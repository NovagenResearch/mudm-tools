#!/usr/bin/env python3
"""PROBE (go/no-go): does MULTI-SCALE input beat single-scale for predicting
spatial *organization* (cell-type intermixing) of a tissue region, under a budget?

Unlike per-cell typing (intrinsic -> fine wins), this label is about ARRANGEMENT:

  * per-cell lineage  = argmax normalized marker score over
        {epithelial/tumor, immune, stromal, endothelial}
  * per-cell mixing   = fraction of its K nearest CELLS of a different lineage
  * region (R um tile) mixing = mean cell-mixing; label = segregated vs intermixed
    (median split), restricted to regions whose 2nd lineage is >= `min_minor`
    (COMPOSITION-CONTROLLED -> the label is arrangement, not composition).

A region holds ~15k transcripts >> budget B, so:
  * fine   : B raw transcripts (single-cell detail, but too sparse to cover the tile)
  * coarse : B merged+counted points (whole-tile composition layout, micro-mixing blurred)
  * multi  : B/2 fine + B/2 coarse  <- muDM multi-LOD
Hypothesis (GO): multi > both -- coarse gives the meso layout, fine the local
adjacency that merge+count destroys; neither alone reads arrangement at this budget.

Usage::
    uv run python scripts/probe_xenium_mixing.py \
        --transcripts ../data/xenium/transcripts.parquet \
        --pyramid ../data/xenium/transcripts_lod_merge.parquet \
        --region-um 150 --coarse-zoom 2 --budgets 64,256,1024 \
        --epochs 40 --output ../results/xenium_mixing_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

LINEAGE_MARKERS = {
    "epithelial": ["EPCAM", "KRT8", "KRT7", "KRT14", "KRT5", "KRT15", "CDH1", "ELF3",
                   "FOXA1", "GATA3", "TACSTD2", "ERBB2", "CEACAM6", "ESR1", "KLF5",
                   "MLPH", "ANKRD30A", "S100A14", "CLDN4"],
    "immune": ["PTPRC", "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "TRAC", "IL7R",
               "CD247", "MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "TCL1A", "MZB1",
               "TNFRSF17", "CD68", "CD163", "LYZ", "ITGAX", "ITGAM", "C1QA", "C1QC",
               "CD14", "AIF1", "TYROBP", "FCER1G", "MRC1", "MNDA", "NKG7", "GNLY",
               "KLRD1", "PRF1", "GZMB", "GZMA", "GZMK", "CPA3", "TPSAB1", "CTSG",
               "LTB", "CCL5"],
    "stromal": ["PDGFRA", "PDGFRB", "LUM", "DPT", "FBLN1", "POSTN", "SFRP4", "SFRP1",
                "MMP2", "LRRC15", "PCOLCE", "CXCL12", "PTN", "CCDC80", "MEDAG",
                "ACTA2", "MYH11", "MYLK", "ACTG2"],
    "endothelial": ["PECAM1", "VWF", "CLDN5", "CLEC14A", "EGFL7", "KDR", "RAMP2",
                    "AQP1", "CD93", "MMRN2", "SOX17", "SOX18", "NOSTRIN"],
}


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"none of {cands} in {cols}")


def load_and_label(transcripts, pyramid, genes_json, region_um, coarse_zoom,
                   knn, min_cells, min_minor, min_marker, qv_min, split, seed):
    gid_map = json.loads(Path(genes_json).read_text())
    n_genes = len(gid_map)
    lineages = list(LINEAGE_MARKERS)
    gene2lin = np.full(n_genes, -1, np.int64)
    for li, name in enumerate(lineages):
        for g in LINEAGE_MARKERS[name]:
            if g in gid_map:
                gene2lin[gid_map[g]] = li
    nmark = np.array([(gene2lin == li).sum() for li in range(len(lineages))], float)

    pf = pq.ParquetFile(transcripts)
    names = pf.schema_arrow.names
    cx = _pick(names, "x_location", "x"); cy = _pick(names, "y_location", "y")
    cg = _pick(names, "feature_name", "gene"); cc_ = _pick(names, "cell_id", "cell")
    cqv = next((c for c in names if c.lower() in ("qv", "quality_value")), None)
    t = pf.read(columns=[cx, cy, cg, cc_] + ([cqv] if cqv else []))
    x = np.asarray(t.column(cx), float); y = np.asarray(t.column(cy), float)
    gene = np.asarray(t.column(cg).cast("string").to_pylist(), object)
    cell = np.asarray(t.column(cc_).cast("string").to_pylist(), object)
    keep = np.ones(len(x), bool)
    if cqv:
        keep &= np.asarray(t.column(cqv)) >= qv_min
    keep &= np.array([g in gid_map for g in gene])
    x, y, gene = x[keep], y[keep], gene[keep]
    cell = cell[keep]
    gid = np.array([gid_map[g] for g in gene], np.int32)
    assigned = np.array([str(c) not in ("UNASSIGNED", "-1", "0", "None", "") for c in cell])

    xmin, ymin = x.min(), y.min()
    NY = int(np.ceil((y.max() - ymin) / region_um)) + 1
    rid_all = (((x - xmin) // region_um).astype(np.int64) * NY
               + ((y - ymin) // region_um).astype(np.int64))

    # --- per-cell lineage + centroid (assigned cells only) ---
    ca = cell[assigned]; xa = x[assigned]; ya = y[assigned]; ga = gid[assigned]
    order = np.argsort(ca, kind="stable")
    ca, xa, ya, ga = ca[order], xa[order], ya[order], ga[order]
    bnds = np.where(ca[1:] != ca[:-1])[0] + 1
    cc_x, cc_y, cc_lin = [], [], []
    for s, e in zip(np.r_[0, bnds], np.r_[bnds, len(ca)]):
        lin = gene2lin[ga[s:e]]
        lin = lin[lin >= 0]
        if len(lin) < min_marker:
            continue
        score = np.bincount(lin, minlength=len(lineages)) / nmark
        if score.max() <= 0:
            continue
        cc_x.append(xa[s:e].mean()); cc_y.append(ya[s:e].mean())
        cc_lin.append(int(score.argmax()))
    cc_x = np.array(cc_x); cc_y = np.array(cc_y); cc_lin = np.array(cc_lin, np.int64)
    print(f"Labeled {len(cc_lin):,} cells; lineage counts="
          f"{ {lineages[i]: int((cc_lin==i).sum()) for i in range(len(lineages))} }",
          flush=True)

    # --- per-cell mixing via KNN over cells ---
    nn = NearestNeighbors(n_neighbors=knn + 1).fit(np.c_[cc_x, cc_y])
    _, idx = nn.kneighbors(np.c_[cc_x, cc_y])
    neigh_lin = cc_lin[idx[:, 1:]]
    cell_mix = (neigh_lin != cc_lin[:, None]).mean(axis=1)

    # --- region stats (from cells) ---
    crid = (((cc_x - xmin) // region_um).astype(np.int64) * NY
            + ((cc_y - ymin) // region_um).astype(np.int64))
    reg = {}
    for r in np.unique(crid):
        m = crid == r
        if m.sum() < min_cells:
            continue
        comp = np.bincount(cc_lin[m], minlength=len(lineages)).astype(float)
        comp /= comp.sum()
        second = np.sort(comp)[-2]
        if second < min_minor:
            continue
        reg[int(r)] = float(cell_mix[m].mean())
    if not reg:
        raise SystemExit("no eligible regions -- relax min_cells/min_minor")
    rids = np.array(sorted(reg)); mix = np.array([reg[r] for r in rids])
    if split == "tertile":
        q1, q2 = np.quantile(mix, [1 / 3, 2 / 3])
        label = {int(r): (0 if reg[r] <= q1 else 1) for r in rids
                 if reg[r] <= q1 or reg[r] >= q2}
        n0 = sum(v == 0 for v in label.values()); n1 = sum(v == 1 for v in label.values())
        print(f"{len(rids)} eligible regions; tertile q1={q1:.3f} q2={q2:.3f}; "
              f"kept {len(label)} (segregated/intermixed = {n0}/{n1})", flush=True)
    else:
        thr = float(np.median(mix))
        label = {int(r): int(reg[r] >= thr) for r in rids}
        print(f"{len(rids)} eligible regions; mixing median={thr:.3f}; "
              f"class balance={int((mix>=thr).sum())}/{int((mix<thr).sum())}", flush=True)

    # --- fine points (raw transcripts) per eligible region ---
    elig = set(label)
    fmask = np.array([r in elig for r in rid_all])
    fr = rid_all[fmask]; fx = x[fmask]; fy = y[fmask]; fg = gid[fmask]
    o = np.argsort(fr, kind="stable"); fr, fx, fy, fg = fr[o], fx[o], fy[o], fg[o]
    fb = np.r_[0, np.where(fr[1:] != fr[:-1])[0] + 1, len(fr)]
    fine = {}
    for i in range(len(fb) - 1):
        s, e = fb[i], fb[i + 1]
        fine[int(fr[s])] = (fx[s:e].astype(np.float32), fy[s:e].astype(np.float32),
                            fg[s:e].astype(np.int64))

    # --- coarse points (merge pyramid @ coarse_zoom) per eligible region ---
    pp = pq.ParquetFile(pyramid)
    tp = pp.read(columns=["zoom", "x", "y", "gene_id", "count"])
    pz = np.asarray(tp.column("zoom")); pm = pz == coarse_zoom
    px = np.asarray(tp.column("x"), float)[pm]; py = np.asarray(tp.column("y"), float)[pm]
    pg = np.asarray(tp.column("gene_id"))[pm].astype(np.int64)
    pc = np.asarray(tp.column("count"))[pm].astype(np.float32)
    prid = (((px - xmin) // region_um).astype(np.int64) * NY
            + ((py - ymin) // region_um).astype(np.int64))
    pmask = np.array([r in elig for r in prid])
    prid, px, py, pg, pc = prid[pmask], px[pmask], py[pmask], pg[pmask], pc[pmask]
    o = np.argsort(prid, kind="stable"); prid, px, py, pg, pc = (a[o] for a in (prid, px, py, pg, pc))
    pb = np.r_[0, np.where(prid[1:] != prid[:-1])[0] + 1, len(prid)]
    coarse = {}
    for i in range(len(pb) - 1):
        s, e = pb[i], pb[i + 1]
        coarse[int(prid[s])] = (px[s:e].astype(np.float32), py[s:e].astype(np.float32),
                                pg[s:e].astype(np.int64), pc[s:e].astype(np.float32))

    ids = [r for r in label if r in fine and r in coarse]
    centers = {r: (xmin + ((r // NY) + 0.5) * region_um,
                   ymin + ((r % NY) + 0.5) * region_um) for r in ids}
    print(f"{len(ids)} regions usable (fine+coarse present)", flush=True)
    return fine, coarse, label, centers, ids, n_genes, region_um / 2.0


class RegionDS(Dataset):
    def __init__(self, ids, fine, coarse, label, centers, half, budget, policy):
        self.ids = ids; self.fine = fine; self.coarse = coarse; self.label = label
        self.centers = centers; self.half = half; self.B = budget; self.policy = policy

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _samp(arr, k):
        n = arr.shape[0]
        if n == 0:
            return np.zeros((k, 5), np.float32)
        return arr[np.random.choice(n, k, replace=n < k)]

    def _fine(self, r, cx, cy):
        fx, fy, fg = self.fine[r]
        return np.c_[(fx - cx) / self.half, (fy - cy) / self.half, fg.astype(np.float32),
                     np.zeros(len(fx), np.float32), np.zeros(len(fx), np.float32)].astype(np.float32)

    def _coarse(self, r, cx, cy):
        px, py, pg, pc = self.coarse[r]
        return np.c_[(px - cx) / self.half, (py - cy) / self.half, pg.astype(np.float32),
                     np.log1p(pc), np.ones(len(px), np.float32)].astype(np.float32)

    def __getitem__(self, i):
        r = self.ids[i]; cx, cy = self.centers[r]
        if self.policy == "fine":
            pts = self._samp(self._fine(r, cx, cy), self.B)
        elif self.policy == "coarse":
            pts = self._samp(self._coarse(r, cx, cy), self.B)
        else:
            h = self.B // 2
            pts = np.concatenate([self._samp(self._fine(r, cx, cy), h),
                                  self._samp(self._coarse(r, cx, cy), self.B - h)])
        return (torch.from_numpy(pts[:, :2]), torch.from_numpy(pts[:, 2].astype(np.int64)),
                torch.from_numpy(pts[:, 3:5].copy()), self.label[r])


class Net(nn.Module):
    def __init__(self, n_genes, n_cls, emb=32):
        super().__init__()
        self.emb = nn.Embedding(n_genes, emb)
        self.mlp = nn.Sequential(nn.Conv1d(2 + emb + 2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, xy, g, feat):
        e = self.emb(g)
        x = torch.cat([xy, e, feat], dim=2).transpose(1, 2)
        x = self.mlp(x).max(dim=2)[0]
        return self.fc(x)


def run_epoch(model, loader, opt, dev, train):
    model.train() if train else model.eval()
    crit = nn.CrossEntropyLoss(); preds = []; labs = []
    torch.set_grad_enabled(train)
    for xy, g, feat, yb in loader:
        xy, g, feat, yb = xy.to(dev), g.to(dev), feat.to(dev), yb.to(dev)
        out = model(xy, g, feat)
        if train:
            opt.zero_grad(); crit(out, yb).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += yb.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0)


def train_eval(tr, va, te, fine, coarse, label, centers, half, budget, policy,
               n_genes, epochs, dev, bs=64):
    def mk(ids, sh):
        return DataLoader(RegionDS(ids, fine, coarse, label, centers, half, budget, policy),
                          bs, shuffle=sh, num_workers=6)
    model = Net(n_genes, 2).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
    best = -1.0; best_te = (0.0, 0.0)
    for ep in range(epochs):
        run_epoch(model, tl, opt, dev, True)
        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            va_acc, _ = run_epoch(model, vl, None, dev, False)
            if va_acc >= best:
                best = va_acc; best_te = run_epoch(model, tel, None, dev, False)
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--pyramid", required=True)
    ap.add_argument("--genes-json", default=None)
    ap.add_argument("--region-um", type=float, default=150.0)
    ap.add_argument("--coarse-zoom", type=int, default=2)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=40)
    ap.add_argument("--min-minor", type=float, default=0.2)
    ap.add_argument("--min-marker", type=int, default=2)
    ap.add_argument("--split", choices=["median", "tertile"], default="median",
                    help="tertile drops the ambiguous middle third of mixing scores")
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--budgets", default="64,256,1024")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="../results/xenium_mixing_probe.json")
    args = ap.parse_args()
    genes_json = args.genes_json or (args.pyramid + ".genes.json")
    budgets = [int(b) for b in args.budgets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)

    t0 = time.perf_counter()
    fine, coarse, label, centers, ids, n_genes, half = load_and_label(
        args.transcripts, args.pyramid, genes_json, args.region_um, args.coarse_zoom,
        args.knn, args.min_cells, args.min_minor, args.min_marker, args.qv_min,
        args.split, args.seed)
    print(f"Setup in {time.perf_counter()-t0:.1f}s", flush=True)

    y = [label[r] for r in ids]
    tr, te = train_test_split(ids, test_size=0.3, random_state=args.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.18, random_state=args.seed,
                              stratify=[label[r] for r in tr])
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = {"budgets": budgets, "n_regions": len(ids), "region_um": args.region_um,
               "coarse_zoom": args.coarse_zoom, "min_minor": args.min_minor, "matrix": {}}
    for policy in ("fine", "coarse", "multi"):
        results["matrix"][policy] = {}
        for B in budgets:
            ts = time.perf_counter()
            acc, f1 = train_eval(tr, va, te, fine, coarse, label, centers, half,
                                 B, policy, n_genes, args.epochs, dev)
            results["matrix"][policy][str(B)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            print(f"  {policy:>6} B={B:<5} acc={acc:.4f} f1={f1:.4f} "
                  f"({time.perf_counter()-ts:.0f}s)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    mat = results["matrix"]
    for B in budgets:
        b = str(B)
        gain = mat["multi"][b]["acc"] - max(mat["fine"][b]["acc"], mat["coarse"][b]["acc"])
        print(f"VERDICT @B={B}: fine={mat['fine'][b]['acc']} coarse={mat['coarse'][b]['acc']} "
              f"multi={mat['multi'][b]['acc']} (multi-best_single={gain:+.4f})")


if __name__ == "__main__":
    main()
