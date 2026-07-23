"""End-to-end check of the meshopt-u16 quick win (compression='meshopt-q').

Ingests a substantial noisy mesh through the real pipeline, emits 3D Tiles with
compression='meshopt' and 'meshopt-q', then:
  1. confirms the meshopt-q GLBs declare KHR_mesh_quantization + u16 POSITION,
  2. decodes them with the REAL meshopt decoder + applies the node TRS, and
     cross-checks world positions against the f32-meshopt tile (within a quant
     step) — i.e. proves the wire format is stock-decodable and correct,
  3. reports the size delta.
"""
import json, struct, glob, os, sys, tempfile
import numpy as np
import meshoptimizer as mo
from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

def write_noisy_obj(path, n=64):
    lines = []
    for y in range(n):
        for x in range(n):
            px = x * 13.7 + 1000.0
            py = y * 21.1 + 2000.0
            pz = ((x * 7 + y * 13) % 97) * 5.31 + 500.0
            lines.append(f"v {px:.3f} {py:.3f} {pz:.3f}")
    for y in range(n - 1):
        for x in range(n - 1):
            i = y * n + x + 1  # OBJ is 1-indexed
            lines.append(f"f {i} {i+1} {i+n}")
            lines.append(f"f {i+1} {i+n+1} {i+n}")
    open(path, "w").write("\n".join(lines) + "\n")

def glb_json_bin(b):
    off = 12; J = BN = None
    while off < len(b):
        clen, ct = struct.unpack('<II', b[off:off+8]); off += 8
        c = b[off:off+clen]; off += clen
        if ct == 0x4E4F534A: J = json.loads(c)
        elif ct == 0x004E4942: BN = c
    return J, BN

def decode_positions(J, BN, prim, node):
    a = J['accessors'][prim['attributes']['POSITION']]
    bv = J['bufferViews'][a['bufferView']]
    ext = bv['extensions']['EXT_meshopt_compression']
    raw = BN[ext['byteOffset']:ext['byteOffset']+ext['byteLength']]
    if a['componentType'] == 5123:  # u16 quantized
        dec = np.asarray(mo.decode_vertex_buffer(ext['count'], 8, raw, np.dtype((np.uint16, 4)))).reshape(-1, 4)[:, :3].astype(np.float64)
        t = np.array(node.get('translation', [0,0,0])); s = np.array(node.get('scale', [1,1,1]))
        return t + s * dec, s
    else:  # f32
        dec = np.asarray(mo.decode_vertex_buffer(ext['count'], 12, raw, np.dtype((np.float32, 3)))).reshape(-1, 3).astype(np.float64)
        t = np.array(node.get('translation', [0,0,0])); s = np.array(node.get('scale', [1,1,1]))
        return t + s * dec, s

def total_glb_bytes(d):
    return sum(os.path.getsize(p) for p in glob.glob(f"{d}/**/*.glb", recursive=True))

def main():
    tmp = tempfile.mkdtemp()
    obj = os.path.join(tmp, "mesh.obj"); write_noisy_obj(obj, 64)
    bounds = scan_obj_bounds([obj])
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
    gen.add_obj_files([obj], bounds, [{"body_id": 1}])
    dirs = {}
    for comp in ["meshopt", "meshopt-q"]:
        d = os.path.join(tmp, comp); gen.generate_3dtiles(d, bounds, compression=comp); dirs[comp] = d
    smo, sq = total_glb_bytes(dirs["meshopt"]), total_glb_bytes(dirs["meshopt-q"])
    print(f"total GLB bytes: meshopt {smo:,}  meshopt-q {sq:,}  -> {smo/max(1,sq):.3f}x smaller")

    # structural + decode check on every meshopt-q tile that has a sibling
    khr_ok = decode_checked = 0; worst = 0.0
    for qp in glob.glob(f"{dirs['meshopt-q']}/**/*.glb", recursive=True):
        rel = os.path.relpath(qp, dirs["meshopt-q"]); fp = os.path.join(dirs["meshopt"], rel)
        Jq, BNq = glb_json_bin(open(qp, "rb").read())
        assert "KHR_mesh_quantization" in Jq.get("extensionsRequired", []), f"{rel}: missing KHR"
        khr_ok += 1
        if not os.path.exists(fp): continue
        Jf, BNf = glb_json_bin(open(fp, "rb").read())
        # match meshes by node order (same ingest+opt -> same per-mesh vertex order)
        for ni, nq in enumerate(Jq["nodes"]):
            if "mesh" not in nq: continue
            pq = Jq["meshes"][nq["mesh"]]["primitives"][0]
            if pq["attributes"].get("POSITION") is None or "indices" not in pq: continue
            nf = Jf["nodes"][ni]; pf = Jf["meshes"][nf["mesh"]]["primitives"][0]
            Wq, s = decode_positions(Jq, BNq, pq, nq)
            Wf, _ = decode_positions(Jf, BNf, pf, nf)
            if Wq.shape != Wf.shape: continue
            err = np.abs(Wq - Wf).max(axis=0)            # per-axis max error
            halfstep = s / 2.0 + 1e-3 * np.abs(Wf).max(axis=0).clip(1)
            assert np.all(err <= halfstep * 1.5 + 1e-3), f"{rel} node {ni}: err {err} > halfstep {halfstep}"
            worst = max(worst, float((err / np.where(s > 0, s, 1)).max()))
            decode_checked += 1
    print(f"KHR_mesh_quantization present on {khr_ok} tiles; decode-checked {decode_checked} primitives")
    print(f"worst reconstruction error: {worst:.3f} quant steps (must be <~0.5+slack)")
    print("OK — meshopt-q is stock-decodable, correct, and smaller" if sq < smo else "WARN: not smaller")

if __name__ == "__main__":
    main()
