"""Native codec benchmark: MMC (real Rust via pyo3) vs draco-oxide (the repo's
PRODUCTION NG encoder, native Rust via pyo3) vs libdraco (DracoPy) vs meshopt
(native C bindings, mirroring encoder_meshopt.rs).

Improves on tools/mmc/bench.py: MMC times here are the ACTUAL Rust codec, not the
Python reference upper bound. All NG-path codecs run the lossless u32 grid path,
so size is apples-to-apples. Any codec whose dep is missing is skipped.

Timing note: MMC and draco-oxide are called through identical pyo3 Vec<T>
marshaling, so their ratio is fair; libdraco/meshopt use numpy-buffer bindings
(less per-call overhead), so MMC's times here are conservative vs those two.
"""
import time
import numpy as np

# ---- codecs (graceful) ----
from mudm_tools._rs import mmc_encode_ng_u32, mmc_decode, draco_encode_ng_u32
try:
    import DracoPy
    HAVE_DRACO = True
except Exception as e:
    HAVE_DRACO = False; print("(no DracoPy:", e, ")")
try:
    import meshoptimizer as mo
    HAVE_MO = True
except Exception as e:
    HAVE_MO = False; print("(no meshoptimizer:", e, ")")
try:
    import zstandard
    HAVE_ZSTD = True
except Exception:
    HAVE_ZSTD = False


# ---- meshes (copied verbatim from tools/mmc/bench.py) ----
def terrain(N=256, qbits=10, seed=1):
    rng = np.random.RandomState(seed); z = np.zeros((N, N))
    for octv in range(1, 6):
        k = 2 ** octv; small = rng.rand(k, k)
        idx = np.linspace(0, k - 1, N); xi, yi = np.meshgrid(idx, idx)
        x0 = np.floor(xi).astype(int); y0 = np.floor(yi).astype(int)
        x1 = np.minimum(x0 + 1, k - 1); y1 = np.minimum(y0 + 1, k - 1)
        fx = xi - x0; fy = yi - y0
        up = (small[y0,x0]*(1-fx)*(1-fy)+small[y0,x1]*fx*(1-fy)+small[y1,x0]*(1-fx)*fy+small[y1,x1]*fx*fy)
        z += up / k
    z = (z - z.min()) / (z.max() - z.min()); qmax = (1 << qbits) - 1
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    pos = np.stack([(xs/(N-1)*qmax).round(),(ys/(N-1)*qmax).round(),(z*qmax).round()],axis=-1).reshape(-1,3).astype(np.uint32)
    idx=[]
    for y in range(N-1):
        for x in range(N-1):
            i=y*N+x; idx+=[i,i+1,i+N,i+1,i+N+1,i+N]
    return pos, np.array(idx, dtype=np.uint32)

def icosphere(subdiv=5, qbits=10):
    t=(1+5**0.5)/2
    verts=[(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    faces=[(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
           (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    verts=[np.array(v)/np.linalg.norm(v) for v in verts]; cache={}
    def mid(a,b):
        key=(min(a,b),max(a,b))
        if key not in cache:
            m=verts[a]+verts[b]; verts.append(m/np.linalg.norm(m)); cache[key]=len(verts)-1
        return cache[key]
    for _ in range(subdiv):
        nf=[]
        for a,b,c in faces:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a); nf+=[(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces=nf
    v=np.array(verts); qmax=(1<<qbits)-1
    q=((v-v.min(0))/(v.max(0)-v.min(0))*qmax+0.5).astype(np.uint32).clip(0,qmax)
    return q.astype(np.uint32), np.array(faces,dtype=np.uint32).reshape(-1)

def dup_heavy(N=128, qbits=10):
    base=[]
    for y in range(N):
        for x in range(N):
            base.append([x*7%(1<<qbits),y*7%(1<<qbits),(x*7+y*13)%97])
    base=np.array(base,dtype=np.uint32); pos=np.concatenate([base,base]); k=len(base); idx=[]; t=0
    for y in range(N-1):
        for x in range(N-1):
            i=y*N+x; r=i+1; d=i+N; dr=d+1
            tw=(lambda v:v+k) if t%2==0 else (lambda v:v); idx+=[tw(i),r,tw(d),r,dr,tw(d)]; t+=1
    return pos.reshape(-1), np.array(idx,dtype=np.uint32)


def best(fn, reps=4):
    fn()  # warmup
    b = 1e9; out = None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); b = min(b, time.perf_counter() - t0)
    return out, b * 1e3  # ms


def bench_mmc(pos_q, idx, qbits):
    pl = pos_q.reshape(-1).tolist(); il = idx.tolist()  # marshal once, outside timing? no—part of call
    blob, te = best(lambda: mmc_encode_ng_u32(pl, il, qbits, 3))
    _, td = best(lambda: mmc_decode(blob))
    return len(blob), te, td

def bench_dracox(pos_q, idx, qbits):
    pl = pos_q.reshape(-1).tolist(); il = idx.tolist()
    blob, te = best(lambda: draco_encode_ng_u32(pl, il, qbits))
    # decode via libdraco (cross-decodable; representative viewer decode)
    td = float("nan")
    if HAVE_DRACO:
        _, td = best(lambda: DracoPy.decode(blob))
    return len(blob), te, td

def bench_libdraco(pos_q, idx, qbits):
    pts = pos_q.reshape(-1,3).astype(np.float32); faces = idx.reshape(-1,3)
    blob, te = best(lambda: DracoPy.encode(pts, faces, quantization_bits=qbits, compression_level=7,
        quantization_range=float((1<<qbits)-1), quantization_origin=np.zeros(3,dtype=np.float32)))
    _, td = best(lambda: DracoPy.decode(blob))
    return len(blob), te, td

def bench_meshopt(pos_q, idx):
    verts = pos_q.reshape(-1,3).astype(np.float32).copy(); n_idx = len(idx)
    def enc():
        oi = np.zeros(n_idx, dtype=np.uint32); mo.optimize_vertex_cache(oi, idx, n_idx, len(verts))
        ov = np.zeros_like(verts); nu = mo.optimize_vertex_fetch(ov, oi, verts); ov = ov[:nu]
        return mo.encode_vertex_buffer(ov, nu, 12), mo.encode_index_buffer(oi, n_idx, nu), nu
    (vb, ib, nu), te = best(enc)
    def dec():
        return mo.decode_vertex_buffer(nu, 12, vb, np.dtype((np.float32,3))), mo.decode_index_buffer(n_idx, 4, ib)
    _, td = best(dec)
    size = len(vb) + len(ib); zsize = None
    if HAVE_ZSTD:
        z = zstandard.ZstdCompressor(level=3); zsize = len(z.compress(vb)) + len(z.compress(ib))
    return size, zsize, te, td


def row(name, ntri, size, te, td, extra=""):
    print(f"  {name:30s} {size:>10,d} B  {size/ntri:6.3f} B/tri   enc {te:8.2f} ms   dec {td:8.2f} ms  {extra}")

def main():
    meshes = [("terrain 256² q10 (NG)", *terrain(256,10), 10),
              ("icosphere subdiv5 q10", *icosphere(5,10), 10),
              ("dup-heavy 128² q10 (NG)", *dup_heavy(128,10), 10),
              ("terrain 512² q14 (GLB-ish)", *terrain(512,14,seed=2), 14)]
    for name, pos, idx, qb in meshes:
        nt = len(idx)//3; nv = len(pos.reshape(-1))//3
        print(f"\n=== {name}: {nv:,} verts, {nt:,} tris ===")
        s,te,td = bench_mmc(pos, idx, qb);      row("MMC v1 (native Rust)", nt, s, te, td)
        s,te,td = bench_dracox(pos, idx, qb);   row("draco-oxide (repo prod NG)", nt, s, te, td, "(dec=libdraco)" if HAVE_DRACO else "(no dec)")
        if HAVE_DRACO:
            s,te,td = bench_libdraco(pos, idx, qb); row("libdraco cl=7 (DracoPy)", nt, s, te, td)
        if HAVE_MO:
            s,zs,te,td = bench_meshopt(pos, idx);   row("meshopt (raw f32, =crate)", nt, s, te, td, f"(+zstd {zs:,} B)" if zs else "")

if __name__ == "__main__":
    main()
