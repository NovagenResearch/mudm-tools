"""Real-neuron-fragment codec benchmark.

Decodes REAL per-neuron mesh fragments from the shipped 3D-Tiles GLBs in
../mudm-data/tiles/<dataset>/3dtiles (organic, irregular connectome surface
meshes at the actual serving granularity), then compares codecs:

  * meshopt (CURRENT shipped): exact bytes already in the GLB (raw f32, lossless)
  * meshopt u16-quant: the doc's quick-win variant (lossy, KHR-style)
  * MMC v1 f32         : bbox-quantize to qbits, native Rust
  * draco-oxide f32    : the repo's production encoder, native Rust
  * libdraco cl=7      : native C++ via DracoPy

The three q-bit codecs (MMC / draco-oxide / libdraco) all do per-mesh bbox
quantization to the SAME qbits, so they are fidelity-matched and directly
comparable. meshopt-raw is lossless f32 (hence bigger) — the current baseline.

Usage: python bench_real_neurons.py [dataset] [n_tiles] [qbits]
"""
import sys, json, struct, glob, os, time
import numpy as np
import meshoptimizer as mo
from mudm_tools._rs import mmc_encode_mesh, mmc_decode, draco_encode_mesh
try:
    import DracoPy; HAVE_DRACO = True
except Exception: HAVE_DRACO = False
try:
    import zstandard; ZC = zstandard.ZstdCompressor(level=3); HAVE_ZSTD = True
except Exception: HAVE_ZSTD = False

CT = {5121:1, 5123:2, 5125:4, 5126:4}          # componentType -> bytes
NC = {'SCALAR':1, 'VEC2':2, 'VEC3':3}

def glb_json_bin(path):
    b = open(path,'rb').read(); off=12; J=BN=None
    while off < len(b):
        clen,ct = struct.unpack('<II', b[off:off+8]); off+=8
        c=b[off:off+clen]; off+=clen
        if ct==0x4E4F534A: J=json.loads(c)
        elif ct==0x004E4942: BN=c
    return J, BN

def read_accessor(J, BN, ai):
    """Return (numpy array [count,nc], meshopt_compressed_byte_len)."""
    a=J['accessors'][ai]; bv=J['bufferViews'][a['bufferView']]
    ext=bv.get('extensions',{}).get('EXT_meshopt_compression')
    nc=NC[a['type']]; comp = '<f4' if a['componentType']==5126 else ('<u4' if a['componentType']==5125 else '<u2')
    if ext:
        raw=BN[ext['byteOffset']:ext['byteOffset']+ext['byteLength']]
        if ext['mode']=='ATTRIBUTES':
            dec=mo.decode_vertex_buffer(ext['count'], ext['byteStride'], raw, np.dtype((np.float32,3)))
            arr=np.asarray(dec).reshape(ext['count'],3)
        else:
            arr=np.asarray(mo.decode_index_buffer(ext['count'], 4, raw)).reshape(-1,1)
        return arr, ext['byteLength']
    # plain
    sz=CT[a['componentType']]; base=bv.get('byteOffset',0)+a.get('byteOffset',0)
    stride=bv.get('byteStride') or sz*nc
    out=np.empty((a['count'],nc), dtype=np.dtype(comp))
    for i in range(a['count']):
        s=base+i*stride; out[i]=np.frombuffer(BN[s:s+sz*nc], dtype=np.dtype(comp))
    return out, sz*nc*a['count']

def fragments(path):
    J,BN = glb_json_bin(path)
    for m in J['meshes']:
        for prim in m['primitives']:
            if 'POSITION' not in prim['attributes'] or 'indices' not in prim: continue
            try:
                P,pbytes = read_accessor(J,BN,prim['attributes']['POSITION'])
                I,ibytes = read_accessor(J,BN,prim['indices'])
            except Exception: continue
            yield P.astype(np.float32), I.reshape(-1).astype(np.uint32), pbytes, ibytes

def meshopt_u16(P, I, qbits=16):
    mins=P.min(0); rng=(P.max(0)-mins); rng[rng==0]=1
    q=np.clip(np.floor((P-mins)/rng*((1<<qbits)-1)+0.5),0,(1<<qbits)-1).astype(np.uint16)
    n=len(q); pad=np.zeros((n,4),dtype=np.uint16); pad[:,:3]=q
    vb=mo.encode_vertex_buffer(pad, n, 8); ib=mo.encode_index_buffer(I, len(I), n)
    s=len(vb)+len(ib)
    return (len(ZC.compress(vb))+len(ZC.compress(ib))) if HAVE_ZSTD else s

def main():
    ds = sys.argv[1] if len(sys.argv)>1 else 'flywire'
    n_tiles = int(sys.argv[2]) if len(sys.argv)>2 else 60
    qbits = int(sys.argv[3]) if len(sys.argv)>3 else 14
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "mudm-data"))
    tiles = sorted(glob.glob(f"{base}/tiles/{ds}/3dtiles/**/*.glb", recursive=True))
    # spread across the tree (zoom levels): take every k-th
    step=max(1,len(tiles)//n_tiles); tiles=tiles[::step][:n_tiles]
    print(f"dataset={ds} sampling {len(tiles)} tiles, qbits={qbits}, DracoPy={HAVE_DRACO}\n")

    agg={k:[0,0.0] for k in ['meshopt_raw','meshopt_u16','mmc','dracox','libdraco']}  # [bytes, enc_seconds]
    nfrag=0; ntris=0; mmc_dec=0.0; mmc_wins_vs_lib=0; lib_cmp=0; mmc_wins_vs_dx=0
    skipped=0
    for ti,t in enumerate(tiles):
        for P,I,pb,ib in fragments(t):
            nt=len(I)//3
            if nt<1 or len(P)<3: continue
            pos=P.reshape(-1).tolist(); idx=I.tolist()
            nfrag+=1; ntris+=nt
            agg['meshopt_raw'][0]+=pb+ib
            try: agg['meshopt_u16'][0]+=meshopt_u16(P,I)
            except Exception: pass
            # MMC f32
            try:
                t0=time.perf_counter(); blob=mmc_encode_mesh(pos,idx,qbits,3); agg['mmc'][1]+=time.perf_counter()-t0
                agg['mmc'][0]+=len(blob)
                t0=time.perf_counter(); mmc_decode(blob); mmc_dec+=time.perf_counter()-t0
            except Exception: blob=None
            # draco-oxide f32
            dx=None
            try:
                t0=time.perf_counter(); dx=draco_encode_mesh(pos,idx,qbits); agg['dracox'][1]+=time.perf_counter()-t0
                agg['dracox'][0]+=len(dx)
            except Exception: pass
            # libdraco
            lib=None
            if HAVE_DRACO:
                try:
                    pts=P.astype(np.float32); faces=I.reshape(-1,3)
                    t0=time.perf_counter()
                    lib=DracoPy.encode(pts,faces,quantization_bits=qbits,compression_level=7)
                    agg['libdraco'][1]+=time.perf_counter()-t0
                    agg['libdraco'][0]+=len(lib)
                except Exception: lib=None
            if blob and lib is not None:
                lib_cmp+=1; mmc_wins_vs_lib += (len(blob)<len(lib))
            if blob and dx is not None:
                mmc_wins_vs_dx += (len(blob)<len(dx))
    print(f"{nfrag:,} real fragments, {ntris:,} triangles total  (mean {ntris/max(1,nfrag):.0f} tris/frag)\n")
    def line(name, key, extra=""):
        by,sec=agg[key]
        print(f"  {name:26s} {by:>12,d} B  {by/max(1,ntris):6.3f} B/tri   enc {sec*1e3:9.1f} ms total ({by/max(1,nfrag):6.0f} B/frag) {extra}")
    line("meshopt RAW f32 (shipped)", 'meshopt_raw', "[lossless f32 — current]")
    if HAVE_ZSTD: line("meshopt u16-quant +zstd", 'meshopt_u16', "[lossy q16]")
    line("MMC v1 f32 (native Rust)", 'mmc', f"[lossy q{qbits}]")
    line("draco-oxide f32 (prod)", 'dracox', f"[lossy q{qbits}]")
    if HAVE_DRACO: line("libdraco cl=7", 'libdraco', f"[lossy q{qbits}]")
    print(f"\n  MMC decode total: {mmc_dec*1e3:.1f} ms (incl. pyo3 list marshaling)")
    if lib_cmp: print(f"  MMC < libdraco on {mmc_wins_vs_lib}/{lib_cmp} fragments ({100*mmc_wins_vs_lib/lib_cmp:.0f}%)")
    print(f"  MMC < draco-oxide on {mmc_wins_vs_dx}/{nfrag} fragments ({100*mmc_wins_vs_dx/max(1,nfrag):.0f}%)")
    # ratios
    if agg['libdraco'][0]: print(f"\n  size: MMC/libdraco = {agg['mmc'][0]/agg['libdraco'][0]:.2f}x   MMC/draco-oxide = {agg['mmc'][0]/max(1,agg['dracox'][0]):.2f}x   MMC/meshopt-raw = {agg['mmc'][0]/max(1,agg['meshopt_raw'][0]):.2f}x")

if __name__=="__main__":
    main()
