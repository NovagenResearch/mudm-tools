"""Real-neuron test of the FLAT-* candidates (the check §7.4 asked for).

Decodes real connectome fragments from the shipped 3D-Tiles GLBs and compares
wire size (post transport-zstd) + app-side decode of FLAT-DELTA / FLAT-HYBRID
against the meshopt baselines and libdraco — the same fidelity-matched per-mesh
quantization as bench_real_neurons.py. Settles whether §7's synthetic win
(FLAT-HYBRID 1.6-2.4x smaller than meshopt-u16+z) holds on organic geometry.

Usage: python bench_flat_real.py [dataset] [n_tiles] [qbits]
"""
import sys, json, struct, glob, os, time
import numpy as np
import meshoptimizer as mo
import zstandard
from bench_flat import (flat_delta_encode, flat_hybrid_encode, first_use_remap,
                        meshopt_pipeline, load_flatdec, ZC, ZD)
try:
    import DracoPy; HAVE_DRACO = True
except Exception: HAVE_DRACO = False

CT={5121:1,5123:2,5125:4,5126:4}; NC={'SCALAR':1,'VEC2':2,'VEC3':3}
def glb_json_bin(path):
    b=open(path,'rb').read(); off=12; J=BN=None
    while off<len(b):
        clen,ct=struct.unpack('<II',b[off:off+8]); off+=8
        c=b[off:off+clen]; off+=clen
        if ct==0x4E4F534A: J=json.loads(c)
        elif ct==0x004E4942: BN=c
    return J,BN
def read_acc(J,BN,ai):
    a=J['accessors'][ai]; bv=J['bufferViews'][a['bufferView']]
    ext=bv.get('extensions',{}).get('EXT_meshopt_compression')
    if ext:
        raw=BN[ext['byteOffset']:ext['byteOffset']+ext['byteLength']]
        if ext['mode']=='ATTRIBUTES':
            return np.asarray(mo.decode_vertex_buffer(ext['count'],ext['byteStride'],raw,np.dtype((np.float32,3)))).reshape(ext['count'],3)
        return np.asarray(mo.decode_index_buffer(ext['count'],4,raw)).reshape(-1,1)
    sz=CT[a['componentType']]; nc=NC[a['type']]; comp='<f4' if a['componentType']==5126 else('<u4' if a['componentType']==5125 else '<u2')
    base=bv.get('byteOffset',0)+a.get('byteOffset',0); stride=bv.get('byteStride') or sz*nc
    out=np.empty((a['count'],nc),dtype=np.dtype(comp))
    for i in range(a['count']): out[i]=np.frombuffer(BN[base+i*stride:base+i*stride+sz*nc],dtype=np.dtype(comp))
    return out
def fragments(path):
    J,BN=glb_json_bin(path)
    for m in J['meshes']:
        for prim in m['primitives']:
            if 'POSITION' not in prim['attributes'] or 'indices' not in prim: continue
            try: P=read_acc(J,BN,prim['attributes']['POSITION']); I=read_acc(J,BN,prim['indices'])
            except Exception: continue
            yield P.astype(np.float32), I.reshape(-1).astype(np.uint32)

def quantize(P, qbits):
    mins=P.min(0); rng=(P.max(0)-mins); safe=np.where(rng>0,rng,1.0); qmax=(1<<qbits)-1
    q=np.clip(np.floor((P-mins)/safe*qmax+0.5),0,qmax)
    return np.where(rng>0,q,0).astype(np.uint32)

def main():
    ds=sys.argv[1] if len(sys.argv)>1 else 'flywire'
    n_tiles=int(sys.argv[2]) if len(sys.argv)>2 else 60
    qbits=int(sys.argv[3]) if len(sys.argv)>3 else 14
    base=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..","..","mudm-data"))
    tiles=sorted(glob.glob(f"{base}/tiles/{ds}/3dtiles/**/*.glb",recursive=True))
    tiles=tiles[::max(1,len(tiles)//n_tiles)][:n_tiles]
    lib=load_flatdec()
    print(f"dataset={ds} {len(tiles)} tiles, q{qbits}, DracoPy={HAVE_DRACO}, flatdec={'ok' if lib else 'MISSING'}\n")
    # wire bytes accumulators
    W={k:0 for k in ['meshopt_raw','meshopt_u16','flat_delta','flat_hybrid','libdraco']}
    HDR=24  # per-fragment dequant transform (min+scale f32x3), counted for FLAT (fairness)
    nfrag=0; ntris=0; hyb_wins_u16=0; hyb_modes={'flat':0,'meshopt':0}
    # decode timing accumulators (native kernels), on fragments large enough to matter
    dec_meshopt=0.0; dec_flatpos=0.0; dec_n=0
    for t in tiles:
        for P,I in fragments(t):
            nt=len(I)//3
            if nt<1 or len(P)<3: continue
            q=quantize(P,qbits); n=len(q)
            nfrag+=1; ntris+=nt
            # meshopt raw f32 + transport z
            try:
                vb,ib,nu=meshopt_pipeline(P.astype(np.float32),I,12); W['meshopt_raw']+=len(ZC.compress(vb))+len(ZC.compress(ib))
            except Exception: pass
            # meshopt u16 + transport z
            p16=np.zeros((n,4),dtype=np.uint16); p16[:,:3]=q.astype(np.uint16)
            try:
                vb2,ib2,nu2=meshopt_pipeline(p16,I,8); mz=len(ZC.compress(vb2))+len(ZC.compress(ib2)); W['meshopt_u16']+=mz
            except Exception: mz=None
            # FLAT-DELTA + z
            try:
                _,_,pz,iz=flat_delta_encode(q,I)[2:4] if False else (None,None,*flat_delta_encode(q,I)[2:4])
            except Exception: pz=iz=None
            if pz is not None: W['flat_delta']+=len(pz)+len(iz)+HDR
            # FLAT-HYBRID + z
            try:
                p2,i2,pb5,pz5,ib5,iz5,mode=flat_hybrid_encode(q,I); hz=len(pz5)+len(iz5)+HDR
                W['flat_hybrid']+=hz; hyb_modes[mode]+=1
                if mz is not None and hz<mz: hyb_wins_u16+=1
            except Exception: hz=None
            # libdraco context
            if HAVE_DRACO:
                try:
                    blob=DracoPy.encode(P.astype(np.float32),I.reshape(-1,3),quantization_bits=qbits,compression_level=7); W['libdraco']+=len(blob)
                except Exception: pass
            # decode race on large frags (native NEON pos prefix-sum vs meshopt vertex decode)
            if lib and hasattr(lib,'decode_pos_neon') and nt>2000 and dec_n<400:
                try:
                    zp=np.frombuffer(pb5,dtype=np.uint16).copy(); op=np.zeros(len(p2)*4,dtype=np.uint16)
                    t0=time.perf_counter(); lib.decode_pos_neon(zp.ctypes.data,len(p2),op.ctypes.data); dec_flatpos+=time.perf_counter()-t0
                    t0=time.perf_counter(); mo.decode_vertex_buffer(nu2,8,vb2,np.dtype((np.uint16,4))); dec_meshopt+=time.perf_counter()-t0
                    dec_n+=1
                except Exception: pass
    print(f"{nfrag:,} fragments, {ntris:,} tris (mean {ntris/max(1,nfrag):.0f} tris/frag)\n")
    def line(name,k,note=""):
        by=W[k]; print(f"  {name:28s} {by:>13,d} B  {by/max(1,ntris):6.3f} B/tri  {note}")
    line("meshopt raw f32 +z (shipped)",'meshopt_raw',"[lossless f32]")
    line("meshopt u16 +z (quick win)",'meshopt_u16',f"[lossy q{qbits}]")
    line("FLAT-DELTA +z",'flat_delta',f"[lossy q{qbits}]")
    line("FLAT-HYBRID +z",'flat_hybrid',f"[lossy q{qbits}] modes={hyb_modes}")
    if HAVE_DRACO: line("libdraco cl7 (context)",'libdraco',f"[lossy q{qbits}]")
    u16=W['meshopt_u16']; h=W['flat_hybrid']; lib_b=W['libdraco']
    print(f"\n  FLAT-HYBRID vs meshopt-u16+z: {u16/max(1,h):.2f}x  (hybrid smaller on {hyb_wins_u16}/{nfrag} frags = {100*hyb_wins_u16/max(1,nfrag):.0f}%)")
    if lib_b: print(f"  FLAT-HYBRID vs libdraco: {h/max(1,lib_b):.2f}x (>1 = hybrid larger)")
    if dec_n: print(f"\n  position decode (native, {dec_n} large frags): FLAT-NEON {dec_flatpos*1e3:.1f}ms vs meshopt {dec_meshopt*1e3:.1f}ms  ({dec_meshopt/max(1e-9,dec_flatpos):.1f}x faster)")

if __name__=="__main__": main()
