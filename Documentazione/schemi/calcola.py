"""Calcola e mette in cache i passaggi intermedi della stima, con le funzioni
del motore cosi' come sono (nessuna modifica al progetto)."""
import os, sys, time
import numpy as np
import cv2

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io
import block_matching as bm
from transform_model import Transform

RIPRESE = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
FIX = os.path.join(RIPRESE, "VIS_FraGalgario_TOT.tif")
MOV = os.path.join(RIPRESE, sys.argv[1] if len(sys.argv) > 1 else "NIR850_FraGalgario_TOT.tif")
OUT = os.path.join(os.path.dirname(__file__), "cache.npz")

t = time.time()
info = image_io.info(FIX)
target = max(info.height, info.width) // 2
pf, ff = image_io.leggi_ridotta(FIX, lato_lungo_target=target, in_grigio=True)
pm, fm = image_io.leggi_ridotta(MOV, lato_lungo_target=target, in_grigio=True)
print("anteprime", pf.shape, pm.shape, ff, fm, f"{time.time()-t:.1f}s")

gf = bm.gradient_magnitude(pf)
gm = bm.gradient_magnitude(pm)
offset = bm.coarse_translation(gf, gm, verbose=True)
print("offset", offset)

pts1 = bm.match_blocks(gf, gm, block=256, step=128, offset=offset, verbose=True)
M, inl, res = bm.fit_affine_robust(pts1, min_conf=0.10, verbose=True)
print("M anteprima\n", M)
T = Transform.from_factor(M, ff, f_moving=fm)
print("M piena\n", T.M)
print(T.describe())

# stima completa, come la fa oggi il lotto: posa, campo grossolano se serve, campo fine
Tc, est = bm.estimate_transform(pf, pm, ff, factor_moving=fm, verbose=True,
                                campo_grossolano_tra_le_passate='auto')
Tp = Tc.at_factor(ff, fm)
print(Tc.describe())
np.savez_compressed(OUT, pf=pf, pm=pm, ff=ff, fm=fm, offset=np.array(offset),
                    pts1=pts1, M=M, inl=inl, res=res, Mfull=T.M,
                    Mc=Tp.M, campo=Tp.field if Tp.field is not None else np.zeros((1, 1, 2)),
                    passo=Tp.grid_step or 0.0, H=info.height, W=info.width)
print("salvato", OUT, f"{time.time()-t:.1f}s")
