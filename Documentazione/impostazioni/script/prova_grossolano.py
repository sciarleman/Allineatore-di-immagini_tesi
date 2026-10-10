"""Il pre-allineamento com'e' oggi contro una versione che riduce le due
immagini con lo STESSO fattore. Solo misura, nello scratchpad."""
import os, sys, json
import numpy as np
import cv2

sys.path.insert(0, r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA\Allineatore")
import image_io, block_matching as bm, align_paintings as ap
from transform_model import Transform

RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)


def grossolano_stessa_scala(ref_grad, mov_grad, lato=512):
    """Come bm.coarse_translation, ma le due immagini si riducono con lo stesso
    fattore e si ritaglia la parte comune dall'origine: un pixel della miniatura
    vale gli stessi pixel veri in tutte e due."""
    h, w = ref_grad.shape[:2]
    hm, wm = mov_grad.shape[:2]
    scala = min(1.0, lato / max(h, w, hm, wm))
    a = cv2.resize(ref_grad, None, fx=scala, fy=scala, interpolation=cv2.INTER_AREA)
    b = cv2.resize(mov_grad, None, fx=scala, fy=scala, interpolation=cv2.INTER_AREA)
    hh, ww = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:hh, :ww], b[:hh, :ww]
    win = cv2.createHanningWindow((ww, hh), cv2.CV_64F)
    (dx, dy), conf = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win)
    return dx / scala, dy / scala, conf


pf, ff = image_io.leggi_ridotta(os.path.join(RUN, "VIS.png"), lato_lungo_target=3600, in_grigio=True, verbose=False)
gf = bm.gradient_magnitude(pf)
W, H = 29049, 20428
gx, gy = np.meshgrid(np.linspace(0, W - 1, 300), np.linspace(0, H - 1, 210))
P = np.column_stack([gx.ravel(), gy.ravel()])
for nome in ("NIR", "UV"):
    pm, fm = image_io.leggi_ridotta(os.path.join(RUN, nome + ".png"), lato_lungo_target=3600, in_grigio=True, verbose=False)
    gm = bm.gradient_magnitude(pm)
    T = Transform.load(os.path.join(RUN, "parametri", f"{nome} (2026-10-05 21-33).json"))
    d = ap._porta_punti(T, P) - P
    vero = np.median(d, axis=0)
    ox, oy = bm.coarse_translation(gf, gm, verbose=False)
    oggi = np.array([ox * ff, oy * ff], float)
    nx, ny, conf = grossolano_stessa_scala(gf, gm)
    nuovo = np.array([round(nx) * ff, round(ny) * ff], float)
    print(f"\n{nome}: anteprime {pf.shape[1]}x{pf.shape[0]} e {pm.shape[1]}x{pm.shape[0]} (fattori {ff:g}, {fm:g})")
    print(f"  spostamento vero (mediana sul dipinto): ({vero[0]:+.0f}, {vero[1]:+.0f}) px")
    for et, off in (("oggi", oggi), ("stessa scala", nuovo), ("ideale (la mediana vera)", np.round(vero))):
        r = np.hypot(*(d - off).T)
        riga = f"  {et:<26} ({off[0]:+5.0f}, {off[1]:+5.0f}) px -> resta mediana {np.median(r):4.0f} px; in portata:"
        for f in (1, 2, 4):
            ok = (np.abs(d[:, 0] - off[0]) <= 64 * f) & (np.abs(d[:, 1] - off[1]) <= 64 * f)
            riga += f"  f{f} {100 * ok.mean():3.0f}%"
        print(riga + (f"   (conf {conf:.2f})" if et == "stessa scala" else ""))
