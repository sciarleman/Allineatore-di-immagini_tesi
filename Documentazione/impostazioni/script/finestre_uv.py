"""Le finestre di verifica dell'UV, una per una: dove sono quelle oltre 5 px,
e se li' l'immagine e' davvero fuori registro o e' la misura ad agganciare male.
Legge i parametri della run delle 21:33; non scrive nulla fuori dallo scratchpad."""
import os, sys, json
import numpy as np
import cv2

sys.path.insert(0, r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA\Allineatore")
import image_io, validation as val, block_matching as bm
from transform_model import Transform

RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)
QUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uv")
os.makedirs(QUI, exist_ok=True)

arr_f, _ = image_io.apri(os.path.join(RUN, "VIS.png"), verbose=False)
arr_m, _ = image_io.apri(os.path.join(RUN, "UV.png"), verbose=False)
T = Transform.load(os.path.join(RUN, "parametri", "UV (2026-10-05 21-33).json"))
H, W = arr_f.shape[:2]

r = val.residual_check(arr_f, arr_m, T, verbose=False, adattiva=False)
c = {(int(v[0]), int(v[1])): (float(v[2]), float(v[3]), float(v[4]))
     for v in (r.get('campioni') or []) + (r.get('campioni_deboli') or [])}
print(f"verifica: mediana {r['residuo_mediano_px']:.3f} px su {r['n_affidabili']}/{r['n_campioni']} finestre, "
      f"oltre 5 px {r['quota_sopra_5px']:.0%}")
xs = sorted({k[0] for k in c}); ys = sorted({k[1] for k in c})
print("griglia: residuo px (confidenza); * = sotto la soglia di confidenza 0.05; '-' = non misurata")
print("         " + "".join(f"x={x:<8d}" for x in xs))
for y in ys:
    riga = f"y={y:<6d} "
    for x in xs:
        if (x, y) in c:
            dx, dy, cf = c[(x, y)]
            riga += f"{np.hypot(dx, dy):5.1f}({cf:.2f}){'*' if cf < 0.05 else ' '}"
        else:
            riga += "     -      "
    print(riga)


def grigio(a):
    a = np.asarray(a)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def misura(x0, y0, s):
    """Spostamento residuo in una finestra s x s con angolo in (x0, y0)."""
    f = grigio(arr_f[y0:y0 + s, x0:x0 + s]); m = grigio(T.warp_tile_from(arr_m, y0, x0, s, s))
    gf, gm = bm.gradient_magnitude(f), bm.gradient_magnitude(m)
    if gf.std() < 1e-4 or gm.std() < 1e-4:
        return None
    (dx, dy), cf = cv2.phaseCorrelate(gf.astype(np.float64), gm.astype(np.float64),
                                      cv2.createHanningWindow((s, s), cv2.CV_64F))
    return dx, dy, cf


def stira(g):
    g = g.astype(np.float32); lo, hi = np.percentile(g, [1, 99])
    return (np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def bordi(g):
    e = bm.gradient_magnitude(g); a, b = np.percentile(e, [1, 99.7])
    return np.clip((e - a) / max(b - a, 1e-6), 0, 1) ** 0.7


aff = sorted(((np.hypot(v[0], v[1]), k) for k, v in c.items() if v[2] >= 0.05), reverse=True)
cattive = [k for res, k in aff if res > 5.0]
buone = [k for res, k in aff[-2:]]
print(f"\nfinestre affidabili oltre 5 px: {len(cattive)}")
print("per ciascuna: la misura a 512 px, poi i 4 quadranti da 256 px, poi una finestra da 1024 px sullo stesso centro")
righe = []
for (cx, cy) in cattive + buone:
    dx, dy, cf = c[(cx, cy)]
    x0, y0 = cx - 256, cy - 256
    quad = [misura(x0 + i * 256, y0 + j * 256, 256) for j in (0, 1) for i in (0, 1)]
    X0, Y0 = max(0, min(W - 1024, cx - 512)), max(0, min(H - 1024, cy - 512))
    grande = misura(X0, Y0, 1024)
    f = grigio(arr_f[y0:y0 + 512, x0:x0 + 512]); m = grigio(T.warp_tile_from(arr_m, y0, x0, 512, 512))
    nero_m = float((m == 0).mean())
    def fmt(q):
        return "   -   " if q is None else f"({q[0]:+6.1f},{q[1]:+6.1f}) c{q[2]:.2f}"
    print(f"  ({cx:5d},{cy:5d}) 512: ({dx:+6.1f},{dy:+6.1f}) c{cf:.2f} | quadranti: " + " ".join(fmt(q) for q in quad)
          + f" | 1024: {fmt(grande)} | contrasto VIS {f.std():.1f} UV {m.std():.1f} | UV nero {nero_m:.0%}")
    # ritaglio: VIS, UV allineato, UV riportato anche dello spostamento misurato, bordi sovrapposti
    ga, gb = bordi(f), bordi(m)
    ov = (np.dstack([ga, gb, ga]) * 255).astype(np.uint8)
    Msh = np.float32([[1, 0, dx], [0, 1, dy]])
    m2 = cv2.warpAffine(m, Msh, (512, 512), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    gb2 = bordi(m2); ov2 = (np.dstack([ga, gb2, ga]) * 255).astype(np.uint8)
    tre = lambda g: cv2.cvtColor(stira(g), cv2.COLOR_GRAY2RGB)
    riga = np.hstack([tre(f), tre(m), ov, ov2])
    cv2.putText(riga, f"({cx},{cy})  misura ({dx:+.1f},{dy:+.1f}) px  conf {cf:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    for i, t in enumerate(("VIS", "UV allineato", "bordi: magenta VIS, verde UV", "bordi dopo aver tolto la misura")):
        cv2.putText(riga, t, (8 + i * 512, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    righe.append(riga)
for i in range(0, len(righe), 3):
    cv2.imwrite(os.path.join(QUI, f"ritagli_{i // 3 + 1}.png"), cv2.cvtColor(np.vstack(righe[i:i + 3]), cv2.COLOR_RGB2BGR))

# mappa: dove stanno le finestre, sul visibile ridotto
passo = 16
mini = np.ascontiguousarray(arr_f[::passo, ::passo]).copy()
for (x, y), (dx, dy, cf) in c.items():
    res = np.hypot(dx, dy)
    col = (150, 150, 150) if cf < 0.05 else (0, 220, 0) if res < 1 else (255, 170, 0) if res < 5 else (255, 0, 0)
    p0, p1 = (int((x - 256) / passo), int((y - 256) / passo)), (int((x + 256) / passo), int((y + 256) / passo))
    cv2.rectangle(mini, p0, p1, col, 3)
    if cf >= 0.05 and res >= 5:
        cv2.putText(mini, f"{res:.0f}", (p0[0], p0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(os.path.join(QUI, "mappa.png"), cv2.cvtColor(mini, cv2.COLOR_RGB2BGR))
print("\nfatto:", len(righe), "ritagli")
