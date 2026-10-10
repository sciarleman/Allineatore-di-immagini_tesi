"""Figure del processo di stima su VIS / NIR850, dai dati messi in cache da
calcola.py. Tutte le misure sono quelle del motore; qui si disegna soltanto.

Unita': le anteprime sono ridotte di un fattore 2. Dove non e' detto
altrimenti gli spostamenti sono riportati in PIXEL A PIENA RISOLUZIONE
(come nei report del programma), cioe' il doppio dei pixel dell'anteprima.
"""
import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import block_matching as bm
from transform_model import Transform

QUI = os.path.dirname(os.path.abspath(__file__))
OUT = QUI

d = np.load(os.path.join(QUI, "cache.npz"))
pf, pm = d["pf"], d["pm"]
F = int(d["ff"])                         # fattore dell'anteprima (2)
ox, oy = (int(v) for v in d["offset"])
pts1, M, inl = d["pts1"], d["M"], d["inl"]
Mfull = d["Mfull"]
Hf, Wf = int(d["H"]), int(d["W"])

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110})
NOME_F, NOME_M = "VIS (riferimento)", "NIR850 (da allineare)"
LEG = "magenta = VIS, verde = NIR850, bianco = coincidono"


def stira(x, lo=1, hi=99):
    x = x.astype(np.float32)
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def _bordi(x):
    g = bm.gradient_magnitude(np.asarray(x))
    return np.clip(stira(g, 1, 99.7) ** 0.7, 0, 1)


def sovrapponi(a, b):
    """Modulo del gradiente (cio' che confronta l'algoritmo): VIS in magenta,
    NIR850 in verde. Dove le strutture combaciano le linee sono bianche, dove
    no si sdoppiano in una magenta e una verde."""
    ga, gb = _bordi(a), _bordi(b)
    return np.dstack([ga, gb, ga])


def scarto_misurato(a, b):
    """Lo spostamento che resta fra i due ritagli, misurato come fa la
    verifica del programma (correlazione di fase sul gradiente); px anteprima."""
    ga = bm.gradient_magnitude(np.asarray(a)).astype(np.float64)
    gb = bm.gradient_magnitude(np.asarray(b)).astype(np.float64)
    w_ = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_64F)
    (sx_, sy_), c_ = cv2.phaseCorrelate(ga, gb, w_)
    return sx_, sy_, c_


def ritaglio_mov(T, x0, y0, w, h):
    """Il ritaglio [y0:y0+h, x0:x0+w] del riferimento come lo vede la moving
    dopo la trasformazione T (a scala anteprima)."""
    return T.warp_tile_from(pm, y0, x0, h, w)


def salva(fig, nome):
    p = os.path.join(OUT, nome)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  ", p)


# ============================================================================
# 1. allineamento grossolano
# ============================================================================
gf = bm.gradient_magnitude(pf)
gm = bm.gradient_magnitude(pm)
h, w = gf.shape
scala = min(1.0, 512 / max(h, w))
dim = (max(8, int(w * scala)), max(8, int(h * scala)))
a512 = cv2.resize(gf, dim, interpolation=cv2.INTER_AREA).astype(np.float64)
b512 = cv2.resize(gm, dim, interpolation=cv2.INTER_AREA).astype(np.float64)
han = cv2.createHanningWindow((a512.shape[1], a512.shape[0]), cv2.CV_64F)
(cdx, cdy), cconf = cv2.phaseCorrelate(a512, b512, han)


def superficie(a, b, win):
    """La stessa correlazione di fase di cv2.phaseCorrelate, fatta a mano per
    poterne mostrare i passaggi. Convenzione identica: il dettaglio che in a
    sta in p, in b sta in p + d, e il picco della superficie cade in d."""
    A = np.fft.fft2(a * win)
    B = np.fft.fft2(b * win)
    R = np.conj(A) * B
    Rn = R / (np.abs(R) + 1e-12)
    r = np.real(np.fft.ifft2(Rn))
    return A, B, Rn, np.fft.fftshift(r)


_, _, _, r512 = superficie(a512, b512, han)

fig = plt.figure(figsize=(13, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
ax = fig.add_subplot(gs[0, 0]); ax.imshow(stira(a512, 1, 99.5), cmap="gray")
ax.set_title(f"{NOME_F}\nmodulo del gradiente, {a512.shape[1]}×{a512.shape[0]} px"); ax.axis("off")
ax = fig.add_subplot(gs[0, 1]); ax.imshow(stira(b512, 1, 99.5), cmap="gray")
ax.set_title(f"{NOME_M}\nstessa riduzione (×{1/scala:.1f} rispetto all'anteprima)"); ax.axis("off")
ax = fig.add_subplot(gs[0, 2])
cyc, cxc = r512.shape[0] // 2, r512.shape[1] // 2
z = 10
ax.imshow(r512[cyc - z:cyc + z + 1, cxc - z:cxc + z + 1], cmap="magma",
          extent=[-z - .5, z + .5, z + .5, -z - .5])
ax.plot(cdx, cdy, "c+", ms=14, mew=2)
ax.axhline(0, color="w", lw=.4); ax.axvline(0, color="w", lw=.4)
ax.set_title("superficie di correlazione di fase (zoom ±10 px)\n"
             f"picco in ({cdx:+.2f}, {cdy:+.2f}) px, altezza {cconf:.2f}")
ax.set_xlabel("dx (px a questa scala)"); ax.set_ylabel("dy")

# ritaglio sull'occhio: prima e dopo lo spostamento intero trovato
X0, Y0, S = 3560, 3380, 320
vis = pf[Y0:Y0 + S, X0:X0 + S]
pre = pm[Y0:Y0 + S, X0:X0 + S]
post = pm[Y0 + oy:Y0 + oy + S, X0 + ox:X0 + ox + S]
s_pre = scarto_misurato(vis, pre)
s_post = scarto_misurato(vis, post)
ax = fig.add_subplot(gs[1, 0]); ax.imshow(sovrapponi(vis, pre))
ax.set_title(f"PRIMA: così come sono\nscarto misurato nel ritaglio "
             f"({s_pre[0] * F:+.1f}, {s_pre[1] * F:+.1f}) px"); ax.axis("off")
ax = fig.add_subplot(gs[1, 1]); ax.imshow(sovrapponi(vis, post))
ax.set_title(f"DOPO lo spostamento d'insieme ({ox * F:+d}, {oy * F:+d}) px\n"
             f"scarto che resta qui ({s_post[0] * F:+.1f}, {s_post[1] * F:+.1f}) px"); ax.axis("off")
fig.text(0.36, 0.035, LEG + "  ·  gradiente dell'anteprima, spostamenti in px a piena risoluzione",
         ha="center", fontsize=8.5)
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.text(0, 1, (
    "Traslazione d'insieme (coarse_translation)\n\n"
    "1. modulo del gradiente delle due anteprime\n"
    f"2. riduzione a {a512.shape[1]}×{a512.shape[0]} px (lato lungo 512)\n"
    "3. finestra di Hann e correlazione di fase\n"
    "   fra le due immagini intere\n"
    f"4. picco in ({cdx:+.2f}, {cdy:+.2f}) px a quella scala\n"
    f"   → ({cdx / scala:+.1f}, {cdy / scala:+.1f}) px dell'anteprima\n"
    f"   → arrotondato ({ox:+d}, {oy:+d}) px dell'anteprima\n"
    f"   = ({ox * F:+d}, {oy * F:+d}) px a piena risoluzione\n\n"
    "È SOLO una traslazione intera, uguale ovunque:\n"
    "serve a far partire ogni blocco vicino al suo\n"
    "bersaglio. Quel che resta varia da zona a zona\n"
    "e lo misurano i blocchi.\n"
    f"Confidenza {cconf:.2f} ≥ 0.03: lo spostamento è accettato.\n\n"
    f"Ritaglio: x {X0 * F}–{(X0 + S) * F}, y {Y0 * F}–{(Y0 + S) * F} px\n"
    "(coordinate a piena risoluzione)"),
    va="top", family="monospace", fontsize=8.6)
fig.suptitle("1 · Traslazione d'insieme", fontsize=13, weight="bold")
salva(fig, "1_traslazione_d_insieme.png")


def it(x, n=1):
    """Un numero con la virgola decimale."""
    return f"{x:.{n}f}".replace(".", ",")


# la stessa figura in versione per la tesi: solo i pannelli, titoli brevi,
# niente riquadro di testo e niente nomi di funzioni
fig = plt.figure(figsize=(12, 8.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(stira(a512, 1, 99.5), cmap="gray"); ax.axis("off")
ax.set_title("(a) riferimento, modulo del gradiente")
ax = fig.add_subplot(gs[0, 2:4]); ax.imshow(stira(b512, 1, 99.5), cmap="gray"); ax.axis("off")
ax.set_title("(b) immagine da allineare")
ax = fig.add_subplot(gs[0, 4:6])
z = 10
ax.imshow(r512[cyc - z:cyc + z + 1, cxc - z:cxc + z + 1], cmap="magma",
          extent=[-z - .5, z + .5, z + .5, -z - .5])
ax.plot(cdx, cdy, "c+", ms=14, mew=2)
ax.axhline(0, color="w", lw=.4); ax.axvline(0, color="w", lw=.4)
ax.set_title("(c) correlazione di fase: il picco\ndà la traslazione d'insieme")
ax.set_xlabel("px delle immagini ridotte"); ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
ax = fig.add_subplot(gs[1, 1:3]); ax.imshow(sovrapponi(vis, pre)); ax.axis("off")
ax.set_title(f"(d) un dettaglio, prima: scarto di {it(np.hypot(s_pre[0], s_pre[1]) * F)} px")
ax = fig.add_subplot(gs[1, 3:5]); ax.imshow(sovrapponi(vis, post)); ax.axis("off")
ax.set_title(f"(e) dopo la traslazione: restano {it(np.hypot(s_post[0], s_post[1]) * F)} px")
fig.tight_layout()
salva(fig, "1_traslazione_d_insieme_tesi.png")

# ============================================================================
# 2. un blocco: trasformata e trasformata inversa
# ============================================================================
bersaglio = np.array([3900, 3600])
cand = pts1[pts1[:, 4] >= 0.5]
k = np.argmin(np.hypot(cand[:, 0] - bersaglio[0], cand[:, 1] - bersaglio[1]))
cx, cy = int(cand[k, 0]), int(cand[k, 1])
half = 128
a = gf[cy - half:cy + half, cx - half:cx + half].astype(np.float64)
b = gm[cy + oy - half:cy + oy + half, cx + ox - half:cx + ox + half].astype(np.float64)
win = cv2.createHanningWindow((2 * half, 2 * half), cv2.CV_64F)
(ldx, ldy), lconf = cv2.phaseCorrelate(a, b, win)
A, B, Rn, rr = superficie(a, b, win)
# controllo: il blocco e' lo stesso che il motore ha misurato
assert abs((ldx + ox) - cand[k, 2]) < 1e-6 and abs((ldy + oy) - cand[k, 3]) < 1e-6
iy, ix = np.unravel_index(np.argmax(rr), rr.shape)
picco_int = (ix - half, iy - half)

img_a = pf[cy - half:cy + half, cx - half:cx + half]
img_b = pm[cy + oy - half:cy + oy + half, cx + ox - half:cx + ox + half]
Msh = np.float32([[1, 0, ldx], [0, 1, ldy]])           # dst(p) = src(p + d)
img_b_post = cv2.warpAffine(img_b, Msh, (2 * half, 2 * half),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REFLECT)

fig, axs = plt.subplots(3, 4, figsize=(15, 11.4))
for axx in axs.ravel():
    axx.set_xticks([]); axx.set_yticks([])
axs[0, 0].imshow(img_a, cmap="gray"); axs[0, 0].set_title(f"blocco nel {NOME_F}\n256×256 px dell'anteprima (512 dell'originale)")
axs[0, 1].imshow(img_b, cmap="gray"); axs[0, 1].set_title(f"stesso posto nel {NOME_M}\n(già spostato della traslazione d'insieme)")
axs[0, 2].imshow(sovrapponi(img_a, img_b)); axs[0, 2].set_title("PRIMA della misura\nmagenta = VIS, verde = NIR850")
axs[0, 3].imshow(sovrapponi(img_a, img_b_post))
axs[0, 3].set_title(f"DOPO: NIR850 riportato di d = ({ldx:+.2f}, {ldy:+.2f}) px\n(sub-pixel, px dell'anteprima)")

axs[1, 0].imshow(a * win, cmap="gray"); axs[1, 0].set_title("gradiente VIS × finestra di Hann")
axs[1, 1].imshow(np.log1p(np.abs(np.fft.fftshift(A))), cmap="viridis")
axs[1, 1].set_title("TRASFORMATA di Fourier del VIS\nlog |F_a|  (centro = frequenza 0)")
axs[1, 2].imshow(np.log1p(np.abs(np.fft.fftshift(B))), cmap="viridis")
axs[1, 2].set_title("TRASFORMATA del NIR850\nlog |F_b|: moduli quasi uguali")
axs[1, 3].imshow(np.angle(np.fft.fftshift(Rn)), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axs[1, 3].set_title("fase di  F_a* · F_b / |F_a* · F_b|\nle frange codificano lo spostamento")

axs[2, 0].imshow(np.sqrt(np.clip(rr, 0, None)), cmap="magma",
                 extent=[-half - .5, half - .5, half - .5, -half - .5])
axs[2, 0].set_title("TRASFORMATA INVERSA della fase = superficie\n"
                    "di correlazione, 256×256 (radice, per vedere il fondo)")
axs[2, 0].set_xticks([-128, 0, 128]); axs[2, 0].set_yticks([-128, 0, 128])
axs[2, 0].plot(ldx, ldy, "o", mfc="none", mec="c", ms=14, mew=1.5)
axs[2, 0].annotate("un solo picco,\nnel punto d", xy=(ldx + 4, ldy - 4), xytext=(30, -80),
                   color="c", fontsize=8.5, arrowprops=dict(arrowstyle="->", color="c"))
somma5 = float(rr[iy - 2:iy + 3, ix - 2:ix + 3].sum())
z = 8
axs[2, 1].imshow(rr[half - z:half + z + 1, half - z:half + z + 1], cmap="magma",
                 extent=[-z - .5, z + .5, z + .5, -z - .5])
axs[2, 1].plot(ldx, ldy, "c+", ms=16, mew=2)
axs[2, 1].axhline(0, color="w", lw=.4); axs[2, 1].axvline(0, color="w", lw=.4)
axs[2, 1].set_xticks(range(-z, z + 1, 4)); axs[2, 1].set_yticks(range(-z, z + 1, 4))
axs[2, 1].set_title(f"zoom ±{z} px attorno all'origine\n+ = picco sub-pixel di OpenCV")
fig.delaxes(axs[2, 2])
ax3 = fig.add_subplot(3, 4, 11, projection="3d")
z3 = 12
Y3, X3 = np.mgrid[-z3:z3 + 1, -z3:z3 + 1]
ax3.plot_surface(X3, Y3, rr[half - z3:half + z3 + 1, half - z3:half + z3 + 1],
                 cmap="magma", linewidth=0, antialiased=True)
ax3.set_title("la stessa superficie in 3D", pad=0)
ax3.set_xlabel("dx"); ax3.set_ylabel("dy"); ax3.view_init(35, -60)
axs[2, 3].axis("off")
axs[2, 3].text(0, 1, (
    "Correlazione di fase su un blocco\n\n"
    f"centro del blocco (anteprima): ({cx}, {cy})\n"
    f"  = ({cx * F}, {cy * F}) px a piena ris.\n\n"
    f"picco intero:      ({picco_int[0]:+d}, {picco_int[1]:+d})\n"
    f"picco sub-pixel d: ({ldx:+.3f}, {ldy:+.3f})\n"
    f"confidenza:         {lconf:.3f}\n"
    f"  (≈ somma della superficie nel\n"
    f"   5×5 attorno al picco: {somma5:.3f};\n"
    f"   il pixel più alto vale {rr.max():.3f})\n\n"
    f"+ traslazione ({ox:+d}, {oy:+d}) →\n"
    f"spostamento del blocco\n"
    f"  ({ldx + ox:+.2f}, {ldy + oy:+.2f}) px anteprima\n"
    f"  ({(ldx + ox) * F:+.2f}, {(ldy + oy) * F:+.2f}) px piena ris.\n\n"
    "Si confronta la sola FASE: una\n"
    "traslazione cambia la fase ma\n"
    "non il modulo, e luminosità e\n"
    "contrasto diversi fra le due bande\n"
    "cambiano il modulo ma non la fase."),
    va="top", family="monospace", fontsize=8.6)
fig.suptitle("2 · Correlazione di fase su un blocco, dalla trasformata di Fourier alla sua inversa",
             fontsize=13, weight="bold")
fig.tight_layout()
salva(fig, "2_correlazione_di_fase_un_blocco.png")

# versione per la tesi: otto pannelli, dalla coppia di blocchi allo spostamento
fig, axs = plt.subplots(2, 4, figsize=(14, 7.6))
for axx in axs.ravel():
    axx.set_xticks([]); axx.set_yticks([])
axs[0, 0].imshow(img_a, cmap="gray"); axs[0, 0].set_title("(a) blocco del riferimento")
axs[0, 1].imshow(img_b, cmap="gray"); axs[0, 1].set_title("(b) stesso blocco, immagine da allineare")
axs[0, 2].imshow(np.log1p(np.abs(np.fft.fftshift(A))), cmap="viridis")
axs[0, 2].set_title("(c) trasformata di Fourier di (a), modulo")
axs[0, 3].imshow(np.log1p(np.abs(np.fft.fftshift(B))), cmap="viridis")
axs[0, 3].set_title("(d) trasformata di Fourier di (b), modulo")
axs[1, 0].imshow(np.angle(np.fft.fftshift(Rn)), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axs[1, 0].set_title("(e) fase del prodotto normalizzato:\nle frange codificano lo spostamento")
z = 8
axs[1, 1].imshow(rr[half - z:half + z + 1, half - z:half + z + 1], cmap="magma",
                 extent=[-z - .5, z + .5, z + .5, -z - .5])
axs[1, 1].plot(ldx, ldy, "c+", ms=16, mew=2)
axs[1, 1].axhline(0, color="w", lw=.4); axs[1, 1].axvline(0, color="w", lw=.4)
axs[1, 1].set_xticks(range(-z, z + 1, 4)); axs[1, 1].set_yticks(range(-z, z + 1, 4))
axs[1, 1].set_title("(f) trasformata inversa: un solo picco,\nnel punto dello spostamento (px)")
axs[1, 2].imshow(sovrapponi(img_a, img_b)); axs[1, 2].set_title("(g) i due blocchi sovrapposti, prima")
axs[1, 3].imshow(sovrapponi(img_a, img_b_post))
axs[1, 3].set_title(f"(h) dopo lo spostamento misurato,\n{it(np.hypot(ldx, ldy) * F)} px sull'originale")
fig.tight_layout()
salva(fig, "2_correlazione_di_fase_un_blocco_tesi.png")

# ============================================================================
# 3. prima e dopo il block matching (a livello di immagine)
# ============================================================================
T_gross = Transform(np.array([[1.0, 0, ox], [0, 1.0, oy]]))
T_aff = Transform(M)
campo = d["campo"] if float(d["passo"]) > 0 else None
T_comp = Transform(d["Mc"], campo, float(d["passo"]) if campo is not None else None)
zone = [("occhio", 3560, 3380, 320), ("dita e catena", 5560, 6000, 320),
        ("catena, in basso", 5380, 7880, 320)]
fasi = [("dopo la sola traslazione d'insieme", T_gross),
        ("dopo la posa\n(affine, blocchi da 512 px)", T_aff),
        ("dopo la stima completa\n(affine e campi di deformazione)", T_comp)]
fig, axs = plt.subplots(3, 3, figsize=(12.5, 13.8))
for i_, (nome, x0, y0, s_) in enumerate(zone):
    v = pf[y0:y0 + s_, x0:x0 + s_]
    for j_, (fase, T_) in enumerate(fasi):
        m_ = ritaglio_mov(T_, x0, y0, s_, s_)
        sx_, sy_, c_ = scarto_misurato(v, m_)
        ax = axs[i_, j_]
        ax.imshow(sovrapponi(v, m_)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((f"{fase}\n" if i_ == 0 else "")
                     + f"scarto misurato ({sx_ * F:+.2f}, {sy_ * F:+.2f}) px, "
                       f"|{np.hypot(sx_, sy_) * F:.2f}| px", fontsize=9)
    axs[i_, 0].set_ylabel(f"{nome}\nx {x0 * F}–{(x0 + s_) * F}\ny {y0 * F}–{(y0 + s_) * F}",
                          fontsize=9)
fig.suptitle("3 · Prima e dopo la stima\n" + LEG
             + "  ·  spostamenti in px a piena risoluzione",
             fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_prima_dopo_la_stima.png"), dpi=170, bbox_inches="tight")
fig.suptitle("")          # versione per la tesi: legenda e unita' vanno nella didascalia
salva(fig, "3_prima_dopo_la_stima_tesi.png")

# ============================================================================
# 4. minimi quadrati pesati con scarto iterativo
# ============================================================================
usabili = pts1[pts1[:, 4] >= 0.10]
scartati_conf = pts1[pts1[:, 4] < 0.10]
# si ripete il ciclo di fit_affine_robust per registrare ogni iterazione
sel = usabili
storia = []
for it in range(8):
    Mi, resi = bm._weighted_affine(sel)
    mag = np.hypot(resi[:, 0], resi[:, 1])
    med = np.median(mag)
    mad = np.median(np.abs(mag - med)) * 1.4826 + 1e-9
    keep = mag <= med + 3.0 * mad
    storia.append((len(sel), med, med + 3.0 * mad, mag))
    if keep.all() or keep.sum() < 6:
        break
    sel = sel[keep]
Mfin, _ = bm._weighted_affine(sel)
assert np.allclose(Mfin, M)
chiavi_in = {(a_, b_) for a_, b_ in sel[:, :2]}
e_in = np.array([(a_, b_) in chiavi_in for a_, b_ in usabili[:, :2]])
x, y, dx, dy, cf = usabili.T
pred = np.column_stack([x, y, np.ones_like(x)]) @ M.T
rx, ry = (x + dx) - pred[:, 0], (y + dy) - pred[:, 1]

miniatura = cv2.resize(pf, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
est = [0, pf.shape[1] * F, pf.shape[0] * F, 0]


def sfondo(ax):
    ax.imshow(miniatura, cmap="gray", extent=est, alpha=.55)
    ax.set_xlim(0, pf.shape[1] * F); ax.set_ylim(pf.shape[0] * F, 0)
    ax.set_xticks([]); ax.set_yticks([])


fig = plt.figure(figsize=(16, 16.5))
gs = fig.add_gridspec(3, 6, height_ratios=[1.45, .95, .8])
# un blocco ogni 3 per lato: 768 px fra una freccia e l'altra
rada = ((np.round(x / 128) % 3 == 0) & (np.round(y / 128) % 3 == 0))
rada_c = ((np.round(scartati_conf[:, 0] / 128) % 3 == 0)
          & (np.round(scartati_conf[:, 1] / 128) % 3 == 0))
# gli anomali si mostrano tutti: sono pochi e sono il punto della figura
vis_in, vis_out = e_in & rada, ~e_in


def frecce(ax, sel_, u, v, col, k, w=.0022):
    ax.quiver(x[sel_] * F, y[sel_] * F, u[sel_] * F, -v[sel_] * F, color=col,
              scale=1 / k, scale_units="xy", angles="xy", width=w, headwidth=3.5)


k_q = 20
ax = fig.add_subplot(gs[0, 0:3]); sfondo(ax)
sc = scartati_conf[rada_c]
ax.quiver(sc[:, 0] * F, sc[:, 1] * F, sc[:, 2] * F, -sc[:, 3] * F, color="0.45",
          scale=1 / k_q, scale_units="xy", angles="xy", width=.0022, headwidth=3.5)
frecce(ax, vis_in, dx, dy, "tab:green", k_q)
frecce(ax, vis_out, dx, dy, "red", k_q, .0028)
ax.set_title(f"spostamento misurato su ogni blocco, frecce ×{k_q} (un blocco su 9)\n"
             f"verde = usato ({e_in.sum()}), rosso = anomalo, tutti ({(~e_in).sum()}), "
             f"grigio = confidenza < 0.10 ({len(scartati_conf)})")
k_r = 70
ax = fig.add_subplot(gs[0, 3:6]); sfondo(ax)
frecce(ax, vis_in, rx, ry, "tab:green", k_r)
frecce(ax, vis_out, rx, ry, "red", k_r, .0028)
ax.set_title(f"residuo dopo l'affine = misurato − previsto, frecce ×{k_r}\n"
             "resta la deformazione non rigida; gli anomali stanno sullo sfondo e sul bordo\n"
             "destro (mano, bottiglia), dove la deformazione vera è più grande della media")


def mappa(valori, sel_mask=None):
    gx = np.round(x / 128).astype(int); gy = np.round(y / 128).astype(int)
    G = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    m = np.ones(len(x), bool) if sel_mask is None else sel_mask
    G[gy[m], gx[m]] = valori[m]
    return G


ext = [0, (x.max() + 64) * F, (y.max() + 64) * F, 0]
spost_x, prev_x = dx * F, (pred[:, 0] - x) * F
spost_y, prev_y = dy * F, (pred[:, 1] - y) * F
lim_x = np.nanpercentile(np.abs(spost_x), 98)
lim_y = np.nanpercentile(np.abs(spost_y), 98)
lim_r = 15
pannelli = [
    (gs[1, 0], mappa(spost_x, e_in), "dx misurato", (-lim_x, lim_x)),
    (gs[1, 1], mappa(prev_x), "dx dell'affine: un piano\n(a−1)·x + b·y + tx", (-lim_x, lim_x)),
    (gs[1, 2], mappa(rx * F, e_in), "dx residuo\n= misurato − piano", (-lim_r, lim_r)),
    (gs[1, 3], mappa(spost_y, e_in), "dy misurato", (-lim_y, lim_y)),
    (gs[1, 4], mappa(prev_y), "dy dell'affine: un piano\nc·x + (d−1)·y + ty", (-lim_y, lim_y)),
    (gs[1, 5], mappa(ry * F, e_in), "dy residuo\n= misurato − piano", (-lim_r, lim_r)),
]
for g, G, tit, (vmin, vmax) in pannelli:
    ax = fig.add_subplot(g)
    im = ax.imshow(G, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=ext)
    ax.set_title(tit, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02, orientation="horizontal",
                      label="px a piena ris.")
    cb.locator = matplotlib.ticker.MaxNLocator(5)
    cb.update_ticks()

ax = fig.add_subplot(gs[2, 0:3])
colori = plt.cm.viridis(np.linspace(0, .85, len(storia)))
for (n, med, soglia, mag), c_ in zip(storia, colori):
    ax.hist(mag * F, bins=np.linspace(0, 40, 161), histtype="step", color=c_, lw=1.2,
            label=f"iter. {storia.index((n, med, soglia, mag)) + 1}: {n} blocchi, "
                  f"soglia {soglia * F:.1f} px")
    ax.axvline(soglia * F, color=c_, ls="--", lw=.9)
ax.set_yscale("log"); ax.set_xlabel("|residuo| dall'affine (px a piena risoluzione)")
ax.set_ylabel("n. blocchi"); ax.legend(fontsize=7.5)
ax.set_title("scarto iterativo: via chi supera mediana + 3·MAD, e si rifà il fit")
ax = fig.add_subplot(gs[2, 3:6]); ax.axis("off")
ax.text(0.03, 1, (
    "Minimi quadrati pesati (_weighted_affine)\n\n"
    "Per ogni blocco i, centro (xᵢ, yᵢ), spostamento (dxᵢ, dyᵢ):\n"
    "   xᵢ + dxᵢ ≈ a·xᵢ + b·yᵢ + tx\n"
    "   yᵢ + dyᵢ ≈ c·xᵢ + d·yᵢ + ty\n\n"
    "si minimizza   Σ confᵢ · ‖(xᵢ+dxᵢ, yᵢ+dyᵢ) − M·(xᵢ, yᵢ, 1)‖²\n"
    "(ogni equazione moltiplicata per √confᵢ): due problemi\n"
    "N×3 con la stessa matrice, uno per riga di M,\n"
    "risolti con np.linalg.lstsq.\n\n"
    "Robustezza (fit_affine_robust):\n"
    "  1. si tengono i blocchi con confidenza ≥ 0.10\n"
    "  2. fit, residui, soglia = mediana + 3·MAD·1.4826\n"
    "  3. si scartano i blocchi oltre soglia e si ripete\n"
    "     al più 8 giri, o finché nessuno viene scartato;\n"
    "     qui si arriva all'ottavo, con 1 blocco tolto\n\n"
    f"Qui: {len(pts1)} blocchi misurati, {len(usabili)} con conf ≥ 0.10,\n"
    f"{len(storia)} iterazioni, {len(sel)} inlier finali.\n"
    f"Residuo mediano degli inlier {storia[-1][1] * F:.2f} px:\n"
    "non è rumore di misura ma deformazione che un'affine\n"
    "non può rappresentare: flessione della tela e, nelle\n"
    "mappe del residuo, salti netti lungo linee dritte,\n"
    "cioè cuciture di mosaico delle riprese (vedi README)."),
    va="top", family="monospace", fontsize=8.8)
fig.suptitle("4 · Dai blocchi all'affine: minimi quadrati pesati e scarto degli anomali",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.975])
salva(fig, "4_minimi_quadrati.png")

# ============================================================================
# 5. i sei parametri
# ============================================================================
a_, b_, tx = Mfull[0]
c_, d_, ty = Mfull[1]
sx, sy = np.hypot(a_, c_), np.hypot(b_, d_)
rot = np.degrees(np.arctan2(c_, a_))
shear = np.degrees(np.arctan2(b_, d_)) + rot
ap, bp, txp = M[0]
cp_, dp, typ = M[1]

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, 0:2]); ax.axis("off")
ax.text(0, 1, (
    "coord. NIR850 = M · [x_VIS, y_VIS, 1]\n\n"
    "      ⎡ a   b   tx ⎤\n"
    "  M = ⎣ c   d   ty ⎦\n\n"
    "sull'anteprima (fattore 2, come stimata):\n"
    f"  a  = {ap:.6f}   b  = {bp:+.6f}   tx = {txp:+.3f}\n"
    f"  c  = {cp_:+.6f}   d  = {dp:.6f}   ty = {typ:+.3f}\n\n"
    "a piena risoluzione (Transform.from_factor):\n"
    f"  a  = {a_:.6f}   b  = {b_:+.6f}   tx = {tx:+.3f}\n"
    f"  c  = {c_:+.6f}   d  = {d_:.6f}   ty = {ty:+.3f}\n\n"
    "a, b, c, d non cambiano con la scala; la traslazione\n"
    "sì: t_piena = 2·t_ant. + c_m − A·c_f, con c = (f−1)/2 = 0.5.\n\n"
    "Lettura (Transform.describe):\n"
    f"  scala x  {sx:.5f}  ({(sx - 1) * 100:+.3f}%)\n"
    f"  scala y  {sy:.5f}  ({(sy - 1) * 100:+.3f}%)\n"
    f"  rotazione {rot:+.4f}°\n"
    f"  taglio    {shear:+.4f}°\n"
    f"  traslazione ({tx:+.1f}, {ty:+.1f}) px"),
    va="top", family="monospace", fontsize=9.6)

# effetto sugli angoli e sul centro, a piena risoluzione, valori veri
ax = fig.add_subplot(gs[0, 2:4])
P = np.array([[0, 0], [Wf, 0], [Wf, Hf], [0, Hf], [Wf / 2, Hf / 2]], float)
Q = P @ Mfull[:, :2].T + Mfull[:, 2]
Dp = Q - P
kk = 150
ax.add_patch(Rectangle((0, 0), Wf, Hf, fill=False, ec="0.4", lw=1))
ax.imshow(miniatura, cmap="gray", extent=[0, Wf, Hf, 0], alpha=.35)
contorno = np.array([[0, 0], [Wf, 0], [Wf, Hf], [0, Hf], [0, 0]], float)
cq = contorno + (contorno @ Mfull[:, :2].T + Mfull[:, 2] - contorno) * kk
ax.plot(cq[:, 0], cq[:, 1], color="tab:orange", lw=1.4, label=f"contorno trasformato, scarti ×{kk}")
for p_, dd in zip(P, Dp):
    ax.annotate("", xy=p_ + dd * kk, xytext=p_,
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
    ax.text(p_[0] + (600 if p_[0] >= Wf / 2 else -600), p_[1] + (900 if p_[1] >= Hf / 2 else -500),
            f"({dd[0]:+.1f}, {dd[1]:+.1f}) px", color="tab:red", fontsize=8.5,
            ha="left" if p_[0] >= Wf / 2 else "right",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
ax.set_xlim(-2500, Wf + 4500); ax.set_ylim(Hf + 2500, -2500)
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("dove l'affine manda angoli e centro del VIS (valori veri a piena ris.)")
ax.set_xticks([0, Wf]); ax.set_yticks([0, Hf])

# i sei parametri uno per uno
def griglia_deformata(ax, Mx, titolo, esagera):
    g = np.linspace(0, 1, 9)
    Mx = np.array(Mx, float)
    A = np.eye(2) + (Mx[:, :2] - np.eye(2)) * esagera
    t = Mx[:, 2] * esagera
    for v in g:
        for linea in (np.column_stack([g, np.full_like(g, v)]),
                      np.column_stack([np.full_like(g, v), g])):
            ax.plot(linea[:, 0], linea[:, 1], color="0.8", lw=.6)
            q = linea @ A.T + t
            ax.plot(q[:, 0], q[:, 1], color="tab:blue", lw=.9)
    ax.set_xlim(-.25, 1.25); ax.set_ylim(1.25, -.25); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titolo, fontsize=8.8)


sub = gs[1, :].subgridspec(1, 6)
voci = [
    ("a", [[a_, 0, 0], [0, 1, 0]], f"a − 1 = {a_ - 1:+.2e}\nscala in x\n→ {(a_ - 1) * Wf:+.1f} px sulla larghezza"),
    ("b", [[1, b_, 0], [0, 1, 0]], f"b = {b_:+.2e}\nx che dipende da y\n→ {b_ * Hf:+.1f} px sull'altezza"),
    ("tx", [[1, 0, tx / Wf], [0, 1, 0]], f"tx = {tx:+.2f} px\ntraslazione in x"),
    ("c", [[1, 0, 0], [c_, 1, 0]], f"c = {c_:+.2e}\ny che dipende da x\n→ {c_ * Wf:+.1f} px sulla larghezza"),
    ("d", [[1, 0, 0], [0, d_, 0]], f"d − 1 = {d_ - 1:+.2e}\nscala in y\n→ {(d_ - 1) * Hf:+.1f} px sull'altezza"),
    ("ty", [[1, 0, 0], [0, 1, ty / Hf]], f"ty = {ty:+.2f} px\ntraslazione in y"),
]
for i, (nome, Mx, tit) in enumerate(voci):
    ax = fig.add_subplot(sub[0, i])
    esag = 0.12 / max(abs(np.array(Mx) - [[1, 0, 0], [0, 1, 0]]).max(), 1e-12)
    griglia_deformata(ax, Mx, f"[{nome}]  {tit}\n(esagerato ×{esag:.0f})", esag)
fig.suptitle("5 · I sei parametri dell'affine VIS → NIR850 (posa)",
             fontsize=13, weight="bold")
fig.tight_layout()
salva(fig, "5_sei_parametri.png")

# riepilogo numerico per il testo
with open(os.path.join(OUT, "numeri.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"coppia: VIS -> NIR850, anteprima fattore {F}, {pf.shape[1]}x{pf.shape[0]} px\n")
    fh.write(f"grossolano: ({ox},{oy}) px anteprima = ({ox*F},{oy*F}) px piena, conf {cconf:.3f}\n")
    fh.write(f"blocco mostrato: centro ({cx},{cy}) anteprima, d=({ldx:.3f},{ldy:.3f}), conf {lconf:.3f}\n")
    fh.write(f"blocchi: {len(pts1)} misurati, {len(usabili)} conf>=0.10, {len(sel)} inlier, "
             f"{len(storia)} iterazioni\n")
    for n, med, so, _m in storia:
        fh.write(f"  iter: {n} blocchi, mediana {med*F:.3f} px, soglia {so*F:.3f} px (piena ris.)\n")
    fh.write(f"M anteprima:\n{M}\nM piena:\n{Mfull}\n")
    fh.write(f"angoli/centro spostati (px piena): {Dp.round(2).tolist()}\n")
print("fatto")
