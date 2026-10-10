"""
Come si misura lo spostamento di un blocco con la correlazione di fase, per la
tesi (sezione «La stima a tre scale»), solo in PNG.

    python schema_correlazione_di_fase.py [--ricalcola]

Usa un blocco vero: 512 px dell'originale, ridotto di 2 (256 px, come alla
precisione consigliata), preso nello stesso punto del visibile e
dell'infrarosso prima dell'allineamento. I dati del blocco stanno in
blocco_correlazione.npz, cosi' lo schema si ridisegna senza le riprese; con
--ricalcola li rilegge dai file originali.
"""

import os
import sys

import numpy as np
import cv2

import disegna_schemi as d
from disegna_schemi import plt, FancyBboxPatch
from schema_pipeline import CORPO_NOTA, CORPO_CORNICE

DATI = os.path.join(d.QUI, 'blocco_correlazione.npz')
CARTELLA = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
X0, Y0B, LATO_PX, FATTORE = 13344, 12192, 512, 2      # il blocco, in pixel dell'originale
SIGMA = 1.5

Y0, HC = 2, 44
P = 14                         # lato dei riquadri piccoli
G = 26                         # lato del riquadro della correlazione
PAD, SP = 2.5, 5
ZOOM = 20                      # semilato della zona mostrata della superficie, in px del blocco
FORMULA = r"$r=\mathcal{F}^{-1}\!\left[\,F_a\overline{F_b}\,/\,|F_a\overline{F_b}|\,\right]$"


def leggi_blocchi():
    sys.path.insert(0, os.path.join(d.QUI, '..', '..', 'Allineatore'))
    import tiff_memmap
    out = {}
    for nome, file in (('vis', 'VIS_FraGalgario_TOT.tif'), ('nir', 'NIR850_FraGalgario_TOT.tif')):
        a, _ = tiff_memmap.open_memmap(os.path.join(CARTELLA, file))
        b = np.ascontiguousarray(a[Y0B:Y0B + LATO_PX, X0:X0 + LATO_PX])
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        if b.dtype != np.uint8:
            b = (b >> 8).astype(np.uint8)
        n = LATO_PX // FATTORE
        out[nome] = cv2.resize(b, (n, n), interpolation=cv2.INTER_AREA)
    np.savez_compressed(DATI, **out)
    return out


def gradiente(img):
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), SIGMA)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def correlazione_di_fase(a, b):
    """La superficie r e lo spostamento (dx, dy) di b rispetto ad a, in pixel del blocco."""
    w = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    Fa, Fb = np.fft.fft2((a - a.mean()) * w), np.fft.fft2((b - b.mean()) * w)
    X = Fa * np.conj(Fb)
    r = np.fft.fftshift(np.real(np.fft.ifft2(X / (np.abs(X) + 1e-9))))
    cy, cx = np.unravel_index(np.argmax(r), r.shape)
    c = a.shape[0] // 2
    return r, (c - cx, c - cy), float(r.max())


def pannello(ax, x, y, lato, img, titolo, cmap='gray', vmax=None):
    ax.imshow(img, extent=(x, x + lato, y, y + lato), aspect="auto", cmap=cmap, zorder=1, interpolation="nearest",
              vmin=0 if vmax else None, vmax=vmax)
    ax.add_patch(plt.Rectangle((x, y), lato, lato, fc='none', ec=d.INK2, lw=1.0, zorder=3))
    ax.text(x + lato / 2, y + lato + 1.2, titolo, ha='center', va='bottom', color=d.INK,
            fontsize=CORPO_NOTA, fontweight='bold')


def cornice(ax, x, w, etichetta):
    ax.add_patch(FancyBboxPatch((x, Y0), w, HC, boxstyle="round,pad=0,rounding_size=1.2",
                                fc='#fafaf8', ec='#c3c2b7', lw=0.9))
    ax.text(x + w / 2, Y0 + HC - 2.6, etichetta, ha='center', va='center', color=d.INK2,
            fontsize=CORPO_CORNICE, fontweight='bold')


def schema():
    if '--ricalcola' in sys.argv or not os.path.exists(DATI):
        blocchi = leggi_blocchi()
    else:
        blocchi = dict(np.load(DATI))
    vis, nir = blocchi['vis'], blocchi['nir']
    ga, gb = gradiente(vis), gradiente(nir)
    r, (dx, dy), picco = correlazione_di_fase(ga, gb)

    w1 = P + 2 * PAD
    w3 = G + 2 * PAD
    x1 = 2
    x2 = x1 + w1 + SP
    x3 = x2 + w1 + SP
    L, A = x3 + w3 + 2, Y0 + HC + 2
    fig, ax = d.tela(L, A)

    cornice(ax, x1, w1, "Blocchi")
    cornice(ax, x2, w1, "Gradiente")
    cornice(ax, x3, w3, "Correlazione di fase")

    ya, yb = Y0 + 21.5, Y0 + 3
    pannello(ax, x1 + PAD, ya, P, vis, "riferimento")
    pannello(ax, x1 + PAD, yb, P, nir, "da allineare")
    vmax = float(np.percentile(np.concatenate([ga.ravel(), gb.ravel()]), 99.5))
    pannello(ax, x2 + PAD, ya, P, ga, "$G_a$", vmax=vmax)
    pannello(ax, x2 + PAD, yb, P, gb, "$G_b$", vmax=vmax)
    for y in (ya, yb):
        d.freccia(ax, (x1 + PAD + P, y + P / 2), (x2 + PAD, y + P / 2))
    d.freccia(ax, (x2 + w1, Y0 + HC / 2 - 2), (x3, Y0 + HC / 2 - 2))

    # la superficie di correlazione: un ingrandimento attorno allo spostamento nullo
    n = r.shape[0]
    c, q = n // 2, ZOOM
    zona = r[c - q:c + q, c - q:c + q]
    yr = Y0 + 9
    pannello(ax, x3 + PAD, yr, G, zona, FORMULA, cmap='magma')
    s = G / (2 * q)                                   # unita' di disegno per pixel del blocco
    xc, yc = x3 + PAD + G / 2, yr + G / 2
    ax.plot([xc - G / 2, xc + G / 2], [yc, yc], color='white', lw=0.6, ls=(0, (3, 3)), zorder=4)
    ax.plot([xc, xc], [yc - G / 2, yc + G / 2], color='white', lw=0.6, ls=(0, (3, 3)), zorder=4)
    xp, yp = xc - dx * s, yc + dy * s                 # il picco: nell'immagine y cresce in basso
    ax.annotate('', xy=(xp, yp), xytext=(xc, yc), zorder=5,
                arrowprops=dict(arrowstyle='-|>', color='#7fd4ff', lw=1.6, shrinkA=0, shrinkB=8))
    ax.plot([xp], [yp], marker='o', ms=15, mfc='none', mec='#7fd4ff', mew=1.6, zorder=6)
    ax.text(x3 + w3 / 2, yr - 1.2,
            f"il picco dà lo spostamento del blocco:\n"
            f"{abs(dx) * FATTORE} px in orizzontale e {abs(dy) * FATTORE} px in verticale",
            ha='center', va='top', color=d.INK2, fontsize=CORPO_NOTA, linespacing=1.25)
    ax.set_xlim(0, L)
    ax.set_ylim(0, A)

    fig.savefig(os.path.join(d.QUI, 'schema_correlazione_di_fase.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print('schema_correlazione_di_fase.png |', f"spostamento nel blocco ridotto: dx {dx}, dy {dy} px; "
          f"sull'originale {dx * FATTORE}, {dy * FATTORE} px; altezza del picco {picco:.3f}")


if __name__ == '__main__':
    schema()
