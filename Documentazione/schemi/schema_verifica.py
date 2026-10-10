"""
Lo schema della verifica, per la tesi (sezione «Verifica e misure d'errore»),
solo in PNG.

    python schema_verifica.py

Stesso stile sintetico di schema_pipeline.py. La miniatura mostra la griglia
iniziale di 8 x 8 finestre da 512 px sopra il visibile, in scala vera
(anteprima_vis.png e' l'originale ridotto di 16 volte, non deformato).
"""

import os

import matplotlib.image as mpimg
import disegna_schemi as d
from disegna_schemi import plt, FancyBboxPatch
from matplotlib.patches import Rectangle
from schema_pipeline import riquadro, LB, HB, SP, PAD, CORPO_NOTA, CORPO_CORNICE

Y0, HC = 2, 50
LARGO_PX, ALTO_PX = 15848, 19850       # dimensioni dell'originale
FINESTRA_PX, N = 512, 8
ALT = 24                               # altezza della miniatura disegnata


def cornice(ax, x, w, etichetta):
    ax.add_patch(FancyBboxPatch((x, Y0), w, HC, boxstyle="round,pad=0,rounding_size=1.2",
                                fc='#fafaf8', ec='#c3c2b7', lw=0.9))
    ax.text(x + w / 2, Y0 + HC - 2.6, etichetta, ha='center', va='center', color=d.INK2,
            fontsize=CORPO_CORNICE, fontweight='bold')


def miniatura(ax, xc, y, immagine, colore):
    """Il visibile con sopra le 8 x 8 finestre, in scala vera."""
    larg = ALT * LARGO_PX / ALTO_PX
    x = xc - larg / 2
    ax.imshow(immagine, extent=(x, x + larg, y, y + ALT), aspect='auto', zorder=1)
    lf = ALT * FINESTRA_PX / ALTO_PX
    for i in range(N):
        for j in range(N):
            cx = x + (i + 0.5) * larg / N
            cy = y + (j + 0.5) * ALT / N
            ax.add_patch(Rectangle((cx - lf / 2, cy - lf / 2), lf, lf, fc='#ffd84a', ec='#ffd84a',
                                   lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x, y), larg, ALT, fc='none', ec=colore, lw=1.2, zorder=3))


def schema():
    immagine = mpimg.imread(os.path.join(d.QUI, 'anteprima_vis.png'))
    w1 = LB + 2 * PAD
    w3 = 2 * LB + SP + 2 * PAD
    xi = 2
    x1 = xi + LB + SP
    x2 = x1 + w1 + SP
    x3 = x2 + w1 + SP
    xu = x3 + w3 + SP
    L, A = xu + LB + 2, Y0 + HC + 2
    fig, ax = d.tela(L, A)

    cornice(ax, x1, w1, "Finestre")
    cornice(ax, x2, w1, "Misura")
    cornice(ax, x3, w3, "Sintesi")

    yf = Y0 + HC - 6 - HB
    ym = yf + HB / 2
    ing = riquadro(ax, xi, yf, "Originali e\ntrasformazione", 'dato')
    gri = riquadro(ax, x1 + PAD, yf, "Griglia\ndi finestre", 'processo')
    res = riquadro(ax, x2 + PAD, yf, "Residuo\nper finestra", 'controllo')
    riquadro(ax, x3 + PAD, yf, "Mediana", 'controllo')
    riquadro(ax, x3 + PAD + LB + SP, yf, "Quote\noltre soglia", 'controllo')
    riquadro(ax, x3 + PAD, yf - HB - 2.5, "Copertura", 'controllo')
    riquadro(ax, x3 + PAD + LB + SP, yf - HB - 2.5, "Distorsione\nmedia", 'controllo')
    usc = riquadro(ax, xu, yf, "Giudizio", 'risultato')

    d.a_destra(ax, ing, gri)
    d.a_destra(ax, gri, res)
    d.freccia(ax, (res[0] + LB, ym), (x3, ym))
    d.freccia(ax, (x3 + w3, ym), (usc[0], ym))

    # la misura esterna: una strada a parte, che non entra nel giudizio del programma
    ys = Y0 + 20
    ax.plot([x3 + 1.5, x3 + w3 - 1.5], [ys, ys], color='#c3c2b7', lw=0.9, ls=(0, (4, 3)))
    ax.text(x3 + w3 / 2, ys + 1.9, "dalle finestre", ha='center', va='center',
            color=d.INK2, fontsize=CORPO_NOTA)
    ax.text(x3 + w3 / 2, ys - 1.9, "dai punti individuati a mano", ha='center', va='center',
            color=d.INK2, fontsize=CORPO_NOTA)
    yt = Y0 + 5
    pun = riquadro(ax, x2 + PAD, yt, "Punti\na mano", 'facoltativo')
    tre = riquadro(ax, x3 + PAD, yt, "Errore\nsui punti", 'facoltativo')
    riquadro(ax, x3 + PAD + LB + SP, yt, "Mediana\ne quote", 'facoltativo')
    d.a_destra(ax, pun, tre)
    d.freccia(ax, (tre[0] + LB, yt + HB / 2), (x3 + PAD + LB + SP, yt + HB / 2))

    yg = Y0 + 9
    miniatura(ax, x1 + w1 / 2, yg, immagine, d.STILI['processo'][1])
    ax.text(x1 + w1 / 2, yg - 1.2, "8 × 8 finestre da 512 px,\nin scala sul riferimento", ha='center', va='top',
            color=d.INK2, fontsize=CORPO_NOTA, linespacing=1.25)
    ax.set_xlim(0, L)
    ax.set_ylim(0, A)

    fig.savefig(os.path.join(d.QUI, 'schema_verifica_sintetico_2.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print('schema_verifica_sintetico_2.png')


if __name__ == '__main__':
    schema()
