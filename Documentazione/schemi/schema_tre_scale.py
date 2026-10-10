"""
Lo schema della stima a tre scale, per la tesi (sezione «La stima a tre
scale»), solo in PNG.

    python schema_tre_scale.py

Stesso stile sintetico di schema_pipeline.py. Sotto ogni scala la griglia dei
blocchi e' disegnata sopra lo stesso ritaglio del visibile, un quadrato di
2048 px dell'originale (ritaglio_vis_2048px.png): le griglie sono in scala
vera, 512, 1024 e 256 px di lato. Il ritaglio non viene deformato.
"""

import os

import matplotlib.image as mpimg
import disegna_schemi as d
from disegna_schemi import plt, FancyBboxPatch
from matplotlib.patches import Rectangle
from schema_pipeline import riquadro, LB, HB, SP, CORPO_NOTA, CORPO_CORNICE

PAD = 3.5
LATO = 16                      # lato del ritaglio disegnato (vale 2048 px dell'originale)
Y0, HC = 2, 45
RITAGLIO_PX = 2048


def cornice(ax, x, w, etichetta):
    ax.add_patch(FancyBboxPatch((x, Y0), w, HC, boxstyle="round,pad=0,rounding_size=1.2",
                                fc='#fafaf8', ec='#c3c2b7', lw=0.9))
    ax.text(x + w / 2, Y0 + HC - 2.6, etichetta, ha='center', va='center', color=d.INK2,
            fontsize=CORPO_CORNICE, fontweight='bold')


def griglia(ax, xc, y, blocco_px, immagine, colore):
    """Il ritaglio, centrato in xc, con sopra i blocchi di `blocco_px` px, in scala vera."""
    x = xc - LATO / 2
    ax.imshow(immagine, extent=(x, x + LATO, y, y + LATO), aspect='auto', zorder=1)
    n = RITAGLIO_PX // blocco_px
    for i in range(1, n):
        t = i * LATO / n
        ax.plot([x + t, x + t], [y, y + LATO], color='white', lw=1.0, zorder=2)
        ax.plot([x, x + LATO], [y + t, y + t], color='white', lw=1.0, zorder=2)
    ax.add_patch(Rectangle((x, y), LATO, LATO, fc='none', ec=colore, lw=1.4, zorder=3))


def schema():
    immagine = mpimg.imread(os.path.join(d.QUI, 'ritaglio_vis_2048px.png'))
    w = LB + 2 * PAD
    xi = 2                                     # ingresso
    x1 = xi + LB + SP
    x2, x3 = x1 + w + SP, x1 + 2 * (w + SP)
    xu = x3 + w + SP                           # uscita
    L, A = xu + LB + 2, Y0 + HC + 2
    fig, ax = d.tela(L, A)

    cornice(ax, x1, w, "Scala 1")
    cornice(ax, x2, w, "Scala 2")
    cornice(ax, x3, w, "Scala 3")

    yf = Y0 + HC - 6 - HB                      # riga del flusso
    ing = riquadro(ax, xi, yf, "Copie\nridotte", 'dato')
    posa = riquadro(ax, x1 + PAD, yf, "Posa\n(affine)", 'processo')
    gros = riquadro(ax, x2 + PAD, yf, "Campo\ngrossolano", 'processo')
    fine = riquadro(ax, x3 + PAD, yf, "Campo\nfine", 'processo')
    usc = riquadro(ax, xu, yf, "Trasformazione", 'risultato')
    d.a_destra(ax, ing, posa)
    d.a_destra(ax, posa, gros)
    d.a_destra(ax, gros, fine)
    d.a_destra(ax, fine, usc)

    blu = d.STILI['processo'][1]
    yg = Y0 + 9
    for x0, blocco, testo in ((x1, 512, "blocchi da 512 px"),
                              (x2, 1024, "blocchi da 1024 px, poi 512\nsolo se supera i controlli"),
                              (x3, 256, "blocchi da 256 px\nsolo se supera i controlli")):
        griglia(ax, x0 + w / 2, yg, blocco, immagine, blu)
        ax.text(x0 + w / 2, yg - 1.2, testo, ha='center', va='top', color=d.INK2,
                fontsize=CORPO_NOTA, linespacing=1.25)
    ax.set_xlim(0, L)
    ax.set_ylim(0, A)

    fig.savefig(os.path.join(d.QUI, 'schema_tre_scale_catena.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print('schema_tre_scale_catena.png')


if __name__ == '__main__':
    schema()
