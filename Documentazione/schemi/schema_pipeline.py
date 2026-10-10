"""
Lo schema della pipeline per la tesi (sezione «Pipeline»), solo in PNG.

    python schema_pipeline.py

Versione sintetica: flusso in orizzontale, un riquadro per passo con il solo
nome, senza descrizioni. Il mosaico non c'e'. Usa i colori di disegna_schemi.py,
con riquadri piu' stretti e caratteri piu' grandi, perche' l'immagine e' larga
e sulla pagina viene rimpicciolita.
"""

import os

import disegna_schemi as d
from disegna_schemi import plt, FancyBboxPatch

LB, HB, SP = 16, 8, 4          # riquadro: larghezza, altezza; spazio fra due riquadri
PAD = 2.5                      # margine dentro le cornici
Y0, HC = 2, 38                 # cornici: base e altezza
CORPO, CORPO_CORNICE, CORPO_NOTA = 12, 11.5, 9.5


def riquadro(ax, x, y, testo, stile):
    """Un riquadro col solo nome, centrato. Torna (x, y, w, h)."""
    fc, ec, ls = d.STILI[stile]
    ax.add_patch(FancyBboxPatch((x, y), LB, HB, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.3, ls=ls))
    ax.text(x + LB / 2, y + HB / 2, testo, ha='center', va='center', color=d.INK,
            fontsize=CORPO, fontweight='bold', linespacing=1.15)
    return x, y, LB, HB


def cornice(ax, x, w, etichetta):
    ax.add_patch(FancyBboxPatch((x, Y0), w, HC, boxstyle="round,pad=0,rounding_size=1.2",
                                fc='#fafaf8', ec='#c3c2b7', lw=0.9))
    ax.text(x + w / 2, Y0 + HC - 2.6, etichetta, ha='center', va='center', color=d.INK2,
            fontsize=CORPO_CORNICE, fontweight='bold')


def schema():
    # larghezze delle tre cornici: una colonna, tre passi, due passi
    w1 = LB + 2 * PAD
    w2 = 3 * LB + 2 * SP + 2 * PAD
    w3 = 2 * LB + SP + 2 * PAD
    x1, x2 = 2, 2 + w1 + SP
    x3 = x2 + w2 + SP
    L, A = x3 + w3 + 2, Y0 + HC + 2
    fig, ax = d.tela(L, A)

    cornice(ax, x1, w1, "1 · Preparazione")
    cornice(ax, x2, w2, "2 · Elaborazione")
    cornice(ax, x3, w3, "3 · Esame")

    # 1. gli ingressi, in colonna
    yb1 = Y0 + 2.5
    passo = HB + 2
    opz = riquadro(ax, x1 + PAD, yb1, "Opzioni", 'facoltativo')
    imm = riquadro(ax, x1 + PAD, yb1 + passo, "Immagini\nda allineare", 'dato')
    rif = riquadro(ax, x1 + PAD, yb1 + 2 * passo, "Riferimento", 'dato')
    ym = imm[1] + HB / 2           # la riga del flusso

    # 2. i tre passi, in riga
    xs = x2 + PAD
    stima = riquadro(ax, xs, imm[1], "Stima", 'processo')
    ver = riquadro(ax, xs + LB + SP, imm[1], "Verifica", 'controllo')
    scr = riquadro(ax, xs + 2 * (LB + SP), imm[1], "Scrittura", 'processo')

    # 3. l'esito
    xe = x3 + PAD
    rie = riquadro(ax, xe, imm[1], "Riepilogo", 'controllo')
    usc = riquadro(ax, xe + LB + SP, imm[1], "Immagini\nallineate", 'risultato')

    d.freccia(ax, (x1 + w1, ym), (stima[0], ym))
    d.a_destra(ax, stima, ver)
    d.a_destra(ax, ver, scr)
    d.a_destra(ax, scr, rie)
    d.a_destra(ax, rie, usc)

    # il ripiego: dalla verifica si torna alla stima, passando sotto
    yr = imm[1] - 4
    xa, xb = ver[0] + LB / 2, stima[0] + LB / 2
    ax.plot([xa, xa, xb], [ver[1], yr, yr], color=d.INK2, lw=1.2)
    d.freccia(ax, (xb, yr), (xb, stima[1]))
    ax.text((xa + xb) / 2, yr - 2.0, "se l'aggancio è fallito", ha='center', va='center',
            color=d.INK2, fontsize=CORPO_NOTA)

    fig.savefig(os.path.join(d.QUI, 'schema_pipeline_sintetico.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print('schema_pipeline_sintetico.png')


if __name__ == '__main__':
    schema()
