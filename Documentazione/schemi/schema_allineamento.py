"""
Lo schema dell'allineamento di una coppia, per la tesi (sezione «Allineamento»),
solo in PNG.

    python schema_allineamento.py

Stesso stile sintetico di schema_pipeline.py: flusso in orizzontale, un
riquadro per passo con il solo nome.
"""

import os

import disegna_schemi as d
from disegna_schemi import plt
from schema_pipeline import riquadro, cornice, LB, HB, SP, PAD, Y0, HC, CORPO_NOTA


def schema():
    # tre cornici: gli originali, il lavoro sulle copie ridotte, il lavoro a piena risoluzione
    w1 = LB + 2 * PAD
    w2 = 2 * LB + SP + 2 * PAD
    w3 = 3 * LB + 2 * SP + 2 * PAD
    x1, x2 = 2, 2 + w1 + SP
    x3 = x2 + w2 + SP
    L, A = x3 + w3 + 2, Y0 + HC + 2
    fig, ax = d.tela(L, A)

    cornice(ax, x1, w1, "Originali")
    cornice(ax, x2, w2, "Sulle copie ridotte")
    cornice(ax, x3, w3, "A piena risoluzione")

    yb = Y0 + 2.5                  # riga bassa
    passo = HB + 2
    yf = yb + passo                # riga del flusso
    ym = yf + HB / 2

    # gli ingressi, a cavallo della riga del flusso
    riquadro(ax, x1 + PAD, yf + passo / 2, "Riferimento", 'dato')
    riquadro(ax, x1 + PAD, yf - passo / 2, "Immagine\nda allineare", 'dato')

    # sulle copie ridotte
    xs = x2 + PAD
    rid = riquadro(ax, xs, yf, "Riduzione", 'processo')
    stima = riquadro(ax, xs + LB + SP, yf, "Stima\na tre scale", 'processo')
    posa = riquadro(ax, xs + LB + SP, yb, "Posa\ngrossolana", 'facoltativo')

    # a piena risoluzione
    xe = x3 + PAD
    ver = riquadro(ax, xe, yf, "Verifica", 'controllo')
    scr = riquadro(ax, xe + LB + SP, yf, "Scrittura", 'processo')
    usc = riquadro(ax, xe + 2 * (LB + SP), yf, "Immagine\nallineata", 'risultato')

    d.freccia(ax, (x1 + w1, ym), (rid[0], ym))
    d.a_destra(ax, rid, stima)
    d.a_destra(ax, stima, ver)
    d.a_destra(ax, ver, scr)
    d.a_destra(ax, scr, usc)

    # il ripiego: dalla verifica alla posa grossolana, e da li' di nuovo alla stima
    xv, yp = ver[0] + LB / 2, posa[1] + HB / 2
    ax.plot([xv, xv], [ver[1], yp], color=d.INK2, lw=1.2)
    d.freccia(ax, (xv, yp), (posa[0] + LB, yp))
    d.freccia(ax, (posa[0] + LB / 2, posa[1] + HB), (posa[0] + LB / 2, stima[1]))
    ax.text(xv + 1.5, yp, "se l'aggancio\nè fallito", ha='left', va='center',
            color=d.INK2, fontsize=CORPO_NOTA, linespacing=1.15)

    fig.savefig(os.path.join(d.QUI, 'schema_allineamento_sintetico.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print('schema_allineamento_sintetico.png')


if __name__ == '__main__':
    schema()
