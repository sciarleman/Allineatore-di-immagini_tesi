"""
I grafici del benchmark delle radiografie, da rx_risultati.json.

    python grafici_rx.py

Produce in figure/:
  rx_1_panoramica.png         TRE, linee del bordo e copertura per configurazione
  rx_2_coda.png               distribuzione del TRE: dove mediana e media divergono
  rx_3_misure_a_confronto.png le altre misure contro il TRE
  rx_4_mappa.png              dove stanno i dettagli che sbagliano di piu'
"""

import json
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.ticker import FuncFormatter         # noqa: E402
from scipy.stats import spearmanr                   # noqa: E402

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)
if PROG not in sys.path:
    sys.path.insert(0, PROG)
import image_io                                     # noqa: E402

FIG = os.path.join(QUI, 'figure')

# Tavolozza: superficie, inchiostri e i primi slot categoriali (i primi tre
# reggono anche quando compaiono tutti insieme, come nei grafici a dispersione).
SUPERFICIE = '#fcfcfb'
INK, INK2, MUTO = '#0b0b0b', '#52514e', '#898781'
GRIGLIA, ASSE = '#e1e0d9', '#c3c2b7'
CATEGORIE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
GRUPPI = {'nostro': ('il nostro allineatore', CATEGORIE[0]),
          'a punti': ('metodi classici a punti di controllo', CATEGORIE[1]),
          'concorrenti': ('metodi concorrenti (affini)', CATEGORIE[2])}
SEQUENZIALE = ['#9ec5f4', '#5598e7', '#256abf', '#0d366b']
CLIC = 3.9      # lo stesso dettaglio cliccato due volte: differenza mediana, px

plt.rcParams.update({
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE,
    'savefig.facecolor': SUPERFICIE, 'font.size': 9, 'font.family': 'DejaVu Sans',
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': ASSE,
    'xtick.color': MUTO, 'ytick.color': INK2, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRIGLIA, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False,
})


NUMERI = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(',', '.') if v >= 1 else f"{v:g}")


def carica():
    with open(os.path.join(QUI, 'rx_risultati.json'), encoding='utf-8') as fh:
        conf = json.load(fh)['configurazioni']
    with open(os.path.join(QUI, 'rx_giudici.json'), encoding='utf-8') as fh:
        giudici = json.load(fh)
    return conf, giudici


def _legenda_gruppi(fig, y=0.995):
    maniglie = [Line2D([], [], marker='o', ls='', ms=7, mfc=c, mec=SUPERFICIE, mew=1.5, label=n)
                for n, c in GRUPPI.values()]
    fig.legend(handles=maniglie, loc='upper left', ncol=3, bbox_to_anchor=(0.01, y),
               handletextpad=0.3, columnspacing=1.6)


def panoramica(conf):
    nomi = [c['nome'] for c in conf][::-1]
    y = np.arange(len(nomi))
    per_nome = {c['nome']: c for c in conf}
    fig, assi = plt.subplots(1, 3, figsize=(13, 6.6), sharey=True,
                             gridspec_kw={'width_ratios': [1.35, 1, 0.8], 'wspace': 0.08})
    a, b, c = assi
    for i, n in enumerate(nomi):
        v = per_nome[n]
        col = GRUPPI[v['gruppo']][1]
        if 'errore' in v:
            for ax in (a, b):
                ax.text(1.2, i, "non converge", va='center', color=MUTO, fontsize=8)
            continue
        pi = v['punti_interni']
        a.plot([pi['mediana'], pi['p90']], [i, i], color=col, lw=2, alpha=0.45,
               solid_capstyle='round')
        a.plot(pi['media'], i, marker='D', ms=5, mfc=SUPERFICIE, mec=col, mew=1.5)
        a.plot(pi['mediana'], i, marker='o', ms=8, mfc=col, mec=SUPERFICIE, mew=2)
        b.plot(v['linee']['mediana_assoluta'], i, marker='o', ms=8, mfc=col, mec=SUPERFICIE, mew=2)
        f = v['finestre']
        c.barh(i, f['misurabili'], height=0.55, color=col, alpha=0.9)
        c.text(f['misurabili'] + 6, i, f"{f['misurabili']}", va='center', color=INK2, fontsize=8)
    for ax in (a, b):
        ax.set_xscale('log')
        ax.set_xlim(1, 12000)
        ax.xaxis.set_major_formatter(NUMERI)
        ax.grid(axis='y', visible=False)
    a.axvline(CLIC, color=ASSE, lw=1)
    a.text(CLIC * 1.05, len(nomi) - 0.4, "ripetibilità\ndel clic", color=MUTO, fontsize=7.5,
           va='top')
    a.set_yticks(y)
    a.set_yticklabels(nomi)
    a.set_title("TRE sui 115 dettagli interni indipendenti")
    a.set_xlabel("px della RX (scala logaritmica)")
    a.legend(handles=[Line2D([], [], marker='o', ls='', ms=7, mfc=INK2, mec=SUPERFICIE, label='mediana'),
                      Line2D([], [], marker='D', ls='', ms=5, mfc=SUPERFICIE, mec=INK2, label='media'),
                      Line2D([], [], color=INK2, lw=2, alpha=0.45, label='fino al p90')],
             loc='lower right', fontsize=8)
    b.set_title("Distanza fra le linee del bordo")
    b.set_xlabel("px della RX, mediana (scala logaritmica)")
    c.set_title("Finestre misurabili")
    c.set_xlabel("su 671")
    c.set_xlim(0, 760)
    c.grid(axis='y', visible=False)
    # separatori fra i gruppi
    gruppi = [per_nome[n]['gruppo'] for n in nomi]
    for i in range(1, len(gruppi)):
        if gruppi[i] != gruppi[i - 1]:
            for ax in assi:
                ax.axhline(i - 0.5, color=ASSE, lw=0.8)
    _legenda_gruppi(fig)
    fig.text(0.01, 0.945, "VIS / RX Cappellochiaro: più a sinistra è meglio nei primi due "
             "pannelli, più lunga è meglio nel terzo.", color=INK2, fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.86, bottom=0.09)
    fig.savefig(os.path.join(FIG, 'rx_1_panoramica.png'), dpi=150)
    plt.close(fig)


def coda(conf, giudici):
    interni = np.array([p[4] >= 250 and p[6] for p in giudici['punti']])
    scelte = ['automatico del lotto', 'tre scale', 'tutto (+ fascia di bordo)',
              'tutto, punti anche come posa', 'solo punti: thin-plate spline', 'NGF + Powell']
    per_nome = {c['nome']: c for c in conf}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for k, n in enumerate(scelte):
        v = per_nome.get(n)
        if not v or 'errore' in v:
            continue
        e = np.sort(np.asarray(v['errori_punti'])[interni])
        q = np.arange(1, len(e) + 1) / len(e)
        col = CATEGORIE[k]
        ax.step(e, 100 * q, where='post', color=col, lw=2,
                label=f"{n}: mediana {np.median(e):.1f}, media {e.mean():.1f} px, "
                      f"oltre 10 px {100 * (e > 10).mean():.0f}%")
        ax.plot(e.mean(), 100 * (e <= e.mean()).mean(), marker='D', ms=6, mfc=SUPERFICIE,
                mec=col, mew=1.8)
    for s in (5, 10):
        ax.axvline(s, color=ASSE, lw=1)
        ax.text(s + 0.6, 3, f"{s} px", color=MUTO, fontsize=8)
    ax.axhline(50, color=ASSE, lw=1)
    ax.text(97, 51.5, "mediana", color=MUTO, fontsize=8, ha='right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("TRE sul dettaglio (px della RX); oltre 100 px le curve escono dal grafico")
    ax.set_ylabel("dettagli interni con errore fino a x (%)")
    ax.set_title("Quanto è lunga la coda: la mediana è dove la curva passa il 50%, "
                 "il rombo è la media")
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'rx_2_coda.png'), dpi=150)
    plt.close(fig)


def misure_a_confronto(conf):
    validi = [c for c in conf if 'errore' not in c]
    x = np.array([c['punti_interni']['mediana'] for c in validi])
    pannelli = [
        ("mediana delle finestre di verifica (px)",
         [c['finestre']['mediana'] for c in validi], True),
        ("finestre misurabili (su 671)", [c['finestre']['misurabili'] for c in validi], False),
        ("distorsione media dei dettagli interni (px, modulo del vettore)",
         [float(np.hypot(c['punti_interni']['distorsione_x'], c['punti_interni']['distorsione_y']))
          for c in validi], True),
        ("costo MIND dopo l'allineamento (adimensionale)",
         [c.get('mind') for c in validi], False),
    ]
    fig, assi = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (titolo, yv, logy) in zip(assi.ravel(), pannelli):
        yv = np.array([np.nan if v is None else v for v in yv], float)
        ok = np.isfinite(yv)
        for c, xi, yi in zip(validi, x, yv):
            if np.isfinite(yi):
                ax.plot(xi, yi, marker='o', ms=8, mfc=GRUPPI[c['gruppo']][1],
                        mec=SUPERFICIE, mew=2)
        rho = spearmanr(x[ok], yv[ok]).correlation if ok.sum() > 3 else np.nan
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(NUMERI)
        if logy:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(NUMERI)
        ax.set_title(f"{titolo}\naccordo con il TRE (Spearman): {rho:+.2f}", fontsize=9.5)
        ax.set_xlabel("TRE sui dettagli interni, mediana (px)")
        # le due configurazioni di riferimento, etichettate
        for c, xi, yi in zip(validi, x, yv):
            if c['nome'] in ('tutto (+ fascia di bordo)', 'automatico del lotto',
                             'SIFT + RANSAC', 'MIRA (Mattes MI, affine)') and np.isfinite(yi):
                ax.annotate(c['nome'].split(' (')[0], (xi, yi), xytext=(6, 4),
                            textcoords='offset points', fontsize=7.5, color=INK2)
    _legenda_gruppi(fig, y=0.998)
    fig.suptitle("Le altre misure seguono il TRE? (tutte le configurazioni riuscite)",
                 x=0.01, y=0.955, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, 'rx_3_misure_a_confronto.png'), dpi=150)
    plt.close(fig)


def mappa(conf, giudici, vis, maschera_png):
    v = next(c for c in conf if c['nome'] == 'tutto (+ fascia di bordo)')
    img, _ = image_io.leggi_ridotta(vis, 1400, in_grigio=True, verbose=False)
    img = image_io.a_8bit(img).astype(float)
    info = image_io.info(vis)
    H, W = float(info.height), float(info.width)
    fondo = 255 - 0.35 * (255 - img)            # schiarito: i punti stanno sopra
    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    ax.imshow(fondo, cmap='gray', vmin=0, vmax=255, extent=(0, W, H, 0))
    m = cv2.imread(maschera_png, cv2.IMREAD_GRAYSCALE)
    cont, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cc in cont:
        cc = cc.reshape(-1, 2).astype(float) * [W / m.shape[1], H / m.shape[0]]
        ax.plot(*np.vstack([cc, cc[:1]]).T, color=MUTO, lw=1)
    lato = 500
    for x0, y0 in v['finestre']['posizioni']:
        ax.add_patch(plt.Rectangle((x0 - lato / 2, y0 - lato / 2), lato, lato, color='#b7d3f6',
                                   alpha=0.55, lw=0))
    P = np.array([p[:2] for p in giudici['punti']])
    interni = np.array([p[4] >= 250 and p[6] for p in giudici['punti']])
    e = np.asarray(v['errori_punti'])
    soglie = [(0, 5), (5, 10), (10, 20), (20, 1e9)]
    nomi = ["fino a 5 px", "5-10 px", "10-20 px", "oltre 20 px"]
    for (lo, hi), col, n, ms in zip(soglie, SEQUENZIALE, nomi, (6, 7, 8, 10)):
        s = interni & (e >= lo) & (e < hi)
        ax.plot(P[s, 0], P[s, 1], ls='', marker='o', ms=ms, mfc=col, mec=SUPERFICIE, mew=1.5,
                label=f"{n} ({s.sum()})")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color='#b7d3f6', alpha=0.55,
                               label='finestre di verifica misurabili'))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Configurazione completa: TRE dei 115 dettagli interni.\n"
                 "Quasi tutti gli errori oltre 20 px cadono fuori dalle finestre misurabili.",
                 fontsize=10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'rx_4_mappa.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    conf, giudici = carica()
    panoramica(conf)
    coda(conf, giudici)
    misure_a_confronto(conf)
    import radiografie
    mappa(conf, giudici, radiografie.VIS, radiografie.MASCHERA)
    print("figure scritte in", FIG)


if __name__ == '__main__':
    main()
