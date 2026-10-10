"""
Gli schemi riassuntivi del programma, per la tesi.

    python disegna_schemi.py

Scrive ogni schema in tre formati nella cartella di questo file: PNG (per
Word), PDF e SVG (vettoriali, per LaTeX o per ritoccarli a mano).

Gli schemi descrivono il programma a livello di passaggi e di funzioni: ogni
riquadro dice cosa succede e, in piccolo, quale funzione lo fa.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

QUI = os.path.dirname(os.path.abspath(__file__))

INK, INK2, MUTO = '#0b0b0b', '#52514e', '#898781'
STILI = {
    # nome: (riempimento, bordo, tratteggio)
    'dato': ('#f3f2ee', '#898781', '-'),
    'processo': ('#e6f0fc', '#2a78d6', '-'),
    'controllo': ('#fdece3', '#eb6834', '-'),
    'facoltativo': ('#ffffff', '#2a78d6', (0, (4, 2))),
    'risultato': ('#e3f5ee', '#1baf7a', '-'),
}
U = 0.1          # pollici per unita' di disegno
RIGA = 2.3       # interlinea, in unita'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def tela(larghezza, altezza):
    fig = plt.figure(figsize=(larghezza * U, altezza * U))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, larghezza)
    ax.set_ylim(0, altezza)
    ax.axis('off')
    return fig, ax


def altezza_scatola(righe, funzione):
    return 3.4 + RIGA * len(righe) + (2.4 if funzione else 0) + 1.0


def scatola(ax, x, y_alto, w, titolo, righe=(), funzione=None, stile='processo', numero=None,
            h=None):
    """Un riquadro con titolo, qualche riga e, in piccolo, la funzione. Torna (x, y, w, h)."""
    h = h or altezza_scatola(righe, funzione)
    y = y_alto - h
    fc, ec, ls = STILI[stile]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.1, ls=ls))
    tx = x + 1.6
    if numero is not None:
        ax.text(x + 2.4, y_alto - 2.3, str(numero), ha='center', va='center', color='white',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.25', fc=ec, ec='none'))
        tx = x + 5.2
    ax.text(tx, y_alto - 2.3, titolo, ha='left', va='center', color=INK, fontweight='bold',
            fontsize=8.6)
    for i, r in enumerate(righe):
        ax.text(x + 1.6, y_alto - 3.4 - RIGA * (i + 0.6), r, ha='left', va='center',
                color=INK2, fontsize=7.8)
    if funzione:
        ax.text(x + 1.6, y + 1.7, funzione, ha='left', va='center', color=MUTO,
                fontsize=6.9, family='DejaVu Sans Mono')
    return x, y, w, h


def freccia(ax, a, b, testo=None, stile='-|>', curva=0.0, colore=INK2, lato='destra'):
    ax.annotate('', xy=b, xytext=a,
                arrowprops=dict(arrowstyle=stile, color=colore, lw=1.2, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={curva}"))
    if testo:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx = 1.0 if lato == 'destra' else -1.0
        ax.text(mx + dx, my, testo, ha='left' if lato == 'destra' else 'right', va='center',
                color=MUTO, fontsize=7.2)


def giu(ax, sopra, sotto, testo=None, x=None):
    """Freccia dal fondo di un riquadro alla cima di quello sotto."""
    xs = x if x is not None else sopra[0] + sopra[2] / 2
    freccia(ax, (xs, sopra[1]), (xs, sotto[1] + sotto[3]), testo)


def a_destra(ax, sinistra, destra, testo=None, y=None):
    ys = y if y is not None else sinistra[1] + sinistra[3] / 2
    freccia(ax, (sinistra[0] + sinistra[2], ys), (destra[0], ys), testo)


def titolo(ax, larghezza, altezza, testo, sotto=None):
    ax.text(1.5, altezza - 2.2, testo, ha='left', va='center', color=INK, fontsize=11,
            fontweight='bold')
    if sotto:
        ax.text(1.5, altezza - 5.2, sotto, ha='left', va='center', color=INK2, fontsize=8.2)


def legenda(ax, x, y, voci):
    for i, (stile, testo) in enumerate(voci):
        fc, ec, ls = STILI[stile]
        ax.add_patch(FancyBboxPatch((x, y - 0.9), 3.2, 1.8,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=fc, ec=ec, lw=1.0, ls=ls))
        ax.text(x + 4.2, y, testo, ha='left', va='center', color=INK2, fontsize=7.4)
        x += 7.0 + 0.82 * len(testo)


def cornice(ax, x, y, w, h, etichetta):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc='#fafaf8', ec='#c3c2b7', lw=0.9))
    ax.text(x + 1.4, y + h - 1.9, etichetta, ha='left', va='center', color=INK2,
            fontsize=8.4, fontweight='bold')


def salva(fig, nome):
    for est in ('png', 'pdf', 'svg'):
        try:
            fig.savefig(os.path.join(QUI, f"{nome}.{est}"), dpi=220, facecolor='white')
        except OSError as exc:
            # un file aperto in un altro programma non si puo' sovrascrivere
            print(f"  [non scritto] {nome}.{est}: {exc}")
    plt.close(fig)
    print(nome)


LEGENDA = [('dato', 'dati'), ('processo', 'elaborazione'), ('controllo', 'controllo o misura'),
           ('facoltativo', 'passo facoltativo'), ('risultato', 'prodotto finale')]


# ---------------------------------------------------------------------------
def _griglia(ax, x, y, lato, n, frecce=(), colore='#2a78d6'):
    """Una griglia n x n di blocchi, con qualche freccia di spostamento."""
    for i in range(n + 1):
        ax.plot([x + i * lato / n] * 2, [y, y + lato], color=colore, lw=0.7)
        ax.plot([x, x + lato], [y + i * lato / n] * 2, color=colore, lw=0.7)
    c = lato / n
    for (i, j, dx, dy) in frecce:
        cx, cy = x + (i + 0.5) * c, y + (j + 0.5) * c
        ax.annotate('', xy=(cx + dx * c, cy + dy * c), xytext=(cx - dx * c, cy - dy * c),
                    arrowprops=dict(arrowstyle='-|>', color='#eb6834', lw=0.9,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))


def schema_0_sintesi():
    from matplotlib.patches import Rectangle, Polygon
    L, A = 159, 60
    fig, ax = tela(L, A)
    titolo(ax, L, A, "La pipeline in sintesi",
           "Dal riferimento e da un'altra ripresa dello stesso dipinto a un'immagine "
           "sovrapponibile pixel per pixel.")
    w, h, gap = 20, 37.5, 2.4
    y_alto = A - 9
    blu, ar, gr, ve = '#2a78d6', '#eb6834', '#898781', '#1baf7a'

    def pannello(k, numero, nome, righe, stile):
        x = 1.8 + k * (w + gap)
        fc, ec, ls = STILI[stile]
        ax.add_patch(FancyBboxPatch((x, y_alto - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=1.1, ls=ls))
        tx = x + 1.4
        if numero:
            ax.text(x + 2.3, y_alto - 2.4, str(numero), ha='center', va='center', color='white',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.25', fc=ec, ec='none'))
            tx = x + 4.8
        ax.text(tx, y_alto - 2.4, nome, ha='left', va='center', color=INK, fontweight='bold',
                fontsize=8.6)
        for i, r in enumerate(righe):
            ax.text(x + 1.4, y_alto - 24.5 - RIGA * i, r, ha='left', va='center', color=INK2,
                    fontsize=7.5)
        if k:
            freccia(ax, (x - gap, y_alto - 13), (x, y_alto - 13))
        return x + 2.0, y_alto - 20.5      # angolo in basso a sinistra del disegno (16 x 15)

    # ingresso: due riprese fuori registro
    ix, iy = pannello(0, None, "Due riprese", ["riferimento (VIS)", "e immagine da", "allineare: fuori",
                                               "registro di decine", "di pixel"], 'dato')
    ax.add_patch(Rectangle((ix + 1, iy + 1), 11, 12.5, fc='#dcdbd5', ec=gr, lw=1))
    ax.add_patch(Polygon([(ix + 4, iy + 2.2), (ix + 15, iy + 3.2), (ix + 14.2, iy + 15),
                          (ix + 3.2, iy + 14)], closed=True, fc='none', ec=blu, lw=1.2))

    ix, iy = pannello(1, 1, "Lettura", ["file da 1–3 GB aperti", "senza caricarli:",
                                        "si legge solo il pezzo", "che serve; stima su",
                                        "copie ridotte"], 'processo')
    ax.add_patch(Rectangle((ix, iy + 1), 10, 13, fc='white', ec=blu, lw=1))
    ax.add_patch(Rectangle((ix + 5, iy + 7), 2.5, 2.5, fc=blu, ec=blu, alpha=0.55))
    ax.add_patch(Rectangle((ix + 11.5, iy + 1), 4.5, 5.8, fc='white', ec=blu, lw=1))
    ax.text(ix + 13.7, iy + 8.2, "½", ha='center', color=INK2, fontsize=7.5)

    ix, iy = pannello(2, 2, "Stima a tre scale", ["posa (affine) →", "campo grossolano →",
                                                  "campo fine: blocchi", "sempre più piccoli,",
                                                  "sul gradiente"], 'processo')
    _griglia(ax, ix - 0.4, iy + 5, 5, 2, [(0, 0, 0.3, 0.12), (1, 1, 0.3, 0.12)])
    _griglia(ax, ix + 5.5, iy + 5, 5, 4, [(0, 3, 0.3, 0.2), (2, 2, 0.32, -0.1), (3, 0, 0.2, 0.3),
                                         (1, 0, -0.25, 0.2)])
    _griglia(ax, ix + 11.4, iy + 5, 5, 6, [(0, 5, 0.3, 0.2), (2, 4, 0.3, -0.2), (4, 2, -0.2, 0.3),
                                          (5, 0, 0.3, 0.1), (1, 1, 0.1, 0.35), (3, 3, -0.3, 0.1)])
    for k_, n_ in enumerate(("posa", "grosso", "fine")):
        ax.text(ix + 2.1 + k_ * 5.9, iy + 3.2, n_, ha='center', color=MUTO, fontsize=6.6)

    ix, iy = pannello(3, 3, "Verifica", ["finestre indipendenti,", "a piena risoluzione:",
                                         "mediana, quote,", "copertura, distorsione"],
                      'controllo')
    ax.add_patch(Rectangle((ix + 1, iy + 1), 14, 13.5, fc='white', ec=ar, lw=1))
    for i in range(4):
        for j in range(4):
            ok = (i, j) not in ((3, 3), (0, 3), (3, 0))
            ax.add_patch(Rectangle((ix + 2.3 + i * 3.1, iy + 2.2 + j * 3.0), 2.1, 2.1,
                                   fc=ve if ok else 'white', ec=ve if ok else gr,
                                   alpha=0.75 if ok else 1, lw=0.8))

    ix, iy = pannello(4, 4, "Solo radiografie", ["fascia di bordo e", "punti a mano: questi",
                                                 "misurano l'errore", "(TRE)"], 'facoltativo')
    ott = [(4, 0.5), (12, 0.5), (15.5, 4.5), (15.5, 10.5), (12, 14.5), (4, 14.5), (0.5, 10.5),
           (0.5, 4.5)]
    ax.add_patch(Polygon([(ix + a, iy + b) for a, b in ott], closed=True, fc='white', ec=blu,
                         lw=3.2, alpha=0.35))
    ax.add_patch(Polygon([(ix + a, iy + b) for a, b in ott], closed=True, fc='none', ec=blu, lw=1))
    for a, b in ((5, 5), (10, 9), (7, 11), (11, 4.5), (4.5, 8.5)):
        ax.plot(ix + a, iy + b, marker='+', color=ar, ms=6, mew=1.4)

    ix, iy = pannello(5, 5, "Parametri", ["trasformazione,", "misure e immagini", "di controllo:",
                                          "tutto rifacibile"], 'processo')
    ax.add_patch(Rectangle((ix + 3, iy + 1), 10, 13.5, fc='white', ec=blu, lw=1))
    for j, lung in enumerate((7, 5, 6.5, 4, 6)):
        ax.plot([ix + 4.5, ix + 4.5 + lung], [iy + 12 - j * 2.3] * 2, color=blu, lw=1.1, alpha=0.6)

    ix, iy = pannello(6, 6, "Scrittura", ["a tasselli, un solo", "ricampionamento:",
                                          "immagine allineata,", "grande quanto il",
                                          "riferimento"], 'risultato')
    for i in range(4):
        for j in range(4):
            fatto = j > 1 or (j == 1 and i < 2)
            ax.add_patch(Rectangle((ix + 1.2 + i * 3.4, iy + 1 + j * 3.4), 3.4, 3.4,
                                   fc='#bfe6d6' if fatto else 'white', ec=ve, lw=0.8))

    ax.text(1.8, 9.6, "Risultati sul Fra Galgario: bande fotografiche 0.2–0.6 px di residuo; "
            "SWIR e radiografie circa 1 px dove si misura; sulla radiografia 5 px sui dettagli "
            "cliccati a mano.", color=INK2, fontsize=7.9)
    ax.text(1.8, 6.6, "Il campo grossolano si applica solo se la deformazione supera la portata "
            "del campo fine (64 px): SWIR e radiografie.", color=INK2, fontsize=7.9)
    legenda(ax, 1.8, 2.4, LEGENDA)
    salva(fig, 'schema_0_pipeline_in_sintesi')


# ---------------------------------------------------------------------------
def schema_1_panoramica():
    L, A = 118, 98
    fig, ax = tela(L, A)
    titolo(ax, L, A, "1. Il sistema nel suo insieme",
           "Tre programmi indipendenti, ciascuno con la sua finestra; l'allineatore è il cuore.")

    riprese = scatola(ax, 2, 84, 30, "Riprese del dipinto",
                      ["visibile (VIS): il riferimento", "UVF, UVR, NIR850, NIR950, SWIR",
                       "radiografie (RX)", "TIFF da 0.9–3 GB l'una"], stile='dato')
    tasselli = scatola(ax, 2, 25, 30, "Tasselli di una ripresa",
                       ["griglia di scatti parziali", "da ricomporre"], stile='dato')

    cornice(ax, 36, 50, 44, 40, "ALLINEATORE")
    lotto = scatola(ax, 38, 85, 19.5, "Lotto", ["le bande fotografiche,", "tutte insieme"],
                    "align_batch\n.allinea_lotto", h=13.5)
    rx = scatola(ax, 58.5, 85, 19.5, "Sezione RX", ["una radiografia", "alla volta"],
                 "finestra_rx\n.allinea_rx", h=13.5)
    coppia = scatola(ax, 38, 67.5, 40, "Allineamento di una coppia",
                     ["stima a tre scale → verifica → scrittura", "(schemi 2, 3 e 4)"],
                     "align_paintings.align")
    giu(ax, lotto, coppia)
    giu(ax, rx, coppia)

    falso = scatola(ax, 36, 44, 44, "FALSO COLORE",
                    ["IRFC: IR → R, R → G, G → B", "UVFC: G → R, B → G, UV → B",
                     "ricontrolla l'allineamento prima di comporre"],
                    "falso_colore.componi_molte")
    mosaico = scatola(ax, 36, 25, 44, "MOSAICO",
                      ["misura le sovrapposizioni, stima le posizioni,", "verifica le cuciture"],
                      "mosaico.componi_mosaico")

    parametri = scatola(ax, 86, 88, 30, "Parametri e report",
                        ["trasformazione (JSON + campo)", "misure d'errore, diagnostica"],
                        stile='risultato')
    allineate = scatola(ax, 86, 74, 30, "Immagini allineate",
                        ["sovrapponibili al VIS pixel a pixel", "TIFF, PNG o JPEG a scelta"],
                        stile='risultato')
    fc_out = scatola(ax, 86, 44, 30, "Falsi colori", ["IRFC e UVFC a piena risoluzione"],
                     stile='risultato')
    mos_out = scatola(ax, 86, 25, 30, "Mosaico", ["immagine unica o livelli", "per Photoshop"],
                      stile='risultato')

    freccia(ax, (32, 78), (36, 78))
    freccia(ax, (80, 83.5), (86, 83.5))
    freccia(ax, (80, 69.5), (86, 69.5))
    # le bande allineate e il visibile entrano nel falso colore
    xa = allineate[0] + allineate[2] / 2
    ax.plot([xa, xa, 58, 58], [allineate[1], 47.5, 47.5, 47.5], color=INK2, lw=1.2)
    freccia(ax, (58, 47.5), (58, 44))
    ax.text(xa - 1, 49, "bande allineate", ha='right', color=MUTO, fontsize=7.2)
    ax.plot([17, 17], [riprese[1], 37], color=INK2, lw=1.2)
    freccia(ax, (17, 37), (36, 37))
    ax.text(18, 38.3, "VIS", color=MUTO, fontsize=7.2)
    freccia(ax, (80, 40.5), (86, 40.5))
    freccia(ax, (32, 20.5), (36, 20.5))
    freccia(ax, (80, 20.5), (86, 20.5))
    legenda(ax, 2, 4, [LEGENDA[0], LEGENDA[1], LEGENDA[4]])
    salva(fig, 'schema_1_panoramica')


# ---------------------------------------------------------------------------
def schema_2_coppia():
    L, A = 104, 116
    fig, ax = tela(L, A)
    titolo(ax, L, A, "2. L'allineamento di una coppia di immagini",
           "align_paintings.align: dal riferimento e dall'immagine da allineare all'immagine "
           "allineata.")
    x, w = 4, 64
    y = A - 9
    passi = []

    def passo(*a, **k):
        nonlocal y
        r = scatola(ax, x, y, w, *a, **k)
        if passi:
            giu(ax, passi[-1], r)
        passi.append(r)
        y = r[1] - 3.2
        return r

    passo("Ingresso", ["riferimento (VIS) e immagine da allineare, a piena risoluzione",
                       "eventuale zona del dipinto, disegnata a mano"], stile='dato')
    lettura = passo("Lettura", ["i TIFF si aprono mappati in memoria: si legge solo ciò che serve",
                                "per la stima si usano anteprime ridotte (fattore 2)"],
                    "image_io.apri · image_io.leggi_ridotta", numero=1)
    stima = passo("Stima della trasformazione",
                  ["tre scale: posa affine, campo grossolano, campo fine (schema 3)",
                   "risultato: un'affine più un campo di spostamenti"],
                  "block_matching.estimate_transform  →  Transform", numero=2)
    verifica = passo("Verifica a piena risoluzione",
                     ["finestre indipendenti dalla stima: residuo, quote, copertura (schema 4)"],
                     "validation.residual_check", stile='controllo', numero=3)
    corr = passo("Correzioni locali (solo radiografie)",
                 ["fascia di bordo: il contorno della zona portato sul bordo della tela",
                  "punti a mano: vincolo locale, oppure solo misura del TRE"],
                 "_correggi_con_fascia · _correggi_con_punti · _tre_lascia_fuori",
                 stile='facoltativo', numero=4)
    par = passo("Parametri, report e immagini di controllo",
                ["la trasformazione si salva: il risultato si può rifare dai soli parametri"],
                "Transform.save · validation.make_overlay", numero=5)
    scr = passo("Scrittura a piena risoluzione",
                ["a tasselli da 2048 px: la memoria non dipende dalla dimensione dell'immagine",
                 "un solo ricampionamento; margine attorno al foglio se l'immagine esce"],
                "scrivi_allineata · Transform.warp_tile_from · margine_necessario", numero=6)
    passo("Immagine allineata", ["grande quanto il riferimento, nel formato scelto"],
          stile='risultato')

    # ripiego: dalla verifica di nuovo alla stima
    xr = x + w
    ripiego = scatola(ax, 73, verifica[1] + verifica[3] + 11, 28, "Ripiego",
                      ["se il residuo supera 10 px", "si riparte da una posa",
                       "stimata su anteprime", "molto ridotte"],
                      "_ripiego_se_serve", stile='controllo')
    freccia(ax, (xr, verifica[1] + verifica[3] / 2), (87, ripiego[1]), curva=0.25)
    freccia(ax, (87, ripiego[1] + ripiego[3]), (xr, stima[1] + stima[3] / 2), curva=0.25)
    legenda(ax, 2, 1.8, LEGENDA)
    salva(fig, 'schema_2_allineamento_di_una_coppia')


# ---------------------------------------------------------------------------
def schema_3_tre_scale():
    L, A = 132, 106
    fig, ax = tela(L, A)
    titolo(ax, L, A, "3. La stima a tre scale",
           "block_matching.estimate_transform. Ogni scala misura con la correlazione di fase "
           "sul gradiente, a blocchi, e lavora sul risultato della precedente.")
    hb = 15.5
    gap = 3.0

    def fila(y_alto, etichetta, sotto, voci):
        cornice(ax, 2, y_alto - hb - 7.5, L - 4, hb + 11.5, etichetta)
        ax.text(3.4, y_alto + 0.2, sotto, color=MUTO, fontsize=7.4, va='center')
        n = len(voci)
        w = (L - 8 - gap * (n - 1)) / n
        fatte = []
        for i, v in enumerate(voci):
            r = scatola(ax, 4 + i * (w + gap), y_alto - 2, w, *v[:3], stile=v[3] if len(v) > 3
                        else 'processo', h=hb)
            if fatte:
                a_destra(ax, fatte[-1], r)
            fatte.append(r)
        return fatte

    y1 = A - 13
    s1 = fila(y1, "SCALA 1 · LA POSA",
              "blocchi da 512 px (a piena risoluzione), portata 128 px",
              [("Gradiente", ["modulo del gradiente:", "confrontabile fra bande", "diverse"],
                "gradient_magnitude"),
               ("Aggancio iniziale", ["una traslazione", "d'insieme"], "coarse_translation"),
               ("Misura a blocchi", ["spostamento e confidenza", "di ogni blocco",
                                     "(correlazione di fase)"], "match_blocks"),
               ("Affine robusta", ["traslazione, scala,", "rotazione, inclinazione;",
                                   "blocchi oltre 3 MAD scartati"], "fit_affine_robust")])
    y2 = y1 - hb - 15
    s2 = fila(y2, "SCALA 2 · IL CAMPO GROSSOLANO",
              "immagine raddrizzata con l'affine; blocchi da 512 px a metà risoluzione, "
              "portata 128 px",
              [("Misura a blocchi", ["sull'immagine già", "raddrizzata"], "match_blocks"),
               ("Filtro di coerenza", ["un blocco si scarta se", "discorda dai vicini",
                                       "oltre 3 σ"], "blocchi_concordi"),
               ("Campo liscio", ["lisciatura scelta per", "validazione incrociata"],
                "build_dense_field"),
               ("Si applica?", ["guadagno ≥ 10% sui blocchi", "non usati, e ampiezza",
                                "oltre 64 px"], "campo_grossolano", 'controllo')])
    y3 = y2 - hb - 15
    s3 = fila(y3, "SCALA 3 · IL CAMPO FINE",
              "immagine raddrizzata con affine e campo grossolano; blocchi da 256 px, "
              "portata 64 px",
              [("Misura a blocchi", ["griglia fitta: le", "deformazioni minute", "della tela"],
                "match_blocks"),
               ("Residuo affine + campo", ["media locale e", "lisciatura scelte per",
                                           "validazione incrociata"],
                "select_smoothing · build_dense_field"),
               ("Si applica?", ["solo se migliora i", "blocchi non usati"],
                "field_crossval_gain", 'controllo'),
               ("Trasformazione", ["affine + somma dei due", "campi, su una griglia:",
                                   "un solo ricampionamento"],
                "Transform.compose_residual", 'risultato')])
    for sopra, sotto in ((s1, s2), (s2, s3)):
        # dal fondo dell'ultimo riquadro, lungo il margine sinistro, fino al
        # fianco del primo riquadro della scala successiva: cosi' la linea non
        # passa sopra le etichette
        a, b = sopra[-1], sotto[0]
        xm, ym = a[0] + a[2] / 2, b[1] + b[3] / 2
        ax.plot([xm, xm, 0.9, 0.9], [a[1], a[1] - 5.6, a[1] - 5.6, ym], color=INK2, lw=1.2)
        freccia(ax, (0.9, ym), (b[0], ym))
    ax.text(3, 5.6, "Perché tre: la scala 3 non vede oltre 64 px, e l'affine della scala 1 "
            "scarta come anomali i blocchi con deformazioni grandi. La scala 2 copre la fascia "
            "da 64 a 128 px.", color=INK2, fontsize=7.8)
    legenda(ax, 3, 2.2, [LEGENDA[1], LEGENDA[2], LEGENDA[4]])
    salva(fig, 'schema_3_stima_a_tre_scale')


# ---------------------------------------------------------------------------
def schema_4_verifica():
    L, A = 124, 84
    fig, ax = tela(L, A)
    titolo(ax, L, A, "4. La verifica e le misure d'errore",
           "Quanto è buono l'allineamento, misurato in modo indipendente da come è stato "
           "stimato.")
    w = 37
    c1 = scatola(ax, 3, A - 9, w, "Verifica a finestre (tutte le bande)", [], stile='dato', h=4.6)
    a1 = scatola(ax, 3, c1[1] - 3, w, "Finestre a piena risoluzione",
                 ["griglia 8×8 da 512 px, dentro la", "zona del dipinto se è disegnata;",
                  "posizioni che non dipendono dalla stima"], "validation.residual_check",
                 numero=1)
    a2 = scatola(ax, 3, a1[1] - 3, w, "Residuo di ogni finestra",
                 ["si applica la trasformazione e si", "misura lo spostamento rimasto:",
                  "vettore (dx, dy) e confidenza"], "Transform.warp_tile_from",
                 stile='controllo', numero=2)
    a3 = scatola(ax, 3, a2[1] - 3, w, "Finestre affidabili",
                 ["si tengono quelle con confidenza", "sufficiente: sono la copertura"],
                 stile='controllo', numero=3)
    giu(ax, c1, a1); giu(ax, a1, a2); giu(ax, a2, a3)

    x2 = 44
    c2 = scatola(ax, x2, A - 9, w, "Le misure riportate", [], stile='dato', h=4.6)
    misure = [("Mediana, con intervallo al 90%", ["il valore tipico; l'intervallo dice",
                                                    "quanto dipende dalle finestre scelte"]),
              ("Quote oltre 1, 2, 5 e 10 px", ["la coda, che la mediana nasconde"]),
              ("Copertura", ["finestre misurabili su quelle tentate:", "se crolla, è un allarme"]),
              ("Distorsione media", ["lo spostamento medio come vettore:", "rivela un errore "
                                     "tutto dalla stessa parte"])]
    y = c2[1] - 3
    fatte = []
    for t, r in misure:
        b = scatola(ax, x2, y, w, t, r, stile='controllo')
        fatte.append(b)
        y = b[1] - 2.2
    giud = scatola(ax, x2, y - 0.8, w, "Giudizio", ["«allineata» oppure «da controllare»"],
                   "align_batch.valuta_qualita", stile='risultato')
    freccia(ax, (a3[0] + a3[2], a3[1] + a3[3] / 2), (x2, a3[1] + a3[3] / 2))
    giu(ax, fatte[-1], giud)

    x3 = 85
    c3 = scatola(ax, x3, A - 9, w - 1, "In più, per le radiografie", [], stile='dato', h=4.6)
    r1 = scatola(ax, x3, c3[1] - 3, w - 1, "Griglia fissa",
                 ["sempre le stesse 671 finestre:", "i tentativi sono confrontabili;",
                  "finestre oltre 10 px in gruppo o isolate"], "finestra_rx.verifica_rx",
                 stile='controllo')
    r2 = scatola(ax, x3, r1[1] - 3, w - 1, "TRE sui punti messi a mano",
                 ["scarto fra dove la trasformazione", "porta un dettaglio e dove è stato",
                  "cliccato: l'unico giudice esterno"], "_tre_lascia_fuori",
                 stile='controllo')
    r3 = scatola(ax, x3, r2[1] - 3, w - 1, "Distanza fra le linee del bordo",
                 ["contorno della zona sul visibile", "contro bordo della tela sulla RX"],
                 "campo_fascia_di_bordo", stile='controllo')
    r4 = scatola(ax, x3, r3[1] - 3, w - 1, "Verità nota",
                 ["test sintetico con deformazione", "imposta: errore vero 0.335 px"],
                 "test_pipeline.py", stile='dato')
    ax.text(3, 5.4, "La verifica a finestre è un controllo di coerenza: usa lo stesso "
            "principio della stima. Il TRE e il test sintetico sono i riferimenti esterni.",
            color=INK2, fontsize=7.8)
    legenda(ax, 2, 1.6, [LEGENDA[0], LEGENDA[1], LEGENDA[2], LEGENDA[4]])
    salva(fig, 'schema_4_verifica_e_misure')


if __name__ == '__main__':
    schema_0_sintesi()
    schema_1_panoramica()
    schema_2_coppia()
    schema_3_tre_scale()
    schema_4_verifica()
