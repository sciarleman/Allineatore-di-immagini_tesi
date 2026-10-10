"""
Produce le figure del confronto a partire da risultati.json.

Scelte di forma, per rendere i grafici leggibili senza spiegazioni:
- i metodi stanno sull'asse, scritti per esteso: l'identita' non e' affidata al
  colore, che serve solo a distinguere il nostro dagli altri;
- l'asse dei residui e' logaritmico perche' i valori coprono due ordini di
  grandezza (da 0.2 a oltre 100 px) e su scala lineare le differenze fra i
  metodi buoni sparirebbero;
- ogni barra porta il suo valore scritto: niente da stimare a occhio;
- un pannello per immagine, cosi' si vede subito che il risultato non dipende
  da quale banda si guarda.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

QUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(QUI, 'figure')
os.makedirs(FIG, exist_ok=True)

# tavolozza validata (vedi references/palette.md della guida di visualizzazione)
SUPERFICIE = '#fcfcfb'
INCHIOSTRO = '#0b0b0b'
SECONDARIO = '#52514e'
ATTENUATO = '#898781'
GRIGLIA = '#e1e0d9'
BLU = '#2a78d6'        # il nostro metodo
ARANCIO = '#eb6834'    # il nostro, solo affine
GRIGIO_BARRA = '#a9a7a0'   # gli altri metodi
ROSSO = '#d03b3b'      # fallimenti

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'figure.facecolor': SUPERFICIE,
    'axes.facecolor': SUPERFICIE,
    'axes.edgecolor': '#c3c2b7',
    'axes.labelcolor': SECONDARIO,
    'text.color': INCHIOSTRO,
    'xtick.color': ATTENUATO,
    'ytick.color': SECONDARIO,
    'axes.grid': True,
    'grid.color': GRIGLIA,
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,
})

ORDINE = ['nessun allineamento', 'SIFT + RANSAC', 'ECC', 'NGF + Powell',
          'NMI + Powell', 'MIND + Powell', 'MIRA (Mattes MI)',
          'Aligner (solo affine)', 'Aligner (completo)']


def carica():
    dati = json.load(open(os.path.join(QUI, 'risultati.json'), encoding='utf-8-sig'))
    per = {}
    for v in dati:
        per.setdefault(v['immagine'], {})[v['metodo']] = v
    return dati, per


def colore_di(metodo):
    if metodo == 'Aligner (completo)':
        return BLU
    if metodo == 'Aligner (solo affine)':
        return ARANCIO
    return GRIGIO_BARRA


# ---------------------------------------------------------------------------
# Figura 1: il residuo, per ogni immagine e ogni metodo
# ---------------------------------------------------------------------------

def figura_residui(per):
    immagini = list(per)
    n = len(immagini)
    righe = (n + 1) // 2
    fig, axes = plt.subplots(righe, 2, figsize=(15, 3.1 * righe), squeeze=False)

    for k, nome_img in enumerate(immagini):
        ax = axes[k // 2][k % 2]
        metodi_presenti = [m for m in ORDINE if m in per[nome_img]]
        valori, colori, etichette, note = [], [], [], []
        for m in metodi_presenti:
            v = per[nome_img][m]
            r = v.get('residuo_px')
            if r is None or not v.get('ok', True):
                valori.append(np.nan)
                note.append('non riuscito' if not v.get('ok', True) else 'non misurabile')
            else:
                valori.append(r)
                note.append('')
            colori.append(colore_di(m))
            etichette.append(m)

        y = np.arange(len(valori))
        finiti = [v if np.isfinite(v) else 0 for v in valori]
        ax.barh(y, finiti, color=colori, height=0.66, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(etichette, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlim(0.15, 400)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
        ax.set_xticks([0.2, 1, 5, 20, 100])
        ax.axvline(1.0, color=SECONDARIO, lw=1, ls='--', zorder=2)
        ax.set_title(nome_img, fontsize=11, color=INCHIOSTRO, loc='left', pad=6)
        ax.tick_params(axis='x', labelsize=8)

        for i, (v, nota) in enumerate(zip(valori, note)):
            if nota:
                ax.text(0.2, i, '  ' + nota, va='center', ha='left',
                        fontsize=8, color=ROSSO, zorder=4)
            else:
                ax.text(v * 1.12, i, f'{v:.2f}', va='center', ha='left',
                        fontsize=8.5, color=SECONDARIO, zorder=4)
        if k // 2 == righe - 1:
            ax.set_xlabel('disallineamento residuo (px, scala logaritmica)', fontsize=9)

    for k in range(n, righe * 2):
        axes[k // 2][k % 2].axis('off')

    fig.suptitle('Quanto restano fuori registro le immagini, dopo ogni metodo',
                 fontsize=14, color=INCHIOSTRO, x=0.012, ha='left', y=0.995)
    fig.text(0.012, 0.963,
             'Misurato su 25 finestre da 512 px a piena risoluzione, identiche per tutti i metodi. '
             'Piu\' corto = meglio. La linea tratteggiata e\' il pixel.',
             fontsize=9, color=SECONDARIO, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = os.path.join(FIG, '1_residui_per_immagine.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figura 2: i cinque criteri concordano?
# ---------------------------------------------------------------------------

def figura_criteri(per, nome_img='UVF'):
    if nome_img not in per:
        nome_img = list(per)[0]
    d = per[nome_img]
    metodi_presenti = [m for m in ORDINE if m in d and d[m].get('ok', True)
                       and d[m].get('mind') is not None]

    criteri = [('residuo_px', 'residuo in px', True, 'fase sui gradienti'),
               ('mind', 'costo MIND', True, 'auto-similarita\' locale'),
               ('nmi', 'NMI', False, 'informazione mutua'),
               ('mattes_mi', 'Mattes MI', True, 'il criterio di MIRA'),
               ('ncc_gradienti', 'NCC sui gradienti', False, 'correlazione')]

    fig, axes = plt.subplots(1, len(criteri), figsize=(16, 5.4), squeeze=False)
    for j, (chiave, titolo, meno_e_meglio, sottotitolo) in enumerate(criteri):
        ax = axes[0][j]
        vals = [d[m].get(chiave) for m in metodi_presenti]
        # si normalizza 0..1 dove 1 = il migliore, per poter affiancare criteri
        # con unita' incomparabili senza mai mescolarli sullo stesso asse
        v = np.array([x if x is not None else np.nan for x in vals], float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        punteggio = (hi - v) / (hi - lo) if meno_e_meglio else (v - lo) / (hi - lo)
        y = np.arange(len(metodi_presenti))
        ax.barh(y, punteggio, color=[colore_di(m) for m in metodi_presenti],
                height=0.66, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(metodi_presenti if j == 0 else [], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['peggiore', 'migliore'], fontsize=8)
        # il sottotitolo va SOPRA il titolo e con uno stacco esplicito: messo
        # appena sopra l'asse finiva sotto il titolo e i due si sovrapponevano
        ax.set_title(titolo, fontsize=10.5, color=INCHIOSTRO, pad=20)
        ax.text(0.5, 1.035, sottotitolo, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8, color=ATTENUATO)
        for i, (p_, raw) in enumerate(zip(punteggio, v)):
            if np.isfinite(raw):
                ax.text(p_ + 0.03, i, f'{raw:.3g}', va='center', fontsize=7.5,
                        color=SECONDARIO, zorder=4)

    fig.suptitle(f'Cinque criteri indipendenti, stessa risposta  —  {nome_img}',
                 fontsize=14, color=INCHIOSTRO, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.915,
             'Ogni pannello e\' un modo diverso di giudicare l\'allineamento, compreso quello che MIRA '
             'ottimizza.',
             fontsize=9, color=SECONDARIO, ha='left', va='top')
    fig.text(0.012, 0.885,
             'Le barre sono normalizzate perche\' le unita\' non sono confrontabili; '
             'il numero accanto e\' il valore vero.',
             fontsize=9, color=SECONDARIO, ha='left', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.845])
    p = os.path.join(FIG, '2_criteri_concordano.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figura 3: quanto costa (tempo), e quanto e' uniforme il risultato
# ---------------------------------------------------------------------------

def figura_costo(dati, per):
    metodi_presenti = [m for m in ORDINE if m != 'nessun allineamento']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    tempi = []
    for m in metodi_presenti:
        t = [v['secondi'] for v in dati if v['metodo'] == m and v.get('secondi')]
        tempi.append(np.mean(t) if t else np.nan)
    y = np.arange(len(metodi_presenti))
    ax1.barh(y, tempi, color=[colore_di(m) for m in metodi_presenti], height=0.66, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(metodi_presenti, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('secondi per immagine (media)', fontsize=9)
    ax1.set_title('Tempo', fontsize=11, color=INCHIOSTRO, loc='left')
    for i, t in enumerate(tempi):
        if np.isfinite(t):
            ax1.text(t + 4, i, f'{t:.0f} s', va='center', fontsize=8.5, color=SECONDARIO)

    # Nelle zone peggiori, di quanto si sbaglia? Si usa il p95 in pixel e NON il
    # rapporto p95/mediana: quel rapporto punisce chi e' molto preciso (con una
    # mediana di 0.24 px anche un p95 piccolissimo da' un rapporto alto) e
    # finirebbe per dire l'opposto del vero.
    FOTOGRAFICHE = ('UVF', 'UVR', 'NIR850', 'NIR950')
    p95 = []
    for m in metodi_presenti:
        r = [v['residuo_p95_px'] for v in dati
             if v['metodo'] == m and v.get('ok') and v.get('residuo_p95_px')
             and v['immagine'] in FOTOGRAFICHE]
        p95.append(np.mean(r) if r else np.nan)
    ax2.barh(y, [v if np.isfinite(v) else 0 for v in p95],
             color=[colore_di(m) for m in metodi_presenti], height=0.66, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xscale('log')
    ax2.set_xlim(0.3, 80)
    ax2.set_xticks([0.5, 1, 5, 20, 50])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax2.axvline(1.0, color=SECONDARIO, lw=1, ls='--', zorder=2)
    ax2.set_xlabel('residuo nelle zone peggiori, p95 in px (log)', fontsize=9)
    ax2.set_title('Quanto sbaglia dove sbaglia di piu\'', fontsize=11,
                  color=INCHIOSTRO, loc='left')
    for i, v in enumerate(p95):
        if np.isfinite(v):
            ax2.text(v * 1.12, i, f'{v:.1f}', va='center', fontsize=8.5, color=SECONDARIO)

    fig.suptitle('Costo e comportamento nelle zone difficili',
                 fontsize=14, color=INCHIOSTRO, x=0.012, ha='left')
    fig.text(0.012, 0.9,
             'A destra: la media del p95 sulle quattro bande fotografiche (le radiografie sono a parte). '
             'Un buon allineamento e\' preciso anche nelle zone peggiori, non solo in media.',
             fontsize=9, color=SECONDARIO, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(FIG, '3_tempo_e_uniformita.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figura 4: il quadro d'insieme, una cella per coppia metodo/immagine
# ---------------------------------------------------------------------------

def figura_riepilogo(per):
    immagini = list(per)
    metodi_presenti = [m for m in ORDINE if any(m in per[i] for i in immagini)]
    M = np.full((len(metodi_presenti), len(immagini)), np.nan)
    testo = [[''] * len(immagini) for _ in metodi_presenti]
    for j, img in enumerate(immagini):
        for i, m in enumerate(metodi_presenti):
            v = per[img].get(m)
            if not v:
                testo[i][j] = '—'
                continue
            r = v.get('residuo_px')
            if r is None or not v.get('ok', True):
                testo[i][j] = 'X'
            else:
                M[i, j] = np.log10(max(r, 0.1))
                testo[i][j] = f'{r:.2f}' if r < 10 else f'{r:.0f}'

    fig, ax = plt.subplots(figsize=(1.55 * len(immagini) + 4.5,
                                    0.52 * len(metodi_presenti) + 2.6))
    # scala sequenziale a una tinta: il blu chiaro e' "vicino a zero"
    im = ax.imshow(M, cmap='Blues', aspect='auto', vmin=np.log10(0.2), vmax=np.log10(100))
    ax.set_xticks(range(len(immagini)))
    ax.set_xticklabels(immagini, fontsize=9)
    ax.set_yticks(range(len(metodi_presenti)))
    ax.set_yticklabels(metodi_presenti, fontsize=9)
    ax.grid(False)
    for i in range(len(metodi_presenti)):
        for j in range(len(immagini)):
            v = M[i, j]
            col = '#ffffff' if np.isfinite(v) and v > np.log10(12) else INCHIOSTRO
            if testo[i][j] == 'X':
                col = ROSSO
            ax.text(j, i, testo[i][j], ha='center', va='center', fontsize=8.5, color=col)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('residuo in px (scala logaritmica)', fontsize=9, color=SECONDARIO)
    cb.set_ticks([np.log10(x) for x in (0.2, 1, 5, 20, 100)])
    cb.set_ticklabels(['0.2', '1', '5', '20', '100'])
    ax.set_title('Tutti i risultati in una tabella\n'
                 'numero = residuo in px;  X = metodo non riuscito;  piu\' chiaro = meglio',
                 fontsize=12, color=INCHIOSTRO, loc='left', pad=12)
    fig.tight_layout()
    p = os.path.join(FIG, '4_riepilogo.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == '__main__':
    dati, per = carica()
    print(f"{len(dati)} risultati, {len(per)} immagini")
    for f in (figura_residui(per), figura_criteri(per), figura_costo(dati, per),
              figura_riepilogo(per)):
        print("scritto", f)
