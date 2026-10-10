"""
Le figure che accompagnano le prove dell'8 ottobre 2026.

    python disegna_figure.py "<cartella delle riprese vere>"

Scrive i PNG in questa cartella. Stessi colori degli schemi in ../schemi.
Le figure 2, 4, 5 e 7 rifanno alcune stime (qualche minuto in tutto); le altre
leggono i risultati gia' salvati in Allineatore/benchmark.
"""
import contextlib, io, json, os, shutil, sys, tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(os.path.dirname(os.path.dirname(QUI)), 'Allineatore')
BENCH = os.path.join(PROG, 'benchmark')
sys.path.insert(0, PROG); sys.path.insert(0, BENCH)
import image_io, tiff_memmap, align_paintings, block_matching as bm
import sintetico, vere_deformate

INK, INK2, MUTO, SUP = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'
BLU, ARANCIO, ACQUA = '#2a78d6', '#eb6834', '#1baf7a'
# una sola tinta dal chiaro allo scuro, per le grandezze
SEQ = LinearSegmentedColormap.from_list('blu', ['#f3f2ee', '#bcd6f5', '#6aa3e6', '#2a78d6', '#123f78'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTO, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 10,
                     'axes.titleweight': 'bold', 'figure.facecolor': SUP, 'axes.facecolor': SUP,
                     'savefig.facecolor': SUP, 'axes.spines.top': False, 'axes.spines.right': False})


def salva(fig, nome):
    if os.environ.get('TESI'):
        # versione per la tesi: titolo e nota vanno nella didascalia, non nell'immagine
        fig.suptitle('')
        for t in list(fig.texts):
            t.remove()
        nome = nome.replace('.png', '_tesi.png')
    p = os.path.join(QUI, nome)
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("scritta", nome, flush=True)


def leggi(nome):
    p = os.path.join(BENCH, nome)
    return [json.loads(r) for r in open(p, encoding='utf-8').read().splitlines()] if os.path.exists(p) else []


def allinea(p_f, p_m, tmp, fattore=2, soglia=None):
    prima = bm.SOGLIA_MINIMA_PASSO
    if soglia is not None:
        bm.SOGLIA_MINIMA_PASSO = soglia
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return align_paintings.align(p_f, p_m, None, params_path=os.path.join(tmp, 'p.json'), factor=fattore,
                                         nonrigid=True, min_conf=0.10, refine_block=128, refine_step=64,
                                         diagnostics_dir=os.path.join(tmp, 'diag'), tre_scale='auto', verbose=False)[0]
    finally:
        bm.SOGLIA_MINIMA_PASSO = prima


# ---------------------------------------------------------------------------
def fig1_deformazioni():
    """Le quattro forme di deformazione, come griglia deformata."""
    W, H = sintetico.W, sintetico.H
    casi = (('Onda', 'onda', "la tela che si flette"), ('Radiale', 'radiale', "un obiettivo diverso"),
            ('Prospettiva', 'prospettiva', "la camera spostata"), ('Rigonfiamento locale', 'locale', "una deformazione in un punto"))
    ESAG = 6
    fig, axs = plt.subplots(1, 4, figsize=(12, 4.3))
    for ax, (titolo, forma, sotto) in zip(axs, casi):
        f = sintetico.mappa_vera(dict(sintetico.BASE, forma=forma, ampiezza=100.0, rotazione=0.0, scala=1.0, trasl=(0.0, 0.0)))
        t = np.linspace(0, 1, 200)
        for k in np.linspace(0, 1, 13):
            for xs, ys in ((np.full_like(t, k * W), t * H), (t * W, np.full_like(t, k * H))):
                ax.plot(xs, ys, color='#d9d7d0', lw=0.6, zorder=1)
                fx, fy = f(xs, ys)
                ax.plot(xs + ESAG * (fx - xs), ys + ESAG * (fy - ys), color=BLU, lw=0.9, zorder=2)
        ax.set_xlim(-0.2 * W, 1.2 * W); ax.set_ylim(1.15 * H, -0.15 * H); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(titolo, loc='left'); ax.text(0, 1.0, sotto, transform=ax.transAxes, color=INK2, fontsize=8.5, va='top')
    fig.suptitle("Le deformazioni imposte nelle prove", x=0.02, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.02, f"In grigio la griglia di partenza, in blu la stessa griglia deformata. Ampiezza 100 px su un'immagine di {W}×{H} px, "
             f"qui ingrandita {ESAG} volte per renderla visibile.", color=INK2, fontsize=8.5)
    salva(fig, "fig1_deformazioni.png")


def fig2_sintetiche():
    """Un ritaglio di immagine artificiale: riferimento, deformata, e i contenuti."""
    tmp = tempfile.mkdtemp(prefix='figure_')
    rit = (slice(2132, 2732), slice(2676, 3276))    # attorno al centro del rigonfiamento
    pannelli = []
    try:
        for nome, diff in (("Da allineare: deformata,\ncontrasto invertito", dict(forma='locale', ampiezza=100.0)),
                           ("Poca struttura in comune\n(come una radiografia)", dict(forma='locale', ampiezza=100.0, contenuto='poca_struttura')),
                           ("Deformata,\ncon rumore", dict(forma='locale', ampiezza=100.0, contenuto='rumore'))):
            p_f, p_m, _ = sintetico.costruisci(dict(sintetico.BASE, **diff), 0, tmp)
            if not pannelli:
                a = tiff_memmap.open_memmap(p_f)[0]; pannelli.append(("Riferimento", np.asarray(a[rit]).copy())); del a
            b = tiff_memmap.open_memmap(p_m)[0]; pannelli.append((nome, np.asarray(b[rit]).copy())); del b
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2))
    for ax, (t, img) in zip(axs, pannelli):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_title(t, loc='left', fontsize=9); ax.axis('off')
        ax.axhline(300, color=ARANCIO, lw=0.7); ax.axvline(300, color=ARANCIO, lw=0.7)
    fig.suptitle("Le immagini artificiali del test", x=0.02, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.02, -0.04, "Stesso ritaglio di 600×600 px nelle quattro immagini, al centro di un rigonfiamento locale da 100 px. Le linee arancioni segnano lo stesso\n"
             "punto del foglio: nelle tre immagini da allineare il disegno sotto è spostato di oltre 100 px rispetto al riferimento.", color=INK2, fontsize=8.5)
    salva(fig, "fig2_immagini_artificiali.png")


def fig3_vera_deformata(cartella):
    """Una ripresa vera prima e dopo la deformazione nota (sull'anteprima)."""
    p = os.path.join(cartella, "NIR850_FraGalgario_TOT.tif")
    img, f = image_io.leggi_ridotta(p, 1500, in_grigio=True, verbose=False)
    img = image_io.a_8bit(img); h, w = img.shape
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.2))
    for ax, (titolo, forma, A) in zip(axs, (("Ripresa originale", None, 0), ("Con distorsione radiale di 100 px", 'radiale', 100.0),
                                            ("Con rigonfiamento locale di 100 px", 'locale', 100.0))):
        ESAG = 4
        if forma is None:
            out = img; D = lambda x, y: (x, y)
        else:
            D = vere_deformate.deformazione(forma, A * ESAG / f, h, w, 1.0)
            X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            ix, iy = X.copy(), Y.copy()
            for _ in range(40):
                fx, fy = D(ix, iy); ix += X - fx; iy += Y - fy
            out = cv2.remap(img, ix.astype(np.float32), iy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ax.imshow(out, cmap='gray'); ax.axis('off'); ax.set_title(titolo, loc='left')
        t = np.linspace(0, 1, 200)
        for k in np.linspace(0, 1, 9):
            for xs, ys in ((np.full_like(t, k * (w - 1)), t * (h - 1)), (t * (w - 1), np.full_like(t, k * (h - 1)))):
                fx, fy = D(xs, ys); ax.plot(fx, fy, color=ARANCIO, lw=0.7, alpha=0.9)
        ax.set_xlim(-0.06 * w, 1.06 * w); ax.set_ylim(1.06 * h, -0.06 * h)
    fig.suptitle("Una ripresa vera deformata in modo noto", x=0.02, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.03, "Infrarosso vicino. La griglia arancione segue la deformazione imposta, qui ingrandita 4 volte: "
             "a grandezza vera 100 px su un'immagine di 6000×8000 px non si vedrebbero.", color=INK2, fontsize=8.5)
    salva(fig, "fig3_ripresa_vera_deformata.png")


def fig4_falso_colore(cartella):
    """Sovrapposizione in falso colore prima e dopo l'allineamento, su un dettaglio."""
    p_f = os.path.join(cartella, "VIS_FraGalgario_TOT.tif"); p_m = os.path.join(cartella, "UVF_FraGalgario_TOT.tif")
    tmp = tempfile.mkdtemp(prefix='figure_')
    try:
        T = allinea(p_f, p_m, tmp)
        af = image_io.apri(p_f, verbose=False)[0]; am = image_io.apri(p_m, verbose=False)[0]
        H, W = af.shape[:2]; L = 700
        # il ritaglio con piu' contorni fra alcuni candidati
        cand = [(int(H * fy), int(W * fx)) for fy in (0.3, 0.45, 0.6) for fx in (0.3, 0.5, 0.7)]
        def grigio(a):
            a = np.asarray(a); return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
        y0, x0 = max(cand, key=lambda c: float(bm.gradient_magnitude(grigio(af[c[0]:c[0] + L, c[1]:c[1] + L])).mean()))
        vf = image_io.a_8bit(grigio(af[y0:y0 + L, x0:x0 + L]))
        prima = image_io.a_8bit(grigio(am[y0:y0 + L, x0:x0 + L]))
        dopo = image_io.a_8bit(grigio(T.warp_tile_from(am, y0, x0, L, L)))
        del af, am
    finally:
        T = None; shutil.rmtree(tmp, ignore_errors=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, t, b in zip(axs, ("Prima dell'allineamento", "Dopo l'allineamento"), (prima, dopo)):
        ax.imshow(np.dstack([b, vf, vf])); ax.axis('off'); ax.set_title(t, loc='left')
    fig.suptitle("Sovrapposizione in falso colore su un dettaglio", x=0.02, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.02, "Fluorescenza ultravioletta nel canale rosso, visibile nel verde e nel blu; ritaglio di 700×700 px a piena risoluzione.\n"
             "Dove le due immagini non coincidono i contorni si sdoppiano in una frangia rossa e una azzurra.", color=INK2, fontsize=8.5)
    salva(fig, "fig4_falso_colore_prima_dopo.png")


def fig5_mappe_errore():
    """Dove sbaglia: mappa dell'errore vero, senza e con la soglia minima."""
    W, H = sintetico.W, sintetico.H
    gx, gy = np.meshgrid(np.arange(25, W - 24, 100, dtype=np.float64), np.arange(25, H - 24, 100, dtype=np.float64))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    casi = (("Prospettiva da 200 px", dict(forma='prospettiva', ampiezza=200.0)), ("Rigonfiamento da 30 px", dict(forma='locale', ampiezza=30.0)),
            ("Rigonfiamento da 100 px", dict(forma='locale', ampiezza=100.0)))
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 9.4))
    for j, (titolo, diff) in enumerate(casi):
        caso = dict(sintetico.BASE, **diff)
        tmp = tempfile.mkdtemp(prefix='figure_')
        try:
            p_f, p_m, _ = sintetico.costruisci(caso, 0, tmp)
            vx, vy = sintetico.mappa_vera(caso)(xy[:, 0], xy[:, 1])
            for i, (et, soglia) in enumerate((("senza soglia minima", 0.0), ("con la soglia minima al 5%", 0.05))):
                T = allinea(p_f, p_m, tmp, soglia=soglia)
                e = np.hypot(*(align_paintings._porta_punti(T, xy) - np.column_stack([vx, vy])).T).reshape(gx.shape)
                T = None
                ax = axs[i, j]
                im = ax.imshow(e, cmap=SEQ, vmin=0, vmax=10, extent=(0, W, H, 0)); ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(MUTO)
                ax.set_title(f"{titolo}\n{et}", loc='left', fontsize=9)
                ax.text(0.02, 0.02, f"oltre 5 px: {100 * (e > 5).mean():.1f}% dei punti\nmassimo {e.max():.0f} px", transform=ax.transAxes,
                        fontsize=8.5, color=INK, va='bottom', bbox=dict(facecolor=SUP, edgecolor='none', alpha=0.85, pad=2))
        finally:
            import gc; gc.collect(); shutil.rmtree(tmp, ignore_errors=True)
    cb = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.03, pad=0.04, aspect=50)
    cb.set_label("errore rispetto alla trasformazione vera, px (la scala si ferma a 10)"); cb.outline.set_visible(False)
    fig.suptitle("Dove sbaglia il programma, e che cosa cambia la soglia minima", x=0.02, ha='left', fontsize=12, fontweight='bold', y=0.97)
    salva(fig, "fig5_mappe_errore_soglia.png")


def fig6_distanza_dal_bordo():
    """Errore di recupero sulle riprese vere, per distanza dal bordo."""
    R = [r for r in leggi('vere_deformate.jsonl') if not r.get('errore')]
    foto = [r for r in R if 'vicino' in r['ripresa'] or 'ultraviolett' in r['ripresa']]
    defs = list(dict.fromkeys(r['deformazione'] for r in foto))
    fasce = (('fascia 0-150', 'entro 150 px dal bordo', '#123f78'), ('fascia 150-500', 'fra 150 e 500 px', '#2a78d6'),
             ('fascia oltre 500', 'oltre 500 px', '#a9c9f1'))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    larg = 0.26
    for k, (chiave, et, col) in enumerate(fasce):
        v = [np.mean([r[chiave]['mediana'] for r in foto if r['deformazione'] == d and r.get(chiave)]) for d in defs]
        x = np.arange(len(defs)) + (k - 1) * (larg + 0.02)
        ax.bar(x, v, larg, color=col, label=et)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.12, f"{vi:.1f}".replace('.', ','), ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(defs))); ax.set_xticklabels([d.replace(' ', chr(10), 1) for d in defs]); ax.tick_params(length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.set_ylabel("errore di recupero mediano, px"); ax.yaxis.grid(True, color='#e6e4dd', lw=0.6); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.legend(frameon=False, loc='upper right', ncol=1)
    ax.set_title("L'errore si concentra vicino al bordo", loc='left', pad=10)
    fig.text(0.02, -0.06, "Riprese vere deformate in modo noto, bande fotografiche (media di quattro riprese), precisione consigliata.", color=INK2, fontsize=8.5)
    salva(fig, "fig6_errore_per_distanza_dal_bordo.png")


def fig7_confronto():
    """Il programma e i metodi deformabili sulle riprese vere."""
    R = [r for r in leggi('deformabili_reali.jsonl') if r.get('fissa')]
    metodi = (('programma (consigliata)', 'Metodo proposto'), ('flusso ottico DIS a mezza risoluzione', 'Flusso ottico'),
              ('Demons a mezza risoluzione', 'Demons'), ('campo denso + correlazione locale', 'Campo denso'),
              ('B-spline + informazione mutua', 'B-spline e inf. mutua'))
    foto = ('UVF', 'UVR', 'NIR850', 'NIR950')
    def media(met, imm, chiave):
        v = [r['fissa'][chiave] for r in R if r['metodo'] == met and r['immagine'] in imm and r['fissa'].get(chiave) is not None]
        return float(np.mean(v)) if v else 0.0
    rx = [i for i in dict.fromkeys(r['immagine'] for r in R) if i.startswith('RX')]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={'wspace': 0.62})
    for ax, (titolo, imm, chiave, fmt, xl) in zip(axs, (
            ("Bande fotografiche: residuo mediano", foto, 'mediana', "{:.2f} px", "px (più basso è meglio)"),
            ("Radiografie: finestre in cui la misura riesce", rx, 'n_affidabili', "{:.0f} su 225", "finestre misurabili (più alto è meglio)"))):
        v = [media(m, imm, chiave) for m, _ in metodi]
        y = np.arange(len(metodi))[::-1]
        ax.barh(y, v, 0.62, color=[BLU] + ['#a9c9f1'] * (len(metodi) - 1))
        for yi, vi in zip(y, v):
            ax.text(vi + max(v) * 0.015, yi, fmt.format(vi).replace('.', ','), va='center', fontsize=8.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([n for _, n in metodi]); ax.tick_params(length=0)
        ax.set_xlim(0, max(v) * 1.28); ax.set_xlabel(xl); ax.set_title(titolo, loc='left')
        ax.xaxis.grid(True, color='#e6e4dd', lw=0.6); ax.set_axisbelow(True); ax.spines['bottom'].set_visible(False)
    fig.suptitle("Confronto con i metodi deformabili sulle riprese vere", x=0.02, ha='left', fontsize=12, fontweight='bold', y=1.04)
    fig.text(0.02, -0.06, "Media su quattro riprese fotografiche e su due radiografie; stessa griglia di finestre per tutti. Flusso ottico e Demons a metà "
             "risoluzione, come il metodo proposto;\ncampo denso e B-spline a un quarto. Tutti partono dalla posa trovata dal metodo proposto.", color=INK2, fontsize=8.5)
    salva(fig, "fig7_confronto_deformabili.png")


def fig8_sintetico():
    """Test sintetico: quota di punti oltre 5 px per forma e ampiezza."""
    R = [r for r in leggi('sintetico_risultati.jsonl') if not r.get('errore')]
    forme = (('onda', 'Onda', (6, 30, 100)), ('radiale', 'Radiale', (30, 100, 200)), ('prospettiva', 'Prospettiva', (30, 100, 200)),
             ('locale', 'Rigonfiamento locale', (30, 100, 200)))
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    for ax, (forma, titolo, amp) in zip(axs, forme):
        for k, (fattore, et, col) in enumerate(((2, 'consigliata', BLU), (1, 'massima', ARANCIO))):
            v = [100 * np.mean([r['vero']['oltre_5px'] for r in R if r['forma'] == forma and r['ampiezza'] == a and r['fattore'] == fattore
                                and r['caso'].startswith('forma')] or [0]) for a in amp]
            x = np.arange(len(amp)) + (k - 0.5) * 0.4
            ax.bar(x, v, 0.37, color=col, label=et)
            for xi, vi in zip(x, v):
                ax.text(xi, vi + 0.6, f"{vi:.0f}", ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(np.arange(len(amp))); ax.set_xticklabels([f"{a} px" for a in amp]); ax.tick_params(length=0)
        ax.set_title(titolo, loc='left'); ax.yaxis.grid(True, color='#e6e4dd', lw=0.6); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
    axs[0].set_ylabel("punti con errore oltre 5 px, %")
    axs[0].legend(frameon=False, loc='upper left', title="precisione", title_fontsize=8.5)
    fig.suptitle("Test su immagini artificiali: dove l'errore supera i 5 px", x=0.02, ha='left', fontsize=12, fontweight='bold', y=1.05)
    fig.text(0.02, -0.04, "Ampiezza della deformazione sull'asse orizzontale. Misure del 9 ottobre, con la soglia minima dei blocchi.", color=INK2, fontsize=8.5)
    salva(fig, "fig8_test_sintetico.png")


if __name__ == '__main__':
    cartella = sys.argv[1] if len(sys.argv) > 1 else None
    solo = sys.argv[2:] or None
    figure = [('1', fig1_deformazioni, False), ('8', fig8_sintetico, False), ('6', fig6_distanza_dal_bordo, False),
              ('7', fig7_confronto, False), ('3', fig3_vera_deformata, True), ('2', fig2_sintetiche, False),
              ('4', fig4_falso_colore, True), ('5', fig5_mappe_errore, False)]
    for n, f, serve in figure:
        if solo and n not in solo:
            continue
        try:
            f(cartella) if serve else f()
        except Exception as exc:
            import traceback
            print(f"figura {n} non riuscita: {exc}"); traceback.print_exc()
