"""
Il test sintetico esteso: il programma contro trasformazioni NOTE.

    python sintetico.py                 # tutta la casistica (qualche ora)
    python sintetico.py --solo onda     # solo i casi il cui nome contiene "onda"
    python sintetico.py --tabelle       # solo le tabelle, da sintetico_risultati.jsonl

Il test di test_pipeline.py e' uno solo: 1300x1600 px, un'onda di 6 px, 81 punti
ad almeno 150 px dal bordo, precisione massima. Sei pixel stanno dentro la
portata del campo fine, quindi il campo grossolano non e' mai messo alla prova,
e i punti lontani dal bordo non dicono niente di dove gli errori si concentrano.

Qui l'immagine e' di 4800x6400 px (31 Mpx, come una ripresa vera ridotta) e si
varia una cosa alla volta attorno a un caso base, piu' la griglia completa
forma della deformazione x ampiezza x precisione:

- forma: onda (la tela che si flette), radiale (un obiettivo diverso, cresce
  verso gli angoli), prospettiva (la camera spostata, come fra le bande riprese
  da cavalletto), locale (un rigonfiamento in un punto);
- ampiezza: da 6 a 200 px, cioe' dentro la portata del campo fine, dentro
  quella del livello stretto del grossolano, dentro quella del livello largo;
- posa d'insieme: traslazione, rotazione, scala, immagine di dimensioni diverse;
- contenuto: stessa banda, contrasto invertito, poca struttura in comune (come
  una radiografia), rumore, sfocatura;
- fondo che si muove in un altro modo (parallasse), senza e con la zona;
- precisione consigliata (fattore 2) e massima (fattore 1).

L'errore e' la distanza fra dove il programma manda un punto e dove lo manda la
trasformazione vera, su una griglia di punti ogni 100 px fino al bordo, e si
riporta per fasce di distanza dal bordo. Ogni caso si ripete su immagini
diverse (semi diversi) per dare media e dispersione. Accanto all'errore vero si
registra la verifica del programma: dice quanto la verifica, che sulle immagini
vere e' l'unica misura, somiglia all'errore vero.

Non tocca il programma: lo chiama come lo chiama il lotto (align_paintings.align
con gli stessi valori di align_batch).
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import cv2

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROG not in sys.path:
    sys.path.insert(0, PROG)

import tiff_memmap
import align_paintings

QUI = os.path.dirname(os.path.abspath(__file__))
RISULTATI = os.path.join(QUI, 'sintetico_risultati.jsonl')
TABELLE = os.path.join(QUI, 'sintetico_tabelle.txt')

H, W = 6400, 4800
BORDO_TELA = 500          # tela dipinta in piu' attorno al foglio, per lato
FASCE = (('0-150', 0, 150), ('150-500', 150, 500), ('oltre 500', 500, 1e9))
BASE = dict(forma='onda', ampiezza=6.0, rotazione=0.25, scala=1.004, trasl=(13.0, -8.0),
            contenuto='invertito', fattore=2, fondo=None, zona=False, scala_file=1.0)


# ---------------------------------------------------------------------------
# le immagini
# ---------------------------------------------------------------------------

def tela(h, w, seme):
    """Rumore a piu' ottave e linee casuali: strutturato ma non periodico (una
    trama periodica darebbe picchi di fase ambigui senza che nulla sia rotto)."""
    rng = np.random.default_rng(seme)
    img = np.zeros((h, w), np.float32)
    for ottava in range(7):
        s = 2 ** ottava
        piccolo = rng.random((max(2, h // (4 * s)), max(2, w // (4 * s)))).astype(np.float32)
        img += cv2.resize(piccolo, (w, h), interpolation=cv2.INTER_CUBIC) / (s ** 0.7)
    img = cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX)
    for _ in range(int(300 * h * w / (1600 * 1300))):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (p0[0] + int(rng.integers(-120, 120)), p0[1] + int(rng.integers(-120, 120)))
        cv2.line(img, p0, p1, float(rng.integers(0, 255)), 1, cv2.LINE_AA)
    return img.astype(np.uint8)


def mappa_vera(caso):
    """La trasformazione vera riferimento -> immagine, come funzione (x, y) ->
    (x', y'), in pixel del riferimento (la scala del file si applica dopo)."""
    th = np.radians(caso['rotazione'])
    s = caso['scala']
    a, b, c, d = s * np.cos(th), -s * np.sin(th), s * np.sin(th), s * np.cos(th)
    cx, cy = W / 2.0, H / 2.0
    tx, ty = caso['trasl']
    A = float(caso['ampiezza'])
    forma = caso['forma']
    r_ang = np.hypot(cx, cy)

    def f(x, y):
        x = np.asarray(x); y = np.asarray(y)
        u, v = x - cx, y - cy
        if forma == 'onda':
            # il periodo cresce con l'ampiezza: la pendenza massima resta 4%
            P = max(900.0, 2 * np.pi * A / 0.04)
            dx, dy = A * np.sin(2 * np.pi * y / P), A * np.cos(2 * np.pi * x / P)
        elif forma == 'radiale':
            # distorsione di un obiettivo: cresce col cubo del raggio, A all'angolo
            k = A * (np.hypot(u, v) / r_ang) ** 2 / r_ang
            dx, dy = k * u, k * v
        elif forma == 'prospettiva':
            # camera inclinata: A di spostamento agli angoli, di segno opposto ai
            # due lati; al centro niente
            den = 1.0 + (A / r_ang) * (0.8 * u + 0.6 * v) / r_ang
            dx, dy = u / den - u, v / den - v
        elif forma == 'locale':
            g = np.exp(-(((x - 0.62 * W) ** 2 + (y - 0.38 * H) ** 2) / (2 * 700.0 ** 2)))
            dx, dy = A * g, -0.6 * A * g
        elif forma == 'nessuna':
            dx = dy = 0.0
        else:
            raise ValueError(forma)
        return (a * u + b * v + cx + tx + dx, c * u + d * v + cy + ty + dy)
    return f


def costruisci(caso, seme, cartella):
    """Scrive riferimento e immagine da allineare; restituisce percorsi e zona."""
    B = BORDO_TELA
    grande = tela(H + 2 * B, W + 2 * B, seme)
    fixed = grande[B:B + H, B:B + W].copy()
    f = mappa_vera(caso)
    sf = caso['scala_file']
    Hm, Wm = int(round(H * sf)), int(round(W * sf))

    # moving(y) = tela(T^-1(y)): l'inversa per punto fisso, a blocchi di righe
    moving = np.zeros((Hm, Wm), np.uint8)
    # in precisione semplice: su coordinate fino a 6400 l'errore e' sotto 0.001 px
    xs = ((np.arange(Wm, dtype=np.float64) + 0.5) / sf - 0.5).astype(np.float32)
    for y0 in range(0, Hm, 800):
        y1 = min(Hm, y0 + 800)
        ys = ((np.arange(y0, y1, dtype=np.float64) + 0.5) / sf - 0.5).astype(np.float32)
        X, Y = np.meshgrid(xs, ys)
        ix, iy = X.copy(), Y.copy()
        for _ in range(40):
            fx, fy = f(ix, iy)
            fx = X - fx; fy = Y - fy
            ix += fx; iy += fy
            if max(float(np.abs(fx).max()), float(np.abs(fy).max())) < 2e-3:
                break
        moving[y0:y1] = cv2.remap(grande, (ix + B).astype(np.float32), (iy + B).astype(np.float32),
                                  cv2.INTER_LINEAR if sf == 1.0 else cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_REFLECT)
    zona = None
    if caso['fondo']:
        # Il dipinto e' un'ellisse; fuori c'e' un fondo con una sua struttura,
        # che fra le due riprese si sposta in un altro modo (parallasse).
        zona = np.zeros((H, W), np.uint8)
        cv2.ellipse(zona, (W // 2, H // 2), (int(W * 0.44), int(H * 0.45)), 0, 0, 360, 1, -1)
        fondo = tela(H + 2 * B, W + 2 * B, seme + 5000)
        fixed = np.where(zona > 0, fixed, fondo[B:B + H, B:B + W])
        # la zona nell'immagine da allineare: i punti il cui antecedente e' nel dipinto
        zm = np.zeros((Hm, Wm), np.uint8)
        px = float(caso['fondo'])
        for y0 in range(0, Hm, 800):
            y1 = min(Hm, y0 + 800)
            ys = ((np.arange(y0, y1, dtype=np.float64) + 0.5) / sf - 0.5).astype(np.float32)
            X, Y = np.meshgrid(xs, ys)
            ix, iy = X.copy(), Y.copy()
            for _ in range(40):
                fx, fy = f(ix, iy)
                fx = X - fx; fy = Y - fy
                ix += fx; iy += fy
                if max(float(np.abs(fx).max()), float(np.abs(fy).max())) < 2e-3:
                    break
            zm[y0:y1] = cv2.remap(zona, ix.astype(np.float32), iy.astype(np.float32), cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            fnd = cv2.remap(fondo, (ix + B + px).astype(np.float32), (iy + B - 0.5 * px).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            moving[y0:y1] = np.where(zm[y0:y1] > 0, moving[y0:y1], fnd)

    rng = np.random.default_rng(seme + 99)
    cont = caso['contenuto']
    m = moving.astype(np.float32)
    if cont in ('invertito', 'rumore', 'sfocata', 'poca_struttura'):
        m = np.clip(((255.0 - m) / 255.0) ** 1.6 * 255.0, 0, 255)
    if cont == 'rumore':
        m = m + rng.normal(0, 25.0, m.shape).astype(np.float32)
    if cont == 'sfocata':
        m = cv2.GaussianBlur(m, (0, 0), 4.0)
    if cont == 'poca_struttura':
        # Come una radiografia: il disegno comune resta solo a chiazze, sfocato
        # e attenuato; sopra c'e' una struttura che il riferimento non ha
        # (telaio, chiodi, stesure sotto) e il rumore.
        chiazze = cv2.resize(rng.random((Hm // 400 + 2, Wm // 400 + 2)).astype(np.float32), (Wm, Hm),
                             interpolation=cv2.INTER_CUBIC)
        peso = np.clip((chiazze - 0.45) * 4.0, 0, 1)
        altro = tela(Hm, Wm, seme + 7000).astype(np.float32)
        barre = np.zeros((Hm, Wm), np.float32)
        for k in range(1, 4):
            cv2.line(barre, (0, Hm * k // 4), (Wm, Hm * k // 4), 60.0, 90)
            cv2.line(barre, (Wm * k // 4, 0), (Wm * k // 4, Hm), 60.0, 90)
        m = 128 + 0.45 * peso * (cv2.GaussianBlur(m, (0, 0), 2.0) - 128) + 0.5 * (cv2.GaussianBlur(altro, (0, 0), 3.0) - 128)
        m = m + cv2.GaussianBlur(barre, (0, 0), 6.0) + rng.normal(0, 8.0, m.shape).astype(np.float32)
    moving = np.clip(m, 0, 255).astype(np.uint8)

    percorsi = []
    for nome, img in (('riferimento.tif', fixed), ('immagine.tif', moving)):
        p = os.path.join(cartella, nome)
        mm = tiff_memmap.create_tiff_memmap(p, img.shape[0], img.shape[1], samples=1, dtype=np.uint8)
        mm[:] = img; mm.flush(); del mm
        percorsi.append(p)
    return percorsi[0], percorsi[1], zona


# ---------------------------------------------------------------------------
# la misura
# ---------------------------------------------------------------------------

def errore_vero(T, caso, zona):
    """Distanza fra la trasformazione stimata e quella vera, su una griglia di
    punti ogni 100 px che arriva a 25 px dal bordo; con la zona, solo dentro."""
    gx, gy = np.meshgrid(np.arange(25, W - 24, 100, dtype=np.float64), np.arange(25, H - 24, 100, dtype=np.float64))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    if zona is not None:
        dist = cv2.distanceTransform(np.pad(zona, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        d = dist[xy[:, 1].astype(int), xy[:, 0].astype(int)]
        xy, d = xy[d > 0], d[d > 0]
    else:
        d = np.minimum.reduce([xy[:, 0], xy[:, 1], W - 1 - xy[:, 0], H - 1 - xy[:, 1]])
    vx, vy = mappa_vera(caso)(xy[:, 0], xy[:, 1])
    sf = caso['scala_file']
    vero = np.column_stack([(vx + 0.5) * sf - 0.5, (vy + 0.5) * sf - 0.5])
    # l'antecedente deve cadere dentro l'immagine da allineare: fuori non c'e'
    # niente da allineare
    dentro = (vero[:, 0] >= 0) & (vero[:, 1] >= 0) & (vero[:, 0] <= W * sf - 1) & (vero[:, 1] <= H * sf - 1)
    stimato = align_paintings._porta_punti(T, xy)
    e = np.hypot(*(stimato - vero).T)[dentro]
    d = d[dentro]
    out = {'n': int(len(e)), 'mediana': float(np.median(e)), 'p95': float(np.percentile(e, 95)),
           'max': float(e.max()), 'oltre_1px': float((e > 1).mean()), 'oltre_5px': float((e > 5).mean())}
    for nome, lo, hi in FASCE:
        s = (d >= lo) & (d < hi)
        if s.any():
            out['fascia ' + nome] = {'n': int(s.sum()), 'mediana': float(np.median(e[s])),
                                     'p95': float(np.percentile(e[s], 95)), 'max': float(e[s].max())}
    return out


def esegui(nome, caso, seme):
    cartella = tempfile.mkdtemp(prefix='sintetico_')
    rec = {'caso': nome, 'seme': seme, **{k: v for k, v in caso.items()}}
    try:
        t0 = time.time()
        p_f, p_m, zona = costruisci(caso, seme, cartella)
        rec['t_costruzione_s'] = time.time() - t0
        usa_zona = zona if caso['zona'] else None
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            T, rep = align_paintings.align(
                p_f, p_m, None, params_path=os.path.join(cartella, 'p.json'), factor=caso['fattore'],
                nonrigid=True, min_conf=0.10, refine_block=128, refine_step=64,
                diagnostics_dir=os.path.join(cartella, 'diag'), mask=usa_zona, tre_scale='auto', verbose=False)
        rec['t_align_s'] = time.time() - t0
        rec['vero'] = errore_vero(T, caso, zona)
        v = rep['verifica']['dopo']
        rec['verifica'] = {k: v.get(k) for k in ('residuo_mediano_px', 'residuo_p95_px', 'residuo_max_px', 'n_affidabili',
                                                 'n_campioni', 'quota_sopra_1px', 'quota_sopra_5px')}
        s = rep['stima']
        rec['livelli'] = [{k: l.get(k) for k in ('riduzione', 'applicato', 'ampiezza_max_px_full')}
                          for l in (s.get('campo_grossolano_livelli') or [])]
        rec['campo_fine'] = bool((s.get('nonrigid') or {}).get('applicato'))
        rec['ripiego'] = bool(rep.get('ripiego')) or bool((rep.get('aggancio') or {}).get('dalla_posa'))
    except Exception as exc:
        rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    finally:
        import gc
        T = rep = None
        gc.collect()
        shutil.rmtree(cartella, ignore_errors=True)
    return rec


# ---------------------------------------------------------------------------
# la casistica
# ---------------------------------------------------------------------------

def casistica():
    """(nome, caso, numero di ripetizioni). Il nome dice la sola cosa cambiata."""
    casi = []

    def c(nome, rip=2, **diff):
        casi.append((nome, dict(BASE, **diff), rip))

    # 1. forma x ampiezza x precisione: la griglia principale
    for fattore, et in ((2, 'consigliata'), (1, 'massima')):
        for forma, ampiezze in (('onda', (6, 30, 100)), ('radiale', (30, 100, 200)),
                                ('prospettiva', (30, 100, 200)), ('locale', (30, 100, 200))):
            for A in ampiezze:
                c(f"forma {forma} {A:3d} px, {et}", rip=3 if fattore == 2 else 1, forma=forma, ampiezza=float(A),
                  fattore=fattore)
    c("forma nessuna (solo posa), consigliata", forma='nessuna', ampiezza=0.0)
    c("forma nessuna (solo posa), massima", rip=1, forma='nessuna', ampiezza=0.0, fattore=1)
    # 2. precisioni piu' veloci, sul caso base e su una deformazione ampia
    for fattore, et in ((4, 'veloce'), (8, 'molto veloce')):
        c(f"precisione {et}, onda 6 px", fattore=fattore)
        c(f"precisione {et}, radiale 100 px", forma='radiale', ampiezza=100.0, fattore=fattore)
    # 3. posa d'insieme (deformazione del caso base)
    for t in (60, 300, 800):
        c(f"posa traslazione {t} px", trasl=(float(t), -0.6 * t))
    for r in (1.0, 3.0, 8.0):
        c(f"posa rotazione {r:g} gradi", rotazione=r)
    for s in (1.02, 1.08):
        c(f"posa scala {s:g}", scala=s)
    for sf in (0.7, 0.4, 1.5):
        c(f"posa immagine di dimensioni {sf:g}x", scala_file=sf)
    c("posa tutto insieme (300 px, 3 gradi, 1.02, file 0.7x, radiale 100)", trasl=(300.0, -180.0), rotazione=3.0,
      scala=1.02, scala_file=0.7, forma='radiale', ampiezza=100.0)
    # 4. contenuto
    for cont in ('uguale', 'rumore', 'sfocata', 'poca_struttura'):
        c(f"contenuto {cont}", contenuto=cont)
        c(f"contenuto {cont}, radiale 100 px", contenuto=cont, forma='radiale', ampiezza=100.0)
    # 5. fondo che si muove in un altro modo, senza e con la zona
    for px in (40, 150):
        c(f"fondo spostato di {px} px, senza zona", fondo=px)
        c(f"fondo spostato di {px} px, con la zona", fondo=px, zona=True)
        c(f"fondo spostato di {px} px, con la zona, radiale 100 px", fondo=px, zona=True, forma='radiale', ampiezza=100.0)
    return casi


def gia_fatti():
    fatti = set()
    if os.path.exists(RISULTATI):
        for r in open(RISULTATI, encoding='utf-8').read().splitlines():
            d = json.loads(r)
            fatti.add((d['caso'], d['seme']))
    return fatti


def tabelle():
    righe = [json.loads(r) for r in open(RISULTATI, encoding='utf-8').read().splitlines()]
    per_caso = {}
    for r in righe:
        per_caso.setdefault(r['caso'], []).append(r)
    out = []

    def ms(v):
        v = [x for x in v if x is not None]
        if not v:
            return '      -      '
        return f"{np.mean(v):7.3f} ±{np.std(v):6.3f}"

    out.append(f"{'caso':<66} n  {'errore vero mediano':>15} {'p95':>15} {'max':>9} {'0-150 dal bordo':>15} {'150-500':>15} "
               f"{'oltre 500':>15} {'verifica progr.':>15} {'>5px':>5}  livelli  s")
    for nome, _, _ in casistica():
        rr = per_caso.get(nome)
        if not rr:
            continue
        ok = [r for r in rr if not r.get('errore')]
        if not ok:
            out.append(f"{nome:<66} ERRORE: {rr[0]['errore']}"); continue
        fa = lambda k: [(r['vero'].get('fascia ' + k) or {}).get('mediana') for r in ok]
        liv = '/'.join('sì' if any((l.get('applicato') for l in r['livelli'][i:i + 1])) else 'no'
                       for i in range(2) for r in ok[:1])
        out.append(f"{nome:<66} {len(ok)}  {ms([r['vero']['mediana'] for r in ok])} {ms([r['vero']['p95'] for r in ok])} "
                   f"{max(r['vero']['max'] for r in ok):9.1f} {ms(fa('0-150'))} {ms(fa('150-500'))} {ms(fa('oltre 500'))} "
                   f"{ms([r['verifica']['residuo_mediano_px'] for r in ok])} "
                   f"{100 * np.mean([r['vero']['oltre_5px'] for r in ok]):4.0f}%  {liv:7s} {np.mean([r['t_align_s'] for r in ok]):4.0f}"
                   + (f"  [{len(rr) - len(ok)} in errore]" if len(ok) < len(rr) else ''))
    testo = "\n".join(out)
    open(TABELLE, 'w', encoding='utf-8').write(testo + "\n")
    print(testo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solo', default=None)
    ap.add_argument('--tabelle', action='store_true')
    ap.add_argument('--ripetizioni', type=int, default=None)
    args = ap.parse_args()
    if not args.tabelle:
        fatti = gia_fatti()
        casi = [x for x in casistica() if not args.solo or args.solo in x[0]]
        tot = sum((args.ripetizioni or rip) for _, _, rip in casi)
        k = 0
        for nome, caso, rip in casi:
            for seme in range(args.ripetizioni or rip):
                k += 1
                if (nome, seme) in fatti:
                    continue
                rec = esegui(nome, caso, seme)
                with open(RISULTATI, 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{k}/{tot}] {nome} (seme {seme}): " + (rec.get('errore') or
                      f"errore vero {rec['vero']['mediana']:.3f} px (p95 {rec['vero']['p95']:.2f}, max {rec['vero']['max']:.1f}), "
                      f"verifica {rec['verifica']['residuo_mediano_px']}, {rec['t_align_s']:.0f} s"), flush=True)
    if os.path.exists(RISULTATI):
        tabelle()


if __name__ == '__main__':
    main()
