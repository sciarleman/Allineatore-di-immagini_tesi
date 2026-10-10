"""
Mosaico da cavalletto: tasselli ripresi RUOTANDO la camera, un'omografia per
tassello, e l'immagine totale.

    python Mosaico\\mosaico_omografia.py --cartella immagini\\tasselli --out-dir risultati\\mosaico

PERCHE' UN MODULO A PARTE. mosaico.py ricompone tasselli ripresi da un carrello:
la camera trasla parallela al dipinto, i vicini stanno su una griglia regolare e
a ogni tassello basta al piu' un'affine. Su un affresco ripreso dal cavalletto
(25 scatti da 36 Mpx, ottobre 2026) quelle ipotesi non reggono: la camera ruota
fra uno scatto e l'altro, i tasselli sono ruotati fino a 3.7 gradi e scalati di
qualche percento, le righe sono sfalsate fino a 2500 px. Misurato: il programma
lasciava le cuciture a 6.3 px di mediana (massimo 215), con 27 coppie agganciate
su 40; sulle stesse corrispondenze un'omografia per tassello dimezza lo scarto
sui punti tenuti da parte (2.8 px contro 5.7 dell'affine). Per una superficie
piana vista da punti diversi la relazione fra due scatti E' un'omografia:
l'affine e' l'approssimazione che vale solo per il carrello. Il resto dello
scarto era la distorsione dell'obiettivo: con un coefficiente radiale comune a
tutti i tasselli lo scarto fuori campione scende a 0.29 px, e le cuciture a
piena risoluzione a 0.17 px di mediana, 1.2 al massimo, con 77 coppie su 77.

Qui dentro cambia quindi tutto cio' che dipendeva da quelle ipotesi, e mosaico.py
resta com'e':

  1. COPPIE SENZA GRIGLIA. I vicini si trovano con punti caratteristici (SIFT) e
     RANSAC fra tutte le coppie di tasselli: non servono righe, colonne e
     percorso, uno scatto doppio e' solo un tassello in piu', e le coppie in
     diagonale chiudono altri anelli.
  2. CORRISPONDENZE FINI come sempre, con la correlazione di fase a blocchi sul
     modulo del gradiente, ma sul vicino GIA' RIPORTATO sul tassello con
     l'omografia dei punti caratteristici: lo spostamento da misurare e' quasi
     nullo anche dove i due scatti sono ruotati e scalati (e' lo stesso motivo
     per cui l'allineatore pre-allinea prima della seconda passata).
  3. STIMA GLOBALE con un'omografia per tassello, scritta attorno al centro del
     tassello: m = t + (d + E u) / (1 + h.u / L), con u = d / L. I due termini
     prospettici h sono, come gli altri, pixel di spostamento sul bordo del
     tassello.
  4. IL FRENO SUI TERMINI PROSPETTICI. Sono i meno vincolati (si stimano da fasce
     strette) e un loro errore, propagato di tassello in tassello, cresce con la
     distanza dall'ancora piu' in fretta di un errore di posizione. Una penalita'
     li tira verso zero: freno infinito = un'affine per tassello, freno zero =
     omografia libera. Misurato su tasselli con le pose note, pero', il freno
     aiuta solo quando la ripresa e' in realta' da carrello; quando la camera
     ruota, frenare deforma il mosaico (vedi scegli_freno). Percio' in automatico
     si usa il freno piu' forte fra quelli che alle cuciture non costano nulla,
     e su una ripresa da cavalletto questo vuol dire quasi sempre nessuno.
  5. LA RETTILINEITA'. Le cuciture dicono se i vicini combaciano, non se il
     mosaico ha la forma vera: per quello serve qualcosa di dritto nella realta'.
     Dato un file con delle linee (il bordo del supporto, una cornice dipinta a
     filo), si cerca il bordo lungo ognuna, tassello per tassello, e si misura di
     quanto i punti trovati si scostano da una retta nel mosaico. Vedi
     rileva_linee: le linee si indicano a mano, come la zona del dipinto.
  6. L'IMMAGINE TOTALE. mosaico.py consegna i livelli e lascia la fusione a
     Photoshop; qui si scrive anche un'immagine unica. La scelta e' solo
     geometrica: in ogni punto vale il tassello di cui quel punto e' piu'
     interno, con una dissolvenza di pochi pixel lungo il confine. Luminosita' e
     colore dei tasselli non vengono raccordati.

Cosa NON fa: i livelli per Photoshop (li scrive mosaico.py, con l'affine), e
nessuna correzione del colore.

Convenzione, dichiarata anche nel JSON: per ogni tassello una matrice 3x3 H con
    [x_mosaico, y_mosaico, 1] ~ H @ [x_tassello, y_tassello, 1]
in pixel a piena risoluzione, origine nel centro del pixel in alto a sinistra.
Se e' stata stimata la distorsione radiale, (x_tassello, y_tassello) sono le
coordinate CORRETTE: p_corretto = c + (p - c) (1 + k |p - c|^2 / L^2), con c il
centro del tassello e L meta' del suo lato lungo.
"""

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime

import numpy as np
import cv2

_QUI = os.path.dirname(os.path.abspath(__file__))
if _QUI not in sys.path:
    sys.path.insert(0, _QUI)

import mosaico as base                      # mette anche l'allineatore sul percorso
import image_io
import tiff_memmap
import block_matching as bm


# Dal freno piu' forte (un'affine per tassello) all'omografia libera. L'unita' e'
# px^2 di scarto per ogni corrispondenza del tassello, per px^2 di termine
# prospettico: a 1e-3 un termine prospettico di 30 px costa quanto uno scarto di
# 1 px su ogni corrispondenza.
SCALA_FRENO = (float('inf'), 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0)
_FRENO_INFINITO = 1e7

MIN_COERENTI = 40          # punti caratteristici coerenti perche' una coppia sia creduta
MIN_SOVRAPPOSIZIONE = 0.03  # frazione del tassello coperta dal vicino


# ---------------------------------------------------------------------------
# Il modello di un tassello
# ---------------------------------------------------------------------------

def _applica_om(par, d, L):
    """
    Posizione nel mosaico di punti del tassello, con le derivate.

    Il tassello e' descritto attorno al suo centro, come in mosaico._applica:
    d = p - centro, u = d / L, e ogni incognita e' in PIXEL di spostamento sul
    raggio L. In piu' dell'affine ci sono i due termini prospettici h:

        m = t + (d + E u) / (1 + h.u / L)

    La divisione agisce attorno al centro del tassello e non attorno all'origine
    del mosaico: cosi' un pixel di h vale un pixel sul bordo del tassello
    dovunque il tassello stia, e il freno ha lo stesso significato per tutti.
    Resta un'omografia qualunque: ((A + t h^T) d + t) / (1 + h.d).

    Args:
        par: (N, 8) una riga di parametri per PUNTO (gia' indicizzata sul
            tassello di ogni punto): tx, ty, e0..e3, h0, h1
        d: (N, 2)

    Returns:
        (m (N, 2), J (N, 2, 8) derivate sui parametri, D (N, 2, 2) derivate su d)
    """
    u = d / L
    n = d + np.column_stack([par[:, 2] * u[:, 0] + par[:, 3] * u[:, 1],
                             par[:, 4] * u[:, 0] + par[:, 5] * u[:, 1]])
    s = 1.0 + (par[:, 6] * u[:, 0] + par[:, 7] * u[:, 1]) / L
    m = par[:, :2] + n / s[:, None]
    J = np.zeros((len(d), 2, 8))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 0, 2], J[:, 0, 3] = u[:, 0] / s, u[:, 1] / s
    J[:, 1, 4], J[:, 1, 5] = u[:, 0] / s, u[:, 1] / s
    J[:, :, 6] = -n * (u[:, 0] / (L * s * s))[:, None]
    J[:, :, 7] = -n * (u[:, 1] / (L * s * s))[:, None]
    D = np.zeros((len(d), 2, 2))
    L2s2 = L * L * s * s
    D[:, 0, 0] = (1.0 + par[:, 2] / L) / s - n[:, 0] * par[:, 6] / L2s2
    D[:, 0, 1] = (par[:, 3] / L) / s - n[:, 0] * par[:, 7] / L2s2
    D[:, 1, 0] = (par[:, 4] / L) / s - n[:, 1] * par[:, 6] / L2s2
    D[:, 1, 1] = (1.0 + par[:, 5] / L) / s - n[:, 1] * par[:, 7] / L2s2
    return m, J, D


def _da_matrice(H, centro, L):
    """Da una 3x3 (pixel del tassello -> mosaico) agli otto parametri."""
    hc = H @ np.array([centro[0], centro[1], 1.0])
    t = hc[:2] / hc[2]
    B = H[:2, :2] / hc[2]
    q = H[2, :2] / hc[2]
    A = B - np.outer(t, q)
    return np.concatenate([t, ((A - np.eye(2)) * L).ravel(), q * L * L])


def _a_matrice(par, centro, L):
    """Dagli otto parametri alla 3x3 pixel del tassello -> mosaico."""
    t = par[:2]
    A = np.eye(2) + par[2:6].reshape(2, 2) / L
    q = par[6:8] / (L * L)
    B = A + np.outer(t, q)
    H = np.eye(3)
    H[:2, :2] = B
    H[:2, 2] = t - B @ centro
    H[2, :2] = q
    H[2, 2] = 1.0 - q @ centro
    return H


class Posa:
    """
    Dove sta un tassello nel mosaico: l'omografia e, se stimata, la distorsione
    radiale dell'obiettivo (un solo coefficiente, uguale per tutti i tasselli).
    """

    def __init__(self, H, centro, L, k=0.0):
        self.H = np.asarray(H, dtype=np.float64)
        self.Hi = np.linalg.inv(self.H)
        self.c = np.asarray(centro, dtype=np.float64)
        self.L = float(L)
        self.k = float(k)

    def spostata(self, dx, dy):
        T = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])
        return Posa(T @ self.H, self.c, self.L, self.k)

    def avanti(self, p):
        """Pixel del tassello -> mosaico."""
        d = np.asarray(p, dtype=np.float64).reshape(-1, 2) - self.c
        if self.k:
            d = d * (1.0 + self.k * (d * d).sum(axis=1) / self.L ** 2)[:, None]
        q = np.column_stack([d + self.c, np.ones(len(d))]) @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def indietro(self, m):
        """Mosaico -> pixel del tassello."""
        m = np.asarray(m, dtype=np.float64).reshape(-1, 2)
        q = np.column_stack([m, np.ones(len(m))]) @ self.Hi.T
        du = q[:, :2] / q[:, 2:3] - self.c
        d = du
        if self.k:
            # d (1 + k |d|^2 / L^2) = du: punto fisso, che per le distorsioni di
            # un obiettivo vero converge in tre o quattro giri. Solo attorno al
            # tassello: il polinomio vale dentro l'immagine, e a molti raggi di
            # distanza si annulla e cambia segno, riportando "dentro" punti
            # lontanissimi. Oltre due volte la distanza dell'angolo il punto e'
            # comunque fuori, e si lascia dov'e'.
            vicino = (du * du).sum(axis=1) <= 4.0 * (self.c * self.c).sum()
            dv = du[vicino]
            for _ in range(6):
                dv = du[vicino] / (1.0 + self.k * (dv * dv).sum(axis=1) / self.L ** 2)[:, None]
            d = du.copy()
            d[vicino] = dv
        return d + self.c

    def mappe(self, X0, Y0, w, h, passo=1.0):
        """Per il rettangolo del mosaico che parte in (X0, Y0), largo w e alto h
        campioni presi ogni `passo` pixel: le coordinate nel tassello di ogni
        campione, pronte per cv2.remap."""
        xs = X0 + passo * np.arange(w, dtype=np.float64)
        ys = Y0 + passo * np.arange(h, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        Hi = self.Hi
        den = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
        ux = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / den - self.c[0]
        uy = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / den - self.c[1]
        if self.k:
            # come in `indietro`: la correzione solo attorno al tassello
            vicino = (ux * ux + uy * uy) <= 4.0 * (self.c * self.c).sum()
            dx, dy = ux, uy
            for _ in range(6):
                f = np.where(vicino, 1.0 + self.k * (dx * dx + dy * dy) / self.L ** 2, 1.0)
                dx, dy = ux / f, uy / f
            ux, uy = dx, dy
        return (ux + self.c[0]).astype(np.float32), (uy + self.c[1]).astype(np.float32)

    def contorno(self, W, H, n=9):
        """Il bordo del tassello nel mosaico, campionato: con la distorsione i
        lati non sono segmenti, e i soli angoli non danno l'ingombro."""
        t = np.linspace(0.0, 1.0, n)
        lati = [np.column_stack([t * (W - 1), np.zeros(n)]),
                np.column_stack([np.full(n, W - 1.0), t * (H - 1)]),
                np.column_stack([(1 - t) * (W - 1), np.full(n, H - 1.0)]),
                np.column_stack([np.zeros(n), (1 - t) * (H - 1)])]
        return self.avanti(np.concatenate([l[:-1] for l in lati]))

    def descrivi(self):
        """Rotazione, scala e termine prospettico, per il log."""
        par = _da_matrice(self.H, self.c, self.L)
        A = np.eye(2) + par[2:6].reshape(2, 2) / self.L
        return {'rotazione_gradi': float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                'scala': float(np.sqrt(abs(np.linalg.det(A)))),
                'prospettico_px': float(np.hypot(par[6], par[7]))}


# ---------------------------------------------------------------------------
# 1. Punti caratteristici e coppie di vicini
# ---------------------------------------------------------------------------

def _a_8bit(g):
    if g.dtype == np.uint8:
        return g
    return image_io.a_8bit(g)


def caratteristiche(arr, riduzione=4, n_max=8000):
    """
    Punti caratteristici SIFT di un tassello, a risoluzione ridotta.

    Il contrasto si alza localmente (CLAHE) prima di cercarli: una parete
    intonacata ha poca dinamica e senza i punti si addensano sui pochi bordi
    forti, lasciando scoperte proprio le fasce di sovrapposizione.

    Returns:
        (punti (N, 2) in pixel a piena risoluzione, descrittori (N, 128))
    """
    H, W = arr.shape[:2]
    g, _ = base._regione_ridotta(arr, 0, 0, W, H, riduzione)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(_a_8bit(g))
    kp, de = cv2.SIFT_create(nfeatures=n_max, contrastThreshold=0.02).detectAndCompute(g, None)
    if de is None or not len(kp):
        return np.zeros((0, 2), np.float32), None
    pt = np.float32([k.pt for k in kp]) * riduzione + (riduzione - 1) / 2.0
    return pt, de


def _poligono_comune(forma_a, forma_b, H_ab):
    """La parte di A che B vede, come poligono in coordinate di A (o None)."""
    Ha, Wa = forma_a
    Hb, Wb = forma_b
    qb = np.array([[0, 0], [Wb, 0], [Wb, Hb], [0, Hb]], dtype=np.float64)
    q = np.column_stack([qb, np.ones(4)]) @ np.linalg.inv(H_ab).T
    if np.any(q[:, 2] <= 1e-9):
        return None
    qa = (q[:, :2] / q[:, 2:3]).astype(np.float32)
    if not cv2.isContourConvex(qa.reshape(-1, 1, 2)):
        return None
    ra = np.array([[0, 0], [Wa, 0], [Wa, Ha], [0, Ha]], dtype=np.float32)
    area, poly = cv2.intersectConvexConvex(ra, qa)
    if poly is None or area <= 0:
        return None
    return poly.reshape(-1, 2), float(area)


def trova_coppie(tasselli, car, riduzione, min_coerenti=MIN_COERENTI,
                 min_sovrapposizione=MIN_SOVRAPPOSIZIONE, avanza=None):
    """
    Quali tasselli si sovrappongono, e con che omografia: tutte le coppie, senza
    griglia.

    Per ogni coppia si accoppiano i descrittori (i due piu' vicini, tenendo
    l'accoppiamento solo se il primo e' nettamente migliore del secondo) e si
    cerca con RANSAC l'omografia che ne mette d'accordo il maggior numero. Due
    tasselli che non si sovrappongono trovano comunque qualche accoppiamento
    per caso, ma non un'omografia sensata che ne spieghi decine: sugli scatti
    dell'affresco le coppie vere ne avevano da 660 a 6600, quelle false nessuna
    oltre la decina.

    Returns:
        lista di coppie {'a', 'b' (Tassello), 'H' (pixel di A -> pixel di B),
        'n_accoppiati', 'n_coerenti', 'sovrapposizione'}
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    n = len(tasselli)
    totale = n * (n - 1) // 2
    coppie, fatto = [], 0
    for ia in range(n):
        for ib in range(ia + 1, n):
            fatto += 1
            if avanza:
                avanza(fatto / totale)
            (pa, da), (pb, db) = car[ia], car[ib]
            if da is None or db is None or len(da) < 8 or len(db) < 8:
                continue
            acc = [m[0] for m in bf.knnMatch(da, db, k=2)
                   if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(acc) < min_coerenti:
                continue
            qa = pa[[m.queryIdx for m in acc]]
            qb = pb[[m.trainIdx for m in acc]]
            H, dentro = cv2.findHomography(qa, qb, cv2.RANSAC, 2.0 * riduzione)
            if H is None or int(dentro.sum()) < min_coerenti:
                continue
            a, b = tasselli[ia], tasselli[ib]
            comune = _poligono_comune((a.H, a.W), (b.H, b.W), H)
            if comune is None or not 0.5 < abs(np.linalg.det(H[:2, :2] / H[2, 2])) < 2.0:
                continue
            sov = comune[1] / float(a.W * a.H)
            if sov < min_sovrapposizione:
                continue
            coppie.append({'a': a, 'b': b, 'H': H / H[2, 2], 'n_accoppiati': len(acc),
                           'n_coerenti': int(dentro.sum()), 'sovrapposizione': sov})
    return coppie


# ---------------------------------------------------------------------------
# 2. Corrispondenze fini, sul vicino gia' riportato sul tassello
# ---------------------------------------------------------------------------

def _blocchi(ga, gb, valido, lato, passo, max_frac=0.25, min_std=1e-3):
    """
    Spostamento locale a blocchi, come block_matching.match_blocks, ma solo sui
    blocchi che stanno TUTTI nella zona valida.

    match_blocks accetta un blocco che sta nella maschera per tre quarti. Qui la
    zona non valida e' quella dove il vicino riportato non ha pixel: un blocco
    che ci sporge confronterebbe del dipinto con un bordo che non esiste.
    """
    h, w = ga.shape[:2]
    half = lato // 2
    win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
    somma = cv2.integral(valido.astype(np.uint8))
    out = []
    for cy in range(half, h - half + 1, passo):
        for cx in range(half, w - half + 1, passo):
            y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
            if somma[y1, x1] - somma[y0, x1] - somma[y1, x0] + somma[y0, x0] < lato * lato:
                continue
            a, b = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
            if a.std() < min_std or b.std() < min_std:
                continue
            (dx, dy), conf = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win)
            if abs(dx) > lato * max_frac or abs(dy) > lato * max_frac:
                continue
            out.append((cx, cy, dx, dy, conf))
    return np.array(out, dtype=np.float64).reshape(-1, 5)


def misura_fine_om(rid_a, rid_b, H_ab, fattore, blocco=256, min_conf=base.MIN_CONF_BLOCCHI):
    """
    Corrispondenze fini fra due tasselli, con la correlazione di fase a blocchi.

    mosaico.misura_fine confronta le due fasce comuni cosi' come sono, e regge
    finche' i due tasselli differiscono per una traslazione. Qui il vicino viene
    prima riportato sul tassello con l'omografia dei punti caratteristici: resta
    da misurare solo cio' che quell'omografia sbaglia, pochi pixel, anche quando
    i due scatti sono ruotati e scalati.

    Args:
        rid_a, rid_b: i due tasselli in grigio, ridotti di `fattore` (intero,
            media d'area: coord_piena = fattore * coord_ridotta + (fattore-1)/2)
        H_ab: 3x3, pixel di A -> pixel di B, a piena risoluzione

    Returns:
        dict con punti_a, punti_b (N, 2) a piena risoluzione e pesi (N,);
        'n_inlier' == 0 se non si e' agganciato.
    """
    f = int(fattore)
    c = (f - 1) / 2.0
    ha, wa = rid_a.shape[:2]
    hb, wb = rid_b.shape[:2]
    comune = _poligono_comune((ha * f, wa * f), (hb * f, wb * f), H_ab)
    if comune is None:
        return {'n_blocchi': 0, 'n_inlier': 0, 'motivo': 'nessuna zona comune'}
    poly = comune[0]
    x0 = max(0, int(np.floor(poly[:, 0].min() / f)))
    y0 = max(0, int(np.floor(poly[:, 1].min() / f)))
    x1 = min(wa, int(np.ceil(poly[:, 0].max() / f)))
    y1 = min(ha, int(np.ceil(poly[:, 1].max() / f)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return {'n_blocchi': 0, 'n_inlier': 0, 'motivo': 'zona comune troppo piccola'}
    sa = np.ascontiguousarray(rid_a[y0:y1, x0:x1])

    # ogni pixel della fascia di A, a piena risoluzione, e dove cade in B
    X, Y = np.meshgrid((np.arange(x0, x1) * f + c).astype(np.float64),
                       (np.arange(y0, y1) * f + c).astype(np.float64))
    den = H_ab[2, 0] * X + H_ab[2, 1] * Y + H_ab[2, 2]
    mx = (((H_ab[0, 0] * X + H_ab[0, 1] * Y + H_ab[0, 2]) / den - c) / f).astype(np.float32)
    my = (((H_ab[1, 0] * X + H_ab[1, 1] * Y + H_ab[1, 2]) / den - c) / f).astype(np.float32)
    valido = (mx >= 0) & (mx <= wb - 1) & (my >= 0) & (my <= hb - 1)
    sb = cv2.remap(rid_b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Almeno tre blocchi di traverso, non due come in mosaico.misura_fine: qui
    # lo spostamento da misurare e' gia' quasi nullo, quindi un blocco piu'
    # piccolo basta, e le sovrapposizioni piccole (i vicini in diagonale, che
    # chiudono gli anelli) danno abbastanza blocchi per essere credute.
    corto = min(sa.shape[0], sa.shape[1])
    lato = base._lato_fft(min(blocco, corto // 3))
    if lato < 32:
        return {'n_blocchi': 0, 'n_inlier': 0,
                'motivo': f'fascia comune troppo stretta ({corto * f} px)'}
    pts = _blocchi(bm.gradient_magnitude(sa), bm.gradient_magnitude(sb), valido, lato, lato // 2)
    esito = {'n_blocchi': int(len(pts)), 'lato_blocco_px': int(lato * f)}
    try:
        _M, inl, res = bm.fit_affine_robust(pts, min_conf=min_conf, verbose=False)
    except RuntimeError:
        esito.update(n_inlier=0, motivo=f'solo {int((pts[:, 4] >= min_conf).sum()) if len(pts) else 0} '
                                        f'blocchi affidabili')
        return esito

    pa = (inl[:, :2] + [x0, y0]) * f + c
    # il dettaglio che in A sta in pa, nel vicino riportato sta in pa + d: da li'
    # l'omografia lo riporta nei pixel veri di B
    q = np.column_stack([pa + inl[:, 2:4] * f, np.ones(len(pa))]) @ H_ab.T
    pb = q[:, :2] / q[:, 2:3]
    esito.update(
        punti_a=pa, punti_b=pb, pesi=np.clip(inl[:, 4], 1e-3, None), n_inlier=int(len(inl)),
        confidenza_mediana=float(np.median(inl[:, 4])),
        # di quanto l'omografia dei punti caratteristici sbagliava, in mediana
        correzione_px=float(np.median(np.hypot(inl[:, 2], inl[:, 3])) * f),
        # quanto i blocchi concordano fra loro sotto un'affine: coerenza interna
        # della misura, non la precisione del mosaico
        coerenza_px=float(np.median(np.hypot(res[:, 0], res[:, 1])) * f))
    return esito


# ---------------------------------------------------------------------------
# 3. Stima globale
# ---------------------------------------------------------------------------

def _raccogli(vincoli):
    """Le corrispondenze di tutte le coppie in vettori unici."""
    return {'i': np.concatenate([np.full(len(v['w']), v['i']) for v in vincoli]),
            'j': np.concatenate([np.full(len(v['w']), v['j']) for v in vincoli]),
            'pa': np.concatenate([v['pa'] for v in vincoli]),
            'pb': np.concatenate([v['pb'] for v in vincoli]),
            'w': np.concatenate([v['w'] for v in vincoli])}


def _residui(par, pk, V, centri, L, derivate=False):
    """
    Per ogni corrispondenza, la distanza nel mosaico fra i due punti (vettore).

    pk e' la distorsione radiale in pixel sul raggio L: il punto del tassello
    viene prima corretto, d -> d (1 + (pk / L) |d|^2 / L^2), e poi portato nel
    mosaico. Un solo valore per tutti i tasselli, perche' l'obiettivo e' lo
    stesso.
    """
    da = V['pa'] - centri[V['i']]
    db = V['pb'] - centri[V['j']]
    ra = (da * da).sum(axis=1) / (L * L)
    rb = (db * db).sum(axis=1) / (L * L)
    k = pk / L
    ma, Ja, Da = _applica_om(par[V['i']], da * (1.0 + k * ra)[:, None], L)
    mb, Jb, Db = _applica_om(par[V['j']], db * (1.0 + k * rb)[:, None], L)
    r = ma - mb
    if not derivate:
        return r
    Jk = (np.einsum('nij,nj->ni', Da, da * (ra / L)[:, None])
          - np.einsum('nij,nj->ni', Db, db * (rb / L)[:, None]))
    return r, Ja, Jb, Jk


def stima_globale_om(n_tasselli, centri, vincoli, ancora, L, par0, freno=0.0, lam=1e-9,
                     distorsione=False, pk0=0.0, n_iter=40):
    """
    Posiziona tutti i tasselli insieme, con un'omografia per tassello.

    Come mosaico.stima_globale: ogni corrispondenza chiede che i due punti
    finiscano nello stesso posto del mosaico, l'ancora resta ferma e `lam` tira
    debolmente la parte affine verso l'identita'. Qui pero' `lam` e' RELATIVO al
    peso delle misure del tassello e piccolissimo: deve solo fermare una
    direzione che nessuna misura vincola, non competere con le misure. Con il
    valore assoluto di mosaico.py (1e-2), su tasselli sintetici con rotazioni di
    2.5 gradi e scale del 5%, il mosaico aveva cuciture a 0.12 px e un errore
    VERO fino a 97 px: la penalita' sulla parte affine (decine di pixel al
    quadrato) pesava cento volte piu' delle poche decine di corrispondenze,
    precise al centesimo, di ogni coppia. Anche un milionesimo del peso delle
    misure lasciava 2 px agli angoli piu' lontani; con un miliardesimo l'errore
    vero e' 0.05 px di mediana e 0.4 al massimo. Quanto poco basti a spostare
    gli angoli lontani dice anche quanto siano poco vincolate le direzioni in
    cui un mosaico di omografie puo' derivare.

    In piu':

    - il FRENO sui due termini prospettici di ogni tassello, pesato con la somma
      dei pesi delle sue corrispondenze: cosi' ha lo stesso significato per un
      tassello con mille corrispondenze e per uno con cento;
    - la DISTORSIONE radiale, se richiesta: un'incognita sola, comune;
    - il problema non e' lineare (c'e' una divisione), quindi si itera con uno
      smorzamento (Levenberg-Marquardt) partendo dalle omografie concatenate
      delle coppie;
    - le corrispondenze che restano lontane pesano meno (Huber, con la soglia a
      tre volte lo scarto mediano): un blocco agganciato male non deve
      trascinare un tassello.

    Returns:
        (parametri (n_tasselli, 8), pk)
    """
    V = _raccogli(vincoli)
    par = np.array(par0, dtype=np.float64)
    par[ancora] = 0.0
    pk = float(pk0) if distorsione else 0.0
    col = np.full(n_tasselli, -1)
    # un tassello senza corrispondenze non ha niente che lo posizioni: resta
    # fuori dalle incognite, dov'e'
    presenti = set(np.unique(np.concatenate([V['i'], V['j']])).tolist())
    liberi = [t for t in range(n_tasselli) if t != ancora and t in presenti]
    col[liberi] = 8 * np.arange(len(liberi))
    P = 8 * len(liberi) + (1 if distorsione else 0)
    N_pt = len(V['w'])
    righe_i = np.flatnonzero(col[V['i']] >= 0)
    righe_j = np.flatnonzero(col[V['j']] >= 0)

    # il freno di ogni tassello e' proporzionale a quante misure ha
    peso_t = np.zeros(n_tasselli)
    np.add.at(peso_t, V['i'], V['w'])
    np.add.at(peso_t, V['j'], V['w'])
    f_eff = _FRENO_INFINITO if not np.isfinite(freno) else float(freno)
    reg = np.zeros(P)
    for t in liberi:
        reg[col[t] + 2:col[t] + 6] = lam * peso_t[t]
        reg[col[t] + 6:col[t] + 8] = f_eff * peso_t[t]

    def impacchetta(par, pk):
        x = par[liberi].ravel()
        return np.append(x, pk) if distorsione else x

    def spacchetta(x):
        p = np.zeros_like(par)
        p[liberi] = x[:8 * len(liberi)].reshape(-1, 8)
        return p, (float(x[-1]) if distorsione else 0.0)

    # le incognite regolarizzate verso zero: tutte tranne le traslazioni e pk
    def costo(x, w):
        p, k = spacchetta(x)
        r = _residui(p, k, V, centri, L)
        return float((w * (r * r).sum(axis=1)).sum() + (reg * x * x).sum())

    x = impacchetta(par, pk)
    mu = 1e-3
    for _ in range(n_iter):
        par, pk = spacchetta(x)
        r, Ja, Jb, Jk = _residui(par, pk, V, centri, L, derivate=True)
        e = np.hypot(r[:, 0], r[:, 1])
        soglia = max(1.0, 3.0 * float(np.median(e)))
        w = V['w'] * np.minimum(1.0, soglia / np.maximum(e, 1e-9))
        J = np.zeros((N_pt, 2, P))
        for q in range(8):
            J[righe_i, :, col[V['i'][righe_i]] + q] = Ja[righe_i, :, q]
            J[righe_j, :, col[V['j'][righe_j]] + q] = -Jb[righe_j, :, q]
        if distorsione:
            J[:, :, P - 1] = Jk
        Jw = (J * np.sqrt(w)[:, None, None]).reshape(-1, P)
        Nm = Jw.T @ Jw + np.diag(reg)
        g = Jw.T @ (r * np.sqrt(w)[:, None]).ravel() + reg * x
        c0 = costo(x, w)
        fatto = False
        for _tentativo in range(10):
            try:
                delta = np.linalg.solve(Nm + mu * np.diag(np.diag(Nm) + 1e-12), -g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            if costo(x + delta, w) < c0:
                x = x + delta
                mu = max(mu / 3.0, 1e-9)
                fatto = True
                break
            mu *= 5.0
        if not fatto or np.abs(delta).max() < 1e-4:
            break
    return spacchetta(x)


def _scarti_om(par, pk, vincoli, centri, L):
    """Per ogni coppia, quanto i punti corrispondenti restano lontani nel
    mosaico (px). E' coerenza con la soluzione globale, non una verifica."""
    out = []
    for v in vincoli:
        r = _residui(par, pk, _raccogli([v]), centri, L)
        out.append(np.hypot(r[:, 0], r[:, 1]))
    return out


def _concatena(n, coppie, radice):
    """
    Una prima posizione per ogni tassello, concatenando le omografie delle
    coppie lungo l'albero che parte da `radice` e passa sempre dalla coppia piu'
    solida disponibile. Serve solo da punto di partenza per la stima globale.

    Returns:
        {indice: 3x3 pixel del tassello -> pixel di `radice`} per i tasselli
        raggiungibili
    """
    G = {radice: np.eye(3)}
    resto = list(coppie)
    while True:
        migliore = None
        for cp in resto:
            ia, ib = cp['a'].indice, cp['b'].indice
            if (ia in G) != (ib in G) and (migliore is None
                                           or cp['n_coerenti'] > migliore['n_coerenti']):
                migliore = cp
        if migliore is None:
            return G
        ia, ib = migliore['a'].indice, migliore['b'].indice
        if ia in G:
            G[ib] = G[ia] @ np.linalg.inv(migliore['H'])
        else:
            G[ia] = G[ib] @ migliore['H']
        resto.remove(migliore)


def _fuori_campione(n, centri, vincoli, ancora, L, par0, freno, distorsione, semi=(0, 1)):
    """Scarto mediano sui punti tenuti da parte: meta' delle corrispondenze di
    ogni coppia per stimare, l'altra meta' per misurare (come scegli_modello)."""
    errori = []
    for s in semi:
        rng = np.random.default_rng(s)
        tr, te = [], []
        for v in vincoli:
            perm = rng.permutation(len(v['w']))
            tr.append(base._sottoinsieme(v, perm[::2]))
            te.append(base._sottoinsieme(v, perm[1::2]))
        par, pk = stima_globale_om(n, centri, tr, ancora, L, par0, freno=freno,
                                   distorsione=distorsione)
        errori.append(float(np.median(np.concatenate(_scarti_om(par, pk, te, centri, L)))))
    return float(np.mean(errori))


def _etichetta_freno(f):
    return "infinito (affine)" if not np.isfinite(f) else ("nessuno (omografia libera)"
                                                           if f == 0 else f"{f:g}")


def scegli_freno(n, centri, vincoli, ancora, L, par0, distorsione=False, tolleranza=0.01,
                 log=print):
    """
    Quanto frenare i termini prospettici, misurando.

    Si sceglie il freno piu' FORTE fra quelli che alle cuciture non costano
    nulla: lo scarto sui punti tenuti da parte entro l'1% dal migliore della
    scala (i freni si confrontano sulle stesse meta' dei punti, quindi l'1% non
    e' rumore). Non e' la regola di mosaico.scegli_modello, che allenta solo
    per un guadagno del 10%, e il motivo e' misurato.

    Su tasselli sintetici con le pose NOTE (12 tasselli da 1200 px; errore di
    posizione VERO nel punto peggiore, e fra parentesi lo scarto fuori
    campione):

        ripresa      misure       affine           freno 1e-4       omografia libera
        cavalletto   esatte       35 px  (0.80)    4.1 px (0.023)   2.2 px (0.020)
        cavalletto   rumore 0.5   37 px  (1.20)    15 px  (0.60)    22 px  (0.61)
        carrello     esatte       0.1 px (0.022)   0.3 px (0.023)   0.4 px (0.023)
        carrello     rumore 0.5   2.4 px (0.65)    16 px  (0.66)    18 px  (0.66)

    Quando le omografie sono vere (cavalletto), ogni freno abbastanza forte da
    toccare le cuciture PEGGIORA la forma del mosaico: tira verso zero termini
    prospettici che non sono zero, e l'errore si accumula allontanandosi
    dall'ancora. Con l'affine le cuciture sembrano ancora buone (0.8 px) e il
    mosaico e' sbagliato di 35 px; gia' un freno che costa il 15% alle cuciture
    raddoppia l'errore vero, e uno che ne costa il 3% lo porta da 0.7 a 1.8 px.
    Il freno serve nel caso opposto: la ripresa da
    carrello con misure rumorose, dove i termini prospettici veri sono zero e
    l'omografia libera deriva di 18 px per inseguire il rumore; li' frenare non
    costa niente alle cuciture, ed e' da questo che lo si riconosce.

    Cio' che il freno NON toglie e' la deriva dovuta al rumore quando le
    omografie sono vere (seconda riga): 15-22 px qualunque freno si scelga, e
    dalle cuciture non si vede. Per quella c'e' il controllo di rettilineita'.

    Returns:
        (freno scelto, {etichetta: scarto fuori campione in px})
    """
    tabella = {}
    for f in SCALA_FRENO:
        tabella[f] = _fuori_campione(n, centri, vincoli, ancora, L, par0, f, distorsione,
                                     semi=(0, 1, 2))
    migliore = min(tabella.values())
    scelto = next(f for f in SCALA_FRENO
                  if tabella[f] <= migliore * (1.0 + tolleranza) + 0.0005)
    log("  [freno] scarto fuori campione (cross-validation)"
        + (", con la distorsione radiale:" if distorsione else ":"))
    for f in SCALA_FRENO:
        log(f"      freno {_etichetta_freno(f):<28} {tabella[f]:.3f} px"
            + ("   <--" if f == scelto else ""))
    return scelto, {_etichetta_freno(f): v for f, v in tabella.items()}


# ---------------------------------------------------------------------------
# 4. Ricampionamento e verifica lungo le cuciture
# ---------------------------------------------------------------------------

def _ricampiona(arr, mx, my, interpolazione=cv2.INTER_LINEAR):
    """
    I pixel del tassello nelle posizioni (mx, my), e dove quelle posizioni
    cadono dentro il tassello. Si passa a cv2.remap solo il ritaglio che serve.

    Returns:
        (pixel o None se nessuna posizione cade nel tassello, maschera `dentro`)
    """
    H, W = arr.shape[:2]
    dentro = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    if not dentro.any():
        return None, dentro
    sx0 = max(0, int(np.floor(mx[dentro].min())) - 2)
    sx1 = min(W, int(np.ceil(mx[dentro].max())) + 3)
    sy0 = max(0, int(np.floor(my[dentro].min())) - 2)
    sy1 = min(H, int(np.ceil(my[dentro].max())) + 3)
    src = np.asarray(arr[sy0:sy1, sx0:sx1])
    if src.ndim == 3 and src.shape[2] > 3:
        src = src[:, :, :3]
    pix = cv2.remap(np.ascontiguousarray(src), mx - sx0, my - sy0, interpolation=interpolazione,
                    borderMode=cv2.BORDER_REPLICATE)
    return pix, dentro


def verifica_cuciture_om(coppie, pose, archivio, lato_max=512, lato_min=96, max_finestre=8,
                         min_conf=0.05, log=None):
    """
    Di quanti pixel due tasselli vicini restano fuori registro, lungo ogni
    cucitura, a PIENA risoluzione: la stessa misura di
    mosaico.verifica_cuciture, con i tasselli ricampionati dalla loro posa.

    E' un controllo di coerenza fra vicini. Non dice se il mosaico ha la forma
    vera: per quello c'e' la rettilineita'.
    """
    tutti = []
    for cp in coppie:
        a, b = cp['a'], cp['b']
        qa = pose[a.indice].avanti(a.angoli()).astype(np.float32)
        qb = pose[b.indice].avanti(b.angoli()).astype(np.float32)
        area, poly = cv2.intersectConvexConvex(qa, qb)
        if poly is None or area <= 0:
            cp['verifica'] = {'n_finestre': 0, 'motivo': 'i due tasselli non si sovrappongono'}
            continue
        poly = poly.reshape(-1, 2)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        corto, lungo = min(x1 - x0, y1 - y0), max(x1 - x0, y1 - y0)
        lato = base._lato_fft(min(lato_max, 0.8 * corto))
        if lato < lato_min:
            cp['verifica'] = {'n_finestre': 0,
                              'motivo': f'fascia comune troppo stretta ({corto:.0f} px)'}
            continue
        n_fin = int(max(1, min(max_finestre, (lungo - lato) // lato + 1)))
        t = np.linspace(0.0, 1.0, n_fin) if n_fin > 1 else np.array([0.5])
        if (x1 - x0) >= (y1 - y0):
            cx = x0 + lato / 2 + 4 + t * max(0.0, x1 - x0 - lato - 8)
            cy = np.full(n_fin, (y0 + y1) / 2.0)
        else:
            cx = np.full(n_fin, (x0 + x1) / 2.0)
            cy = y0 + lato / 2 + 4 + t * max(0.0, y1 - y0 - lato - 8)

        win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
        arr_a, arr_b = archivio(a), archivio(b)
        campioni = []
        for xc, yc in zip(cx, cy):
            X0, Y0 = int(round(xc - lato / 2)), int(round(yc - lato / 2))
            finestre = []
            for tass, arr in ((a, arr_a), (b, arr_b)):
                mx, my = pose[tass.indice].mappe(X0, Y0, lato, lato)
                # la finestra deve stare tutta nel tassello, con un margine
                if (mx.min() < 2 or my.min() < 2 or mx.max() > tass.W - 3
                        or my.max() > tass.H - 3):
                    break
                finestre.append(base._grigio(_ricampiona(arr, mx, my)[0]))
            if len(finestre) < 2:
                continue
            ga, gb = bm.gradient_magnitude(finestre[0]), bm.gradient_magnitude(finestre[1])
            if ga.std() < 1e-4 or gb.std() < 1e-4:
                continue
            (dx, dy), conf = cv2.phaseCorrelate(ga.astype(np.float64), gb.astype(np.float64), win)
            campioni.append((X0 + lato / 2.0, Y0 + lato / 2.0, dx, dy, conf))

        c = np.array(campioni, dtype=np.float64).reshape(-1, 5)
        buoni = c[c[:, 4] >= min_conf]
        mag = np.hypot(buoni[:, 2], buoni[:, 3])
        cp['verifica'] = {
            'n_finestre': int(len(c)), 'n_affidabili': int(len(buoni)),
            'lato_finestra_px': int(lato),
            'residuo_mediano_px': float(np.median(mag)) if len(mag) else None,
            'residuo_max_px': float(mag.max()) if len(mag) else None,
            'campioni': c.tolist(),
        }
        tutti += list(mag)

    tutti = np.array(tutti)
    riepilogo = {'n_affidabili': int(len(tutti))}
    if len(tutti):
        riepilogo.update(residuo_mediano_px=float(np.median(tutti)),
                         residuo_p90_px=float(np.percentile(tutti, 90)),
                         residuo_max_px=float(tutti.max()),
                         quota_sopra_1px=float((tutti > 1.0).mean()),
                         quota_sopra_2px=float((tutti > 2.0).mean()),
                         quota_sopra_5px=float((tutti > 5.0).mean()))
    if log:
        if len(tutti):
            log(f"  residuo lungo le cuciture, su {len(tutti)} finestre a piena risoluzione: "
                f"mediano {riepilogo['residuo_mediano_px']:.2f} px, 90-esimo percentile "
                f"{riepilogo['residuo_p90_px']:.2f} px, max {riepilogo['residuo_max_px']:.2f} px; "
                f"oltre 1 px {100 * riepilogo['quota_sopra_1px']:.0f}%, "
                f"oltre 2 px {100 * riepilogo['quota_sopra_2px']:.0f}%, "
                f"oltre 5 px {100 * riepilogo['quota_sopra_5px']:.0f}%")
        else:
            log("  nessuna finestra misurabile lungo le cuciture")
    return riepilogo


# ---------------------------------------------------------------------------
# 5. Rettilineita'
# ---------------------------------------------------------------------------

def leggi_linee(path, tasselli):
    """
    Le linee che nella realta' sono dritte, da un file JSON:

        {"linee": [{"nome": "bordo alto del supporto", "direzione": "orizzontale",
                    "estremi": [{"tassello": "scatto_1.jpg", "x": 120, "y": 340},
                                {"tassello": "scatto_5.jpg", "x": 7100, "y": 310}]}]}

    Ogni estremo e' un punto DI UN TASSELLO, in pixel di quel file: cosi' il file
    vale per qualunque soluzione, anche se il mosaico cambia forma o origine.
    `direzione` e' facoltativa e serve solo a confrontare fra loro le linee che
    dovrebbero essere parallele o perpendicolari. In testa al file si puo'
    mettere "ricerca_px": di quanti pixel, da ogni lato della linea indicata,
    cercare il bordo (120 se manca).

    Le linee si indicano a mano, come la zona del dipinto: cercarle in
    automatico vorrebbe dire decidere al posto di chi guarda che cosa e' dritto
    nella realta', e un bordo d'intonaco frastagliato o una crepa non lo sono.
    """
    with open(path, encoding='utf-8-sig') as fh:
        dati = json.load(fh)
    per_nome = {t.nome.lower(): t for t in tasselli}
    linee = []
    for l in dati.get('linee', []):
        estremi = []
        for e in l['estremi']:
            t = per_nome.get(os.path.basename(e['tassello']).lower())
            if t is None:
                raise ValueError(f"linea «{l.get('nome')}»: il tassello {e['tassello']} non e' "
                                 f"fra quelli del mosaico")
            estremi.append((t, float(e['x']), float(e['y'])))
        if len(estremi) != 2:
            raise ValueError(f"linea «{l.get('nome')}»: servono due estremi")
        linee.append({'nome': l.get('nome', f"linea {len(linee) + 1}"),
                      'direzione': l.get('direzione'), 'estremi': estremi})
    return linee, int(dati.get('ricerca_px', 120))


def _retta(P):
    """Retta ai minimi quadrati totali: (centro, direzione, normale)."""
    c = P.mean(axis=0)
    _u, _s, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    return c, d, np.array([-d[1], d[0]])


def rileva_linee(linee, tasselli, pose, archivio, passo=150.0, ricerca=120, larghezza=64,
                 finestra_fine=18):
    """
    Cerca il bordo lungo ogni linea, in ogni tassello che la vede.

    Lungo il segmento fra i due estremi, ogni `passo` pixel, si legge dal
    tassello una strisciolina perpendicolare alla linea (larga `larghezza` e
    lunga `ricerca` da ogni lato), se ne fa la media lungo la linea e si prende
    il punto dove il profilo cambia piu' in fretta: e' il bordo, al decimo di
    pixel. Due accorgimenti perche' sia lo stesso bordo dall'inizio alla fine:
    il verso del salto (da chiaro a scuro o viceversa) e' quello prevalente
    sulla linea, e dopo un primo giro si cerca solo vicino all'andamento
    trovato, cosi' un secondo bordo parallelo a pochi pixel non viene scambiato
    per il primo.

    Ogni tassello cerca per conto suo: dove due tasselli si sovrappongono lo
    stesso tratto viene trovato due volte, e la distanza fra i due e' la
    cucitura vista su una linea vera.

    Vicino al margine di un tassello la strisciolina non ci sta tutta: se ne usa
    il tratto che cade dentro, purche' sia lungo almeno 41 px. Scartare il
    campione, come si faceva, lasciava senza misura proprio i bordi del
    supporto, che stanno quasi sempre vicino al margine degli scatti: sul NIR
    dell'affresco il bordo alto, a 140-200 px dal margine con una ricerca di
    240, non dava nemmeno un punto.

    Il risultato e' in pixel DEI TASSELLI, quindi non dipende dalla soluzione
    usata per cercare: la stessa serie di punti si puo' valutare con qualunque
    altra posa (valuta_linee).

    Returns:
        lista di {'nome', 'direzione', 'tassello' (indici), 'punti' (N, 2)}
    """
    out = []
    o = np.arange(-ricerca, ricerca + 1, dtype=np.float64)
    a = np.arange(larghezza, dtype=np.float64) - (larghezza - 1) / 2.0
    for l in linee:
        (t0, x0, y0), (t1, x1, y1) = l['estremi']
        P0 = pose[t0.indice].avanti([[x0, y0]])[0]
        P1 = pose[t1.indice].avanti([[x1, y1]])[0]
        lung = float(np.hypot(*(P1 - P0)))
        dirv = (P1 - P0) / max(lung, 1e-9)
        nrm = np.array([-dirv[1], dirv[0]])
        profili = []                    # (tassello, ascissa, gradiente del profilo)
        for t in tasselli:
            posa = pose[t.indice]
            # il tratto di linea che cade in questo tassello
            estremi_t = posa.indietro(np.array([P0, P1]))
            if (estremi_t[:, 0].max() < 0 or estremi_t[:, 0].min() > t.W
                    or estremi_t[:, 1].max() < 0 or estremi_t[:, 1].min() > t.H):
                continue
            arr = None
            for s in np.arange(passo / 2.0, lung, passo):
                C = P0 + dirv * s
                pm = (C[None, None, :] + o[:, None, None] * nrm[None, None, :]
                      + a[None, :, None] * dirv[None, None, :]).reshape(-1, 2)
                pt = posa.indietro(pm)
                dentro = ((pt[:, 0] >= 2) & (pt[:, 1] >= 2) & (pt[:, 0] <= t.W - 3)
                          & (pt[:, 1] <= t.H - 3)).reshape(len(o), len(a)).all(axis=1)
                if not dentro.any():
                    continue
                # il tratto piu' lungo della strisciolina che sta tutto nel tassello
                bordi_tratti = np.flatnonzero(np.diff(np.concatenate([[0], dentro.view(np.int8), [0]])))
                inizi, fini = bordi_tratti[0::2], bordi_tratti[1::2]
                q = int(np.argmax(fini - inizi))
                i0, i1 = int(inizi[q]), int(fini[q])
                if i1 - i0 < 41:
                    continue
                if arr is None:
                    arr = archivio(t)
                mx = pt[:, 0].reshape(len(o), len(a))[i0:i1].astype(np.float32)
                my = pt[:, 1].reshape(len(o), len(a))[i0:i1].astype(np.float32)
                striscia = base._grigio(_ricampiona(arr, mx, my)[0]).astype(np.float32)
                prof = cv2.GaussianBlur(striscia.mean(axis=1).reshape(-1, 1), (0, 0), 2.0).ravel()
                g = np.full(len(o), np.nan)
                # tre campioni per parte restano fuori: li' il gradiente risente
                # del taglio del profilo
                g[i0 + 3:i1 - 3] = np.gradient(prof)[3:-3]
                profili.append((t.indice, s, g))
        if len(profili) < 4:
            out.append({'nome': l['nome'], 'direzione': l['direzione'],
                        'tassello': np.zeros(0, int), 'punti': np.zeros((0, 2))})
            continue
        G = np.array([p[2] for p in profili])
        S = np.array([p[1] for p in profili])
        k0 = np.nanargmax(np.abs(G), axis=1)
        verso = 1.0 if G[np.arange(len(G)), k0].sum() >= 0 else -1.0
        G = G * verso
        # primo giro: il salto piu' forte; poi l'andamento robusto di quei punti
        primo = o[np.nanargmax(G, axis=1)]
        forza = np.nanmax(G, axis=1)
        buoni = forza >= 0.35 * np.median(forza)
        coef = np.polyfit(S[buoni], primo[buoni], 1)
        for _ in range(4):
            scarto = primo - np.polyval(coef, S)
            mad = np.median(np.abs(scarto[buoni] - np.median(scarto[buoni]))) * 1.4826 + 0.5
            buoni = (forza >= 0.35 * np.median(forza)) & (np.abs(scarto) <= 3.0 * mad)
            if buoni.sum() < 4:
                break
            coef = np.polyfit(S[buoni], primo[buoni], 1)
        # secondo giro: lo stesso bordo, cercato solo vicino all'andamento
        punti, quale = [], []
        for (ti, s, _g), g in zip(profili, G):
            atteso = np.polyval(coef, s)
            sel = np.flatnonzero((np.abs(o - atteso) <= finestra_fine) & np.isfinite(g))
            if len(sel) < 3:
                continue
            k = sel[g[sel].argmax()]
            if (g[k] < 0.35 * np.median(forza) or k in (sel[0], sel[-1]) or k in (0, len(o) - 1)
                    or not (np.isfinite(g[k - 1]) and np.isfinite(g[k + 1]))):
                continue
            # vertice della parabola per i tre campioni attorno al massimo
            den = g[k - 1] - 2 * g[k] + g[k + 1]
            fine = o[k] + (0.5 * (g[k - 1] - g[k + 1]) / den if abs(den) > 1e-12 else 0.0)
            Pm = P0 + dirv * s + nrm * fine
            punti.append(pose[ti].indietro(Pm[None, :])[0])
            quale.append(ti)
        out.append({'nome': l['nome'], 'direzione': l['direzione'],
                    'tassello': np.array(quale, dtype=int),
                    'punti': np.array(punti, dtype=np.float64).reshape(-1, 2)})
    return out


def valuta_linee(rilevate, pose):
    """
    Di quanto i punti di ogni linea si scostano da una retta nel mosaico.

    Per ogni linea: la retta ai minimi quadrati totali, lo scarto quadratico
    medio e massimo dei punti (px), e la FRECCIA: quanto la linea si incurva al
    centro rispetto agli estremi, dalla parabola che meglio segue gli scarti.
    Lo scarto mescola il rumore della ricerca del bordo con la deformazione del
    mosaico; la freccia isola la parte sistematica, quella che cresce se le
    posizioni derivano.

    Una prospettiva d'insieme (tutto il mosaico visto di sbieco) lascia dritte
    le rette e qui non si vede: si vede, se le linee hanno una `direzione`, nel
    confronto fra gli angoli di quelle che dovrebbero essere parallele o
    perpendicolari. E' un indizio, non una prova: il dipinto non e' tenuto a
    essere un rettangolo.

    Returns:
        dict con 'linee' (una voce per linea) e il riepilogo
    """
    voci = []
    for r in rilevate:
        if len(r['punti']) < 6:
            voci.append({'nome': r['nome'], 'direzione': r['direzione'], 'n_punti': int(len(r['punti']))})
            continue
        P = np.concatenate([pose[t].avanti(r['punti'][r['tassello'] == t])
                            for t in np.unique(r['tassello'])])
        quale = np.concatenate([np.full(int((r['tassello'] == t).sum()), t)
                                for t in np.unique(r['tassello'])])
        c, d, nrm = _retta(P)
        buoni = np.ones(len(P), bool)
        for _ in range(4):
            sc = (P - c) @ nrm
            mad = np.median(np.abs(sc[buoni] - np.median(sc[buoni]))) * 1.4826
            nuovi = np.abs(sc - np.median(sc[buoni])) <= max(3.0 * mad, 2.0)
            if nuovi.sum() < 6 or (nuovi == buoni).all():
                break
            buoni = nuovi
            c, d, nrm = _retta(P[buoni])
        s = (P - c) @ d
        sc = (P - c) @ nrm
        lung = float(s[buoni].max() - s[buoni].min())
        s0 = 0.5 * (s[buoni].max() + s[buoni].min())
        par = np.polyfit(s[buoni] - s0, sc[buoni], 2)
        per_tassello = {int(t): float(np.mean(sc[buoni & (quale == t)]))
                        for t in np.unique(quale) if (buoni & (quale == t)).sum() >= 2}
        voci.append({
            'nome': r['nome'], 'direzione': r['direzione'],
            'n_punti': int(buoni.sum()), 'n_scartati': int((~buoni).sum()),
            'lunghezza_px': lung,
            'scarto_rms_px': float(np.sqrt(np.mean(sc[buoni] ** 2))),
            'scarto_max_px': float(np.abs(sc[buoni]).max()),
            'freccia_px': float(par[0] * (lung / 2.0) ** 2),
            'angolo_gradi': float(np.degrees(np.arctan2(d[1], d[0])) % 180.0),
            'retta': {'centro': c.tolist(), 'direzione': d.tolist()},
            'scarto_medio_per_tassello_px': per_tassello,
            'punti_mosaico': P[buoni].tolist(), 'scarti_px': sc[buoni].tolist(),
        })
    misurate = [v for v in voci if 'scarto_rms_px' in v]
    riepilogo = {'linee': voci, 'n_linee': len(misurate)}
    if misurate:
        riepilogo.update(
            scarto_rms_px=float(np.sqrt(np.mean([v['scarto_rms_px'] ** 2 for v in misurate]))),
            scarto_max_px=float(max(v['scarto_max_px'] for v in misurate)),
            freccia_max_px=float(max(abs(v['freccia_px']) for v in misurate)))
        gruppi = {}
        for v in misurate:
            if v['direzione']:
                gruppi.setdefault(v['direzione'], []).append(v['angolo_gradi'])

        def media_angoli(ang):
            # angoli di rette: si mediano sul cerchio di periodo 180 gradi
            z = np.exp(2j * np.radians(ang)).mean()
            return float(np.degrees(np.angle(z)) / 2.0 % 180.0)

        def differenza(a, b):
            return float((a - b + 90.0) % 180.0 - 90.0)

        par = {}
        for nome, ang in gruppi.items():
            m = media_angoli(ang)
            par[nome] = {'angolo_medio_gradi': m, 'n': len(ang),
                         'apertura_gradi': float(max(abs(differenza(x, m)) for x in ang) * 2)
                         if len(ang) > 1 else None}
        riepilogo['direzioni'] = par
        if len(par) == 2:
            (n1, g1), (n2, g2) = par.items()
            riepilogo['fuori_squadra_gradi'] = float(
                abs(differenza(g1['angolo_medio_gradi'], g2['angolo_medio_gradi'])) - 90.0) \
                if abs(differenza(g1['angolo_medio_gradi'], g2['angolo_medio_gradi'])) > 45 else None
    return riepilogo


def focale_implicita(par, L):
    """
    La focale che i termini prospettici implicano, se la camera ruotava.

    Per una camera che ruota attorno al suo centro, il tassello che nel mosaico
    sta in t ha un termine prospettico h = -(L^2 / f^2) t: piu' lo scatto e'
    girato, piu' e' lontano dal centro e piu' converge, con una costante che
    dipende solo dalla focale. Quanto bene i tasselli stanno su quella retta
    dice se la ripresa e' davvero da cavalletto (sugli scatti dell'affresco la
    correlazione e' -0.999, e la focale che ne esce e' 185 mm contro i 170
    nominali: l'allungamento atteso mettendo a fuoco a due metri).

    Returns:
        (focale in pixel, correlazione fra posizione e termine prospettico), o
        None se la relazione non c'e' (ripresa da carrello)
    """
    t, h = par[:, :2].ravel(), par[:, 6:8].ravel()
    if len(t) < 6 or float((t * t).sum()) <= 0 or h.std() <= 0:
        return None
    pendenza = float((t * h).sum() / (t * t).sum())
    if pendenza >= 0:
        return None
    return float(L / np.sqrt(-pendenza)), float(np.corrcoef(t, h)[0, 1])


def raddrizzamento(esito_linee, centro_ancora, focale):
    """
    L'omografia che mette il mosaico DI FRONTE al dipinto, dalle linee.

    Il mosaico nasce nel piano del sensore dello scatto ancora: e' il dipinto
    come lo vedrebbe una sola fotografia fatta da quel punto. Se in quello
    scatto la camera non era perpendicolare al dipinto, tutto il mosaico e'
    visto di sbieco: le rette restano rette, ma i lati paralleli convergono.
    Nessuna misura fra tasselli puo' accorgersene, perche' una camera che ruota
    attorno a un punto non vede la profondita'. Lo dicono solo le linee.

    Qui si assume che le linee di una stessa `direzione` siano parallele nella
    realta'. Dove convergono nel mosaico c'e' il loro punto di fuga; due punti
    di fuga danno la retta di fuga del piano del dipinto, e con la focale la sua
    inclinazione rispetto al sensore. L'omografia e' la rotazione della camera
    che annulla quell'inclinazione, K R K^-1 attorno al centro dell'ancora, e
    mette la prima direzione in orizzontale.

    Che le due direzioni siano anche PERPENDICOLARI non si impone: si misura. Lo
    scarto dall'angolo retto fra le due direzioni nello spazio, che resta dopo il
    raddrizzamento, dice quanto il dipinto somiglia a un rettangolo visto di
    sbieco. Se e' grande, o i lati non sono paralleli nella realta' o non sono
    perpendicolari, e il raddrizzamento non va creduto.

    Servono almeno due linee per direzione. Provato e scartato: con una linea
    sola in una direzione, assumere che sia ad angolo retto con le altre. Sul
    visibile dell'affresco, togliendo una linea alla volta, l'inclinazione
    trovata passava da 4.9 a 4.2-5.3 gradi e la forma cambiava fino a 70 px;
    sulla fluorescenza UV, dove il bordo basso non si distingue, il mosaico
    cosi' raddrizzato differiva da quello del visibile di 57 px in mediana e
    320 al massimo, cioe' non meglio di quello non raddrizzato (49 px). Un
    raddrizzamento che si puo' sbagliare cosi' senza che niente lo segnali e'
    peggio di nessun raddrizzamento.

    Returns:
        (G 3x3 mosaico -> mosaico raddrizzato, descrizione) oppure (None, motivo)
    """
    if not focale:
        return None, {'motivo': "i termini prospettici non dicono la focale: la ripresa non "
                                "sembra da cavalletto"}
    gruppi = {}
    for v in (esito_linee or {}).get('linee', []):
        if 'retta' in v and v.get('direzione'):
            gruppi.setdefault(v['direzione'], []).append(v['retta'])
    if len(gruppi) != 2 or any(len(g) < 2 for g in gruppi.values()):
        return None, {'motivo': "servono due direzioni, ognuna con almeno due linee misurate"}

    def piani(rette):
        # ogni retta, in coordinate centrate sull'ancora e divise per la focale,
        # e' un piano per il centro della camera
        out = []
        for r in rette:
            c = (np.array(r['centro']) - centro_ancora) / focale
            d = np.array(r['direzione'])
            nrm = np.array([-d[1], d[0]])
            out.append([nrm[0], nrm[1], -float(nrm @ c)])
        return np.array(out)

    def fuga(rette):
        # il punto di fuga: la direzione comune a tutti i piani
        v = np.linalg.svd(piani(rette))[2][-1]
        return v / np.linalg.norm(v)

    nomi = sorted(gruppi, key=lambda k: -np.mean([abs(r['direzione'][0]) for r in gruppi[k]]))
    dh, dv = fuga(gruppi[nomi[0]]), fuga(gruppi[nomi[1]])
    dh = dh if dh[0] > 0 else -dh
    dv = dv if dv[1] > 0 else -dv
    z = np.cross(dh, dv)
    z /= np.linalg.norm(z)
    R = np.array([dh, np.cross(z, dh), z])
    K = np.diag([focale, focale, 1.0])
    Tc = np.array([[1.0, 0, centro_ancora[0]], [0, 1.0, centro_ancora[1]], [0, 0, 1.0]])
    G = Tc @ K @ R @ np.linalg.inv(K) @ np.linalg.inv(Tc)
    return G / G[2, 2], {
        'focale_px': float(focale),
        'direzioni': nomi,
        'inclinazione_gradi': float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))),
        'inclinazione_attorno_alla_verticale_gradi': float(np.degrees(np.arctan2(z[0], z[2]))),
        'inclinazione_attorno_alla_orizzontale_gradi': float(np.degrees(np.arctan2(z[1], z[2]))),
        'rotazione_nel_piano_gradi': float(np.degrees(np.arctan2(dh[1], dh[0]))),
        'fuori_squadra_nello_spazio_gradi': float(np.degrees(np.arcsin(np.clip(dh @ dv, -1, 1)))),
    }


def _stampa_linee(esito, log, titolo="  [rettilineita']"):
    if not esito.get('n_linee'):
        log(f"{titolo} nessuna linea misurabile")
        return
    log(f"{titolo} {esito['n_linee']} linee: scarto dalla retta {esito['scarto_rms_px']:.2f} px "
        f"(rms), massimo {esito['scarto_max_px']:.1f} px; freccia massima "
        f"{esito['freccia_max_px']:.1f} px")
    for v in esito['linee']:
        if 'scarto_rms_px' not in v:
            log(f"      {v['nome']:<34} non misurabile ({v['n_punti']} punti)")
            continue
        log(f"      {v['nome']:<34} lunga {v['lunghezza_px']:>6.0f} px, {v['n_punti']:>3} punti: "
            f"rms {v['scarto_rms_px']:.2f} px, max {v['scarto_max_px']:.1f} px, "
            f"freccia {v['freccia_px']:+.1f} px")
    for nome, g in (esito.get('direzioni') or {}).items():
        if g['apertura_gradi'] is not None:
            log(f"      le {g['n']} linee «{nome}» divergono fra loro di "
                f"{g['apertura_gradi']:.3f} gradi")
    if esito.get('fuori_squadra_gradi') is not None:
        log(f"      fuori squadra fra le due direzioni: {esito['fuori_squadra_gradi']:+.3f} gradi "
            f"(un indizio: il dipinto non e' tenuto a essere un rettangolo)")


# ---------------------------------------------------------------------------
# 6. L'immagine totale e l'anteprima
# ---------------------------------------------------------------------------

def _crea_tiff(path, altezza, larghezza, colori, dtype, icc=None, dpi=None):
    """
    TIFF non compresso da riempire a bande, come quello dei livelli di
    mosaico.py ma senza trasparenza: con il profilo colore e la risoluzione dei
    tasselli, perche' chi apre il mosaico veda i colori dei file di partenza.
    """
    dtype = np.dtype(dtype)
    n_byte = altezza * larghezza * colori * dtype.itemsize
    if n_byte >= 4 * 1024 ** 3:
        raise ValueError(f"mosaico da {n_byte / 1024 ** 3:.1f} GB, oltre i 4 GB di un TIFF "
                         "classico (servirebbe BigTIFF).")
    ris = 300.0
    if dpi:
        try:
            ris = float(dpi[0]) or 300.0
        except (TypeError, ValueError, IndexError):
            pass
    razionale = (int(round(ris * 100)), 100)
    voci = [(256, 4, [larghezza]), (257, 4, [altezza]),
            (258, 3, [dtype.itemsize * 8] * colori), (259, 3, [1]),
            (262, 3, [2 if colori == 3 else 1]), (273, 4, [0]), (277, 3, [colori]),
            (278, 4, [altezza]), (279, 4, [n_byte]), (282, 5, [razionale]),
            (283, 5, [razionale]), (284, 3, [1]), (296, 3, [2])]
    if icc:
        voci.append((34675, 7, bytes(icc)))
    ifd = base._ifd(voci, 8)
    inizio_dati = (8 + len(ifd) + 7) // 8 * 8
    voci[5] = (273, 4, [inizio_dati])
    ifd = base._ifd(voci, 8)
    fh = open(path, 'wb')
    fh.write(b'II' + struct.pack('<HL', 42, 8) + ifd + b'\0' * (inizio_dati - 8 - len(ifd)))
    return tiff_memmap.ScrittoreRighe(fh, altezza, larghezza, colori, dtype)


def _banda(tasselli, pose, archivio, Y0, n_righe, Wm, ingombri, canali, passo=1.0,
           sfumatura=32.0):
    """
    Una banda di righe dell'immagine totale.

    In ogni punto vale il tassello di cui quel punto e' piu' INTERNO (piu'
    lontano dal bordo del tassello): e' la regola che tiene ogni tassello nella
    sua parte migliore, lontano dagli angoli dove obiettivo e ricampionamento
    rendono peggio. Lungo il confine fra due tasselli c'e' una dissolvenza di
    `sfumatura` pixel, e quanto farla larga dipende dalle cuciture. Dove due
    tasselli non combaciano al pixel la media di due immagini sfalsate e' una
    sfocatura, e va tenuta in una striscia stretta (32 px). Dove combaciano
    (sugli scatti dell'affresco le cuciture stanno a 0.17 px) mediare non costa
    nitidezza, e una dissolvenza larga (512 px) serve: i tasselli hanno
    esposizioni un po' diverse, e con 32 px lungo ogni confine restava uno
    scalino di luminosita' ben visibile.

    I valori dei pixel non sono corretti in nessun modo: una differenza di
    esposizione fra due tasselli resta. La dissolvenza la distende, non la
    toglie.

    Returns:
        (banda (n_righe, Wm, canali) float32, copertura (n_righe, Wm) bool)
    """
    somma = np.zeros((n_righe, Wm, canali), np.float32)
    peso = np.zeros((n_righe, Wm), np.float32)
    massimo = np.full((n_righe, Wm), -1.0, np.float32)
    attivi = []
    for t in tasselli:
        bx0, by0, bx1, by1 = ingombri[t.indice]
        if by1 <= Y0 or by0 >= Y0 + n_righe * passo:
            continue
        c0 = max(0, int(np.floor(bx0 / passo)))
        c1 = min(Wm, int(np.ceil(bx1 / passo)) + 1)
        if c1 <= c0:
            continue
        mx, my = pose[t.indice].mappe(c0 * passo, Y0, c1 - c0, n_righe, passo)
        # quanto ogni punto e' interno al tassello; sotto 1 non c'e' intorno per
        # interpolare
        interno = np.minimum(np.minimum(mx, my), np.minimum(t.W - 1 - mx, t.H - 1 - my))
        if interno.max() < 1.0:
            continue
        np.maximum(massimo[:, c0:c1], interno, out=massimo[:, c0:c1])
        attivi.append((t, c0, c1, mx, my, interno))
    for t, c0, c1, mx, my, interno in attivi:
        a = np.clip((interno - massimo[:, c0:c1]) / max(sfumatura, 1e-6) + 1.0, 0.0, 1.0)
        # verso il bordo del tassello il peso scende comunque a zero: dove la
        # sovrapposizione e' piu' stretta della dissolvenza, il tassello che
        # finisce non lascia uno scalino
        a *= np.clip(interno / 16.0, 0.0, 1.0)
        a[interno < 1.0] = 0.0
        if not a.any():
            continue
        pix, _ = _ricampiona(archivio(t), mx, my)
        if pix is None:
            continue
        if pix.ndim == 2:
            pix = pix[:, :, None]
        somma[:, c0:c1] += pix[:, :, :canali].astype(np.float32) * a[:, :, None]
        peso[:, c0:c1] += a
    coperto = peso > 0
    somma /= np.maximum(peso, 1e-6)[:, :, None]
    return somma, coperto


def scrivi_totale(tasselli, pose, forma, path, archivio, banda=256, sfumatura=32.0,
                  avanza=None):
    """
    L'immagine totale a piena risoluzione, scritta a bande di righe: in memoria
    stanno una banda e i tasselli che la attraversano, mai il mosaico intero.

    Returns:
        frazione del foglio coperta da almeno un tassello
    """
    Hm, Wm = forma
    primo = archivio(tasselli[0])
    dtype = primo.dtype
    if dtype not in (np.uint8, np.uint16):
        raise ValueError(f"profondita' {dtype} non supportata (servono 8 o 16 bit interi)")
    canali = 3 if primo.ndim == 3 else 1
    icc, dpi = base._profilo_e_dpi(tasselli[0].path)
    ingombri = {}
    for t in tasselli:
        q = pose[t.indice].contorno(t.W, t.H)
        ingombri[t.indice] = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
    temp = path + '.parziale'
    scrittore = _crea_tiff(temp, Hm, Wm, canali, dtype, icc, dpi)
    coperti = 0
    try:
        for yb in range(0, Hm, banda):
            bh = min(banda, Hm - yb)
            riga, coperto = _banda(tasselli, pose, archivio, yb, bh, Wm, ingombri, canali,
                                   sfumatura=sfumatura)
            coperti += int(coperto.sum())
            out = np.clip(np.rint(riga), 0, np.iinfo(dtype).max).astype(dtype)
            out[~coperto] = 0
            scrittore.scrivi(out if canali == 3 else out[:, :, 0])
            if avanza:
                avanza((yb + bh) / Hm)
        scrittore.chiudi()
    except BaseException:
        try:
            scrittore.f.close()
        except Exception:
            pass
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    os.replace(temp, path)
    return coperti / float(Hm * Wm)


def salva_anteprima_om(tasselli, pose, forma, path, coppie=(), linee=None, in_composizione=None,
                       lato=2400):
    """
    Il mosaico in piccolo, composto con la stessa regola dell'immagine totale,
    con il contorno e il numero di ogni tassello, l'esito della verifica lungo
    le cuciture (verde sotto 1 px, arancio fino a 2, rosso oltre) e, se ci
    sono, i punti trovati lungo le linee. Per questi le soglie sono piu'
    larghe (verde sotto 5 px dalla retta, arancio fino a 15, rosso oltre): un
    bordo d'intonaco non e' dritto al pixel nemmeno nella realta', e con le
    soglie delle cuciture sarebbe tutto rosso senza che il mosaico c'entri.
    """
    Hm, Wm = forma
    passo = max(1.0, max(Hm, Wm) / float(lato))
    Wp, Hp = int(np.ceil(Wm / passo)), int(np.ceil(Hm / passo))

    def leggi(t):
        # letto per intero e subito lasciato: con una banda sola ogni tassello
        # serve una volta
        return image_io.a_8bit(image_io.apri(t.path, verbose=False)[0])

    usati = [t for t in tasselli if in_composizione is None or in_composizione.get(t.indice, True)]
    ingombri = {}
    for t in tasselli:
        q = pose[t.indice].contorno(t.W, t.H)
        ingombri[t.indice] = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
    canali = 3 if tasselli[0].info.samples >= 3 else 1
    riga, coperto = _banda(usati, pose, leggi, 0.0, Hp, Wp, ingombri, canali, passo=passo,
                           sfumatura=32.0)
    if canali == 1:
        riga = np.repeat(riga, 3, axis=2)
    tela = np.clip(riga, 0, 255).astype(np.uint8)
    tela[~coperto] = 40
    del riga

    lato_t = max(t.W for t in tasselli) / passo
    fs = float(np.clip(lato_t / 220.0, 0.35, 2.5))
    spess = max(1, int(round(fs * 1.5)))
    for t in tasselli:
        q = pose[t.indice].contorno(t.W, t.H) / passo
        colore = (255, 255, 255) if (in_composizione is None
                                     or in_composizione.get(t.indice, True)) else (150, 150, 150)
        cv2.polylines(tela, [np.round(q).astype(np.int32)], True, colore, 1, cv2.LINE_AA)

    def pallino(x, y, r):
        colore = (60, 200, 60) if r < 1.0 else (255, 160, 0) if r < 2.0 else (230, 40, 40)
        cv2.circle(tela, (int(x / passo), int(y / passo)), max(2, spess + 1), colore, -1, cv2.LINE_AA)

    for cp in coppie:
        for x, y, dx, dy, conf in cp.get('verifica', {}).get('campioni', []):
            if conf < 0.05:
                cv2.circle(tela, (int(x / passo), int(y / passo)), max(2, spess + 1),
                           (140, 140, 140), -1, cv2.LINE_AA)
            else:
                pallino(x, y, float(np.hypot(dx, dy)))
    for v in (linee or {}).get('linee', []):
        for (x, y), sc in zip(v.get('punti_mosaico', []), v.get('scarti_px', [])):
            colore = (60, 200, 60) if abs(sc) < 5.0 else (255, 160, 0) if abs(sc) < 15.0 \
                else (230, 40, 40)
            c = (int(x / passo), int(y / passo))
            cv2.drawMarker(tela, c, colore, cv2.MARKER_TILTED_CROSS, max(5, 2 * spess), 1,
                           cv2.LINE_AA)
    for t in tasselli:
        c = pose[t.indice].avanti([t.centro])[0] / passo
        testo = str(t.indice + 1)
        (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, fs, spess)
        org = (int(c[0] - tw / 2), int(c[1] + th / 2))
        cv2.putText(tela, testo, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), spess + 3, cv2.LINE_AA)
        cv2.putText(tela, testo, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), spess,
                    cv2.LINE_AA)
    image_io.scrivi_immagine(path, cv2.cvtColor(tela, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# 7. Tutto insieme
# ---------------------------------------------------------------------------

def _doppioni(tasselli, pose, nitidezza, soglia=0.85):
    """
    Gli scatti doppi: due tasselli che inquadrano quasi la stessa cosa (uno
    copre l'altro per oltre l'85%). Restano tutti e due nella stima, dove sono
    una misura in piu'; nell'immagine totale entra solo il piu' nitido, perche'
    la media di due scatti sfalsati anche di un pixel e' una sfocatura su tutto
    il tassello.

    Returns:
        {indice: True se entra nell'immagine totale}
    """
    dentro = {t.indice: True for t in tasselli}
    quad = {t.indice: pose[t.indice].avanti(t.angoli()).astype(np.float32) for t in tasselli}
    for a in tasselli:
        for b in tasselli:
            if b.indice <= a.indice or not (dentro[a.indice] and dentro[b.indice]):
                continue
            area, _ = cv2.intersectConvexConvex(quad[a.indice], quad[b.indice])
            piccolo = min(cv2.contourArea(quad[a.indice]), cv2.contourArea(quad[b.indice]))
            if piccolo > 0 and area / piccolo >= soglia:
                peggiore = a if nitidezza[a.indice] < nitidezza[b.indice] else b
                dentro[peggiore.indice] = False
    return dentro


def componi(immagini, cartella_out, fattore=2, freno='auto', distorsione='auto', linee=None,
            raddrizza=False, scrivi_immagine=True, formati=None, sfumatura='auto',
            riduzione_sift=4, min_coerenti=MIN_COERENTI, verbose=True, progress=None):
    """
    Dal gruppo di tasselli alle pose e all'immagine totale.

    Args:
        immagini: i file, in un ordine qualunque
        fattore: riduzione usata per la misura fine (2 = come la precisione
            normale dell'allineatore)
        freno: 'auto' (scelto in cross-validation, vedi scegli_freno) oppure un
            numero: 0 = omografia libera, inf = un'affine per tassello
        distorsione: 'auto' aggiunge la distorsione radiale dell'obiettivo solo
            se lo scarto fuori campione scende di almeno il 10% e di 0.05 px;
            'si' e 'no' la impongono
        linee: file JSON con le linee dritte per il controllo di rettilineita'
            (vedi leggi_linee), o None
        raddrizza: True scrive ANCHE una seconda immagine, «raddrizzata»: il
            mosaico rimesso di fronte al dipinto usando le linee (vedi
            raddrizzamento). La prima resta com'e', senza nessuna ipotesi.
        scrivi_immagine: False = solo stima e verifica
        formati: formati dell'immagine totale ('TIFF', 'PNG', 'JPEG'); senza
            scelta, il TIFF piu' il formato dei tasselli
        sfumatura: larghezza in px della dissolvenza fra tasselli nell'immagine
            totale; 'auto' = 512 se nove cuciture su dieci stanno sotto 0.75 px,
            altrimenti 32 (vedi _banda)

    Returns:
        il report (dict), salvato anche come mosaico.json
    """
    t_inizio = time.time()

    def avanza(testo, frazione):
        if progress:
            progress(testo, frazione)

    def log(*a, **k):
        if verbose:
            print(*a, **k, flush=True)

    os.makedirs(cartella_out, exist_ok=True)
    tasselli = [base.Tassello(k, p, 0, 0) for k, p in enumerate(immagini)]
    n = len(tasselli)
    if n < 2:
        raise ValueError("servono almeno due tasselli")
    L = max(max(t.W, t.H) for t in tasselli) / 2.0
    centri = np.array([t.centro for t in tasselli])

    log("=" * 72)
    log(f"MOSAICO DA CAVALLETTO  {n} tasselli, un'omografia per tassello")
    log("=" * 72)
    for t in tasselli:
        log(f"  {t.indice + 1:>3}  {t.info!r}")

    # ---- 1. punti caratteristici ------------------------------------------
    log(f"\n[1/6] punti caratteristici (riduzione {riduzione_sift}) e tasselli ridotti di "
        f"{fattore} per la misura fine")
    car, ridotte, nitidezza = {}, {}, {}
    for k, t in enumerate(tasselli):
        arr, _ = image_io.apri(t.path, verbose=False)
        car[t.indice] = caratteristiche(arr, riduzione_sift)
        ridotte[t.indice], _ = base._regione_ridotta(arr, 0, 0, t.W, t.H, fattore)
        g = cv2.resize(_a_8bit(ridotte[t.indice]), None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA)
        nitidezza[t.indice] = float(cv2.Laplacian(g, cv2.CV_32F).var())
        del arr
        avanza("punti caratteristici", 0.10 * (k + 1) / n)

    # ---- 2. coppie -----------------------------------------------------------
    log(f"\n[2/6] ricerca dei vicini fra tutte le {n * (n - 1) // 2} coppie")
    coppie = trova_coppie(tasselli, car, riduzione_sift, min_coerenti=min_coerenti,
                          avanza=lambda f: avanza("ricerca dei vicini", 0.10 + 0.10 * f))
    del car
    for cp in coppie:
        log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} {cp['n_coerenti']:>5} punti "
            f"coerenti su {cp['n_accoppiati']}, sovrapposizione "
            f"{100 * cp['sovrapposizione']:.0f}%")
    if not coppie:
        raise RuntimeError("Nessuna coppia di tasselli si sovrappone in modo riconoscibile: "
                           "controlla che i file siano gli scatti di uno stesso dipinto.")

    # ---- 3. corrispondenze fini -------------------------------------------
    log(f"\n[3/6] corrispondenze fini sul vicino riportato (riduzione {fattore})")
    vincoli = []
    for k, cp in enumerate(coppie):
        fi = misura_fine_om(ridotte[cp['a'].indice], ridotte[cp['b'].indice], cp['H'], fattore)
        cp['fine'] = fi
        cp['stato'] = 'misurata' if fi.get('n_inlier', 0) >= 12 else 'non misurata'
        if cp['stato'] == 'misurata':
            vincoli.append({'i': cp['a'].indice, 'j': cp['b'].indice, 'pa': fi['punti_a'],
                            'pb': fi['punti_b'], 'w': fi['pesi'], 'coppia': cp})
            log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} {fi['n_inlier']}/"
                f"{fi['n_blocchi']} blocchi coerenti, correzione {fi['correzione_px']:.2f} px, "
                f"coerenza interna {fi['coerenza_px']:.2f} px")
        else:
            log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} NON MISURATA: "
                f"{fi.get('motivo', 'troppo pochi blocchi coerenti')}")
        avanza("corrispondenze fini", 0.20 + 0.20 * (k + 1) / len(coppie))
    del ridotte
    misurate = [cp for cp in coppie if cp['stato'] == 'misurata']
    if not misurate:
        raise RuntimeError("Nessuna coppia misurata: le fasce comuni non hanno abbastanza "
                           "struttura per la misura fine.")

    # ---- 4. stima globale ---------------------------------------------------
    log("\n[4/6] stima globale")
    avanza("stima globale", 0.42)
    conta = {t.indice: sum(t in (cp['a'], cp['b']) for cp in misurate) for t in tasselli}
    radice = max(conta, key=conta.get)
    G = _concatena(n, misurate, radice)
    fuori = [t for t in tasselli if t.indice not in G]
    if fuori:
        log(f"  [attenzione] {len(fuori)} tasselli non sono legati agli altri da nessuna coppia "
            f"misurata e restano fuori dal mosaico: " + ", ".join(t.nome for t in fuori))
        tasselli_in = [t for t in tasselli if t.indice in G]
    else:
        tasselli_in = list(tasselli)
    # l'ancora: fra i tasselli con piu' coppie, il piu' vicino al centro del
    # mosaico, come in mosaico._scegli_ancora e per le stesse ragioni
    pos = {i: (H @ np.append(centri[i], 1.0)) for i, H in G.items()}
    pos = {i: p[:2] / p[2] for i, p in pos.items()}
    baricentro = np.mean(list(pos.values()), axis=0)
    molte = np.median(list(conta.values()))
    ancora = min((i for i in G if conta[i] >= molte),
                 key=lambda i: float(np.hypot(*(pos[i] - baricentro))))
    Gi = np.linalg.inv(G[ancora])
    sposta = np.array([[1.0, 0, -centri[ancora][0]], [0, 1.0, -centri[ancora][1]], [0, 0, 1.0]])
    par0 = np.zeros((n, 8))
    for i, H in G.items():
        par0[i] = _da_matrice(sposta @ Gi @ H, centri[i], L)
    vincoli = [v for v in vincoli if v['i'] in G and v['j'] in G]
    # i tasselli fuori dal grafo non hanno corrispondenze: le loro incognite
    # restano dove sono, tenute ferme dal termine verso l'identita'
    log(f"  ancora il tassello {ancora + 1} ({tasselli[ancora].nome}); "
        f"{len(misurate)}/{len(coppie)} coppie misurate, "
        f"{sum(len(v['w']) for v in vincoli)} corrispondenze")

    usa_dist = distorsione == 'si'
    tabelle = {}
    if freno == 'auto':
        freno_scelto, tabelle['freno'] = scegli_freno(n, centri, vincoli, ancora, L, par0,
                                                      distorsione=usa_dist, log=log)
    else:
        freno_scelto = float(freno)
    avanza("stima globale", 0.50)
    if distorsione == 'auto':
        senza = _fuori_campione(n, centri, vincoli, ancora, L, par0, freno_scelto, False)
        con = _fuori_campione(n, centri, vincoli, ancora, L, par0, freno_scelto, True)
        usa_dist = con < senza * 0.90 and senza - con >= 0.05
        tabelle['distorsione'] = {'senza_px': senza, 'con_px': con}
        log(f"  [distorsione] scarto fuori campione: senza {senza:.3f} px, con {con:.3f} px"
            f" -> {'adottata' if usa_dist else 'non adottata'}")
        if usa_dist and freno == 'auto':
            # con la distorsione tolta, quanto serve l'omografia va rimisurato
            freno_scelto, tabelle['freno_con_distorsione'] = scegli_freno(
                n, centri, vincoli, ancora, L, par0, distorsione=True, log=log)
    avanza("stima globale", 0.56)

    par, pk = stima_globale_om(n, centri, vincoli, ancora, L, par0, freno=freno_scelto,
                               distorsione=usa_dist)
    # una coppia che non torna con le altre: come il controllo degli anelli
    scarti = _scarti_om(par, pk, vincoli, centri, L)
    med_coppie = np.array([float(np.median(s)) for s in scarti])
    soglia_anelli = max(3.0, 5.0 * float(np.median(med_coppie)))
    for v, m in zip(vincoli, med_coppie):
        v['coppia']['scarto_globale_px'] = float(m)
    via = [v for v, m in zip(vincoli, med_coppie) if m > soglia_anelli]
    if via and len(vincoli) - len(via) >= len(tasselli_in) - 1:
        for v in via:
            v['coppia']['stato'] = 'incoerente con le altre'
            log(f"  [anelli] {v['i'] + 1} - {v['j'] + 1} esclusa: scarto "
                f"{v['coppia']['scarto_globale_px']:.1f} px dalla soluzione globale, contro una "
                f"soglia di {soglia_anelli:.1f} px")
        vincoli = [v for v in vincoli if v not in via]
        if set(_concatena(n, [v['coppia'] for v in vincoli], ancora)) == set(G):
            par, pk = stima_globale_om(n, centri, vincoli, ancora, L, par0, freno=freno_scelto,
                                       distorsione=usa_dist)
            scarti = _scarti_om(par, pk, vincoli, centri, L)
    tutti_scarti = np.concatenate(scarti)
    k_dist = pk / L

    # ---- il sistema del mosaico: origine nell'angolo dell'ingombro ---------
    pose = {t.indice: Posa(_a_matrice(par[t.indice], centri[t.indice], L), centri[t.indice], L,
                           k_dist) for t in tasselli_in}
    bordi = np.concatenate([pose[t.indice].contorno(t.W, t.H) for t in tasselli_in])
    origine = np.floor(bordi.min(axis=0))
    pose = {i: p.spostata(-origine[0], -origine[1]) for i, p in pose.items()}
    Wm, Hm = (int(v) for v in np.ceil(bordi.max(axis=0) - origine) + 1)
    descr = {i: p.descrivi() for i, p in pose.items()}
    log(f"  freno {_etichetta_freno(freno_scelto)}"
        + (f"; distorsione radiale {pk:+.2f} px sul bordo del tassello" if usa_dist else ""))
    log(f"  scarto delle corrispondenze dalla soluzione globale: mediano "
        f"{np.median(tutti_scarti):.2f} px, p95 {np.percentile(tutti_scarti, 95):.2f} px")
    log(f"  rotazioni fra {min(d['rotazione_gradi'] for d in descr.values()):+.3f} e "
        f"{max(d['rotazione_gradi'] for d in descr.values()):+.3f} gradi, scale fra "
        f"{min(d['scala'] for d in descr.values()):.4f} e "
        f"{max(d['scala'] for d in descr.values()):.4f}, termine prospettico fino a "
        f"{max(d['prospettico_px'] for d in descr.values()):.1f} px")
    log(f"  mosaico {Wm} x {Hm} px")

    in_comp = _doppioni(tasselli_in, pose, nitidezza)
    for t in tasselli_in:
        if not in_comp[t.indice]:
            log(f"  [doppione] {t.nome} inquadra quasi la stessa zona di un altro scatto piu' "
                f"nitido: resta nella stima, non entra nell'immagine totale")

    # ---- 5. verifica --------------------------------------------------------
    log("\n[5/6] verifica a piena risoluzione")
    avanza("verifica lungo le cuciture", 0.60)
    archivio = base._Archivio(quanti=8)
    da_verificare = [cp for cp in coppie if cp['a'].indice in pose and cp['b'].indice in pose]
    verifica = verifica_cuciture_om(da_verificare, pose, archivio, log=log)
    esito_linee, rilevate = None, []
    if linee:
        avanza("rettilineita'", 0.68)
        definite, ricerca = leggi_linee(linee, tasselli_in)
        rilevate = rileva_linee(definite, tasselli_in, pose, archivio, ricerca=ricerca)
        esito_linee = valuta_linee(rilevate, pose)
        _stampa_linee(esito_linee, log)
        # la stessa serie di punti, giudicata con gli altri freni: e' il
        # confronto che dice se allentare il freno ha deformato il mosaico
        confronto = {}
        for f in SCALA_FRENO:
            if f == freno_scelto:
                confronto[_etichetta_freno(f)] = esito_linee
                continue
            pf, kf = stima_globale_om(n, centri, vincoli, ancora, L, par0, freno=f,
                                      distorsione=usa_dist)
            pose_f = {t.indice: Posa(_a_matrice(pf[t.indice], centri[t.indice], L),
                                     centri[t.indice], L, kf / L) for t in tasselli_in}
            confronto[_etichetta_freno(f)] = valuta_linee(rilevate, pose_f)
        log("  [rettilineita'] le stesse linee con gli altri freni:")
        for f in SCALA_FRENO:
            e = confronto[_etichetta_freno(f)]
            if e.get('n_linee'):
                fs = e.get('fuori_squadra_gradi')
                log(f"      freno {_etichetta_freno(f):<28} rms {e['scarto_rms_px']:.2f} px, "
                    f"max {e['scarto_max_px']:.1f} px, freccia max {e['freccia_max_px']:.1f} px"
                    + (f", fuori squadra {fs:+.3f} gradi" if fs is not None else "")
                    + ("   <--" if f == freno_scelto else ""))
        esito_linee['con_gli_altri_freni'] = {
            k: {q: e.get(q) for q in ('scarto_rms_px', 'scarto_max_px', 'freccia_max_px',
                                      'fuori_squadra_gradi', 'direzioni')}
            for k, e in confronto.items()}

    momento = datetime.now().strftime('%Y-%m-%d %H-%M')
    icc_tasselli, dpi_tasselli = base._profilo_e_dpi(tasselli[0].path)
    if sfumatura == 'auto':
        p90 = verifica.get('residuo_p90_px')
        sfumatura = 512.0 if (p90 is not None and p90 <= 0.75) else 32.0
        if scrivi_immagine:
            log(f"\n  dissolvenza fra i tasselli: {sfumatura:.0f} px"
                + (f" (cuciture a {p90:.2f} px al 90-esimo percentile)" if p90 is not None else ""))
    sfumatura = float(sfumatura)

    def consegna(pose_u, forma, etichetta, esito_l, f0, f1):
        """Anteprima e immagine totale per una serie di pose."""
        Hu, Wu = forma
        path_ant = os.path.join(cartella_out, f"anteprima_mosaico{etichetta}.png")
        avanza("anteprima", f0)
        salva_anteprima_om(tasselli_in, pose_u, (Hu, Wu), path_ant,
                           coppie=() if etichetta else da_verificare, linee=esito_l,
                           in_composizione=in_comp)
        tot, altri_u, cop = None, [], None
        if scrivi_immagine:
            log(f"\n[6/6] immagine totale{etichetta} {Wu} x {Hu} px")
            import formati as fmt
            est_tasselli = os.path.splitext(tasselli[0].path)[1].lower()
            scelti = fmt.normalizza(formati) or ['.tif'] + (
                [fmt.normalizza([est_tasselli])[0]] if est_tasselli in ('.png', '.jpg', '.jpeg')
                else [])
            tot = os.path.join(cartella_out, f"mosaico{etichetta} ({momento}).tif")
            usati = [t for t in tasselli_in if in_comp[t.indice]]
            # una banda attraversa due file di tasselli: se ne tengono aperti
            # abbastanza da non rileggere gli stessi JPEG a ogni banda
            arch = base._Archivio(quanti=14)
            cop = scrivi_totale(usati, pose_u, (Hu, Wu), tot, arch, sfumatura=sfumatura,
                                avanza=lambda f: avanza("immagine totale", f0 + (f1 - f0) * f))
            arch.aperti.clear()
            log(f"  scritta: {tot} ({os.path.getsize(tot) / 1024 ** 3:.2f} GB), "
                f"{100 * cop:.1f}% del foglio coperto dai tasselli")
            for e in scelti:
                if e == '.tif':
                    continue
                dest = fmt.con_estensione(tot, e)
                _converti_con_profilo(tot, dest, icc_tasselli, dpi_tasselli)
                log(f"  anche in {e[1:].upper()}: {dest} "
                    f"({os.path.getsize(dest) / 1024 ** 3:.2f} GB)")
                altri_u.append(dest)
            if '.tif' not in scelti:
                os.unlink(tot)
                tot, altri_u = altri_u[0], altri_u[1:]
        return path_ant, tot, altri_u, cop

    # ---- 6. immagine totale -------------------------------------------------
    archivio.aperti.clear()
    if not scrivi_immagine:
        log("\n[6/6] immagine totale non scritta (solo stima)")
    path_anteprima, totale, altri, copertura = consegna(
        pose, (Hm, Wm), "", esito_linee, 0.72, 0.84 if raddrizza else 0.97)

    # ---- la versione raddrizzata, se chiesta e se le linee la reggono ------
    foc = focale_implicita(par[[t.indice for t in tasselli_in]], L)
    if foc:
        log(f"\n  [camera] i termini prospettici seguono la posizione con correlazione "
            f"{foc[1]:+.3f}: focale implicita {foc[0]:.0f} px")
    raddrizzato = None
    if raddrizza:
        centro_anc = pose[ancora].avanti([centri[ancora]])[0]
        G, dr = raddrizzamento(esito_linee, centro_anc, foc[0] if foc else None)
        if G is None:
            log(f"  [raddrizzamento] non fatto: {dr['motivo']}")
            raddrizzato = dr
        else:
            pose_r = {i: Posa(G @ p.H, p.c, p.L, p.k) for i, p in pose.items()}
            bordi_r = np.concatenate([pose_r[t.indice].contorno(t.W, t.H) for t in tasselli_in])
            orig_r = np.floor(bordi_r.min(axis=0))
            pose_r = {i: p.spostata(-orig_r[0], -orig_r[1]) for i, p in pose_r.items()}
            Wr, Hr = (int(v) for v in np.ceil(bordi_r.max(axis=0) - orig_r) + 1)
            esito_r = valuta_linee(rilevate, pose_r)
            log(f"  [raddrizzamento] il piano del dipinto era inclinato di "
                f"{dr['inclinazione_gradi']:.2f} gradi rispetto al sensore dello scatto ancora "
                f"({dr['inclinazione_attorno_alla_verticale_gradi']:+.2f} attorno alla verticale, "
                f"{dr['inclinazione_attorno_alla_orizzontale_gradi']:+.2f} attorno "
                f"all'orizzontale), e ruotato di {dr['rotazione_nel_piano_gradi']:+.2f} gradi")
            log(f"      assunto: le linee «{dr['direzioni'][0]}» parallele fra loro, e cosi' le "
                f"«{dr['direzioni'][1]}». Misurato: fra le due direzioni mancano "
                f"{dr['fuori_squadra_nello_spazio_gradi']:+.3f} gradi all'angolo retto")
            _stampa_linee(esito_r, log, titolo="  [rettilineita' dopo il raddrizzamento]")
            ant_r, tot_r, altri_r, cop_r = consegna(pose_r, (Hr, Wr), " raddrizzato", esito_r,
                                                    0.85, 0.97)
            raddrizzato = dict(dr, mosaico={'larghezza': Wr, 'altezza': Hr},
                               mosaico_verso_raddrizzato=(
                                   np.array([[1.0, 0, -orig_r[0]], [0, 1.0, -orig_r[1]],
                                             [0, 0, 1.0]]) @ G).tolist(),
                               tasselli={str(i + 1): p.H.tolist() for i, p in pose_r.items()},
                               rettilineita=esito_r, anteprima=os.path.abspath(ant_r),
                               immagine_totale=os.path.abspath(tot_r) if tot_r else None,
                               immagine_totale_altri_formati=[os.path.abspath(p) for p in altri_r],
                               copertura_del_foglio=cop_r)

    report = {
        'programma': 'mosaico_omografia.py',
        'data': datetime.now().isoformat(timespec='seconds'),
        'convenzione': '[x_mosaico, y_mosaico, 1] ~ H @ [x_tassello, y_tassello, 1], in pixel a '
                       'piena risoluzione, origine nel centro del pixel in alto a sinistra. Con '
                       'la distorsione, x e y del tassello sono quelle corrette: '
                       'p_corretto = c + (p - c)(1 + k |p - c|^2 / L^2)',
        'parametri': {'fattore_misura_fine': fattore, 'freno_richiesto': str(freno),
                      'distorsione_richiesta': distorsione, 'sfumatura_px': sfumatura,
                      'linee': os.path.abspath(linee) if linee else None},
        'mosaico': {'larghezza': Wm, 'altezza': Hm, 'ancora': ancora + 1},
        'modello': {'tipo': 'omografia per tassello',
                    'freno': None if not np.isfinite(freno_scelto) else freno_scelto,
                    'freno_descrizione': _etichetta_freno(freno_scelto),
                    'distorsione_px_sul_bordo': pk if usa_dist else None,
                    'distorsione_k': k_dist if usa_dist else None,
                    'raggio_L_px': L, 'crossval_px': tabelle,
                    'scarto_corrispondenze_px': {'mediano': float(np.median(tutti_scarti)),
                                                 'p95': float(np.percentile(tutti_scarti, 95))}},
        'tasselli': [{
            'numero': t.indice + 1, 'file': os.path.abspath(t.path),
            'nel_mosaico': t.indice in pose,
            'nell_immagine_totale': bool(in_comp.get(t.indice, False)),
            'nitidezza': nitidezza[t.indice],
            'tassello_verso_mosaico': pose[t.indice].H.tolist() if t.indice in pose else None,
            'posa': descr.get(t.indice),
        } for t in tasselli],
        'coppie': [{
            'a': cp['a'].indice + 1, 'b': cp['b'].indice + 1, 'stato': cp['stato'],
            'punti_caratteristici_coerenti': cp['n_coerenti'],
            'sovrapposizione': cp['sovrapposizione'],
            'fine': {k: v for k, v in cp['fine'].items()
                     if k not in ('punti_a', 'punti_b', 'pesi')},
            'scarto_globale_px': cp.get('scarto_globale_px'),
            'verifica': cp.get('verifica'),
        } for cp in coppie],
        'verifica': verifica,
        'rettilineita': esito_linee,
        'focale_implicita': {'px': foc[0], 'correlazione': foc[1]} if foc else None,
        'raddrizzato': raddrizzato,
        'anteprima': os.path.abspath(path_anteprima),
        'immagine_totale': os.path.abspath(totale) if totale else None,
        'immagine_totale_altri_formati': [os.path.abspath(p) for p in altri],
        'copertura_del_foglio': copertura,
        'tempo_s': round(time.time() - t_inizio, 1),
    }
    with open(os.path.join(cartella_out, 'mosaico.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    avanza("fatto", 1.0)
    log(f"\nanteprima: {path_anteprima}")
    log(f"report:    {os.path.join(cartella_out, 'mosaico.json')}")
    log(f"tempo totale: {report['tempo_s']:.1f} s")
    return report


def _converti_con_profilo(path_tif, path_out, icc, dpi=None):
    """PNG o JPEG dal TIFF appena scritto, con lo stesso profilo colore. Serve
    l'immagine intera in memoria, come sempre per questi due formati."""
    from PIL import Image
    arr, _ = image_io.apri(path_tif, verbose=False)
    img = np.asarray(arr)
    est = os.path.splitext(path_out)[1].lower()
    if est == '.jpg' and img.dtype != np.uint8:
        img = (img.astype(np.float32) * (255.0 / np.iinfo(img.dtype).max) + 0.5).astype(np.uint8)
    massimo = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        im = Image.fromarray(img)
        extra = {'icc_profile': icc} if icc else {}
        if dpi:
            extra['dpi'] = tuple(float(v) for v in dpi[:2])
        if est == '.jpg':
            im.save(path_out, quality=95, subsampling=0, **extra)
        else:
            im.save(path_out, **extra)
    finally:
        Image.MAX_IMAGE_PIXELS = massimo
    del img, arr


def main():
    ap = argparse.ArgumentParser(
        description="Ricompone un dipinto ripreso a tasselli dal cavalletto: un'omografia per "
                    "tassello, verifica delle cuciture e della rettilineita', immagine totale.")
    sorgente = ap.add_mutually_exclusive_group(required=True)
    sorgente.add_argument('--cartella', help="cartella con i tasselli (l'ordine non conta)")
    sorgente.add_argument('--immagini', nargs='+', help="in alternativa, i file")
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--freno', default='auto',
                    help="freno sui termini prospettici: 'auto' (default, scelto misurando), "
                         "0 = omografia libera, inf = un'affine per tassello")
    ap.add_argument('--distorsione', choices=('auto', 'si', 'no'), default='auto',
                    help="distorsione radiale dell'obiettivo, comune a tutti i tasselli")
    ap.add_argument('--linee', default=None,
                    help="file JSON con le linee dritte per il controllo di rettilineita'")
    ap.add_argument('--raddrizza', action='store_true',
                    help="scrive anche il mosaico rimesso di fronte al dipinto, assumendo "
                         "parallele le linee di una stessa direzione (serve --linee)")
    ap.add_argument('--fattore', type=int, default=2,
                    help="riduzione per la misura fine: 1 = piena risoluzione, 2 = default")
    ap.add_argument('--sfumatura', default='auto',
                    help="larghezza in px della dissolvenza fra due tasselli nell'immagine "
                         "totale; 'auto' (default) = 512 se le cuciture sono sotto il pixel, "
                         "altrimenti 32")
    ap.add_argument('--formati', nargs='*', default=None,
                    help="formati dell'immagine totale: TIFF PNG JPEG (default: TIFF e il "
                         "formato dei tasselli)")
    ap.add_argument('--solo-stima', action='store_true',
                    help="stima e verifica senza scrivere l'immagine totale")
    a = ap.parse_args()
    immagini = base.elenca_immagini(a.cartella) if a.cartella else a.immagini
    componi(immagini, a.out_dir, fattore=a.fattore,
            freno=a.freno if a.freno == 'auto' else float(a.freno), distorsione=a.distorsione,
            linee=a.linee, raddrizza=a.raddrizza, scrivi_immagine=not a.solo_stima,
            formati=a.formati,
            sfumatura=a.sfumatura if a.sfumatura == 'auto' else float(a.sfumatura))


if __name__ == '__main__':
    main()
