"""
Mosaico: ricomporre un dipinto ripreso a tasselli, come livelli per Photoshop.

    python Mosaico\\mosaico.py --cartella immagini\\tasselli --righe 3 --colonne 4 ^
                              --percorso serpentina_righe --partenza alto_sinistra ^
                              --out-dir risultati\\mosaico

Quando un dipinto e' troppo grande per un solo scatto alla risoluzione voluta
lo si riprende a tasselli: la camera (o il dipinto) si sposta su una griglia di
righe e colonne, e ogni scatto si sovrappone ai vicini per una fascia. Questo
programma ritrova dove sta ogni tassello rispetto agli altri e lo consegna come
LIVELLO gia' posizionato, da aprire in Photoshop.

COSA FA E COSA NO. La fusione -- dove passa il taglio fra due tasselli, come se
ne raccordano luminosita' e colore -- resta in Photoshop, perche' tocca il
colore ed e' una decisione di chi cura l'immagine. Qui si decide solo la
GEOMETRIA: ogni tassello viene ricampionato una volta sola nel sistema del
mosaico, con al piu' una trasformazione affine, i valori dei pixel non vengono
toccati in nessun altro modo e il profilo colore dell'originale viene copiato
nel livello.

COME
  1. Il percorso di ripresa (serpentina o righe/colonne, angolo di partenza)
     assegna a ogni file la sua casella, seguendo l'ordine naturale dei nomi.
  2. Per ogni coppia di caselle ADIACENTI si misura la sovrapposizione: prima
     grossolana su anteprime ridotte (correlazione di fase a piu' picchi, con
     ogni candidato verificato), poi fine sulla sola fascia comune, con lo
     stesso block matching dell'allineatore (block_matching.py).
  3. Le misure diventano corrispondenze fra punti, e una stima GLOBALE ai minimi
     quadrati posiziona tutti i tasselli insieme.
  4. Una verifica a piena risoluzione misura, lungo ogni cucitura, di quanti
     pixel due tasselli vicini restano fuori registro.

PERCHE' TUTTE LE COPPIE ADIACENTI E NON SOLO LE CONSECUTIVE. Concatenare le
sole coppie consecutive nello scatto (1-2, 2-3, 3-4...) da' una catena: ogni
anello porta il suo piccolo errore e gli errori si sommano. In una serpentina
per righe da quattro colonne il tassello 8 sta sotto il tassello 1, ma nella
catena ci arriva dopo sette anelli, e la cucitura fra la prima e la seconda
riga raccoglie tutto l'errore accumulato. Misurando anche le coppie verticali
ogni riquadro di quattro tasselli chiude un anello: la stima globale
distribuisce gli errori invece di sommarli, e un aggancio sbagliato si
riconosce perche' non torna con gli altri. E' l'impostazione dei programmi di
riferimento per i mosaici a griglia: lo Stitching di Fiji (Preibisch et al.,
Bioinformatics 2009) e MIST del NIST (Chalfoun et al., Scientific Reports 2017),
da cui viene anche il controllo del passo del carrello (vedi
controlla_passo).

IL MODELLO DI OGNI TASSELLO, dal piu' semplice al piu' libero:
    traslazione   camera che scorre su un carrello senza ruotare
    rigida        + una piccola rotazione fra uno scatto e l'altro
    similarita    + una scala: la distanza dal dipinto cambia, o la messa a
                  fuoco rifatta a ogni scatto cambia l'ingrandimento
    affine        + scala diversa in x e in y, e taglio
Con 'auto' il modello non si decide a priori: si misura in cross-validation
quanto ciascuno riduce lo scarto su corrispondenze tenute da parte, e si passa
a un modello piu' libero solo se guadagna almeno il 10% (lo stesso criterio
con cui l'allineatore decide se il campo non rigido e' segnale o rumore) e
almeno 0.05 px: vedi scegli_modello per il perche', misurato.

Nessun modello qui corregge la DISTORSIONE DELL'OBIETTIVO, che e' radiale e
massima proprio ai bordi dello scatto, cioe' nelle fasce di sovrapposizione.
Se c'e', la verifica delle cuciture la mostra come uno scarto che cresce verso
le estremita' della fascia; e un'affine scelta in automatico con una scala
lontana da 1 e' spesso il segno che sta cercando di compensarla. Il rimedio
giusto e' a monte (profilo dell'obiettivo nello sviluppo del RAW, oppure
calibration_setup.py), uguale per tutti i tasselli e senza effetti sul colore.

POSIZIONI E BANDE. Le posizioni trovate valgono per QUEI file. Le riprese di
un'altra banda sulla stessa griglia non si possono posizionare con le stesse
matrici: il filtro cambia ingrandimento e spostamento, e fra uno scatto e
l'altro la tela si flette. Ogni banda si ricompone per conto suo, e i mosaici
delle bande si allineano poi fra loro con l'allineatore.

Convenzione, diversa da quella di transform_model.py e dichiarata nel JSON:
    coord_mosaico = M @ [x_tassello, y_tassello, 1]
cioe' la matrice porta il tassello nel mosaico. Per ricampionare serve
l'inversa (mosaico -> tassello), che e' esattamente un Transform
dell'allineatore con il mosaico nel ruolo della fixed.
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# Lettura delle immagini, block matching e trasformazioni sono quelli
# dell'allineatore, nella cartella accanto: si riusano invece di copiarli.
_MOTORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Allineatore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import image_io
import tiff_memmap
import block_matching as bm
from transform_model import Transform


ESTENSIONI = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')

PERCORSI = {
    'serpentina_righe': "serpentina per righe (una riga in un verso, la successiva "
                        "nel verso opposto)",
    'serpentina_colonne': "serpentina per colonne",
    'righe': "per righe, tutte nello stesso verso",
    'colonne': "per colonne, tutte nello stesso verso",
}
PARTENZE = {
    'alto_sinistra': "in alto a sinistra",
    'alto_destra': "in alto a destra",
    'basso_sinistra': "in basso a sinistra",
    'basso_destra': "in basso a destra",
}
MODELLI = ('traslazione', 'rigida', 'similarita', 'affine')
_N_PAR = {'traslazione': 2, 'rigida': 3, 'similarita': 4, 'affine': 6}

# Soglie non ancora tarate su tasselli veri: sono scelte prudenti, da rivedere
# con il primo mosaico reale.
NCC_MIN_GROSSOLANA = 0.20   # sotto, l'aggancio grossolano non e' creduto
MIN_CONF_BLOCCHI = 0.10     # come l'allineatore
PESO_DAL_PASSO = 0.05       # peso di un punto "dal passo del carrello"


# ---------------------------------------------------------------------------
# Percorso di ripresa
# ---------------------------------------------------------------------------

def ordina_naturale(percorsi):
    """Ordina i file come li conta una persona: scatto_2 prima di scatto_10.

    L'ordine alfabetico puro li invertirebbe ("1", "10", "11", "2"...), e con
    esso il percorso: tutti i tasselli dal decimo in poi finirebbero nella
    casella sbagliata senza che nulla lo segnali.
    """
    def chiave(p):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', os.path.basename(p))]
    return sorted(percorsi, key=chiave)


def elenca_immagini(cartella):
    """Le immagini di una cartella, in ordine naturale."""
    return ordina_naturale([os.path.join(cartella, n) for n in os.listdir(cartella)
                            if n.lower().endswith(ESTENSIONI)
                            and os.path.isfile(os.path.join(cartella, n))])


def posizioni_griglia(n_righe, n_colonne, percorso='serpentina_righe',
                      partenza='alto_sinistra'):
    """
    La casella di ogni scatto, nell'ordine di ripresa.

    Returns:
        lista di (riga, colonna), con riga 0 in alto e colonna 0 a sinistra
        DEL DIPINTO. L'elemento k e' la casella dello scatto k.
    """
    if percorso not in PERCORSI:
        raise ValueError(f"percorso '{percorso}' sconosciuto: {', '.join(PERCORSI)}")
    if partenza not in PARTENZE:
        raise ValueError(f"partenza '{partenza}' sconosciuta: {', '.join(PARTENZE)}")
    if n_righe < 1 or n_colonne < 1:
        raise ValueError("righe e colonne devono essere almeno 1")

    serpentina = percorso.startswith('serpentina')
    dal_basso = partenza.startswith('basso')
    da_destra = partenza.endswith('destra')
    out = []
    if percorso in ('serpentina_righe', 'righe'):
        righe = range(n_righe - 1, -1, -1) if dal_basso else range(n_righe)
        for k, r in enumerate(righe):
            colonne = list(range(n_colonne))
            # verso della riga: quello di partenza, invertito a righe alterne
            # nella serpentina
            if da_destra != (serpentina and k % 2 == 1):
                colonne.reverse()
            out += [(r, c) for c in colonne]
    else:
        colonne = range(n_colonne - 1, -1, -1) if da_destra else range(n_colonne)
        for k, c in enumerate(colonne):
            righe = list(range(n_righe))
            if dal_basso != (serpentina and k % 2 == 1):
                righe.reverse()
            out += [(r, c) for r in righe]
    return out


def schema_testuale(n_righe, n_colonne, posizioni):
    """La griglia con il numero di scatto in ogni casella, per il log: e' il
    modo piu' rapido di accorgersi di un percorso scelto male."""
    tab = np.zeros((n_righe, n_colonne), dtype=int)
    for k, (r, c) in enumerate(posizioni):
        tab[r, c] = k + 1
    w = max(4, len(str(n_righe * n_colonne)) + 2)
    righe = ["         " + "".join(f"{'col ' + str(c + 1):>{w + 3}}" for c in range(n_colonne))]
    for r in range(n_righe):
        righe.append(f"  riga {r + 1:<3}" + "".join(f"{tab[r, c]:>{w + 3}}"
                                                   for c in range(n_colonne)))
    return "\n".join(righe)


# ---------------------------------------------------------------------------
# Tasselli e lettura
# ---------------------------------------------------------------------------

class Tassello:
    """Un file e la sua casella. Le coordinate di un tassello sono i suoi pixel
    a piena risoluzione, con l'origine nell'angolo in alto a sinistra."""

    def __init__(self, indice, path, riga, colonna):
        self.indice = indice                 # ordine di ripresa, da 0
        self.path = path
        self.nome = os.path.basename(path)
        self.riga, self.colonna = riga, colonna
        self.info = image_io.info(path)
        self.H, self.W = self.info.height, self.info.width

    @property
    def centro(self):
        return np.array([self.W / 2.0, self.H / 2.0])

    def angoli(self):
        return np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], dtype=np.float64)

    def __repr__(self):
        return f"{self.indice + 1} r{self.riga + 1}c{self.colonna + 1} {self.nome}"


class _Archivio:
    """
    Tiene aperti gli ultimi tasselli letti a piena risoluzione.

    Per un TIFF non compresso l'array e' un memmap e non costa nulla. Per JPEG e
    TIFF compressi vuol dire l'immagine intera in RAM, quindi se ne tengono
    pochi: le coppie si visitano in ordine di griglia, e tre bastano a non
    rileggere lo stesso file due volte di fila.
    """

    def __init__(self, quanti=3):
        self.quanti = quanti
        self.aperti = OrderedDict()

    def __call__(self, tassello):
        p = tassello.path
        if p in self.aperti:
            self.aperti.move_to_end(p)
            return self.aperti[p]
        arr, _ = image_io.apri(p, verbose=False)
        self.aperti[p] = arr
        while len(self.aperti) > self.quanti:
            self.aperti.popitem(last=False)
        return arr


def _grigio(x):
    """Canale singolo, per la misura. Un eventuale canale alfa si ignora."""
    x = np.asarray(x)
    if x.ndim == 3:
        if x.shape[2] >= 3:
            return cv2.cvtColor(np.ascontiguousarray(x[:, :, :3]), cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(x[:, :, 0])
    return x


def _regione_ridotta(arr, x0, y0, x1, y1, fattore, banda=4096):
    """
    La regione [y0:y1, x0:x1] a piena risoluzione, in grigio e ridotta di un
    fattore INTERO per media d'area, letta a bande di righe.

    Con il fattore intero la relazione geometrica e' esatta, la stessa di
    image_io.leggi_ridotta e di transform_model:

        coord_piena = fattore * coord_ridotta + (fattore - 1) / 2 + origine

    Returns:
        (immagine, (x0, y0) origine effettiva dopo il ritaglio ai bordi)
    """
    H, W = arr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    f = max(1, int(fattore))
    w, h = (x1 - x0) // f * f, (y1 - y0) // f * f
    if w < f or h < f:
        return None, (x0, y0)
    parti = []
    passo = max(f, banda // f * f)
    for yy in range(y0, y0 + h, passo):
        blocco = _grigio(arr[yy:min(y0 + h, yy + passo), x0:x0 + w])
        if f > 1:
            blocco = cv2.resize(blocco, (w // f, blocco.shape[0] // f),
                                interpolation=cv2.INTER_AREA)
        parti.append(blocco)
    return np.concatenate(parti, axis=0), (x0, y0)


def _lato_fft(n, minimo=16):
    """
    Il lato pari piu' grande, non oltre n, che cv2.phaseCorrelate usa cosi'
    com'e'.

    phaseCorrelate allarga la finestra alla dimensione "ottimale" per la FFT
    (getOptimalDFTSize) e poi prende il centro a meta' di quella. Quando la
    dimensione ottimale e' DISPARI -- 44 diventa 45, 74 diventa 75, 402 diventa
    405 -- il centro cade su mezzo pixel e ogni misura esce spostata di circa
    +0.5 px in entrambi gli assi. Misurato: su finestre da 44, 74 e 402 px lo
    scarto medio dal valore vero e' di +0.3/+0.6 px, contro -0.1/+0.1 px per i
    lati vicini che non vengono allargati. Nel mosaico era diventato un errore
    vero: una fascia di sovrapposizione alta 180 px, ridotta di 2, dava blocchi
    da 44 px e una cucitura sbagliata di 1.4 px.

    Fra 16 e 1024 i lati pari colpiti sono 52; le potenze di due non lo sono mai.
    """
    n = int(n) // 2 * 2
    while n > minimo and cv2.getOptimalDFTSize(n) != n:
        n -= 2
    return n


def _sovrapposizione(forma_a, forma_b, d):
    """
    Il rettangolo comune, in coordinate di A, quando un dettaglio che in A sta
    in p in B sta in p + d. Con d intero.

    Returns:
        (x0, y0, x1, y1) in coordinate di A, oppure None
    """
    ha, wa = forma_a
    hb, wb = forma_b
    dx, dy = int(d[0]), int(d[1])
    x0, x1 = max(0, -dx), min(wa, wb - dx)
    y0, y1 = max(0, -dy), min(ha, hb - dy)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


# ---------------------------------------------------------------------------
# Misura di una coppia: aggancio grossolano
# ---------------------------------------------------------------------------

def _tukey(n, frazione):
    """Finestra piatta al centro e raccordata a zero solo su una fascia stretta.

    La finestra di Hann che usa l'allineatore qui sarebbe sbagliata: attenua i
    bordi, e in una coppia di tasselli la parte in comune sta proprio sul bordo
    di ciascuno dei due.
    """
    w = np.ones(n, np.float64)
    k = int(round(n * frazione / 2))
    if k > 0:
        rampa = 0.5 * (1 - np.cos(np.pi * (np.arange(k) + 0.5) / k))
        w[:k] = rampa
        w[n - k:] = rampa[::-1]
    return w


def _picchi_correlazione(a, b, n_picchi=6):
    """
    Correlazione di fase che restituisce PIU' picchi, non solo il piu' alto.

    cv2.phaseCorrelate da' un solo picco, ed e' giusto quando le due immagini
    sono quasi sovrapposte. Fra due tasselli no: la parte comune e' una frazione
    di ciascuno, il picco giusto puo' non essere il piu' alto, e comunque la
    correlazione e' circolare, quindi un picco in (px, py) puo' voler dire
    anche (px - larghezza, py - altezza). Si tengono allora i migliori candidati
    e si decide dopo, guardando le immagini (e' lo schema dello Stitching di
    Fiji).

    Convenzione di cv2.phaseCorrelate: il dettaglio che in `a` sta in p, in `b`
    sta in p + d.

    Returns:
        lista di (dx, dy, altezza) con dx in [0, larghezza), dy in [0, altezza)
    """
    h, w = a.shape
    fin = np.outer(_tukey(h, 0.1), _tukey(w, 0.1))
    fa = np.fft.rfft2((a - a.mean()) * fin)
    fb = np.fft.rfft2((b - b.mean()) * fin)
    R = np.conj(fa) * fb
    R /= np.abs(R) + 1e-12 * float(np.abs(R).max() + 1e-30)
    r = np.fft.irfft2(R, s=(h, w)).astype(np.float32)

    # massimi locali, sulla superficie circolare
    rp = np.pad(r, 2, mode='wrap')
    dil = cv2.dilate(rp, np.ones((5, 5), np.uint8))[2:-2, 2:-2]
    cand = np.argwhere(r >= dil)
    vals = r[cand[:, 0], cand[:, 1]]
    ordine = np.argsort(vals)[::-1][:n_picchi]

    def parabola(m1, c0, p1):
        den = m1 - 2 * c0 + p1
        return 0.0 if abs(den) < 1e-12 else float(0.5 * (m1 - p1) / den)

    out = []
    for (py, px), v in zip(cand[ordine], vals[ordine]):
        ox = parabola(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
        oy = parabola(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
        out.append((px + ox, py + oy, float(v)))
    return out


def _ncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else -1.0


def misura_grossolana(ga, gb, relazione, sov_min, sov_max, tolleranza_trasversale=0.25):
    """
    Dove sta B rispetto ad A, su anteprime ridotte.

    Si correla solo la fascia che PUO' essere in comune -- il lato destro di A
    col lato sinistro di B, o il basso di A con l'alto di B -- larga quanto la
    sovrapposizione massima ammessa: correlando i tasselli interi, la parte
    comune sarebbe un quinto del segnale e il resto solo rumore per il picco.

    Ogni picco candidato viene poi giudicato in due modi: deve essere
    compatibile col percorso (B a destra di A, o sotto, con una sovrapposizione
    nell'intervallo ammesso) e, fra quelli compatibili, vince quello con la
    correlazione normalizzata piu' alta nella zona comune.

    Args:
        ga, gb: modulo del gradiente delle anteprime di A e di B
        relazione: 'destra' (B e' a destra di A) oppure 'sotto'
        sov_min, sov_max: sovrapposizione ammessa, in frazione del lato

    Returns:
        dict con d (spostamento in px dell'anteprima: dettaglio in p in A ->
        p + d in B), ncc, picco e sovrapposizione; None se nessun candidato e'
        compatibile.
    """
    ha, wa = ga.shape
    hb, wb = gb.shape
    if relazione == 'destra':
        s = int(min(wa, wb, np.ceil(sov_max * wa) + 8))
        h = min(ha, hb)
        A, B = ga[:h, wa - s:], gb[:h, :s]
        base = (-(wa - s), 0)
    else:
        s = int(min(ha, hb, np.ceil(sov_max * ha) + 8))
        w = min(wa, wb)
        A, B = ga[ha - s:, :w], gb[:s, :w]
        base = (0, -(ha - s))
    if A.std() < 1e-6 or B.std() < 1e-6:
        return None

    hs, ws = A.shape
    migliore = None
    for px, py, altezza in _picchi_correlazione(A, B):
        for dxs in (px, px - ws):
            for dys in (py, py - hs):
                d = (dxs + base[0], dys + base[1])
                if relazione == 'destra':
                    sov, trasv = (wa + d[0]) / wa, abs(d[1]) / ha
                else:
                    sov, trasv = (ha + d[1]) / ha, abs(d[0]) / wa
                if not (sov_min <= sov <= sov_max) or trasv > tolleranza_trasversale:
                    continue
                di = (int(round(d[0])), int(round(d[1])))
                box = _sovrapposizione(ga.shape, gb.shape, di)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                c = _ncc(ga[y0:y1, x0:x1], gb[y0 + di[1]:y1 + di[1], x0 + di[0]:x1 + di[0]])
                if migliore is None or c > migliore['ncc']:
                    migliore = {'d': (float(d[0]), float(d[1])), 'ncc': c,
                                'picco': altezza, 'sovrapposizione': float(sov)}
    return migliore


# ---------------------------------------------------------------------------
# Misura di una coppia: corrispondenze fini
# ---------------------------------------------------------------------------

def misura_fine(arr_a, arr_b, D, fattore, blocco=256, min_conf=MIN_CONF_BLOCCHI):
    """
    Corrispondenze fini nella fascia comune, con il block matching
    dell'allineatore.

    Si legge SOLO la fascia comune dei due tasselli, a risoluzione di lavoro
    (fattore 2 di default, come la precisione normale dell'allineatore): per un
    TIFF non compresso e' una lettura da memmap di un quinto del file, e la
    precisione resta quella del sub-pixel a quella scala.

    Il lato dei blocchi si adatta alla fascia: ne devono stare almeno due di
    traverso, altrimenti sulle sovrapposizioni strette si misurerebbe una sola
    colonna di blocchi e la rotazione relativa resterebbe indeterminata.

    Args:
        arr_a, arr_b: i due tasselli a piena risoluzione (memmap o array)
        D: spostamento stimato a piena risoluzione (dettaglio in p in A ->
           p + D in B)

    Returns:
        dict con punti_a, punti_b (N, 2) a piena risoluzione, pesi (N,), e la
        descrizione della misura; 'n_inlier' == 0 se non si e' agganciato.
    """
    forma_a, forma_b = arr_a.shape[:2], arr_b.shape[:2]
    Di = (int(round(D[0])), int(round(D[1])))
    box = _sovrapposizione(forma_a, forma_b, Di)
    if box is None:
        return {'n_blocchi': 0, 'n_inlier': 0, 'motivo': 'nessuna zona comune'}
    x0, y0, x1, y1 = box
    sa, oa = _regione_ridotta(arr_a, x0, y0, x1, y1, fattore)
    sb, ob = _regione_ridotta(arr_b, x0 + Di[0], y0 + Di[1], x1 + Di[0], y1 + Di[1], fattore)
    if sa is None or sb is None:
        return {'n_blocchi': 0, 'n_inlier': 0, 'motivo': 'zona comune vuota'}

    corto = min(sa.shape[0], sa.shape[1], sb.shape[0], sb.shape[1])
    lato = _lato_fft(min(blocco, corto // 2))
    if lato < 32:
        return {'n_blocchi': 0, 'n_inlier': 0,
                'motivo': f'fascia comune troppo stretta ({corto * fattore} px)'}

    # Un dettaglio che nella fascia di A sta in s, in quella di B sta in
    # s + (oa + D - ob) / fattore: l'origine di B e' stata presa spostata di D
    # arrotondato, quindi questo scarto e' sotto il pixel e si arrotonda a zero.
    off = (int(round((oa[0] + D[0] - ob[0]) / fattore)),
           int(round((oa[1] + D[1] - ob[1]) / fattore)))
    ga, gb = bm.gradient_magnitude(sa), bm.gradient_magnitude(sb)
    pts = bm.match_blocks(ga, gb, block=lato, step=lato // 2, offset=off,
                          lati_ridotti=False, verbose=False)
    esito = {'n_blocchi': int(len(pts)), 'lato_blocco_px': int(lato * fattore)}
    try:
        _M, inl, res = bm.fit_affine_robust(pts, min_conf=min_conf, verbose=False)
    except RuntimeError:
        esito.update(n_inlier=0, motivo=f'solo {int((pts[:, 4] >= min_conf).sum()) if len(pts) else 0} '
                                        f'blocchi affidabili')
        return esito

    c = (fattore - 1) / 2.0
    pa = inl[:, :2] * fattore + c + np.array(oa, dtype=np.float64)
    pb = (inl[:, :2] + inl[:, 2:4]) * fattore + c + np.array(ob, dtype=np.float64)
    esito.update(
        punti_a=pa, punti_b=pb, pesi=np.clip(inl[:, 4], 1e-3, None),
        n_inlier=int(len(inl)),
        confidenza_mediana=float(np.median(inl[:, 4])),
        # quanto i blocchi della coppia concordano fra loro sotto un'affine: e'
        # coerenza interna della misura, non la precisione del mosaico
        coerenza_px=float(np.median(np.hypot(res[:, 0], res[:, 1])) * fattore),
        spostamento_px=[float(v) for v in np.median(pb - pa, axis=0)],
    )
    return esito


# ---------------------------------------------------------------------------
# Controllo del passo del carrello
# ---------------------------------------------------------------------------

def controlla_passo(coppie, lato_tassello, k_mad=6.0, frazione_min=0.02, min_coppie=4):
    """
    Scarta le coppie il cui spostamento non somiglia a quello delle altre.

    Il carrello (o chi sposta la camera) fa sempre lo stesso passo, con
    l'imprecisione della meccanica: gli spostamenti misurati fra vicini
    orizzontali si somigliano tutti, e cosi' quelli fra vicini verticali. Un
    aggancio sbagliato invece da' un valore qualunque. E' l'idea di MIST, che
    stima il passo dalle coppie stesse e ripara quelle che se ne discostano.

    Serve soprattutto dove la stima globale non puo' accorgersene da sola: in
    un mosaico di una riga sola non c'e' nessun anello da chiudere, e una coppia
    sbagliata non contraddice nessun'altra misura.

    La tolleranza non e' fissa: e' k_mad volte la dispersione robusta (MAD)
    degli spostamenti misurati, quindi un carrello impreciso si prende da solo
    la tolleranza che gli serve; con un minimo del 2% del lato, perche' un
    carrello molto preciso non renda sospetta ogni frazione di pixel.

    Returns:
        {relazione: passo mediano [dx, dy] in px, o None}
    """
    passi = {}
    for rel in ('destra', 'sotto'):
        mis = [cp for cp in coppie if cp['relazione'] == rel and cp['stato'] == 'misurata']
        if not mis:
            passi[rel] = None
            continue
        D = np.array([cp['fine']['spostamento_px'] for cp in mis])
        med = np.median(D, axis=0)
        if len(mis) >= min_coppie:
            mad = np.median(np.abs(D - med), axis=0) * 1.4826
            tol = np.maximum(k_mad * mad, frazione_min * lato_tassello)
            for cp, d in zip(mis, D):
                if np.any(np.abs(d - med) > tol):
                    cp['stato'] = 'incoerente col passo'
                    cp['nota'] = (f"spostamento ({d[0]:+.1f}, {d[1]:+.1f}) px contro un passo "
                                  f"mediano ({med[0]:+.1f}, {med[1]:+.1f}) px")
            buoni = [cp['fine']['spostamento_px'] for cp in mis if cp['stato'] == 'misurata']
            med = np.median(np.array(buoni), axis=0) if buoni else med
        passi[rel] = [float(med[0]), float(med[1])]
    return passi


def controlla_coerenza(coppie, volte_mediana=5.0, minimo_px=1.0, min_coppie=3):
    """
    Scarta le coppie i cui blocchi non concordano fra loro.

    Un aggancio vero da' blocchi che, dentro la fascia comune, descrivono tutti
    lo stesso spostamento: la loro coerenza interna (lo scarto mediano da
    un'affine) e' una frazione di pixel. Un aggancio sul rumore -- una fascia
    senza struttura, uno sfondo uniforme -- trova qualche blocco che supera la
    soglia di confidenza per caso, e quei blocchi puntano ognuno per conto suo.
    Sul test con un tassello senza struttura le coppie buone stavano a
    0.11-0.15 px e le due false a 4.0 e 10.8 px.

    La soglia e' relativa alle altre coppie dello stesso mosaico (5 volte la
    mediana, almeno 1 px) e non assoluta, perche' la coerenza normale dipende
    dai dati: dalla riduzione usata e, sulle riprese vere, dalla distorsione
    dell'obiettivo, che un'affine sulla fascia non segue del tutto.
    """
    mis = [cp for cp in coppie if cp['stato'] == 'misurata']
    if len(mis) < min_coppie:
        return
    c = np.array([cp['fine']['coerenza_px'] for cp in mis])
    soglia = max(minimo_px, volte_mediana * float(np.median(c)))
    for cp, v in zip(mis, c):
        if v > soglia:
            cp['stato'] = 'blocchi discordi'
            cp['nota'] = (f"coerenza interna {v:.2f} px contro una soglia di {soglia:.2f} px "
                          f"(mediana delle coppie {np.median(c):.2f} px)")


def _vincolo_dal_passo(a, b, D, peso=PESO_DAL_PASSO):
    """
    Un punto fittizio che dice solo "B sta a D da A", preso a meta' strada fra
    i centri dei due tasselli. Tiene al suo posto un tassello per cui la misura
    e' mancata, e pesa cosi' poco che dove ci sono misure vere non conta nulla.

    Un punto solo, e non una fila lungo la fascia, apposta: una fila
    imporrebbe anche l'orientamento relativo dei due tasselli, che nessuno ha
    misurato. Il passo tipico viene da ALTRE coppie e sulla coppia mancante
    sbaglia dell'imprecisione del carrello; con piu' punti, un tassello libero
    di ruotare e scalarsi si deformerebbe per inseguire quell'errore.
    """
    D = np.asarray(D, dtype=np.float64)
    pa = ((a.centro + (b.centro - D)) / 2.0).reshape(1, 2)
    return {'i': a.indice, 'j': b.indice, 'pa': pa, 'pb': pa + D,
            'w': np.array([peso]), 'dal_passo': True}


# ---------------------------------------------------------------------------
# Stima globale
# ---------------------------------------------------------------------------

def _applica(modello, par, d, L):
    """
    Posizione nel mosaico di punti del tassello, e derivate rispetto ai suoi
    parametri.

    Il tassello e' descritto attorno al suo centro: m = A (p - centro) + t, con
    d = p - centro. La parte lineare A e' scritta come identita' piu' una
    correzione espressa in PIXEL sul raggio L del tassello (quanto si sposta il
    bordo): cosi' tutte le incognite hanno la stessa unita' e lo stesso ordine di
    grandezza, e il sistema resta ben condizionato anche su tasselli da 15000 px.

    Returns:
        (m (N, 2), J (N, 2, k))
    """
    k = _N_PAR[modello]
    n = len(d)
    J = np.zeros((n, 2, k))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    u = d / L
    if modello == 'traslazione':
        m = d.copy()
    elif modello == 'rigida':
        th = par[2] / L
        c, s = np.cos(th), np.sin(th)
        m = np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])
        J[:, 0, 2] = (-s * d[:, 0] - c * d[:, 1]) / L
        J[:, 1, 2] = (c * d[:, 0] - s * d[:, 1]) / L
    elif modello == 'similarita':
        a, b = par[2], par[3]
        m = d + np.column_stack([a * u[:, 0] - b * u[:, 1], b * u[:, 0] + a * u[:, 1]])
        J[:, 0, 2], J[:, 1, 2] = u[:, 0], u[:, 1]
        J[:, 0, 3], J[:, 1, 3] = -u[:, 1], u[:, 0]
    else:
        e = par[2:6]
        m = d + np.column_stack([e[0] * u[:, 0] + e[1] * u[:, 1],
                                 e[2] * u[:, 0] + e[3] * u[:, 1]])
        J[:, 0, 2], J[:, 0, 3] = u[:, 0], u[:, 1]
        J[:, 1, 4], J[:, 1, 5] = u[:, 0], u[:, 1]
    return m + par[:2], J


def _parte_lineare(modello, par, L):
    if modello == 'traslazione':
        return np.eye(2)
    if modello == 'rigida':
        th = par[2] / L
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if modello == 'similarita':
        a, b = par[2] / L, par[3] / L
        return np.array([[1 + a, -b], [b, 1 + a]])
    e = par[2:6] / L
    return np.array([[1 + e[0], e[1]], [e[2], 1 + e[3]]])


def _matrice(modello, par, centro, L):
    """Matrice 2x3 tassello -> mosaico: m = A (p - centro) + t."""
    A = _parte_lineare(modello, par, L)
    return np.hstack([A, (par[:2] - A @ centro).reshape(2, 1)])


def _inversa(M):
    """Da tassello -> mosaico a mosaico -> tassello (e viceversa)."""
    A = M[:, :2]
    Ai = np.linalg.inv(A)
    return np.hstack([Ai, (-Ai @ M[:, 2]).reshape(2, 1)])


def stima_globale(tasselli, vincoli, modello, ancora, L, solo_traslazione=(), lam=1e-2,
                  n_iter=12):
    """
    Posiziona tutti i tasselli insieme, ai minimi quadrati pesati.

    Ogni corrispondenza (p nel tassello i, q nel tassello j) chiede che i due
    punti finiscano nello stesso posto del mosaico: G_i(p) = G_j(q). Il peso e'
    la confidenza del blocco, come nel fit affine dell'allineatore. Un tassello,
    l'ANCORA, resta fermo: fissa il sistema di riferimento, che altrimenti
    potrebbe scorrere tutto insieme (vedi _scegli_ancora).

    Il termine `lam` tira debolmente scala, rotazione e taglio verso l'identita'.
    Non cambia nulla dove le misure ci sono; impedisce che un tassello con
    poche misure prese su una sola fascia stretta si deformi a piacere nella
    direzione che quella fascia non vincola.

    I tasselli in `solo_traslazione` sono quelli senza nessuna coppia misurata:
    della loro rotazione e della loro scala non si sa niente, quindi restano
    fermi all'identita' e si spostano soltanto.

    Per traslazione, similarita' e affine il problema e' lineare e si risolve in
    un passo; per la rigida si itera (Gauss-Newton), perche' la rotazione entra
    con seno e coseno.

    Returns:
        parametri (n_tasselli, k)
    """
    bloccati = set(solo_traslazione)
    k = _N_PAR[modello]
    n = len(tasselli)
    centri = [t.centro for t in tasselli]
    par = np.zeros((n, k))
    col = {}
    for t in range(n):
        if t != ancora:
            col[t] = len(col) * k
    P = len(col) * k
    for _ in range(n_iter):
        N = np.zeros((P, P))
        g = np.zeros(P)
        for v in vincoli:
            i, j = v['i'], v['j']
            mi, Ji = _applica(modello, par[i], v['pa'] - centri[i], L)
            mj, Jj = _applica(modello, par[j], v['pb'] - centri[j], L)
            r = (mi - mj).reshape(-1)
            w = np.repeat(v['w'], 2)
            parti = []
            if i in col:
                parti.append((col[i], Ji.reshape(-1, k)))
            if j in col:
                parti.append((col[j], -Jj.reshape(-1, k)))
            for ca, Ja in parti:
                g[ca:ca + k] += Ja.T @ (w * r)
                for cb, Jb in parti:
                    N[ca:ca + k, cb:cb + k] += Ja.T @ (w[:, None] * Jb)
        for t, c0 in col.items():
            peso = 1e8 if t in bloccati else lam
            for q in range(2, k):
                N[c0 + q, c0 + q] += peso
                g[c0 + q] += peso * par[t, q]
        try:
            delta = np.linalg.solve(N, -g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(N, -g, rcond=None)[0]
        for t, c0 in col.items():
            par[t] += delta[c0:c0 + k]
        if modello != 'rigida' or np.abs(delta).max() < 1e-6:
            break
    return par


def _scarti(modello, par, vincoli, tasselli, L):
    """Per ogni vincolo, quanto i punti corrispondenti restano lontani nel
    mosaico (px). E' coerenza con la soluzione globale, non una verifica."""
    out = []
    for v in vincoli:
        mi, _ = _applica(modello, par[v['i']], v['pa'] - tasselli[v['i']].centro, L)
        mj, _ = _applica(modello, par[v['j']], v['pb'] - tasselli[v['j']].centro, L)
        out.append(np.hypot(mi[:, 0] - mj[:, 0], mi[:, 1] - mj[:, 1]))
    return out


def _sottoinsieme(v, idx):
    return dict(v, pa=v['pa'][idx], pb=v['pb'][idx], w=v['w'][idx])


def _scegli_ancora(tasselli, coppie, n_righe, n_colonne):
    """
    Il tassello che resta fermo e da' il sistema di riferimento al mosaico.

    Fra quelli con piu' coppie misurate, il piu' vicino al centro della griglia.
    Il centro, perche' dal centro il resto del mosaico si raggiunge con meno
    passaggi. Le coppie misurate prima di tutto, per un motivo che il test col
    tassello vuoto ha mostrato: un'ancora legata agli altri solo dal passo
    tipico lascia che tutto il resto del mosaico ruoti e si scali come un blocco
    unico per inseguire quei vincoli deboli (-1 grado e scala 0.98, su quel
    test).
    """
    def n_misurate(t):
        return sum(cp['stato'] == 'misurata' and t in (cp['a'], cp['b']) for cp in coppie)

    return min(tasselli, key=lambda t: (-n_misurate(t),
                                        (t.riga - (n_righe - 1) / 2) ** 2
                                        + (t.colonna - (n_colonne - 1) / 2) ** 2)).indice


def scegli_modello(tasselli, misurati, dal_passo, ancora, L, solo_traslazione=(),
                   semi=(0, 1, 2), guadagno_min=0.10, guadagno_min_px=0.05, verbose=True):
    """
    Sceglie il modello misurando, non a priori.

    Per ogni modello si stimano le posizioni con meta' delle corrispondenze di
    ogni coppia e si misura lo scarto sull'altra meta', mai vista. Si parte
    dalla traslazione e si passa a un modello piu' libero solo se riduce lo
    scarto fuori campione di almeno il 10% E di almeno 0.05 px rispetto a
    quello scelto fin li'.

    Perche' anche il minimo assoluto. Un grado di liberta' in piu' ha un costo
    che le cuciture non mostrano: sul test con tasselli rigidi, l'affine lascia
    le cuciture a 0.04 px come il rigido ma porta l'errore vero da 0.045 a
    0.47 px, perche' deforma il mosaico in modo coerente con tutte le
    sovrapposizioni. E la cross-validation, da sola, a volte la sceglieva: i
    punti tenuti da parte stanno nella stessa fascia di quelli usati, ne
    condividono gli errori sistematici, e un modello piu' libero li impara.
    Misurato su tasselli sintetici con quattro verita' diverse (traslazione,
    rigida, similarita', affine) e tre combinazioni di riduzione: quando la
    scelta era sbagliata il guadagno era di 0.016-0.026 px, quando era giusta
    di 0.19 px e oltre. Col solo 10% si sbagliavano 3 casi su 12, e su una
    corsa a parte i tasselli rigidi diventavano affini per 0.026 px; con il
    minimo di 0.05 px nessuno. E' una soglia tarata su dati sintetici, da
    ricontrollare sui primi mosaici veri.

    Provata e scartata: tenere da parte COPPIE intere invece di punti, che e'
    una prova piu' indipendente. Non sceglie mai l'affine, nemmeno quando e'
    vera (sceglieva la similarita', con 1.1 px di errore vero contro 0.14),
    perche' da una sola fascia stretta la deformazione del tassello si
    estrapola male fino alla cucitura opposta.

    Returns:
        (modello scelto, {modello: scarto fuori campione in px})
    """
    tabella = {}
    for mod in MODELLI:
        errori = []
        for s in semi:
            rng = np.random.default_rng(s)
            tr, te = [], []
            for v in misurati:
                perm = rng.permutation(len(v['w']))
                tr.append(_sottoinsieme(v, perm[::2]))
                te.append(_sottoinsieme(v, perm[1::2]))
            par = stima_globale(tasselli, tr + dal_passo, mod, ancora, L, solo_traslazione)
            errori.append(float(np.median(np.concatenate(_scarti(mod, par, te, tasselli, L)))))
        tabella[mod] = float(np.mean(errori))
    scelto = 'traslazione'
    for mod in MODELLI[1:]:
        if (tabella[mod] < tabella[scelto] * (1.0 - guadagno_min)
                and tabella[scelto] - tabella[mod] >= guadagno_min_px):
            scelto = mod
    if verbose:
        print("  [modello] scarto fuori campione (cross-validation):")
        for mod in MODELLI:
            print(f"      {mod:<12} {tabella[mod]:.3f} px" + ("   <--" if mod == scelto else ""))
    return scelto, tabella


# ---------------------------------------------------------------------------
# Verifica a piena risoluzione lungo le cuciture
# ---------------------------------------------------------------------------

def _dentro(T, tassello, x0, y0, lato, margine=2.0):
    """La finestra [x0, x0+lato) x [y0, y0+lato) del mosaico cade tutta nel
    tassello?"""
    q = np.array([[x0, y0], [x0 + lato, y0], [x0, y0 + lato], [x0 + lato, y0 + lato]],
                 dtype=np.float64)
    p = q @ T.M[:, :2].T + T.M[:, 2]
    return bool((p[:, 0] >= margine).all() and (p[:, 0] <= tassello.W - 1 - margine).all()
                and (p[:, 1] >= margine).all() and (p[:, 1] <= tassello.H - 1 - margine).all())


def verifica_cuciture(tasselli, coppie, matrici, archivio, lato_max=512, lato_min=96,
                      max_finestre=8, min_conf=0.05, verbose=True):
    """
    Di quanti pixel due tasselli vicini restano fuori registro, lungo ogni
    cucitura, a PIENA risoluzione.

    Si mettono fino a `max_finestre` finestre in fila lungo la fascia comune,
    si ricampionano ENTRAMBI i tasselli in quella finestra con la posizione
    finale e si misura lo spostamento che resta fra i due, con la stessa
    correlazione di fase sul gradiente della verifica dell'allineatore. Il
    numero e' direttamente quello che si vedrebbe in Photoshop sovrapponendo i
    due livelli: un bordo doppio di tanti pixel.

    E' indipendente dalla stima per risoluzione (piena contro ridotta) e per
    posizione delle finestre, non per principio: e' un controllo di coerenza,
    come quello dell'allineatore, non un confronto con una verita' esterna.

    La fila di finestre serve anche a un'altra cosa: se lo scarto cresce verso
    le estremita' della fascia, invece di essere uniforme, e' il segno della
    distorsione dell'obiettivo, che nessuna trasformazione affine puo' togliere.
    """
    trasf = {t.indice: Transform(_inversa(matrici[t.indice])) for t in tasselli}
    tutti = []
    for cp in coppie:
        a, b = cp['a'], cp['b']
        qa = (a.angoli() @ matrici[a.indice][:, :2].T + matrici[a.indice][:, 2]).astype(np.float32)
        qb = (b.angoli() @ matrici[b.indice][:, :2].T + matrici[b.indice][:, 2]).astype(np.float32)
        area, poly = cv2.intersectConvexConvex(qa, qb)
        if poly is None or area <= 0:
            cp['verifica'] = {'n_finestre': 0, 'motivo': 'i due tasselli non si sovrappongono'}
            continue
        poly = poly.reshape(-1, 2)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        corto, lungo = min(x1 - x0, y1 - y0), max(x1 - x0, y1 - y0)
        lato = _lato_fft(min(lato_max, 0.8 * corto))
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
            if not (_dentro(trasf[a.indice], a, X0, Y0, lato)
                    and _dentro(trasf[b.indice], b, X0, Y0, lato)):
                continue
            fa = _grigio(trasf[a.indice].warp_tile_from(arr_a, Y0, X0, lato, lato))
            fb = _grigio(trasf[b.indice].warp_tile_from(arr_b, Y0, X0, lato, lato))
            ga, gb = bm.gradient_magnitude(fa), bm.gradient_magnitude(fb)
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
                         residuo_max_px=float(tutti.max()),
                         quota_sopra_1px=float((tutti > 1.0).mean()),
                         quota_sopra_2px=float((tutti > 2.0).mean()))
    if verbose:
        if len(tutti):
            print(f"  residuo lungo le cuciture, su {len(tutti)} finestre a piena risoluzione: "
                  f"mediano {riepilogo['residuo_mediano_px']:.2f} px, "
                  f"max {riepilogo['residuo_max_px']:.2f} px; "
                  f"oltre 1 px {100 * riepilogo['quota_sopra_1px']:.0f}%, "
                  f"oltre 2 px {100 * riepilogo['quota_sopra_2px']:.0f}%")
        else:
            print("  nessuna finestra misurabile lungo le cuciture")
        for cp in coppie:
            v = cp.get('verifica', {})
            testo = (f"mediano {v['residuo_mediano_px']:.2f} px, max {v['residuo_max_px']:.2f} px "
                     f"su {v['n_affidabili']}/{v['n_finestre']} finestre"
                     if v.get('residuo_mediano_px') is not None else
                     v.get('motivo', 'nessuna finestra affidabile'))
            print(f"    {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} ({cp['relazione']:<6}) {testo}")
    return riepilogo


# ---------------------------------------------------------------------------
# Livelli per Photoshop
# ---------------------------------------------------------------------------

def _ifd(voci, inizio):
    """
    Costruisce una IFD TIFF (little endian) a partire da voci (tag, tipo, valori).
    Tipi: 3 SHORT, 4 LONG, 5 RATIONAL (coppie), 7 UNDEFINED (byte).
    I valori che non stanno nei 4 byte della voce vanno in coda, all'offset
    dichiarato, allineati alla parola come chiede lo standard.
    """
    n = len(voci)
    dopo = inizio + 2 + 12 * n + 4
    corpo, coda = bytearray(struct.pack('<H', n)), bytearray()
    for tag, tipo, valori in sorted(voci, key=lambda v: v[0]):
        if tipo == 7:
            grezzo, conta = bytes(valori), len(valori)
        elif tipo == 5:
            grezzo = b''.join(struct.pack('<LL', a, b) for a, b in valori)
            conta = len(valori)
        else:
            fmt = {3: 'H', 4: 'L'}[tipo]
            grezzo, conta = struct.pack('<' + fmt * len(valori), *valori), len(valori)
        if len(grezzo) <= 4:
            corpo += struct.pack('<HHL', tag, tipo, conta) + grezzo.ljust(4, b'\0')
        else:
            if len(coda) % 2:
                coda += b'\0'
            corpo += struct.pack('<HHLL', tag, tipo, conta, dopo + len(coda))
            coda += grezzo
    corpo += struct.pack('<L', 0)
    return bytes(corpo + coda)


def _crea_tiff_con_alfa(path, altezza, larghezza, colori, dtype, icc=None, dpi=None):
    """
    TIFF non compresso con un canale di trasparenza, da riempire a bande.

    E' lo stesso formato di tiff_memmap.create_tiff_memmap (un'unica striscia,
    niente compressione) con in piu' tre cose che servono a un livello:
      - il canale alfa, dichiarato come trasparenza (ExtraSamples = 2, alfa non
        premoltiplicato: i colori restano quelli veri anche sul bordo);
      - il profilo ICC dell'originale, cosi' Photoshop interpreta i colori come
        nel file di partenza e non li converte;
      - la risoluzione dell'originale.
    Non tocca tiff_memmap: la scrittura dei pixel usa il suo ScrittoreRighe
    cosi' com'e'.
    """
    dtype = np.dtype(dtype)
    campioni = colori + 1
    n_byte = altezza * larghezza * campioni * dtype.itemsize
    if n_byte >= 4 * 1024 ** 3:
        raise ValueError(f"{os.path.basename(path)}: livello da {n_byte / 1024 ** 3:.1f} GB, "
                         "oltre i 4 GB di un TIFF classico (servirebbe BigTIFF).")
    ris = 300.0
    if dpi:
        try:
            ris = float(dpi[0]) or 300.0
        except (TypeError, ValueError, IndexError):
            pass
    razionale = (int(round(ris * 100)), 100)
    voci = [(256, 4, [larghezza]), (257, 4, [altezza]),
            (258, 3, [dtype.itemsize * 8] * campioni), (259, 3, [1]),
            (262, 3, [2 if colori == 3 else 1]), (273, 4, [0]), (277, 3, [campioni]),
            (278, 4, [altezza]), (279, 4, [n_byte]), (282, 5, [razionale]),
            (283, 5, [razionale]), (284, 3, [1]), (296, 3, [2]), (338, 3, [2])]
    if icc:
        voci.append((34675, 7, bytes(icc)))
    ifd = _ifd(voci, 8)
    inizio_dati = (8 + len(ifd) + 7) // 8 * 8
    voci[5] = (273, 4, [inizio_dati])
    ifd = _ifd(voci, 8)          # stessa lunghezza: cambia solo un valore
    fh = open(path, 'wb')
    fh.write(b'II' + struct.pack('<HL', 42, 8) + ifd + b'\0' * (inizio_dati - 8 - len(ifd)))
    return tiff_memmap.ScrittoreRighe(fh, altezza, larghezza, campioni, dtype)


def _profilo_e_dpi(path):
    try:
        with Image.open(path) as im:
            return im.info.get('icc_profile'), im.info.get('dpi')
    except Exception:
        return None, None


def scrivi_livello(tassello, matrice, arr, path_out, forma_mosaico, banda=512):
    """
    Il tassello ricampionato nel sistema del mosaico, ritagliato al rettangolo
    che occupa, con la trasparenza fuori dal suo contorno.

    Si scrive solo il rettangolo del tassello, non l'intero mosaico: un livello
    grande quanto il mosaico peserebbe, a 16 bit e con l'alfa, diversi GB anche
    se quasi tutto trasparente. La posizione del rettangolo finisce nel JSON e
    nello script per Photoshop.

    Un solo ricampionamento (cv2.remap bilineare, come l'allineatore). Il
    contorno e' netto: un pixel e' opaco se TUTTO il suo intorno di
    interpolazione cade nel tassello, altrimenti e' trasparente. Si perde al piu'
    un pixel sul bordo, che sta comunque dentro la fascia di sovrapposizione, e
    in cambio non esistono pixel semitrasparenti dal colore incerto.

    Returns:
        (x0, y0, x1, y1) del rettangolo nel mosaico
    """
    Hm, Wm = forma_mosaico
    q = tassello.angoli() @ matrice[:, :2].T + matrice[:, 2]
    x0, y0 = max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min())))
    x1, y1 = min(Wm, int(np.ceil(q[:, 0].max()))), min(Hm, int(np.ceil(q[:, 1].max())))
    w, h = x1 - x0, y1 - y0
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{tassello.nome}: profondita' {arr.dtype} non supportata "
                         "(servono 8 o 16 bit interi)")
    colori = 3 if (arr.ndim == 3 and arr.shape[2] >= 3) else 1
    alfa_max = np.iinfo(arr.dtype).max
    T = Transform(_inversa(matrice))
    icc, dpi = _profilo_e_dpi(tassello.path)

    temp = path_out + '.parziale'
    scrittore = _crea_tiff_con_alfa(temp, h, w, colori, arr.dtype, icc, dpi)
    try:
        for yb in range(y0, y1, banda):
            bh = min(banda, y1 - yb)
            mx, my = T.maps_for_tile(yb, x0, bh, w)
            dentro = (mx >= 0) & (mx <= tassello.W - 1) & (my >= 0) & (my <= tassello.H - 1)
            out = np.zeros((bh, w, colori + 1), dtype=arr.dtype)
            if dentro.any():
                sx0 = max(0, int(np.floor(mx[dentro].min())) - 2)
                sx1 = min(tassello.W, int(np.ceil(mx[dentro].max())) + 3)
                sy0 = max(0, int(np.floor(my[dentro].min())) - 2)
                sy1 = min(tassello.H, int(np.ceil(my[dentro].max())) + 3)
                src = np.asarray(arr[sy0:sy1, sx0:sx1])
                if src.ndim == 3:
                    src = src[:, :, :3] if colori == 3 else src[:, :, 0]
                pix = cv2.remap(np.ascontiguousarray(src), mx - sx0, my - sy0,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
                if pix.ndim == 2:
                    pix = pix[:, :, None]
                pix[~dentro] = 0
                out[:, :, :colori] = pix
                out[:, :, colori] = np.where(dentro, alfa_max, 0)
            scrittore.scrivi(out)
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
    os.replace(temp, path_out)
    return x0, y0, x1, y1


_JSX = r"""// Mosaico: apre i livelli gia' posizionati in un unico documento.
// Generato da mosaico.py il %(quando)s.
//
// In Photoshop: File > Script > Sfoglia... e scegli questo file (oppure
// trascinalo sulla finestra di Photoshop). I file dei livelli devono restare
// nella cartella "livelli" accanto a questo script.
//
// Il documento nuovo prende profilo colore, modo e profondita' dal primo
// livello, cosi' nessun livello viene convertito entrando.
// Un mosaico oltre i 30000 px di lato si salva come PSB (formato documento
// grande), non come PSD.
#target photoshop
(function () {
    var W = %(W)d, H = %(H)d;
    var livelli = %(livelli)s;
    var cartella = File($.fileName).parent;
    var unita = app.preferences.rulerUnits;
    var dialoghi = app.displayDialogs;
    app.preferences.rulerUnits = Units.PIXELS;
    app.displayDialogs = DialogModes.NO;
    var mosaico = null, vuoto = null;

    function px(u) { return u.as("px"); }

    // Se Photoshop ha letto la trasparenza del TIFF come un canale alfa a
    // parte, la si trasforma in trasparenza del livello; se l'ha gia' letta
    // come trasparenza, non c'e' nulla da fare.
    function trasparenzaDaAlfa(d, strato) {
        var alfa = null;
        for (var k = 0; k < d.channels.length; k++) {
            if (d.channels[k].kind != ChannelType.COMPONENT) { alfa = d.channels[k]; break; }
        }
        if (alfa === null) return;
        d.activeLayer = strato;
        d.selection.load(alfa, SelectionType.REPLACE, true);
        try { d.selection.clear(); } catch (e) { /* tassello tutto opaco */ }
        d.selection.deselect();
        alfa.remove();
    }

    function creaDocumento(modello) {
        var modo = (modello.mode == DocumentMode.GRAYSCALE) ? NewDocumentMode.GRAYSCALE
                                                            : NewDocumentMode.RGB;
        var profilo = null;
        try { profilo = modello.colorProfileName; } catch (e) { profilo = null; }
        var d;
        if (profilo) {
            d = app.documents.add(W, H, modello.resolution, "Mosaico", modo,
                                  DocumentFill.TRANSPARENT, 1.0, modello.bitsPerChannel, profilo);
        } else {
            d = app.documents.add(W, H, modello.resolution, "Mosaico", modo,
                                  DocumentFill.TRANSPARENT, 1.0, modello.bitsPerChannel);
        }
        return d;
    }

    try {
        for (var i = 0; i < livelli.length; i++) {
            var L = livelli[i];
            var f = new File(cartella.fsName + "/" + L.file);
            if (!f.exists) { alert("Manca il livello:\n" + f.fsName); return; }
            var src = app.open(f);
            var strato = src.activeLayer;
            if (strato.isBackgroundLayer) strato.isBackgroundLayer = false;
            trasparenzaDaAlfa(src, strato);
            if (mosaico === null) {
                mosaico = creaDocumento(src);
                vuoto = mosaico.activeLayer;
                app.activeDocument = src;
            }
            // La posizione si fissa confrontando i contorni prima e dopo la
            // copia, cosi' non dipende da dove Photoshop decide di incollarla.
            var prima = strato.bounds;
            var copia = strato.duplicate(mosaico, ElementPlacement.PLACEATBEGINNING);
            src.close(SaveOptions.DONOTSAVECHANGES);
            app.activeDocument = mosaico;
            var dopo = copia.bounds;
            copia.translate(L.x + px(prima[0]) - px(dopo[0]), L.y + px(prima[1]) - px(dopo[1]));
            copia.name = L.nome;
        }
        if (vuoto !== null && mosaico.artLayers.length > 1) vuoto.remove();
    } finally {
        app.preferences.rulerUnits = unita;
        app.displayDialogs = dialoghi;
    }
})();
"""


def scrivi_jsx(path, forma_mosaico, livelli):
    """Lo script che apre i livelli in Photoshop alle coordinate giuste."""
    Hm, Wm = forma_mosaico
    voci = ",\n        ".join(
        "{file: %s, x: %d, y: %d, nome: %s}" % (json.dumps(l['file']), l['x'], l['y'],
                                               json.dumps(l['nome'], ensure_ascii=False))
        for l in livelli)
    testo = _JSX % {'quando': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'W': Wm, 'H': Hm, 'livelli': "[\n        " + voci + "\n    ]"}
    # ExtendScript legge l'UTF-8 se c'e' il BOM: servono i nomi dei file con accenti
    with open(path, 'w', encoding='utf-8-sig', newline='\r\n') as fh:
        fh.write(testo)


# ---------------------------------------------------------------------------
# Anteprima
# ---------------------------------------------------------------------------

def salva_anteprima(tasselli, matrici, forma_mosaico, path, coppie=(), lato=2400):
    """
    Il mosaico in piccolo, con il contorno e il numero di ogni tassello, il
    percorso di ripresa e l'esito della verifica lungo le cuciture (verde sotto
    1 px, arancio fino a 2, rosso oltre). Serve a controllare in un colpo
    d'occhio che il percorso sia quello giusto: un tassello nella casella
    sbagliata si vede subito.
    """
    Hm, Wm = forma_mosaico
    s = min(1.0, lato / float(max(Hm, Wm)))
    Wp, Hp = max(1, int(np.ceil(Wm * s))), max(1, int(np.ceil(Hm * s)))
    tela = np.full((Hp, Wp, 3), 40, np.uint8)
    for t in tasselli:
        M = matrici[t.indice]
        mini, passo = image_io.anteprima_veloce(t.path, int(max(t.W, t.H) * s * 1.5) + 1)
        mini = image_io.a_8bit(mini)
        if mini.ndim == 2:
            mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2RGB)
        elif mini.shape[2] > 3:
            mini = np.ascontiguousarray(mini[:, :, :3])
        Mp = np.hstack([s * passo * M[:, :2], (s * M[:, 2]).reshape(2, 1)])
        img = cv2.warpAffine(mini, Mp, (Wp, Hp), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(np.full(mini.shape[:2], 255, np.uint8), Mp, (Wp, Hp),
                             flags=cv2.INTER_NEAREST)
        tela[msk > 0] = img[msk > 0]

    lato_t = max(t.W for t in tasselli) * s
    fs = float(np.clip(lato_t / 220.0, 0.35, 2.5))
    spess = max(1, int(round(fs * 1.5)))
    centri = []
    for t in sorted(tasselli, key=lambda t: t.indice):
        M = matrici[t.indice]
        q = (t.angoli() @ M[:, :2].T + M[:, 2]) * s
        cv2.polylines(tela, [np.round(q).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        centri.append((t.centro @ M[:, :2].T + M[:, 2]) * s)
    for a, b in zip(centri[:-1], centri[1:]):
        lung = max(1.0, float(np.hypot(*(b - a))))
        cv2.arrowedLine(tela, tuple(int(v) for v in a), tuple(int(v) for v in b),
                        (90, 170, 255), spess, cv2.LINE_AA, tipLength=min(0.3, 8.0 * spess / lung))
    for cp in coppie:
        for x, y, dx, dy, conf in cp.get('verifica', {}).get('campioni', []):
            r = float(np.hypot(dx, dy))
            colore = ((140, 140, 140) if conf < 0.05 else (60, 200, 60) if r < 1.0
                      else (255, 160, 0) if r < 2.0 else (230, 40, 40))
            cv2.circle(tela, (int(x * s), int(y * s)), max(2, spess + 1), colore, -1, cv2.LINE_AA)
    for t, c in zip(sorted(tasselli, key=lambda t: t.indice), centri):
        testo = str(t.indice + 1)
        (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, fs, spess)
        org = (int(c[0] - tw / 2), int(c[1] + th / 2))
        cv2.putText(tela, testo, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), spess + 3, cv2.LINE_AA)
        cv2.putText(tela, testo, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), spess, cv2.LINE_AA)
    image_io.scrivi_immagine(path, cv2.cvtColor(tela, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Tutto insieme
# ---------------------------------------------------------------------------

def _intervallo_sovrapposizione(sovrapposizione):
    """Da una sovrapposizione attesa in percento (o None) all'intervallo
    ammesso, in frazione del lato."""
    if sovrapposizione is None or sovrapposizione <= 0:
        return 0.03, 0.60
    f = float(sovrapposizione) / 100.0
    return max(0.02, f - 0.15), min(0.90, f + 0.15)


def _descrivi_posa(M):
    A = M[:, :2]
    return {'rotazione_gradi': float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
            'scala_x': float(np.hypot(A[0, 0], A[1, 0])),
            'scala_y': float(np.hypot(A[0, 1], A[1, 1]))}


def componi_mosaico(immagini, n_righe, n_colonne, cartella_out,
                    percorso='serpentina_righe', partenza='alto_sinistra',
                    modello='auto', sovrapposizione=None, fattore=2,
                    lato_grossolano=1500, scrivi_livelli=True, verbose=True,
                    progress=None):
    """
    Dal gruppo di tasselli ai livelli posizionati.

    Args:
        immagini: i file, NELL'ORDINE DI RIPRESA
        percorso, partenza: vedi posizioni_griglia
        modello: 'auto' oppure uno di MODELLI
        sovrapposizione: sovrapposizione attesa fra vicini, in percento; None se
            non la si conosce (si cerca fra il 3 e il 60%)
        fattore: riduzione usata per la misura fine (2 = come la precisione
            normale dell'allineatore)
        lato_grossolano: lato lungo delle anteprime per l'aggancio grossolano
        scrivi_livelli: False = solo stima e verifica, senza scrivere i livelli

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

    n = len(immagini)
    if n != n_righe * n_colonne:
        raise ValueError(f"{n} immagini per una griglia di {n_righe} righe e {n_colonne} "
                         f"colonne: ne servono esattamente {n_righe * n_colonne}.")
    if modello != 'auto' and modello not in MODELLI:
        raise ValueError(f"modello '{modello}' sconosciuto: auto, {', '.join(MODELLI)}")
    os.makedirs(cartella_out, exist_ok=True)
    posizioni = posizioni_griglia(n_righe, n_colonne, percorso, partenza)
    tasselli = [Tassello(k, p, r, c) for k, (p, (r, c)) in enumerate(zip(immagini, posizioni))]
    griglia = {(t.riga, t.colonna): t for t in tasselli}
    sov_min, sov_max = _intervallo_sovrapposizione(sovrapposizione)
    lato_max = max(max(t.W, t.H) for t in tasselli)
    L = lato_max / 2.0
    archivio = _Archivio()

    log("=" * 72)
    log(f"MOSAICO  {n_righe} righe x {n_colonne} colonne, {PERCORSI[percorso]}, "
        f"partenza {PARTENZE[partenza]}")
    log("=" * 72)
    log(schema_testuale(n_righe, n_colonne, posizioni))
    for t in tasselli:
        log(f"  {t.indice + 1:>3}  riga {t.riga + 1}, colonna {t.colonna + 1}   {t.info!r}")
    if len({(t.H, t.W) for t in tasselli}) > 1:
        log("  [attenzione] i tasselli non hanno tutti le stesse dimensioni")
    log(f"  sovrapposizione cercata fra {100 * sov_min:.0f}% e {100 * sov_max:.0f}% del lato")

    # ---- 1. anteprime per l'aggancio grossolano ---------------------------
    avanza("anteprime", 0.0)
    f_g = max(1, int(lato_max // lato_grossolano))
    log(f"\n[1/5] anteprime ridotte di {f_g} per l'aggancio grossolano")
    anteprime = {}
    for k, t in enumerate(tasselli):
        anteprime[t.indice], _ = _regione_ridotta(archivio(t), 0, 0, t.W, t.H, f_g)
        avanza("anteprime", 0.15 * (k + 1) / n)

    # ---- 2. coppie adiacenti: aggancio grossolano -------------------------
    coppie = []
    for r in range(n_righe):
        for c in range(n_colonne):
            if c + 1 < n_colonne:
                coppie.append({'a': griglia[(r, c)], 'b': griglia[(r, c + 1)], 'relazione': 'destra'})
            if r + 1 < n_righe:
                coppie.append({'a': griglia[(r, c)], 'b': griglia[(r + 1, c)], 'relazione': 'sotto'})
    log(f"\n[2/5] aggancio grossolano di {len(coppie)} coppie di vicini")
    for k, cp in enumerate(coppie):
        ga = bm.gradient_magnitude(anteprime[cp['a'].indice])
        gb = bm.gradient_magnitude(anteprime[cp['b'].indice])
        g = misura_grossolana(ga, gb, cp['relazione'], sov_min, sov_max)
        cp['grossolana'] = g
        if g is not None and g['ncc'] >= NCC_MIN_GROSSOLANA:
            cp['D0'] = (g['d'][0] * f_g, g['d'][1] * f_g)
            cp['partenza_da'] = 'aggancio grossolano'
        else:
            cp['D0'] = None
        avanza("aggancio grossolano", 0.15 + 0.10 * (k + 1) / len(coppie))
    # Dove l'aggancio grossolano non c'e', si parte dal passo tipico dei vicini
    # nella stessa direzione: il carrello fa sempre lo stesso passo.
    for rel in ('destra', 'sotto'):
        buone = [cp['D0'] for cp in coppie if cp['relazione'] == rel and cp['D0'] is not None]
        for cp in coppie:
            if cp['relazione'] != rel or cp['D0'] is not None:
                continue
            if buone:
                cp['D0'] = tuple(np.median(np.array(buone), axis=0))
                cp['partenza_da'] = 'passo tipico dei vicini'
            elif sovrapposizione:
                a = cp['a']
                ov = sovrapposizione / 100.0
                cp['D0'] = (-(a.W * (1 - ov)), 0.0) if rel == 'destra' else (0.0, -(a.H * (1 - ov)))
                cp['partenza_da'] = 'sovrapposizione dichiarata'
    for cp in coppie:
        g = cp['grossolana']
        stato = (f"sovrapposizione {100 * g['sovrapposizione']:.1f}%, correlazione {g['ncc']:.2f}"
                 if g else "nessun candidato compatibile col percorso")
        if cp['D0'] is None:
            raise RuntimeError(
                f"Nessuna coppia agganciata in direzione '{cp['relazione']}', e nessuna "
                f"sovrapposizione dichiarata da cui partire.\nControlla righe, colonne e "
                f"percorso (l'anteprima non c'e' ancora: guarda lo schema qui sopra), oppure "
                f"indica la sovrapposizione attesa.")
        nota = "" if cp['partenza_da'] == 'aggancio grossolano' else f"  -> parto dal {cp['partenza_da']}"
        log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} ({cp['relazione']:<6}) {stato}{nota}")

    # ---- 3. corrispondenze fini -------------------------------------------
    log(f"\n[3/5] corrispondenze fini nelle fasce comuni (riduzione {fattore})")
    for k, cp in enumerate(coppie):
        fi = misura_fine(archivio(cp['a']), archivio(cp['b']), cp['D0'], fattore)
        cp['fine'] = fi
        cp['stato'] = 'misurata' if fi.get('n_inlier', 0) >= 6 else 'non misurata'
        if cp['stato'] == 'misurata':
            d = fi['spostamento_px']
            log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} ({cp['relazione']:<6}) "
                f"{fi['n_inlier']}/{fi['n_blocchi']} blocchi coerenti, spostamento "
                f"({d[0]:+.1f}, {d[1]:+.1f}) px, coerenza interna {fi['coerenza_px']:.2f} px")
        else:
            log(f"  {cp['a'].indice + 1:>3} - {cp['b'].indice + 1:<3} ({cp['relazione']:<6}) "
                f"NON MISURATA: {fi.get('motivo', '')}")
        avanza("corrispondenze fini", 0.25 + 0.30 * (k + 1) / len(coppie))

    controlla_coerenza(coppie)
    for cp in coppie:
        if cp['stato'] == 'blocchi discordi':
            log(f"  [coerenza] {cp['a'].indice + 1} - {cp['b'].indice + 1} esclusa: {cp['nota']}")
    passi = controlla_passo(coppie, lato_max)
    for cp in coppie:
        if cp['stato'] == 'incoerente col passo':
            log(f"  [passo] {cp['a'].indice + 1} - {cp['b'].indice + 1} esclusa: {cp['nota']}")

    # ---- 4. stima globale -------------------------------------------------
    avanza("stima globale", 0.55)
    log("\n[4/5] stima globale")

    def senza_misure():
        return [t for t in tasselli
                if not any(cp['stato'] == 'misurata' and t in (cp['a'], cp['b'])
                           for cp in coppie)]

    def vincolo_misurato(cp):
        fi = cp['fine']
        return {'i': cp['a'].indice, 'j': cp['b'].indice, 'pa': fi['punti_a'],
                'pb': fi['punti_b'], 'w': fi['pesi'], 'coppia': cp}

    def vincolo_passo(cp):
        D = passi.get(cp['relazione']) or cp['D0']
        return _vincolo_dal_passo(cp['a'], cp['b'], D)

    # Le coppie che non tornano con le altre si escludono una alla volta, la
    # peggiore per prima. Per riconoscerle si usa il modello piu' libero: con
    # uno troppo rigido anche le coppie giuste lascerebbero scarti grandi (una
    # rotazione vera non rappresentata), e si rischierebbe di togliere quelle.
    esclusioni = 0
    while True:
        misurati = [vincolo_misurato(cp) for cp in coppie if cp['stato'] == 'misurata']
        dal_passo = [vincolo_passo(cp) for cp in coppie if cp['stato'] != 'misurata']
        if not misurati:
            break
        ancora = _scegli_ancora(tasselli, coppie, n_righe, n_colonne)
        solo_trasl = [t.indice for t in senza_misure()]
        par = stima_globale(tasselli, misurati + dal_passo, 'affine', ancora, L, solo_trasl)
        sc = [float(np.median(s)) for s in _scarti('affine', par, misurati, tasselli, L)]
        peggiore = int(np.argmax(sc))
        soglia = max(3.0, 5.0 * float(np.median(sc)))
        if sc[peggiore] <= soglia or esclusioni >= max(1, len(coppie) // 3):
            break
        cp = misurati[peggiore]['coppia']
        cp['stato'] = 'incoerente con le altre'
        cp['nota'] = (f"scarto {sc[peggiore]:.1f} px dalla soluzione globale, contro una "
                      f"soglia di {soglia:.1f} px")
        log(f"  [anelli] {cp['a'].indice + 1} - {cp['b'].indice + 1} esclusa: {cp['nota']}")
        esclusioni += 1

    ancora = _scegli_ancora(tasselli, coppie, n_righe, n_colonne)
    isolati = senza_misure()
    solo_trasl = [t.indice for t in isolati]
    if modello == 'auto':
        if misurati:
            modello_usato, tabella_cv = scegli_modello(tasselli, misurati, dal_passo, ancora, L,
                                                       solo_trasl, verbose=verbose)
        else:
            modello_usato, tabella_cv = 'traslazione', {}
    else:
        modello_usato, tabella_cv = modello, {}
    par = stima_globale(tasselli, misurati + dal_passo, modello_usato, ancora, L, solo_trasl)
    scarti = _scarti(modello_usato, par, misurati, tasselli, L)
    for v, s in zip(misurati, scarti):
        v['coppia']['scarto_globale_px'] = float(np.median(s))
    matrici = {t.indice: _matrice(modello_usato, par[t.indice], t.centro, L) for t in tasselli}

    # origine del mosaico nell'angolo in alto a sinistra di tutto cio' che c'e'
    q = np.concatenate([t.angoli() @ matrici[t.indice][:, :2].T + matrici[t.indice][:, 2]
                        for t in tasselli])
    ox, oy = np.floor(q.min(axis=0))
    for M in matrici.values():
        M[:, 2] -= (ox, oy)
    Wm = int(np.ceil(q[:, 0].max() - ox))
    Hm = int(np.ceil(q[:, 1].max() - oy))
    pose = [_descrivi_posa(matrici[t.indice]) for t in tasselli]
    rot = np.array([p['rotazione_gradi'] for p in pose])
    sca = np.array([[p['scala_x'], p['scala_y']] for p in pose])
    n_mis = sum(cp['stato'] == 'misurata' for cp in coppie)
    log(f"  modello {modello_usato}, ancora il tassello {ancora + 1}; "
        f"{n_mis}/{len(coppie)} coppie misurate")
    if scarti:
        tutti = np.concatenate(scarti)
        log(f"  scarto delle corrispondenze dalla soluzione globale: mediano "
            f"{np.median(tutti):.2f} px, p95 {np.percentile(tutti, 95):.2f} px")
    log(f"  rotazioni fra {rot.min():+.3f} e {rot.max():+.3f} gradi, "
        f"scale fra {sca.min():.5f} e {sca.max():.5f}")
    if modello_usato in ('similarita', 'affine') and np.abs(sca - 1).max() > 0.002:
        log("  [attenzione] scale diverse da 1 di oltre lo 0.2%: se il carrello non cambia "
            "distanza dal dipinto, e' probabile che stiano compensando la distorsione "
            "dell'obiettivo (vedi il commento in testa al file)")
    log(f"  mosaico {Wm} x {Hm} px")
    for t in isolati:
        log(f"  [attenzione] il tassello {t.indice + 1} non ha nessuna coppia misurata: "
            f"e' posizionato solo dal passo tipico del carrello, senza rotazione ne' scala; "
            f"controllalo a occhio")

    # ---- 5. verifica ------------------------------------------------------
    avanza("verifica delle cuciture", 0.60)
    log("\n[5/5] verifica lungo le cuciture, a piena risoluzione")
    verifica = verifica_cuciture(tasselli, coppie, matrici, archivio, verbose=verbose)

    # ---- uscite -----------------------------------------------------------
    livelli = []
    if scrivi_livelli:
        cartella_livelli = os.path.join(cartella_out, 'livelli')
        os.makedirs(cartella_livelli, exist_ok=True)
        log(f"\nscrittura dei livelli in {cartella_livelli}")
        for k, t in enumerate(tasselli):
            nome = f"{t.indice + 1:02d}_r{t.riga + 1}c{t.colonna + 1}_" \
                   f"{os.path.splitext(t.nome)[0]}.tif"
            path_liv = os.path.join(cartella_livelli, nome)
            x0, y0, x1, y1 = scrivi_livello(t, matrici[t.indice], archivio(t), path_liv, (Hm, Wm))
            livelli.append({'file': 'livelli/' + nome, 'x': x0, 'y': y0,
                            'nome': f"{t.indice + 1:02d} r{t.riga + 1}c{t.colonna + 1} {t.nome}",
                            'rettangolo': [x0, y0, x1, y1]})
            log(f"  {nome}  {x1 - x0} x {y1 - y0} px in ({x0}, {y0})")
            avanza("scrittura dei livelli", 0.70 + 0.28 * (k + 1) / n)
        scrivi_jsx(os.path.join(cartella_out, 'apri_in_photoshop.jsx'), (Hm, Wm), livelli)

    path_anteprima = os.path.join(cartella_out, 'anteprima_mosaico.png')
    salva_anteprima(tasselli, matrici, (Hm, Wm), path_anteprima, coppie)

    report = {
        'programma': 'mosaico.py',
        'data': datetime.now().isoformat(timespec='seconds'),
        'convenzione': 'coord_mosaico = M @ [x_tassello, y_tassello, 1], in pixel a piena '
                       'risoluzione; la matrice inversa (mosaico -> tassello) e\' quella da '
                       'usare con Transform di transform_model.py',
        'parametri': {'righe': n_righe, 'colonne': n_colonne, 'percorso': percorso,
                      'partenza': partenza, 'modello_richiesto': modello,
                      'sovrapposizione_attesa_pct': sovrapposizione,
                      'sovrapposizione_cercata': [sov_min, sov_max],
                      'fattore_misura_fine': fattore, 'fattore_grossolano': f_g},
        'mosaico': {'larghezza': Wm, 'altezza': Hm, 'ancora': ancora + 1},
        'modello': {'usato': modello_usato, 'crossval_px': tabella_cv},
        'passo_tipico_px': passi,
        'tasselli': [{
            'numero': t.indice + 1, 'file': os.path.abspath(t.path),
            'riga': t.riga + 1, 'colonna': t.colonna + 1,
            'tassello_verso_mosaico': matrici[t.indice].tolist(),
            'mosaico_verso_tassello': _inversa(matrici[t.indice]).tolist(),
            **{k: round(v, 6) for k, v in _descrivi_posa(matrici[t.indice]).items()},
            'livello': next((l for l in livelli if l['nome'].startswith(f"{t.indice + 1:02d} ")), None),
        } for t in tasselli],
        'coppie': [{
            'a': cp['a'].indice + 1, 'b': cp['b'].indice + 1, 'relazione': cp['relazione'],
            'stato': cp['stato'], 'nota': cp.get('nota'),
            'partenza_da': cp.get('partenza_da'),
            'grossolana': cp['grossolana'],
            'fine': {k: v for k, v in cp['fine'].items()
                     if k not in ('punti_a', 'punti_b', 'pesi')},
            'scarto_globale_px': cp.get('scarto_globale_px'),
            'verifica': cp.get('verifica'),
        } for cp in coppie],
        'verifica': verifica,
        'anteprima': os.path.abspath(path_anteprima),
        'tempo_s': round(time.time() - t_inizio, 1),
    }
    path_json = os.path.join(cartella_out, 'mosaico.json')
    with open(path_json, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    avanza("fatto", 1.0)
    log(f"\nanteprima: {path_anteprima}")
    log(f"report:    {path_json}")
    if scrivi_livelli:
        log(f"Photoshop: File > Script > Sfoglia... -> "
            f"{os.path.join(cartella_out, 'apri_in_photoshop.jsx')}")
    log(f"tempo totale: {time.time() - t_inizio:.1f} s")
    return report


def main():
    ap = argparse.ArgumentParser(
        description="Ricompone un dipinto ripreso a tasselli: posiziona ogni tassello e lo "
                    "scrive come livello da aprire in Photoshop.")
    sorgente = ap.add_mutually_exclusive_group(required=True)
    sorgente.add_argument('--cartella', help="cartella con i tasselli: vengono presi in ordine "
                                             "naturale dei nomi, che deve essere l'ordine di ripresa")
    sorgente.add_argument('--immagini', nargs='+', help="in alternativa, i file nell'ordine di ripresa")
    ap.add_argument('--righe', type=int, required=True)
    ap.add_argument('--colonne', type=int, required=True)
    ap.add_argument('--percorso', choices=list(PERCORSI), default='serpentina_righe')
    ap.add_argument('--partenza', choices=list(PARTENZE), default='alto_sinistra',
                    help="angolo del dipinto da cui parte la ripresa")
    ap.add_argument('--modello', choices=('auto',) + MODELLI, default='auto')
    ap.add_argument('--sovrapposizione', type=float, default=None,
                    help="sovrapposizione attesa fra vicini, in percento (es. 20); "
                         "se non la sai, non metterla")
    ap.add_argument('--fattore', type=int, default=2,
                    help="riduzione per la misura fine: 1 = piena risoluzione, 2 = default")
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--solo-stima', action='store_true',
                    help="stima e verifica senza scrivere i livelli")
    a = ap.parse_args()
    immagini = elenca_immagini(a.cartella) if a.cartella else a.immagini
    componi_mosaico(immagini, a.righe, a.colonne, a.out_dir, percorso=a.percorso,
                    partenza=a.partenza, modello=a.modello, sovrapposizione=a.sovrapposizione,
                    fattore=a.fattore, scrivi_livelli=not a.solo_stima)


if __name__ == '__main__':
    main()
