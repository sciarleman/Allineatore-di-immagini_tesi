"""
Stima della trasformazione VIS -> UV con block matching a phase correlation.

Perche' questo approccio invece di ottimizzare direttamente MIND con Powell:

- Costo. Powell su 6 parametri ricalcola il descriptor MIND dell'immagine
  warpata ad OGNI valutazione della funzione obiettivo (centinaia di volte).
  Qui invece ogni blocco viene misurato UNA volta, con una FFT: una griglia di
  ~1000 blocchi si misura in pochi secondi, e il costo non dipende da quante
  iterazioni servirebbero a un ottimizzatore.

- Robustezza cross-modale. La phase correlation lavora sulla FASE dello
  spettro, quindi ignora le differenze di luminosita'/contrasto fra VIS e UV;
  applicata al MODULO DEL GRADIENTE ignora anche le inversioni di contrasto
  (una craquelure scura in VIS e chiara in UV ha lo stesso modulo di gradiente).

- Diagnostica. Ogni blocco restituisce una CONFIDENZA (l'altezza del picco di
  correlazione): si scartano le zone senza struttura (lo sfondo nero della UV,
  la parete della stanza visibile solo in VIS) invece di lasciare che inquinino
  la stima globale. Ed e' cio' che permette di distinguere una deformazione
  reale da rumore di misura.

- Nessuna finestra di ricerca. Il bug della versione precedente (tutti i blocchi
  che saturavano esattamente a +/- search_radius) nasceva da una ricerca
  esaustiva su una finestra fissa: se il minimo vero e' fuori finestra, o se la
  metrica e' piatta, il risultato finisce sul bordo della finestra e sembra un
  vero spostamento. La phase correlation non ha finestra di ricerca e da' una
  stima sub-pixel con una confidenza associata.

Le metriche robuste di robust_metrics.py (MIND/SSC/NGF/NMI) restano il
riferimento del progetto e vengono usate qui per VALIDARE il risultato
(quality_report), oltre che per l'eventuale raffinamento con la piramide.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from transform_model import Transform


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def gradient_magnitude(img, sigma=1.5, band=1024):
    """
    Modulo del gradiente, normalizzato. E' la rappresentazione su cui facciamo
    il matching: dipende dalla STRUTTURA (bordi, craquelure, pennellate) e non
    dal fatto che una zona sia chiara in VIS e scura in UV o viceversa.

    Il calcolo procede a bande di righe: su un'anteprima da 79 Mpx la versione
    "tutta in una volta" crea cinque array float32 temporanei da ~315 MB
    ciascuno (conversione, blur, sobel x, sobel y, modulo), cioe' oltre 1.5 GB
    di picco per una sola chiamata. A bande il picco resta quello dell'array di
    uscita piu' qualche decina di MB. Le bande hanno un margine (`pad`) per far
    vedere al blur e al Sobel il contesto che sta appena fuori: senza, ogni
    giunzione produrrebbe una riga di falsi bordi.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    out = np.empty((h, w), dtype=np.float32)
    pad = int(np.ceil(4 * sigma)) + 3

    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        ys, ye = max(0, y0 - pad), min(h, y1 + pad)
        f = img[ys:ye].astype(np.float32)
        if f.size and f.max() > 1.0:
            f /= 255.0
        f = cv2.GaussianBlur(f, (0, 0), sigma)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        out[y0:y1] = cv2.magnitude(gx, gy)[y0 - ys:y1 - ys]

    # A cosa serve dividere: phaseCorrelate e' invariante per un fattore
    # moltiplicativo (verificato: stesso spostamento a 1e-9 con fattori da 1e-4
    # a 1e4, e anche con fattori diversi fra le due immagini), quindi questa
    # divisione NON cambia nulla di cio' che viene misurato. Serve solo a dare
    # un significato stabile alle soglie assolute che decidono se un blocco e'
    # troppo piatto per contenere informazione di posizione (min_std qui,
    # 1e-4 nella verifica).
    #
    # Proprio per questo il riferimento non puo' essere il solo massimo. Un
    # singolo bordo molto forte -- una cornice, un riflesso speculare, un
    # graffio sul supporto -- schiaccerebbe tutto il resto verso lo zero e
    # farebbe scartare come "piatti" blocchi di dipinto perfettamente
    # utilizzabili. Il margine non e' teorico: sulla VIS di questo progetto il
    # blocco piu' piatto sta a 1.08e-3 contro una soglia di 1e-3, cioe' a filo.
    #
    # Si tiene quindi il massimo, ma senza lasciargli superare 12 volte il
    # 99.5-esimo percentile, che a differenza del massimo e' fissato dal grosso
    # dell'immagine. Il tetto e' scelto sui dati: sulle sette riprese del
    # progetto il rapporto misurato va da 2.2 (RX) a 6.5 (UVF), quindi a 12 il
    # tetto non interviene mai e nessun risultato gia' validato cambia; entra
    # in funzione solo sul caso patologico da cui protegge.
    mx = float(out.max())
    campione = out[::8, ::8]  # >1 Mpx su un'anteprima da 79: basta e avanza
    p995 = float(np.percentile(campione, 99.5)) if campione.size else 0.0
    rif = min(mx, 12.0 * p995) if p995 > 0 else mx
    if rif > 0:
        out /= rif
    return out


# ---------------------------------------------------------------------------
# Block matching
# ---------------------------------------------------------------------------

def coarse_translation(ref_grad, mov_grad, lato=512, min_conf=0.03, verbose=True):
    """
    Traslazione grossolana fra le due immagini, misurata in una volta sola su
    versioni molto ridotte.

    Serve quando le due riprese non sono gia' quasi sovrapposte: inquadrature
    diverse, sensori diversi, ritagli diversi. In quel caso un blocco preso
    nella stessa posizione nelle due immagini inquadra zone diverse del dipinto
    e il matching a blocchi non ha nulla da agganciare. Misurando prima lo
    scostamento d'insieme, i blocchi partono gia' vicini al bersaglio.

    Se le immagini sono gia' quasi allineate (il caso VIS/UV di questo progetto)
    restituisce uno spostamento minimo e non cambia nulla.

    Returns:
        (ox, oy) in pixel dell'immagine passata, arrotondati all'intero.
    """
    h, w = ref_grad.shape[:2]
    scala = min(1.0, lato / max(h, w))
    if scala < 1.0:
        dim = (max(8, int(w * scala)), max(8, int(h * scala)))
        a = cv2.resize(ref_grad, dim, interpolation=cv2.INTER_AREA)
        b = cv2.resize(mov_grad, dim, interpolation=cv2.INTER_AREA)
    else:
        a, b = ref_grad, mov_grad

    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0, 0
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_64F)
    (dx, dy), conf = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win)
    if conf < min_conf:
        if verbose:
            print(f"  [pre-allineamento] picco debole (conf {conf:.3f}): parto dall'identita'")
        return 0, 0
    ox, oy = int(round(dx / max(scala, 1e-9))), int(round(dy / max(scala, 1e-9)))
    if verbose and (abs(ox) > 2 or abs(oy) > 2):
        print(f"  [pre-allineamento] scostamento d'insieme ({ox:+d}, {oy:+d}) px "
              f"(conf {conf:.3f}): i blocchi partono da li'")
    return ox, oy


def match_blocks(ref_grad, mov_grad, block=256, step=128, min_std=1e-3,
                 max_shift_frac=0.25, roi_frac=1.0, mask=None, offset=(0, 0),
                 n_thread=None, lati_ridotti=True, verbose=True):
    """
    Misura lo spostamento locale fra due immagini gia' pre-processate
    (modulo del gradiente), su una griglia regolare di blocchi.

    Convenzione (verificata sperimentalmente): cv2.phaseCorrelate(a, b)
    restituisce d tale che il dettaglio che in `a` sta in p, in `b` sta in p+d.
    Qui a = ref (VIS) e b = mov (UV), quindi d mappa VIS -> UV, coerente con la
    convenzione di Transform.

    Args:
        mask: maschera opzionale (stessa shape di ref_grad, non zero = da usare).
            Serve solo nel caso patologico in cui il fondo abbia struttura in
            ENTRAMBE le immagini: li' i blocchi sul fondo correlano bene ma,
            essendo a distanza diversa dalla camera, il loro spostamento e'
            genuinamente diverso da quello del dipinto (parallasse) e falserebbe
            la stima. Negli altri casi i filtri automatici (blocco piatto,
            confidenza, outlier) bastano da soli.
        offset: spostamento (ox, oy) da cui partire, tipicamente da
            coarse_translation. Il blocco nella moving viene preso in
            (cx+ox, cy+oy) e lo spostamento riportato ne tiene conto.
        n_thread: thread usati per misurare i blocchi, che sono indipendenti fra
            loro. Il lavoro sta quasi tutto dentro cv2.phaseCorrelate, che
            rilascia il GIL, quindi i thread danno un guadagno reale: misurato
            21.7s -> 5.8s su 8 thread, con risultati identici al caso seriale.
            None sceglie automaticamente, 1 forza il vecchio comportamento.

    Returns:
        array (N, 5) con colonne (cx, cy, dx, dy, confidenza), in coordinate
        dell'immagine passata (tipicamente un'anteprima). L'ordine e' quello
        della scansione per righe, identico qualunque sia il numero di thread.
    """
    h, w = ref_grad.shape[:2]
    hm, wm = mov_grad.shape[:2]
    # Il blocco estratto misura 2*half: con un `block` dispari sarebbe un pixel
    # piu' piccolo della finestra di Hann, e phaseCorrelate rifiuterebbe la
    # chiamata. Si lavora quindi sempre con il lato pari effettivo.
    half = max(8, block // 2)
    lato = 2 * half
    ox, oy = int(offset[0]), int(offset[1])

    y_lo, y_hi = int(h * (1 - roi_frac) / 2), int(h - h * (1 - roi_frac) / 2)
    x_lo, x_hi = int(w * (1 - roi_frac) / 2), int(w - w * (1 - roi_frac) / 2)

    # Lati candidati: quello nominale, poi meta' e un quarto. Servono solo
    # quando c'e' una maschera: vicino al contorno un blocco grande sporge
    # fuori dal soggetto e verrebbe scartato, lasciando cieca una fascia larga
    # mezzo blocco lungo TUTTO il contorno, qualunque forma abbia. Un blocco
    # piu' piccolo ci sta, e misura dove prima non si misurava niente.
    # Il criterio e' la frazione di blocco dentro la maschera, quindi non
    # dipende dall'orientamento del contorno: funziona identico su un lato
    # verticale, sulla diagonale di un ottagono e sull'arco di un tondo.
    # I lati devono restare PARI: il blocco estratto misura 2*(L//2), e con L
    # dispari sarebbe un pixel piu' piccolo della finestra di Hann, con
    # phaseCorrelate che rifiuta la chiamata e fa fallire l'intera stima. Era
    # un errore reale: con un blocco nominale da 190 px il lato dimezzato
    # veniva 95. Si dimezza quindi arrotondando al pari (190 -> 94).
    lati = [lato]
    if lati_ridotti:
        while (lati[-1] // 4) * 2 >= max(48, lato // 4):
            lati.append((lati[-1] // 4) * 2)
    finestre = {L: cv2.createHanningWindow((L, L), cv2.CV_64F) for L in lati}
    righe = list(range(max(half, y_lo + half), min(h - half, y_hi - half) + 1, step))
    colonne = list(range(max(half, x_lo + half), min(w - half, x_hi - half) + 1, step))

    def misura_riga(cy):
        """Una riga di blocchi. Non scrive nulla di condiviso: si puo' eseguire
        in parallelo su righe diverse senza alcun lock."""
        trovati, scartati, ridotti = [], 0, 0
        for cx in colonne:
            # Fuori da una delle due immagini il blocco si salta, come sempre.
            # Rimpicciolirlo per farlo rientrare sarebbe controproducente: sulla
            # radiografia Cappellochiaro, che ha inquadratura diversa, quei
            # blocchi di bordo misurano male e peggioravano il residuo da 1.00 a
            # 2.42 px. Un blocco piu' piccolo con lo stesso centro sta dentro
            # ogni volta che ci sta quello nominale, quindi basta provare questo.
            my, mx = cy + oy, cx + ox
            if (cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w or
                    my - half < 0 or mx - half < 0 or my + half > hm or mx + half > wm):
                continue
            # Vicino al CONTORNO DELLA MASCHERA invece si rimpicciolisce: li' un
            # blocco grande sporge fuori dal soggetto e andrebbe perso, mentre
            # uno piu' piccolo ci sta e misura dove prima non si misurava nulla.
            scelto = None
            for L in lati:
                hh = L // 2
                if mask is not None:
                    # Non basta guardare il centro: un blocco centrato appena
                    # dentro la maschera sporge per meta' fuori, e quella meta'
                    # e' proprio cio' che si voleva escludere.
                    if mask[cy - hh:cy + hh, cx - hh:cx + hh].mean() < 0.75:
                        continue
                scelto = (L, hh)
                break
            if scelto is None:
                if mask is not None:
                    scartati += 1
                continue
            L, hh = scelto
            if L != lato:
                ridotti += 1
            my, mx = cy + oy, cx + ox
            a = ref_grad[cy - hh:cy + hh, cx - hh:cx + hh]
            b = mov_grad[my - hh:my + hh, mx - hh:mx + hh]
            # Un blocco piatto (sfondo nero della UV, parete uniforme) non
            # contiene informazione di posizione: misurarlo produce solo rumore.
            if a.std() < min_std or b.std() < min_std:
                continue
            (dx, dy), conf = cv2.phaseCorrelate(a.astype(np.float64),
                                                b.astype(np.float64), finestre[L])
            # Lo spostamento ammesso e' una frazione del blocco effettivamente
            # usato, non di quello nominale.
            if abs(dx) > L * max_shift_frac or abs(dy) > L * max_shift_frac:
                continue  # spostamento implausibile: quasi certamente un falso picco
            trovati.append((cx, cy, dx + ox, dy + oy, conf))
        return trovati, scartati, ridotti

    n = n_thread if n_thread else min(8, (os.cpu_count() or 1))
    if n > 1 and len(righe) > 1:
        with ThreadPoolExecutor(max_workers=n) as ex:
            per_riga = list(ex.map(misura_riga, righe))
    else:
        per_riga = [misura_riga(cy) for cy in righe]

    # Si ricompone nell'ordine delle righe: il risultato non dipende dall'ordine
    # con cui i thread hanno finito.
    out = [p for trovati, _, _ in per_riga for p in trovati]
    scartati_maschera = sum(s for _, s, _ in per_riga)
    ridotti = sum(r for _, _, r in per_riga)

    pts = np.array(out, dtype=np.float64).reshape(-1, 5)
    if verbose:
        extra = f", {scartati_maschera} esclusi dalla maschera" if scartati_maschera else ""
        if ridotti:
            extra += f", {ridotti} misurati con blocco ridotto vicino al contorno"
        if len(pts):
            print(f"  [match] {len(pts)} blocchi misurati (block={block}, step={step}){extra}, "
                  f"confidenza mediana {np.median(pts[:, 4]):.3f}")
        else:
            print(f"  [match] nessun blocco valido (block={block}, step={step}){extra}")
    return pts


# ---------------------------------------------------------------------------
# Fit affine robusto
# ---------------------------------------------------------------------------

def fit_affine_robust(pts, min_conf=0.10, n_iter=8, sigma_reject=3.0, verbose=True):
    """
    Adatta un'affine al campo di spostamenti misurato, scartando iterativamente
    i blocchi anomali.

    Perche' robusto e non semplici minimi quadrati: qualche blocco sbaglia
    sempre (zone senza struttura, riflessi speculari presenti in una sola
    modalita', il bordo della cornice). Un solo blocco con 70 px di errore,
    elevato al quadrato, sposta la stima globale piu' di cento blocchi buoni.
    Si pesa per confidenza e si scartano gli outlier oltre 3 MAD.

    Returns:
        (M 2x3 fixed->moving, pts_inlier, residui_inlier (N,2))
    """
    pts = pts[pts[:, 4] >= min_conf]
    if len(pts) < 6:
        raise RuntimeError(
            f"Solo {len(pts)} blocchi affidabili: impossibile stimare la trasformazione.\n"
            "Cause tipiche, in ordine di probabilita':\n"
            "  - le due immagini non inquadrano la stessa scena, o una e' un "
            "dettaglio/ritaglio dell'altra (il metodo assume inquadrature simili);\n"
            "  - una delle due e' ruotata di 90 gradi o capovolta rispetto all'altra;\n"
            "  - il soggetto ha pochissima struttura (superficie liscia e uniforme);\n"
            "  - soglia di confidenza troppo alta: prova --min-conf 0.05.\n"
            "Guarda diagnostica/overlay_falsocolore_PRIMA.png per capire da dove si parte.")

    sel = pts
    for it in range(n_iter):
        M, res = _weighted_affine(sel)
        mag = np.hypot(res[:, 0], res[:, 1])
        med = np.median(mag)
        mad = np.median(np.abs(mag - med)) * 1.4826 + 1e-9
        keep = mag <= med + sigma_reject * mad
        if keep.all() or keep.sum() < 6:
            break
        sel = sel[keep]

    M, res = _weighted_affine(sel)
    if verbose:
        mag = np.hypot(res[:, 0], res[:, 1])
        print(f"  [affine] {len(sel)}/{len(pts)} blocchi inlier, "
              f"residuo mediano {np.median(mag):.3f} px, p95 {np.percentile(mag, 95):.3f} px")
    return M, sel, res


# ---------------------------------------------------------------------------
# La soglia minima con cui un blocco e' giudicato concorde con i vicini, e un
# interruttore rimasto in prova (8 ottobre 2026).
#
# SOGLIA_MINIMA_PASSO: un blocco si tiene se si discosta dalla mediana dei vicini
#   meno di una soglia ricavata dal rumore degli altri blocchi (mediana + 3 MAD).
#   Dove il rumore e' bassissimo quella soglia diventa cosi' stretta da scartare
#   una deformazione liscia ma curva, cioe' i bordi e i rigonfiamenti: sul test
#   sintetico era 0.07 px. Ora non scende sotto questa frazione del passo dei
#   blocchi. Misurato con benchmark/prova_locale.py su 11 casi artificiali e 7
#   riprese vere, a 0, 3, 5 e 8%. Alla precisione consigliata, punti oltre 5 px:
#   prospettiva da 200 px 26.3% -> 1.8%, radiale da 200 px 13.5% -> 6.4%,
#   rigonfiamento da 30 px 11.2% -> 0%; casi senza deformazione invariati. Sulle
#   riprese vere l'effetto e' piccolo, perche' li' il rumore tiene gia' larga la
#   soglia: bande fotografiche e SWIR del primo dipinto identiche, NIR
#   dell'affresco da 1.1% a 0.2% di finestre oltre 5 px, UV pari (56 finestre
#   migliori e 55 peggiori). Il 3% aiuta meno e peggiora l'UV dell'affresco
#   (massimo da 19.5 a 54 px); l'8% e' il migliore sul sintetico ma attiva il
#   livello largo sul NIR850 del primo dipinto e alza la mediana di SWIR (0.996
#   -> 1.031) e radiografia (1.069 -> 1.178). Con 0.0 la stima torna quella di
#   prima.
# PROVA_CODA (spento, in prova): un livello grossolano si applica solo se riduce
#   di almeno il 10% il residuo MEDIANO dei blocchi tenuti da parte, e una
#   deformazione che riguarda una minoranza di blocchi la mediana non la
#   sposta. Con True, se quel criterio non passa, si guarda il guadagno sui soli
#   blocchi oltre la portata di chi viene dopo (almeno l'1%, residuo ridotto di
#   almeno il 50%). Da solo non cambia niente, con la soglia risolve un solo
#   caso artificiale e sulle riprese vere non si e' mai attivato: resta spento.
SOGLIA_MINIMA_PASSO = 0.05
PROVA_CODA = False


def blocchi_coerenti(pts, M, step, inlier, min_conf=0.10, raggio=2, soglia_sigma=3.0,
                     min_vicini=3, verbose=True):
    """
    Sceglie i blocchi da cui costruire il campo non rigido, con un criterio
    LOCALE invece che globale.

    fit_affine_robust scarta un blocco quando il suo residuo rispetto
    all'affine e' grande rispetto a quello di TUTTI gli altri. Per stimare
    l'affine e' giusto: una deformazione locale non deve trascinarla. Ma usare
    lo stesso insieme per il campo e' un errore, e sul Fra Galgario e' stato
    misurato: vicino al bordo della tela la deformazione vera e' piu' grande
    della media, e il criterio globale scartava come anomali 40 blocchi su 81
    sulla mano e 65 su 87 al vertice sinistro -- proprio quelli con lo
    spostamento maggiore (~20 px), pur con confidenza buona (0.42-0.44). Il
    campo non li riceveva e restava a 12-14 px: da li' i 5-14 px di
    disallineamento residuo visibili in falso colore.

    Qui un blocco scartato dal criterio globale viene RIAMMESSO se concorda con
    i suoi vicini: una deformazione vera e' liscia, quindi blocchi adiacenti la
    misurano quasi uguale; un falso aggancio invece da' un valore che non
    c'entra con quelli attorno. Conta la coerenza, non la grandezza.

    Perche' solo riammettere e non ri-selezionare tutto da capo: provato, e sul
    test con trasformazione nota peggiorava (errore mediano da 0.338 a 0.362 px,
    massimo da 0.57 a 0.91). Una deformazione liscia ma curva fa differire un
    blocco dalla mediana dei vicini, e il criterio locale puro tagliava proprio
    i picchi. Cosi' invece gli inlier globali restano tutti, e dove il criterio
    globale non scartava nulla il risultato e' identico a prima per costruzione.

    Args:
        pts: (N, 5) come da match_blocks, sulla griglia regolare di passo `step`
        M: affine 2x3 rispetto a cui calcolare i residui (quella di
            fit_affine_robust, cosi' il campo completa esattamente quell'affine)
        inlier: i blocchi accettati da fit_affine_robust; restano tutti
        raggio: i vicini sono quelli in una finestra (2*raggio+1)^2 di blocchi
        soglia_sigma: si riammette entro mediana + soglia_sigma * MAD della
            deviazione dai vicini, misurata sugli inlier (cioe' il rumore locale
            normale), come nel fit robusto
        min_vicini: un blocco scartato con meno vicini validi non si puo'
            giudicare, e resta scartato

    Returns:
        (pts_scelti, residui (N, 2))
    """
    pts = pts[pts[:, 4] >= min_conf]
    x, y, dx, dy = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    pred = np.stack([x, y, np.ones_like(x)], axis=1) @ M.T
    res = np.stack([(x + dx) - pred[:, 0], (y + dy) - pred[:, 1]], axis=1)

    # I blocchi stanno su una griglia regolare: si riportano su una matrice e i
    # vicini diventano semplici spostamenti di indice, senza strutture di ricerca.
    gx = np.round((x - x.min()) / step).astype(int)
    gy = np.round((y - y.min()) / step).astype(int)
    G = np.full((gy.max() + 1, gx.max() + 1, 2), np.nan)
    G[gy, gx] = res
    r = int(raggio)
    P = np.pad(G, ((r, r), (r, r), (0, 0)), constant_values=np.nan)
    vicini = np.stack([P[r + oy:r + oy + G.shape[0], r + ox:r + ox + G.shape[1]]
                       for oy in range(-r, r + 1) for ox in range(-r, r + 1)
                       if (oy, ox) != (0, 0)], axis=0)
    n_vicini = np.sum(~np.isnan(vicini[..., 0]), axis=0)[gy, gx]
    # Le celle della griglia senza blocchi danno "All-NaN slice": e' atteso, e
    # il loro valore non viene mai letto.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mediana_vicini = np.nanmedian(vicini, axis=0)[gy, gx]
    deviazione = np.hypot(res[:, 0] - mediana_vicini[:, 0],
                          res[:, 1] - mediana_vicini[:, 1])

    posizioni_inlier = {(float(a), float(b)) for a, b in inlier[:, :2]}
    e_inlier = np.array([(float(a), float(b)) in posizioni_inlier
                         for a, b in zip(x, y)], dtype=bool)
    giudicabili = n_vicini >= min_vicini
    d = deviazione[giudicabili & e_inlier]
    if len(d) < 12:
        return pts[e_inlier], res[e_inlier]
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-9
    soglia = max(med + soglia_sigma * mad, SOGLIA_MINIMA_PASSO * step)
    ripresi = ~e_inlier & giudicabili & (deviazione <= soglia)
    tieni = e_inlier | ripresi

    if verbose:
        print(f"  [campo] {e_inlier.sum()} inlier dell'affine + {ripresi.sum()} blocchi "
              f"riammessi perche' coerenti con i vicini (deviazione tipica {med:.2f} px, "
              f"soglia {soglia:.2f} px); {(~e_inlier & ~ripresi).sum()} restano scartati")
    return pts[tieni], res[tieni]


def blocchi_concordi(pts, step, min_conf=0.10, raggio=2.5, soglia_sigma=3.0, min_vicini=3):
    """
    Tiene i blocchi che concordano con i vicini, senza nessun modello globale.

    E' il criterio di blocchi_coerenti applicato a TUTTI i blocchi, invece che
    solo a quelli scartati dall'affine: serve dove l'affine non c'e' ancora o
    non e' il riferimento giusto, come nel campo grossolano. Una deformazione
    vera e' liscia, quindi blocchi vicini la misurano quasi uguale; un falso
    aggancio da' un valore che non c'entra con quelli attorno.

    I vicini sono i blocchi entro `raggio` passi di griglia (2.5 = l'anello dei
    primi e secondi vicini, angoli esclusi). Si tiene un blocco se si discosta
    dalla mediana dei vicini meno di mediana + soglia_sigma * MAD di questa
    stessa deviazione; uno con meno di `min_vicini` vicini non si puo'
    giudicare e si scarta.

    Returns:
        (blocchi tenuti, quanti avevano confidenza sufficiente)
    """
    p = pts[pts[:, 4] >= min_conf]
    if len(p) < 12:
        return p[:0], len(p)
    gx = np.round((p[:, 0] - p[:, 0].min()) / step).astype(int)
    gy = np.round((p[:, 1] - p[:, 1].min()) / step).astype(int)
    G = np.full((gy.max() + 1, gx.max() + 1, 2), np.nan)
    G[gy, gx] = p[:, 2:4]
    r = int(np.floor(raggio))
    P = np.pad(G, ((r, r), (r, r), (0, 0)), constant_values=np.nan)
    anello = [(oy, ox) for oy in range(-r, r + 1) for ox in range(-r, r + 1)
              if (oy, ox) != (0, 0) and oy * oy + ox * ox <= raggio * raggio]
    vicini = np.stack([P[r + oy:r + oy + G.shape[0], r + ox:r + ox + G.shape[1]]
                       for oy, ox in anello], axis=0)
    n_vicini = np.sum(~np.isnan(vicini[..., 0]), axis=0)[gy, gx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mediana_vicini = np.nanmedian(vicini, axis=0)[gy, gx]
    deviazione = np.hypot(p[:, 2] - mediana_vicini[:, 0], p[:, 3] - mediana_vicini[:, 1])
    giudicabili = n_vicini >= min_vicini
    if giudicabili.sum() < 12:
        return p[:0], len(p)
    d = deviazione[giudicabili]
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-9
    soglia = max(med + soglia_sigma * mad, SOGLIA_MINIMA_PASSO * step)
    return p[giudicabili & (deviazione <= soglia)], len(p)


class AggancioDubbio(RuntimeError):
    """
    La prima passata ha trovato cosi' pochi blocchi buoni che non si sa se
    l'aggancio sia riuscito. La solleva estimate_transform, e solo se glielo si
    chiede con `soglia_aggancio`: chi la riceve decide come proseguire (vedi
    align_paintings._scegli_partenza).

    Attributi: `T` l'affine della prima passata a piena risoluzione, o None se
    non c'erano nemmeno i blocchi per stimarla; `info` la diagnostica raccolta
    fin li'.
    """

    def __init__(self, T, info, messaggio=""):
        super().__init__(messaggio or "aggancio dubbio dopo la prima passata")
        self.T, self.info = T, info


def blocchi_possibili(shape, block, step, mask=None):
    """
    Quante posizioni di blocco ci sono sulla griglia della prima passata: il
    denominatore della quota di blocchi buoni. Con una maschera contano solo le
    posizioni che la maschera copre per almeno il 75%, come in match_blocks.
    """
    h, w = shape[:2]
    half = max(8, block // 2)
    righe = np.arange(half, h - half + 1, step)
    colonne = np.arange(half, w - half + 1, step)
    if len(righe) == 0 or len(colonne) == 0:
        return 0
    if mask is None:
        return int(len(righe) * len(colonne))
    # La copertura si misura su una maschera rimpicciolita, non su quella
    # intera. La prima versione sfocava la maschera a piena scala in float32:
    # su un'anteprima da 593 Mpx erano due array da 2.4 GB, e una stima a
    # fattore 1 con la zona disegnata passava da 11 a 14 GB su una macchina da
    # 15.8, con la RAM libera a zero. Qui serve solo un conteggio, e ridotta
    # di `r` la maschera pesa pochi megabyte.
    r = max(1, half // 16)
    piccola = cv2.resize(np.asarray(mask).astype(np.uint8) * 255,
                         (max(1, w // r), max(1, h // r)), interpolation=cv2.INTER_AREA)
    lato = max(1, (2 * half) // r)
    cop = cv2.blur(piccola, (lato, lato))
    ri = np.clip(righe // r, 0, cop.shape[0] - 1)
    ci = np.clip(colonne // r, 0, cop.shape[1] - 1)
    return int((cop[np.ix_(ri, ci)] >= 0.75 * 255).sum())


def riduzione_per_il_grossolano(fattore):
    """
    Di quanto ridurre la scala di lavoro per misurare il campo grossolano.

    Fino a qui era sempre 2, quindi la portata del campo grossolano dipendeva
    dalla precisione scelta: 128 px veri a fattore 2, ma solo 64 a fattore 1.
    Cosi' la precisione massima era anche quella che vedeva meno lontano, e su
    una ripresa con deformazioni grandi ai bordi allineava peggio di quella
    normale. Misurato su un affresco da 593 Mpx, banda UV, a fattore 1: in 8
    finestre di verifica su 51 la deformazione vera rispetto all'affine era di
    80-117 px su un asse, oltre la portata; restavano 23-102 px di errore
    (16% delle finestre oltre 5 px, massimo 101.5). Con la portata a 128 px
    le stesse 8 finestre scendono a 0.1-0.5 px (0% oltre 5 px, massimo 1.8),
    e le 40 finestre gia' sotto il pixel restano dov'erano (mediana 0.210 ->
    0.219 px).

    La regola tiene fissa la portata in pixel VERI: i blocchi del campo
    grossolano misurano almeno 512 px a piena risoluzione. A fattore 2 e oltre
    la riduzione resta 2, quindi li' la stima e' identica a prima per
    costruzione; cambia solo sotto il fattore 2. Sulle quattro bande
    fotografiche del Fra Galgario a fattore 1: NIR850 e NIR950 identiche
    finestra per finestra (il campo grossolano resta dentro la portata fine e
    non si applica); UVF mediana 0.171 -> 0.160 px; UVR 10 finestre migliorate
    e 3 peggiorate su 53, oltre 1 px dal 13% al 4%, ma una finestra a bassa
    confidenza (0.08) passa da 1.7 a 12.2 px.

    Provata anche una portata di 256 px (riduzione 8 a fattore 1): sull'UV
    dell'affresco da' lo stesso esito (0% oltre 5 px), con blocchi da 1024 px
    che seguono peggio le deformazioni minute (oltre 1 px: 12% contro 6%).
    """
    return max(2, int(round(4.0 / max(float(fattore), 1e-6))))


def livelli_del_grossolano(fattore):
    """
    Le riduzioni dei campi grossolani, dalla piu' larga alla piu' stretta: un
    livello largo con portata doppia, poi quello di riduzione_per_il_grossolano.

    Un solo livello vede fino a 128 px. Sull'UV di un affresco da 593 Mpx la
    deformazione negli angoli arrivava a ~280 px e restava fuori: sulle finestre
    da 512 px degli angoli, a fattore 1, il 42% oltre 5 px (mediana 0.85,
    massimo 254). Con un livello largo davanti (portata 256 px) che porta la
    deformazione dentro la portata di quello stretto: 3% oltre 5 px, mediana
    0.21, massimo 77; l'interno del dipinto resta dov'era (0.24 -> 0.25 px), la
    stima costa 24 s in piu'.

    Il livello largo non sostituisce quello stretto (i suoi blocchi da 1024 px
    seguono peggio le deformazioni minute) e si applica solo se almeno l'1% dei
    blocchi supera la portata di quello stretto: vedi estimate_transform. Dove
    non si applica la stima e' quella a un livello, finestra per finestra:
    verificato su NIR850, UVR, UVF, SWIR e una radiografia del Fra Galgario e
    sul NIR dell'affresco.
    """
    r = riduzione_per_il_grossolano(fattore)
    return (2 * r, r)


def campo_grossolano(img_fixed, mov_allineata, griglia, mask=None, block=128, step=64,
                     min_conf=0.10, fattore=1.0, riduzione=2, verbose=True,
                     portata_dopo_px_full=None):
    """
    Il campo di deformazione a scala grossolana: la terza scala, fra l'affine e
    il campo fine.

    Perche' serve. La seconda passata misura con blocchi da `refine_block` e
    scarta ogni spostamento oltre un quarto del blocco (max_shift_frac): con i
    valori di sempre la sua portata e' di 64 px a piena risoluzione. La prima
    passata ha il doppio della portata, ma il fit affine robusto scarta come
    anomali i blocchi che misurano una deformazione grande, e da quei blocchi
    nessuno costruisce un campo. Una deformazione oltre i 64 px che l'affine non
    rappresenta resta quindi invisibile a tutte e due.

    Misurato sulla radiografia Cappellochiaro: sul lato destro il dipinto stava
    ~100 px piu' all'esterno di dove lo metteva l'automatica. Lo dicevano i
    punti messi a mano su dettagli interni (+103 px), il bordo tracciato a mano
    sulle due immagini, e le tre sole finestre di verifica misurabili in quel
    quarto (72, 104 e 97 px). Blocchi grandi misurati sulla moving pre-allineata
    con l'affine lo vedono: +102 px.

    Come. I blocchi si misurano a META' della risoluzione di lavoro, con il lato
    e il passo della prima passata dimezzati: a fattore 2 sono blocchi da 512 px
    pieni ogni 256, con portata 128 px. Si tengono quelli che concordano con i
    vicini (blocchi_concordi: qui non c'e' un'affine rispetto a cui giudicarli),
    se ne fa un campo liscio con la stessa normalized convolution del campo
    fine e lo smoothing scelto per cross-validation. Il campo si costruisce
    sulla griglia `griglia` della scala di lavoro, cioe' la stessa del campo
    fine: cosi' i due si sommano senza ricampionarne nessuno.

    Si applica solo se la cross-validation mostra un guadagno fuori campione di
    almeno il 10%, lo stesso criterio del campo fine: sotto, e' rumore.

    Args:
        img_fixed, mov_allineata: anteprime grayscale alla scala di lavoro, la
            seconda gia' riportata nel sistema della prima con l'affine.
        griglia: passo della griglia del campo, in px della scala di lavoro.
        mask: maschera del soggetto alla scala di lavoro, o None.
        block, step: lato e passo dei blocchi, in px della scala RIDOTTA.
        riduzione: di quanto si riduce la scala di lavoro per misurare i
            blocchi (2 = la mezza scala di sempre). E' cio' che fissa la portata
            in pixel pieni: block * 0.25 * riduzione * fattore.
        fattore: fattore di riduzione della scala di lavoro, solo per riportare
            i numeri in pixel pieni.

    Returns:
        (campo (Gh, Gw, 2) in px della scala di lavoro, info) oppure (None, info)
    """
    h, w = img_fixed.shape[:2]
    riduzione = max(1, int(riduzione))
    piccola = (max(8, w // riduzione), max(8, h // riduzione))
    sx, sy = w / float(piccola[0]), h / float(piccola[1])
    gfs = gradient_magnitude(cv2.resize(img_fixed, piccola, interpolation=cv2.INTER_AREA))
    gms = gradient_magnitude(cv2.resize(mov_allineata, piccola, interpolation=cv2.INTER_AREA))
    mk = None
    if mask is not None:
        mk = cv2.resize(np.asarray(mask).astype(np.uint8), piccola,
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    pts = match_blocks(gfs, gms, block=block, step=step, mask=mk, verbose=False)
    del gms
    buoni, n_conf = blocchi_concordi(pts, step, min_conf=min_conf)
    f_px = float(riduzione) * fattore    # da px della scala ridotta a px pieni
    info = {'applicato': False, 'blocchi_misurati': int(len(pts)),
            'blocchi_affidabili': int(n_conf), 'blocchi_concordi': int(len(buoni)),
            'lato_blocco_px_full': float(block * f_px),
            'portata_px_full': float(block * 0.25 * f_px)}
    if len(buoni) < 12:
        info['motivo'] = 'troppi pochi blocchi concordi'
        return None, info
    # Quanto si spostano i blocchi, al 99-esimo percentile: a differenza del
    # massimo del campo, un blocco sbagliato da solo non lo muove.
    info['ampiezza_p99_blocchi_px_full'] = float(
        np.percentile(np.hypot(buoni[:, 2], buoni[:, 3]), 99) * f_px)

    candidati = [step * k for k in (0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)]
    soglia_coda = (portata_dopo_px_full / f_px
                   if PROVA_CODA and portata_dopo_px_full is not None else None)
    s, stats, tabella = select_smoothing(buoni, buoni[:, 2:4], gfs.shape, step, candidati,
                                         verbose=False, soglia_coda=soglia_coda)
    if not stats.get('usable'):
        info['motivo'] = 'cross-validation non utilizzabile'
        return None, info
    info.update(smooth_px_full=float(s * f_px), guadagno_pct=float(stats['guadagno_pct']),
                residuo_prima_px_full=float(stats['residuo_prima_px'] * f_px),
                residuo_dopo_px_full=float(stats['residuo_dopo_px'] * f_px))
    if stats['guadagno_pct'] < 10.0:
        sulla_coda = [r for r in tabella if r.get('quota_coda', 0.0) >= 0.01
                      and r.get('guadagno_coda_pct', 0.0) >= 50.0]
        if not sulla_coda:
            info['motivo'] = ("la cross-validation non mostra un guadagno reale: alla scala "
                              "grossolana non c'e' deformazione da correggere")
            if verbose:
                print(f"  [grossolano] nessun guadagno fuori campione "
                      f"({stats['guadagno_pct']:+.0f}%): campo non applicato")
            return None, info
        # In prova (PROVA_CODA): la mediana non vede guadagno, ma i blocchi
        # oltre la portata di chi viene dopo si'. Lo smoothing e' quello che li
        # segue meglio.
        migliore = min(sulla_coda, key=lambda r: r['residuo_dopo_coda_px'])
        s = migliore['smooth_px']
        info.update(smooth_px_full=float(s * f_px), criterio='coda',
                    quota_coda=float(migliore['quota_coda']),
                    guadagno_coda_pct=float(migliore['guadagno_coda_pct']))
        if verbose:
            print(f"  [grossolano] la mediana non mostra guadagno ({stats['guadagno_pct']:+.0f}%), "
                  f"ma il {100 * migliore['quota_coda']:.0f}% dei blocchi oltre "
                  f"{portata_dopo_px_full:.0f} px si' ({migliore['guadagno_coda_pct']:+.0f}%)")

    # Dalla mezza scala a quella di lavoro: posizioni e spostamenti si
    # riscalano, e lo smoothing pure, cosi' resta lo stesso in pixel pieni.
    pw = buoni.copy()
    pw[:, 0] = (buoni[:, 0] + 0.5) * sx - 0.5
    pw[:, 1] = (buoni[:, 1] + 0.5) * sy - 0.5
    pw[:, 2] = buoni[:, 2] * sx
    pw[:, 3] = buoni[:, 3] * sy
    campo = build_dense_field(pw, pw[:, 2:4], (h, w), griglia, s * 0.5 * (sx + sy),
                              mask=mask, verbose=False)
    ampiezza = float(np.hypot(campo[..., 0], campo[..., 1]).max())
    info.update(applicato=True, ampiezza_max_px_full=ampiezza * fattore)
    if verbose:
        print(f"  [grossolano] {len(buoni)} blocchi concordi su {n_conf} affidabili "
              f"(blocchi da {block * f_px:.0f} px pieni, portata {block * 0.25 * f_px:.0f} px); "
              f"fuori campione {info['residuo_prima_px_full']:.1f} -> "
              f"{info['residuo_dopo_px_full']:.1f} px ({stats['guadagno_pct']:+.0f}%), "
              f"sigma {s * f_px:.0f} px; ampiezza max {ampiezza * fattore:.0f} px")
    return campo, info


def _weighted_affine(pts):
    """Minimi quadrati pesati per confidenza. Restituisce (M 2x3, residui)."""
    x, y, dx, dy, conf = pts.T
    wgt = np.sqrt(np.clip(conf, 0.0, None))
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    Aw = A * wgt[:, None]
    cx, *_ = np.linalg.lstsq(Aw, (x + dx) * wgt, rcond=None)
    cy, *_ = np.linalg.lstsq(Aw, (y + dy) * wgt, rcond=None)
    M = np.stack([cx, cy], axis=0)
    pred = A @ M.T
    res = np.stack([(x + dx) - pred[:, 0], (y + dy) - pred[:, 1]], axis=1)
    return M, res


# ---------------------------------------------------------------------------
# Campo denso
# ---------------------------------------------------------------------------

def _campo_ordine1(pts, vectors, forma_griglia, grid_step, sigma_grid, ripiego,
                   ridge=0.1, copertura_min=0.02):
    """
    Campo per media locale di primo ordine (normalized convolution di ordine 1).

    Per ogni cella si adatta v = a + b*(x-X) + c*(y-Y) con pesi gaussiani e si
    tiene a. Tutte le somme pesate servono solo come sfocature gaussiane di
    momenti accumulati sulla griglia, quindi il costo resta quello di qualche
    GaussianBlur. Dove il peso totale e' troppo basso per fidarsi si torna al
    campo di ordine 0 passato in `ripiego`.
    """
    gh, gw = forma_griglia
    x, y = pts[:, 0] / grid_step, pts[:, 1] / grid_step
    ix = np.clip(np.round(x).astype(int), 0, gw - 1)
    iy = np.clip(np.round(y).astype(int), 0, gh - 1)
    wg = np.clip(pts[:, 4], 0.0, None)

    def sfoca(val):
        A = np.zeros((gh, gw), np.float64)
        np.add.at(A, (iy, ix), val)
        # bordo a zero: fuori dalla griglia non ci sono blocchi, e riflettere i
        # dati falserebbe i momenti, che dipendono dalle coordinate assolute
        return cv2.GaussianBlur(A, (0, 0), sigma_grid, borderType=cv2.BORDER_CONSTANT)

    Y, X = np.mgrid[0:gh, 0:gw].astype(np.float64)
    S0, SX, SY = sfoca(wg), sfoca(wg * x), sfoca(wg * y)
    SXX, SXY, SYY = sfoca(wg * x * x), sfoca(wg * x * y), sfoca(wg * y * y)
    Sx, Sy = SX - X * S0, SY - Y * S0
    Sxx = SXX - 2 * X * SX + X * X * S0
    Sxy = SXY - X * SY - Y * SX + X * Y * S0
    Syy = SYY - 2 * Y * SY + Y * Y * S0
    r = ridge * S0 * sigma_grid * sigma_grid          # tiene stabile il sistema
    A = np.stack([np.stack([S0, Sx, Sy], -1),
                  np.stack([Sx, Sxx + r, Sxy], -1),
                  np.stack([Sy, Sxy, Syy + r], -1)], -2)
    rhs = []
    for k in range(2):
        T0 = sfoca(wg * vectors[:, k])
        rhs.append(np.stack([T0, sfoca(wg * vectors[:, k] * x) - X * T0,
                             sfoca(wg * vectors[:, k] * y) - Y * T0], -1))
    rhs = np.stack(rhs, -1)

    out = np.asarray(ripiego, dtype=np.float64).copy()
    ok = S0 > copertura_min * float(S0.max() + 1e-12)
    if ok.any():
        sol = np.linalg.solve(A[ok] + np.eye(3) * 1e-12, rhs[ok])
        out[ok] = sol[:, 0, :]
    return out.astype(np.float32)


# Fuori dalla zona disegnata il campo si spegne in una lunghezza pari a questo
# numero di volte il suo spostamento sul contorno (vedi build_dense_field).
# Quattro: campo grossolano e campo fine si spengono ciascuno per conto suo e i
# due stiramenti si sommano, al piu' 1/4 + 1/4.
SFUMATURA_PER_AMPIEZZA = 4.0


def build_dense_field(pts, vectors, shape, grid_step, smooth_px, mask=None,
                      ordine=0, verbose=True):
    """
    Interpola vettori sparsi (con peso) in un campo denso e liscio, con la
    "normalized convolution": si accumulano i vettori pesati su una griglia
    rada, si sfoca sia il numeratore sia il denominatore e si divide.

    Rispetto a scipy.interpolate.griddata:
    - non produce NaN fuori dall'inviluppo convesso dei punti (le zone senza
      dati vengono riempite da quelle vicine, con sigma via via piu' grande);
    - pesa i blocchi per confidenza invece di trattarli tutti uguali;
    - e' O(N) e non richiede triangolazione.

    Il campo risultante e' liscio per costruzione: e' cio' che vogliamo, perche'
    una tela che si flette produce deformazioni a bassa frequenza spaziale, non
    salti da blocco a blocco.

    Args:
        pts: (N, 5) come da match_blocks (servono cx, cy, confidenza)
        vectors: (N, 2) vettori da interpolare (tipicamente i residui dopo affine)
        shape: (H, W) dell'immagine a cui si riferisce il campo
        grid_step: passo della griglia di output, in px di quell'immagine
        smooth_px: sigma dello smoothing, in px di quell'immagine. Va scelto
            dell'ordine della distanza fra blocchi: piu' piccolo insegue il
            rumore di misura, piu' grande appiattisce la deformazione vera.
        mask: maschera del soggetto (stessa shape di `shape`, non zero = dentro).
            Se data, il campo viene fatto sfumare a zero appena fuori dal
            soggetto, cioe' li' la trasformazione si riduce alla sola affine.

    Returns:
        campo (Gh, Gw, 2) float32
    """
    h, w = shape[:2]
    gh = int(np.ceil(h / grid_step)) + 1
    gw = int(np.ceil(w / grid_step)) + 1

    gx = np.clip(np.round(pts[:, 0] / grid_step).astype(int), 0, gw - 1)
    gy = np.clip(np.round(pts[:, 1] / grid_step).astype(int), 0, gh - 1)
    wgt = np.clip(pts[:, 4], 0.0, None).astype(np.float32)

    num = np.zeros((gh, gw, 2), dtype=np.float32)
    den = np.zeros((gh, gw), dtype=np.float32)
    np.add.at(num, (gy, gx, 0), (vectors[:, 0] * wgt).astype(np.float32))
    np.add.at(num, (gy, gx, 1), (vectors[:, 1] * wgt).astype(np.float32))
    np.add.at(den, (gy, gx), wgt)

    sigma_grid = max(0.8, smooth_px / grid_step)
    field = np.zeros((gh, gw, 2), dtype=np.float32)
    filled = np.zeros((gh, gw), dtype=bool)

    # Scale crescenti: le zone coperte da blocchi usano il sigma richiesto, le
    # zone vuote (bordi, sfondo senza struttura) vengono estrapolate da quelle
    # vicine con un sigma sempre piu' grande, invece di restare a zero (che
    # produrrebbe uno scalino proprio al bordo del dipinto).
    for k in range(6):
        s = sigma_grid * (2 ** k)
        nb = cv2.GaussianBlur(num, (0, 0), s)
        db = cv2.GaussianBlur(den, (0, 0), s)
        # Dove il denominatore e' quasi nullo la divisione amplificherebbe il
        # rumore: si considera "coperta" solo una cella con almeno il 2% del
        # peso massimo presente a questa scala di smoothing.
        ok = db > 0.02 * float(db.max() + 1e-12)
        new = ok & ~filled
        if new.any():
            safe = np.maximum(db, 1e-12)
            field[new] = (nb[new] / safe[new][:, None])
            filled |= new
        if filled.all():
            break

    if ordine == 1:
        # Media locale di PRIMO ordine: invece di una costante si adatta
        # localmente un piano (valore + pendenza) e si prende il suo valore al
        # centro. Dove i blocchi stanno tutti da un lato -- al bordo del
        # soggetto, o accanto a una zona senza misure -- la media di ordine 0 e'
        # tirata verso l'interno; una pendenza locale non lo e'. Si usa solo
        # dove il supporto e' sufficiente, e altrove resta il campo di ordine 0.
        field = _campo_ordine1(pts, vectors, (gh, gw), grid_step, sigma_grid, field)

    # Fuori dal soggetto non c'e' stata nessuna misura: quello che si vede la'
    # e' solo estrapolazione, e su una forma non rettangolare (un ottagono, un
    # tondo) la zona estrapolata e' vasta -- i quattro angoli dell'immagine.
    # Applicarci una deformazione inventata non ha senso e puo' allontanare lo
    # sfondo dalla sua posizione corretta. Si sfuma quindi il campo a zero
    # appena fuori dal contorno: la' resta la sola affine, che e' l'ipotesi
    # piu' prudente. La sfumatura e' lunga almeno quanto il sigma del campo, e
    # comunque SFUMATURA_PER_AMPIEZZA volte lo spostamento che il campo ha sul
    # contorno in quel punto: spegnere uno spostamento di A pixel in una
    # lunghezza L stira o comprime l'immagine di A / L, e con A > L la ripiega
    # su se stessa. Con il solo sigma succedeva: sull'UV di un affresco il
    # campo sul contorno valeva ~218 px e si spegneva in 96, e lungo il bordo
    # della zona l'immagine usciva strappata.
    if mask is not None:
        m_grid = cv2.resize((np.asarray(mask) > 0).astype(np.uint8), (gw, gh),
                            interpolation=cv2.INTER_NEAREST)
        if 0 < int(m_grid.sum()) < m_grid.size:
            dist_fuori, vicino = cv2.distanceTransformWithLabels(
                (m_grid == 0).astype(np.uint8), cv2.DIST_L2, 3,
                labelType=cv2.DIST_LABEL_PIXEL)
            dist_fuori = dist_fuori * grid_step
            # l'ampiezza del campo nella cella del soggetto piu' vicina
            dentro = m_grid > 0
            per_etichetta = np.zeros(int(vicino.max()) + 1, dtype=np.float32)
            per_etichetta[vicino[dentro]] = np.hypot(field[..., 0], field[..., 1])[dentro]
            lunghezza = np.maximum(smooth_px, SFUMATURA_PER_AMPIEZZA * per_etichetta[vicino])
            rampa = np.clip(1.0 - dist_fuori / np.maximum(lunghezza, 1e-6), 0.0, 1.0)
            field *= rampa[:, :, None].astype(np.float32)

    if verbose:
        mag = np.hypot(field[:, :, 0], field[:, :, 1])
        print(f"  [campo] griglia {gw}x{gh} (passo {grid_step:.0f} px), "
              f"ampiezza rms {np.sqrt((mag ** 2).mean()):.2f} px, max {mag.max():.2f} px")
    return field


def field_crossval_gain(pts, vectors, shape, grid_step, smooth_px, seed=0, ordine=0,
                        soglia_coda=None):
    """
    Misura se il campo non rigido e' SEGNALE o rumore, con una cross-validation:
    si costruisce il campo usando meta' dei blocchi e si misura di quanto riduce
    il residuo sull'altra meta' (blocchi mai visti).

    Se il guadagno fuori campione e' piccolo (o negativo), la deformazione
    stimata e' rumore di misura e conviene restare all'affine: e' esattamente il
    controllo che mancava alla versione precedente della pipeline, dove un campo
    tutto rumore veniva applicato come se fosse reale.

    Returns:
        dict con residuo mediano prima/dopo (fuori campione) e guadagno relativo.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pts))
    tr, te = idx[::2], idx[1::2]
    if len(tr) < 8 or len(te) < 8:
        return {'usable': False}

    field_tr = build_dense_field(pts[tr], vectors[tr], shape, grid_step, smooth_px,
                                 ordine=ordine, verbose=False)
    gx = np.clip(pts[te, 0] / grid_step, 0, field_tr.shape[1] - 1).astype(np.float32)
    gy = np.clip(pts[te, 1] / grid_step, 0, field_tr.shape[0] - 1).astype(np.float32)
    # A pezzi: cv2.remap rifiuta mappe con 32767 righe o piu', e i blocchi
    # tenuti da parte stanno in colonna, uno per riga. A fattore 2 sono ~10 000,
    # ma a fattore 1 con i blocchi da 96 px diventano ~68 000, e la stima
    # falliva. Ogni blocco si interpola per conto suo, quindi a pezzi il
    # risultato e' identico.
    pezzo = 30000
    pred = np.concatenate([
        cv2.remap(field_tr, gx[i:i + pezzo].reshape(-1, 1), gy[i:i + pezzo].reshape(-1, 1),
                  interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
        for i in range(0, len(gx), pezzo)])

    before = np.hypot(vectors[te, 0], vectors[te, 1])
    after = np.hypot(vectors[te, 0] - pred[:, 0], vectors[te, 1] - pred[:, 1])
    out = {
        'usable': True,
        'residuo_prima_px': float(np.median(before)),
        'residuo_dopo_px': float(np.median(after)),
        'guadagno_pct': float(100.0 * (1 - np.median(after) / max(np.median(before), 1e-9))),
    }
    if soglia_coda is not None:
        # In prova: lo stesso guadagno, ma sui soli blocchi tenuti da parte che
        # misurano uno spostamento oltre `soglia_coda`.
        coda = before > soglia_coda
        out['quota_coda'] = float(coda.mean())
        if coda.sum() >= 4:
            out['residuo_dopo_coda_px'] = float(np.median(after[coda]))
            out['guadagno_coda_pct'] = float(
                100.0 * (1 - np.median(after[coda]) / max(np.median(before[coda]), 1e-9)))
    return out


# ---------------------------------------------------------------------------
# Stima completa
# ---------------------------------------------------------------------------

def select_smoothing(pts, vectors, shape, grid_step, candidates, seeds=(0, 1, 2), verbose=True,
                     soglia_coda=None):
    """
    Sceglie l'ampiezza dello smoothing del campo non rigido misurando, per ogni
    candidato, il residuo FUORI CAMPIONE (cross-validation).

    Perche' non un valore fisso: lo smoothing giusto dipende da quanto e' "a
    onda lunga" la deformazione reale, che non si conosce in anticipo. Troppo
    smoothing e le deformazioni a frequenza piu' alta vengono attenuate (su un
    test sintetico con un'onda di periodo 900 px, un sigma di 144 px ne
    cancellava il 40%); troppo poco e il campo insegue il rumore di misura dei
    singoli blocchi. La cross-validation misura direttamente il compromesso sui
    dati veri, senza doverlo indovinare.

    Returns:
        (sigma_migliore, stats_del_migliore, tabella_completa)
    """
    table = []
    for s in candidates:
        gains = [field_crossval_gain(pts, vectors, shape, grid_step, s, seed=sd,
                                     soglia_coda=soglia_coda) for sd in seeds]
        gains = [g for g in gains if g.get('usable')]
        if not gains:
            continue
        coda = {}
        if soglia_coda is not None and all('guadagno_coda_pct' in g for g in gains):
            coda = {k: float(np.mean([g[k] for g in gains]))
                    for k in ('quota_coda', 'residuo_dopo_coda_px', 'guadagno_coda_pct')}
        table.append({
            **coda,
            'smooth_px': float(s),
            'residuo_dopo_px': float(np.mean([g['residuo_dopo_px'] for g in gains])),
            'residuo_prima_px': float(np.mean([g['residuo_prima_px'] for g in gains])),
            'guadagno_pct': float(np.mean([g['guadagno_pct'] for g in gains])),
        })
    if not table:
        return None, {'usable': False}, []

    best = min(table, key=lambda r: r['residuo_dopo_px'])
    if verbose:
        print("  [smoothing] scelta per cross-validation "
              "(residuo fuori campione, px alla scala di lavoro):")
        for row in table:
            mark = " <--" if row is best else ""
            print(f"      sigma {row['smooth_px']:6.0f} px -> {row['residuo_dopo_px']:.3f} "
                  f"({row['guadagno_pct']:+.0f}%){mark}")
    best['usable'] = True
    return best['smooth_px'], best, table


def estimate_transform(img_fixed, img_moving, factor, factor_moving=None,
                       block=256, step=128, min_conf=0.10,
                       nonrigid=True, refine_block=128, refine_step=64,
                       smooth_px=None, grid_step=None, roi_frac=1.0,
                       mask=None, coarse_init=True, verbose=True, diagnostica=None,
                       campo_grossolano_tra_le_passate=False, riduzione_grossolano=None,
                       soglia_aggancio=None):
    """
    Stima completa su una coppia di anteprime ridotte.

    `diagnostica`, se e' un dizionario, riceve i dati che servono alle immagini
    di controllo ma non vanno nel report JSON. Oggi non ne riceve nessuno:
    l'unico uso era la mappa delle cuciture di mosaico, che e' stata tolta.

    Due passate:
      1. blocchi grandi sulle immagini originali -> affine globale robusta;
      2. blocchi piu' piccoli sulla moving GIA' PRE-ALLINEATA con l'affine ->
         residuo. Pre-allineare e' importante: al secondo giro gli spostamenti
         da misurare sono quasi nulli, quindi il picco di correlazione e' piu'
         netto e blocchi piu' piccoli (cioe' un campo piu' fitto) restano
         affidabili. Dal residuo si ricava sia una correzione affine sia, se la
         cross-validation lo giustifica, il campo non rigido.

    Args:
        img_fixed, img_moving: anteprime grayscale (uint8)
        factor: fattore di riduzione dell'immagine FIXED
        factor_moving: fattore di riduzione della MOVING. Se None si assume lo
            stesso della fixed. Sono diversi quando le due riprese hanno
            dimensioni diverse (sensori o ritagli diversi): ognuna viene ridotta
            al fattore intero che la porta vicino alla risoluzione di lavoro.
        mask: maschera opzionale alla scala dell'anteprima fixed (non zero = zona
            da usare per la stima). Vedi match_blocks.
        coarse_init: misura prima uno scostamento d'insieme, utile quando le due
            riprese non sono gia' quasi sovrapposte.
        nonrigid: se False si ferma all'affine
        smooth_px: sigma dello smoothing del campo, in px dell'anteprima
            (default: scelto per cross-validation)
        grid_step: passo della griglia del campo, in px dell'anteprima
            (default: refine_step)
        campo_grossolano_tra_le_passate: se True, fra la prima e la seconda
            passata si stima anche un campo a scala grossolana (vedi
            campo_grossolano), e la seconda passata parte da li'. Spento per
            impostazione predefinita: lo usa solo la sezione delle radiografie,
            e con False la stima e' esattamente quella di sempre.
        riduzione_grossolano: di quanto si riduce la scala di lavoro per
            misurare il campo grossolano: un numero, o una sequenza dalla
            piu' larga alla piu' stretta per piu' campi in fila. None (il
            default) = due livelli, portata 256 e poi 128 px veri qualunque
            sia la precisione: vedi livelli_del_grossolano.
        soglia_aggancio: se data, dopo la prima passata si guarda la quota di
            blocchi buoni sul totale delle posizioni; sotto la soglia ci si
            ferma e si solleva AggancioDubbio invece di completare la stima.
            None (il default) = si prosegue sempre, come prima.

    Returns:
        (Transform a piena risoluzione, dizionario di diagnostica)
    """
    f_f = float(factor)
    f_m = float(factor if factor_moving is None else factor_moving)
    if riduzione_grossolano is None:
        riduzione_grossolano = livelli_del_grossolano(f_f)
    info = {'factor': f_f, 'factor_moving': f_m,
            'preview_shape': list(img_fixed.shape[:2]),
            'preview_shape_moving': list(img_moving.shape[:2])}

    # I blocchi devono starci: su un'anteprima piccola (immagine piccola, o
    # fattore di riduzione alto) blocchi da 256 px ne farebbero entrare due o
    # tre in tutto, e con due o tre misure non si stima nulla. Si chiede almeno
    # una griglia di ~5 blocchi per lato, riducendo la dimensione se serve.
    lato_minimo = min(img_fixed.shape[0], img_fixed.shape[1])
    block_max = max(48, lato_minimo // 5)
    if block > block_max:
        nuovo_refine = max(32, min(refine_block, block_max // 2))
        if verbose:
            print(f"  [blocchi] anteprima {img_fixed.shape[1]}x{img_fixed.shape[0]}: "
                  f"blocchi ridotti da {block} a {block_max} px "
                  f"(e da {refine_block} a {nuovo_refine} per la seconda passata), "
                  f"altrimenti non ce ne starebbero abbastanza")
        block, refine_block = block_max, nuovo_refine
        step, refine_step = max(8, block // 2), max(8, nuovo_refine // 2)
    info['block'] = int(block)
    info['refine_block'] = int(refine_block)

    gf = gradient_magnitude(img_fixed)
    gm = gradient_magnitude(img_moving)

    if verbose:
        print("\n[stima 1/2] affine globale")
    offset = coarse_translation(gf, gm, verbose=verbose) if coarse_init else (0, 0)
    info['pre_allineamento_px'] = [int(offset[0]), int(offset[1])]

    pts1 = match_blocks(gf, gm, block=block, step=step, roi_frac=roi_frac,
                        mask=mask, offset=offset, verbose=verbose)
    n_possibili = blocchi_possibili(img_fixed.shape, block, step, mask)
    try:
        M_prev, inl1, res1 = fit_affine_robust(pts1, min_conf=min_conf, verbose=verbose)
    except RuntimeError as exc:
        if soglia_aggancio is None:
            raise
        # Nemmeno i sei blocchi per un'affine: e' il caso estremo dell'aggancio
        # dubbio, e chi ha chiesto di esserne avvisato puo' ancora ripartire da
        # un'altra posa invece di fermarsi qui.
        info['pass1'] = {'blocchi': int(len(pts1)), 'inlier': 0,
                         'possibili': int(n_possibili), 'quota_inlier': 0.0}
        raise AggancioDubbio(None, info, str(exc).splitlines()[0]) from exc

    del gm  # ~300 MB a fattore 2: non serve piu', la seconda passata usa la moving pre-allineata
    T = Transform.from_factor(M_prev, f_f, ref_shape=None, f_moving=f_m)
    info['pass1'] = {
        'blocchi': int(len(pts1)),
        'inlier': int(len(inl1)),
        'residuo_mediano_px_full': float(np.median(np.hypot(res1[:, 0], res1[:, 1])) * f_f),
        'possibili': int(n_possibili),
        'quota_inlier': float(len(inl1) / max(n_possibili, 1)),
    }
    if soglia_aggancio is not None and info['pass1']['quota_inlier'] < soglia_aggancio:
        raise AggancioDubbio(T, info)

    if not nonrigid:
        info['nonrigid'] = {'applicato': False, 'motivo': 'disattivato dall\'utente'}
        return T, info

    # ---- fra le due passate: il campo grossolano, solo se richiesto -------
    # Va prima della passata fine, non dopo: aggiunto dopo, il suo rumore
    # resterebbe anche dove il campo fine aveva gia' allineato bene (provato:
    # al centro della radiografia il residuo saliva da 1.56 a 1.72 px). Prima,
    # la passata fine ricorregge tutto cio' che il grossolano lascia.
    if campo_grossolano_tra_le_passate:
        # `riduzione_grossolano` puo' essere un numero (un solo campo
        # grossolano, come sempre) oppure una sequenza dalla piu' larga alla
        # piu' stretta: ogni livello si misura sulla moving gia' corretta dai
        # precedenti, e porta la deformazione dentro la portata del successivo.
        livelli = ([int(riduzione_grossolano)] if np.isscalar(riduzione_grossolano)
                   else [int(r) for r in riduzione_grossolano])
        blocco_g = max(32, block // 2)
        griglia = grid_step or refine_step
        portata_fine = refine_block * 0.25 * f_f
        registro = []
        for n_liv, rid in enumerate(livelli):
            if verbose:
                print(f"\n[stima 1b] campo grossolano, riduzione {rid}, sulla moving "
                      f"allineata fin qui")
            mov_affine = T.at_factor(f_f, f_m).apply_to_image(img_moving, img_fixed.shape[:2])
            campo_g, info_g = campo_grossolano(
                img_fixed, mov_affine, griglia, mask=mask, block=blocco_g,
                step=max(8, step // 2), min_conf=min_conf, fattore=f_f,
                riduzione=rid, verbose=verbose,
                portata_dopo_px_full=(portata_fine if n_liv == len(livelli) - 1
                                      else blocco_g * 0.25 * livelli[n_liv + 1] * f_f))
            del mov_affine
            # Con 'auto' il campo grossolano si applica solo se va oltre la
            # portata di chi viene dopo: il livello successivo, o la passata
            # fine (un quarto del blocco fine, a piena risoluzione) per
            # l'ultimo. Dentro quella portata chi viene dopo lo vede da solo, e
            # il grossolano aggiunge soltanto il suo rumore: misurato sulle
            # bande fotografiche del Fra Galgario (ampiezza 26-33 px, portata
            # 64), la mediana sulle finestre comuni peggiorava da 0.57 a 0.65
            # px (UVF) e da 0.21 a 0.26 (NIR950). Oltre la portata e' l'unico a
            # vederla: SWIR e radiografie, 104-131 px.
            portata_dopo = (portata_fine if n_liv == len(livelli) - 1
                            else blocco_g * 0.25 * livelli[n_liv + 1] * f_f)
            ampiezza = (info_g or {}).get('ampiezza_max_px_full')
            ultimo = n_liv == len(livelli) - 1
            if (campo_g is not None
                    and (campo_grossolano_tra_le_passate == 'auto' or not ultimo)
                    and ampiezza is not None and ampiezza <= portata_dopo):
                info_g.update(
                    applicato=False, portata_passata_fine_px_full=float(portata_dopo),
                    motivo=(f"ampiezza massima {ampiezza:.0f} px, dentro la portata della "
                            f"passata successiva ({portata_dopo:.0f} px): basta quella"))
                if verbose:
                    print(f"  [grossolano] non applicato: {info_g['motivo']}")
                campo_g = None
            # Un livello che non e' l'ultimo deve anche essere sostenuto da
            # piu' blocchi: almeno l'1% di quelli concordi oltre la portata del
            # livello dopo. Con il solo massimo bastava un blocco sbagliato:
            # sul NIR850 del Fra Galgario uno su 924 misurava 195 px (99-esimo
            # percentile 21) e il livello, applicato, portava le finestre degli
            # angoli oltre 5 px dal 2% al 10%. Sull'UV dell'affresco, dove la
            # deformazione e' vera, i blocchi oltre portata sono il 12%.
            p99 = (info_g or {}).get('ampiezza_p99_blocchi_px_full')
            if (campo_g is not None and not ultimo and p99 is not None
                    and p99 <= portata_dopo):
                info_g.update(
                    applicato=False, portata_passata_fine_px_full=float(portata_dopo),
                    motivo=(f"meno dell'1% dei blocchi oltre la portata del livello "
                            f"successivo ({portata_dopo:.0f} px; 99-esimo percentile "
                            f"{p99:.0f} px): l'ampiezza massima di {ampiezza:.0f} px "
                            f"viene da blocchi isolati"))
                if verbose:
                    print(f"  [grossolano] non applicato: {info_g['motivo']}")
                campo_g = None
            if campo_g is not None:
                T = T.compose_residual(np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                                       delta_field=campo_g * f_f,
                                       delta_grid_step=griglia * f_f)
            registro.append(dict(info_g or {}, riduzione=rid))
        info['campo_grossolano'] = registro[-1]
        if len(registro) > 1:
            info['campo_grossolano_livelli'] = registro

    # ---- passata 2: residuo sulla moving pre-allineata -------------------
    if verbose:
        print("\n[stima 2/2] residuo su moving pre-allineata")
    T_prev = T.at_factor(f_f, f_m)
    mov_aligned = T_prev.apply_to_image(img_moving, img_fixed.shape[:2])
    gm2 = gradient_magnitude(mov_aligned)

    del mov_aligned
    # La moving pre-allineata vive ormai nel sistema della fixed: niente offset,
    # e gli spostamenti misurati sono in pixel dell'anteprima FIXED.
    pts2 = match_blocks(gf, gm2, block=refine_block, step=refine_step,
                        roi_frac=roi_frac, mask=mask, verbose=verbose)
    del gf, gm2
    if len(pts2) < 12:
        info['nonrigid'] = {'applicato': False, 'motivo': 'troppi pochi blocchi affidabili'}
        return T, info

    # Correzione affine residua (di solito piccolissima: recupera l'effetto di
    # aver stimato la prima affine con blocchi grandi e pesi diversi).
    dM, inl2, res2 = fit_affine_robust(pts2, min_conf=min_conf, verbose=verbose)
    # dM e' una mappa fixed -> fixed (residuo misurato dopo il pre-allineamento),
    # quindi entrambi i fattori sono quelli della fixed.
    T = T.compose_residual(_delta_at_full(dM, f_f))

    # L'affine si stima sugli inlier GLOBALI (una deformazione locale non deve
    # trascinarla), ma il campo si costruisce sui blocchi coerenti con i propri
    # vicini: vedi blocchi_coerenti per il perche', misurato sul bordo della tela.
    pts_campo, res_campo = blocchi_coerenti(pts2, dM, refine_step, inl2,
                                            min_conf=min_conf, verbose=verbose)

    grid_step = grid_step or refine_step

    if smooth_px is None:
        candidates = [refine_step * k for k in (0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)]
        smooth_px, cv_stats, cv_table = select_smoothing(
            pts_campo, res_campo, img_fixed.shape[:2], grid_step, candidates,
            verbose=verbose)
        info['smoothing_crossval'] = cv_table
    else:
        cv_stats = field_crossval_gain(pts_campo, res_campo, img_fixed.shape[:2],
                                       grid_step, smooth_px)

    # Ordine della media locale: si sceglie come il sigma, misurando fuori
    # campione invece di decidere a priori. L'ordine 1 aiuta dove i blocchi
    # stanno da un lato solo, ma costa qualche grado di liberta' in piu' e non
    # conviene sempre.
    ordine = 0
    cv_ordine1 = [g for g in (field_crossval_gain(pts_campo, res_campo,
                                                  img_fixed.shape[:2], grid_step,
                                                  smooth_px, seed=sd, ordine=1)
                              for sd in (0, 1, 2)) if g.get('usable')]
    if cv_ordine1 and cv_stats.get('usable'):
        residuo0 = cv_stats['residuo_dopo_px']
        residuo1 = float(np.mean([g['residuo_dopo_px'] for g in cv_ordine1]))
        if residuo1 < residuo0:
            ordine = 1
            cv_stats = dict(cv_stats, residuo_dopo_px=residuo1,
                            guadagno_pct=float(np.mean([g['guadagno_pct'] for g in cv_ordine1])))
        if verbose:
            print(f"  [ordine] media locale fuori campione: ordine 0 -> {residuo0:.3f} px, "
                  f"ordine 1 -> {residuo1:.3f} px; scelto ordine {ordine}")
    info['ordine_media_locale'] = int(ordine)
    info['nonrigid_crossval'] = cv_stats
    if verbose and cv_stats.get('usable'):
        print(f"  [cross-val] residuo fuori campione: {cv_stats['residuo_prima_px'] * f_f:.2f} px "
              f"-> {cv_stats['residuo_dopo_px'] * f_f:.2f} px a piena risoluzione "
              f"({cv_stats['guadagno_pct']:+.0f}%), sigma scelto {smooth_px * f_f:.0f} px")

    if not cv_stats.get('usable') or cv_stats['guadagno_pct'] < 10.0:
        info['nonrigid'] = {'applicato': False,
                            'motivo': 'la cross-validation non mostra un guadagno reale: '
                                      'il residuo e\' rumore di misura, non deformazione'}
        return T, info

    field_prev = build_dense_field(pts_campo, res_campo, img_fixed.shape[:2], grid_step,
                                   smooth_px, mask=mask, ordine=ordine, verbose=verbose)
    # field_prev e' in pixel dell'anteprima FIXED: si porta a piena risoluzione
    # con il fattore della fixed, e compose_residual lo converte poi in pixel
    # della moving applicando la parte lineare della trasformazione.
    T = T.compose_residual(np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                           delta_field=field_prev * f_f,
                           delta_grid_step=grid_step * f_f)
    info['nonrigid'] = {'applicato': True,
                        'grid_step_px_full': float(grid_step * f_f),
                        'smooth_px_full': float(smooth_px * f_f),
                        'blocchi': int(len(pts_campo)),
                        'inlier_affine': int(len(inl2))}
    return T, info


def _delta_at_full(dM_prev, factor):
    """
    Converte una correzione affine misurata a scala anteprima in una correzione
    valida a piena risoluzione. Una correzione e' uno spostamento, quindi si
    converte come una trasformazione, non come dei semplici parametri scalati.
    """
    return Transform.from_factor(dM_prev, factor).M
