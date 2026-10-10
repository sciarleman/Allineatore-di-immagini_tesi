"""
Accesso uniforme alle immagini, qualunque sia il formato.

E' LO STRATO ALTO: chi deve aprire un'immagine chiama questo modulo, non
tiff_memmap. Sotto c'e' appunto tiff_memmap.py, che sa mappare in memoria un TIFF
NON COMPRESSO ed e' la via veloce (un tile da 2048x2048 costa ~12 MB di RAM
invece di 900 MB). Ma le riprese di un progetto reale arrivano in formati misti:
TIFF compressi LZW/ZIP, JPEG, PNG, magari di dimensioni diverse fra loro.

Questo modulo espone un'unica interfaccia e sceglie da solo la strada:

    apri()            -> array + info, via memmap se possibile, altrimenti in RAM
    info()            -> dimensioni e tipo senza leggere i pixel
    leggi_ridotta()   -> anteprima ridotta di un fattore intero
    copia_tiff_temporanea() / rilascia_copia()
                      -> per la durata dell'analisi, un formato compresso si
                         legge da una copia TIFF non compressa, che poi sparisce

L'array restituito si usa allo stesso modo nei due casi (slicing normale), quindi
il resto della pipeline non deve sapere quale strada e' stata presa. Quando si
ricade sulla lettura in RAM il modulo lo dice, perche' cambia il picco di memoria
e quindi cosa puo' girare in parallelo.

Convenzione colore: si usa sempre PILLOW, quindi i canali sono in ordine RGB
(non BGR come cv2.imread). Tutto il resto della pipeline assume RGB.
"""

import atexit
import gc
import os
import shutil
import tempfile
import time

import numpy as np
import cv2
from PIL import Image

import tiff_memmap

Image.MAX_IMAGE_PIXELS = None

# Le copie TIFF temporanee: percorso dell'originale -> percorso della copia.
# Chi apre un'immagine continua a passare il percorso dell'originale; le
# funzioni di questo modulo leggono dalla copia, se c'e'.
_COPIE = {}
_CARTELLA_COPIE = None
_PREFISSO_COPIE = 'allineatore_tiff_'


def _chiave(path):
    return os.path.normcase(os.path.abspath(path))


def _da_leggere(path):
    return _COPIE.get(_chiave(path), path)


class InfoImmagine:
    """Dimensioni, canali e tipo di un'immagine, senza leggerne i pixel."""

    def __init__(self, path):
        self.path = path
        # Da dove si leggono i pixel: l'originale, o la sua copia temporanea.
        self.path_lettura = _da_leggere(path)
        self.memmappabile = False
        self.motivo_fallback = None
        try:
            t = tiff_memmap.TiffInfo(self.path_lettura)
            self.height, self.width = t.height, t.width
            self.samples, self.dtype = t.samples, np.dtype(t.dtype)
            self.memmappabile = True
            return
        except Exception as err:
            testo = str(err).split('\n')[0]
            # Un file non-TIFF non ha i tag TIFF: l'eccezione che ne deriva e'
            # un dettaglio interno, non un'informazione utile a chi legge.
            self.motivo_fallback = (testo if 'TIFF' in testo or 'strip' in testo
                                    else "formato non mappabile in memoria")

        with Image.open(path) as im:
            self.width, self.height = im.size
            modo = im.mode
            if self.motivo_fallback == "formato non mappabile in memoria" and im.format:
                self.motivo_fallback = f"formato {im.format}, non mappabile in memoria"
        if modo in ('L', 'P', '1'):
            self.samples, self.dtype = 1, np.dtype(np.uint8)
        elif modo in ('I;16', 'I;16B', 'I;16L', 'I'):
            self.samples, self.dtype = 1, np.dtype(np.uint16)
        elif modo in ('RGB', 'RGBA', 'CMYK', 'YCbCr'):
            self.samples, self.dtype = 3, np.dtype(np.uint8)
        elif modo == 'F':
            self.samples, self.dtype = 1, np.dtype(np.float32)
        else:
            self.samples, self.dtype = 3, np.dtype(np.uint8)

    @property
    def shape(self):
        return (self.height, self.width) if self.samples == 1 else \
               (self.height, self.width, self.samples)

    @property
    def megapixel(self):
        return self.width * self.height / 1e6

    def __repr__(self):
        via = "memmap" if self.memmappabile else "RAM"
        return (f"{os.path.basename(self.path)} {self.width}x{self.height} "
                f"{self.samples}ch {self.dtype.name} [{via}]")


def info(path):
    """Metadati senza leggere i pixel."""
    return InfoImmagine(path)


def a_8bit(img, percentili=(0.5, 99.5)):
    """
    Porta un'immagine a 8 bit per la visualizzazione.

    Serve perche' una ripresa a 16 bit (radiografie, sensori scientifici) non si
    puo' convertire con un semplice .astype(np.uint8): quello taglia i valori
    modulo 256 e produce un'immagine irriconoscibile, a bande. Qui si mappa
    invece l'intervallo utile dei valori su 0-255.

    Si usano i percentili e non min/max perche' un solo pixel bruciato o un
    pixel morto schiaccerebbe tutto il resto in una fetta minuscola della scala.

    Solo per le immagini di controllo: la misura lavora sui dati originali.
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    lo, hi = np.percentile(a, percentili)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _pil_ad_array(im):
    """Converte un'immagine Pillow in array numpy con canali RGB o grigio."""
    if im.mode in ('P', '1', 'RGBA', 'CMYK', 'YCbCr'):
        im = im.convert('RGB')
    arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def apri(path, verbose=True):
    """
    Apre un'immagine per la lettura.

    Returns:
        (array, InfoImmagine). L'array e' un np.memmap (lettura su richiesta dal
        disco, RAM trascurabile) se il file e' un TIFF non compresso; altrimenti
        e' un normale array con TUTTA l'immagine in RAM.

    Chi chiama non deve distinguere i due casi: l'uso e' identico. Ma se
    `info.memmappabile` e' False, quell'immagine occupa la sua dimensione piena
    in memoria finche' non viene liberata.
    """
    meta = InfoImmagine(path)
    if meta.memmappabile:
        arr, _ = tiff_memmap.open_memmap(meta.path_lettura)
        return arr, meta

    costo_gb = meta.width * meta.height * meta.samples * meta.dtype.itemsize / 1024 ** 3
    # L'avviso serve solo quando il costo in memoria e' rilevante: per
    # un'immagine piccola sarebbe solo rumore a schermo.
    if verbose and costo_gb >= 0.25:
        print(f"  [lettura in RAM] {os.path.basename(path)} ({meta.motivo_fallback}): "
              f"caricata per intero, ~{costo_gb:.2f} GB. "
              f"Un TIFF non compresso userebbe la via veloce a memmap.")
    with Image.open(path) as im:
        arr = _pil_ad_array(im)
    return arr, meta


def anteprima_veloce(path, lato_lungo_target=2400, in_grigio=False):
    """
    Anteprima per lo SCHERMO, presa a passo invece che mediando.

    leggi_ridotta fa una media d'area esatta e per farlo legge ogni pixel del
    file: e' giusto per la stima, dove un errore sub-pixel conta, ma per
    un'anteprima da guardare e' quasi un secondo di lavoro buttato per ogni
    gigabyte. Qui si prende un pixel ogni `passo`, quindi si legge 1/passo del
    file. Misurato sulle immagini di questo progetto: 3.75 s -> 0.04 s per il
    visibile da 0.94 GB, 1.18 s -> 0.02 s per la radiografia. E' il motivo per
    cui le finestre della maschera e dei punti si aprivano lente.

    La relazione geometrica resta semplice ed esatta, senza il +(passo-1)/2
    della media d'area, perche' il pixel dell'anteprima E' un pixel vero:

        coord_piena = passo * coord_anteprima

    Il bersaglio e' piu' grande di quanto serva a schermo apposta: sovracampionare
    attenua la scalettatura che il prendere a passo introduce sui dettagli fini,
    come le craquelure.

    Returns:
        (immagine, passo intero)
    """
    meta = InfoImmagine(path)
    passo = max(1, int(max(meta.height, meta.width) // max(1, lato_lungo_target)))
    if meta.memmappabile:
        arr, _ = tiff_memmap.open_memmap(meta.path_lettura)
        out = np.ascontiguousarray(arr[::passo, ::passo])
        del arr
    else:
        pieno, _ = apri(path, verbose=False)
        out = np.ascontiguousarray(pieno[::passo, ::passo])
    if in_grigio and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    return out, passo


def leggi_ridotta(path, lato_lungo_target=2000, in_grigio=True, verbose=True):
    """
    Legge una versione ridotta dell'immagine, per fattore INTERO.

    La riduzione per fattore intero rende esatta la relazione fra anteprima e
    piena risoluzione (vedi transform_model.py):

        coord_piena = fattore * coord_anteprima + (fattore - 1) / 2

    Per i TIFF non compressi la lettura avviene a bande, senza mai caricare tutto.
    Per gli altri formati si carica (una volta) e si riduce; sui JPEG si sfrutta
    il ridimensionamento veloce in fase di decodifica.

    Returns:
        (immagine_ridotta, fattore)
    """
    meta = InfoImmagine(path)
    if meta.memmappabile:
        return tiff_memmap.read_downsampled(meta.path_lettura, target_long_side=lato_lungo_target,
                                        to_gray=in_grigio)

    fattore = max(1, int(max(meta.height, meta.width) // max(1, lato_lungo_target)))
    with Image.open(path) as im:
        if im.format == 'JPEG':
            # draft() riduce gia' in fase di decodifica JPEG: molto piu' veloce
            # e molto piu' leggero che decodificare a piena risoluzione.
            im.draft('L' if in_grigio else 'RGB',
                     (meta.width // fattore, meta.height // fattore))
        arr = _pil_ad_array(im)

    if in_grigio and arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # draft() puo' aver gia' ridotto l'immagine, di 2, 4 o 8: lo si ricava dalle
    # dimensioni reali dell'array. Il rapporto fra le altezze non e' esattamente
    # quell'intero (draft arrotonda per eccesso le righe che avanzano), quindi
    # si arrotonda.
    h, w = arr.shape[:2]
    fattore_gia_applicato = max(1, int(round(meta.height / h)))
    fattore_residuo = max(1, int(fattore / fattore_gia_applicato))
    if fattore_residuo > 1:
        out_h, out_w = h // fattore_residuo, w // fattore_residuo
        arr = cv2.resize(arr[:out_h * fattore_residuo, :out_w * fattore_residuo],
                         (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Il fattore restituito e' l'INTERO per cui si e' ridotto, non il rapporto
    # fra le altezze. Quando l'altezza non e' multipla del fattore le righe che
    # avanzano in fondo si scartano, ma ogni pixel ridotto resta la media di un
    # blocco fattore x fattore che parte dall'origine: la relazione fra le
    # coordinate e' quella del fattore intero. Qui si restituiva altezza /
    # altezza ridotta (8003 / 4001 = 2.00025), cioe' una scala sbagliata dello
    # 0.0125%, che non si cancella quando riferimento e immagine da allineare
    # hanno resti diversi. Misurato sugli stessi pixel letti da TIFF (dove il
    # fattore e' sempre stato l'intero) e da PNG, riferimento 6001x8003 e
    # immagine 6000x8000: residuo mediano 0.22 contro 0.73 px a fattore 2, 0.56
    # contro 1.95 a fattore 4, con le due affini distanti 1.25 e 3.75 px
    # all'angolo opposto all'origine.
    return arr, float(fattore_gia_applicato * fattore_residuo)


def _cartella_copie():
    """La cartella delle copie di QUESTO processo, creata alla prima richiesta.
    Passando si tolgono quelle lasciate da un processo finito male."""
    global _CARTELLA_COPIE
    if _CARTELLA_COPIE is None:
        radice = tempfile.gettempdir()
        for nome in os.listdir(radice):
            vecchia = os.path.join(radice, nome)
            try:
                if (nome.startswith(_PREFISSO_COPIE) and os.path.isdir(vecchia)
                        and time.time() - os.path.getmtime(vecchia) > 86400):
                    shutil.rmtree(vecchia, ignore_errors=True)
            except OSError:
                pass
        _CARTELLA_COPIE = tempfile.mkdtemp(prefix=_PREFISSO_COPIE)
        atexit.register(_rilascia_tutto)
    return _CARTELLA_COPIE


def copia_tiff_temporanea(path, verbose=True, righe=1024):
    """
    Per la durata dell'analisi, fa leggere `path` da una copia TIFF non compressa.

    Un PNG, un JPEG o un TIFF compresso non si possono leggere a pezzi: per
    averne un ritaglio bisogna decomprimerli tutti. Senza copia restano quindi
    per intero in memoria per tutta l'elaborazione, riferimento e immagine
    insieme, e si decomprimono una volta per l'anteprima e una per la piena
    risoluzione. Misurato su immagini da 593 Mpx in PNG, a precisione massima:
    11 GB di picco contro 7.7 con gli stessi pixel in TIFF non compresso, e
    circa 2 minuti in piu' per immagine.

    Qui l'immagine si decomprime UNA volta, si scrive a bande in un TIFF nella
    cartella temporanea del sistema e da quel momento apri(), info() e
    leggi_ridotta() la leggono da li', mappata in memoria, senza che chi chiama
    cambi niente: il percorso resta quello dell'originale, e cosi' i nomi e il
    formato dei file prodotti. I pixel sono gli stessi (stessa conversione di
    _pil_ad_array). La copia si toglie con rilascia_copia(), e comunque alla
    chiusura del programma: non e' un risultato e non resta su disco.

    Non si fa, e si continua a leggere in memoria come prima, se il file e'
    gia' mappabile, se supera i 4 GB (il nostro TIFF non li scrive) o se sul
    disco temporaneo non c'e' posto.

    Returns:
        il percorso della copia, o None se non e' stata fatta.
    """
    chiave = _chiave(path)
    if chiave in _COPIE:
        return _COPIE[chiave]
    meta = InfoImmagine(path)
    if meta.memmappabile:
        return None
    peso = meta.width * meta.height * meta.samples * meta.dtype.itemsize
    if peso >= 4 * 1024 ** 3 - 65536:
        return None
    copia = None
    try:
        cartella = _cartella_copie()
        if shutil.disk_usage(cartella).free < peso + 1024 ** 3:
            if verbose:
                print(f"  [copia TIFF non fatta] {os.path.basename(path)}: sul disco temporaneo "
                      f"non ci sono {peso / 1024 ** 3:.1f} GB liberi; si legge in memoria.")
            return None
        t0 = time.time()
        copia = os.path.join(cartella, f"{len(os.listdir(cartella)):03d}_"
                                       f"{os.path.splitext(os.path.basename(path))[0]}.tif")
        with Image.open(path) as im:
            if im.mode in ('P', '1', 'RGBA', 'CMYK', 'YCbCr'):
                im = im.convert('RGB')
            # In sequenza e non attraverso la mappatura in memoria: misurato
            # su un PNG da 593 Mpx, 1.7 GB di pixel, 10 s contro 235.
            scrittore = tiff_memmap.create_tiff_memmap(copia, meta.height, meta.width, samples=meta.samples,
                                                       dtype=meta.dtype, sequenziale=True)
            # A bande: Pillow tiene l'immagine decompressa una volta, e non se
            # ne fa una seconda copia intera per passarla a numpy.
            for y0 in range(0, meta.height, righe):
                y1 = min(meta.height, y0 + righe)
                banda = np.asarray(im.crop((0, y0, meta.width, y1)))
                if banda.ndim == 3 and banda.shape[2] > meta.samples:
                    banda = banda[:, :, :meta.samples]
                scrittore.scrivi(banda)
            scrittore.chiudi()
        _COPIE[chiave] = copia
        # Come per l'avviso di apri(): su un'immagine piccola sarebbe rumore.
        if verbose and peso >= 0.25 * 1024 ** 3:
            print(f"  [copia TIFF temporanea] {os.path.basename(path)}: {peso / 1024 ** 3:.2f} GB "
                  f"in {time.time() - t0:.0f} s; si toglie a fine analisi.")
        return copia
    except Exception as exc:
        # La copia e' un'ottimizzazione: se non riesce, l'immagine si legge
        # come prima.
        _COPIE.pop(chiave, None)
        gc.collect()
        try:
            if copia:
                os.remove(copia)
        except OSError:
            pass
        if verbose:
            print(f"  [copia TIFF non fatta] {os.path.basename(path)}: {exc}; si legge in memoria.")
        return None


def rilascia_copia(path):
    """Toglie la copia temporanea di `path`, se c'e'. Chi la stava leggendo
    deve aver gia' lasciato andare il suo array: su Windows un file ancora
    mappato non si cancella, e in quel caso resta per la pulizia finale."""
    copia = _COPIE.pop(_chiave(path), None)
    if copia is None:
        return
    gc.collect()
    try:
        os.remove(copia)
    except OSError:
        pass


def _rilascia_tutto():
    _COPIE.clear()
    gc.collect()
    if _CARTELLA_COPIE:
        shutil.rmtree(_CARTELLA_COPIE, ignore_errors=True)


def rilascia_pagine():
    """
    Restituisce al sistema le pagine dei file gia' letti.

    Le immagini si aprono mappate in memoria, e ogni pezzo letto resta contato
    nella memoria del processo finche' il processo non finisce, anche se non
    serve piu'. Misurato componendo quattro falsi colori: il programma usava di
    suo 0.6 GB, ma teneva 5.3 GB di file gia' letti, e la RAM libera del
    computer scendeva a 2 GB. Chiamata ogni tanto durante un ciclo di lettura,
    questa funzione dice a Windows che quelle pagine si possono riusare: restano
    nella cache dei file (rileggerle e' immediato), ma tornano disponibili.

    Sugli altri sistemi non fa niente: li' il sistema le recupera da solo.
    """
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(kernel32.GetCurrentProcess()))
    except Exception:
        pass


def crea_output(path, height, width, samples=3, dtype=np.uint8):
    """
    Crea il file di output e restituisce un array scrivibile.

    Per i TIFF si usa la scrittura a memmap (memoria costante). Per gli altri
    formati si prepara un array in RAM che verra' salvato da `salva_output`:
    JPEG e PNG non permettono la scrittura incrementale a tile.
    """
    est = os.path.splitext(path)[1].lower()
    if est in ('.tif', '.tiff'):
        return tiff_memmap.create_tiff_memmap(path, height, width, samples=samples, dtype=dtype), True
    shape = (height, width) if samples == 1 else (height, width, samples)
    return np.zeros(shape, dtype=dtype), False


def scrivi_immagine(path, img, parametri=None):
    """
    cv2.imwrite che funziona anche se nel percorso c'e' una lettera accentata.

    Su Windows cv2.imwrite passa il percorso a OpenCV come stringa a 8 bit, e
    con un carattere non ASCII non trova la cartella: restituisce False senza
    sollevare nulla. E' successo con una cartella di salvataggio il cui nome
    finiva con una "a" accentata: dopo 21 minuti di lotto le due immagini
    allineate non erano state
    scritte ("Impossibile scrivere") e le cartelle di diagnostica erano vuote
    senza nessun avviso, mentre JSON, .npy e TIFF, che scrive Python, c'erano.

    Con un percorso ASCII si chiama cv2.imwrite come sempre, quindi per chi
    non ha accenti nel percorso non cambia niente. Altrimenti si codifica in
    memoria con lo stesso encoder (cv2.imencode) e il file lo scrive Python;
    costa in piu' la RAM del file compresso, che per un PNG da 600 Mpx e'
    intorno a 1 GB.

    `img` e' in ordine BGR, come per cv2.imwrite. Returns: True se scritto.
    """
    parametri = list(parametri or [])
    if path.isascii():
        return bool(cv2.imwrite(path, img, parametri))
    try:
        ok, buf = cv2.imencode(os.path.splitext(path)[1], img, parametri)
        if not ok:
            return False
        with open(path, 'wb') as fh:
            fh.write(buf.data)
        return True
    except (cv2.error, OSError):
        return False


def salva_output(path, arr, memmappato):
    """Chiude/salva il file di output creato da `crea_output`."""
    if memmappato:
        arr.flush()
        return
    img = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not scrivi_immagine(path, img):
        raise IOError(f"Impossibile scrivere {path}")
