"""
STRATO BASSO dell'I/O: la via veloce per i TIFF NON COMPRESSI.

Non si usa direttamente. Sopra c'e' image_io.py, che e' l'unico a chiamare questo
modulo: prova questa strada e, se il file non e' un TIFF non compresso, ripiega
sulla lettura con Pillow. Chi deve aprire un'immagine usa image_io, non questo.

Qui si manipola il formato a livello di byte (si leggono e si scrivono i tag TIFF
con struct), quindi e' la parte piu' delicata del progetto ed e' tenuta separata
per poterla leggere e verificare da sola.

I/O per TIFF molto grandi (centinaia di MB, decine di migliaia di pixel di lato)
SENZA dipendenze extra: solo numpy + Pillow (per leggere i tag) + cv2.

Perche' non tifffile: sulla macchina di destinazione tifffile non e' installato e
non c'e' garanzia di poterlo installare. Ma non serve: i TIFF prodotti dallo
scanner/fotocamera in questo progetto sono NON COMPRESSI e con i dati dei pixel
in un blocco contiguo del file. Un TIFF non compresso e' letteralmente l'array
dei pixel scritto di fila dopo un header: si puo' quindi mappare in memoria con
np.memmap e leggerne qualsiasi sotto-regione senza caricare nulla in RAM.

Il sistema operativo carica solo le pagine effettivamente toccate, quindi:
- leggere un tile 2048x2048 costa ~12 MB di RAM, non 900 MB;
- leggere un'anteprima ridotta costa quanto un blocco di righe alla volta.

Limite noto: se un giorno lavorerai con TIFF COMPRESSI (LZW/JPEG/Deflate), questo
modulo solleva un errore esplicito invece di produrre risultati sbagliati. In quel
caso o si ri-esporta il file non compresso, oppure si installa tifffile.
"""

import os
import struct
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disattiva il guard anti "decompression bomb"

# Tag TIFF usati
_TAG_WIDTH = 256
_TAG_LENGTH = 257
_TAG_BITS = 258
_TAG_COMPRESSION = 259
_TAG_PHOTOMETRIC = 262
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTECOUNTS = 279
_TAG_PLANAR = 284
_TAG_TILE_WIDTH = 322


class TiffInfo:
    """Metadati di un TIFF non compresso, sufficienti per mapparlo con numpy."""

    def __init__(self, path):
        self.path = path
        with Image.open(path) as im:
            t = im.tag_v2
            self.width = int(t[_TAG_WIDTH])
            self.height = int(t[_TAG_LENGTH])
            self.samples = int(t.get(_TAG_SAMPLES, 1))
            bits = t.get(_TAG_BITS, (8,))
            bits = tuple(bits) if isinstance(bits, (tuple, list)) else (int(bits),)
            compression = int(t.get(_TAG_COMPRESSION, 1))
            planar = int(t.get(_TAG_PLANAR, 1))
            offsets = t.get(_TAG_STRIP_OFFSETS, None)
            counts = t.get(_TAG_STRIP_BYTECOUNTS, None)
            self.rows_per_strip = int(t.get(_TAG_ROWS_PER_STRIP, self.height))
            self.is_tiled = _TAG_TILE_WIDTH in t
            self.photometric = int(t.get(_TAG_PHOTOMETRIC, 2))

        if compression != 1:
            raise ValueError(
                f"{os.path.basename(path)}: TIFF compresso (Compression={compression}). "
                "Questo modulo legge solo TIFF non compressi. Ri-esporta il file senza "
                "compressione, oppure installa tifffile e usa quello."
            )
        if planar != 1:
            raise ValueError(f"{os.path.basename(path)}: PlanarConfiguration={planar} non supportata "
                             "(servono i campioni interlacciati, RGBRGB...).")
        if self.is_tiled:
            raise ValueError(f"{os.path.basename(path)}: TIFF a tile non supportato da questo modulo.")
        if len(set(bits)) != 1:
            raise ValueError(f"{os.path.basename(path)}: BitsPerSample eterogenei {bits}.")

        bits_per_sample = bits[0]
        if bits_per_sample == 8:
            self.dtype = np.uint8
        elif bits_per_sample == 16:
            self.dtype = np.uint16
        else:
            raise ValueError(f"{os.path.basename(path)}: BitsPerSample={bits_per_sample} non supportato "
                             "(supportati: 8 e 16 bit).")

        offsets = tuple(offsets) if isinstance(offsets, (tuple, list)) else (int(offsets),)
        counts = tuple(counts) if isinstance(counts, (tuple, list)) else (int(counts),)

        # I dati devono formare un blocco contiguo: e' il caso normale per un TIFF
        # non compresso scritto in un colpo solo. Se cosi' non fosse, meglio
        # accorgersene subito che leggere pixel sbagliati.
        expected = offsets[0]
        for off, cnt in zip(offsets, counts):
            if off != expected:
                raise ValueError(f"{os.path.basename(path)}: strip non contigui, memmap non applicabile.")
            expected = off + cnt

        self.data_offset = int(offsets[0])
        self.data_bytes = int(sum(counts))

        expected_bytes = self.height * self.width * self.samples * (bits_per_sample // 8)
        if self.data_bytes < expected_bytes:
            raise ValueError(f"{os.path.basename(path)}: dati troncati "
                             f"({self.data_bytes} byte, attesi {expected_bytes}).")

    @property
    def shape(self):
        if self.samples == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.samples)

    def __repr__(self):
        return (f"TiffInfo({os.path.basename(self.path)}, {self.width}x{self.height}, "
                f"{self.samples}ch, {np.dtype(self.dtype).name}, offset={self.data_offset})")


def open_memmap(path, mode='r'):
    """
    Mappa in memoria i pixel di un TIFF non compresso.

    Restituisce (array_memmap, info). L'array si usa come un normale array numpy
    (slicing, fancy indexing), ma i dati vengono letti dal disco su richiesta:
    `arr[1000:1200, 500:800]` legge davvero solo quella regione.

    ATTENZIONE: fare arr.astype(...) o operazioni su TUTTO l'array materializza
    l'intera immagine in RAM. Lavora sempre per sotto-regioni.
    """
    info = TiffInfo(path)
    arr = np.memmap(path, dtype=info.dtype, mode=mode,
                    offset=info.data_offset, shape=info.shape)
    return arr, info


def read_downsampled(path, target_long_side=2000, to_gray=True, block_rows=2048):
    """
    Legge una versione ridotta dell'immagine leggendo il file a blocchi di righe.

    La riduzione avviene per fattore INTERO (area averaging con INTER_AREA), cosi'
    la relazione geometrica fra anteprima e piena risoluzione e' esatta e nota:

        coord_full = factor * coord_preview + (factor - 1) / 2

    (il +(factor-1)/2 e' il centro del blocco di pixel mediati; ignorarlo
    introdurrebbe uno spostamento sistematico di qualche pixel a piena
    risoluzione, che e' esattamente l'ordine di grandezza dell'errore che
    stiamo cercando di correggere).

    Args:
        path: TIFF non compresso
        target_long_side: lato lungo desiderato (approssimato per difetto dal
            fattore intero piu' vicino)
        to_gray: converte in grayscale mentre legge (meta' memoria, ed e' cio'
            che serve alle metriche)
        block_rows: righe di output elaborate per volta (controlla il picco RAM)

    Returns:
        (img_small, factor) con factor = fattore intero di riduzione.
    """
    arr, info = open_memmap(path)
    h, w = info.height, info.width

    factor = max(1, int(max(h, w) // max(1, target_long_side)))
    out_h, out_w = h // factor, w // factor

    if info.samples == 1 or to_gray:
        out = np.empty((out_h, out_w), dtype=np.uint8 if info.dtype == np.uint8 else np.uint16)
    else:
        out = np.empty((out_h, out_w, info.samples), dtype=arr.dtype)

    for y_out in range(0, out_h, block_rows):
        n_out = min(block_rows, out_h - y_out)
        y0 = y_out * factor
        block = np.asarray(arr[y0:y0 + n_out * factor, :out_w * factor])
        if block.ndim == 3 and to_gray:
            block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        out[y_out:y_out + n_out] = cv2.resize(block, (out_w, n_out), interpolation=cv2.INTER_AREA)
        del block

    del arr
    return out, factor


# ---------------------------------------------------------------------------
# Scrittura
# ---------------------------------------------------------------------------

def create_tiff_memmap(path, height, width, samples=3, dtype=np.uint8,
                       sequenziale=False):
    """
    Crea un TIFF non compresso vuoto e restituisce un memmap SCRIVIBILE sui suoi
    pixel: si riempie tile per tile senza mai avere l'immagine intera in RAM.

    Il file viene scritto come: header (8 byte) + IFD + dati pixel. Sono gli
    stessi tag che hanno i file sorgente di questo progetto, quindi il risultato
    si apre in Photoshop/ImageJ/GIMP come gli originali.

    Returns:
        memmap di shape (H, W) o (H, W, samples). Chiamare .flush() a fine
        scrittura (o cancellare il riferimento) per garantire il salvataggio.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("Supportati solo uint8 e uint16 in scrittura.")
    bits = dtype.itemsize * 8
    data_bytes = height * width * samples * dtype.itemsize
    if data_bytes >= 4 * 1024 ** 3:
        raise ValueError("Immagine >= 4 GB: servirebbe BigTIFF, non supportato da questo writer.")

    photometric = 2 if samples == 3 else 1

    # Tag (tag, tipo, count, valore/offset). Tipi: 3=SHORT, 4=LONG, 5=RATIONAL.
    # Le voci con piu' valori di quelli che stanno in 4 byte (BitsPerSample con
    # 3 canali, le RATIONAL) puntano a dati scritti in coda all'IFD.
    entries = []
    extra = b''           # blocco dati extra, posizionato subito dopo l'IFD
    # Le voci sono 11 in entrambi i casi (BitsPerSample c'e' sempre: in linea
    # con un canale, come puntatore con piu' canali). Fino al 4 ottobre qui
    # c'era 10 per un canale solo: l'intestazione diceva che i pixel
    # cominciavano 10 byte prima di dove venivano scritti, e chi leggeva vedeva
    # l'immagine spostata a destra di 10 byte, cioe' 5 px a 16 bit (tutte le
    # radiografie allineate) e 10 px a 8 bit. Misurato: +5.00 px esatti.
    n_entries = 11
    ifd_offset = 8
    ifd_size = 2 + n_entries * 12 + 4
    extra_offset = ifd_offset + ifd_size

    def add_extra(raw):
        nonlocal extra
        off = extra_offset + len(extra)
        extra += raw
        return off

    if samples > 1:
        bits_off = add_extra(struct.pack('<' + 'H' * samples, *([bits] * samples)))
    xres_off = add_extra(struct.pack('<LL', 300, 1))
    yres_off = add_extra(struct.pack('<LL', 300, 1))

    # I dati pixel iniziano dopo IFD + extra, allineati a 8 byte (numpy.memmap
    # non lo richiede, ma allineare evita letture disallineate piu' lente).
    data_offset = (extra_offset + len(extra) + 7) // 8 * 8

    entries.append((_TAG_WIDTH, 4, 1, width))
    entries.append((_TAG_LENGTH, 4, 1, height))
    if samples > 1:
        entries.append((_TAG_BITS, 3, samples, bits_off))
    else:
        entries.append((_TAG_BITS, 3, 1, bits))
    entries.append((_TAG_COMPRESSION, 3, 1, 1))
    entries.append((_TAG_PHOTOMETRIC, 3, 1, photometric))
    entries.append((_TAG_STRIP_OFFSETS, 4, 1, data_offset))
    entries.append((_TAG_SAMPLES, 3, 1, samples))
    entries.append((_TAG_ROWS_PER_STRIP, 4, 1, height))
    entries.append((_TAG_STRIP_BYTECOUNTS, 4, 1, data_bytes))
    entries.append((282, 5, 1, xres_off))
    entries.append((283, 5, 1, yres_off))
    entries.sort(key=lambda e: e[0])

    buf = bytearray()
    buf += b'II' + struct.pack('<HL', 42, ifd_offset)
    buf += struct.pack('<H', len(entries))
    for tag, typ, count, value in entries:
        if typ == 3 and count == 1:
            buf += struct.pack('<HHLHH', tag, typ, count, value, 0)
        else:
            buf += struct.pack('<HHLL', tag, typ, count, value)
    buf += struct.pack('<L', 0)  # nessun IFD successivo
    buf += extra
    # Controllo esplicito: se l'intestazione sconfina oltre l'inizio dichiarato
    # dei pixel, il file sarebbe letto spostato. Meglio fermarsi.
    if len(buf) > data_offset or len(entries) != n_entries:
        raise RuntimeError(f"intestazione TIFF incoerente: {len(buf)} byte con i pixel "
                           f"dichiarati a {data_offset}, {len(entries)} voci su {n_entries}")
    buf += b'\x00' * (data_offset - len(buf))

    if sequenziale:
        # Chi scrive dall'alto in basso non ha bisogno dell'accesso sparso, e
        # cosi' si evita il truncate: su NTFS pre-dimensionare un file da
        # 900 MB costa 14-42 s (misurato), perche' il sistema lo riempie di
        # zeri che poi riscriviamo comunque. In piu' la riscrittura via memmap
        # sporca pagine che il sistema riversa su disco in ordine sparso.
        # Misurato sullo stesso file da 900 MB: 77-125 s con truncate+memmap
        # contro 4-27 s scrivendo in sequenza.
        f = open(path, 'wb')
        f.write(bytes(buf))
        return ScrittoreRighe(f, height, width, samples, dtype)

    with open(path, 'wb') as f:
        f.write(bytes(buf))
        f.truncate(data_offset + data_bytes)

    shape = (height, width) if samples == 1 else (height, width, samples)
    return np.memmap(path, dtype=dtype, mode='r+', offset=data_offset, shape=shape)


class ScrittoreRighe:
    """
    Scrive un TIFF una banda di righe alla volta, in ordine, senza memmap.

    Va usato solo da chi produce le righe dall'alto in basso: l'intestazione
    dichiara gia' quante righe ci saranno, quindi saltarne o scriverne di piu'
    produrrebbe un file corrotto. `chiudi()` verifica che il conto torni.
    """

    def __init__(self, f, height, width, samples, dtype):
        self.f = f
        self.height, self.width = height, width
        self.samples, self.dtype = samples, np.dtype(dtype)
        self.scritte = 0

    def scrivi(self, righe):
        """Accoda una banda di righe, come array (n, W) o (n, W, samples)."""
        a = np.ascontiguousarray(righe, dtype=self.dtype)
        attesa = (a.shape[0], self.width) if self.samples == 1 else \
                 (a.shape[0], self.width, self.samples)
        if a.shape != attesa:
            raise ValueError(f"Banda di forma {a.shape}, attesa {attesa}")
        if self.scritte + a.shape[0] > self.height:
            raise ValueError("Scritte piu' righe di quelle dichiarate nell'intestazione")
        self.f.write(a.tobytes())
        self.scritte += a.shape[0]

    def chiudi(self):
        if self.scritte != self.height:
            self.f.close()
            raise ValueError(f"Scritte {self.scritte} righe su {self.height}: "
                             f"il file sarebbe incompleto")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, tipo, valore, traccia):
        if tipo is None:
            self.chiudi()
        else:
            self.f.close()
