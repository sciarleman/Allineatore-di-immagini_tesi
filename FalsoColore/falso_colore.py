"""
Falso colore per l'imaging dei beni culturali: IRFC e UVFC.

Prende immagini GIA' ALLINEATE (l'output di align_paintings.py o
align_batch.py) e compone le due rappresentazioni standard.

    IRFC  infrarosso falso colore      IR -> R,  R(VIS) -> G,  G(VIS) -> B
    UVFC  ultravioletto falso colore   G(VIS) -> R,  B(VIS) -> G,  UV -> B

I canali scorrono in direzioni opposte, ed e' il motivo per cui le due immagini
si leggono in modo speculare: l'IRFC scarta il blu del visibile, l'UVFC scarta
il rosso.

Per l'UVFC si usa la ripresa in ULTRAVIOLETTO RIFLESSO (UVR), non la
fluorescenza (UVF): la fluorescenza si guarda com'e', non entra in un falso
colore, perche' il segnale che porta e' emissione del materiale e non riflessione
della banda.

Perche' e' utile anche per il nostro lavoro: un disallineamento residuo, in
falso colore, si vede come frangia colorata sui bordi, e l'occhio e' molto piu'
allenato a coglierla li' che in una scacchiera o in una mappa di vettori.

AVVERTENZA SULLA RADIOMETRIA. Il falso colore viene usato anche per
identificare i pigmenti confrontando le tinte con tavole di riferimento, e
QUEL confronto richiede riprese calibrate su uno standard colorimetrico in
posa. Qui non c'e' nessun bersaglio di calibrazione, quindi il programma non
altera i livelli: compone i canali cosi' come sono stati ripresi, e stampa
quanto i canali di ogni banda differiscono fra loro, in modo che si sappia
cosa si sta guardando. C'era un'opzione per stirare i livelli fra due
percentili, piu' leggibile ma non confrontabile con le tavole: e' stata tolta
perche' non serviva.

Esempi, dalla cartella del progetto:
    python FalsoColore\\falso_colore.py --vis immagini\\VIS.tif ^
        --ir Allineamenti\\NIR850_allineata.tif --out-dir risultati

    python FalsoColore\\falso_colore.py --vis immagini\\VIS.tif ^
        --uv Allineamenti\\UVR_allineata.tif
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import cv2

# image_io (lettura memmap, creazione dell'output, riduzioni) vive nel motore
# dell'allineatore, nella cartella accanto: si riusa invece di riscriverlo.
_MOTORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Allineatore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import formati as fmt
import image_io
import tiff_memmap
import validation as val


# ---------------------------------------------------------------------------
# Lettura delle bande
# ---------------------------------------------------------------------------

def _scala_dtype(dtype):
    """Valore di fondo scala per portare un tipo intero in [0, 1]."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def statistiche_banda(path, canale='media', lato=1600):
    """
    Guarda un'anteprima della banda e dice quanto i tre canali differiscono
    fra loro.

    Il numero conta: una ripresa infrarossa o ultravioletta e'
    monocromatica, ma spesso viene salvata come RGB. Se i tre canali sono
    praticamente uguali, ridurli alla media e' innocuo. Se invece differiscono
    parecchio -- succede con i sensori a matrice di Bayer, la cui sensibilita'
    fuori dal visibile non e' uguale nei tre filtri -- la scelta del canale
    cambia il risultato, e va fatta sapendo che lo si sta facendo.
    """
    prev, _ = image_io.leggi_ridotta(path, lato, in_grigio=False, verbose=False)
    fondo = _scala_dtype(prev.dtype)
    f = prev.astype(np.float32) / fondo
    if f.ndim == 3 and f.shape[2] >= 3:
        scarto = float(np.abs(f[:, :, 0] - f[:, :, 2]).mean())
        rid = _riduci(f, canale)
    else:
        scarto = 0.0
        rid = f if f.ndim == 2 else f[:, :, 0]
    return {
        'media': float(rid.mean()),
        'scarto_fra_canali': scarto,
    }


def _riduci(f, canale):
    """Porta una banda a canale singolo, in float [0, 1]."""
    if f.ndim == 2:
        return f
    if f.shape[2] == 1:
        return f[:, :, 0]
    if canale == 'media':
        return f[:, :, :3].mean(axis=2)
    return f[:, :, {'r': 0, 'g': 1, 'b': 2}[canale]]


# ---------------------------------------------------------------------------
# Composizione
# ---------------------------------------------------------------------------

# Per ogni tipo: da dove viene ciascun canale di uscita, in ordine R, G, B.
# 'banda' e' l'immagine aggiunta (IR oppure UV); ('vis', i) e' il canale i del
# visibile, con 0=R, 1=G, 2=B.
RICETTE = {
    'irfc': {
        'sorgenti': ['banda', ('vis', 0), ('vis', 1)],
        'descrizione': "IR -> R, R(VIS) -> G, G(VIS) -> B  (il blu del visibile non viene usato)",
    },
    'uvfc': {
        'sorgenti': [('vis', 1), ('vis', 2), 'banda'],
        'descrizione': "G(VIS) -> R, B(VIS) -> G, UV -> B  (il rosso del visibile non viene usato)",
    },
}


# ---------------------------------------------------------------------------
# Una striscia di righe, in due modi che danno lo stesso risultato
# ---------------------------------------------------------------------------
#
# Un falso colore e' solo uno SPOSTAMENTO DI CANALI: l'IR va nel rosso, il
# rosso del visibile nel verde, e cosi' via. La via decimale lo fa passando ogni
# valore per un numero fra 0 e 1 e tornando indietro, e costa tre conversioni
# in float32 di tutta l'immagine. La via intera fa gli stessi conti con numeri
# interi, e il risultato e' identico bit per bit:
#
#   canale a 8 bit        v/255*255 + 0.5, troncato      = v
#   16 bit -> 8 bit       x/65535*255 + 0.5 = x/257 + .5 = (x + 128) // 257
#   media di 3 a 8 bit    s/3 + 0.5, troncato            = (s + 1) // 3
#   media di 3 a 16 bit   s/771 + 0.5, troncato          = (s + 385) // 771
#
# Identico perche' in nessuno dei quattro casi il valore esatto cade abbastanza
# vicino a un mezzo da far arrotondare diversamente l'errore del float32: x/257
# ne sta ad almeno 0.002, s/3 e s/771 non ci cadono mai. Verificato su tutti i
# valori possibili (test_falsocolore.py) e sulle riprese vere: 0 pixel diversi
# su NIR850, UVR, SWIR a 16 bit e radiografia, anche contro la versione
# precedente del programma. Tempo: tre bande insieme da 48-49 s a 21-25 s,
# quasi tutto ormai lettura e scrittura.
#
# Le divisioni si fanno una volta sola per ogni valore possibile, in tabelle:
# la divisione intera di numpy su 315 milioni di valori era piu' lenta dei
# decimali (39.6 s contro 24.1 sulla SWIR).
#
# La via decimale resta per le immagini che non sono a 8 o 16 bit interi, e come
# termine di confronto del test.

_DA16 = ((np.arange(65536, dtype=np.uint32) + 128) // 257).astype(np.uint8)
_MEDIA8 = ((np.arange(3 * 255 + 1, dtype=np.uint32) + 1) // 3).astype(np.uint8)
_MEDIA16 = ((np.arange(3 * 65535 + 1, dtype=np.uint32) + 385) // 771).astype(np.uint8)


def via_intera_possibile(dtype_vis, forma_banda, dtype_banda):
    """La via intera vale per 8 e 16 bit, con la banda in grigio o con almeno
    tre canali (sui due canali la media decimale mescola grigio e trasparenza,
    e non la si vuole riprodurre)."""
    interi = (np.dtype(np.uint8), np.dtype(np.uint16))
    canali_ok = len(forma_banda) == 2 or forma_banda[2] in (1, 3, 4)
    return (np.dtype(dtype_vis) in interi and np.dtype(dtype_banda) in interi
            and canali_ok)


def _a_8bit(x):
    return x if x.dtype == np.uint8 else _DA16[x]


def _grigio_intero(b, canale):
    if b.ndim == 2:
        return _a_8bit(b)
    if b.shape[2] == 1:
        return _a_8bit(b[:, :, 0])
    if canale != 'media':
        return _a_8bit(b[:, :, {'r': 0, 'g': 1, 'b': 2}[canale]])
    if b.dtype == np.uint8:
        s = b[:, :, 0].astype(np.uint16)
        s += b[:, :, 1]
        s += b[:, :, 2]
        return _MEDIA8[s]
    s = b[:, :, 0].astype(np.uint32)
    s += b[:, :, 1]
    s += b[:, :, 2]
    return _MEDIA16[s]


def striscia_intera(tipo, vis, ban, canale='media'):
    """La striscia composta spostando canali. vis e ban come letti dal file."""
    grigio = _grigio_intero(ban, canale)
    out = np.empty(vis.shape[:2] + (3,), dtype=np.uint8)
    for i, sorgente in enumerate(RICETTE[tipo]['sorgenti']):
        out[:, :, i] = grigio if sorgente == 'banda' else _a_8bit(vis[:, :, sorgente[1]])
    return out


def _visibile_decimale(vis):
    """Il visibile in decimali fra 0 e 1. A parte, perche' nel ciclo si fa una
    volta sola per tutte le bande."""
    return vis.astype(np.float32) / _scala_dtype(vis.dtype)


def striscia_decimale(tipo, vis, ban, canale='media'):
    """La striscia composta passando per i decimali: il ripiego per i formati
    non interi, e il termine di confronto della via intera."""
    return _componi_decimale(tipo, _visibile_decimale(vis), ban, canale)


def _componi_decimale(tipo, v, ban, canale):
    b = _riduci(ban.astype(np.float32) / _scala_dtype(ban.dtype), canale)
    out = np.empty(v.shape[:2] + (3,), dtype=np.uint8)
    for i, sorgente in enumerate(RICETTE[tipo]['sorgenti']):
        piano = b if sorgente == 'banda' else v[:, :, sorgente[1]]
        out[:, :, i] = np.clip(piano * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def verifica_allineamento(arr_vis, arr_banda, n_finestre=4, verbose=True):
    """
    Controlla che la banda sia DAVVERO allineata al visibile, non solo delle
    stesse dimensioni.

    Serve perche' il solo controllo delle dimensioni da' una falsa
    rassicurazione: le riprese originali di questo progetto hanno esattamente
    la stessa dimensione dell'allineata, perche' escono dallo stesso sistema.
    Comporre un falso colore con le originali produce un'immagine che a colpo
    d'occhio sembra giusta e che invece ha frange ovunque -- e' successo.

    Misura il residuo su n_finestre^2 campioni da 512 px: un paio di secondi,
    contro i due minuti e i 900 MB di un'uscita sbagliata.
    """
    r = val.residual_check(arr_vis, arr_banda, None, n_side=n_finestre, patch=512,
                           min_conf=0.05, verbose=False, n_boot=200)
    if not r.get('n_affidabili'):
        return {'verificabile': False, 'residuo_mediano_px': None,
                'n_affidabili': 0, 'n_campioni': r.get('n_campioni', 0)}
    return {'verificabile': True,
            'residuo_mediano_px': r['residuo_mediano_px'],
            'n_affidabili': r['n_affidabili'], 'n_campioni': r['n_campioni']}


def componi_molte(path_vis, lavori, canale='media',
                  altezza_banda=512, anteprima_px=0, max_residuo=3.0,
                  forza=False, verbose=True):
    """
    Compone piu' falsi colori in una volta sola, leggendo e scrivendo a bande
    di righe.

    Si lavora a bande e non tutto in una volta perche' le immagini di questo
    progetto pesano ~900 MB l'una: caricare visibile, banda aggiunta e uscita
    insieme vorrebbe dire quasi 3 GB, mentre a bande il picco resta di qualche
    decina di MB. E' la stessa ragione per cui align_paintings scrive a tile.

    Piu' bande si compongono NELLO STESSO ciclo di righe, cosi' il visibile si
    legge una volta sola invece di una per banda: su quattro bande sono 2.7 GB
    di letture in meno.

    Args:
        lavori: lista di (tipo, path_banda, path_out)
        max_residuo: soglia oltre la quale la banda non e' considerata allineata
        forza: compone lo stesso, nonostante il controllo
    """
    arr_vis, _ = image_io.apri(path_vis, verbose=False)
    H, W = arr_vis.shape[:2]
    if arr_vis.ndim != 3 or arr_vis.shape[2] < 3:
        raise ValueError("Il visibile deve avere tre canali: serve il suo RGB.")

    preparati = []
    for tipo, path_banda, path_out in lavori:
        arr_ban, _ = image_io.apri(path_banda, verbose=False)
        if arr_ban.shape[:2] != (H, W):
            raise ValueError(
                f"Le due immagini hanno dimensioni diverse: visibile {W}x{H}, "
                f"banda {arr_ban.shape[1]}x{arr_ban.shape[0]}.\n"
                f"Il falso colore presuppone immagini GIA' allineate e delle stesse "
                f"dimensioni: passa l'uscita di align_paintings.py, non l'originale.")
        st = statistiche_banda(path_banda, canale=canale)
        allineamento = verifica_allineamento(arr_vis, arr_ban)
        image_io.rilascia_pagine()      # le letture del controllo, banda per banda
        nome = os.path.basename(path_banda)
        if verbose:
            print(f"\n{'=' * 72}\n{tipo.upper()}  {RICETTE[tipo]['descrizione']}\n{'=' * 72}")
            print(f"  visibile : {os.path.basename(path_vis)}  {W}x{H}")
            print(f"  banda    : {nome}  canale '{canale}'")
            print(f"  scarto fra i canali della banda: {st['scarto_fra_canali']:.4f}"
                  + ("  (di fatto monocromatica)" if st['scarto_fra_canali'] < 0.02
                     else "  <- i tre canali differiscono: la scelta del canale conta"))
        if not allineamento['verificabile']:
            print(f"  [attenzione] allineamento non verificabile: {nome} e il visibile "
                  f"non condividono abbastanza struttura. Si procede comunque.")
        else:
            residuo = allineamento['residuo_mediano_px']
            if residuo > max_residuo and not forza:
                raise ValueError(
                    f"{nome} non sembra allineata al visibile: residuo mediano "
                    f"{residuo:.1f} px, oltre la soglia di {max_residuo:.1f} px.\n"
                    f"Le dimensioni coincidono, ma questo non basta: le riprese "
                    f"originali hanno le stesse dimensioni dell'allineata.\n"
                    f"Allineala prima (interfaccia.py), oppure componi lo stesso "
                    f"con l'opzione 'procedi comunque' (--forza da riga di comando).")
            if verbose:
                stato = "ok" if residuo <= max_residuo else "SOPRA SOGLIA, si procede su richiesta"
                print(f"  allineamento: residuo mediano {residuo:.2f} px su "
                      f"{allineamento['n_affidabili']} finestre -> {stato}")
        preparati.append((tipo, path_banda, path_out, arr_ban, st, allineamento))

    if verbose:
        print("\n  nessuna alterazione radiometrica (livelli come ripresi)")

    # Si scrive su un nome temporaneo e si rinomina alla fine: un'esecuzione
    # interrotta lascia cosi' un file ".parziale", riconoscibile, invece di uno
    # col nome giusto e l'aria di essere completo (era successo: 59 MB su 900,
    # intestazione da immagine intera).
    # Si compone spostando canali (vedi striscia_intera): stesso file della via
    # decimale, bit per bit, in circa la meta' del tempo.
    intera = [via_intera_possibile(arr_vis.dtype, arr_ban.shape, arr_ban.dtype)
              for _t, _p, _o, arr_ban, _s, _a in preparati]
    if verbose:
        print("  composizione " + ("per spostamento di canali" if all(intera) else
                                   "in decimali" if not any(intera) else
                                   "per spostamento di canali dove possibile"))
    scrittori, temporanei = [], []
    try:
        for tipo, path_banda, path_out, arr_ban, st, _a in preparati:
            temp = path_out + '.parziale'
            temporanei.append(temp)
            scrittori.append(tiff_memmap.create_tiff_memmap(
                temp, H, W, samples=3, dtype=np.uint8, sequenziale=True))

        for y0 in range(0, H, altezza_banda):
            y1 = min(H, y0 + altezza_banda)
            vis = np.asarray(arr_vis[y0:y1])
            v_dec = None
            for scrittore, via_intera, (tipo, _p, _o, arr_ban, st, _a) in zip(
                    scrittori, intera, preparati):
                ban = np.asarray(arr_ban[y0:y1])
                if via_intera:
                    banda = striscia_intera(tipo, vis, ban, canale)
                else:
                    if v_dec is None:
                        v_dec = _visibile_decimale(vis)
                    banda = _componi_decimale(tipo, v_dec, ban, canale)
                scrittore.scrivi(banda)
            # Ogni quattro fasce si restituiscono al sistema le pagine gia' lette
            # del visibile e delle bande: senza, restano contate nel processo
            # fino alla fine (5.3 GB con quattro bande).
            if (y0 // altezza_banda) % 4 == 3:
                image_io.rilascia_pagine()
            if verbose:
                print(f"      righe {y1}/{H}  ({100.0 * y1 / H:.0f}%)",
                      end='\r', flush=True)
        for scrittore in scrittori:
            scrittore.chiudi()
    except BaseException:
        for scrittore in scrittori:
            try:
                scrittore.f.close()
            except Exception:
                pass
        for temp in temporanei:
            try:
                os.unlink(temp)
            except OSError:
                pass
        raise

    report = []
    for temp, (tipo, path_banda, path_out, _arr, st, allineamento) in zip(temporanei, preparati):
        os.replace(temp, path_out)
        if verbose:
            print(f"\n  scritto {path_out} "
                  f"({os.path.getsize(path_out) / 1024 ** 3:.2f} GB)")
        anteprima = None
        if anteprima_px:
            anteprima = os.path.splitext(path_out)[0] + '_anteprima.png'
            riletto, _ = image_io.apri(path_out, verbose=False)
            _salva_anteprima(riletto, anteprima, anteprima_px)
            del riletto
            if verbose:
                print(f"  anteprima: {anteprima}")
        report.append({
            'tipo': tipo, 'ricetta': RICETTE[tipo]['descrizione'],
            'visibile': os.path.abspath(path_vis), 'banda': os.path.abspath(path_banda),
            'canale': canale,
            'allineamento': allineamento,
            'statistiche_banda': st,
            'uscita': os.path.abspath(path_out),
            'anteprima': os.path.abspath(anteprima) if anteprima else None,
        })
    return report


def altri_formati(percorsi_tif, formati, verbose=True, originale=None, report=None):
    """
    Porta i falsi colori nei formati voluti, partendo dal TIFF che il
    compositore scrive sempre.

    Il TIFF si scrive a fasce di righe, con la memoria costante; gli altri si
    ricavano da lui. Se fra i formati voluti il TIFF non c'e', dopo la
    conversione si cancella.

    Senza scelta il formato e' quello di `originale`, cioe' del visibile: e' la
    regola di tutto il programma, un'immagine elaborata esce nel formato in cui
    e' entrata a meno che dal menu non si chieda altro. Un formato che non si
    sa scrivere (BMP) resta TIFF.

    Args:
        originale: il file del visibile, da cui prendere il formato quando
            `formati` e' vuoto. None = resta il TIFF.
        report: le voci prodotte da componi_molte, nello stesso ordine: se il
            TIFF viene tolto, la loro 'uscita' passa al file rimasto.

    Returns:
        lista dei file prodotti, nell'ordine
    """
    scelti = fmt.normalizza(formati)
    if not scelti and originale:
        try:
            scelti = fmt.normalizza([os.path.splitext(originale)[1]])
        except ValueError:
            scelti = []
    if not scelti or scelti == ['.tif']:
        return list(percorsi_tif)
    prodotti = []
    for i, p in enumerate(percorsi_tif):
        nuovi = fmt.scrivi_altri(p, [e for e in scelti if e != '.tif'], verbose=verbose)
        prodotti += nuovi
        if '.tif' in scelti:
            prodotti.append(p)
        else:
            os.remove(p)
            if report is not None and i < len(report) and nuovi:
                report[i]['uscita'] = os.path.abspath(nuovi[0])
        if report is not None and i < len(report):
            report[i]['uscite'] = [os.path.abspath(n) for n in nuovi] + (
                [os.path.abspath(p)] if '.tif' in scelti else [])
    return prodotti


def componi(tipo, path_vis, path_banda, path_out, **kw):
    """Una sola composizione: comodita' attorno a componi_molte."""
    return componi_molte(path_vis, [(tipo, path_banda, path_out)], **kw)[0]


def _salva_anteprima(arr, path, lato_lungo):
    """PNG ridotto, per guardare il risultato senza aprire un file da 1 GB."""
    h, w = arr.shape[:2]
    scala = min(1.0, lato_lungo / float(max(h, w)))
    piccola = cv2.resize(np.asarray(arr), (max(1, int(w * scala)),
                                           max(1, int(h * scala))),
                         interpolation=cv2.INTER_AREA)
    image_io.scrivi_immagine(path, cv2.cvtColor(piccola, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Riga di comando
# ---------------------------------------------------------------------------

def _nome(cartella, tipo, path_banda, momento):
    base = os.path.splitext(os.path.basename(path_banda))[0]
    base = base.replace('_allineata', '')
    quando = momento.strftime('%Y-%m-%d %H-%M')
    return os.path.join(cartella, f"{tipo.upper()}_{base} ({quando}).tif")


def main():
    ap = argparse.ArgumentParser(
        description="Falso colore IRFC e UVFC da immagini gia' allineate.")
    ap.add_argument('--vis', required=True, help="ripresa nel visibile (RGB)")
    ap.add_argument('--ir', nargs='+', metavar='IMG',
                    help="una o piu' riprese infrarosse allineate: un IRFC per ciascuna")
    ap.add_argument('--uv', nargs='+', metavar='IMG',
                    help="una o piu' riprese in UV RIFLESSO allineate: un UVFC per ciascuna")
    ap.add_argument('--out-dir', default='risultati', help="cartella di uscita")
    ap.add_argument('--canale', default='media', choices=('media', 'r', 'g', 'b'),
                    help="come ridurre a canale singolo la banda aggiunta; conta "
                         "solo se i tre canali differiscono fra loro")
    ap.add_argument('--anteprima-px', type=int, default=0,
                    help="lato lungo di un'anteprima PNG (0 = non produrla)")
    ap.add_argument('--max-residuo', type=float, default=3.0,
                    help="residuo oltre il quale la banda non e' considerata allineata")
    ap.add_argument('--formati', nargs='+', default=None, metavar='FORMATO',
                    help="uno o piu' fra TIFF, PNG, JPEG; senza, il formato del visibile")
    ap.add_argument('--forza', action='store_true',
                    help="componi anche se la banda non sembra allineata")
    args = ap.parse_args()

    if not args.ir and not args.uv:
        ap.error("serve almeno --ir (per l'IRFC) o --uv (per l'UVFC)")

    os.makedirs(args.out_dir, exist_ok=True)
    momento = datetime.now()
    lavori = [(tipo, p, _nome(args.out_dir, tipo, p, momento))
              for tipo, bande in (('irfc', args.ir or []), ('uvfc', args.uv or []))
              for p in bande]
    report = componi_molte(args.vis, lavori, canale=args.canale,
                           anteprima_px=args.anteprima_px,
                           max_residuo=args.max_residuo, forza=args.forza)
    altri_formati([l[2] for l in lavori], args.formati, originale=args.vis, report=report)

    percorso = os.path.join(
        args.out_dir, f"falso_colore ({momento.strftime('%Y-%m-%d %H-%M')}).json")
    with open(percorso, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nparametri salvati: {percorso}")


if __name__ == '__main__':
    main()
