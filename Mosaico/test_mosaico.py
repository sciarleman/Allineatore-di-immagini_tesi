"""
Test del mosaico su tasselli sintetici con posizioni NOTE.

    python Mosaico\\test_mosaico.py

Non ci sono ancora tasselli veri, quindi si fabbricano: si prende un "dipinto"
sintetico (la texture non periodica di test_pipeline.py, in tre canali
correlati come in una foto) e lo si ritaglia su una griglia di tasselli
sovrapposti, come farebbe una camera su un carrello imperfetto:

- il passo nominale e' sbagliato di qualche decina di pixel a ogni scatto;
- ogni scatto ha una sua piccola rotazione;
- ogni tassello ha una dominante di colore, un'esposizione e una vignettatura
  sue, piu' rumore. E' proprio la differenza di colore che il programma NON
  deve correggere, e che non deve disturbare la misura.

I file vengono numerati nell'ordine di una serpentina per colonne che parte in
basso a destra, cosi' il test passa anche per la decodifica del percorso.

Poi si confrontano le posizioni stimate con quelle vere, agli angoli e al
centro di ogni tassello. Il mosaico ha un sistema di riferimento suo (quello
del tassello ancora), quindi prima del confronto si toglie la trasformazione
d'insieme che lo porta sul dipinto: quel che resta e' l'errore di
posizionamento RELATIVO, cioe' quello che si vedrebbe sulle cuciture.

Tre prove in piu':
- un tassello senza nessuna struttura: le sue coppie non si possono misurare,
  deve essere messo al suo posto dal passo tipico e segnalato;
- tasselli a 16 bit in grigio e tasselli JPEG, per le due strade di lettura e
  per il livello in grigio con trasparenza;
- i livelli scritti: si riaprono e si controlla che, messi alle coordinate
  dello script per Photoshop, cadano davvero sul dipinto.
"""

import os
import shutil
import struct
import sys
import tempfile

import numpy as np
import cv2
from PIL import Image

# il motore e la texture sintetica stanno nella cartella dell'allineatore
_MOTORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Allineatore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import tiff_memmap
import mosaico
from test_pipeline import synthetic_texture


def leggi_tiff_grezzo(path):
    """
    Tag e pixel di un TIFF non compresso a striscia unica, letti a mano.

    Non si usa Pillow apposta: serve un lettore che non condivida nulla con chi
    ha scritto il file, e Pillow comunque non apre i TIFF in grigio a 16 bit con
    trasparenza (e' un suo limite: libtiff, e quindi Photoshop e GIMP, li
    leggono).
    """
    dimensioni = {3: 2, 4: 4, 5: 8, 7: 1}
    formati = {3: 'H', 4: 'L', 5: 'LL'}
    with open(path, 'rb') as fh:
        testa = fh.read(8)
        if testa[:4] != b'II*\0':
            raise ValueError("intestazione TIFF little endian assente")
        fh.seek(struct.unpack('<L', testa[4:8])[0])
        tag = {}
        for _ in range(struct.unpack('<H', fh.read(2))[0]):
            t, tipo, conta, valore = struct.unpack('<HHL4s', fh.read(12))
            n = dimensioni[tipo] * conta
            if n > 4:
                qui = fh.tell()
                fh.seek(struct.unpack('<L', valore)[0])
                grezzo = fh.read(n)
                fh.seek(qui)
            else:
                grezzo = valore[:n]
            tag[t] = grezzo if tipo == 7 else struct.unpack('<' + formati[tipo] * conta, grezzo)
    dt = {8: np.uint8, 16: np.uint16}[tag[258][0]]
    arr = np.memmap(path, dtype=dt, mode='r', offset=tag[273][0],
                    shape=(tag[257][0], tag[256][0], tag[277][0]))
    return arr, tag


def dipinto_sintetico(h, w):
    g = synthetic_texture(h, w, seed=0).astype(np.float32)
    t2 = synthetic_texture(h, w, seed=1).astype(np.float32)
    t3 = synthetic_texture(h, w, seed=2).astype(np.float32)
    return np.stack([g, 0.75 * g + 0.25 * t2, 0.55 * g + 0.45 * t3], axis=2)


def fabbrica_tasselli(cartella, dipinto, n_righe, n_colonne, lato, sov, percorso, partenza,
                      rng, ruota_gradi=0.4, errore_passo=20.0, formato='tif', grigio16=False,
                      vuoto=None, icc=None):
    """
    Ritaglia il dipinto in tasselli e li salva nell'ordine di ripresa.

    Returns:
        {indice di ripresa: matrice 2x3 VERA tassello -> dipinto}
    """
    W_t, H_t = lato
    passo_x, passo_y = W_t * (1 - sov[0]), H_t * (1 - sov[1])
    os.makedirs(cartella, exist_ok=True)
    vere = {}
    for k, (r, c) in enumerate(mosaico.posizioni_griglia(n_righe, n_colonne, percorso, partenza)):
        centro = np.array([80 + W_t / 2 + c * passo_x, 80 + H_t / 2 + r * passo_y])
        centro += rng.uniform(-errore_passo, errore_passo, 2)
        th = np.radians(rng.uniform(-ruota_gradi, ruota_gradi))
        A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        M = np.hstack([A, (centro - A @ np.array([W_t / 2, H_t / 2])).reshape(2, 1)])
        vere[k] = M
        X, Y = np.meshgrid(np.arange(W_t, dtype=np.float32), np.arange(H_t, dtype=np.float32))
        mx = (M[0, 0] * X + M[0, 1] * Y + M[0, 2]).astype(np.float32)
        my = (M[1, 0] * X + M[1, 1] * Y + M[1, 2]).astype(np.float32)
        t = cv2.remap(dipinto, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # colore, esposizione e vignettatura diversi per ogni scatto
        guadagno = rng.uniform(0.8, 1.2, 3).astype(np.float32)
        rr = ((X - W_t / 2) ** 2 + (Y - H_t / 2) ** 2) / ((W_t / 2) ** 2 + (H_t / 2) ** 2)
        t = t * guadagno * (1 - 0.3 * rr)[:, :, None] + rng.uniform(-15, 15)
        t += rng.normal(0, 2.0, t.shape).astype(np.float32)
        if vuoto is not None and k == vuoto:
            t = np.full_like(t, 128.0) + rng.normal(0, 2.0, t.shape).astype(np.float32)
        nome = os.path.join(cartella, f"scatto_{k + 1}.{formato}")
        if grigio16:
            t = np.clip(t.mean(axis=2) * 180 + 3000, 0, 65535).astype(np.uint16)
            mm = tiff_memmap.create_tiff_memmap(nome, H_t, W_t, samples=1, dtype=np.uint16)
            mm[:] = t
            mm.flush()
            del mm
        else:
            t = np.clip(t, 0, 255).astype(np.uint8)
            if formato == 'tif':
                mm = tiff_memmap.create_tiff_memmap(nome, H_t, W_t, samples=3, dtype=np.uint8)
                mm[:] = t
                mm.flush()
                del mm
            else:
                Image.fromarray(t).save(nome, quality=93, icc_profile=icc)
    return vere


def errore_relativo(report, vere, lato, escludi=()):
    """
    Errore di posizionamento relativo, in px, agli angoli e al centro dei
    tasselli, dopo aver tolto la trasformazione d'insieme mosaico -> dipinto.

    La trasformazione d'insieme si stima senza i tasselli in `escludi`: uno
    posizionato solo dal passo del carrello sbaglia di decine di pixel per
    costruzione, e lasciarlo entrare nella stima sporcherebbe il confronto di
    tutti gli altri.

    Returns:
        (errori (n_tasselli, 5), S 2x3 mosaico -> dipinto)
    """
    W_t, H_t = lato
    punti = np.array([[0, 0], [W_t, 0], [W_t, H_t], [0, H_t], [W_t / 2, H_t / 2]], float)
    stimati, veri = [], []
    for t in report['tasselli']:
        k = t['numero'] - 1
        M = np.array(t['tassello_verso_mosaico'])
        stimati.append(punti @ M[:, :2].T + M[:, 2])
        veri.append(punti @ vere[k][:, :2].T + vere[k][:, 2])
    s, v = np.concatenate(stimati), np.concatenate(veri)
    usa = np.repeat([t['numero'] - 1 not in escludi for t in report['tasselli']], len(punti))
    A = np.column_stack([s, np.ones(len(s))])
    S = np.linalg.lstsq(A[usa], v[usa], rcond=None)[0].T
    err = np.hypot(*(s @ S[:, :2].T + S[:, 2] - v).T)
    return err.reshape(len(report['tasselli']), 5), S


def controlla_livelli(report, cartella_out, dipinto, S, rng, lato=128, icc=None):
    """
    I livelli, messi alle coordinate dello script, cadono sul dipinto?

    Per ogni livello si prendono alcune finestre interamente opache, si ricava
    dal dipinto quello che DOVREBBE esserci in quella posizione del mosaico e si
    misura lo spostamento fra le due con la correlazione di fase. Si controlla
    anche il formato: niente compressione, alfa dichiarato come trasparenza non
    premoltiplicata, profilo colore copiato dall'originale se ce n'era uno.
    """
    scarti, problemi = [], []
    for t in report['tasselli']:
        liv = t['livello']
        arr, tag = leggi_tiff_grezzo(os.path.join(cartella_out, liv['file']))
        if tag.get(259) != (1,) or tag.get(338) != (2,):
            problemi.append(f"{liv['file']}: compressione {tag.get(259)}, "
                            f"ExtraSamples {tag.get(338)}")
        if icc is not None and tag.get(34675) != icc:
            problemi.append(f"{liv['file']}: profilo colore non copiato")
        x0, y0, x1, y1 = liv['rettangolo']
        if arr.shape[:2] != (y1 - y0, x1 - x0):
            problemi.append(f"{liv['file']}: dimensioni {arr.shape[:2]} invece di "
                            f"{(y1 - y0, x1 - x0)}")
            continue
        alfa = np.asarray(arr[:, :, -1])
        massimo = np.iinfo(arr.dtype).max
        if not np.isin(np.unique(alfa), [0, massimo]).all():
            problemi.append(f"{liv['file']}: trasparenza non netta")
        colore = np.asarray(arr[:, :, :-1]).astype(np.float32)
        colore = colore.mean(axis=2) if colore.shape[2] == 3 else colore[:, :, 0]
        win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
        prese = 0
        for _ in range(60):
            u = int(rng.integers(0, max(1, arr.shape[1] - lato)))
            v = int(rng.integers(0, max(1, arr.shape[0] - lato)))
            if (alfa[v:v + lato, u:u + lato] < massimo).any():
                continue
            X, Y = np.meshgrid(np.arange(x0 + u, x0 + u + lato, dtype=np.float64),
                               np.arange(y0 + v, y0 + v + lato, dtype=np.float64))
            mx = (S[0, 0] * X + S[0, 1] * Y + S[0, 2]).astype(np.float32)
            my = (S[1, 0] * X + S[1, 1] * Y + S[1, 2]).astype(np.float32)
            atteso = cv2.remap(dipinto.mean(axis=2), mx, my, cv2.INTER_LINEAR)
            ga = mosaico.bm.gradient_magnitude(atteso)
            gb = mosaico.bm.gradient_magnitude(colore[v:v + lato, u:u + lato])
            (dx, dy), conf = cv2.phaseCorrelate(ga.astype(np.float64), gb.astype(np.float64), win)
            scarti.append(float(np.hypot(dx, dy)))
            prese += 1
            if prese == 3:
                break
        del arr
    return np.array(scarti), problemi


def prova_percorsi():
    """La decodifica del percorso, contro tabelle scritte a mano (2 righe x 3
    colonne). Non usa la funzione per costruire l'atteso: la verifica."""
    attesi = {
        ('serpentina_righe', 'alto_sinistra'): [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)],
        ('serpentina_righe', 'basso_sinistra'): [(1, 0), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)],
        ('serpentina_colonne', 'alto_sinistra'): [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)],
        ('serpentina_colonne', 'basso_destra'): [(1, 2), (0, 2), (0, 1), (1, 1), (1, 0), (0, 0)],
        ('righe', 'basso_destra'): [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)],
        ('righe', 'alto_sinistra'): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        ('colonne', 'alto_destra'): [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)],
        ('colonne', 'basso_sinistra'): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)],
    }
    sbagliati = [k for k, v in attesi.items() if mosaico.posizioni_griglia(2, 3, *k) != v]
    ordine = mosaico.ordina_naturale(['s_10.tif', 's_2.tif', 's_1.tif', 'S_11.tif'])
    ordine_ok = ordine == ['s_1.tif', 's_2.tif', 's_10.tif', 'S_11.tif']
    return sbagliati, ordine_ok


def main():
    lavoro = tempfile.mkdtemp(prefix='mosaico_test_')
    print(f"cartella di lavoro: {lavoro}\n")
    esiti = []
    try:
        # ---- percorsi --------------------------------------------------
        sbagliati, ordine_ok = prova_percorsi()
        print("PERCORSI: " + ("ok" if not sbagliati else f"SBAGLIATI {sbagliati}")
              + ", ordine naturale " + ("ok" if ordine_ok else "SBAGLIATO"))
        esiti.append(('percorsi e ordine naturale', not sbagliati and ordine_ok))

        rng = np.random.default_rng(7)
        lato = (960, 720)
        sov = (0.22, 0.28)
        dipinto = dipinto_sintetico(1950, 3450)

        # ---- 1. griglia 3x4, serpentina per colonne dal basso a destra --
        cart = os.path.join(lavoro, 'griglia')
        vere = fabbrica_tasselli(cart, dipinto, 3, 4, lato, sov, 'serpentina_colonne',
                                 'basso_destra', rng)
        out = os.path.join(lavoro, 'uscita')
        rep = mosaico.componi_mosaico(mosaico.elenca_immagini(cart), 3, 4, out,
                                      percorso='serpentina_colonne', partenza='basso_destra',
                                      fattore=1, lato_grossolano=400)
        err, S = errore_relativo(rep, vere, lato)
        print("\n" + "=" * 72)
        print("GRIGLIA 3x4 (rotazioni fino a 0.4 gradi, passo sbagliato fino a 20 px)")
        print(f"  modello scelto: {rep['modello']['usato']}   "
              f"(cross-validation: {', '.join(f'{k} {v:.3f}' for k, v in rep['modello']['crossval_px'].items())})")
        print(f"  errore di posizione relativo, agli angoli e al centro dei tasselli: "
              f"mediano {np.median(err):.3f} px, max {err.max():.3f} px")
        misurate = sum(c['stato'] == 'misurata' for c in rep['coppie'])
        print(f"  coppie misurate: {misurate}/{len(rep['coppie'])}")
        esiti.append(('posizioni 3x4', np.median(err) < 0.5 and err.max() < 1.5
                      and misurate == len(rep['coppie'])))
        esiti.append(('modello rigido riconosciuto', rep['modello']['usato'] == 'rigida'))

        scarti, problemi = controlla_livelli(rep, out, dipinto, S, rng)
        print(f"  livelli riaperti e confrontati col dipinto su {len(scarti)} finestre: "
              f"scarto mediano {np.median(scarti):.3f} px, max {scarti.max():.3f} px")
        for p in problemi:
            print(f"  PROBLEMA: {p}")
        jsx = open(os.path.join(out, 'apri_in_photoshop.jsx'), encoding='utf-8-sig').read()
        jsx_ok = all(f"x: {t['livello']['x']}, y: {t['livello']['y']}" in jsx
                     and t['livello']['file'] in jsx for t in rep['tasselli'])
        with Image.open(os.path.join(out, rep['tasselli'][0]['livello']['file'])) as im:
            modo_pil = im.mode
        print(f"  script per Photoshop coerente col report: {'si' if jsx_ok else 'NO'}; "
              f"Pillow apre il livello come {modo_pil}")
        esiti.append(('livelli al posto giusto', len(scarti) >= 24 and np.median(scarti) < 0.3
                      and scarti.max() < 1.0 and not problemi))
        esiti.append(('script e formato dei livelli', jsx_ok and modo_pil == 'RGBA'))

        # ---- 1b. gli stessi tasselli alla precisione di default --------
        # E' il caso che aveva fatto emergere l'errore di mezzo pixel di
        # phaseCorrelate: la fascia comune piu' stretta, ridotta di 2, dava
        # blocchi da 44 px (vedi mosaico._lato_fft).
        cart = os.path.join(lavoro, 'griglia')
        rep = mosaico.componi_mosaico(mosaico.elenca_immagini(cart), 3, 4,
                                      os.path.join(lavoro, 'uscita_f2'),
                                      percorso='serpentina_colonne', partenza='basso_destra',
                                      fattore=2, lato_grossolano=400, scrivi_livelli=False,
                                      verbose=False)
        err, _ = errore_relativo(rep, vere, lato)
        cuciture = [c['verifica']['residuo_mediano_px'] for c in rep['coppie']]
        lati_ok = all(cv2.getOptimalDFTSize(mosaico._lato_fft(n)) == mosaico._lato_fft(n)
                      for n in range(16, 1025))
        print(f"  alla precisione normale (fattore 2): modello {rep['modello']['usato']}, errore "
              f"mediano {np.median(err):.3f} px, max {err.max():.3f} px; cucitura peggiore "
              f"{max(cuciture):.2f} px; lati delle finestre sempre accettati da OpenCV: "
              f"{'si' if lati_ok else 'NO'}")
        esiti.append(('posizioni 3x4 a fattore 2', np.median(err) < 0.3 and err.max() < 0.6
                      and max(cuciture) < 0.5 and lati_ok))

        # ---- 2. un tassello senza struttura ----------------------------
        cart = os.path.join(lavoro, 'vuoto')
        vere = fabbrica_tasselli(cart, dipinto, 2, 3, lato, sov, 'serpentina_righe',
                                 'alto_sinistra', rng, vuoto=1)
        out = os.path.join(lavoro, 'uscita_vuoto')
        rep = mosaico.componi_mosaico(mosaico.elenca_immagini(cart), 2, 3, out,
                                      fattore=1, lato_grossolano=400, scrivi_livelli=False)
        err, _ = errore_relativo(rep, vere, lato, escludi=(1,))
        buoni = [t['numero'] - 1 for t in rep['tasselli'] if t['numero'] != 2]
        stati_vuoto = [c['stato'] for c in rep['coppie'] if 2 in (c['a'], c['b'])]
        vuoto = next(t for t in rep['tasselli'] if t['numero'] == 2)
        print("\n" + "=" * 72)
        print("TASSELLO SENZA STRUTTURA (il numero 2)")
        print(f"  coppie del tassello 2: {', '.join(stati_vuoto)}")
        print(f"  errore sugli altri tasselli: mediano {np.median(err[buoni]):.3f} px, "
              f"max {err[buoni].max():.3f} px;  sul tassello 2: max {err[1].max():.1f} px "
              f"(atteso dell'ordine dell'errore del passo, 20 px), rotazione "
              f"{vuoto['rotazione_gradi']:+.3f} gradi, scala {vuoto['scala_x']:.4f}")
        esiti.append(('tassello vuoto segnalato, altri a posto',
                      all(s != 'misurata' for s in stati_vuoto)
                      and np.median(err[buoni]) < 0.5 and err[buoni].max() < 1.5
                      and err[1].max() < 60.0 and abs(vuoto['rotazione_gradi']) < 1e-3
                      and abs(vuoto['scala_x'] - 1) < 1e-4))

        # ---- 3. 16 bit in grigio e JPEG con profilo colore -------------
        try:
            from PIL import ImageCms
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        except Exception:
            srgb = None
        for etichetta, kw in (('16 BIT IN GRIGIO', {'grigio16': True}),
                              ('JPEG CON PROFILO sRGB', {'formato': 'jpg', 'icc': srgb})):
            cart = os.path.join(lavoro, etichetta.replace(' ', '_'))
            vere = fabbrica_tasselli(cart, dipinto, 2, 2, lato, sov, 'righe', 'alto_sinistra',
                                     rng, **kw)
            out = cart + '_uscita'
            rep = mosaico.componi_mosaico(mosaico.elenca_immagini(cart), 2, 2, out,
                                          percorso='righe', fattore=1, lato_grossolano=400,
                                          verbose=False)
            err, S = errore_relativo(rep, vere, lato)
            scarti, problemi = controlla_livelli(rep, out, dipinto, S, rng, icc=kw.get('icc'))
            arr, _ = leggi_tiff_grezzo(os.path.join(out, rep['tasselli'][0]['livello']['file']))
            forma = f"{arr.shape[2]} canali {arr.dtype}"
            del arr
            print("\n" + "=" * 72)
            print(f"{etichetta}: errore mediano {np.median(err):.3f} px, max {err.max():.3f} px; "
                  f"livelli {forma}, scarto col dipinto mediano {np.median(scarti):.3f} px")
            for p in problemi:
                print(f"  PROBLEMA: {p}")
            atteso = "2 canali uint16" if kw.get('grigio16') else "4 canali uint8"
            esiti.append((etichetta.lower(), np.median(err) < 0.5 and err.max() < 1.5
                          and forma == atteso and not problemi and np.median(scarti) < 0.3))

        print("\n" + "=" * 72)
        for nome, ok in esiti:
            print(f"  {'ok     ' if ok else 'FALLITO'}  {nome}")
        tutto = all(ok for _, ok in esiti)
        print("TEST SUPERATO" if tutto else "TEST FALLITO")
        return 0 if tutto else 1
    finally:
        shutil.rmtree(lavoro, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
