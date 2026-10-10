"""
Test end-to-end su dati sintetici con trasformazione NOTA.

    python test_pipeline.py

Genera una coppia di immagini che imitano il caso reale:
- texture non periodica (rumore multi-ottava + "craquelure" casuali): le texture
  periodiche creano ambiguita' di fase e falserebbero il test senza che ci sia
  nulla di rotto nella pipeline;
- la seconda immagine e' la prima trasformata con un'affine nota PIU' una
  deformazione ondulata nota (simula la tela che si e' leggermente flessa);
- contrasto invertito e curva di luminosita' diversa, per verificare che il
  matching sia davvero cross-modale e non stia semplicemente confrontando
  livelli di grigio.

Poi esegue la pipeline e confronta la trasformazione stimata con quella vera,
pixel per pixel su una griglia di punti di controllo. Il test passa se l'errore
mediano e' sotto il pixel.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import cv2
from PIL import Image as PILImage

import tiff_memmap
import image_io
import align_paintings
import align_batch
import apply_transform
from transform_model import Transform

H, W = 1600, 1300
ANGLE = 0.25          # gradi
SCALE = 1.004
TX, TY = 13.0, -8.0
WAVE_AMP = 6.0        # ampiezza della deformazione non rigida, in px
WAVE_PERIOD = 900.0


def synthetic_texture(h, w, seed=0):
    """Rumore multi-ottava + linee casuali: strutturato ma non periodico."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for octave in range(5):
        s = 2 ** octave
        small = rng.random((max(2, h // (8 * s)), max(2, w // (8 * s)))).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) / s
    img = cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX)
    for _ in range(300):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (p0[0] + int(rng.integers(-120, 120)), p0[1] + int(rng.integers(-120, 120)))
        cv2.line(img, p0, p1, float(rng.integers(0, 255)), 1, cv2.LINE_AA)
    return img.astype(np.uint8)


def true_mapping(x, y):
    """Trasformazione vera fixed -> moving (affine nota + onda nota)."""
    th = np.radians(ANGLE)
    a, b = SCALE * np.cos(th), -SCALE * np.sin(th)
    c, d = SCALE * np.sin(th), SCALE * np.cos(th)
    cx, cy = W / 2.0, H / 2.0
    xr = a * (x - cx) + b * (y - cy) + cx + TX
    yr = c * (x - cx) + d * (y - cy) + cy + TY
    xr = xr + WAVE_AMP * np.sin(2 * np.pi * y / WAVE_PERIOD)
    yr = yr + WAVE_AMP * np.cos(2 * np.pi * x / WAVE_PERIOD)
    return xr, yr


def main():
    work = tempfile.mkdtemp(prefix='align_test_')
    print(f"cartella di lavoro: {work}\n")
    try:
        base = synthetic_texture(H, W)

        # moving: campiona la texture nella posizione trasformata (cosi' la
        # mappa fixed->moving e' esattamente true_mapping), poi cambia la
        # "modalita'": contrasto invertito e gamma diversa.
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        mx, my = true_mapping(X, Y)
        # per costruire la moving serve la mappa inversa; la si ottiene
        # numericamente con qualche iterazione di punto fisso (la deformazione
        # e' piccola, converge subito).
        ix, iy = X.copy(), Y.copy()
        for _ in range(12):
            fx, fy = true_mapping(ix, iy)
            ix += (X - fx)
            iy += (Y - fy)
        moving = cv2.remap(base, ix.astype(np.float32), iy.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        moving = (255 - moving).astype(np.float32)
        moving = np.clip(((moving / 255.0) ** 1.6) * 255.0, 0, 255).astype(np.uint8)

        p_fixed = os.path.join(work, 'fixed.tif')
        p_moving = os.path.join(work, 'moving.tif')
        for path, img in ((p_fixed, base), (p_moving, moving)):
            mm = tiff_memmap.create_tiff_memmap(path, H, W, samples=1, dtype=np.uint8)
            mm[:] = img
            mm.flush()
            del mm

        p_out = os.path.join(work, 'aligned.tif')
        p_par = os.path.join(work, 'params.json')
        T, _ = align_paintings.align(p_fixed, p_moving, p_out, p_par, factor=1,
                                     block=192, step=96, refine_block=96, refine_step=48,
                                     tile_size=512, diagnostics_dir=os.path.join(work, 'diag'))

        # --- confronto con la trasformazione vera, su punti di controllo ---
        gx, gy = np.meshgrid(np.linspace(150, W - 150, 9), np.linspace(150, H - 150, 9))
        gx, gy = gx.ravel(), gy.ravel()
        tx_true, ty_true = true_mapping(gx, gy)

        ones = np.ones_like(gx)
        pts = np.stack([gx, gy, ones], axis=1)
        est = pts @ T.M.T
        if T.field is not None:
            fx = cv2.remap(T.field, (gx / T.grid_step).astype(np.float32).reshape(-1, 1),
                           (gy / T.grid_step).astype(np.float32).reshape(-1, 1),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
            est = est + fx
        err = np.hypot(est[:, 0] - tx_true, est[:, 1] - ty_true)

        print("\n" + "=" * 72)
        print("CONFRONTO CON LA TRASFORMAZIONE VERA (81 punti di controllo)")
        print(f"  errore mediano: {np.median(err):.3f} px")
        print(f"  errore p95:     {np.percentile(err, 95):.3f} px")
        print(f"  errore max:     {err.max():.3f} px")

        # --- apply_transform: caso VALIDO, stessa ripresa riprocessata ---
        # Non e' "un canale IR": e' la STESSA ripresa della moving (stesso remap
        # ix/iy, quindi geometria identica per costruzione) esportata senza il
        # cambio di modalita'. E' il caso d'uso legittimo di apply_transform.py:
        # un raster che vive nelle coordinate di quello scatto.
        p_same = os.path.join(work, 'stessa_ripresa.tif')
        same = cv2.remap(base, ix.astype(np.float32), iy.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mm = tiff_memmap.create_tiff_memmap(p_same, H, W, samples=1, dtype=np.uint8)
        mm[:] = same
        mm.flush()
        del mm
        p_same_out = os.path.join(work, 'stessa_ripresa_allineata.tif')
        print("\n" + "=" * 72)
        print("CASO VALIDO: riapplicazione a un raster della stessa ripresa")
        apply_transform.apply_saved(p_par, p_same, p_same_out, reference_path=p_fixed)

        arr_f, _ = tiff_memmap.open_memmap(p_fixed)
        arr_o, _ = tiff_memmap.open_memmap(p_same_out)
        inner_f = np.asarray(arr_f[200:H - 200, 200:W - 200]).astype(np.float32)
        inner_o = np.asarray(arr_o[200:H - 200, 200:W - 200]).astype(np.float32)
        rms = float(np.sqrt(np.mean((inner_f - inner_o) ** 2)))
        print(f"  differenza RMS fixed vs riallineata (stessa modalita'): {rms:.2f} livelli/255")

        # --- apply_transform: caso da RIFIUTARE, altra esposizione ---
        # Simula un secondo scatto: stessa scena, ma la camera si e' spostata di
        # ~25 px nel frattempo. La trasformazione salvata non la allinea, e lo
        # script deve accorgersene PRIMA di scrivere.
        print("\n" + "=" * 72)
        print("CASO DA RIFIUTARE: altra esposizione (camera spostata di 25 px)")
        p_other = os.path.join(work, 'altra_esposizione.tif')
        shifted = cv2.warpAffine(moving, np.float32([[1, 0, 25], [0, 1, -25]]), (W, H),
                                 borderMode=cv2.BORDER_REFLECT)
        mm = tiff_memmap.create_tiff_memmap(p_other, H, W, samples=1, dtype=np.uint8)
        mm[:] = shifted
        mm.flush()
        del mm
        p_other_out = os.path.join(work, 'non_deve_esistere.tif')
        rifiutato = False
        try:
            apply_transform.apply_saved(p_par, p_other, p_other_out, reference_path=p_fixed)
        # Attenzione: la variabile di `except ... as <nome>` viene CANCELLATA da
        # Python all'uscita del blocco, anche se esisteva prima. Qui non si puo'
        # usare `err`, che piu' sopra contiene gli errori misurati sui punti di
        # controllo e serve ancora dopo.
        except apply_transform.TrasformazioneNonApplicabile as exc:
            rifiutato = True
            print(f"  rifiutata correttamente: {str(exc).splitlines()[0]}")
        non_scritto = not os.path.exists(p_other_out)
        if not rifiutato:
            print("  ERRORE: la trasformazione e' stata applicata a un'esposizione diversa")
        if not non_scritto:
            print("  ERRORE: e' stato creato un file di output nonostante il rifiuto")

        # --- lotto con formati e dimensioni diversi ---------------------
        # Il caso d'uso vero: piu' bande sullo stesso riferimento, salvate in
        # formati diversi. Interessa soprattutto che funzionino le due strade
        # nuove: la lettura di formati non mappabili in memoria (JPEG, PNG, TIFF
        # compresso) e la normalizzazione di scala quando una ripresa ha
        # dimensioni diverse dal riferimento.
        print("\n" + "=" * 72)
        print("LOTTO: formati misti e dimensioni diverse")
        cartella_lotto = os.path.join(work, 'lotto')
        os.makedirs(cartella_lotto, exist_ok=True)

        p_jpg = os.path.join(cartella_lotto, 'banda_jpeg.jpg')
        PILImage.fromarray(moving).save(p_jpg, quality=92)

        p_png = os.path.join(cartella_lotto, 'banda_png_piccola.png')
        piccola = cv2.resize(moving, (int(W * 0.7), int(H * 0.7)), interpolation=cv2.INTER_AREA)
        PILImage.fromarray(piccola).save(p_png)

        p_lzw = os.path.join(cartella_lotto, 'banda_tiff_lzw.tif')
        PILImage.fromarray(moving).save(p_lzw, compression='tiff_lzw')

        # 16 bit che occupa solo una fetta stretta della scala, come una
        # radiografia. Verifica due cose insieme: che la stima non si faccia
        # confondere dalla profondita' diversa, e che le immagini di controllo
        # restino leggibili (un .astype(np.uint8) diretto le ridurrebbe a rumore).
        p_16 = os.path.join(cartella_lotto, 'banda_16bit.tif')
        mm = tiff_memmap.create_tiff_memmap(p_16, H, W, samples=1, dtype=np.uint16)
        mm[:] = moving.astype(np.uint16) * 12 + 30000
        mm.flush()
        del mm

        riepilogo = align_batch.allinea_lotto(
            p_fixed, [p_moving, p_jpg, p_png, p_lzw, p_16],
            os.path.join(work, 'risultati_lotto'),
            factor=1, refine_block=96, tile_size=512, verbose=False)

        lotto_ok = all(v['esito'] == 'ok' for v in riepilogo['immagini'])
        prodotti = [v for v in riepilogo['immagini']
                    if v.get('output') and os.path.exists(v['output'])]
        if not lotto_ok:
            print("  ERRORE: non tutte le immagini del lotto sono state allineate")
        if len(prodotti) != 5:
            print(f"  ERRORE: prodotti {len(prodotti)} file su 5")

        # l'output a 16 bit deve restare a 16 bit
        out16 = [v for v in riepilogo['immagini'] if '16bit' in v['nome']]
        prof_ok = bool(out16) and image_io.info(out16[0]['output']).dtype == np.uint16
        if not prof_ok:
            print("  ERRORE: la profondita' a 16 bit non e' stata preservata in uscita")

        # Il file scritto deve contenere ESATTAMENTE cio' che la trasformazione
        # produce. Le verifiche qui sopra misurano la trasformazione sulle
        # sorgenti, non il file: per questo non si e' visto per dieci giorni che
        # i TIFF a un canale venivano letti spostati di 10 byte (5 px a 16 bit).
        # Si rilegge con Pillow, un lettore indipendente dal nostro.
        T_out = Transform.load(p_par)
        letto = np.asarray(PILImage.open(p_out))
        atteso = T_out.warp_tile_from(image_io.apri(p_moving, verbose=False)[0], 0, 0, H, W)
        file_ok = bool(np.array_equal(letto, atteso))
        print(f"  file allineato riletto con Pillow: "
              f"{'coincide' if file_ok else 'NON coincide'} con la trasformazione")

        ok = (np.median(err) < 1.0 and rms < 12.0 and rifiutato and non_scritto
              and lotto_ok and len(prodotti) == 5 and prof_ok and file_ok)
        print("\n" + "=" * 72)
        print("TEST SUPERATO" if ok else "TEST FALLITO")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
