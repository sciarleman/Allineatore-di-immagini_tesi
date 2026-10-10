"""Effetto del fattore di riduzione non intero (ingressi non TIFF).

Stessi pixel in due formati: TIFF non compresso (la lettura ridotta restituisce
il fattore intero giusto) e PNG (restituisce altezza/altezza ridotta). Le
dimensioni sono scelte perche' riferimento e immagine da allineare abbiano
RESTI DIVERSI nella divisione per il fattore: e' li' che l'errore non si
cancella. Si confronta il residuo verificato a piena risoluzione.

    python prova_fattore.py            misura
"""
import os, sys, io, json, time, contextlib, shutil
import numpy as np
from PIL import Image

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, tiff_memmap, align_paintings as ap, validation as val

Image.MAX_IMAGE_PIXELS = None
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
QUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fattore"); os.makedirs(QUI, exist_ok=True)
# ritaglio dal centro del dipinto; il riferimento ha lati non multipli di 2, 4, 8
X0, Y0 = 4000, 5000
FORME = {"VIS": (8003, 6001), "NIR850": (8000, 6000)}          # (altezza, larghezza)
SORG = {"VIS": "VIS_FraGalgario_TOT.tif", "NIR850": "NIR850_FraGalgario_TOT.tif"}


def prepara():
    for nome, (h, w) in FORME.items():
        tif, png = os.path.join(QUI, nome + ".tif"), os.path.join(QUI, nome + ".png")
        if os.path.exists(tif) and os.path.exists(png):
            continue
        arr, _ = image_io.apri(os.path.join(FG, SORG[nome]), verbose=False)
        rit = np.ascontiguousarray(arr[Y0:Y0 + h, X0:X0 + w])
        s = 1 if rit.ndim == 2 else rit.shape[2]
        wr = tiff_memmap.create_tiff_memmap(tif, h, w, samples=s, dtype=rit.dtype, sequenziale=True)
        wr.scrivi(rit); wr.chiudi()
        Image.fromarray(rit).save(png, compress_level=1)


def corsa(est, fattore):
    fx, mv = os.path.join(QUI, "VIS" + est), os.path.join(QUI, "NIR850" + est)
    tmp = os.path.join(QUI, "tmp"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T, rep = ap.align(fx, mv, None, params_path=os.path.join(tmp, "p.json"), factor=fattore,
                          diagnostics_dir=os.path.join(tmp, "d"), tre_scale='auto', verbose=False)
    log = buf.getvalue()
    fattori = [l.strip() for l in log.splitlines() if "fattore" in l and "riferimento" in l]
    d = rep['verifica']['dopo']
    # il residuo come vettore, finestra per finestra, sui TIFF (stessi pixel): dice se l'errore cresce verso l'angolo
    af, _ = image_io.apri(os.path.join(QUI, "VIS.tif"), verbose=False); am, _ = image_io.apri(os.path.join(QUI, "NIR850.tif"), verbose=False)
    r = val.residual_check(af, am, T, verbose=False, adattiva=False)
    c = np.array([[v[0], v[1], v[2], v[3]] for v in r['campioni']])
    A = np.column_stack([c[:, 0], c[:, 1], np.ones(len(c))])
    gx = np.linalg.lstsq(A, c[:, 2], rcond=None)[0]; gy = np.linalg.lstsq(A, c[:, 3], rcond=None)[0]
    return {'est': est, 'fattore': fattore, 'fattori_letti': fattori, 'mediana': d['residuo_mediano_px'],
            'ic': d['residuo_mediano_ic90'], 'n': d['n_affidabili'], 'q1': d['quota_sopra_1px'], 'max': d['residuo_max_px'],
            'pendenza_x_px_su_larghezza': float(gx[0] * FORME["VIS"][1]), 'pendenza_y_px_su_altezza': float(gy[1] * FORME["VIS"][0]),
            'M': T.M.tolist()}


if __name__ == "__main__":
    prepara()
    etichetta = sys.argv[1] if len(sys.argv) > 1 else "prima"
    FORMATI = tuple(sys.argv[2:]) or (".tif", ".png")
    for fattore in (2, 4, 8):
        ris = {}
        for est in FORMATI:
            ris[est] = corsa(est, fattore)
            r = ris[est]
            print(f"[{etichetta}] f{fattore} {est}: mediana {r['mediana']:.3f} px [{r['ic'][0]:.2f}-{r['ic'][1]:.2f}] su {r['n']} finestre, "
                  f">1px {100 * r['q1']:.0f}%, max {r['max']:.2f} | residuo che cresce: {r['pendenza_x_px_su_larghezza']:+.2f} px sulla larghezza, "
                  f"{r['pendenza_y_px_su_altezza']:+.2f} px sull'altezza | {r['fattori_letti'][:1]}", flush=True)
            with open(os.path.join(QUI, "risultati.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(r, etichetta=etichetta)) + "\n")
        Mt, Mp = np.array(ris[FORMATI[0]]["M"]), np.array(ris[FORMATI[-1]]["M"])
        H, W = FORME["VIS"]
        ang = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1], [W / 2, H / 2, 1]], float)
        d = np.linalg.norm(ang @ Mt.T - ang @ Mp.T, axis=1)
        print(f"[{etichetta}] f{fattore}: le due affini (TIFF contro PNG) distano {d[0]:.2f} px all'origine, {d[4]:.2f} al centro, "
              f"{d[2]:.2f} all'angolo opposto", flush=True)
