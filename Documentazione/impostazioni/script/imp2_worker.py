"""Una configurazione delle impostazioni avanzate, in un processo a se'.

Passa per align_paintings.align con gli argomenti che gli da' il lotto
dell'interfaccia (tre scale 'auto', solo stima e verifica) e poi misura in tre
modi: la verifica del programma, la verifica a griglia fissa che il lotto ora
aggiunge, e una griglia FITTA di finestre da 512 px su tutto il foglio, bordi e
angoli compresi. Scrive una riga in risultati.jsonl.

    python imp2_worker.py FG NIR850 2 128 0.10
"""
import os, sys, io, json, time, contextlib, shutil
import numpy as np
import cv2

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, block_matching as bm, verifica_fissa as vf

QUI = os.path.dirname(os.path.abspath(__file__))
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
CK = r"C:\Users\carlo\Desktop\checkà\tiff"
DATI = {
    "FG": (os.path.join(FG, "VIS_FraGalgario_TOT.tif"), {
        "NIR850": "NIR850_FraGalgario_TOT.tif", "NIR950": "NIR950_FraGalgario_TOT.tif",
        "UVF": "UVF_FraGalgario_TOT.tif", "UVR": "UVR_FraGalgario_TOT.tif",
        "SWIR": "big_SWIR_FraGalgario_TOT.tif", "RX_Cappello": "RX_FraGalgario_TOT_Cappellochiaro.tif",
        "RX_Testa": "RX_FraGalgarioTOT_TestaChiara.tif"}, FG),
    "CK": (os.path.join(CK, "VIS.tif"), {"NIR": "NIR.tif", "UV": "UV.tif"}, CK),
}
ins, banda, fattore, blocco, conf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
fixed = DATI[ins][0]
mov = os.path.join(DATI[ins][2], DATI[ins][1][banda])
ident = f"{ins}_{banda}_f{fattore}_b{blocco}_c{int(round(conf * 100)):02d}"


def griglia_fitta(arr_f, arr_m, T, passo=768, lato=512):
    H, W = arr_f.shape[:2]
    win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
    out = []
    for cy in range(lato // 2, H - lato // 2 + 1, passo):
        for cx in range(lato // 2, W - lato // 2 + 1, passo):
            y0, x0 = cy - lato // 2, cx - lato // 2
            f = np.asarray(arr_f[y0:y0 + lato, x0:x0 + lato]); m = T.warp_tile_from(arr_m, y0, x0, lato, lato)
            if f.ndim == 3: f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
            gf, gm = bm.gradient_magnitude(f), bm.gradient_magnitude(m)
            if gf.std() < 1e-4 or gm.std() < 1e-4:
                continue
            (dx, dy), c = cv2.phaseCorrelate(gf.astype(np.float64), gm.astype(np.float64), win)
            out.append((cx, cy, float(np.hypot(dx, dy)), float(c)))
    return np.array(out)


tmp = os.path.join(QUI, "tmp_" + ident); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
rec = {'insieme': ins, 'banda': banda, 'fattore': fattore, 'refine_block': blocco, 'min_conf': conf}
buf = io.StringIO()
t0 = time.time()
try:
    with contextlib.redirect_stdout(buf):
        T, rep = ap.align(fixed, mov, None, params_path=os.path.join(tmp, "p.json"), factor=fattore,
                          nonrigid=True, min_conf=conf, refine_block=blocco,
                          refine_step=max(8, blocco // 2), diagnostics_dir=os.path.join(tmp, "diag"),
                          tre_scale='auto', verbose=False)
    rec['t_align_s'] = time.time() - t0
    s = rep['stima']
    liv = s.get('campo_grossolano_livelli') or [s.get('campo_grossolano') or {}]
    rec['livelli'] = [{k: l.get(k) for k in ('riduzione', 'applicato', 'ampiezza_max_px_full')} for l in liv]
    rec['ripiego'] = bool(rep.get('ripiego'))
    nr = s.get('nonrigid') or {}
    rec['campo_fine'] = {k: nr.get(k) for k in ('applicato', 'smooth_px_full', 'grid_step_px_full', 'blocchi', 'inlier_affine')}
    rec['crossval'] = {k: (s.get('nonrigid_crossval') or {}).get(k) for k in ('residuo_prima_px', 'residuo_dopo_px', 'guadagno_pct')}
    rec['pass1'] = s.get('pass1')
    if T.field is not None:
        mag = np.hypot(T.field[..., 0], T.field[..., 1])
        rec['campo'] = {'rms_px': float(np.sqrt((mag ** 2).mean())), 'max_px': float(mag.max())}
    v = rep['verifica']['dopo']
    rec['verifica'] = {k: v.get(k) for k in ('residuo_mediano_px', 'residuo_mediano_ic90', 'n_affidabili', 'n_campioni',
                                             'quota_sopra_1px', 'quota_sopra_2px', 'quota_sopra_5px', 'residuo_max_px')}
    mem = ap.peak_memory_gb()
    rec['picco_memoria_gb'] = None if mem is None else mem[0]
    arr_f, _ = image_io.apri(fixed, verbose=False); arr_m, _ = image_io.apri(mov, verbose=False)
    rec['HW'] = [int(arr_f.shape[0]), int(arr_f.shape[1])]
    t1 = time.time()
    with contextlib.redirect_stdout(buf):
        fissa = vf.verifica_griglia_fissa(fixed, mov, T, None, rep, os.path.join(tmp, "diag"), arr_fixed=arr_f)
    rec['t_fissa_s'] = time.time() - t1
    rec['fissa'] = {k: fissa.get(k) for k in ('residuo_mediano_px', 'n_affidabili', 'n_campioni', 'quota_sopra_5px',
                                              'quota_sopra_10px', 'residuo_max_px')}
    rec['fissa']['n_oltre_10px'] = len(fissa.get('oltre_10px') or [])
    rec['fissa']['n_in_gruppo'] = sum(bool(o.get('in_gruppo')) for o in fissa.get('oltre_10px') or [])
    rec['fissa_finestre'] = fissa.get('finestre')
    rec['finestre'] = griglia_fitta(arr_f, arr_m, T).tolist()
    np.save(os.path.join(QUI, "trasformazioni", ident + "_M.npy"), T.M)
    if T.field is not None:
        np.save(os.path.join(QUI, "trasformazioni", ident + "_campo.npy"), T.field)
except Exception as exc:
    import traceback
    rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    buf.write("\n" + traceback.format_exc())
rec['t_totale_s'] = time.time() - t0
open(os.path.join(QUI, "log", ident + ".txt"), "w", encoding="utf-8").write(buf.getvalue())
shutil.rmtree(tmp, ignore_errors=True)
with open(os.path.join(QUI, "risultati.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
print(ident, "->", rec.get('errore') or (
    f"verifica {rec['verifica']['residuo_mediano_px']:.3f} px ({rec['verifica']['n_affidabili']}/{rec['verifica']['n_campioni']}); "
    f"griglia fissa {rec['fissa']['residuo_mediano_px']:.3f} px, oltre 10 px {rec['fissa']['n_oltre_10px']}; "
    f"{rec['t_align_s']:.0f} s di align, {rec['picco_memoria_gb']:.1f} GB"), flush=True)
