"""UV o NIR dell'affresco a precisione massima con la zona disegnata dall'utente
(il poligono della run delle 13:39 del 6 ottobre), per la strada vera del
programma di oggi: due livelli grossolani, verifica di sempre e griglia fissa
dentro la zona. Poi la griglia fitta su tutto il foglio, da leggere dentro la
zona. Sola stima e verifica: non scrive immagini.

    python zona_worker.py UV
    python zona_worker.py NIR
    python zona_worker.py NIR 2     (precisa)
"""
import os, sys, io, json, time, contextlib, shutil, threading, ctypes
import numpy as np
import cv2

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, block_matching as bm, verifica_fissa as vf

QUI = os.path.dirname(os.path.abspath(__file__))
USC = os.path.join(QUI, "zona"); os.makedirs(USC, exist_ok=True)
CK = r"C:\Users\carlo\Desktop\checkà\tiff"
PAR = "C:/Users/carlo/Desktop/check" + chr(0xe0) + "/parametri"
banda = sys.argv[1]
fattore = int(sys.argv[2]) if len(sys.argv) > 2 else 1
fixed, mov = os.path.join(CK, "VIS.tif"), os.path.join(CK, banda + ".tif")
ident = f"{banda}_f{fattore}_zona"


class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                          "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def libera_gb():
    m = _Mem(); m.dwLength = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


minimo = [libera_gb()]


def guardia():
    # il PC e' gia' andato in crash per memoria esaurita: sotto 0.5 GB liberi la
    # prova si ferma da sola invece di trascinare giu' Windows
    while True:
        g = libera_gb(); minimo[0] = min(minimo[0], g)
        if g < 0.5:
            open(os.path.join(USC, ident + "_INTERROTTA.txt"), "w").write(f"memoria libera {g:.2f} GB dopo {time.time() - t0:.0f} s\n")
            os._exit(3)
        time.sleep(0.5)


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


t0 = time.time()
threading.Thread(target=guardia, daemon=True).start()
maschera = cv2.imdecode(np.fromfile(os.path.join(PAR, "UV (2026-10-06 13-39)_maschera.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
tmp = os.path.join(USC, "tmp_" + ident); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
rec = {'banda': banda, 'fattore': fattore, 'zona': '13-39', 'maschera_shape': list(maschera.shape), 'libera_all_inizio_gb': minimo[0]}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        T, rep = ap.align(fixed, mov, None, params_path=os.path.join(tmp, "p.json"), factor=fattore,
                          nonrigid=True, min_conf=0.10, refine_block=128, refine_step=64,
                          diagnostics_dir=os.path.join(tmp, "diag"), tre_scale='auto', mask=maschera,
                          verbose=False)
    rec['t_align_s'] = time.time() - t0
    s = rep['stima']
    liv = s.get('campo_grossolano_livelli') or [s.get('campo_grossolano') or {}]
    rec['livelli'] = [{k: l.get(k) for k in ('riduzione', 'applicato', 'portata_px_full', 'ampiezza_max_px_full')} for l in liv]
    rec['ripiego'] = bool(rep.get('ripiego'))
    v = rep['verifica']
    CHIAVI = ('residuo_mediano_px', 'residuo_p95_px', 'residuo_max_px', 'n_affidabili', 'n_campioni', 'quota_sopra_1px',
              'quota_sopra_5px', 'quota_sopra_10px', 'lato_finestra_px')
    rec['verifica'] = {k: v['dopo'].get(k) for k in CHIAVI}
    rec['verifica_prima'] = {k: v['prima'].get(k) for k in CHIAVI}
    rec['bordo'] = (v.get('bordo') or {}).get('fasce')
    rec['zone_critiche'] = len(v.get('zone_critiche') or [])
    np.save(os.path.join(USC, ident + "_M.npy"), T.M)
    if T.field is not None:
        np.save(os.path.join(USC, ident + "_campo.npy"), T.field)
        rec['passo_campo'] = float(T.grid_step)
    mem = ap.peak_memory_gb(); rec['picco_dopo_align_gb'] = None if mem is None else mem[0]
    t1 = time.time()
    with contextlib.redirect_stdout(buf):
        fissa = vf.verifica_griglia_fissa(fixed, mov, T, maschera, rep, os.path.join(USC, ident + "_diag"))
    rec['t_fissa_s'] = time.time() - t1
    rec['fissa'] = {k: fissa.get(k) for k in CHIAVI + ('fasce', 'oltre_10px', 'contorno_cieco', 'finestre')}
    del maschera
    arr_f, _ = image_io.apri(fixed, verbose=False); arr_m, _ = image_io.apri(mov, verbose=False)
    rec['HW'] = [int(arr_f.shape[0]), int(arr_f.shape[1])]
    t1 = time.time()
    rec['finestre'] = griglia_fitta(arr_f, arr_m, T).tolist()
    rec['t_fitta_s'] = time.time() - t1
    mem = ap.peak_memory_gb()
    rec['picco_memoria_gb'] = None if mem is None else mem[0]
except Exception as exc:
    import traceback
    rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    buf.write("\n" + traceback.format_exc())
rec['t_totale_s'] = time.time() - t0
rec['libera_minima_gb'] = minimo[0]
open(os.path.join(USC, ident + "_log.txt"), "w", encoding="utf-8").write(buf.getvalue())
shutil.rmtree(tmp, ignore_errors=True)
with open(os.path.join(USC, "risultati.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
print(ident, "->", rec.get('errore') or (
    f"verifica {rec['verifica']['residuo_mediano_px']:.3f} px, {rec['verifica']['n_affidabili']}/{rec['verifica']['n_campioni']}, "
    f">5px {rec['verifica']['quota_sopra_5px']:.0%}, max {rec['verifica']['residuo_max_px']:.1f}; "
    f"fissa {rec['fissa']['residuo_mediano_px']:.3f} px, {rec['fissa']['n_affidabili']}/{rec['fissa']['n_campioni']}, oltre 10 px: {len(rec['fissa']['oltre_10px'] or [])}; "
    f"livelli {rec['livelli']}; {rec['t_align_s']:.0f} s di align, {rec['t_totale_s']:.0f} s in tutto; "
    f"picco {rec.get('picco_memoria_gb')} GB, memoria libera minima {rec['libera_minima_gb']:.1f} GB"), flush=True)
