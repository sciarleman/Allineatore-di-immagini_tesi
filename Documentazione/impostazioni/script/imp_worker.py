"""Una configurazione del programma, in un processo a se': tempo, memoria, esito.

Passa per align_paintings.align con gli stessi argomenti che gli da' il lotto
dell'interfaccia (tre scale 'auto', solo stima e verifica). Scrive una riga JSON
in impostazioni/risultati.jsonl.

    python imp_worker.py '<json della configurazione>'
"""
import os, sys, io, json, time, re, contextlib, shutil
import numpy as np

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, validation as val

QUI = os.path.dirname(os.path.abspath(__file__))
USCITA = os.path.join(QUI, "impostazioni"); os.makedirs(USCITA, exist_ok=True)
cfg = json.loads(sys.argv[1])
tmp = os.path.join(USCITA, "tmp_" + cfg['id']); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)

rec = dict(cfg)
buf = io.StringIO()
t0 = time.time()
try:
    with contextlib.redirect_stdout(buf):
        T, rep = ap.align(cfg['fixed'], cfg['moving'], None, params_path=os.path.join(tmp, "p.json"),
                          factor=cfg['factor'], nonrigid=cfg['nonrigid'], min_conf=cfg['min_conf'],
                          refine_block=cfg['refine_block'], refine_step=max(8, cfg['refine_block'] // 2),
                          diagnostics_dir=os.path.join(tmp, "diag"), tre_scale='auto', verbose=False)
    rec['t_totale_s'] = time.time() - t0
    mem = ap.peak_memory_gb()
    rec['picco_memoria_gb'] = None if mem is None else mem[0]
    log = buf.getvalue()
    m = re.search(r"stima completata in ([\d.]+)(s| min)", log)
    rec['t_stima_log'] = None if not m else float(m.group(1)) * (60 if 'min' in m.group(2) else 1)
    m = re.search(r"Lettura anteprime.*?\n.*? in ([\d.]+)(s| min)", log)
    rec['t_anteprime_log'] = None if not m else float(m.group(1)) * (60 if 'min' in m.group(2) else 1)
    v = rep['verifica']
    rec['prima'] = {k: v['prima'].get(k) for k in ('residuo_mediano_px', 'n_affidabili', 'n_campioni')}
    rec['dopo'] = {k: v['dopo'].get(k) for k in ('residuo_mediano_px', 'residuo_mediano_ic90', 'n_affidabili', 'n_campioni',
                                                 'quota_sopra_1px', 'quota_sopra_2px', 'quota_sopra_5px', 'quota_sopra_10px',
                                                 'residuo_p95_px', 'residuo_max_px', 'copertura')}
    rec['bordo'] = [{k: f[k] for k in ('da_px', 'a_px', 'n_affidabili', 'n_misurate', 'mediana_px', 'p90_px', 'quota_sopra_5px')}
                    for f in v['bordo']['fasce']]
    rec['zone'] = {'non_seguito': sum(z['tipo'] == 'non seguito' for z in v['zone_critiche']),
                   'non_misurabile': sum(z['tipo'] == 'non misurabile' for z in v['zone_critiche'])}
    rec['mind'] = v.get('mind')
    rec['ripiego'] = rep.get('ripiego')
    s = rep['stima']
    rec['stima'] = {'pass1': s.get('pass1'), 'pre_allineamento_px': s.get('pre_allineamento_px'),
                    'block': s.get('block'), 'refine_block': s.get('refine_block'),
                    'campo_grossolano': {k: (s.get('campo_grossolano') or {}).get(k) for k in
                                         ('applicato', 'portata_px_full', 'ampiezza_max_px_full', 'guadagno_pct')},
                    'nonrigid': s.get('nonrigid'), 'nonrigid_crossval': s.get('nonrigid_crossval'),
                    'ordine': s.get('ordine_media_locale')}
    rec['primo_tentativo'] = (rep.get('stima_primo_tentativo') or {}).get('pass1')
    A = T.M[:, :2]
    rec['affine'] = {'scala_x': float(np.hypot(A[0, 0], A[1, 0])), 'scala_y': float(np.hypot(A[0, 1], A[1, 1])),
                     'rot_gradi': float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 't': T.M[:, 2].tolist()}
    if T.field is not None:
        mag = np.hypot(T.field[..., 0], T.field[..., 1])
        rec['campo'] = {'rms_px': float(np.sqrt((mag ** 2).mean())), 'max_px': float(mag.max()), 'passo_px': T.grid_step}
    # le finestre della griglia fissa, per confrontare le configurazioni sulle stesse
    arr_f, _ = image_io.apri(cfg['fixed'], verbose=False); arr_m, _ = image_io.apri(cfg['moving'], verbose=False)
    r = val.residual_check(arr_f, arr_m, T, verbose=False, adattiva=False)
    rec['finestre'] = [[int(c[0]), int(c[1]), float(np.hypot(c[2], c[3])), float(c[4])]
                       for c in (r.get('campioni') or []) + (r.get('campioni_deboli') or [])]
    np.save(os.path.join(USCITA, cfg['id'] + "_M.npy"), T.M)
    if T.field is not None:
        np.save(os.path.join(USCITA, cfg['id'] + "_campo.npy"), T.field)
    open(os.path.join(USCITA, cfg['id'] + "_log.txt"), "w", encoding="utf-8").write(log)
except Exception as exc:
    rec['t_totale_s'] = time.time() - t0
    rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    open(os.path.join(USCITA, cfg['id'] + "_log.txt"), "w", encoding="utf-8").write(buf.getvalue())
shutil.rmtree(tmp, ignore_errors=True)
with open(os.path.join(USCITA, "risultati.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
d = rec.get('dopo') or {}
print(cfg['id'], "->", rec.get('errore') or f"{d.get('residuo_mediano_px')} px, {d.get('n_affidabili')}/{d.get('n_campioni')}, "
      f"{rec['t_totale_s']:.0f} s, {rec.get('picco_memoria_gb')} GB", flush=True)
