"""Confronto generale: come partire (traslazione di oggi, traslazione corretta,
posa grossolana subito) e con che portata del campo grossolano, su due dipinti.

Non modifica il programma: le varianti si ottengono chiamando le funzioni del
motore con argomenti diversi o sostituendo a tempo la sola traslazione
grossolana. Ogni stima viene scritta subito su risultati.jsonl, cosi' una
interruzione non perde il lavoro fatto.

    python confronto_generale.py FG 2      (Fra Galgario, fattore 2)
    python confronto_generale.py CK 1      (dipinto nuovo, fattore 1)
"""
import os, sys, time, json
import numpy as np
import cv2

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, validation as val, block_matching as bm
from transform_model import Transform

QUI = os.path.dirname(os.path.abspath(__file__))
USCITA = os.path.join(QUI, "generale"); os.makedirs(USCITA, exist_ok=True)
RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)
IMM = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
DATI = {
    "FG": (os.path.join(IMM, "VIS_FraGalgario_TOT.tif"), {
        "NIR850": os.path.join(IMM, "NIR850_FraGalgario_TOT.tif"),
        "NIR950": os.path.join(IMM, "NIR950_FraGalgario_TOT.tif"),
        "UVF": os.path.join(IMM, "UVF_FraGalgario_TOT.tif"),
        "UVR": os.path.join(IMM, "UVR_FraGalgario_TOT.tif"),
        "RX_Cappello": os.path.join(IMM, "RX_FraGalgario_TOT_Cappellochiaro.tif"),
        "RX_Testa": os.path.join(IMM, "RX_FraGalgarioTOT_TestaChiara.tif")}),
    "CK": (os.path.join(RUN, "VIS.png"), {"NIR": os.path.join(RUN, "NIR.png"),
                                         "UV": os.path.join(RUN, "UV.png")}),
}
insieme, fattore = sys.argv[1], int(sys.argv[2])
solo = sys.argv[3:]
_originale = bm.coarse_translation


def dico(*a):
    print(*a, flush=True)


def traslazione_corretta(ref_grad, mov_grad, lato=512, min_conf=0.03, verbose=True):
    """bm.coarse_translation, ma le due miniature si riducono con lo STESSO
    fattore e si confronta la parte comune a partire dall'origine."""
    h, w = ref_grad.shape[:2]
    scala = min(1.0, lato / max(h, w))
    if scala < 1.0:
        a = cv2.resize(ref_grad, None, fx=scala, fy=scala, interpolation=cv2.INTER_AREA)
        b = cv2.resize(mov_grad, None, fx=scala, fy=scala, interpolation=cv2.INTER_AREA)
    else:
        a, b = ref_grad, mov_grad
    hh, ww = min(a.shape[0], b.shape[0]) // 2 * 2, min(a.shape[1], b.shape[1]) // 2 * 2
    a, b = a[:hh, :ww], b[:hh, :ww]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0, 0
    win = cv2.createHanningWindow((ww, hh), cv2.CV_64F)
    (dx, dy), conf = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win)
    if conf < min_conf:
        return 0, 0
    return int(round(dx / max(scala, 1e-9))), int(round(dy / max(scala, 1e-9)))


def prepara(fixed, moving, factor):
    """Le anteprime come le prepara align(): riduzione e normalizzazione di scala."""
    inf = image_io.info(fixed)
    target = max(inf.height, inf.width) // factor
    pf, ff = image_io.leggi_ridotta(fixed, lato_lungo_target=target, in_grigio=True, verbose=False)
    pm, fm = image_io.leggi_ridotta(moving, lato_lungo_target=target, in_grigio=True, verbose=False)
    rapporto = float(np.sqrt((pf.shape[0] / pm.shape[0]) * (pf.shape[1] / pm.shape[1])))
    if abs(rapporto - 1.0) > 0.02:
        interp = cv2.INTER_AREA if rapporto < 1 else cv2.INTER_LINEAR
        pm = cv2.resize(pm, None, fx=rapporto, fy=rapporto, interpolation=interp)
        fm = fm / rapporto
    return pf, ff, pm, fm, inf.height, inf.width


def riassunto(r):
    chiavi = ('n_campioni', 'n_affidabili', 'residuo_mediano_px', 'residuo_mediano_ic90', 'residuo_p95_px',
              'residuo_max_px', 'quota_sopra_1px', 'quota_sopra_2px', 'quota_sopra_5px', 'quota_sopra_10px')
    out = {k: r.get(k) for k in chiavi}
    out['campioni'] = [[int(c[0]), int(c[1]), float(np.hypot(c[2], c[3])), float(c[4])]
                       for c in (r.get('campioni') or []) + (r.get('campioni_deboli') or [])]
    return out


def scrivi(rec):
    with open(os.path.join(USCITA, "risultati.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


fixed, bande = DATI[insieme]
riduzioni = (2, 4) if fattore == 1 else (2,)
for nome, moving in bande.items():
    if solo and nome not in solo:
        continue
    dico(f"\n===== {insieme} {nome} fattore {fattore} =====")
    t0 = time.time()
    pf, ff, pm, fm, H, W = prepara(fixed, moving, fattore)
    t_ant = time.time() - t0
    stesse_dim = pf.shape == pm.shape
    # le due traslazioni, per sapere se differiscono (costa meno di un secondo)
    gf, gm = bm.gradient_magnitude(pf), bm.gradient_magnitude(pm)
    off_oggi = _originale(gf, gm, verbose=False); off_corr = traslazione_corretta(gf, gm)
    del gf, gm
    dico(f"anteprime {pf.shape[1]}x{pf.shape[0]} e {pm.shape[1]}x{pm.shape[0]} in {t_ant:.0f} s; "
         f"traslazione oggi {off_oggi}, corretta {off_corr} (px anteprima)")
    t0 = time.time()
    Tg, _ = ap._posa_grossolana(fixed, moving, H, W, None)
    t_posa = time.time() - t0
    pm2 = Tg.at_factor(ff, fm).apply_to_image(pm, pf.shape[:2])
    stime = []
    for rid in riduzioni:
        varianti = [("A1", "traslazione di oggi")]
        if tuple(off_oggi) != tuple(off_corr):
            varianti.append(("B1", "traslazione corretta"))
        varianti.append(("C", "posa grossolana subito"))
        for var, descr in varianti:
            rec = {'insieme': insieme, 'banda': nome, 'fattore': fattore, 'riduzione': rid, 'variante': var,
                   'descrizione': descr, 't_anteprime_s': t_ant, 't_posa_s': t_posa, 'stesse_dimensioni': bool(stesse_dim),
                   'traslazione_oggi': list(map(int, off_oggi)), 'traslazione_corretta': list(map(int, off_corr))}
            if insieme == "CK" and fattore == 1 and var == "A1":
                rec['saltata'] = "fallisce di sicuro (0% del dipinto in portata): vale la run delle 21:33"
                scrivi(rec); dico(f"  rid {rid} {var}: saltata"); continue
            if insieme == "CK" and fattore == 1 and var == "C" and rid == 2:
                T = Transform.load(os.path.join(RUN, "parametri", f"{nome} (2026-10-05 21-33).json"))
                rec['t_stima_s'] = None; rec['nota'] = "trasformazione salvata dalla run delle 21:33"
                rec['stima'] = {k: v for k, v in T.meta['estimazione'].items() if k != 'smoothing_crossval'}
                stime.append((rec, T)); dico(f"  rid {rid} {var}: dalla run"); continue
            t0 = time.time()
            try:
                if var == "C":
                    T, est = bm.estimate_transform(pf, pm2, ff, factor_moving=ff, coarse_init=False,
                                                   campo_grossolano_tra_le_passate='auto',
                                                   riduzione_grossolano=rid, verbose=False)
                    T = Tg.compose_residual(T.M, delta_field=T.field, delta_grid_step=T.grid_step)
                else:
                    bm.coarse_translation = traslazione_corretta if var == "B1" else _originale
                    T, est = bm.estimate_transform(pf, pm, ff, factor_moving=fm, coarse_init=True,
                                                   campo_grossolano_tra_le_passate='auto',
                                                   riduzione_grossolano=rid, verbose=False)
                    bm.coarse_translation = _originale
            except Exception as exc:
                bm.coarse_translation = _originale
                rec['t_stima_s'] = time.time() - t0; rec['errore'] = str(exc).split("\n")[0]
                scrivi(rec); dico(f"  rid {rid} {var}: ERRORE {rec['errore']}"); continue
            T.ref_shape = (H, W)
            rec['t_stima_s'] = time.time() - t0
            rec['stima'] = {k: v for k, v in est.items() if k != 'smoothing_crossval'}
            np.save(os.path.join(USCITA, f"{insieme}_{nome}_f{fattore}_r{rid}_{var}_M.npy"), T.M)
            if T.field is not None:
                np.save(os.path.join(USCITA, f"{insieme}_{nome}_f{fattore}_r{rid}_{var}_campo.npy"), T.field)
            stime.append((rec, T))
            dico(f"  rid {rid} {var}: stima in {rec['t_stima_s']:.0f} s, pass1 {est['pass1']['inlier']}/{est['pass1']['blocchi']}")
    del pf, pm, pm2
    arr_f, _ = image_io.apri(fixed, verbose=False)
    arr_m, _ = image_io.apri(moving, verbose=False)
    for rec, T in stime:
        t0 = time.time()
        r = val.residual_check(arr_f, arr_m, T, verbose=False)
        rec['t_verifica_s'] = time.time() - t0
        rec['verifica'] = riassunto(r)
        rf = val.residual_check(arr_f, arr_m, T, verbose=False, adattiva=False)
        rec['verifica_griglia_fissa'] = riassunto(rf)
        b = val.verifica_bordo(arr_f, arr_m, T, verbose=False)
        rec['bordo'] = [{k: f[k] for k in ('da_px', 'a_px', 'n_affidabili', 'n_misurate', 'mediana_px', 'p90_px', 'quota_sopra_5px')}
                        for f in b['fasce']]
        scrivi(rec)
        v = rec['verifica']
        med = v.get('residuo_mediano_px')
        dico(f"  rid {rec['riduzione']} {rec['variante']:<2}: "
             + ("nessuna finestra misurabile" if med is None else
                f"mediana {med:.3f} px, finestre {v['n_affidabili']}/{v['n_campioni']}, >1px {v['quota_sopra_1px']:.0%}, "
                f">5px {v['quota_sopra_5px']:.0%}, max {v['residuo_max_px']:.1f}")
             + f"  (verifica {rec['t_verifica_s']:.0f} s)")
    del arr_f, arr_m
dico("fatto", insieme, fattore)
