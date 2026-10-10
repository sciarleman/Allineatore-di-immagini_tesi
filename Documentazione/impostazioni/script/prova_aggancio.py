"""Riconoscere presto un primo aggancio fallito: misura del criterio.

Per ogni coppia: la sola prima passata dalla traslazione di oggi (affine T1),
la posa grossolana (affine Tg), e la verifica a piena risoluzione delle due
affini sulle stesse 64 finestre. Solo misura, nessuna modifica al programma.

    python prova_aggancio.py FG|CK fattore [bande...]
"""
import os, sys, time, json
import numpy as np
import cv2

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, validation as val, block_matching as bm

RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
DATI = {"FG": (os.path.join(FG, "VIS_FraGalgario_TOT.tif"), {
            "NIR850": "NIR850_FraGalgario_TOT.tif", "UVF": "UVF_FraGalgario_TOT.tif", "UVR": "UVR_FraGalgario_TOT.tif",
            "SWIR": "big_SWIR_FraGalgario_TOT.tif",
            "RX_Cappello": "RX_FraGalgario_TOT_Cappellochiaro.tif", "RX_Testa": "RX_FraGalgarioTOT_TestaChiara.tif"}, FG),
        "CK": (os.path.join(RUN, "VIS.png"), {"NIR": "NIR.png", "UV": "UV.png"}, RUN)}
ins, fattore = sys.argv[1], int(sys.argv[2])
fixed, bande, cart = DATI[ins]
QUI = os.path.dirname(os.path.abspath(__file__))


def possibili(shape, block, step):
    h, w = shape[:2]; half = block // 2
    return len(range(half, h - half + 1, step)) * len(range(half, w - half + 1, step))


def ver(arr_f, arr_m, T):
    r = val.residual_check(arr_f, arr_m, T, verbose=False, adattiva=False)
    return r.get('residuo_mediano_px'), r.get('n_affidabili', 0), r.get('n_campioni', 0), r.get('quota_sopra_10px')


info = image_io.info(fixed); H, W = info.height, info.width
arr_f, _ = image_io.apri(fixed, verbose=False)
for nome, f in bande.items():
    if sys.argv[3:] and nome not in sys.argv[3:]:
        continue
    mov = os.path.join(cart, f)
    target = max(H, W) // fattore
    pf, ff = image_io.leggi_ridotta(fixed, lato_lungo_target=target, in_grigio=True, verbose=False)
    pm, fm = image_io.leggi_ridotta(mov, lato_lungo_target=target, in_grigio=True, verbose=False)
    rapporto = float(np.sqrt((pf.shape[0] / pm.shape[0]) * (pf.shape[1] / pm.shape[1])))
    if abs(rapporto - 1.0) > 0.02:
        pm = cv2.resize(pm, None, fx=rapporto, fy=rapporto, interpolation=cv2.INTER_AREA if rapporto < 1 else cv2.INTER_LINEAR)
        fm = fm / rapporto
    t0 = time.time(); err = None
    try:
        T1, i1 = bm.estimate_transform(pf, pm, ff, factor_moving=fm, nonrigid=False, verbose=False)
        T1.ref_shape = (H, W)
    except RuntimeError as exc:
        T1, i1, err = None, None, str(exc).splitlines()[0]
    t_p1 = time.time() - t0
    t0 = time.time()
    Tg, ig = ap._posa_grossolana(fixed, mov, H, W, None)
    t_posa = time.time() - t0
    del pm
    arr_m, _ = image_io.apri(mov, verbose=False)
    t0 = time.time()
    v1 = ver(arr_f, arr_m, T1) if T1 is not None else (None, 0, 0, None)
    vg = ver(arr_f, arr_m, Tg)
    t_ver = (time.time() - t0) / (2 if T1 is not None else 1)
    blk = (i1 or {}).get('block', 256)
    q = None if i1 is None else i1['pass1']['inlier'] / possibili(pf.shape, blk, blk // 2)
    rec = {'insieme': ins, 'banda': nome, 'fattore': fattore, 'errore_prima_passata': err,
           'pass1': None if i1 is None else i1['pass1'], 'quota_inlier': q,
           'pre': None if i1 is None else i1['pre_allineamento_px'],
           'affine_traslazione': v1, 'affine_posa': vg, 't_prima_passata_s': t_p1, 't_posa_s': t_posa, 't_verifica_s': t_ver}
    with open(os.path.join(QUI, "aggancio.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    f1 = "errore" if T1 is None else f"{i1['pass1']['inlier']}/{i1['pass1']['blocchi']} inlier ({100 * q:.2f}% dei possibili)"
    m = lambda v: "non mis." if v[0] is None else f"{v[0]:6.1f} px su {v[1]}/{v[2]}, >10px {100 * v[3]:.0f}%"
    print(f"{ins} {nome:<11} f{fattore}: 1a passata {f1} | affine da traslazione: {m(v1)} | affine della posa: {m(vg)} "
          f"| tempi: 1a passata {t_p1:.0f} s, posa {t_posa:.0f} s, verifica {t_ver:.0f} s", flush=True)
    del arr_m, pf
