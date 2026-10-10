"""Portata del campo grossolano slegata dalla precisione: misura sui dati veri.

Stessa strada della run a precisione massima (posa grossolana del ripiego, poi
stima a fattore 1), cambiando solo `riduzione_grossolano`. Il risultato di oggi
(riduzione 2) e' quello salvato dalla run delle 21:33 e si rimisura sulle stesse
finestre. Non scrive nulla fuori dallo scratchpad."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA\Allineatore")
import image_io, align_paintings as ap, validation as val, block_matching as bm
from transform_model import Transform

RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)
QUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portata")
os.makedirs(QUI, exist_ok=True)
VIS = os.path.join(RUN, "VIS.png")
PROVE = {"UV": (4, 8), "NIR": (4, 8)}
if len(sys.argv) > 1:
    PROVE = {k: v for k, v in PROVE.items() if k in sys.argv[1:]}


def dico(*a):
    print(*a, flush=True)


def campioni(arr_f, arr_m, T):
    r = val.residual_check(arr_f, arr_m, T, verbose=False, adattiva=False)
    c = {(int(v[0]), int(v[1])): (float(np.hypot(v[2], v[3])), float(v[4]))
         for v in (r.get('campioni') or []) + (r.get('campioni_deboli') or [])}
    return r, c


info_f = image_io.info(VIS)
H, W = info_f.height, info_f.width
for nome, riduzioni in PROVE.items():
    mov = os.path.join(RUN, nome + ".png")
    dico(f"\n===== {nome} =====")
    t0 = time.time()
    prev_f, ff = image_io.leggi_ridotta(VIS, lato_lungo_target=max(H, W), in_grigio=True, verbose=False)
    prev_m, fm = image_io.leggi_ridotta(mov, lato_lungo_target=max(H, W), in_grigio=True, verbose=False)
    Tg, _ = ap._posa_grossolana(VIS, mov, H, W, None)
    prev_m2 = Tg.at_factor(ff, fm).apply_to_image(prev_m, prev_f.shape[:2])
    del prev_m
    dico(f"anteprime e posa grossolana in {time.time() - t0:.0f} s (fattori {ff:g}, {fm:g})")
    stime = {}
    for rid in riduzioni:
        t0 = time.time()
        T2, est = bm.estimate_transform(prev_f, prev_m2, ff, factor_moving=ff, coarse_init=False,
                                        campo_grossolano_tra_le_passate='auto',
                                        riduzione_grossolano=rid, verbose=False)
        T2 = Tg.compose_residual(T2.M, delta_field=T2.field, delta_grid_step=T2.grid_step)
        T2.ref_shape = (H, W); T2.meta = {'estimazione': est}
        T2.save(os.path.join(QUI, f"{nome}_rid{rid}.json"))
        cg = est.get('campo_grossolano') or {}
        dico(f"riduzione {rid}: stima in {(time.time() - t0) / 60:.1f} min | grossolano: applicato {cg.get('applicato')}, "
             f"blocchi concordi {cg.get('blocchi_concordi')}/{cg.get('blocchi_affidabili')}, portata {cg.get('portata_px_full')}, "
             f"guadagno {cg.get('guadagno_pct')}, ampiezza max {cg.get('ampiezza_max_px_full')}, sigma {cg.get('smooth_px_full')} | "
             f"pass1 inlier {est['pass1']['inlier']}/{est['pass1']['blocchi']} | fine: blocchi {est['nonrigid'].get('blocchi')}, "
             f"cv {est['nonrigid_crossval'].get('residuo_prima_px')} -> {est['nonrigid_crossval'].get('residuo_dopo_px')}")
        stime[rid] = T2
    del prev_f, prev_m2
    arr_f, _ = image_io.apri(VIS, verbose=False)
    arr_m, _ = image_io.apri(mov, verbose=False)
    T0 = Transform.load(os.path.join(RUN, "parametri", f"{nome} (2026-10-05 21-33).json"))
    r0, c0 = campioni(arr_f, arr_m, T0)
    tutti = {2: (r0, c0)}
    for rid, T2 in stime.items():
        tutti[rid] = campioni(arr_f, arr_m, T2)
    for rid, (r, c) in tutti.items():
        et = "oggi (riduzione 2)" if rid == 2 else f"riduzione {rid}"
        dico(f"  {et:<20} mediana {r['residuo_mediano_px']:.3f} px IC {[round(v, 3) for v in r['residuo_mediano_ic90']]}"
             f"  finestre {r['n_affidabili']}/{r['n_campioni']}  >1px {r['quota_sopra_1px']:.0%} >5px {r['quota_sopra_5px']:.0%}"
             f" >10px {r['quota_sopra_10px']:.0%}  p95 {r['residuo_p95_px']:.2f}  max {r['residuo_max_px']:.2f}")
    for rid in stime:
        r, c = tutti[rid]
        com = sorted(k for k in c0 if k in c and c0[k][1] >= 0.05 and c[k][1] >= 0.05)
        a = np.array([c0[k][0] for k in com]); b = np.array([c[k][0] for k in com])
        dico(f"  riduzione {rid} contro oggi, sulle {len(com)} finestre misurabili in entrambe: mediana {np.median(a):.3f} -> {np.median(b):.3f} px;"
             f" peggiorate di oltre 0.2 px: {(b > a + 0.2).sum()}, migliorate: {(b < a - 0.2).sum()};"
             f" fra quelle gia' sotto 1 px ({(a < 1).sum()}): mediana {np.median(a[a < 1]):.3f} -> {np.median(b[a < 1]):.3f}, max {b[a < 1].max():.2f}")
        cattive = [k for k in c0 if c0[k][1] >= 0.05 and c0[k][0] > 5]
        dico("    le finestre oggi oltre 5 px: " + "  ".join(
            f"({k[0]},{k[1]}) {c0[k][0]:.0f}->{c[k][0]:.1f}{'' if c[k][1] >= 0.05 else '*'}" for k in cattive))
        b_ = val.verifica_bordo(arr_f, arr_m, stime[rid], verbose=False)
        dico("    bordo: " + " | ".join(f"{f['da_px']}-{f['a_px']}: n {f['n_affidabili']}/{f['n_misurate']} med "
                                         f"{'-' if f['mediana_px'] is None else round(f['mediana_px'], 2)} >5px "
                                         f"{'-' if f['quota_sopra_5px'] is None else round(f['quota_sopra_5px'], 2)}" for f in b_['fasce']))
    del arr_f, arr_m
dico("\nfatto")
