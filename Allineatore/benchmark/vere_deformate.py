"""
Riprese VERE deformate in modo noto: quanto bene il programma ritrova una
deformazione che gli abbiamo aggiunto noi.

    python vere_deformate.py <cartella delle riprese>     # esegue e stampa la tabella
    python vere_deformate.py --tabelle

Il test sintetico (sintetico.py) conosce l'errore esattamente, ma su immagini
artificiali. Le riprese vere hanno la struttura, il rumore e le differenze fra
bande che contano, ma li' la trasformazione vera non si conosce. Questa prova
sta in mezzo:

1. si allinea la ripresa al visibile com'e' -> T0;
2. si deforma la ripresa con una trasformazione D nota (in pixel della ripresa)
   e la si riallinea da capo -> T1;
3. se il programma fosse esatto, T1 = D o T0. L'errore di recupero e' la
   distanza fra T1(x) e D(T0(x)), su una griglia di punti del visibile.

Che cosa misura e che cosa no. Misura quanto il programma segue una
deformazione nota aggiunta a una ripresa vera, con il contenuto vero. Non misura
l'errore di T0: se T0 sbaglia di 3 px in un punto, quell'errore c'e' in tutte e
due le stime e qui si cancella. E dove T0 e' incerta (le zone di una
radiografia senza struttura in comune) anche l'errore di recupero lo e': per
questo si riporta anche ristretto ai punti vicini a una finestra di verifica
misurabile.

Le deformazioni sono quelle del test sintetico, con ampiezze espresse in pixel
del visibile: radiale e prospettica da 100 px agli angoli, onda da 30 px,
rigonfiamento locale da 30 e da 100 px.
"""

import contextlib
import gc
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import cv2

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)
sys.path.insert(0, PROG)

import image_io
import tiff_memmap
import align_paintings
import validation as val

RIS = os.path.join(QUI, 'vere_deformate.jsonl')
TABELLE = os.path.join(QUI, 'vere_deformate_tabelle.txt')
RIFERIMENTO = "VIS_FraGalgario_TOT.tif"
RIPRESE = (("infrarosso vicino 1", "NIR850_FraGalgario_TOT.tif"), ("infrarosso vicino 2", "NIR950_FraGalgario_TOT.tif"),
           ("fluorescenza ultravioletta", "UVF_FraGalgario_TOT.tif"), ("ultravioletto riflesso", "UVR_FraGalgario_TOT.tif"),
           ("infrarosso a onde corte", "big_SWIR_FraGalgario_TOT.tif"), ("radiografia", "RX_FraGalgario_TOT_Cappellochiaro.tif"))
DEFORMAZIONI = (("radiale 100 px", 'radiale', 100.0), ("prospettiva 100 px", 'prospettiva', 100.0),
                ("onda 30 px", 'onda', 30.0), ("rigonfiamento 30 px", 'locale', 30.0),
                ("rigonfiamento 100 px", 'locale', 100.0))
FASCE = (('0-150', 0, 150), ('150-500', 150, 500), ('oltre 500', 500, 1e9))


def deformazione(forma, A, Hm, Wm, scala):
    """D: (x, y) -> (x', y') in pixel della ripresa. `A` e' in pixel del
    visibile, `scala` = pixel della ripresa per pixel del visibile."""
    A = A * scala
    cx, cy = Wm / 2.0, Hm / 2.0
    r_ang = np.hypot(cx, cy)

    def D(x, y):
        x = np.asarray(x); y = np.asarray(y)
        u, v = x - cx, y - cy
        if forma == 'onda':
            P = max(900.0 * scala, 2 * np.pi * A / 0.04)
            return x + A * np.sin(2 * np.pi * y / P), y + A * np.cos(2 * np.pi * x / P)
        if forma == 'radiale':
            k = A * (np.hypot(u, v) / r_ang) ** 2 / r_ang
            return x + k * u, y + k * v
        if forma == 'prospettiva':
            den = 1.0 + (A / r_ang) * (0.8 * u + 0.6 * v) / r_ang
            return cx + u / den, cy + v / den
        if forma == 'locale':
            g = np.exp(-(((x - 0.62 * Wm) ** 2 + (y - 0.38 * Hm) ** 2) / (2 * (700.0 * scala) ** 2)))
            return x + A * g, y - 0.6 * A * g
        raise ValueError(forma)
    return D


def scrivi_deformata(path_in, path_out, D, righe=1024):
    """ripresa'(p) = ripresa(D^-1(p)): l'inversa per punto fisso, a bande di
    righe, leggendo dal file mappato solo le righe che servono."""
    arr, meta = image_io.apri(path_in, verbose=False)
    Hm, Wm = meta.height, meta.width
    scrittore = tiff_memmap.create_tiff_memmap(path_out, Hm, Wm, samples=meta.samples, dtype=meta.dtype,
                                               sequenziale=True)
    xs = np.arange(Wm, dtype=np.float32)
    for y0 in range(0, Hm, righe):
        y1 = min(Hm, y0 + righe)
        X, Y = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float32))
        ix, iy = X.copy(), Y.copy()
        for _ in range(40):
            fx, fy = D(ix, iy)
            fx = X - fx; fy = Y - fy
            ix += fx; iy += fy
            if max(float(np.abs(fx).max()), float(np.abs(fy).max())) < 2e-3:
                break
        a = int(np.clip(np.floor(iy.min()) - 2, 0, Hm - 1)); b = int(np.clip(np.ceil(iy.max()) + 3, a + 1, Hm))
        pezzo = np.asarray(arr[a:b])
        banda = cv2.remap(pezzo, ix.astype(np.float32), (iy - a).astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        scrittore.scrivi(banda)
        del pezzo
    scrittore.chiudi()
    del arr


def allinea(p_f, p_m, cartella):
    with contextlib.redirect_stdout(io.StringIO()):
        return align_paintings.align(p_f, p_m, None, params_path=os.path.join(cartella, 'p.json'), factor=2,
                                     nonrigid=True, min_conf=0.10, refine_block=128, refine_step=64,
                                     diagnostics_dir=os.path.join(cartella, 'diag'), tre_scale='auto', verbose=False)


def fatti():
    out = set()
    if os.path.exists(RIS):
        for r in open(RIS, encoding='utf-8').read().splitlines():
            d = json.loads(r)
            if not d.get('errore'):
                out.add((d['ripresa'], d['deformazione']))
    return out


def esegui(cartella):
    p_f = os.path.join(cartella, RIFERIMENTO)
    mf = image_io.info(p_f)
    Hf, Wf = mf.height, mf.width
    gx, gy = np.meshgrid(np.arange(100, Wf - 99, 200, dtype=np.float64), np.arange(100, Hf - 99, 200, dtype=np.float64))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    d_bordo = np.minimum.reduce([xy[:, 0], xy[:, 1], Wf - 1 - xy[:, 0], Hf - 1 - xy[:, 1]])
    gia = fatti()
    for et, nome in RIPRESE:
        p_m = os.path.join(cartella, nome)
        if not os.path.exists(p_m) or all((et, d[0]) in gia for d in DEFORMAZIONI):
            continue
        mm = image_io.info(p_m)
        Hm, Wm = mm.height, mm.width
        scala = float(np.sqrt((Hm * Wm) / float(Hf * Wf)))
        tmp = tempfile.mkdtemp(prefix='vere_def_')
        try:
            t0 = time.time()
            T0, rep0 = allinea(p_f, p_m, tmp)
            q0 = align_paintings._porta_punti(T0, xy)
            dentro = (q0[:, 0] >= 0) & (q0[:, 1] >= 0) & (q0[:, 0] <= Wm - 1) & (q0[:, 1] <= Hm - 1)
            # i punti a meno di 600 px da una finestra di verifica misurabile:
            # li' T0 poggia su una misura
            # (griglia fissa, la piu' fitta: il report di align non tiene le posizioni)
            arr_f = image_io.apri(p_f, verbose=False)[0]; arr_m = image_io.apri(p_m, verbose=False)[0]
            vf0 = val.residual_check(arr_f, arr_m, T0, target_affidabili=10 ** 9, verbose=False)
            del arr_f, arr_m
            c = np.asarray(vf0.get('campioni') or [], np.float64).reshape(-1, 5)
            if len(c):
                dm = np.min(np.hypot(xy[:, None, 0] - c[None, :, 0], xy[:, None, 1] - c[None, :, 1]), axis=1)
                misurato = dm <= 600
            else:
                misurato = np.zeros(len(xy), bool)
            v0 = rep0['verifica']['dopo']
            print(f"[{et}] allineamento di partenza: verifica {v0['residuo_mediano_px']:.3f} px, "
                  f"{v0['n_affidabili']}/{v0['n_campioni']} finestre, {time.time() - t0:.0f} s", flush=True)
            for nome_d, forma, A in DEFORMAZIONI:
                if (et, nome_d) in gia:
                    continue
                rec = {'ripresa': et, 'deformazione': nome_d, 'scala_ripresa': scala,
                       'verifica_partenza': v0['residuo_mediano_px']}
                p_d = os.path.join(tmp, 'deformata.tif')
                try:
                    D = deformazione(forma, A, Hm, Wm, scala)
                    t0 = time.time()
                    scrivi_deformata(p_m, p_d, D)
                    rec['t_costruzione_s'] = time.time() - t0
                    t0 = time.time()
                    T1, rep1 = allinea(p_f, p_d, tmp)
                    rec['t_align_s'] = time.time() - t0
                    atteso = np.column_stack(D(q0[:, 0], q0[:, 1]))
                    e = np.hypot(*(align_paintings._porta_punti(T1, xy) - atteso).T) / scala   # in px del visibile
                    ok = dentro & (atteso[:, 0] >= 0) & (atteso[:, 1] >= 0) & (atteso[:, 0] <= Wm - 1) & (atteso[:, 1] <= Hm - 1)

                    def riassunto(sel):
                        v = e[sel]
                        if len(v) == 0:
                            return None
                        return {'n': int(len(v)), 'mediana': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
                                'max': float(v.max()), 'oltre_1px': float((v > 1).mean()), 'oltre_5px': float((v > 5).mean())}
                    rec['tutti'] = riassunto(ok)
                    rec['dove_si_misura'] = riassunto(ok & misurato)
                    for nf, lo, hi in FASCE:
                        rec['fascia ' + nf] = riassunto(ok & (d_bordo >= lo) & (d_bordo < hi))
                    v1 = rep1['verifica']['dopo']
                    rec['verifica'] = {k: v1.get(k) for k in ('residuo_mediano_px', 'n_affidabili', 'n_campioni', 'quota_sopra_5px')}
                    rec['livelli'] = [bool(l.get('applicato')) for l in (rep1['stima'].get('campo_grossolano_livelli') or [])]
                    testo = (f"errore di recupero {rec['tutti']['mediana']:.3f} px (p95 {rec['tutti']['p95']:.2f}, max {rec['tutti']['max']:.1f}, "
                             f">5px {100 * rec['tutti']['oltre_5px']:.1f}%); dove si misura {rec['dove_si_misura']['mediana'] if rec['dove_si_misura'] else float('nan'):.3f}; "
                             f"verifica {v1['residuo_mediano_px']:.3f} ({v1['n_affidabili']}/{v1['n_campioni']}); "
                             f"{rec['t_costruzione_s']:.0f} + {rec['t_align_s']:.0f} s")
                except Exception as exc:
                    rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                    testo = "ERRORE " + rec['errore']
                T1 = rep1 = None
                gc.collect()
                with open(RIS, 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{et}] {nome_d}: {testo}", flush=True)
        finally:
            T0 = rep0 = None
            gc.collect()
            shutil.rmtree(tmp, ignore_errors=True)


def tabelle():
    R = [json.loads(r) for r in open(RIS, encoding='utf-8').read().splitlines()]
    R = [r for r in R if not r.get('errore')]
    out = ["RIPRESE VERE DEFORMATE IN MODO NOTO, precisione consigliata. Errore di recupero in px del visibile.",
           "mediana (p95; quota oltre 5 px) su tutti i punti | mediana dove la verifica di partenza misura | mediana per distanza dal bordo: 0-150 / 150-500 / oltre 500 | verifica del programma dopo (finestre)"]
    for et in dict.fromkeys(r['ripresa'] for r in R):
        out.append(f"\n{et}")
        for r in [r for r in R if r['ripresa'] == et]:
            t, m = r['tutti'], r.get('dove_si_misura')
            fa = " / ".join(f"{(r.get('fascia ' + n) or {}).get('mediana', float('nan')):.2f}" for n, _, _ in FASCE)
            v = r['verifica']
            out.append(f"  {r['deformazione']:<22} {t['mediana']:6.3f} (p95 {t['p95']:6.2f}; >5px {100 * t['oltre_5px']:5.1f}%) | "
                       f"{(m or {}).get('mediana', float('nan')):6.3f} (n {(m or {}).get('n', 0):4d}; >5px {100 * (m or {}).get('oltre_5px', float('nan')):5.1f}%) | {fa} | "
                       f"{v['residuo_mediano_px']:.3f} ({v['n_affidabili']}/{v['n_campioni']})  livelli {r['livelli']}")
    out.append("\nPER DEFORMAZIONE, media sulle riprese fotografiche (infrarosso vicino e ultravioletto) e, a parte, onde corte e radiografia:")
    for nome_d, _, _ in DEFORMAZIONI:
        foto = [r for r in R if r['deformazione'] == nome_d and ('vicino' in r['ripresa'] or 'ultraviolett' in r['ripresa'])]
        altri = [r for r in R if r['deformazione'] == nome_d and r not in foto]
        riga = f"  {nome_d:<22}"
        if foto:
            riga += (f" fotografiche (n {len(foto)}): mediana {np.mean([r['tutti']['mediana'] for r in foto]):.3f} "
                     f"[{min(r['tutti']['mediana'] for r in foto):.3f}-{max(r['tutti']['mediana'] for r in foto):.3f}], "
                     f">5px {100 * np.mean([r['tutti']['oltre_5px'] for r in foto]):.1f}%")
        for r in altri:
            riga += f" | {r['ripresa']}: {r['tutti']['mediana']:.2f}, >5px {100 * r['tutti']['oltre_5px']:.0f}%"
        out.append(riga)
    testo = "\n".join(out)
    open(TABELLE, 'w', encoding='utf-8').write(testo + "\n")
    print(testo)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != '--tabelle':
        esegui(sys.argv[1])
    if os.path.exists(RIS):
        tabelle()
