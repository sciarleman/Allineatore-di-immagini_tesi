"""
Prova di block_matching.SOGLIA_MINIMA_PASSO e PROVA_CODA
sul rigonfiamento locale che il test sintetico ha mostrato non corretto.

    python prova_locale.py sintetico            # casi artificiali, spento e acceso
    python prova_locale.py reali <cartella FG>  # riprese vere, spento e acceso

Tre domande: i casi col rigonfiamento migliorano? gli altri casi artificiali
restano com'erano? le riprese vere restano com'erano? Scrive
prova_locale.jsonl e stampa la tabella.
"""
import contextlib, io, json, os, shutil, sys, tempfile, time
import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)
sys.path.insert(0, PROG)
import block_matching as bm, align_paintings, image_io, validation as val
import sintetico

RIS = os.path.join(QUI, 'prova_locale.jsonl')
VARIANTI = (('come oggi', False, 0.0), ('soglia 3%', False, 0.03), ('soglia', False, 0.05), ('soglia 8%', False, 0.08),
            ('coda + soglia', True, 0.05))   # 'come oggi' = prima dell'8 ottobre, 'soglia' = 5%, quella adottata
CASI = ("forma locale  30 px, consigliata", "forma locale 100 px, consigliata", "forma locale 200 px, consigliata",
        "forma locale 100 px, massima", "forma nessuna (solo posa), consigliata", "forma onda   6 px, consigliata",
        "forma radiale 100 px, consigliata", "forma radiale 200 px, consigliata", "forma prospettiva 200 px, consigliata",
        "contenuto poca_struttura", "contenuto poca_struttura, radiale 100 px")
REALI = (("NIR850", "NIR850_FraGalgario_TOT.tif"), ("UVF", "UVF_FraGalgario_TOT.tif"), ("UVR", "UVR_FraGalgario_TOT.tif"),
         ("SWIR", "big_SWIR_FraGalgario_TOT.tif"), ("RX Cappello", "RX_FraGalgario_TOT_Cappellochiaro.tif"))


def allinea(p_f, p_m, fattore, coda, soglia, cartella):
    prima = (bm.PROVA_CODA, bm.SOGLIA_MINIMA_PASSO)
    bm.PROVA_CODA, bm.SOGLIA_MINIMA_PASSO = coda, soglia
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return align_paintings.align(p_f, p_m, None, params_path=os.path.join(cartella, 'p.json'), factor=fattore,
                                         nonrigid=True, min_conf=0.10, refine_block=128, refine_step=64,
                                         diagnostics_dir=os.path.join(cartella, 'diag'), tre_scale='auto', verbose=False)
    finally:
        bm.PROVA_CODA, bm.SOGLIA_MINIMA_PASSO = prima


def livelli(rep):
    s = rep['stima']
    return [(l.get('applicato'), l.get('criterio')) for l in (s.get('campo_grossolano_livelli') or [])]


def scrivi(rec):
    with open(RIS, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print({k: v for k, v in rec.items() if k != 'finestre'}, flush=True)


def prova_sintetica():
    casi = {n: c for n, c, _ in sintetico.casistica()}
    for nome in CASI:
        caso = casi[nome]
        tmp = tempfile.mkdtemp(prefix='sintetico_pl_')
        try:
            p_f, p_m, zona = sintetico.costruisci(caso, 0, tmp)
            for et, coda, soglia in VARIANTI:
                rec = {'insieme': 'sintetico', 'caso': nome, 'variante': et}
                try:
                    t0 = time.time()
                    T, rep = allinea(p_f, p_m, caso['fattore'], coda, soglia, tmp)
                    v = sintetico.errore_vero(T, caso, zona)
                    rec.update(mediana=round(v['mediana'], 3), p95=round(v['p95'], 2), max=round(v['max'], 1),
                               oltre_5px=round(v['oltre_5px'], 3), livelli=livelli(rep), s=round(time.time() - t0))
                except Exception as exc:
                    rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                T = rep = None
                scrivi(rec)
        finally:
            import gc; gc.collect(); shutil.rmtree(tmp, ignore_errors=True)


def prova_reale(cartella, coppie=REALI, rif="VIS_FraGalgario_TOT.tif", fattore=2):
    p_f = os.path.join(cartella, rif)
    arr_f = image_io.apri(p_f, verbose=False)[0]
    for et_img, nome in coppie:
        p_m = os.path.join(cartella, nome)
        arr_m = image_io.apri(p_m, verbose=False)[0]
        for et, coda, soglia in VARIANTI:
            tmp = tempfile.mkdtemp(prefix='reale_pl_')
            rec = {'insieme': 'reale', 'caso': et_img, 'variante': et}
            try:
                t0 = time.time()
                T, rep = allinea(p_f, p_m, fattore, coda, soglia, tmp)
                d = rep['verifica']['dopo']
                v = val.residual_check(arr_f, arr_m, T, target_affidabili=10 ** 9, verbose=False)
                c = np.asarray(v.get('campioni') or [], np.float64).reshape(-1, 5)
                rec.update(verifica=round(d['residuo_mediano_px'], 3), fissa=round(v['residuo_mediano_px'], 3),
                           n=f"{v['n_affidabili']}/{v['n_campioni']}", oltre_5px=round(v['quota_sopra_5px'], 3),
                           max=round(v['residuo_max_px'], 1), livelli=livelli(rep), s=round(time.time() - t0),
                           finestre=[[int(x), int(y), round(float(np.hypot(dx, dy)), 3)] for x, y, dx, dy, _ in c])
            except Exception as exc:
                rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            T = rep = None
            scrivi(rec)
            shutil.rmtree(tmp, ignore_errors=True)
        del arr_m


def tabella():
    R = [json.loads(r) for r in open(RIS, encoding='utf-8').read().splitlines()]
    for ins in ('sintetico', 'reale'):
        casi = list(dict.fromkeys(r['caso'] for r in R if r['insieme'] == ins))
        if not casi:
            continue
        print(f"\n== {ins}: " + ("errore vero mediano (p95; quota oltre 5 px)" if ins == 'sintetico'
                                 else "griglia fissa, mediana (misurabili; oltre 5 px; max) | sulle finestre comuni con «come oggi»: migliori/peggiori di oltre 0.5 px"))
        for c in casi:
            print(f"  {c}")
            base = next((r for r in R if r['insieme'] == ins and r['caso'] == c and r['variante'] == 'come oggi'), None)
            for r in [r for r in R if r['insieme'] == ins and r['caso'] == c]:
                if r.get('errore'):
                    print(f"     {r['variante']:<14} ERRORE {r['errore']}"); continue
                if ins == 'sintetico':
                    print(f"     {r['variante']:<14} {r['mediana']:7.3f} (p95 {r['p95']:7.2f}; >5px {100 * r['oltre_5px']:4.1f}%; max {r['max']:6.1f})  livelli {r['livelli']}  {r['s']} s")
                else:
                    conf = ''
                    if base and base.get('finestre'):
                        b = {(x, y): e for x, y, e in base['finestre']}; d = {(x, y): e for x, y, e in r['finestre']}
                        com = [k for k in d if k in b]
                        a1 = np.array([d[k] for k in com]); b1 = np.array([b[k] for k in com])
                        conf = f" | {len(com)} comuni: {int((b1 - a1 > 0.5).sum())} migliori, {int((a1 - b1 > 0.5).sum())} peggiori"
                    print(f"     {r['variante']:<14} {r['fissa']:7.3f} ({r['n']}; >5px {100 * r['oltre_5px']:4.1f}%; max {r['max']:6.1f})  livelli {r['livelli']}{conf}  {r['s']} s")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sintetico':
        prova_sintetica()
    elif len(sys.argv) > 2 and sys.argv[1] == 'reali':
        prova_reale(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == 'affresco':
        prova_reale(sys.argv[2], coppie=(("affresco UV", "UV.tif"), ("affresco NIR", "NIR.tif")), rif="VIS.tif")
    if os.path.exists(RIS):
        tabella()
