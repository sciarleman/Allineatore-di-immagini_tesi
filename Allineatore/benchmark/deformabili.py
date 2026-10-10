"""
Il programma contro i metodi DEFORMABILI di uso comune.

    python deformabili.py --sintetico      # immagini artificiali, trasformazione nota
    python deformabili.py --reali          # le riprese vere, con i criteri di esegui.py
    python deformabili.py --tabelle

Il confronto di esegui.py mette il programma contro sei metodi che stimano
tutti una trasformazione affine: non dice niente su chi, come noi, stima anche
una deformazione. Qui i concorrenti sono i quattro modi consueti di farlo, tutti
con librerie gia' installate (SimpleITK e OpenCV), senza installare elastix:

- B-spline + informazione mutua: una griglia di nodi che si infittisce
  (8 -> 16 -> 32 per lato), mossa per massimizzare l'informazione mutua di
  Mattes. E' l'impostazione di elastix e la piu' usata in ambito medico per
  immagini di modalita' diverse. A differenza dei file di parametri di elastix
  qui non c'e' la penalita' sull'energia di curvatura, che SimpleITK non ha: la
  regolarita' viene solo dalla spaziatura dei nodi;
- campo denso + correlazione locale: un vettore per pixel, lisciato a ogni
  passo, che massimizza la correlazione in un intorno (la metrica di ANTs). E'
  l'impostazione di ANTs/SyN, senza la simmetria;
- Demons: un vettore per pixel mosso dalla differenza di intensita'. Nasce per
  immagini della stessa modalita', quindi gli si danno i moduli del gradiente,
  come facciamo noi;
- flusso ottico denso (DIS, OpenCV): il modo consueto in visione artificiale,
  anch'esso sui moduli del gradiente.

Tutti partono dalla posa trovata dal programma (la sua affine, senza campi):
cosi' si confronta la sola parte deformabile. La prima versione li faceva
partire da un'affine a informazione mutua, che su due casi artificiali (stessa
banda, rotazione di 3 gradi) sbagliava di oltre 100 px e li trascinava con se':
quella affine resta in tabella come metodo a parte. Lavorano
sull'anteprima ridotta di un fattore 4 (un vettore per pixel o una B-spline a
piena risoluzione su decine di megapixel non sono praticabili), e il risultato
si riporta a piena risoluzione nel nostro Transform, cosi' la valutazione e'
la stessa per tutti.

Limite da dichiarare: ogni metodo e' usato con una configurazione ragionevole,
non tarata coppia per coppia; nemmeno il programma lo e' (valori predefiniti).
"""

import argparse
import contextlib
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
if PROG not in sys.path:
    sys.path.insert(0, PROG)
sys.path.insert(0, QUI)

import image_io
import block_matching as bm
import align_paintings
from transform_model import Transform
import metodi

RIS_SINT = os.path.join(QUI, 'deformabili_sintetico.jsonl')
RIS_REALI = os.path.join(QUI, 'deformabili_reali.jsonl')
TABELLE = os.path.join(QUI, 'deformabili_tabelle.txt')
FATTORE = 4


# ---------------------------------------------------------------------------
# preparazione comune
# ---------------------------------------------------------------------------

_POSE = {}


def _posa_del_programma(path_rif, path_tgt):
    """L'affine del programma (senza campi), a piena risoluzione: la posa di
    partenza che si da' ai deformabili. Una per coppia, riusata fra i metodi."""
    chiave = (path_rif, path_tgt)
    if chiave not in _POSE:
        tmp = tempfile.mkdtemp(prefix='deformabili_p_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                T, _ = align_paintings.align(path_rif, path_tgt, None, params_path=os.path.join(tmp, 'p.json'),
                                             factor=2, nonrigid=False, diagnostics_dir=os.path.join(tmp, 'diag'),
                                             verbose=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        _POSE.clear()
        _POSE[chiave] = np.array(T.M, np.float64)
    return _POSE[chiave]


def _prepara(path_rif, path_tgt, fattore, partenza='programma'):
    """Le due anteprime alla stessa scala e con la stessa forma (la seconda
    ritagliata o riempita, senza cambiarne la geometria), piu' l'affine di
    partenza comune a tutti i deformabili."""
    import SimpleITK as sitk
    a, b, fa, fb = metodi._coppia(path_rif, path_tgt, fattore, con_clahe=True)
    if b.shape != a.shape:
        c = np.zeros_like(a)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        c[:h, :w] = b[:h, :w]
        b = c
    ox, oy = bm.coarse_translation(bm.gradient_magnitude(a), bm.gradient_magnitude(b), verbose=False)
    fix = sitk.Cast(sitk.GetImageFromArray(a), sitk.sitkFloat32)
    mov = sitk.Cast(sitk.GetImageFromArray(b), sitk.sitkFloat32)
    if partenza == 'programma':
        # dalla piena risoluzione all'anteprima: x = f x_ant + (f-1)/2 sui due lati
        M = _posa_del_programma(path_rif, path_tgt)
        A = M[:, :2] * fa / fb
        t = (M[:, :2] @ np.full(2, (fa - 1) / 2.0) + M[:, 2] - (fb - 1) / 2.0) / fb
        tx_aff = sitk.AffineTransform(2)
        tx_aff.SetMatrix(A.ravel().tolist())
        tx_aff.SetTranslation(t.tolist())
        return a, b, fa, fb, fix, mov, tx_aff
    aff0 = sitk.AffineTransform(2)
    aff0.SetCenter([a.shape[1] / 2.0, a.shape[0] / 2.0])
    aff0.SetTranslation([float(ox), float(oy)])
    r = _registrazione(sitk)
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.25, 1)
    r.SetOptimizerAsRegularStepGradientDescent(1.5, 1e-5, 300)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(aff0, inPlace=False)
    tx_aff = r.Execute(fix, mov)
    return a, b, fa, fb, fix, mov, tx_aff


def _registrazione(sitk):
    r = sitk.ImageRegistrationMethod()
    r.SetInterpolator(sitk.sitkLinear)
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return r


def _spostamento(sitk, tx, fix):
    """Lo spostamento (h, w, 2) che la trasformazione ITK da' a ogni pixel
    dell'anteprima del riferimento: mappa(x) = x + spostamento(x)."""
    f = sitk.TransformToDisplacementFieldFilter()
    f.SetReferenceImage(fix)
    f.SetOutputPixelType(sitk.sitkVectorFloat64)
    return sitk.GetArrayFromImage(f.Execute(tx)).astype(np.float32)


def _matrice(tx_aff):
    """L'affine ITK come matrice 2x3 riferimento -> immagine, in px di anteprima."""
    o, ex, ey = (np.array(tx_aff.TransformPoint(p)) for p in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    return np.column_stack([ex - o, ey - o, o])


def _a_transform(disp, fa, fb, forma_rif, passo=64):
    """Dallo spostamento sull'anteprima al nostro Transform, a piena
    risoluzione. Il campo si campiona nei punti dell'anteprima che
    corrispondono ESATTAMENTE ai nodi della griglia piena (x = f x_ant +
    (f-1)/2): con M identita' e il campo uguale allo spostamento totale non
    serve separare la parte lineare."""
    Hf, Wf = forma_rif
    gy, gx = np.meshgrid(np.arange(0, Hf + passo, passo, dtype=np.float64),
                         np.arange(0, Wf + passo, passo, dtype=np.float64), indexing='ij')
    xa, ya = (gx - (fa - 1) / 2.0) / fa, (gy - (fa - 1) / 2.0) / fa
    d = cv2.remap(disp, xa.astype(np.float32), ya.astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    # fuori dall'anteprima lo spostamento si prolunga con l'ultimo valore
    mx, my = xa + d[..., 0], ya + d[..., 1]
    campo = np.stack([fb * mx + (fb - 1) / 2.0 - gx, fb * my + (fb - 1) / 2.0 - gy], axis=2).astype(np.float32)
    return Transform(np.array([[1.0, 0, 0], [0, 1.0, 0]]), campo, float(passo), forma_rif, {})


def _forma(path):
    m = image_io.info(path)
    return (m.height, m.width)


def _gradienti_8bit(a, b_riportata):
    out = []
    for img in (a, b_riportata):
        g = bm.gradient_magnitude(img)
        hi = float(np.percentile(g, 99.5)) or 1.0
        out.append(np.clip(g / hi * 255.0, 0, 255).astype(np.uint8))
    return out


def _riporta(b, M):
    """L'immagine da allineare portata sul riferimento con la sola affine."""
    return cv2.warpAffine(b, M.astype(np.float64), (b.shape[1], b.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def _componi(flusso, M):
    """mappa(x) = A (x + flusso(x)): lo spostamento totale sull'anteprima."""
    h, w = flusso.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    px, py = X + flusso[..., 0], Y + flusso[..., 1]
    return np.stack([M[0, 0] * px + M[0, 1] * py + M[0, 2] - X,
                     M[1, 0] * px + M[1, 1] * py + M[1, 2] - Y], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# i metodi
# ---------------------------------------------------------------------------

def solo_affine_mi(path_rif, path_tgt, fattore=FATTORE):
    """L'affine di partenza comune: dice quanto aggiunge la parte deformabile."""
    import SimpleITK as sitk
    t0 = time.time()
    a, b, fa, fb, fix, mov, tx_aff = _prepara(path_rif, path_tgt, fattore, partenza='mi')
    return _a_transform(_spostamento(sitk, tx_aff, fix), fa, fb, _forma(path_rif)), {'secondi': time.time() - t0}


def bspline_mi(path_rif, path_tgt, fattore=FATTORE, nodi=8, iterazioni=40):
    import SimpleITK as sitk
    t0 = time.time()
    a, b, fa, fb, fix, mov, tx_aff = _prepara(path_rif, path_tgt, fattore)
    bs = sitk.BSplineTransformInitializer(fix, [nodi, nodi], order=3)
    r = _registrazione(sitk)
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.10, 1)
    r.SetMovingInitialTransform(tx_aff)
    # Discesa del gradiente con ricerca in linea, come negli esempi di SimpleITK
    # per la B-spline a piu' risoluzioni: con L-BFGS-B il numero di parametri
    # che cambia da un livello all'altro manda in errore l'ottimizzatore.
    r.SetOptimizerAsGradientDescentLineSearch(5.0, iterazioni, convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5, lineSearchMaximumIterations=6)
    r.SetOptimizerScalesFromPhysicalShift()
    # la griglia si infittisce a ogni livello: 8 -> 16 -> 32 nodi per lato
    r.SetInitialTransformAsBSpline(bs, inPlace=True, scaleFactors=[1, 2, 4])
    r.Execute(fix, mov)
    tot = sitk.CompositeTransform([tx_aff, bs])
    return (_a_transform(_spostamento(sitk, tot, fix), fa, fb, _forma(path_rif)),
            {'secondi': time.time() - t0, 'nodi_finali': nodi * 4, 'metrica': float(r.GetMetricValue())})


def campo_denso_cc(path_rif, path_tgt, fattore=FATTORE, iterazioni=80):
    import SimpleITK as sitk
    t0 = time.time()
    a, b, fa, fb, fix, mov, tx_aff = _prepara(path_rif, path_tgt, fattore)
    zero = sitk.TransformToDisplacementField(sitk.Transform(2, sitk.sitkIdentity), sitk.sitkVectorFloat64,
                                             fix.GetSize(), fix.GetOrigin(), fix.GetSpacing(), fix.GetDirection())
    df = sitk.DisplacementFieldTransform(zero)
    df.SetSmoothingGaussianOnUpdate(varianceForUpdateField=3.0, varianceForTotalField=1.0)
    r = _registrazione(sitk)
    r.SetMetricAsANTSNeighborhoodCorrelation(4)
    r.SetMovingInitialTransform(tx_aff)
    r.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iterazioni,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10,
                                    estimateLearningRate=r.EachIteration)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(df, inPlace=True)
    r.Execute(fix, mov)
    tot = sitk.CompositeTransform([tx_aff, df])
    return (_a_transform(_spostamento(sitk, tot, fix), fa, fb, _forma(path_rif)),
            {'secondi': time.time() - t0, 'metrica': float(r.GetMetricValue())})


def demons_gradienti(path_rif, path_tgt, fattore=FATTORE, iterazioni=60):
    import SimpleITK as sitk
    t0 = time.time()
    a, b, fa, fb, fix, mov, tx_aff = _prepara(path_rif, path_tgt, fattore)
    M = _matrice(tx_aff)
    gf, gm = _gradienti_8bit(a, _riporta(b, M))
    flusso = None
    for rid in (4, 2, 1):
        h, w = a.shape[0] // rid, a.shape[1] // rid
        f_ = sitk.Cast(sitk.GetImageFromArray(cv2.resize(gf, (w, h), interpolation=cv2.INTER_AREA)), sitk.sitkFloat32)
        m_ = sitk.Cast(sitk.GetImageFromArray(cv2.resize(gm, (w, h), interpolation=cv2.INTER_AREA)), sitk.sitkFloat32)
        m_ = sitk.HistogramMatching(m_, f_, 128, 16)
        dem = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        dem.SetNumberOfIterations(iterazioni)
        dem.SetStandardDeviations(2.0)
        if flusso is None:
            campo = dem.Execute(f_, m_)
        else:
            su = cv2.resize(flusso, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
            iniz = sitk.GetImageFromArray(su.astype(np.float64), isVector=True)
            campo = dem.Execute(f_, m_, iniz)
        flusso = sitk.GetArrayFromImage(campo).astype(np.float32)
    return _a_transform(_componi(flusso, M), fa, fb, _forma(path_rif)), {'secondi': time.time() - t0}


def flusso_ottico_dis(path_rif, path_tgt, fattore=FATTORE):
    t0 = time.time()
    a, b, fa, fb, fix, mov, tx_aff = _prepara(path_rif, path_tgt, fattore)
    M = _matrice(tx_aff)
    gf, gm = _gradienti_8bit(a, _riporta(b, M))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    flusso = dis.calc(gf, gm, None)
    return _a_transform(_componi(flusso, M), fa, fb, _forma(path_rif)), {'secondi': time.time() - t0}


def programma(path_rif, path_tgt, fattore=2):
    """Il programma come lo chiama il lotto, alla precisione consigliata."""
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix='deformabili_')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T, rep = align_paintings.align(path_rif, path_tgt, None, params_path=os.path.join(tmp, 'p.json'),
                                           factor=fattore, nonrigid=True, min_conf=0.10, refine_block=128,
                                           refine_step=64, diagnostics_dir=os.path.join(tmp, 'diag'),
                                           tre_scale='auto', verbose=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return T, {'secondi': time.time() - t0}


METODI = {
    'programma (consigliata)': programma,
    'affine + informazione mutua': solo_affine_mi,
    'B-spline + informazione mutua': bspline_mi,
    'campo denso + correlazione locale': campo_denso_cc,
    'Demons sui gradienti': demons_gradienti,
    'flusso ottico DIS sui gradienti': flusso_ottico_dis,
    # Gli stessi due alla risoluzione a cui lavora il programma (fattore 2), per
    # togliere l'obiezione che a un quarto partano svantaggiati.
    'Demons a mezza risoluzione': lambda a, b: demons_gradienti(a, b, fattore=2),
    'flusso ottico DIS a mezza risoluzione': lambda a, b: flusso_ottico_dis(a, b, fattore=2),
}


# ---------------------------------------------------------------------------
# le due prove
# ---------------------------------------------------------------------------

def _fatti(percorso, chiavi):
    # una corsa finita in errore non conta come fatta: si ritenta
    fatti = set()
    if os.path.exists(percorso):
        for r in open(percorso, encoding='utf-8').read().splitlines():
            d = json.loads(r)
            if not d.get('errore'):
                fatti.add(tuple(d[k] for k in chiavi))
    return fatti


def _scrivi(percorso, rec):
    with open(percorso, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


CASI_SINTETICI = ("forma onda   6 px, consigliata", "forma onda  30 px, consigliata",
                  "forma onda 100 px, consigliata", "forma radiale  30 px, consigliata",
                  "forma radiale 100 px, consigliata", "forma radiale 200 px, consigliata",
                  "forma prospettiva 100 px, consigliata", "forma locale 100 px, consigliata",
                  "contenuto uguale", "contenuto rumore", "contenuto poca_struttura",
                  "posa rotazione 3 gradi", "posa immagine di dimensioni 0.7x")


def prova_sintetica(solo=None, semi=2):
    import sintetico
    casi = {n: c for n, c, _ in sintetico.casistica()}
    fatti = _fatti(RIS_SINT, ('caso', 'seme', 'metodo'))
    for nome in CASI_SINTETICI:
        caso = casi[nome]
        for seme in range(semi):
            da_fare = [m for m in METODI if (not solo or any(s in m for s in solo)) and (nome, seme, m) not in fatti]
            if not da_fare:
                continue
            cartella = tempfile.mkdtemp(prefix='deformabili_s_')
            try:
                p_f, p_m, zona = sintetico.costruisci(caso, seme, cartella)
                for met in da_fare:
                    rec = {'caso': nome, 'seme': seme, 'metodo': met}
                    try:
                        T, info = METODI[met](p_f, p_m)
                        rec.update(info)
                        rec['vero'] = sintetico.errore_vero(T, caso, zona)
                        testo = (f"errore vero {rec['vero']['mediana']:.2f} px (p95 {rec['vero']['p95']:.1f}, "
                                 f"max {rec['vero']['max']:.0f}), {info['secondi']:.0f} s")
                    except Exception as exc:
                        rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                        testo = "ERRORE " + rec['errore']
                    _scrivi(RIS_SINT, rec)
                    print(f"[{nome} | seme {seme}] {met}: {testo}", flush=True)
            finally:
                import gc
                T = None
                gc.collect()
                shutil.rmtree(cartella, ignore_errors=True)


def prova_reale(solo=None, cartella=None, bersagli=None):
    import esegui
    import validation as val
    if not os.path.exists(esegui.RIFERIMENTO) and cartella:
        esegui.IMMAGINI = cartella
        esegui.RIFERIMENTO = os.path.join(cartella, os.path.basename(esegui.RIFERIMENTO))
    arr_f, meta_f = image_io.apri(esegui.RIFERIMENTO, verbose=False)
    pos = esegui._posizioni(meta_f.height, meta_f.width)
    fatti = set()
    if os.path.exists(RIS_REALI):
        for r in open(RIS_REALI, encoding='utf-8').read().splitlines():
            d = json.loads(r)
            if not d.get('errore') and d.get('fissa'):
                fatti.add((d['immagine'], d['metodo']))
    for nome_img in esegui.BERSAGLI:
        if bersagli and not any(b in nome_img for b in bersagli):
            continue
        path_t = os.path.join(esegui.IMMAGINI, nome_img)
        if not os.path.exists(path_t):
            print(f"manca {nome_img}"); continue
        et = esegui.etichetta_di(nome_img)
        arr_m, _ = image_io.apri(path_t, verbose=False)
        for met in METODI:
            if (solo and not any(s in met for s in solo)) or (et, met) in fatti:
                continue
            rec = {'immagine': et, 'metodo': met}
            try:
                T, info = METODI[met](esegui.RIFERIMENTO, path_t)
                rec.update(info)
                rec.update(esegui.valuta(arr_f, arr_m, T, pos))
                # La verifica di esegui.py infittisce la griglia finche' trova
                # abbastanza finestre misurabili, quindi metodi diversi finiscono
                # su finestre diverse. Qui la griglia e' la piu' fitta possibile,
                # la stessa per tutti: si confronta sulle finestre comuni.
                v = val.residual_check(arr_f, arr_m, T, target_affidabili=10 ** 9, verbose=False)
                c = np.asarray(v.get('campioni') or [], np.float64).reshape(-1, 5)
                rec['fissa'] = {'mediana': v.get('residuo_mediano_px'), 'n_affidabili': v.get('n_affidabili'),
                                'n_campioni': v.get('n_campioni'), 'oltre_5px': v.get('quota_sopra_5px'),
                                'finestre': [[int(x), int(y), round(float(np.hypot(dx, dy)), 3)] for x, y, dx, dy, _ in c]}
                cart = os.path.join(QUI, 'lavoro', 'deformabili', et)
                os.makedirs(cart, exist_ok=True)
                T.save(os.path.join(cart, met.replace(' ', '_').replace('+', 'e') + '.json'))
                r = rec.get('residuo_px')
                testo = (f"residuo {r:.2f} px" if r is not None else "non misurabile") + \
                    f", {rec.get('campioni_affidabili')}/{rec.get('campioni_totali')} finestre, >5px " \
                    f"{100 * (rec.get('quota_sopra_5px') or 0):.0f}%, {info['secondi']:.0f} s"
            except Exception as exc:
                rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                testo = "ERRORE " + rec['errore']
            rec['memoria_picco_gb'] = esegui.memoria_gb()
            _scrivi(RIS_REALI, rec)
            print(f"[{et}] {met}: {testo}", flush=True)
        del arr_m


def tabelle():
    out = []
    if os.path.exists(RIS_SINT):
        righe = [json.loads(r) for r in open(RIS_SINT, encoding='utf-8').read().splitlines()]
        out.append("IMMAGINI ARTIFICIALI: errore vero mediano in px (media sui semi); fra parentesi il 95-esimo percentile")
        out.append(f"{'caso':<42}" + "".join(f"{m[:26]:>28}" for m in METODI))
        for nome in CASI_SINTETICI:
            riga = f"{nome:<42}"
            for m in METODI:
                rr = [r for r in righe if r['caso'] == nome and r['metodo'] == m]
                ok = [r for r in rr if not r.get('errore')]
                riga += (f"{np.mean([r['vero']['mediana'] for r in ok]):14.2f} ({np.mean([r['vero']['p95'] for r in ok]):7.1f})    "
                         if ok else f"{'errore' if rr else '-':>28}")
            out.append(riga)
        out.append(f"{'tempo medio, s':<42}" + "".join(
            f"{np.mean([r['secondi'] for r in righe if r['metodo'] == m and 'secondi' in r] or [np.nan]):28.0f}" for m in METODI))
    if os.path.exists(RIS_REALI):
        righe = [json.loads(r) for r in open(RIS_REALI, encoding='utf-8').read().splitlines()]
        out.append("")
        out.append("RIPRESE VERE: residuo mediano in px sulle 64 finestre di esegui.py; fra parentesi finestre misurabili e quota oltre 5 px")
        imm = list(dict.fromkeys(r['immagine'] for r in righe))
        out.append(f"{'immagine':<16}" + "".join(f"{m[:26]:>28}" for m in METODI))
        for et in imm:
            riga = f"{et:<16}"
            for m in METODI:
                rr = [r for r in righe if r['immagine'] == et and r['metodo'] == m]
                if not rr:
                    riga += f"{'-':>28}"
                elif rr[0].get('errore') or rr[0].get('residuo_px') is None:
                    riga += f"{'errore' if rr[0].get('errore') else 'non misurabile':>28}"
                else:
                    r = rr[0]
                    riga += f"{r['residuo_px']:12.2f} ({r['campioni_affidabili']:2d}/{r['campioni_totali']}, {100 * (r['quota_sopra_5px'] or 0):3.0f}%)   "
            out.append(riga)
        for chiave, et in (('ncc_gradienti', 'correlazione dei gradienti (piu\' alta = meglio)'), ('mind', 'MIND (piu\' basso = meglio)'),
                           ('secondi', 'tempo, s'), ('memoria_picco_gb', 'memoria di picco del processo, GB')):
            out.append(f"  {et}")
            for e in imm:
                riga = f"{e:<16}"
                for m in METODI:
                    rr = [r for r in righe if r['immagine'] == e and r['metodo'] == m and r.get(chiave) is not None]
                    riga += f"{rr[0][chiave]:28.3f}" if rr else f"{'-':>28}"
                out.append(riga)
    testo = "\n".join(out)
    open(TABELLE, 'w', encoding='utf-8').write(testo + "\n")
    print(testo)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sintetico', action='store_true')
    ap.add_argument('--reali', action='store_true')
    ap.add_argument('--tabelle', action='store_true')
    ap.add_argument('--solo', nargs='+', default=None)
    ap.add_argument('--semi', type=int, default=2)
    ap.add_argument('--bersagli', nargs='+', default=None, help="solo le riprese il cui nome contiene una di queste parole")
    ap.add_argument('--immagini', default=None, help="cartella delle riprese vere, se non e' immagini/")
    args = ap.parse_args()
    if args.sintetico:
        prova_sintetica(args.solo, args.semi)
    if args.reali:
        prova_reale(args.solo, args.immagini, args.bersagli)
    tabelle()
