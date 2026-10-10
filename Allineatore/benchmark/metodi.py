"""
I metodi di allineamento messi a confronto.

Ognuno riceve la stessa coppia di percorsi e restituisce la stessa cosa: un
oggetto Transform nella convenzione del progetto (mappa riferimento -> immagine
da allineare, in pixel a piena risoluzione) piu' un dizionario di informazioni.
Cosi' la valutazione successiva e' identica per tutti e non favorisce nessuno.

Ogni metodo lavora alla risoluzione che gli e' propria:
- il nostro e MIRA usano quella prevista dai rispettivi autori;
- i metodi basati su un ottimizzatore generico (MIND, NGF, NMI + Powell) non
  reggono la piena risoluzione, quindi usano una piramide a due livelli, che e'
  il modo in cui vengono normalmente impiegati;
- SIFT ed ECC lavorano su un'anteprima ridotta, che e' la prassi.

La risoluzione di lavoro e' registrata nel dizionario di informazioni e riportata
nei risultati: fa parte delle caratteristiche del metodo, non e' un dettaglio da
nascondere.
"""

import os
import sys
import time

import numpy as np
import cv2

# Il motore sta nella cartella sopra questa (Allineatore/).
PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROG not in sys.path:
    sys.path.insert(0, PROG)

import image_io
import block_matching as bm
import align_paintings
from transform_model import Transform
from robust_metrics import RobustMetrics


# ---------------------------------------------------------------------------
# utilita' comuni
# ---------------------------------------------------------------------------

def _clahe(img, clip=3.0):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)


def _anteprima(path, fattore, con_clahe=False):
    """Anteprima in grigio a 8 bit, ridotta del fattore indicato."""
    meta = image_io.info(path)
    target = max(meta.height, meta.width) // fattore
    img, f = image_io.leggi_ridotta(path, target, in_grigio=True, verbose=False)
    img = image_io.a_8bit(img)
    if con_clahe:
        img = _clahe(img)
    return img, f


def _coppia(path_rif, path_tgt, fattore, con_clahe=False):
    """
    Due anteprime riportate alla stessa scala.

    Le radiografie hanno dimensioni diverse dal visibile, quindi la seconda
    viene riscalata isotropicamente: senza questo passaggio nessun metodo
    potrebbe funzionare, e non e' un vantaggio per nessuno in particolare.
    """
    a, fa = _anteprima(path_rif, fattore, con_clahe)
    b, fb = _anteprima(path_tgt, fattore, con_clahe)
    rap = float(np.sqrt((a.shape[0] / b.shape[0]) * (a.shape[1] / b.shape[1])))
    if abs(rap - 1.0) > 0.02:
        nuova = (max(8, int(round(b.shape[1] * rap))), max(8, int(round(b.shape[0] * rap))))
        b = cv2.resize(b, nuova, interpolation=cv2.INTER_AREA if rap < 1 else cv2.INTER_LINEAR)
        fb = fb / rap
    return a, b, fa, fb


def _forma_riferimento(path_rif):
    m = image_io.info(path_rif)
    return (m.height, m.width)


# ---------------------------------------------------------------------------
# 1-2. il nostro allineatore
# ---------------------------------------------------------------------------

def aligner_completo(path_rif, path_tgt, cartella_diag=None):
    t0 = time.time()
    T, rep = align_paintings.align(
        path_rif, path_tgt, out_path=None,
        params_path=os.path.join(cartella_diag or '.', 'par.json'),
        factor=2, nonrigid=True, tre_scale='auto',      # come il lotto
        diagnostics_dir=cartella_diag or '.', verbose=False)
    return T, {'secondi': time.time() - t0, 'fattore_lavoro': 2,
               'campo_applicato': rep['stima'].get('nonrigid', {}).get('applicato', False)}


def aligner_affine(path_rif, path_tgt, cartella_diag=None):
    t0 = time.time()
    T, _ = align_paintings.align(
        path_rif, path_tgt, out_path=None,
        params_path=os.path.join(cartella_diag or '.', 'par_aff.json'),
        factor=2, nonrigid=False,
        diagnostics_dir=cartella_diag or '.', verbose=False)
    return T, {'secondi': time.time() - t0, 'fattore_lavoro': 2}


# ---------------------------------------------------------------------------
# 3. MIRA (SimpleITK, Mattes Mutual Information, affine, ROI centrale, CLAHE)
# ---------------------------------------------------------------------------

def mira(path_rif, path_tgt, cartella_diag=None, iterazioni=800, roi_pct=60):
    import SimpleITK as sitk
    t0 = time.time()

    def grigio(path):
        arr, meta = image_io.apri(path, verbose=False)
        h, w = meta.height, meta.width
        out = np.empty((h, w), np.uint8)
        for y0 in range(0, h, 4096):
            y1 = min(h, y0 + 4096)
            b = np.asarray(arr[y0:y1])
            if meta.dtype == np.uint16:
                b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if b.ndim == 3:
                b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            out[y0:y1] = b
        del arr
        return _clahe(out)

    ref_g, tgt_g = grigio(path_rif), grigio(path_tgt)
    # MIRA richiede immagini della stessa dimensione: se non lo sono (le
    # radiografie) si riporta la seconda a quella del riferimento, che e'
    # esattamente cio' che il suo manuale chiede all'utente di fare prima.
    if tgt_g.shape != ref_g.shape:
        tgt_g = cv2.resize(tgt_g, (ref_g.shape[1], ref_g.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = ref_g.shape
    m = (100 - roi_pct) / 200.0
    sx, sy = int(w * m), int(h * m)
    ex, ey = int(w * (1 - m)), int(h * (1 - m))
    ref_p, tgt_p = ref_g[sy:ey, sx:ex], tgt_g[sy:ey, sx:ex]

    sitk_fixed = sitk.Cast(sitk.GetImageFromArray(tgt_p), sitk.sitkFloat32)
    sitk_moving = sitk.Cast(sitk.GetImageFromArray(ref_p), sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(
        sitk_fixed, sitk_moving, sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(1.5, 1e-5, iterazioni)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    finale = reg.Execute(sitk_fixed, sitk_moving)

    ap, fp = np.array(finale.GetParameters()), np.array(finale.GetFixedParameters())
    mat, off, cen = ap[0:4].reshape(2, 2), ap[4:6], fp[0:2]
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = mat
    M[:, 2] = off + cen - mat @ cen
    M[0, 2] += sx - (mat[0, 0] * sx + mat[0, 1] * sy)
    M[1, 2] += sy - (mat[1, 0] * sx + mat[1, 1] * sy)

    # MIRA lavora sul bersaglio ridimensionato come il riferimento: se era di
    # dimensioni diverse va rimessa la scala, altrimenti la trasformazione
    # punterebbe a coordinate sbagliate nel file originale.
    meta_t = image_io.info(path_tgt)
    s = np.array([meta_t.width / ref_g.shape[1], meta_t.height / ref_g.shape[0]])
    M_tgt = M.copy()
    M_tgt[0, :] *= s[0]
    M_tgt[1, :] *= s[1]
    T = Transform(cv2.invertAffineTransform(M_tgt.astype(np.float32)),
                  ref_shape=_forma_riferimento(path_rif))
    return T, {'secondi': time.time() - t0, 'fattore_lavoro': 1,
               'iterazioni': int(reg.GetOptimizerIteration()),
               'mi_finale': float(reg.GetMetricValue())}


# ---------------------------------------------------------------------------
# 4-6. metriche robuste + ottimizzatore Powell, su piramide a due livelli
# ---------------------------------------------------------------------------

def _powell_piramide(path_rif, path_tgt, metrica, livelli=(16, 8), evals=(300, 250)):
    """
    L'approccio "classico": si sceglie una metrica robusta e la si massimizza
    con un ottimizzatore generico sui sei parametri affini.

    Serve una piramide perche' l'ottimizzatore da solo si perde: si parte da
    un'immagine molto ridotta, dove il problema e' facile, e si raffina.

    ATTENZIONE alla convenzione: optimize_affine_mind passa la matrice a
    cv2.warpAffine per portare la MOVING nel sistema della FIXED, quindi la
    matrice che restituisce mappa moving -> fixed. La convenzione del progetto e'
    l'opposta, percio' alla fine si inverte. Verificato con un caso a verita'
    nota: applicandola senza invertire il residuo peggiora di sei volte.

    Come al nostro metodo, si concede un'inizializzazione grossolana della
    traslazione: senza, l'ottimizzatore parte troppo lontano e si ferma al primo
    minimo locale. Il segno e' negato proprio per la convenzione di cui sopra.
    """
    t0 = time.time()
    params = None
    f_rif = f_tgt = None
    costo = float('nan')
    for i, fatt in enumerate(livelli):
        a, b, f_rif, f_tgt = _coppia(path_rif, path_tgt, fatt)
        if params is None:
            ox, oy = bm.coarse_translation(bm.gradient_magnitude(a),
                                           bm.gradient_magnitude(b), verbose=False)
            params = np.array([1, 0, -ox, 0, 1, -oy], np.float32)
        else:
            rapporto = livelli[i - 1] / fatt
            params = params.copy()
            params[2] *= rapporto
            params[5] *= rapporto
        params, costo = RobustMetrics.optimize_affine_mind(
            a, b, initial_params=params, metric=metrica,
            max_iterations=300, max_function_evals=evals[i],
            max_megapixels=60.0, verbose=False)

    M_mov_fix = np.asarray(params, np.float32).reshape(2, 3)
    M_prev = cv2.invertAffineTransform(M_mov_fix).astype(np.float64)
    T = Transform.from_factor(M_prev, f_rif, f_moving=f_tgt,
                              ref_shape=_forma_riferimento(path_rif))
    return T, {'secondi': time.time() - t0, 'fattore_lavoro': livelli[-1],
               'costo_finale': float(costo)}


def bspline_mi(path_rif, path_tgt, cartella_diag=None, fattore=8, nodi=8,
               iterazioni=200):
    """
    Registrazione NON RIGIDA con B-spline e Mattes Mutual Information (SimpleITK).

    E' il confronto che mancava: MIRA con un modello non rigido al posto
    dell'affine. E' anche l'impostazione classica della registrazione deformabile
    multimodale in ambito medico, quella di elastix e ANTs: una griglia di nodi
    di controllo B-spline la cui deformazione massimizza l'informazione mutua.

    Si lavora su un'anteprima ridotta perche' una B-spline a piena risoluzione
    su 300 Mpx non sta in memoria: il campo viene poi riportato alla scala piena
    campionandolo sulla griglia del nostro Transform, che e' esattamente lo
    stesso modo in cui trattiamo il nostro campo.

    Prima della B-spline si stima un'affine con lo stesso criterio, perche'
    partire dall'identita' con una deformabile e' il modo classico di finire in
    un minimo locale.

    ATTENZIONE: questa configurazione NON e' tarata e i suoi risultati non vanno
    usati come giudizio sul metodo. Misurato: sull'immagine ridotta non scende
    sotto i 5.8 px (peggio della sola affine), perche' a fattore 8 la
    deformazione da trovare vale meno di un pixel; su ritagli a piena
    risoluzione arriva a 0.03 px su un ritaglio e diverge a 370 px su un altro.
    Manca la penalita' sull'energia di curvatura che evita quelle divergenze e
    che i file di parametri di elastix includono di serie. Vedi README.md del
    benchmark.
    """
    import SimpleITK as sitk
    t0 = time.time()
    a, b, fa, fb = _coppia(path_rif, path_tgt, fattore, con_clahe=True)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    # In SimpleITK il "fixed" e' l'immagine nel cui sistema si resta, e la
    # trasformazione mappa fixed -> moving: la stessa convenzione del progetto.
    fix = sitk.Cast(sitk.GetImageFromArray(a), sitk.sitkFloat32)
    mov = sitk.Cast(sitk.GetImageFromArray(b), sitk.sitkFloat32)

    def base():
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        r.SetMetricSamplingStrategy(r.NONE)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetShrinkFactorsPerLevel([4, 2, 1])
        r.SetSmoothingSigmasPerLevel([2, 1, 0])
        r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return r

    # --- 1. affine, con le stesse impostazioni che funzionano in MIRA ---
    aff0 = sitk.CenteredTransformInitializer(
        fix, mov, sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r_aff = base()
    r_aff.SetOptimizerAsRegularStepGradientDescent(1.5, 1e-5, iterazioni)
    r_aff.SetOptimizerScalesFromPhysicalShift()
    r_aff.SetInitialTransform(aff0, inPlace=False)
    tx_aff = r_aff.Execute(fix, mov)

    # --- 2. B-spline SOPRA l'affine ---
    # SetMovingInitialTransform fissa l'affine gia' trovata e NON la ottimizza:
    # la B-spline stima solo cio' che resta. Senza questo, la B-spline
    # ripartirebbe da zero e componendola con l'affine lo spostamento verrebbe
    # contato due volte (misurato: campo rms 113 px invece di ~10).
    bs = sitk.BSplineTransformInitializer(fix, [nodi, nodi], order=3)
    r_bs = base()
    r_bs.SetMovingInitialTransform(tx_aff)
    r_bs.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                              numberOfIterations=iterazioni)
    r_bs.SetInitialTransform(bs, inPlace=False)
    try:
        tx_bs = r_bs.Execute(fix, mov)
        mi = float(r_bs.GetMetricValue())
    except Exception as exc:
        return None, {'errore': f'B-spline fallita: {exc}', 'secondi': time.time() - t0}

    composita = sitk.CompositeTransform([tx_aff, tx_bs])

    # --- 3. dal modello ITK al nostro Transform ---
    # Si campiona lo spostamento completo (affine inclusa) sulla griglia del
    # campo e si lascia M all'identita'. Il nostro Transform vale
    # q = M*p + F(p): con M identita' e F lo spostamento totale il risultato e'
    # identico, e non serve separare la parte lineare (una separazione ai minimi
    # quadrati introdurrebbe un errore proprio dove il campo e' grande).
    # L'interpolazione bilineare del campo su griglia regolare e' esatta per la
    # parte lineare, quindi non si perde nulla.
    filtro = sitk.TransformToDisplacementFieldFilter()
    filtro.SetReferenceImage(fix)
    filtro.SetOutputPixelType(sitk.sitkVectorFloat64)
    disp = sitk.GetArrayFromImage(filtro.Execute(composita))   # (h, w, 2) dx, dy

    passo = 16                                   # px di anteprima fra i nodi
    gh = int(np.ceil(a.shape[0] / passo)) + 1
    gw = int(np.ceil(a.shape[1] / passo)) + 1
    campo = np.zeros((gh, gw, 2), np.float32)
    for i in range(gh):
        y = min(i * passo, disp.shape[0] - 1)
        for j in range(gw):
            x = min(j * passo, disp.shape[1] - 1)
            campo[i, j] = disp[y, x]

    T = Transform.from_factor(np.array([[1.0, 0, 0], [0, 1.0, 0]]), fa,
                              field=campo, grid_step=passo,
                              f_moving=fb, ref_shape=_forma_riferimento(path_rif))
    amp = float(np.sqrt((campo ** 2).sum(axis=2).mean())) * fa
    return T, {'secondi': time.time() - t0, 'fattore_lavoro': fattore,
               'nodi_bspline': nodi, 'mi_finale': mi,
               'spostamento_rms_px_full': amp}


def mind_powell(path_rif, path_tgt, cartella_diag=None):
    return _powell_piramide(path_rif, path_tgt, 'mind')


def ngf_powell(path_rif, path_tgt, cartella_diag=None):
    return _powell_piramide(path_rif, path_tgt, 'ngf')


def nmi_powell(path_rif, path_tgt, cartella_diag=None):
    return _powell_piramide(path_rif, path_tgt, 'nmi')


# ---------------------------------------------------------------------------
# 7. SIFT + RANSAC (con CLAHE), l'approccio "a punti notevoli"
# ---------------------------------------------------------------------------

def sift_ransac(path_rif, path_tgt, cartella_diag=None, fattore=4):
    t0 = time.time()
    a, b, fa, fb = _coppia(path_rif, path_tgt, fattore, con_clahe=True)

    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    info = {'secondi': time.time() - t0, 'fattore_lavoro': fattore,
            'punti_rif': len(ka), 'punti_tgt': len(kb)}
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        info['errore'] = 'troppi pochi punti notevoli'
        return None, info

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    corrisp = flann.knnMatch(db, da, k=2)
    buone = [m for m, n in corrisp if m.distance < 0.75 * n.distance]
    info['corrispondenze'] = len(buone)
    if len(buone) < 10:
        info['errore'] = f'solo {len(buone)} corrispondenze affidabili'
        return None, info

    src = np.float32([kb[m.queryIdx].pt for m in buone]).reshape(-1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in buone]).reshape(-1, 2)
    # src sta nell'immagine da allineare, dst nel riferimento: serve la mappa
    # riferimento -> immagine, quindi (dst -> src)
    M, inl = cv2.estimateAffine2D(dst, src, method=cv2.RANSAC,
                                  ransacReprojThreshold=3.0, maxIters=5000)
    if M is None:
        info['errore'] = 'RANSAC non ha trovato un modello'
        return None, info
    info['inlier'] = int(inl.sum())
    info['secondi'] = time.time() - t0
    T = Transform.from_factor(M, fa, f_moving=fb, ref_shape=_forma_riferimento(path_rif))
    return T, info


# ---------------------------------------------------------------------------
# 8. ECC (correlazione migliorata), l'approccio di TestUI
# ---------------------------------------------------------------------------

def ecc(path_rif, path_tgt, cartella_diag=None, fattore=8):
    """
    ECC parte da una stima grossolana, come in TestUI.py, dove veniva sempre
    dopo un allineamento globale. Da solo, a partire dall'identita', su queste
    immagini non converge: non e' un difetto del metodo, e' il modo in cui va
    impiegato.
    """
    t0 = time.time()
    a, b, fa, fb = _coppia(path_rif, path_tgt, fattore, con_clahe=True)

    # ECC si applica al MODULO DEL GRADIENTE, non ai livelli di grigio. Il suo
    # criterio e' una correlazione, che presuppone una relazione lineare fra le
    # intensita' delle due immagini: fra bande diverse quella relazione non
    # esiste, e sulle intensita' grezze ECC non converge mai (verificato su
    # tutte le coppie di questo progetto). Sui gradienti converge, perche' li'
    # la struttura e' condivisa.
    A, B = bm.gradient_magnitude(a), bm.gradient_magnitude(b)

    # Raggio di cattura misurato: circa 8 px. Oltre, non converge. Serve quindi
    # una stima grossolana di partenza, come in TestUI.py dove ECC veniva
    # sempre dopo un allineamento globale.
    ox, oy = bm.coarse_translation(A, B, verbose=False)
    warp = np.array([[1, 0, ox], [0, 1, oy]], np.float32)

    info = {'fattore_lavoro': fattore, 'inizializzazione_px': [int(ox), int(oy)],
            'nota': 'applicato al modulo del gradiente'}
    ultimo = ''
    for tentativo, (iterazioni, eps) in enumerate(((500, 1e-7), (200, 1e-5), (100, 1e-4))):
        try:
            criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterazioni, eps)
            cc, w = cv2.findTransformECC(A, B, warp.copy(), cv2.MOTION_AFFINE,
                                         criterio, None, 5)
            info['correlazione'] = float(cc)
            info['tentativo'] = tentativo + 1
            info['secondi'] = time.time() - t0
            # ECC restituisce gia' la mappa fisso -> moving: non va invertita
            # (verificato con un caso a verita' nota).
            T = Transform.from_factor(w.astype(np.float64), fa, f_moving=fb,
                                      ref_shape=_forma_riferimento(path_rif))
            return T, info
        except cv2.error as exc:
            ultimo = str(exc).splitlines()[-1][:80]
    info['errore'] = f'ECC non converge: {ultimo}'
    info['secondi'] = time.time() - t0
    return None, info


METODI = {
    'Aligner (completo)': aligner_completo,
    'Aligner (solo affine)': aligner_affine,
    'B-spline + MI (SimpleITK)': bspline_mi,
    'MIRA (Mattes MI)': mira,
    'MIND + Powell': mind_powell,
    'NGF + Powell': ngf_powell,
    'NMI + Powell': nmi_powell,
    'SIFT + RANSAC': sift_ransac,
    'ECC': ecc,
}
