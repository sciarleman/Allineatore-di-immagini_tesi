"""
Metriche robuste per registrazione cross-modale di dipinti.

Contenuto:
- MIND (Modality-Independent Neighborhood Descriptor): invarianza all'inversione di contrasto
- NGF (Normalized Gradient Fields): allineamento solo di direzioni, non modulo
- NMI (Normalized Mutual Information): baseline robusta
- SSC (Self-Similarity Context): versione robusta di MIND

Rispetto alla versione originale:
- np.roll (wraparound) sostituito con shift a bordi riflessi (np.pad + slicing),
  perché il wraparound introduce falsa similarità tra i due lati opposti
  dell'immagine (o del tile, quando lavoriamo a tile).
- Le funzioni ora accettano anche piccoli tile senza cambiare comportamento.

Utilizzo: applica queste metriche dopo la calibrazione/undistortion e la
inizializzazione con phase correlation. Per immagini grandi, usa
pyramid_registration.py invece di chiamare optimize_affine_mind direttamente
sull'immagine a piena risoluzione.
"""

import cv2
import numpy as np
from scipy import optimize
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def _shift_reflect(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Trasla un'immagine di (dy, dx) pixel usando riflessione ai bordi,
    invece del wraparound di np.roll.

    np.roll(img, dy, axis=0) prende i pixel che "escono" da un lato e li
    reinserisce dal lato opposto: per un dipinto questo vuol dire confrontare
    il bordo superiore con quello inferiore come se fossero vicini, il che
    è un artefatto puro. Con la riflessione, il bordo viene invece
    specchiato su se stesso, che è l'assunzione standard (usata anche da
    cv2.BORDER_REFLECT) per dati che non sono periodici.
    """
    pad_y = abs(dy)
    pad_x = abs(dx)
    if pad_y == 0 and pad_x == 0:
        return img.copy()

    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

    h, w = img.shape
    start_y = pad_y - dy
    start_x = pad_x - dx
    return padded[start_y:start_y + h, start_x:start_x + w]


class RobustMetrics:
    """Metriche robuste per registrazione cross-modale."""

    # ========================================================================
    # MIND: Modality-Independent Neighborhood Descriptor
    # ========================================================================

    @staticmethod
    def mind(img: np.ndarray,
             sigma: float = 1.0,
             patch_size: int = 1,
             offsets: Optional[list] = None) -> np.ndarray:
        """
        MIND descriptor: codifica l'auto-similarità locale.

        Invariante a:
        - Inversione locale di contrasto (usando rapporto SSD/varianza)
        - Differenze fotometriche non stazionarie

        Args:
            img: immagine di input (grayscale, float32, [0, 1] o [0, 255])
            sigma: smoothing gaussiano pre-descriptor
            patch_size: raggio della patch per SSD (1 = 3x3)
            offsets: lista di (dy, dx) offset del vicinato
                    default: 8-connected (+ diagonali)

        Returns:
            array (n_offsets, H, W) float32: descriptor per pixel.

            NOTA: l'asse dei canali è il PRIMO (n_offsets, H, W), non
            l'ultimo come in una prima versione. Motivo: np.stack lungo
            l'ultimo asse produce un array non contiguo in memoria (le
            singole slice non sono adiacenti), il che rende le successive
            riduzioni (.mean(axis=2), .max(axis=2)) molto più lente perché
            NumPy deve saltare in memoria in modo non sequenziale — misurato
            un fattore ~2x più lento in totale, e fino a 15x più lento sulla
            sola operazione di max, rispetto a tenere i canali sul primo
            asse (row-major friendly). Dato che optimize_affine_mind
            ricalcola questo descriptor ad ogni valutazione della funzione
            obiettivo (centinaia di volte per ottimizzazione), questo
            fattore 2x si traduce direttamente in metà del tempo totale di
            registrazione. Il confronto tra descriptor (MSE) è invariante
            all'ordine degli assi, quindi non serve convertire indietro.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0

        I = cv2.GaussianBlur(img, (0, 0), sigma)

        if offsets is None:
            offsets = [
                (0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (-1, -1), (1, -1), (-1, 1)
            ]

        patch_side = 2 * patch_size + 1
        kernel_size = (patch_side, patch_side)

        diffs = []
        for dy, dx in offsets:
            I_offset = _shift_reflect(I, dy, dx)
            diff_sq = (I - I_offset) ** 2
            ssd = cv2.boxFilter(diff_sq, -1, kernel_size, normalize=False)
            diffs.append(ssd)

        diffs = np.stack(diffs, axis=0)

        V = np.maximum(diffs.mean(axis=0, keepdims=True), 1e-6)
        mind_desc = np.exp(-diffs / V)

        mind_max = np.maximum(mind_desc.max(axis=0, keepdims=True), 1e-6)
        mind_desc = mind_desc / mind_max

        return mind_desc.astype(np.float32)

    @staticmethod
    def ssc(img: np.ndarray,
            sigma: float = 1.0,
            patch_size: int = 1) -> np.ndarray:
        """
        Self-Similarity Context: versione più robusta di MIND.

        Confronta coppie di vicini fra loro, non il centro con i vicini.
        Riduce il rumore sul pixel centrale ed è migliore su strutture
        ripetitive (come le craquelure).
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0

        I = cv2.GaussianBlur(img, (0, 0), sigma)

        patch_side = 2 * patch_size + 1
        kernel_size = (patch_side, patch_side)

        offsets_pairs = [
            ((1, 0), (0, 1)),
            ((1, 0), (-1, 0)),
            ((1, 0), (0, -1)),
            ((0, 1), (-1, 0)),
            ((0, 1), (0, -1)),
            ((-1, 0), (0, -1)),
            ((1, 1), (-1, -1)),
            ((1, -1), (-1, 1)),
        ]

        comparisons = []
        for (dy1, dx1), (dy2, dx2) in offsets_pairs:
            patch1 = _shift_reflect(I, dy1, dx1)
            patch2 = _shift_reflect(I, dy2, dx2)

            diff_sq = (patch1 - patch2) ** 2
            ssd = cv2.boxFilter(diff_sq, -1, kernel_size, normalize=False)
            comparisons.append(ssd)

        # axis=0 (non axis=2): stessa ragione di performance spiegata in mind().
        comparisons = np.stack(comparisons, axis=0)

        V = np.maximum(comparisons.mean(axis=0, keepdims=True), 1e-6)
        exp_term = np.exp(-comparisons / V)
        ssc_desc = exp_term / np.maximum(exp_term.max(axis=0, keepdims=True), 1e-6)

        return ssc_desc.astype(np.float32)

    # ========================================================================
    # NGF: Normalized Gradient Fields
    # ========================================================================

    @staticmethod
    def ngf(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        Normalized Gradient Fields: misura l'allineamento delle direzioni
        dei gradienti. Ignora modulo e segno del gradiente: utile per crepe
        che hanno contrasto invertito tra VIS e UV.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0

        img_smooth = cv2.GaussianBlur(img, (0, 0), sigma)

        gx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)

        norm = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        ngf = np.stack([gx / norm, gy / norm], axis=2)

        return ngf.astype(np.float32)

    # ========================================================================
    # Mutual Information e varianti
    # ========================================================================

    @staticmethod
    def compute_histogram_2d(a: np.ndarray, b: np.ndarray, bins: int = 64) -> np.ndarray:
        hist, _, _ = np.histogram2d(
            a.ravel(), b.ravel(),
            bins=bins,
            range=[[a.min(), a.max()], [b.min(), b.max()]]
        )
        hist = hist / hist.sum()
        return hist

    @staticmethod
    def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
        hist = RobustMetrics.compute_histogram_2d(a, b, bins)

        pa = hist.sum(axis=1)
        pb = hist.sum(axis=0)

        H_a = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
        H_b = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
        H_ab = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))

        return float(H_a + H_b - H_ab)

    @staticmethod
    def normalized_mutual_information(a: np.ndarray, b: np.ndarray,
                                       bins: int = 64) -> float:
        """NMI = (H(A) + H(B)) / H(A, B)"""
        hist = RobustMetrics.compute_histogram_2d(a, b, bins)

        pa = hist.sum(axis=1)
        pb = hist.sum(axis=0)

        H_a = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
        H_b = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
        H_ab = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))

        return float((H_a + H_b) / np.maximum(H_ab, 1e-6))

    # ========================================================================
    # Trasformazione affine
    # ========================================================================

    @staticmethod
    def param_to_affine_matrix(params: np.ndarray) -> np.ndarray:
        """params = [a, b, tx, c, d, ty] -> matrice 2x3"""
        return np.asarray(params, dtype=np.float32).reshape(2, 3)

    @staticmethod
    def affine_to_param(mat: np.ndarray) -> np.ndarray:
        return mat.flatten()

    @staticmethod
    def warp_image_affine(img: np.ndarray, params: np.ndarray,
                           output_shape: Optional[Tuple] = None,
                           border_mode: int = cv2.BORDER_REFLECT) -> np.ndarray:
        """Applica trasformazione affine a un'immagine (o tile)."""
        mat = RobustMetrics.param_to_affine_matrix(params)

        if output_shape is None:
            output_shape = img.shape[:2]

        warped = cv2.warpAffine(img, mat, (output_shape[1], output_shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=border_mode)
        return warped

    # ========================================================================
    # Ottimizzazione con scipy (uso: SOLO su immagini/downsample piccoli,
    # vedi pyramid_registration.py per la strategia multi-risoluzione)
    # ========================================================================

    @staticmethod
    def optimize_affine_mind(img_fixed: np.ndarray,
                              img_moving: np.ndarray,
                              initial_params: Optional[np.ndarray] = None,
                              metric: str = 'mind',
                              max_iterations: int = 100,
                              max_function_evals: Optional[int] = None,
                              max_megapixels: float = 4.0,
                              verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Ottimizza la trasformazione affine usando MIND o altre metriche.

        ATTENZIONE PERFORMANCE: questa funzione ricalcola il descriptor
        dell'immagine moving ad ogni VALUTAZIONE della funzione obiettivo
        (non ad ogni "iterazione" di Powell: Powell fa molte più valutazioni
        di funzione che iterazioni del suo ciclo esterno — misurato: ~900
        valutazioni per un'immagine 400x300px con maxiter=200). Per questo
        `max_function_evals` è il parametro che conta davvero per limitare
        il tempo, non `max_iterations` da solo: entrambi vengono passati a
        scipy, che si ferma al primo limite raggiunto.

        FIX IMPORTANTE rispetto a una prima versione: i 6 parametri affini
        hanno scale fisicamente diverse (a,b,c,d sono adimensionali, ~1;
        tx,ty sono in pixel, possono essere decine). Powell (e Nelder-Mead)
        esplorano tutte le direzioni con lo stesso passo iniziale, quindi
        senza normalizzazione convergono molto più lentamente sulle
        direzioni "piccole" (a,b,c,d). Qui ottimizziamo in uno spazio
        riscalato (parametri divisi per `param_scale`) e poi denormalizziamo
        il risultato: a parità di valutazioni, converge circa 2x meglio
        (verificato sperimentalmente: cost 0.0037 con 400 eval scalate,
        contro cost 0.026 con 400 eval non scalate, sullo stesso problema).

        Va usata SOLO dentro una piramide multi-risoluzione (vedi
        pyramid_registration.PyramidRegistration) per immagini grandi, mai
        sull'immagine a piena risoluzione (15000x20000): anche con questi
        fix, il costo per valutazione resta proporzionale ai pixel.
        """
        n_mpx = (img_fixed.shape[0] * img_fixed.shape[1]) / 1e6
        if n_mpx > max_megapixels:
            raise MemoryError(
                f"Immagine da {n_mpx:.1f} Mpx: troppo grande per questa funzione.\n"
                f"Ogni valutazione della funzione obiettivo costruisce un descriptor MIND di "
                f"{n_mpx * 8 * 4:.0f} MB e ne servono centinaia -> la macchina va in swap e "
                f"il processo viene ucciso (e' esattamente il crash che si voleva evitare).\n"
                f"Usa align_paintings.py, che lavora su anteprime ridotte e poi applica la "
                f"trasformazione a piena risoluzione in streaming; oppure, se vuoi davvero "
                f"forzare, alza max_megapixels consapevolmente.")

        if initial_params is None:
            initial_params = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        else:
            initial_params = np.asarray(initial_params, dtype=np.float32).flatten()

        if max_function_evals is None:
            # Budget di default se non specificato esplicitamente dal chiamante
            # (pyramid_registration.py lo calibra in base alla dimensione immagine).
            max_function_evals = max_iterations * 4

        if verbose:
            print(f"  [optimize_affine_mind] metrica={metric}, "
                  f"shape={img_fixed.shape}, max_iter={max_iterations}, "
                  f"max_fev={max_function_evals}")

        if len(img_fixed.shape) == 3:
            img_fixed_gray = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2GRAY)
        else:
            img_fixed_gray = img_fixed

        if len(img_moving.shape) == 3:
            img_moving_gray = cv2.cvtColor(img_moving, cv2.COLOR_BGR2GRAY)
        else:
            img_moving_gray = img_moving

        img_fixed_gray = img_fixed_gray.astype(np.float32)
        img_moving_gray = img_moving_gray.astype(np.float32)

        if img_fixed_gray.max() > 1:
            img_fixed_gray = img_fixed_gray / 255.0
        if img_moving_gray.max() > 1:
            img_moving_gray = img_moving_gray / 255.0

        h, w = img_fixed_gray.shape

        if metric in ('mind', 'ssc'):
            desc_fn = RobustMetrics.mind if metric == 'mind' else RobustMetrics.ssc
            desc_fixed = desc_fn(img_fixed_gray)

            def objective(params):
                img_warped = RobustMetrics.warp_image_affine(img_moving_gray, params, (h, w))
                desc_moving = desc_fn(img_warped)
                return float(np.mean((desc_fixed - desc_moving) ** 2))

        elif metric == 'ngf':
            ngf_fixed = RobustMetrics.ngf(img_fixed_gray)

            def objective(params):
                img_warped = RobustMetrics.warp_image_affine(img_moving_gray, params, (h, w))
                ngf_moving = RobustMetrics.ngf(img_warped)
                dot = (ngf_fixed * ngf_moving).sum(axis=2)
                return float(-dot.mean())

        elif metric == 'nmi':
            def objective(params):
                img_warped = RobustMetrics.warp_image_affine(img_moving_gray, params, (h, w))
                nmi = RobustMetrics.normalized_mutual_information(img_fixed_gray, img_warped, bins=32)
                return -nmi

        else:
            raise ValueError(f"Metrica non supportata: {metric}")

        # Scala tipica di ciascun parametro: 0.05 per i coefficienti lineari
        # (a,b,c,d intorno a 1, variazioni tipiche di rotazione/scala/shear
        # dell'ordine di qualche percento), 5.0 px per le traslazioni (tx,ty).
        # Con questo scaling, uno spostamento di "1" nello spazio interno di
        # Powell corrisponde a variazioni di grandezza comparabile sul costo,
        # in tutte le direzioni.
        param_scale = np.array([0.05, 0.05, 5.0, 0.05, 0.05, 5.0], dtype=np.float64)
        identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)

        x0_scaled = (initial_params.astype(np.float64) - identity) / param_scale

        def objective_scaled(p_scaled):
            p = p_scaled * param_scale + identity
            return objective(p.astype(np.float32))

        result = optimize.minimize(
            objective_scaled,
            x0_scaled,
            method='Powell',
            options={
                'maxiter': max_iterations,
                'maxfev': max_function_evals,
                'xtol': 1e-2,
                'ftol': 1e-5,
                'disp': False,
            }
        )

        params_final = (result.x * param_scale + identity).astype(np.float32)

        if verbose:
            print(f"  -> convergenza={result.success}, iter={result.nit}, "
                  f"nfev={result.nfev}, costo_finale={result.fun:.6f}")

        return params_final, float(result.fun)

    # ========================================================================
    # Validazione
    # ========================================================================

    @staticmethod
    def compute_registration_error(img_fixed: np.ndarray,
                                    img_moving_warped: np.ndarray,
                                    control_points: Optional[np.ndarray] = None):
        if len(img_fixed.shape) == 3:
            img_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2GRAY)
        if len(img_moving_warped.shape) == 3:
            img_moving_warped = cv2.cvtColor(img_moving_warped, cv2.COLOR_BGR2GRAY)

        img_fixed = img_fixed.astype(np.float32)
        img_moving_warped = img_moving_warped.astype(np.float32)

        diff = img_fixed - img_moving_warped
        rms_pixel = np.sqrt(np.mean(diff ** 2))

        result = {'rms_pixel': float(rms_pixel)}

        if control_points is not None and len(control_points) > 0:
            errors = []
            for xf, yf, xm, ym in control_points:
                xf, yf = int(xf), int(yf)
                xm, ym = int(xm), int(ym)

                if 0 <= xf < img_fixed.shape[1] and 0 <= yf < img_fixed.shape[0]:
                    if 0 <= xm < img_moving_warped.shape[1] and 0 <= ym < img_moving_warped.shape[0]:
                        val_fixed = img_fixed[yf, xf]
                        val_moving = img_moving_warped[ym, xm]
                        errors.append(abs(val_fixed - val_moving))

            if errors:
                result['rms_control_points'] = float(np.sqrt(np.mean(np.array(errors) ** 2)))

        return result
