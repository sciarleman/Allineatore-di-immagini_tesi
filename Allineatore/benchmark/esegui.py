"""
Esegue il confronto fra i metodi e ne misura il risultato.

    python esegui.py --prova          una sola coppia, per controllare che tutto giri
    python esegui.py                  tutte le coppie

La valutazione e' IDENTICA per tutti i metodi e usa cinque criteri di famiglie
diverse, fra cui quello che MIRA stessa ottimizza. Nessun metodo viene giudicato
con il proprio criterio soltanto: e' l'unico modo per non barare a favore di uno.

I risultati finiscono in risultati.json, da cui grafici.py produce le figure.
"""

import argparse
import ctypes
import json
import os
import sys
import time
import traceback
from ctypes import wintypes

import numpy as np
import cv2

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)                   # Allineatore/, dove sta il motore
RADICE = os.path.dirname(PROG)                # la cartella del progetto, con immagini/
sys.path.insert(0, PROG)

import image_io
import validation as val
from robust_metrics import RobustMetrics
import metodi

IMMAGINI = os.path.join(RADICE, 'immagini')
if not os.path.exists(os.path.join(IMMAGINI, 'VIS_FraGalgario_TOT.tif')):
    # le riprese sono dati di ricerca, fuori dal repository (come in radiografie.py)
    IMMAGINI = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
RIFERIMENTO = os.path.join(IMMAGINI, 'VIS_FraGalgario_TOT.tif')
BERSAGLI = ['UVF_FraGalgario_TOT.tif', 'UVR_FraGalgario_TOT.tif',
            'NIR850_FraGalgario_TOT.tif', 'NIR950_FraGalgario_TOT.tif',
            'RX_FraGalgarioTOT_TestaChiara.tif',
            'RX_FraGalgario_TOT_Cappellochiaro.tif']
# La seconda opera (--affresco <cartella delle copie TIFF>): riprese da 593 Mpx,
# di dimensioni diverse dal riferimento.
RIFERIMENTO_AFFRESCO = 'VIS.tif'
BERSAGLI_AFFRESCO = ['NIR.tif', 'UV.tif']
GUARDIA_GB = 0.6

PATCH = 512
# 8x8 = 64 finestre. Con 25 (la versione precedente) la mediana oscillava fra
# 0.35 e 0.67 px contro un valore vero di 0.50: differenze di un pixel fra i
# metodi erano dentro il rumore della misura.
N_LATO = 8


def memoria_gb():
    """Memoria privata di picco del processo (Windows)."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(n, ctypes.c_size_t) for n in
                    ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                     "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                     "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        gi = ctypes.windll.psapi.GetProcessMemoryInfo
        gi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        gi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakPagefileUsage / 1024 ** 3
    except Exception:
        return None


def etichetta_di(nome_file):
    """
    Nome breve per i grafici, che distingua le due radiografie.

    Prendere solo la prima parola darebbe 'RX' a entrambe, che finirebbero per
    sovrascriversi nei risultati: si aggiunge quindi l'ultima parola utile del
    nome, quella che le distingue davvero.
    """
    parti = os.path.splitext(nome_file)[0].split('_')
    coda = [p for p in parti[1:]
            if p.lower() != 'tot' and 'fragalgario' not in p.lower()]
    return parti[0] + (' ' + coda[-1] if coda else '')


def _posizioni(H, W):
    """Le stesse finestre di verifica per tutti i metodi e tutte le immagini."""
    pos = []
    for fy in np.linspace(0.15, 0.85, N_LATO):
        for fx in np.linspace(0.15, 0.85, N_LATO):
            pos.append((int(fy * H) - PATCH // 2, int(fx * W) - PATCH // 2))
    return pos


def _mattes(a, b):
    import SimpleITK as sitk
    ia = sitk.Cast(sitk.GetImageFromArray(a.astype(np.float32)), sitk.sitkFloat32)
    ib = sitk.Cast(sitk.GetImageFromArray(b.astype(np.float32)), sitk.sitkFloat32)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    r.SetMetricSamplingStrategy(r.NONE)
    r.SetInitialTransform(sitk.TranslationTransform(2))
    r.SetInterpolator(sitk.sitkLinear)
    return float(r.MetricEvaluate(ia, ib))


def valuta(arr_f, arr_m, T, pos):
    """
    Cinque criteri indipendenti, sugli stessi campioni a piena risoluzione.

    - residuo in px: di quanto le due immagini sono ancora fuori registro.
      E' l'unico direttamente interpretabile, ed e' quello che interessa;
    - MIND: somiglianza di struttura locale, invariante al contrasto;
    - NMI: informazione mutua normalizzata, la baseline storica;
    - Mattes MI: il criterio che MIRA ottimizza (piu' negativo = meglio);
    - NCC sui gradienti: correlazione delle strutture.
    """
    mind, nmi, mattes, ncc = [], [], [], []
    for (y0, x0) in pos:
        f = np.asarray(arr_f[y0:y0 + PATCH, x0:x0 + PATCH])
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        f = image_io.a_8bit(f)
        if T is None:
            m = np.asarray(arr_m[y0:y0 + PATCH, x0:x0 + PATCH])
            if m.shape[:2] != f.shape[:2]:
                continue
        else:
            m = T.warp_tile_from(arr_m, y0, x0, PATCH, PATCH)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = image_io.a_8bit(m)
        if f.std() < 3 or m.std() < 3:
            continue
        mind.append(float(np.mean((RobustMetrics.mind(f) - RobustMetrics.mind(m)) ** 2)))
        nmi.append(RobustMetrics.normalized_mutual_information(f, m, bins=64))
        mattes.append(_mattes(f, m))
        ga = cv2.Sobel(f.astype(np.float32), cv2.CV_32F, 1, 1)
        gb = cv2.Sobel(m.astype(np.float32), cv2.CV_32F, 1, 1)
        ncc.append(float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]))

    r = val.residual_check(arr_f, arr_m, T, n_side=N_LATO, patch=PATCH, verbose=False)
    return {
        'residuo_px': r.get('residuo_mediano_px'),
        'residuo_ic90': r.get('residuo_mediano_ic90'),
        'residuo_p95_px': r.get('residuo_p95_px'),
        'quota_sopra_1px': r.get('quota_sopra_1px'),
        'quota_sopra_2px': r.get('quota_sopra_2px'),
        'quota_sopra_5px': r.get('quota_sopra_5px'),
        'campioni_affidabili': r.get('n_affidabili'),
        'campioni_totali': r.get('n_campioni'),
        'mind': float(np.mean(mind)) if mind else None,
        'nmi': float(np.mean(nmi)) if nmi else None,
        'mattes_mi': float(np.mean(mattes)) if mattes else None,
        'ncc_gradienti': float(np.mean(ncc)) if ncc else None,
        'n_finestre': len(mind),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prova', action='store_true', help="solo la prima coppia")
    p.add_argument('--solo', nargs='+', default=None, help="solo questi metodi")
    p.add_argument('--affresco', default=None, metavar='CARTELLA',
                   help="la seconda opera, dalle copie TIFF in questa cartella")
    args = p.parse_args()

    immagini, riferimento, bersagli = IMMAGINI, RIFERIMENTO, BERSAGLI
    if args.affresco:
        immagini, bersagli = args.affresco, BERSAGLI_AFFRESCO
        riferimento = os.path.join(immagini, RIFERIMENTO_AFFRESCO)
        # a questa taglia un metodo puo' chiedere piu' memoria di quanta ce n'e':
        # meglio che il processo esca prima che il computer si blocchi
        import threading
        import bande_precisioni

        def guardia():
            while True:
                if bande_precisioni._memoria_libera_gb() < GUARDIA_GB:
                    print(f"\nGUARDIA: memoria libera sotto {GUARDIA_GB} GB, esco", flush=True)
                    os._exit(3)
                time.sleep(0.5)
        threading.Thread(target=guardia, daemon=True).start()
    if args.prova:
        bersagli = bersagli[:1]
    nomi_metodi = args.solo if args.solo else list(metodi.METODI)

    arr_f, meta_f = image_io.apri(riferimento, verbose=False)
    pos = _posizioni(meta_f.height, meta_f.width)
    print(f"riferimento: {meta_f}")
    print(f"{len(bersagli)} immagini x {len(nomi_metodi)} metodi, "
          f"valutazione su {len(pos)} finestre da {PATCH} px\n")

    risultati = []
    percorso = os.path.join(QUI, 'risultati.json')
    if os.path.exists(percorso):
        # utf-8-sig e non utf-8: se il file e' stato toccato da un editor o da
        # PowerShell puo' avere un BOM davanti, che json rifiuta.
        risultati = json.load(open(percorso, encoding='utf-8-sig'))
    fatti = {(r['immagine'], r['metodo']) for r in risultati}

    for nome_img in bersagli:
        path_t = os.path.join(immagini, nome_img)
        arr_m, meta_m = image_io.apri(path_t, verbose=False)
        etichetta = ('affresco ' if args.affresco else '') + etichetta_di(nome_img)

        if (etichetta, 'nessun allineamento') not in fatti:
            base = valuta(arr_f, arr_m, None, pos)
            risultati.append({'immagine': etichetta, 'file': nome_img,
                              'metodo': 'nessun allineamento', 'ok': True,
                              'secondi': 0.0, **base})
            r0 = base.get('residuo_px')
            print(f"[{etichetta}] partenza: "
                  + (f"residuo {r0:.1f} px" if r0 is not None else
                     "non misurabile (dimensioni diverse dal riferimento)"))

        for nome_met in nomi_metodi:
            if (etichetta, nome_met) in fatti:
                print(f"[{etichetta}] {nome_met}: gia' fatto, salto")
                continue
            print(f"[{etichetta}] {nome_met} ... ", end='', flush=True)
            diag = os.path.join(QUI, 'lavoro', etichetta, nome_met.replace(' ', '_'))
            os.makedirs(diag, exist_ok=True)
            voce = {'immagine': etichetta, 'file': nome_img, 'metodo': nome_met}
            t0 = time.time()
            try:
                T, info = metodi.METODI[nome_met](riferimento, path_t, diag)
                if T is None:
                    voce.update({'ok': False, 'motivo': info.get('errore', 'fallito'),
                                 'secondi': info.get('secondi', time.time() - t0)})
                    print(f"FALLITO ({voce['motivo']})")
                else:
                    T.save(os.path.join(diag, 'trasformazione.json'))
                    m = valuta(arr_f, arr_m, T, pos)
                    voce.update({'ok': True, **info, **m})
                    res = m.get('residuo_px')
                    testo = f"{res:.2f} px" if res is not None else "non misurabile"
                    print(f"{testo}  ({info.get('secondi', 0):.0f}s, "
                          f"{m.get('campioni_affidabili', 0)}/{m.get('campioni_totali', 0)} campioni)")
            except Exception as exc:
                voce.update({'ok': False, 'motivo': f"{type(exc).__name__}: {exc}",
                             'secondi': time.time() - t0})
                print(f"ERRORE: {exc}")
                traceback.print_exc()
            voce['memoria_picco_gb'] = memoria_gb()
            risultati.append(voce)
            json.dump(risultati, open(percorso, 'w', encoding='utf-8'),
                      indent=2, ensure_ascii=False)

    print(f"\nsalvato {percorso} ({len(risultati)} righe)")


if __name__ == '__main__':
    main()
