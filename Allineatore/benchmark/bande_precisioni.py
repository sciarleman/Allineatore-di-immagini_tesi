"""
Le bande fotografiche alle due precisioni: sola affine, completa a fattore 2
(consigliata) e completa a fattore 1 (massima), su tutte le riprese delle due
opere. Sono i numeri della tabella delle bande nella tesi.

    python bande_precisioni.py "<cartella dei TIFF dell'affresco>"            tutte le corse
    python bande_precisioni.py "<cartella dei TIFF dell'affresco>" FG         solo un insieme
    python bande_precisioni.py "<cartella dei TIFF dell'affresco>" CK 2       un insieme e un fattore
    python bande_precisioni.py --tabella

Passa per align_paintings.align con gli argomenti del lotto (tre scale 'auto',
nessuna zona, solo stima e verifica) e registra la verifica del programma e
quella a griglia fissa. Ogni corsa e' un processo a se', cosi' la memoria di una
non pesa sulla successiva; una guardia la ferma se la memoria libera scende
sotto GUARDIA_GB (l'affresco a massima e' al limite su questo computer).

Scrive una riga per corsa in bande_precisioni.jsonl; le corse gia' fatte si
saltano.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)
if PROG not in sys.path:
    sys.path.insert(0, PROG)

RISULTATI = os.path.join(QUI, 'bande_precisioni.jsonl')
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
GUARDIA_GB = 0.6

# insieme -> (riferimento, {nome: (banda, file)})
INSIEMI = {
    'FG': ('VIS_FraGalgario_TOT.tif', {
        'NIR850': ('NIR', 'NIR850_FraGalgario_TOT.tif'),
        'NIR950': ('NIR', 'NIR950_FraGalgario_TOT.tif'),
        'UVF': ('UV', 'UVF_FraGalgario_TOT.tif'),
        'UVR': ('UV', 'UVR_FraGalgario_TOT.tif')}),
    'CK': ('VIS.tif', {
        'NIR': ('NIR', 'NIR.tif'),
        'UV': ('UV', 'UV.tif')}),
}
# nome, fattore, campo non rigido
CONFIG = [('sola affine', 2, False), ('completa', 2, True), ('completa', 1, True)]


def _memoria_libera_gb():
    """Memoria fisica libera (Windows), senza dipendenze esterne."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in
                    ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                     "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024 ** 3


def _guardia():
    while True:
        if _memoria_libera_gb() < GUARDIA_GB:
            print(f"GUARDIA: memoria libera sotto {GUARDIA_GB} GB, esco", flush=True)
            os._exit(3)
        time.sleep(0.5)


def una_corsa(cartella, ins, nome, config, fattore, nonrigid):
    import align_paintings as ap
    import verifica_fissa as vf
    threading.Thread(target=_guardia, daemon=True).start()
    rif, immagini = INSIEMI[ins]
    banda, file = immagini[nome]
    fixed, mov = os.path.join(cartella, rif), os.path.join(cartella, file)
    rec = {'insieme': ins, 'immagine': nome, 'banda': banda, 'configurazione': config,
           'fattore': fattore}
    tmp = tempfile.mkdtemp(prefix='bande_prec_')
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            T, rep = ap.align(fixed, mov, None, params_path=os.path.join(tmp, 'p.json'),
                              factor=fattore, nonrigid=nonrigid,
                              diagnostics_dir=os.path.join(tmp, 'diag'),
                              tre_scale='auto', verbose=False)
        rec['secondi'] = time.time() - t0
        chiavi = ('residuo_mediano_px', 'residuo_mediano_ic90', 'residuo_p95_px', 'residuo_max_px',
                  'n_affidabili', 'n_campioni', 'quota_sopra_2px', 'quota_sopra_5px')
        rec['prima'] = {k: rep['verifica']['prima'].get(k) for k in chiavi}
        rec['verifica'] = {k: rep['verifica']['dopo'].get(k) for k in chiavi}
        s = rep['stima']
        liv = s.get('campo_grossolano_livelli') or [s.get('campo_grossolano') or {}]
        rec['livelli'] = [{k: l.get(k) for k in ('riduzione', 'applicato', 'ampiezza_max_px_full')}
                          for l in liv]
        mem = ap.peak_memory_gb()
        rec['picco_memoria_gb'] = None if mem is None else mem[0]
        with contextlib.redirect_stdout(buf):
            fissa = vf.verifica_griglia_fissa(fixed, mov, T, None, rep, os.path.join(tmp, 'diag'))
        rec['fissa'] = {k: fissa.get(k) for k in ('residuo_mediano_px', 'n_affidabili', 'n_campioni',
                                                  'quota_sopra_5px', 'quota_sopra_10px')}
        import image_io
        info = image_io.info(fixed)
        rec['megapixel'] = info.height * info.width / 1e6
    except Exception as exc:
        rec['errore'] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        rec['secondi'] = time.time() - t0
    shutil.rmtree(tmp, ignore_errors=True)
    with open(RISULTATI, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    v = rec.get('verifica') or {}
    print(f"{ins} {nome}, {config}, fattore {fattore}: " + (rec.get('errore') or (
        f"{v['residuo_mediano_px']:.3f} px ({v['n_affidabili']}/{v['n_campioni']}), griglia fissa "
        f"{rec['fissa']['residuo_mediano_px']:.3f} px; {rec['secondi']:.0f} s, "
        f"{rec['picco_memoria_gb']:.1f} GB")), flush=True)


def _leggi():
    if not os.path.exists(RISULTATI):
        return []
    with open(RISULTATI, encoding='utf-8') as fh:
        return [json.loads(r) for r in fh if r.strip()]


def tabella():
    import numpy as np
    corse = [r for r in _leggi() if 'errore' not in r]
    print(f"{'immagine':12s}{'configurazione':16s}{'fatt.':>6s}{'prima':>9s}{'residuo':>9s}"
          f"{'IC 90%':>16s}{'>5px':>6s}{'fissa':>8s}{'s':>7s}{'s/Mpx':>7s}{'GB':>6s}{'byte/px':>8s}")
    for r in corse:
        v, ic = r['verifica'], r['verifica'].get('residuo_mediano_ic90') or [float('nan')] * 2
        gb = r.get('picco_memoria_gb') or float('nan')
        print(f"{r['insieme'] + ' ' + r['immagine']:12s}{r['configurazione']:16s}{r['fattore']:6d}"
              f"{r['prima']['residuo_mediano_px'] or float('nan'):9.2f}{v['residuo_mediano_px']:9.3f}"
              f"{f'[{ic[0]:.2f}-{ic[1]:.2f}]':>16s}{100 * (v['quota_sopra_5px'] or 0):5.0f}%"
              f"{r['fissa']['residuo_mediano_px']:8.3f}{r['secondi']:7.0f}"
              f"{r['secondi'] / r['megapixel']:7.2f}{gb:6.1f}{gb * 1024 ** 3 / (r['megapixel'] * 1e6):8.1f}")
    print("\nper banda: media sulle immagini (minimo-massimo), verifica del programma")
    for banda in ('NIR', 'UV'):
        for config, fattore, _ in CONFIG:
            x = np.array([r['verifica']['residuo_mediano_px'] for r in corse
                          if (r['banda'], r['configurazione'], r['fattore']) == (banda, config, fattore)])
            if len(x):
                print(f"  {banda:4s}{config:14s}fattore {fattore}: {x.mean():6.2f} px "
                      f"({x.min():.2f}-{x.max():.2f}), {len(x)} immagini")


def main():
    if sys.argv[1:2] == ['--tabella']:
        return tabella()
    if sys.argv[1:2] == ['--una']:
        cartella, ins, nome, config, fattore, nonrigid = sys.argv[2:8]
        return una_corsa(cartella, ins, nome, config, int(fattore), nonrigid == '1')
    ck = sys.argv[1]
    filtro = sys.argv[2:4]
    fatti = {(r['insieme'], r['immagine'], r['configurazione'], r['fattore']) for r in _leggi()}
    for ins, (_, immagini) in INSIEMI.items():
        for config, fattore, nonrigid in CONFIG:
            if filtro and (ins != filtro[0] or (len(filtro) > 1 and str(fattore) != filtro[1])):
                continue
            for nome in immagini:
                if (ins, nome, config, fattore) in fatti:
                    print(f"[gia' fatta] {ins} {nome}, {config}, fattore {fattore}")
                    continue
                esito = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--una',
                                        FG if ins == 'FG' else ck, ins, nome, config, str(fattore),
                                        '1' if nonrigid else '0']).returncode
                if esito:
                    print(f"  {ins} {nome}, {config}, fattore {fattore}: processo uscito con {esito}",
                          flush=True)
    tabella()


if __name__ == '__main__':
    main()
