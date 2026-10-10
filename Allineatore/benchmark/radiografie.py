"""
Benchmark delle radiografie: VIS / RX Cappellochiaro, senza scrivere immagini.

Mette a confronto le configurazioni del nostro allineatore (dall'automatico
del lotto fino a "tutto": tre scale, zona, punti, fascia di bordo), i metodi
classici a punti di controllo (affine e thin-plate spline dai soli punti) e i
metodi concorrenti del benchmark generale (metodi.py).

Tutti sono giudicati allo stesso modo, con tre criteri indipendenti fra loro:

1. la verifica a griglia fissa (verifica_fissa.verifica_griglia_fissa),
   sempre con la stessa zona del dipinto, cosi' che le finestre siano le
   stesse per tutti: quante sono misurabili e quanto sbagliano;
2. i dettagli cliccati a mano (rx_giudici.json): i 115 interni lontani dai 17
   punti usati come vincolo, che nessuna configurazione vede, e gli 82 sul
   bordo della tela;
3. la distanza fra le due linee del bordo: il contorno della zona sul VIS e il
   bordo della tela tracciato sulla RX.

Accanto, per vedere come si comportano, le altre misure che il programma sa
dare: il costo MIND, la distorsione media come vettore, la copertura (quante
finestre si misurano) e, dove ci sono punti come vincolo, il TRE sui 17 punti
lasciandone fuori uno alla volta.

Le trasformazioni si salvano in lavoro/RX benchmark/trasformazioni, e gli
errori punto per punto e finestra per finestra in rx_risultati.json: i grafici
(grafici_rx.py) si rifanno senza rifare le corse.

Ogni configurazione si salva appena finita in rx_risultati.json: rilanciando, le
configurazioni gia' fatte si saltano (per rifarle, toglierle dal file).

    python radiografie.py
"""

import json
import os
import re
import sys
import time
import traceback

import numpy as np
import cv2

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUI = os.path.dirname(os.path.abspath(__file__))
if PROG not in sys.path:
    sys.path.insert(0, PROG)

import align_paintings                      # noqa: E402
from datetime import datetime               # noqa: E402
import verifica_fissa                       # noqa: E402
import image_io                             # noqa: E402
import metodi                               # noqa: E402
import punti_manuali                        # noqa: E402
import validation as val                    # noqa: E402
from transform_model import Transform       # noqa: E402

# Le riprese: dati di ricerca, fuori dal repository.
A = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
VIS = os.path.join(A, "VIS_FraGalgario_TOT.tif")
RX = os.path.join(A, "RX_FraGalgario_TOT_Cappellochiaro.tif")
PAR27 = os.path.join(A, "27-09-Tutto", "parametri")
CORSA = "RX_FraGalgario_TOT_Cappellochiaro (2026-09-27 18-45)"
MASCHERA = os.path.join(PAR27, CORSA + "_maschera.png")
CONTORNO = os.path.join(PAR27, CORSA + "_contorno.json")

LAVORO = os.path.join(QUI, 'lavoro', 'RX benchmark')
RISULTATI = os.path.join(QUI, 'rx_risultati.json')


def carica_dati():
    info = image_io.info(VIS)
    H, W = info.height, info.width
    maschera = cv2.imread(MASCHERA, cv2.IMREAD_GRAYSCALE)
    with open(CONTORNO, encoding='utf-8-sig') as fh:
        C = np.asarray(json.load(fh)['contorno'], float).reshape(-1, 2)
    with open(os.path.join(QUI, 'rx_giudici.json'), encoding='utf-8') as fh:
        g = json.load(fh)
    R = np.array([p[:4] for p in g['punti']], float)
    d_bordo = np.array([p[4] for p in g['punti']], float)
    lontani = np.array([p[6] for p in g['punti']], bool)
    P17 = np.asarray(g['punti_vincolo_17'], float)
    return dict(H=H, W=W, maschera=maschera, C=C, R=R, P17=P17,
                interni=(d_bordo >= 250) & lontani, bordo=d_bordo < 100)


# ---------------------------------------------------------------------------
# le configurazioni
# ---------------------------------------------------------------------------

def _una_corsa(punti=None, maschera=None, contorno_moving=None, factor=2, nonrigid=True,
               min_conf=0.10, refine_block=128, n_maniglie=0, tre_scale=True,
               punti_come_posa=True, punti_come_vincolo=False, posa_dal_bordo=False,
               fascia_di_bordo=0):
    """Stima e verifica di una radiografia con gli strumenti a mano, senza
    scrivere l'immagine. Era finestra_rx.allinea_rx, con gli stessi valori
    predefiniti: la sezione a parte non c'e' piu' (gli strumenti si assegnano
    alle immagini del lotto), ma queste corse restano ripetibili."""
    base = f"{os.path.splitext(os.path.basename(RX))[0]} ({datetime.now().strftime('%Y-%m-%d %H-%M-%S')})"
    params_path = os.path.join(LAVORO, 'parametri', base + '.json')
    diagnostica = os.path.join(LAVORO, 'diagnostica', base)
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    if punti is not None and len(punti) < 3:
        punti = None
    T, rep = align_paintings.align(
        VIS, RX, None, params_path=params_path, factor=factor, nonrigid=nonrigid,
        min_conf=min_conf, refine_block=refine_block, refine_step=max(8, refine_block // 2),
        diagnostics_dir=diagnostica, punti=punti, mask=maschera,
        contorno_moving=contorno_moving, n_maniglie=n_maniglie, tre_scale=tre_scale,
        punti_come_posa=punti_come_posa, punti_come_vincolo=punti_come_vincolo,
        posa_dal_bordo=posa_dal_bordo, fascia_di_bordo=fascia_di_bordo, verbose=False)
    ver = verifica_fissa.verifica_griglia_fissa(VIS, RX, T, maschera, rep, diagnostica)
    return {'parametri': params_path, 'verifica_fissa': ver,
            'tre_punti': (rep.get('correzione_punti') or {}).get('tre')}


def nostro(**kw):
    """Una corsa con gli strumenti a mano, senza scrivere l'immagine."""
    def corri(d):
        k = dict(kw)
        uso_maschera = k.pop('_maschera', False)
        uso_punti = k.pop('_punti', False)
        uso_contorno = k.get('fascia_di_bordo') or k.get('posa_dal_bordo')
        e = _una_corsa(
            punti=d['P17'] if uso_punti else None,
            maschera=d['maschera'] if uso_maschera else None,
            contorno_moving=d['C'] if uso_contorno else None, **k)
        T = Transform.load(e['parametri'])
        info = {'parametri': os.path.basename(e['parametri'])}
        if e.get('tre_punti'):
            info['tre_17'] = {k: v for k, v in e['tre_punti'].items() if k != 'per_punto_px'}
        return T, info, (
            e['verifica_fissa'] if uso_maschera else None)
    return corri


def solo_punti_affine(d):
    """Il metodo classico: un'affine ai minimi quadrati sui punti di controllo."""
    ctrl = punti_manuali.controlla_punti(d['P17'])
    return Transform(ctrl['M'], ref_shape=(d['H'], d['W'])), {'modello': ctrl['modello']}, None


def solo_punti_tps(d, passo=128):
    """Thin-plate spline attraverso i punti: il non rigido classico a punti."""
    from scipy.interpolate import RBFInterpolator
    P = d['P17']
    ctrl = punti_manuali.controlla_punti(P)
    M = ctrl['M']
    tps = RBFInterpolator(P[:, :2], P[:, 2:], kernel='thin_plate_spline', degree=1)
    gh, gw = int(np.ceil(d['H'] / passo)) + 1, int(np.ceil(d['W'] / passo)) + 1
    yy, xx = np.mgrid[0:gh, 0:gw].astype(float) * passo
    nodi = np.column_stack([xx.ravel(), yy.ravel()])
    campo = tps(nodi) - (nodi @ M[:, :2].T + M[:, 2])
    return (Transform(M, campo.reshape(gh, gw, 2).astype(np.float32), passo, (d['H'], d['W'])),
            {}, None)


def solo_bordo_affine(d):
    """L'affine che sovrappone le sole due linee del bordo, senza guardare l'interno."""
    G = punti_manuali.contorno_da_maschera(d['maschera'], (d['H'], d['W']))
    M0 = punti_manuali.controlla_punti(d['P17'])['M']
    M, info = punti_manuali.posa_dai_contorni(G, d['C'], M0)
    return Transform(M, ref_shape=(d['H'], d['W'])), info, None


def concorrente(funzione):
    def corri(d):
        T, info = funzione(VIS, RX, cartella_diag=os.path.join(LAVORO, funzione.__name__))
        if T is None:
            raise RuntimeError(info.get('errore', 'nessuna trasformazione'))
        return T, {k: v for k, v in info.items() if isinstance(v, (int, float, str))}, None
    return corri


CONFIGURAZIONI = [
    # nome, gruppo, funzione
    ('automatico del lotto', 'nostro', nostro(tre_scale=False)),
    ('tre scale', 'nostro', nostro(tre_scale=True)),
    ('tre scale + zona', 'nostro', nostro(tre_scale=True, _maschera=True)),
    ('tre scale + zona + punti', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=False,
            punti_come_vincolo=True)),
    ('tutto (+ fascia di bordo)', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=False,
            punti_come_vincolo=True, fascia_di_bordo=150)),
    ('predefinita: zona + fascia, punti solo per verifica', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=False,
            punti_come_vincolo=False, fascia_di_bordo=150)),
    ('tutto, punti anche come posa', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=True,
            punti_come_vincolo=True, fascia_di_bordo=150)),
    ('tutto, posa dal bordo', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=False,
            punti_come_vincolo=True, fascia_di_bordo=150, posa_dal_bordo=True)),
    ('tutto, a fattore 4', 'nostro',
     nostro(tre_scale=True, _maschera=True, _punti=True, punti_come_posa=False,
            punti_come_vincolo=True, fascia_di_bordo=150, factor=4)),
    ('solo punti: affine', 'a punti', solo_punti_affine),
    ('solo punti: thin-plate spline', 'a punti', solo_punti_tps),
    ('solo bordo: affine sulle due linee', 'a punti', solo_bordo_affine),
    ('MIRA (Mattes MI, affine)', 'concorrenti', concorrente(metodi.mira)),
    ('SIFT + RANSAC', 'concorrenti', concorrente(metodi.sift_ransac)),
    ('ECC sui gradienti', 'concorrenti', concorrente(metodi.ecc)),
    ('NGF + Powell', 'concorrenti', concorrente(metodi.ngf_powell)),
    ('MIND + Powell', 'concorrenti', concorrente(metodi.mind_powell)),
    ('NMI + Powell', 'concorrenti', concorrente(metodi.nmi_powell)),
]


# ---------------------------------------------------------------------------
# la valutazione, identica per tutti
# ---------------------------------------------------------------------------

def _riassunto(v):
    """Media, mediana, coda e distorsione di un insieme di vettori d'errore."""
    v = np.asarray(v, float).reshape(-1, 2)
    if not len(v):
        return None
    m = np.hypot(v[:, 0], v[:, 1])
    return {'n': int(len(m)), 'media': float(m.mean()), 'mediana': float(np.median(m)),
            'p90': float(np.percentile(m, 90)), 'massimo': float(m.max()),
            'rms': float(np.sqrt((m ** 2).mean())),
            'quota_oltre_5': float((m > 5).mean()), 'quota_oltre_10': float((m > 10).mean()),
            # lo spostamento medio come vettore: un errore sistematico (tutto
            # spostato dalla stessa parte) si vede qui e non nel modulo
            'distorsione_x': float(v[:, 0].mean()), 'distorsione_y': float(v[:, 1].mean())}


def _ic90(x, statistica, n=2000):
    """Intervallo al 90% di una statistica, per ricampionamento."""
    x = np.asarray(x, float)
    if len(x) < 8:
        return None
    rng = np.random.default_rng(0)
    v = statistica(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


def _jacobiano_min(T):
    if T.field is None:
        return float(np.linalg.det(T.M[:, :2]))
    f, p = T.field.astype(float), float(T.grid_step)
    a = T.M[0, 0] + np.gradient(f[..., 0], p, axis=1)
    b = T.M[0, 1] + np.gradient(f[..., 0], p, axis=0)
    c = T.M[1, 0] + np.gradient(f[..., 1], p, axis=1)
    e = T.M[1, 1] + np.gradient(f[..., 1], p, axis=0)
    return float((a * e - b * c).min())


def valuta(T, d, ver=None, nome=''):
    if ver is None:
        ver = verifica_fissa.verifica_griglia_fissa(
            VIS, RX, T, d['maschera'], {},
            os.path.join(LAVORO, 'verifiche', re.sub(r'[^\w +-]', '_', nome)))
    fin = np.asarray([f[2] for f in ver['finestre']], float)
    arr_f = image_io.apri(VIS, verbose=False)[0]
    arr_m = image_io.apri(RX, verbose=False)[0]
    mind = val.mind_quality(arr_f, arr_m, T, verbose=False).get('mind_dopo')
    del arr_f, arr_m
    finestre = {'misurabili': int(ver['n_affidabili']), 'totali': int(ver['n_campioni'])}
    if len(fin):
        finestre.update({'media': float(fin.mean()), 'mediana': float(np.median(fin)),
                         'p90': float(np.percentile(fin, 90)),
                         'quota_oltre_5': float((fin > 5).mean()),
                         'oltre_10': int((fin > 10).sum()),
                         'distorsione_px': ver.get('distorsione_px'),
                         'residui': [round(float(x), 3) for x in fin],
                         'posizioni': [[int(f[0]), int(f[1])] for f in ver['finestre']]})
    R = d['R']
    err = R[:, 2:] - align_paintings._porta_punti(T, R[:, :2])
    # le due linee
    G = punti_manuali._ricampiona_chiuso(
        punti_manuali.contorno_da_maschera(d['maschera'], (d['H'], d['W']), passo_px=10.0), 10.0)
    n = punti_manuali.normali_esterne(G)
    Cd = punti_manuali._ricampiona_chiuso(d['C'], 5.0)
    from scipy.spatial import cKDTree
    PG = align_paintings._porta_punti(T, G)
    gap = ((Cd[cKDTree(Cd).query(PG)[1]] - PG) * n).sum(axis=1)
    m_int = np.hypot(*err[d['interni']].T)
    pi = _riassunto(err[d['interni']])
    pi['ic90_mediana'] = _ic90(m_int, np.median)
    pi['ic90_media'] = _ic90(m_int, np.mean)
    return {'finestre': finestre,
            'punti_interni': pi,
            'punti_bordo': _riassunto(err[d['bordo']]),
            # gli errori uno per uno, nell'ordine di rx_giudici.json
            'errori_punti': [round(float(x), 2) for x in np.hypot(*err.T)],
            'mind': mind,
            'linee': {'mediana_con_segno': float(np.median(gap)),
                      'mediana_assoluta': float(np.median(np.abs(gap))),
                      'p90_assoluto': float(np.percentile(np.abs(gap), 90))},
            'jacobiano_min': _jacobiano_min(T)}


def main():
    os.makedirs(LAVORO, exist_ok=True)
    d = carica_dati()
    fatti = {}
    if os.path.exists(RISULTATI):
        with open(RISULTATI, encoding='utf-8') as fh:
            fatti = {r['nome']: r for r in json.load(fh)['configurazioni']}
    for nome, gruppo, corri in CONFIGURAZIONI:
        if nome in fatti and 'errore' not in fatti[nome]:
            print(f"[gia' fatta] {nome}")
            continue
        print(f"\n=== {nome} ===", flush=True)
        voce = {'nome': nome, 'gruppo': gruppo}
        t0 = time.time()
        try:
            T, info, ver = corri(d)
            voce['secondi'] = time.time() - t0
            cartella_T = os.path.join(LAVORO, 'trasformazioni')
            os.makedirs(cartella_T, exist_ok=True)
            T.save(os.path.join(cartella_T, re.sub(r'[^\w +-]', '_', nome) + '.json'))
            voce['info'] = info
            voce.update(valuta(T, d, ver, nome))
            pi, pb, f = voce['punti_interni'], voce['punti_bordo'], voce['finestre']
            print(f"  {voce['secondi'] / 60:.1f} min; interni {pi['mediana']:.1f} px "
                  f"(media {pi['media']:.1f}); bordo {pb['mediana']:.1f} px; linee "
                  f"{voce['linee']['mediana_assoluta']:.1f} px; finestre "
                  f"{f['misurabili']}/{f['totali']}"
                  + (f", mediana {f['mediana']:.2f} px" if 'mediana' in f else ""), flush=True)
        except Exception as exc:
            voce['errore'] = f"{type(exc).__name__}: {exc}"
            voce['secondi'] = time.time() - t0
            print(f"  ERRORE {voce['errore']}")
            traceback.print_exc()
        fatti[nome] = voce
        ordine = [c[0] for c in CONFIGURAZIONI]
        with open(RISULTATI, 'w', encoding='utf-8') as fh:
            json.dump({'coppia': 'VIS / RX Cappellochiaro',
                       'configurazioni': sorted(fatti.values(),
                                                key=lambda v: ordine.index(v['nome'])
                                                if v['nome'] in ordine else 99)},
                      fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
