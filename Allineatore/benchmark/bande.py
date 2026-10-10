"""
Le tre scale su tutte le bande: un solo algoritmo per tutto?

Tre configurazioni: le due passate di sempre, le tre scale, e l'algoritmo
unico (tre_scale='auto'): il campo grossolano si stima sempre ma si applica
solo se supera la portata della passata fine.

Per ogni banda del Fra Galgario si allinea sul VIS nelle condizioni del lotto
(fattore 2, campo non rigido, nessuna maschera nella stima), una volta con le
due passate di sempre e una con le tre scale (block_matching.campo_grossolano
fra le due passate). Per la RX TestaChiara si prova anche la configurazione
predefinita della sezione RX (tre scale + zona del dipinto; niente fascia di
bordo, perche' sulla TestaChiara il bordo della tela non e' tracciato, e
niente TRE, perche' non ci sono punti cliccati).

Il giudizio e' uguale per tutte: la verifica a griglia fissa della sezione RX
(verifica_fissa.verifica_griglia_fissa), dentro la zona del dipinto del 27/9 18:45, cosi'
che le finestre siano le stesse per ogni banda e configurazione; il confronto
fra due configurazioni si fa sulle finestre misurate da entrambe.

Nessuna immagine viene scritta. Rilanciando, le corse gia' fatte si saltano.

    python bande.py
"""

import json
import os
import re
import sys
import time
import traceback

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)
if PROG not in sys.path:
    sys.path.insert(0, PROG)

import align_paintings                      # noqa: E402
import verifica_fissa                       # noqa: E402
import radiografie                          # noqa: E402

A = radiografie.A
BANDE = [
    ('UVF', 'UVF_FraGalgario_TOT.tif'),
    ('UVR', 'UVR_FraGalgario_TOT.tif'),
    ('NIR850', 'NIR850_FraGalgario_TOT.tif'),
    ('NIR950', 'NIR950_FraGalgario_TOT.tif'),
    ('SWIR', 'big_SWIR_FraGalgario_TOT.tif'),
    ('RX TestaChiara', 'RX_FraGalgarioTOT_TestaChiara.tif'),
    ('RX Cappellochiaro', 'RX_FraGalgario_TOT_Cappellochiaro.tif'),
]
CONFIG = [
    ('due passate (lotto di sempre)', dict(tre_scale=False)),
    ('tre scale', dict(tre_scale=True)),
    ('algoritmo unico (tre scale, grossolano solo oltre la portata fine)', dict(tre_scale='auto')),
]
SOLO_RX = [('sezione RX predefinita (tre scale + zona)', dict(tre_scale=True, _maschera=True))]

LAVORO = os.path.join(QUI, 'lavoro', 'bande')
RISULTATI = os.path.join(QUI, 'bande_risultati.json')


def corri(banda, file, nome, kw, maschera):
    k = dict(kw)
    usa_maschera = k.pop('_maschera', False)
    base = re.sub(r'[^\w +-]', '_', f"{banda} - {nome}")
    t0 = time.time()
    T, rep = align_paintings.align(
        radiografie.VIS, os.path.join(A, file), out_path=None,
        params_path=os.path.join(LAVORO, 'parametri', base + '.json'),
        diagnostics_dir=os.path.join(LAVORO, 'diagnostica', base),
        mask=maschera if usa_maschera else None, verbose=False, **k)
    secondi = time.time() - t0
    ver = verifica_fissa.verifica_griglia_fissa(
        radiografie.VIS, os.path.join(A, file), T, maschera, rep,
        os.path.join(LAVORO, 'verifiche', base))
    cg = rep['stima'].get('campo_grossolano') or {}
    return {
        'banda': banda, 'configurazione': nome, 'secondi': secondi,
        'mediana_px': ver.get('residuo_mediano_px'),
        'ic90': ver.get('residuo_mediano_ic90'),
        'quota_sopra_1px': ver.get('quota_sopra_1px'),
        'quota_sopra_2px': ver.get('quota_sopra_2px'),
        'quota_sopra_5px': ver.get('quota_sopra_5px'),
        'quota_sopra_10px': ver.get('quota_sopra_10px'),
        'misurabili': ver.get('n_affidabili'), 'totali': ver.get('n_campioni'),
        'distorsione_px': ver.get('distorsione_px'),
        'oltre_10': len(ver.get('oltre_10px') or []),
        'campo_grossolano': {k_: cg.get(k_) for k_ in ('applicato', 'guadagno_pct',
                                                        'ampiezza_max_px_full', 'motivo')
                             if k_ in cg},
        'ripiego': bool(rep.get('ripiego', {}).get('usato')) if isinstance(rep.get('ripiego'), dict) else None,
        'finestre': ver['finestre'],
    }


def main():
    os.makedirs(os.path.join(LAVORO, 'parametri'), exist_ok=True)
    d = radiografie.carica_dati()
    fatti = []
    if os.path.exists(RISULTATI):
        with open(RISULTATI, encoding='utf-8') as fh:
            fatti = [r for r in json.load(fh)['corse'] if 'errore' not in r]
    chiavi = {(r['banda'], r['configurazione']) for r in fatti}
    for banda, file in BANDE:
        for nome, kw in CONFIG + (SOLO_RX if banda == 'RX TestaChiara' else []):
            if (banda, nome) in chiavi:
                print(f"[gia' fatta] {banda} - {nome}")
                continue
            print(f"\n=== {banda} - {nome} ===", flush=True)
            try:
                r = corri(banda, file, nome, kw, d['maschera'])
                print(f"  {r['secondi'] / 60:.1f} min; mediana {r['mediana_px']:.2f} px su "
                      f"{r['misurabili']}/{r['totali']}; oltre 5 px "
                      f"{100 * (r['quota_sopra_5px'] or 0):.0f}%; campo grossolano "
                      f"{r['campo_grossolano']}", flush=True)
            except Exception as exc:
                r = {'banda': banda, 'configurazione': nome,
                     'errore': f"{type(exc).__name__}: {exc}"}
                print(f"  ERRORE {r['errore']}")
                traceback.print_exc()
            fatti.append(r)
            with open(RISULTATI, 'w', encoding='utf-8') as fh:
                json.dump({'riferimento': 'VIS_FraGalgario_TOT.tif', 'corse': fatti}, fh,
                          indent=1, ensure_ascii=False)


def tabella():
    """La tabella per il README, con il confronto sulle finestre comuni."""
    with open(RISULTATI, encoding='utf-8') as fh:
        corse = json.load(fh)['corse']
    per = {}
    for r in corse:
        per.setdefault(r['banda'], {})[r['configurazione']] = r
    print("| banda | configurazione | mediana | oltre 5 px | oltre 10 px | finestre | "
          "distorsione | campo grossolano | sulle finestre comuni | minuti |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for banda, conf in per.items():
        base = conf.get(CONFIG[0][0])
        for nome, r in conf.items():
            if 'errore' in r:
                print(f"| {banda} | {nome} | errore: {r['errore'][:50]} ||||||||")
                continue
            comuni = '—'
            if base and r is not base and 'finestre' in base:
                a = {(f[0], f[1]): f[2] for f in base['finestre']}
                b = {(f[0], f[1]): f[2] for f in r['finestre']}
                k = sorted(set(a) & set(b))
                if k:
                    va, vb = np.array([a[x] for x in k]), np.array([b[x] for x in k])
                    comuni = (f"{np.median(va):.2f} → {np.median(vb):.2f} px su {len(k)} "
                              f"({(vb < va - 0.2).sum()} meglio, {(vb > va + 0.2).sum()} peggio)")
            cg = r.get('campo_grossolano') or {}
            cg_s = ('—' if not cg else
                    f"applicato, +{cg.get('guadagno_pct', 0):.0f}%" if cg.get('applicato') else
                    "non applicato")
            dist = r.get('distorsione_px') or [0, 0]
            print(f"| {banda} | {nome} | {r['mediana_px']:.2f} | {100 * r['quota_sopra_5px']:.0f}% "
                  f"| {100 * r['quota_sopra_10px']:.0f}% | {r['misurabili']}/{r['totali']} "
                  f"| ({dist[0]:+.2f}, {dist[1]:+.2f}) | {cg_s} | {comuni} | {r['secondi'] / 60:.1f} |")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'tabella':
        tabella()
    else:
        main()
