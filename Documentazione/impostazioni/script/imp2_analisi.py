"""Tabelle dello studio delle impostazioni avanzate: per ogni coppia, ogni
configurazione contro quella predefinita (blocchi 128, confidenza 0.10), sulle
stesse finestre.   python imp2_analisi.py > tabelle.txt"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
QUI = os.path.dirname(os.path.abspath(__file__))
righe = [json.loads(r) for r in open(os.path.join(QUI, "risultati.jsonl"), encoding="utf-8")]
ult = {}
for r in righe:
    ult[(r['insieme'], r['banda'], r['fattore'], r['refine_block'], r['min_conf'])] = r
NOMI = {(128, 0.10): "predefinita (128, 0.10)", (96, 0.10): "densita' fitta (96)", (192, 0.10): "densita' larga (192)",
        (128, 0.05): "confidenza bassa (0.05)", (128, 0.20): "confidenza alta (0.20)"}
PREC = {1: "massima", 2: "precisa", 4: "veloce"}


def zone(g, H, W):
    ok = g[:, 3] >= 0.05
    d = np.minimum.reduce([g[:, 0], g[:, 1], W - g[:, 0], H - g[:, 1]])
    ang = ((g[:, 0] < 4608) | (g[:, 0] > W - 4608)) & ((g[:, 1] < 3072) | (g[:, 1] > H - 3072))
    return ok, d, ang


def stat(v):
    if len(v) == 0:
        return "    -      -     -"
    return f"{np.median(v):5.2f} {100 * (v > 5).mean():5.0f}% {v.max():5.0f}"


riassunto = {}
for (ins, banda, fat) in sorted({k[:3] for k in ult}, key=lambda k: (k[2] != 2, k[0], k[1])):
    base = ult.get((ins, banda, fat, 128, 0.10))
    print("=" * 150)
    print(f"{ins} {banda}, precisione {PREC.get(fat, fat)}")
    print(f"{'configurazione':<26}{'verifica programma':^24}{'griglia fissa':^26}{'fitta: tutte':^20}{'bordo 0-1500':^20}{'angoli':^20}"
          f"{'campo fine':^22}{'tempo':>6}{'RAM':>6}")
    print(f"{'':<26}{'med.  n   >5px':^24}{'med.  n   >10px(gruppo)':^26}{'med.  >5px   max':^20}{'med.  >5px   max':^20}{'med.  >5px   max':^20}"
          f"{'sigma  blocchi':^22}{'s':>6}{'GB':>6}")
    for cfg, nome in NOMI.items():
        r = ult.get((ins, banda, fat) + cfg)
        if r is None:
            continue
        if r.get('errore'):
            print(f"{nome:<26}ERRORE: {r['errore']}"); continue
        g = np.array(r['finestre']); H, W = r['HW']
        ok, d, ang = zone(g, H, W)
        v, f = r['verifica'], r['fissa']
        cf = r['campo_fine']
        print(f"{nome:<26}"
              f"{v['residuo_mediano_px']:6.3f} {v['n_affidabili']:>3}/{v['n_campioni']:<3} {100 * (v['quota_sopra_5px'] or 0):3.0f}%   "
              f"{f['residuo_mediano_px']:6.3f} {f['n_affidabili']:>3}/{f['n_campioni']:<3} {f['n_oltre_10px']:>3} ({f['n_in_gruppo']})    "
              f"{stat(g[ok, 2]):^20}{stat(g[ok & (d <= 1500), 2]):^20}{stat(g[ok & ang, 2]):^20}"
              f"{(cf.get('smooth_px_full') or 0):6.0f} {(cf.get('blocchi') or 0):>7}        "
              f"{r['t_align_s']:6.0f}{r['picco_memoria_gb']:6.1f}")
    if base is None or base.get('errore'):
        continue
    gb = np.array(base['finestre']); H, W = base['HW']
    db = {(x, y): (e, c) for x, y, e, c in gb}
    print("  rispetto alla predefinita, sulle finestre misurabili in entrambe (migliorata/peggiorata = oltre 0.5 px di differenza):")
    for cfg, nome in NOMI.items():
        r = ult.get((ins, banda, fat) + cfg)
        if r is None or r.get('errore') or cfg == (128, 0.10):
            continue
        g = np.array(r['finestre'])
        dc = {(x, y): (e, c) for x, y, e, c in g}
        com = [k for k in db if k in dc and db[k][1] >= 0.05 and dc[k][1] >= 0.05]
        e1 = np.array([db[k][0] for k in com]); e2 = np.array([dc[k][0] for k in com])
        xy = np.array(com)
        dist = np.minimum.reduce([xy[:, 0], xy[:, 1], W - xy[:, 0], H - xy[:, 1]])
        bordo = dist <= 1500
        solo_b = sum(1 for k in db if db[k][1] >= 0.05 and not (k in dc and dc[k][1] >= 0.05))
        solo_c = sum(1 for k in dc if dc[k][1] >= 0.05 and not (k in db and db[k][1] >= 0.05))
        print(f"    {nome:<26} {len(com):>4} finestre: mediana {np.median(e1):.3f} -> {np.median(e2):.3f} px; "
              f"migliorate {int((e1 - e2 > 0.5).sum()):>3}, peggiorate {int((e2 - e1 > 0.5).sum()):>3} "
              f"(al bordo {int((e1 - e2 > 0.5)[bordo].sum())}/{int((e2 - e1 > 0.5)[bordo].sum())}); "
              f"differenza mediana in valore assoluto {np.median(np.abs(e2 - e1)):.3f} px, la piu' grande {np.abs(e2 - e1).max():.1f} px; "
              f"misurabili solo con la predefinita {solo_b}, solo con questa {solo_c}")
        riassunto.setdefault(cfg, []).append({
            'coppia': f"{ins} {banda} {PREC.get(fat, fat)}", 'd_mediana': float(np.median(e2) - np.median(e1)),
            'migliorate': int((e1 - e2 > 0.5).sum()), 'peggiorate': int((e2 - e1 > 0.5).sum()), 'n': len(com),
            'dt': r['t_align_s'] - base['t_align_s'], 'dram': r['picco_memoria_gb'] - base['picco_memoria_gb'],
            'd_ver': r['verifica']['residuo_mediano_px'] - base['verifica']['residuo_mediano_px']})

print("\n" + "=" * 150)
print("IN SINTESI: ogni impostazione contro la predefinita, su tutte le coppie")
for cfg, nome in NOMI.items():
    if cfg not in riassunto:
        continue
    v = riassunto[cfg]
    dm = np.array([x['d_mediana'] for x in v])
    print(f"\n{nome}: {len(v)} coppie")
    print(f"   mediana sulla griglia fitta: cambia fra {dm.min():+.3f} e {dm.max():+.3f} px (mediana delle coppie {np.median(dm):+.3f})")
    print(f"   finestre migliorate di oltre 0.5 px: {sum(x['migliorate'] for x in v)}; peggiorate: {sum(x['peggiorate'] for x in v)}; su {sum(x['n'] for x in v)}")
    print(f"   tempo di stima: fra {min(x['dt'] for x in v):+.0f} e {max(x['dt'] for x in v):+.0f} s; memoria: fra {min(x['dram'] for x in v):+.1f} e {max(x['dram'] for x in v):+.1f} GB")
    for x in v:
        print(f"      {x['coppia']:<26} mediana {x['d_mediana']:+.3f} px (verifica del programma {x['d_ver']:+.3f}), migliorate {x['migliorate']:>3}, peggiorate {x['peggiorate']:>3} su {x['n']}")
