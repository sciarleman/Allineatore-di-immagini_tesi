"""Tabelle del confronto generale, da generale/risultati.jsonl."""
import os, json, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
QUI = os.path.dirname(os.path.abspath(__file__))
rec = {}
for riga in open(os.path.join(QUI, "generale", "risultati.jsonl"), encoding="utf-8"):
    r = json.loads(riga)
    rec[(r['insieme'], r['banda'], r['fattore'], r['riduzione'], r['variante'])] = r   # l'ultima vince


def med(r):
    return (r.get('verifica') or {}).get('residuo_mediano_px')


def fallita(r):
    """La stessa regola di align_paintings._posa_da_rifare."""
    v = r.get('verifica') or {}
    return r.get('saltata') is not None or 'errore' in r or v.get('residuo_mediano_px') is None \
        or not v.get('n_affidabili') or v['residuo_mediano_px'] > 10.0


def riga(v):
    if v is None or v.get('residuo_mediano_px') is None:
        return "non misurabile"
    ic = v['residuo_mediano_ic90']
    return (f"{v['residuo_mediano_px']:6.3f} px [{ic[0]:.2f}-{ic[1]:.2f}] {v['n_affidabili']:3d}/{v['n_campioni']:<3d} "
            f">1 {100 * v['quota_sopra_1px']:3.0f}% >5 {100 * v['quota_sopra_5px']:3.0f}% max {v['residuo_max_px']:6.1f}")


casi = sorted({k[:4] for k in rec})
for (ins, banda, f, rid) in casi:
    g = lambda var: rec.get((ins, banda, f, rid, var))
    A, B, C = g("A1"), g("B1"), g("C")
    if C is None or 'verifica' not in C:
        continue
    base = A or C
    tC = C.get('t_stima_s')
    if tC is None:                      # CK f1 rid 2: la stima e' quella della run, il tempo quello della gemella
        gem = rec.get((ins, banda, f, 4, "C")); tC = gem.get('t_stima_s') if gem else float('nan')
    tv = C.get('t_verifica_s') or 0.0
    tp = C['t_posa_s']
    print(f"\n{ins} {banda}  fattore {f}  portata grossolano {64 * rid * f // 2 if False else int(32 * rid * f)} px (riduzione {rid})"
          f"   [anteprime {C['t_anteprime_s']:.0f} s, traslazione oggi {C['traslazione_oggi']} corretta {C['traslazione_corretta']}]")
    scen = []
    for et, X in (("oggi: traslazione, poi ripiego se serve", A), ("traslazione corretta, poi ripiego se serve", B if B else A)):
        if X is None:
            continue
        stessa = (et.startswith("traslazione corretta") and B is None)
        tX = X.get('t_stima_s')
        if tX is None:
            tX = (B or C).get('t_stima_s') or tC          # A1 saltata: costa come una stima
        tXv = X.get('t_verifica_s') or tv
        if fallita(X):
            fin, t, nota = C, tX + tXv + tp + tC + tv, f"primo tentativo fallito ({'-' if med(X) is None else f'{med(X):.0f} px'}) -> ripiego"
            if med(X) is not None and med(C) is not None and med(X) < med(C):
                fin, nota = X, nota + " (ma si tiene il primo)"
        else:
            fin, t, nota = X, tX + tXv, "primo tentativo riuscito, niente ripiego"
        if stessa:
            nota += "  [identico a oggi: stesse dimensioni]"
        scen.append((et, fin, t, nota))
    scen.append(("posa grossolana subito", C, tp + tC + tv, ""))
    for et, fin, t, nota in scen:
        print(f"  {et:<44} {riga(fin.get('verifica')):<70} {t / 60:5.1f} min  {nota}")
    # confronto sulle stesse finestre (griglia fissa) fra i finali
    def camp(r):
        return {(c[0], c[1]): c[2] for c in (r.get('verifica_griglia_fissa') or {}).get('campioni', []) if c[3] >= 0.05}
    rif = camp(scen[0][1])
    for et, fin, t, nota in scen[1:]:
        c = camp(fin); com = sorted(set(rif) & set(c))
        if com and fin is not scen[0][1]:
            a = np.array([rif[k] for k in com]); b = np.array([c[k] for k in com])
            print(f"      rispetto a oggi su {len(com)} finestre comuni: mediana {np.median(a):.3f} -> {np.median(b):.3f} px, "
                  f"peggiorate >0.2 px: {(b > a + 0.2).sum()}, migliorate: {(b < a - 0.2).sum()}")
    cg = (C.get('stima') or {}).get('campo_grossolano') or {}
    print(f"      (posa) grossolano applicato {cg.get('applicato')}, ampiezza max {cg.get('ampiezza_max_px_full')}; "
          f"bordo: " + " | ".join(f"{b_['da_px']}-{b_['a_px']}: {b_['n_affidabili']}/{b_['n_misurate']} med "
                                   f"{'-' if b_['mediana_px'] is None else round(b_['mediana_px'], 2)}" for b_ in C.get('bordo', [])))
