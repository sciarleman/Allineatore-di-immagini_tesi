import json, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
R = {}
for l in open("impostazioni/risultati.jsonl", encoding="utf-8"):
    r = json.loads(l); R[r['id']] = r
def fin(r): return {(c[0], c[1]): c[2] for c in r.get('finestre', []) if c[3] >= 0.05}
ordine = sorted(R.values(), key=lambda r: (r['insieme'], r['banda'], r['gruppo'] != 'precisione', r['gruppo'], r['factor'], r['refine_block'], r['min_conf']))
ultimo = None
for r in ordine:
    k = (r['insieme'], r['banda'])
    if k != ultimo:
        print(f"\n=== {r['insieme']} {r['banda']} ==="); ultimo = k
    et = f"f{r['factor']} blocchi {r['refine_block']} conf {r['min_conf']:.2f} {'campo' if r['nonrigid'] else 'SOLO AFFINE'}"
    if 'errore' in r:
        print(f"{r['gruppo']:<11} {et:<40} ERRORE: {r['errore']}  ({r['t_totale_s']:.0f} s)"); continue
    d = r['dopo']; base = R.get(f"{r['insieme']}_{r['banda']}_f2_b128_c10_nr")
    conf = ""
    if base and base is not r and 'finestre' in base:
        a, b = fin(base), fin(r); com = sorted(set(a) & set(b))
        if com:
            va = np.array([a[k] for k in com]); vb = np.array([b[k] for k in com])
            conf = f" | vs predef. su {len(com)}: {np.median(va):.3f}->{np.median(vb):.3f}, peggio {(vb > va + 0.2).sum()} meglio {(vb < va - 0.2).sum()}"
    s = r['stima']; cg = s.get('campo_grossolano') or {}; rip = r.get('ripiego') or {}
    cam = r.get('campo') or {}
    b0 = r['bordo'][0]; b1 = r['bordo'][1]
    print(f"{r['gruppo']:<11} {et:<40} {d['residuo_mediano_px']:6.3f} [{d['residuo_mediano_ic90'][0]:.2f}-{d['residuo_mediano_ic90'][1]:.2f}] "
          f"{d['n_affidabili']:3d}/{d['n_campioni']:<3d} >1 {100*d['quota_sopra_1px']:3.0f}% >5 {100*d['quota_sopra_5px']:3.0f}% max {d['residuo_max_px']:6.1f} | "
          f"{r['t_totale_s']:4.0f} s (stima {r.get('t_stima_log')}) {r['picco_memoria_gb']:.1f} GB | ripiego {'si' if rip.get('usato') else ('provato' if rip else 'no')} | "
          f"gross {'si' if cg.get('applicato') else 'no'} {'' if cg.get('ampiezza_max_px_full') is None else round(cg['ampiezza_max_px_full'])} | "
          f"campo rms {cam.get('rms_px', 0):.1f} max {cam.get('max_px', 0):.0f} passo {cam.get('passo_px')} | bordo<500 {b0['n_affidabili']}/{b0['n_misurate']} med {b0['mediana_px'] if b0['mediana_px'] is None else round(b0['mediana_px'],2)}; 500-1000 {b1['n_affidabili']}/{b1['n_misurate']} med {b1['mediana_px'] if b1['mediana_px'] is None else round(b1['mediana_px'],2)}{conf}")
