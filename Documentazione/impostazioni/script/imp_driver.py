"""Lancia in fila le configurazioni dello studio delle impostazioni, una per
processo. Riprende da dove si era fermato (salta quelle gia' in risultati.jsonl).

    python imp_driver.py [FG] [CK]
"""
import os, sys, json, subprocess, time

QUI = os.path.dirname(os.path.abspath(__file__))
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
CK = os.path.join(QUI, "ck_tiff")          # i tre PNG del dipinto nuovo, portati in TIFF non compresso
IMG = {
    "FG": (os.path.join(FG, "VIS_FraGalgario_TOT.tif"), {
        "NIR850": "NIR850_FraGalgario_TOT.tif", "UVF": "UVF_FraGalgario_TOT.tif",
        "UVR": "UVR_FraGalgario_TOT.tif", "RX_Cappello": "RX_FraGalgario_TOT_Cappellochiaro.tif"}, FG),
    "CK": (os.path.join(CK, "VIS.tif"), {"NIR": "NIR.tif", "UV": "UV.tif"}, CK),
}
BASE = dict(factor=2, refine_block=128, min_conf=0.10, nonrigid=True)
SECONDARIE = {"FG": ("NIR850", "UVF"), "CK": ("NIR", "UV")}   # dove si provano anche le altre manopole

configurazioni = []
for ins in [a for a in sys.argv[1:] if a in IMG] or list(IMG):
    fixed, bande, cartella = IMG[ins]
    for banda, f in bande.items():
        def aggiungi(etichetta, **kw):
            c = dict(BASE, **kw)
            c.update(insieme=ins, banda=banda, fixed=fixed, moving=os.path.join(cartella, f),
                     gruppo=etichetta, id=f"{ins}_{banda}_f{c['factor']}_b{c['refine_block']}_c{int(100 * c['min_conf']):02d}"
                                         f"_{'nr' if c['nonrigid'] else 'aff'}")
            if c['id'] not in {x['id'] for x in configurazioni}:
                configurazioni.append(c)
        for fat in (2, 4, 8, 1):
            aggiungi("precisione", factor=fat)
        if banda in SECONDARIE[ins]:
            for rb in (96, 192):
                aggiungi("densita", refine_block=rb)
            for mc in (0.05, 0.20):
                aggiungi("confidenza", min_conf=mc)
            aggiungi("solo_affine", nonrigid=False)

fatti = set()
p = os.path.join(QUI, "impostazioni", "risultati.jsonl")
if os.path.exists(p):
    fatti = {json.loads(r)['id'] for r in open(p, encoding="utf-8")}
da_fare = [c for c in configurazioni if c['id'] not in fatti]
print(f"{len(configurazioni)} configurazioni, {len(da_fare)} da fare", flush=True)
for i, c in enumerate(da_fare):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", os.path.join(QUI, "imp_worker.py"), json.dumps(c)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ultima = (r.stdout.strip().splitlines() or ["(nessuna uscita)"])[-1]
    print(f"[{i + 1}/{len(da_fare)}] {time.strftime('%H:%M:%S')} {ultima}"
          + ("" if r.returncode == 0 else f"  !! exit {r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}"),
          flush=True)
print("STUDIO_FINITO", flush=True)
