import os, sys, json, subprocess
QUI = os.path.dirname(os.path.abspath(__file__)); I = os.path.join(QUI, "impostazioni"); S = os.path.join(QUI, "scritture"); os.makedirs(S, exist_ok=True)
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
R = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(I, "risultati.jsonl"), encoding="utf-8")}
casi = {"FG": ("FG_NIR850_f2_b128_c10_nr", os.path.join(FG, "NIR850_FraGalgario_TOT.tif"), 19850, 15848),
        "CK": ("CK_NIR_f2_b128_c10_nr", os.path.join(QUI, "ck_tiff", "NIR.tif"), 20428, 29049)}
lavori = []
for ins, (idr, mov, H, W) in casi.items():
    base = dict(azione='scrivi', moving=mov, H=H, W=W, M=os.path.join(I, idr + "_M.npy"), campo=os.path.join(I, idr + "_campo.npy"),
                passo=R[idr]['campo']['passo_px'], insieme=ins)
    for tile in ((1024, 2048, 4096) if ins == "FG" else (2048,)):
        lavori.append(dict(base, id=f"{ins}_tif_tile{tile}", tile=tile, out=os.path.join(S, f"{ins}_t{tile}.tif")))
    lavori.append(dict(base, id=f"{ins}_png_diretto", tile=2048, out=os.path.join(S, f"{ins}_diretto.png")))
    lavori.append(dict(base, id=f"{ins}_jpg_diretto", tile=2048, out=os.path.join(S, f"{ins}_diretto.jpg")))
    for est in ("png", "jpg"):
        lavori.append(dict(azione='converti', insieme=ins, id=f"{ins}_{est}_da_tif", da=os.path.join(S, f"{ins}_t2048.tif"), out=os.path.join(S, f"{ins}_conv.{est}")))
for l in lavori:
    r = subprocess.run([sys.executable, "-u", os.path.join(QUI, "scrittura_worker.py"), json.dumps(l)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print((r.stdout.strip().splitlines() or ["?"])[-1], "" if r.returncode == 0 else "!! " + r.stderr.strip().splitlines()[-1], flush=True)
    if l['azione'] == 'converti' or 'tile1024' in l['id'] or 'tile4096' in l['id'] or 'diretto' in l['id']:
        try: os.remove(l['out'])
        except OSError: pass
for ins in casi:
    try: os.remove(os.path.join(S, f"{ins}_t2048.tif"))
    except OSError: pass
print("SCRITTURE_FINITE")
