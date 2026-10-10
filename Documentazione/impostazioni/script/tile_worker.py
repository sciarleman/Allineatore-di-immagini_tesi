"""Il lato dei tile in scrittura cambia qualcosa nell'immagine allineata?
Stima a «veloce» (uguale per tutti), poi scrittura con il lato dato: tempo,
picco di memoria e impronta del file.   python tile_worker.py 2048"""
import os, sys, io, json, time, hashlib, contextlib, shutil
PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import align_paintings as ap
QUI = os.path.dirname(os.path.abspath(__file__))
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
lato = int(sys.argv[1])
tmp = os.path.join(QUI, f"tmp_tile_{lato}"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
out = os.path.join(tmp, "NIR850_allineata.tif")
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    T, rep = ap.align(os.path.join(FG, "VIS_FraGalgario_TOT.tif"), os.path.join(FG, "NIR850_FraGalgario_TOT.tif"), out,
                      params_path=os.path.join(tmp, "p.json"), factor=4, tile_size=lato,
                      diagnostics_dir=os.path.join(tmp, "diag"), tre_scale='auto', verbose=True)
dt = time.time() - t0
log = buf.getvalue()
import re
m = re.search(r"scrittura.*?([\d.]+) ?(s|min)", log, re.I)
h = hashlib.md5()
with open(out, 'rb') as fh:
    for blocco in iter(lambda: fh.read(1 << 24), b''):
        h.update(blocco)
mem = ap.peak_memory_gb()
rec = {'tile': lato, 't_totale_s': dt, 'picco_memoria_gb': mem[0] if mem else None, 'md5': h.hexdigest(),
       'byte': os.path.getsize(out), 'residuo': rep['verifica']['dopo']['residuo_mediano_px'],
       'righe_scrittura': [r for r in log.splitlines() if 'scritt' in r.lower()][:4]}
shutil.rmtree(tmp, ignore_errors=True)
with open(os.path.join(QUI, "tile.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"tile {lato}: {dt:.0f} s in tutto, picco {rec['picco_memoria_gb']:.2f} GB, md5 {rec['md5']}, residuo {rec['residuo']:.3f} px", flush=True)
