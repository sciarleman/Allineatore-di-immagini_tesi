"""Tempo, memoria e peso su disco della SCRITTURA dell'immagine allineata, per
lato dei tile e per formato. Una misura per processo.

    python scrittura_worker.py '<json>'
"""
import os, sys, json, time
import numpy as np

PROG = r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA"
sys.path.insert(0, os.path.join(PROG, "Allineatore"))
import image_io, align_paintings as ap, formati
from transform_model import Transform

QUI = os.path.dirname(os.path.abspath(__file__))
c = json.loads(sys.argv[1])
rec = dict(c)
t0 = time.time()
if c['azione'] == 'scrivi':
    campo = np.load(c['campo']) if c.get('campo') and os.path.exists(c['campo']) else None
    T = Transform(np.load(c['M']), campo, c.get('passo') if campo is not None else None)
    arr_m, info = image_io.apri(c['moving'], verbose=False)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ap.scrivi_allineata(T, arr_m, c['out'], c['H'], c['W'], info.samples, info.dtype, tile_size=c['tile'])
else:                                   # 'converti': da un file gia' scritto a un altro formato
    formati.converti(c['da'], c['out'])
rec['t_s'] = time.time() - t0
mem = ap.peak_memory_gb()
rec['picco_memoria_gb'] = None if mem is None else mem[0]
rec['peso_gb'] = os.path.getsize(c['out']) / 1024 ** 3
with open(os.path.join(QUI, "impostazioni", "scrittura.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
print(c['id'], f"{rec['t_s']:.0f} s, {rec['picco_memoria_gb']:.2f} GB di picco, {rec['peso_gb']:.2f} GB su disco", flush=True)
