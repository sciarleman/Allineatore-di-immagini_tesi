"""Copia i tre PNG del dipinto nuovo in TIFF non compresso nello scratchpad,
per lo studio delle impostazioni: stessi pixel, ma leggibili a memmap, cosi' le
decine di run dello studio non pagano ogni volta la decodifica del PNG."""
import os, sys, time, hashlib
import numpy as np

sys.path.insert(0, r"C:\Users\carlo\Desktop\Uni\Tesi\script\Aligner_BETA\Allineatore")
import image_io, tiff_memmap

RUN = "C:/Users/carlo/Desktop/check" + chr(0xe0)
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck_tiff"); os.makedirs(DEST, exist_ok=True)
for nome in ("VIS", "NIR", "UV"):
    out = os.path.join(DEST, nome + ".tif")
    if os.path.exists(out):
        print(nome, "gia' presente"); continue
    t0 = time.time()
    arr, info = image_io.apri(os.path.join(RUN, nome + ".png"), verbose=False)
    H, W = arr.shape[:2]; samples = 1 if arr.ndim == 2 else arr.shape[2]
    w = tiff_memmap.create_tiff_memmap(out + ".parziale", H, W, samples=samples, dtype=arr.dtype, sequenziale=True)
    for y in range(0, H, 2048):
        w.scrivi(arr[y:y + 2048])
    w.chiudi(); os.replace(out + ".parziale", out)
    riletto, _ = image_io.apri(out, verbose=False)
    uguale = all(np.array_equal(arr[y:y + 4096], riletto[y:y + 4096]) for y in range(0, H, 4096))
    print(f"{nome}: {W}x{H} {samples}ch {arr.dtype} -> {os.path.getsize(out) / 1e9:.2f} GB in {time.time() - t0:.0f} s; pixel identici: {uguale}", flush=True)
    del arr, riletto
