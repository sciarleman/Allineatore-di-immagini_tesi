"""
Test del falso colore: la composizione per spostamento di canali deve dare
ESATTAMENTE lo stesso file della composizione in decimali.

    python FalsoColore\\test_falsocolore.py

Il programma compone con numeri interi (striscia_intera),
perche' e' circa due volte piu' veloce. La garanzia che il risultato non
cambi non sta in un ragionamento, sta qui:

1. su TUTTI i valori possibili: ogni livello a 8 e a 16 bit, tutte le terne di
   canali a 8 bit, e dieci milioni di terne a 16 bit con gli estremi, per
   entrambe le ricette e per ogni scelta di canale della banda;
2. da capo a fondo: riprese sintetiche in tre formati (8 bit RGB, 16 bit RGB,
   16 bit in grigio), composte dal programma e confrontate con la via
   decimale ricalcolata.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
_MOTORE = os.path.join(os.path.dirname(QUI), 'Allineatore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import falso_colore as fc
import image_io
import tiff_memmap
from test_pipeline import synthetic_texture


def confronta(nome, vis, ban, canali=('media',)):
    """Via intera contro via decimale sulla stessa striscia, due ricette."""
    diversi = 0
    for tipo in fc.RICETTE:
        for canale in canali:
            a = fc.striscia_intera(tipo, vis, ban, canale)
            b = fc.striscia_decimale(tipo, vis, ban, canale)
            diversi += int((a != b).sum())
    print(f"  {nome:<44} pixel diversi: {diversi}")
    return diversi == 0


def tutti_i_valori():
    rng = np.random.default_rng(0)
    esiti = []
    print("1. tutti i valori possibili")

    # visibile a 8 bit: ogni livello in ogni canale; banda a 8 bit: ogni terna
    terne = np.stack(np.meshgrid(np.arange(256), np.arange(256), np.arange(256),
                                 indexing='ij'), -1).reshape(4096, 4096, 3).astype(np.uint8)
    vis8 = np.roll(terne, 1, axis=2)
    esiti.append(confronta("8 bit, tutte le 16.7 milioni di terne", vis8, terne,
                           canali=('media', 'r', 'g', 'b')))
    del terne, vis8

    # 16 bit: ogni livello, in grigio e come visibile
    livelli = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    vis16 = np.stack([livelli, livelli[::-1], livelli.T], -1)
    esiti.append(confronta("16 bit in grigio, tutti i 65 536 livelli", vis16, livelli))

    # 16 bit, terne: dieci milioni a caso piu' gli estremi
    t = rng.integers(0, 65536, (2500, 4000, 3), dtype=np.uint16)
    t[0, :8] = [[0, 0, 0], [65535, 65535, 65535], [0, 0, 65535], [65535, 0, 0],
                [1, 0, 0], [65534, 65535, 65535], [128, 128, 128], [32767, 32768, 32767]]
    v = rng.integers(0, 65536, t.shape, dtype=np.uint16)
    esiti.append(confronta("16 bit RGB, dieci milioni di terne", v, t,
                           canali=('media', 'r', 'g', 'b')))
    return all(esiti)


def salva(path, arr):
    campioni = 1 if arr.ndim == 2 else arr.shape[2]
    mm = tiff_memmap.create_tiff_memmap(path, arr.shape[0], arr.shape[1],
                                        samples=campioni, dtype=arr.dtype)
    mm[:] = arr
    mm.flush()
    del mm


def da_capo_a_fondo(lavoro):
    print("\n2. da capo a fondo, su riprese sintetiche")
    H, W = 1200, 1000
    g = synthetic_texture(H, W, seed=0).astype(np.float32)
    vis = np.stack([g, 0.8 * g + 20, 0.6 * g + 40], -1).clip(0, 255).astype(np.uint8)
    ir = 255.0 - g
    bande = {
        'banda_8bit_rgb.tif': np.stack([ir, ir * 0.9 + 10, ir], -1).clip(0, 255).astype(np.uint8),
        'banda_16bit_rgb.tif': np.stack([ir * 250, ir * 240 + 900, ir * 255], -1).astype(np.uint16),
        'banda_16bit_grigio.tif': (ir * 200 + 3000).astype(np.uint16),
    }
    p_vis = os.path.join(lavoro, 'vis.tif')
    salva(p_vis, vis)
    lavori = []
    for nome, arr in bande.items():
        salva(os.path.join(lavoro, nome), arr)
        for tipo in fc.RICETTE:
            lavori.append((tipo, os.path.join(lavoro, nome),
                           os.path.join(lavoro, f"{tipo}_{nome}")))

    esiti = []
    report = fc.componi_molte(p_vis, lavori, verbose=False)
    for (tipo, p_banda, p_out), r in zip(lavori, report):
        fatto, _ = image_io.apri(p_out, verbose=False)
        ban, _ = image_io.apri(p_banda, verbose=False)
        atteso = fc.striscia_decimale(tipo, vis, np.asarray(ban), 'media')
        diversi = int((np.asarray(fatto) != atteso).sum())
        print(f"  {tipo} da {os.path.basename(p_banda):<24} pixel diversi: {diversi}")
        esiti.append(diversi == 0)
        del fatto, ban
    return all(esiti)


def main():
    lavoro = tempfile.mkdtemp(prefix='falsocolore_test_')
    try:
        esiti = [('tutti i valori possibili', tutti_i_valori()),
                 ('da capo a fondo', da_capo_a_fondo(lavoro))]
    finally:
        shutil.rmtree(lavoro, ignore_errors=True)
    print()
    for nome, ok in esiti:
        print(f"  {'ok     ' if ok else 'FALLITO'}  {nome}")
    tutto = all(ok for _, ok in esiti)
    print("TEST SUPERATO" if tutto else "TEST FALLITO")
    return 0 if tutto else 1


if __name__ == '__main__':
    sys.exit(main())
