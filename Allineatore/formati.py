"""
I formati in cui salvare le immagini prodotte, e la scelta nell'interfaccia.

Si puo' chiedere piu' di un formato. Se non se ne sceglie nessuno si salva
SOLO nel formato dell'originale.

Il primo file si scrive sempre nel modo piu' leggero: il TIFF, se e' fra quelli
chiesti, perche' si scrive a tile con la memoria costante; gli altri formati si
ricavano poi da quel file. PNG e JPEG non si possono scrivere a pezzi: per
salvarli serve l'immagine intera in memoria (0.9 GB per un RGB a 8 bit di
queste dimensioni, il doppio a 16 bit), qualunque sia la strada.
"""

import os

import numpy as np
import cv2

# etichetta mostrata -> estensione
FORMATI = {'TIFF': '.tif', 'PNG': '.png', 'JPEG': '.jpg'}


def normalizza(formati):
    """Da etichette o estensioni a una lista di estensioni, senza doppioni."""
    out = []
    for f in formati or []:
        e = FORMATI.get(f, f).lower()
        e = {'.tiff': '.tif', '.jpeg': '.jpg'}.get(e, e if e.startswith('.') else '.' + e)
        if e not in FORMATI.values():
            raise ValueError(f"formato non previsto: {f} (si possono usare "
                             f"{', '.join(FORMATI)})")
        if e not in out:
            out.append(e)
    return out


def piano(formati, estensione_originale):
    """
    Quale formato scrivere per primo e quali ricavare dopo.

    Returns:
        (primo, altri): il primo e' il TIFF se chiesto, altrimenti il primo
        della lista; senza scelta, l'estensione dell'originale e nessun altro.
    """
    scelti = normalizza(formati)
    if not scelti:
        return estensione_originale.lower(), []
    primo = '.tif' if '.tif' in scelti else scelti[0]
    return primo, [e for e in scelti if e != primo]


def con_estensione(path, estensione):
    return os.path.splitext(path)[0] + estensione


def converti(path_in, path_out, qualita_jpeg=95):
    """Ricava path_out (PNG o JPEG) da un'immagine gia' scritta."""
    import image_io
    arr, _ = image_io.apri(path_in, verbose=False)
    img = np.asarray(arr)
    est = os.path.splitext(path_out)[1].lower()
    if est == '.jpg' and img.dtype != np.uint8:
        # il JPEG non ha i 16 bit: si scala in 8 bit, senza stirare i livelli
        img = (img.astype(np.float32) * (255.0 / np.iinfo(img.dtype).max) + 0.5).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2BGR)
    parametri = [cv2.IMWRITE_JPEG_QUALITY, qualita_jpeg] if est == '.jpg' else []
    if not image_io.scrivi_immagine(path_out, img, parametri):
        raise IOError(f"Impossibile scrivere {path_out}")
    del img, arr
    return path_out


def scrivi_altri(path_scritto, altri, verbose=True):
    """Ricava dal file scritto tutti gli altri formati chiesti."""
    fatti = []
    for e in altri:
        dest = con_estensione(path_scritto, e)
        converti(path_scritto, dest)
        if verbose:
            print(f"      anche in {e[1:].upper()}: {dest} "
                  f"({os.path.getsize(dest) / 1024 ** 3:.2f} GB)")
        fatti.append(dest)
    return fatti


def menu_formati(padre, scelti=None, testo_vuoto="come l'originale"):
    """
    Un menu a tendina con le crocette: se ne possono spuntare piu' d'uno.

    Returns:
        (widget, funzione che restituisce le etichette spuntate)
    """
    import tkinter as tk
    variabili = {n: tk.BooleanVar(value=n in (scelti or [])) for n in FORMATI}
    bottone = tk.Menubutton(padre, relief='raised', indicatoron=True, width=22, anchor='w')
    menu = tk.Menu(bottone, tearoff=False)
    bottone.configure(menu=menu)

    def etichetta():
        sp = [n for n, v in variabili.items() if v.get()]
        bottone.configure(text=", ".join(sp) if sp else testo_vuoto)

    for n, v in variabili.items():
        menu.add_checkbutton(label=n, variable=v, command=etichetta)
    menu.add_separator()
    menu.add_command(label=f"nessuno ({testo_vuoto})",
                     command=lambda: ([v.set(False) for v in variabili.values()], etichetta()))
    etichetta()
    return bottone, lambda: [n for n, v in variabili.items() if v.get()]
