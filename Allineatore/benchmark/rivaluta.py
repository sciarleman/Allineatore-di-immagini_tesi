"""
Ricalcola le misure senza rieseguire i metodi.

Le trasformazioni stimate sono salvate in lavoro/<immagine>/<metodo>/, quindi
quando cambia il MODO di misurare (numero di finestre, statistiche riportate)
non serve rifare ore di allineamenti: basta rileggerle e rimisurare.

    python rivaluta.py
"""

import json
import os
import sys

QUI = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(QUI)                   # Allineatore/, dove sta il motore
sys.path.insert(0, PROG)
sys.path.insert(0, QUI)

import image_io
from transform_model import Transform
import esegui


def main():
    percorso = os.path.join(QUI, 'risultati.json')
    dati = json.load(open(percorso, encoding='utf-8-sig'))

    arr_f, meta_f = image_io.apri(esegui.RIFERIMENTO, verbose=False)
    pos = esegui._posizioni(meta_f.height, meta_f.width)
    print(f"rimisuro {len(dati)} risultati su {len(pos)} finestre da {esegui.PATCH} px\n")

    cache = {}
    for i, v in enumerate(dati):
        nome_file = v.get('file')
        if not nome_file:
            continue
        if nome_file not in cache:
            cache[nome_file] = image_io.apri(
                os.path.join(esegui.IMMAGINI, nome_file), verbose=False)[0]
        arr_m = cache[nome_file]

        if v['metodo'] == 'nessun allineamento':
            T = None
        else:
            p = os.path.join(QUI, 'lavoro', v['immagine'],
                             v['metodo'].replace(' ', '_'), 'trasformazione.json')
            if not os.path.exists(p):
                # metodo fallito: non c'e' nulla da rimisurare
                continue
            T = Transform.load(p)

        m = esegui.valuta(arr_f, arr_m, T, pos)
        v.update(m)
        r = m.get('residuo_px')
        testo = f"{r:.2f} px" if r is not None else "non misurabile"
        print(f"  [{v['immagine']:<18}] {v['metodo']:<26} {testo:>15}"
              f"   ({m.get('campioni_affidabili', 0)}/{m.get('campioni_totali', 0)} finestre)")

        if i % 5 == 0:
            json.dump(dati, open(percorso, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    json.dump(dati, open(percorso, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\naggiornato {percorso}")


if __name__ == '__main__':
    main()
