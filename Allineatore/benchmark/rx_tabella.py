"""
Le tabelle del benchmark delle radiografie, in Markdown, da rx_risultati.json.

    python rx_tabella.py
"""

import json
import os

QUI = os.path.dirname(os.path.abspath(__file__))


def f1(x):
    return '—' if x is None else f"{x:.1f}"


def main():
    with open(os.path.join(QUI, 'rx_risultati.json'), encoding='utf-8') as fh:
        conf = json.load(fh)['configurazioni']

    print("| configurazione | dettagli interni: mediana | media | p90 | distorsione (x, y) "
          "| dettagli sul bordo | distanza fra le linee | finestre misurabili | "
          "mediana finestre | finestre oltre 10 px | minuti |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conf:
        if 'errore' in c:
            print(f"| {c['nome']} | non riuscito: {c['errore'][:60]} |||||||||| "
                  f"{c['secondi'] / 60:.1f} |")
            continue
        pi, pb, f, li = c['punti_interni'], c['punti_bordo'], c['finestre'], c['linee']
        print(f"| {c['nome']} | {f1(pi['mediana'])} | {f1(pi['media'])} | {f1(pi['p90'])} "
              f"| {pi['distorsione_x']:+.1f}, {pi['distorsione_y']:+.1f} "
              f"| {f1(pb['mediana'])} | {f1(li['mediana_assoluta'])} "
              f"| {f['misurabili']}/{f['totali']} "
              f"| {f.get('mediana', 0):.2f} | {f.get('oltre_10', '—')} "
              f"| {c['secondi'] / 60:.1f} |")


if __name__ == '__main__':
    main()
