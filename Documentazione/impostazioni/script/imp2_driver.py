"""Le impostazioni avanzate che cambiano l'allineamento, una alla volta attorno ai
valori predefiniti (blocchi da 128 px, confidenza 0.10), su ogni coppia. Una
corsa alla volta; quelle gia' fatte si saltano, quindi si puo' riprendere.

    python imp2_driver.py            tutte
    python imp2_driver.py FG 2       solo un insieme e un fattore
"""
import json, os, subprocess, sys
QUI = os.path.dirname(os.path.abspath(__file__))
for d in ("log", "trasformazioni"):
    os.makedirs(os.path.join(QUI, d), exist_ok=True)
CONFIG = [(128, 0.10), (96, 0.10), (192, 0.10), (128, 0.05), (128, 0.20)]
GRUPPI = [("FG", 2, ["NIR850", "UVF", "UVR", "SWIR", "RX_Cappello"]),
          ("CK", 2, ["NIR", "UV"]),
          ("FG", 1, ["NIR850", "UVF"])]
filtro = sys.argv[1:3]
fatti = set()
ris = os.path.join(QUI, "risultati.jsonl")
if os.path.exists(ris):
    for riga in open(ris, encoding="utf-8"):
        r = json.loads(riga)
        fatti.add((r['insieme'], r['banda'], r['fattore'], r['refine_block'], r['min_conf']))
for ins, fattore, bande in GRUPPI:
    if filtro and (ins != filtro[0] or str(fattore) != filtro[1]):
        continue
    for banda in bande:
        for blocco, conf in CONFIG:
            if (ins, banda, fattore, blocco, conf) in fatti:
                continue
            subprocess.run([sys.executable, "-u", os.path.join(QUI, "imp2_worker.py"), ins, banda,
                            str(fattore), str(blocco), str(conf)])
print("IMPOSTAZIONI_FINITO", flush=True)
