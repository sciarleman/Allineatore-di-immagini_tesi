"""Riprova, con il controllo dell'aggancio, dei casi che contano. Usa i PNG veri
dell'affresco e i TIFF del Fra Galgario; una configurazione per processo."""
import os, sys, json, subprocess, time

QUI = os.path.dirname(os.path.abspath(__file__))
FG = r"C:\Users\carlo\Desktop\Analisi tesi\FotoTesi\Totali l'allegrezza"
CK = "C:/Users/carlo/Desktop/check" + chr(0xe0)
vis = {"FG": os.path.join(FG, "VIS_FraGalgario_TOT.tif"), "CK": os.path.join(CK, "VIS.png")}
f = {"NIR850": "NIR850_FraGalgario_TOT.tif", "UVR": "UVR_FraGalgario_TOT.tif", "SWIR": "big_SWIR_FraGalgario_TOT.tif",
     "RX_Cappello": "RX_FraGalgario_TOT_Cappellochiaro.tif", "RX_Testa": "RX_FraGalgarioTOT_TestaChiara.tif",
     "NIR": "NIR.png", "UV": "UV.png"}
casi = [("FG", "NIR850", 2, 128, 0.10), ("FG", "UVR", 2, 128, 0.10), ("FG", "RX_Cappello", 2, 128, 0.10),
        ("FG", "RX_Testa", 2, 128, 0.10), ("FG", "SWIR", 2, 128, 0.10), ("FG", "RX_Cappello", 1, 128, 0.10),
        ("CK", "NIR", 2, 128, 0.10), ("CK", "NIR", 2, 192, 0.10), ("CK", "NIR", 2, 128, 0.20),
        ("CK", "UV", 2, 128, 0.10), ("CK", "NIR", 1, 128, 0.10), ("CK", "UV", 1, 128, 0.10)]
etichetta = sys.argv[1] if len(sys.argv) > 1 else "v2"
for ins, banda, fat, rb, mc in casi:
    c = dict(insieme=ins, banda=banda, fixed=vis[ins], moving=os.path.join(FG if ins == "FG" else CK, f[banda]),
             factor=fat, refine_block=rb, min_conf=mc, nonrigid=True, gruppo=etichetta,
             id=f"{etichetta}_{ins}_{banda}_f{fat}_b{rb}_c{int(100 * mc):02d}")
    r = subprocess.run([sys.executable, "-u", os.path.join(QUI, "imp_worker.py"), json.dumps(c)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(time.strftime('%H:%M:%S'), (r.stdout.strip().splitlines() or ["(nessuna uscita)"])[-1],
          "" if r.returncode == 0 else "!! " + (r.stderr.strip().splitlines() or ["?"])[-1], flush=True)
print("V2_FINITO", flush=True)
