"""
Il TRE sulla radiografia contato per dettaglio distinto, per la tesi.

Legge i due file che il benchmark delle radiografie lascia in
Allineatore/benchmark (rx_giudici.json e rx_risultati.json) e non rifa' nessuna
corsa: va lanciato dopo `radiografie.py`. Stampa i numeri del capoverso sul TRE,
la colonna «Radiografia» della tabella del confronto e la ripetibilita'
dell'individuazione a mano.

Le individuazioni a mano contengono dettagli individuati piu' volte in sessioni
diverse. Due individuazioni contano come lo stesso dettaglio se nel visibile
distano meno di SOGLIA px; a ogni dettaglio si assegna la mediana degli errori
delle sue individuazioni. Si usano i soli dettagli interni: ad almeno
BORDO px dal contorno e lontani dai punti usati come vincolo.

    python tre_per_dettaglio.py
"""
import json
import os

import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))
SOGLIA = 30.0       # px nel visibile: sotto, due individuazioni sono lo stesso dettaglio
BORDO = 250.0       # px dal contorno: sotto, il punto e' «di bordo» e non entra nel conto
RAGGIO = 800.0      # px: una finestra di verifica entro questo raggio «vede» il dettaglio
PROPOSTO = 'predefinita: zona + fascia, punti solo per verifica'
SENZA_GROSSOLANO = 'automatico del lotto'


def gruppi(xy, soglia):
    """Etichetta i punti: stessa etichetta a quelli collegati da distanze sotto soglia."""
    n = len(xy)
    et = np.arange(n)

    def trova(i):
        while et[i] != i:
            et[i] = et[et[i]]
            i = et[i]
        return i
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < soglia:
                a, b = trova(i), trova(j)
                if a != b:
                    et[b] = a
    return np.unique(np.array([trova(i) for i in range(n)]), return_inverse=True)[1]


def per_dettaglio(errori, et):
    return np.array([np.median(errori[et == k]) for k in range(et.max() + 1)])


def intervallo_90(x, n=4000):
    rng = np.random.default_rng(0)
    m = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(m, [5, 95])


def coppie(idx):
    return [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]


def main():
    g = json.load(open(os.path.join(BENCH, 'rx_giudici.json'), encoding='utf-8'))
    ris = json.load(open(os.path.join(BENCH, 'rx_risultati.json'), encoding='utf-8'))
    conf = {c['nome']: c for c in ris['configurazioni']}
    P = np.array([p[:4] for p in g['punti']], float)
    d_bordo = np.array([p[4] for p in g['punti']], float)
    lontani = np.array([p[6] for p in g['punti']], bool)
    interni = (d_bordo >= BORDO) & lontani
    Q = P[interni]
    et = gruppi(Q[:, :2], SOGLIA)
    volte = np.bincount(et)

    print(f"individuazioni: {len(P)} in tutto; {int((d_bordo < BORDO).sum())} a meno di {BORDO:.0f} px "
          f"dal bordo; {int(((d_bordo >= BORDO) & ~lontani).sum())} interne ma vicine ai punti di vincolo; "
          f"{len(Q)} interne e indipendenti")
    print(f"dettagli distinti (soglia {SOGLIA:.0f} px): {len(volte)}; individuati una volta "
          f"{int((volte == 1).sum())}, due volte {int((volte == 2).sum())}, "
          f"tre o piu' volte {int((volte >= 3).sum())}")

    print(f"\n{'configurazione':52s}{'mediana':>9s}{'IC 90%':>14s}{'p90':>8s}{'>5 px':>7s}{'>10 px':>7s}{'>30':>5s}")
    for nome, c in conf.items():
        if 'errori_punti' not in c:
            print(f"{nome[:52]:52s}   non riuscita")
            continue
        x = per_dettaglio(np.asarray(c['errori_punti'], float)[interni], et)
        lo, hi = intervallo_90(x)
        print(f"{nome[:52]:52s}{np.median(x):9.1f}{f'[{lo:.1f}-{hi:.1f}]':>14s}{np.percentile(x, 90):8.1f}"
              f"{100 * (x > 5).mean():6.0f}%{100 * (x > 10).mean():6.0f}%{int((x > 30).sum()):5d}")

    if PROPOSTO not in conf or SENZA_GROSSOLANO not in conf:
        print("\nconfigurazioni del capoverso non trovate; nomi disponibili:", list(conf))
        return
    c = conf[PROPOSTO]
    e = np.asarray(c['errori_punti'], float)[interni]
    x = per_dettaglio(e, et)
    x0 = per_dettaglio(np.asarray(conf[SENZA_GROSSOLANO]['errori_punti'], float)[interni], et)
    print(f"\ncapoverso sul TRE ({PROPOSTO}):")
    print(f"   {len(x)} dettagli, {len(e)} individuazioni; mediana {np.median(x):.1f} px; "
          f"90% entro {np.percentile(x, 90):.1f} px; oltre 10 px {100 * (x > 10).mean():.0f}%; "
          f"senza campo grossolano {np.median(x0):.1f} px")
    print(f"   per confronto, contando le {len(e)} individuazioni una per una: mediana {np.median(e):.1f} px; "
          f"90% entro {np.percentile(e, 90):.1f} px; oltre 10 px {100 * (e > 10).mean():.0f}%")

    # gli errori grandi e le finestre di verifica vicine
    centro = np.array([Q[et == k, :2].mean(axis=0) for k in range(len(x))])
    F = np.array(c['finestre']['posizioni'], float)
    R = np.array(c['finestre']['residui'], float)
    viste = np.array([(np.hypot(F[:, 0] - p[0], F[:, 1] - p[1]) < RAGGIO).any() for p in centro])
    grandi = x > 30
    print(f"   dettagli oltre 30 px: {int(grandi.sum())}, di cui {int((grandi & ~viste).sum())} senza finestre "
          f"misurabili entro {RAGGIO:.0f} px e {int((grandi & (volte > 1)).sum())} individuati piu' volte")
    for k in np.argsort(-x)[:int(grandi.sum())]:
        v = np.hypot(F[:, 0] - centro[k, 0], F[:, 1] - centro[k, 1]) < RAGGIO
        print(f"      {x[k]:6.1f} px, finestre vicine {int(v.sum())}"
              + (f" (residuo mediano {np.median(R[v]):.1f} px)" if v.any() else ""))
    print(f"   mediana dove la verifica a finestre misura: {np.median(x[viste]):.1f} px "
          f"({int(viste.sum())} dettagli); dove non misura: {np.median(x[~viste]):.1f} px")

    # ripetibilita' dell'individuazione: lo spostamento indicato a mano (radiografia
    # meno visibile) per lo stesso dettaglio, in due individuazioni diverse
    def ripetibilita(punti, etichette):
        sp = punti[:, 2:] - punti[:, :2]
        diff, diversi = [], 0
        for k in range(etichette.max() + 1):
            idx = np.where(etichette == k)[0]
            d = [float(np.hypot(*(sp[a] - sp[b]))) for a, b in coppie(idx)]
            diff += d
            diversi += int(bool(d) and max(d) > 30)
        return np.array(diff), diversi

    print("\nripetibilita' dell'individuazione (differenza fra due individuazioni dello stesso dettaglio):")
    for nome, punti, etichette in (('tutti i punti', P, gruppi(P[:, :2], SOGLIA)), ('soli interni', Q, et)):
        d, diversi = ripetibilita(punti, etichette)
        print(f"   {nome}: {len(d)} ripetizioni, mediana {np.median(d):.1f} px, oltre 10 px {int((d > 10).sum())}; "
              f"dettagli in cui e' stato indicato un punto diverso (oltre 30 px): {diversi}")


if __name__ == '__main__':
    main()
