"""
Verifica indipendente della qualita' dell'allineamento, A PIENA RISOLUZIONE.

Perche' serve un controllo separato dalla stima: il costo finale di un
ottimizzatore dice quanto l'ottimizzatore e' contento di se stesso, non quanto
l'allineamento sia buono. Qui invece si prendono N campioni a piena risoluzione
in posizioni scelte indipendentemente, si applica la trasformazione stimata e si
misura quanto ancora si spostano l'uno rispetto all'altro. Il numero che ne esce
("residuo mediano in px") e' direttamente interpretabile: e' l'errore di
allineamento residuo che vedresti sovrapponendo le due immagini.

Viene calcolato anche il costo MIND (la metrica robusta cross-modale del
progetto) prima e dopo l'allineamento, come secondo indicatore indipendente.
"""

import numpy as np
import cv2

import image_io
from block_matching import gradient_magnitude
from robust_metrics import RobustMetrics


def residual_check(fixed_arr, moving_arr, transform, n_side=8, patch=None,
                   margin_frac=0.12, min_conf=0.05, verbose=True, n_boot=2000,
                   mask=None, copertura_min=0.98, target_affidabili=40,
                   patch_min=256, patch_max=512, adattiva=True):
    """
    Misura l'errore di allineamento residuo su una griglia di campioni a piena
    risoluzione.

    Sulla scelta di n_side=8 (64 finestre). Prima erano 25 (5x5), un valore
    scelto per rapidita' e mai verificato. Misurato poi sulla coppia VIS/UVF,
    con 256 finestre come riferimento: con 25 finestre la mediana oscillava fra
    0.35 e 0.67 px (intervallo al 90%) contro un valore vero di 0.50, e il
    massimo riportato era ~3 px contro gli 11.3 px effettivamente presenti.
    Costo: 25 finestre 1 s, 64 finestre 2 s, 256 finestre 9 s, a fronte dei ~43 s
    della sola stima. Il risparmio non giustificava l'imprecisione.

    Sul p95: con poche finestre non e' un percentile ma quasi il massimo (su 22
    misure cade fra la 21esima e la 22esima), quindi oscilla moltissimo. Resta
    nel risultato perche' con 64 finestre ha piu' senso, ma il modo affidabile
    di descrivere la coda sono le QUOTE di finestre sopra 1, 2 e 5 px, che sono
    frequenze e non posizioni in una graduatoria corta.

    Args:
        fixed_arr, moving_arr: memmap (o array) delle due immagini a piena risoluzione
        transform: Transform da verificare (None = identita', per il "prima")
        n_side: lato INIZIALE della griglia. Con adattiva=True la griglia si
            infittisce da sola finche' le finestre affidabili non raggiungono
            target_affidabili, riusando i campioni gia' misurati.
        patch: lato del campione in px. None = scelto in proporzione al
            soggetto (lato/30) fra patch_min e patch_max. Un lato fisso in px
            non si adatta a dipinti fotografati a risoluzioni diverse: su
            un'immagine piccola 512 px sono mezza tela, su una grande un
            trentesimo. Sul Fra Galgario (15680 px di lato corto del soggetto)
            la regola restituisce 512, cioe' il valore storico: i numeri gia'
            misurati non cambiano.
        target_affidabili: quante finestre MISURABILI si vogliono prima di
            fermarsi. E' il vero invariante da tenere costante fra un dipinto e
            l'altro: non la dimensione dell'immagine, ne' il numero di
            posizioni tentate. Misurato su questa stessa immagine, a parita' di
            n_side, le finestre utili vanno da 12 (radiografia) a 42 (UVF):
            quel che cambia non e' quanti pixel ha il file, e' quanto le due
            bande si somigliano. Con 11 finestre l'intervallo di confidenza
            della mediana era [0.81-3.47] px, cioe' un fattore quattro.
        patch_min, patch_max: limiti assoluti del lato della finestra. Il
            minimo e' fisica della misura, non estetica: sotto i ~256 px la
            correlazione di fase sul gradiente perde affidabilita'.
        adattiva: False torna al comportamento storico, una sola griglia
            n_side x n_side.
        margin_frac: margine dai bordi (li' spesso c'e' cornice/sfondo). E' un
            margine RETTANGOLARE, quindi e' un ripiego: vale solo quando non si
            sa dove sia il soggetto. Se viene passata una maschera, non si usa.
        n_boot: ricampionamenti per l'intervallo di confidenza della mediana
        mask: maschera del soggetto, a qualunque risoluzione (viene riscalata).
            Senza, si misura anche sullo sfondo, e sullo sfondo il residuo non
            dice nulla sull'allineamento del dipinto: essendo a una distanza
            diversa dalla camera si sposta genuinamente in modo diverso
            (parallasse). Misurato sul Fra Galgario: le finestre a cavallo del
            bordo della tela davano 3.47 px di mediana su NIR850 contro 0.18 px
            all'interno, venti volte tanto, e con confidenza ALTA (0.43-0.59),
            perche' lo spigolo della tela e' un dettaglio nettissimo. Non erano
            misure sbagliate: erano misure giuste di un oggetto sbagliato.
        copertura_min: frazione di finestra che deve stare dentro la maschera.
            Alta di proposito (0.98): una finestra a cavallo del contorno e'
            esattamente il caso patologico che si vuole escludere.

    Returns:
        dict con i residui e la lista dei campioni (x, y, dx, dy, confidenza).
    """
    H, W = fixed_arr.shape[:2]

    mk = None
    if mask is not None:
        mk = (np.asarray(mask) > 0)
        if not mk.any():
            mk = None
    if mk is not None:
        # La griglia si concentra sul rettangolo che contiene il soggetto: su
        # una forma non rettangolare, spargerla sull'intera immagine sprecherebbe
        # buona parte dei campioni negli angoli, che sono sfondo.
        sy, sx = mk.shape[0] / float(H), mk.shape[1] / float(W)
        # Dalle righe e dalle colonne che contengono almeno un punto, non da
        # np.nonzero(mk): quello restituisce due interi a 64 bit per OGNI pixel
        # della zona, cioe' circa 8 GB quando la maschera arriva a piena
        # risoluzione su un'immagine da 593 Mpx (a precisione massima, e nella
        # verifica a griglia fissa a ogni precisione). Il rettangolo e' lo stesso.
        righe, colonne = np.flatnonzero(mk.any(axis=1)), np.flatnonzero(mk.any(axis=0))
        y_lo, y_hi = righe[0] / sy, righe[-1] / sy
        x_lo, x_hi = colonne[0] / sx, colonne[-1] / sx
    else:
        sy = sx = None
        y_lo, y_hi = H * margin_frac, H * (1 - margin_frac)
        x_lo, x_hi = W * margin_frac, W * (1 - margin_frac)

    lato_soggetto = min(y_hi - y_lo, x_hi - x_lo)
    if patch is None:
        # Il lato si lega al soggetto solo quando il soggetto si sa dov'e'. Senza
        # maschera il rettangolo dei margini e' una supposizione, e dimensionare
        # la finestra su una supposizione non e' piu' generale, e' solo piu'
        # fragile: li' si tiene il valore storico.
        patch = (int(np.clip(lato_soggetto / 30.0, patch_min, patch_max)) // 2 * 2
                 if mk is not None else patch_max)
    patch = max(16, min(int(patch), H, W))
    half = patch // 2

    if mk is not None:
        gy_lo, gy_hi = max(half, y_lo + half), min(H - half, y_hi - half)
        gx_lo, gx_hi = max(half, x_lo + half), min(W - half, x_hi - half)
    else:
        gy_lo, gy_hi, gx_lo, gx_hi = y_lo, y_hi, x_lo, x_hi

    # Tetto all'infittimento: due finestre che si sovrappongono non sono due
    # misure indipendenti, e contarle come tali renderebbe l'intervallo di
    # confidenza qui sotto una bugia. Quindi il passo non scende mai sotto il
    # lato della finestra.
    n_max = max(int(n_side), int(lato_soggetto // patch))

    win = cv2.createHanningWindow((patch, patch), cv2.CV_64F)
    visti = {}

    def misura(cy, cx):
        y0, x0 = int(cy) - half, int(cx) - half
        if y0 < 0 or x0 < 0 or y0 + patch > H or x0 + patch > W:
            return
        if mk is not None:
            # La finestra deve stare quasi tutta dentro il soggetto. Il
            # criterio e' una frazione di area, quindi non dipende
            # dall'orientamento del contorno: vale identico sul lato
            # verticale di un rettangolo, sulla diagonale di un ottagono e
            # sull'arco di un tondo.
            my0, my1 = int(y0 * sy), int((y0 + patch) * sy)
            mx0, mx1 = int(x0 * sx), int((x0 + patch) * sx)
            finestra = mk[my0:max(my1, my0 + 1), mx0:max(mx1, mx0 + 1)]
            if finestra.size == 0 or finestra.mean() < copertura_min:
                return
        f_patch = np.asarray(fixed_arr[y0:y0 + patch, x0:x0 + patch])
        if transform is None:
            # Confronto grezzo, senza trasformazione: se la moving e' piu'
            # piccola della fixed la finestra risulta tagliata e le due non
            # sono confrontabili. Si salta il campione invece di misurare
            # una cosa senza senso.
            m_patch = np.asarray(moving_arr[y0:y0 + patch, x0:x0 + patch])
            if m_patch.shape[:2] != f_patch.shape[:2]:
                return
        else:
            m_patch = transform.warp_tile_from(moving_arr, y0, x0, patch, patch)
        if f_patch.ndim == 3:
            f_patch = cv2.cvtColor(f_patch, cv2.COLOR_RGB2GRAY)
        if m_patch.ndim == 3:
            m_patch = cv2.cvtColor(m_patch, cv2.COLOR_RGB2GRAY)

        gf = gradient_magnitude(f_patch)
        gm = gradient_magnitude(m_patch)
        if gf.std() < 1e-4 or gm.std() < 1e-4:
            return
        (dx, dy), conf = cv2.phaseCorrelate(gf.astype(np.float64),
                                            gm.astype(np.float64), win)
        visti[(int(cx), int(cy))] = (int(cx), int(cy), dx, dy, conf)

    # Infittimento a raddoppio: linspace(a, b, 2n-1) contiene esattamente le
    # posizioni di linspace(a, b, n), quindi ogni giro riusa tutto il lavoro
    # del precedente e paga solo le posizioni nuove.
    livello = int(n_side)
    n_usato = livello
    giri = []
    while True:
        ys = np.linspace(gy_lo, gy_hi, livello).astype(int)
        xs = np.linspace(gx_lo, gx_hi, livello).astype(int)
        for cy in ys:
            for cx in xs:
                if (int(cx), int(cy)) not in visti:
                    misura(cy, cx)
        n_usato = livello
        n_aff = sum(1 for v in visti.values() if v[4] >= min_conf)
        giri.append((livello, len(visti), n_aff))
        prossimo = 2 * livello - 1
        if not adattiva or n_aff >= target_affidabili or prossimo > n_max:
            break
        livello = prossimo

    samples = np.array(sorted(visti.values(), key=lambda v: (v[1], v[0])),
                       dtype=np.float64).reshape(-1, 5)
    good = samples[samples[:, 4] >= min_conf]
    if len(good) == 0:
        return {'n_campioni': int(len(samples)), 'n_affidabili': 0}

    mag = np.hypot(good[:, 2], good[:, 3])

    # Intervallo di confidenza della mediana per ricampionamento: dice quanto
    # del numero riportato e' sostanza e quanto e' l'aver guardato solo alcune
    # finestre. Senza, una mediana di 0.41 e una di 0.58 sembrano diverse anche
    # quando sono la stessa cosa misurata due volte.
    ic_lo = ic_hi = float(np.median(mag))
    if len(mag) >= 8:
        rng = np.random.default_rng(0)
        campioni_boot = rng.choice(mag, size=(n_boot, len(mag)), replace=True)
        mediane = np.median(campioni_boot, axis=1)
        ic_lo, ic_hi = (float(v) for v in np.percentile(mediane, [5, 95]))

    out = {
        'n_campioni': int(len(samples)),
        'n_affidabili': int(len(good)),
        'n_side_usato': int(n_usato),
        'lato_finestra_px': int(patch),
        'griglia_infittimenti': giri,
        'residuo_mediano_px': float(np.median(mag)),
        'residuo_mediano_ic90': [ic_lo, ic_hi],
        'residuo_medio_px': float(mag.mean()),
        'residuo_p95_px': float(np.percentile(mag, 95)),
        'residuo_max_px': float(mag.max()),
        # Le quote sono il modo robusto di descrivere la coda: sono frequenze,
        # non posizioni in una graduatoria di pochi elementi.
        'quota_sopra_1px': float((mag > 1.0).mean()),
        'quota_sopra_2px': float((mag > 2.0).mean()),
        'quota_sopra_5px': float((mag > 5.0).mean()),
        'quota_sopra_10px': float((mag > 10.0).mean()),
        # Quante delle finestre tentate danno una misura: la mediana vale per
        # quella parte del soggetto, non per il resto.
        'copertura': float(len(good) / max(len(samples), 1)),
        # Lo spostamento medio come vettore: un errore sistematico, tutto
        # spostato dalla stessa parte, si vede qui e non nel modulo. Cinque
        # pixel tutti verso destra e cinque pixel in direzioni a caso danno la
        # stessa mediana, ma solo il primo e' un difetto da cercare.
        'distorsione_px': [float(good[:, 2].mean()), float(good[:, 3].mean())],
        'campioni': good.tolist(),
        # Le finestre in cui la misura non e' riuscita non vanno perse: dicono
        # DOVE le due immagini non condividono struttura, cioe' dove il
        # risultato non e' verificato. Non sono un elenco di posti in cui
        # mettere punti a mano: di solito li' non riconosce niente neanche
        # l'occhio, perche' le due riprese mostrano oggetti fisici diversi.
        'campioni_deboli': samples[samples[:, 4] < min_conf].tolist(),
    }
    if verbose:
        coda = (f", griglia {n_usato}x{n_usato}" if len(giri) == 1 else
                f", griglia infittita {giri[0][0]}x{giri[0][0]} -> {n_usato}x{n_usato}"
                f" per arrivare a {target_affidabili} misure")
        print(f"  residuo su {len(good)}/{len(samples)} finestre da {patch}px: "
              f"mediano {out['residuo_mediano_px']:.2f} px "
              f"[{ic_lo:.2f}-{ic_hi:.2f} al 90%]{coda}")
        print(f"    finestre oltre 1 px: {100 * out['quota_sopra_1px']:.0f}%, "
              f"oltre 2 px: {100 * out['quota_sopra_2px']:.0f}%, "
              f"oltre 5 px: {100 * out['quota_sopra_5px']:.0f}%"
              f"   (p95 {out['residuo_p95_px']:.2f}, max {out['residuo_max_px']:.2f})")
        print(f"    oltre 10 px: {100 * out['quota_sopra_10px']:.0f}%; distorsione media "
              f"({out['distorsione_px'][0]:+.2f}, {out['distorsione_px'][1]:+.2f}) px")
    return out


FASCE_BORDO = ((0, 500), (500, 1000), (1000, 2000), (2000, None))


def verifica_bordo(fixed_arr, moving_arr, transform, mask=None, lato=256,
                   passo=320, max_per_fascia=60, min_conf=0.2, copertura_min=0.98,
                   verbose=True):
    """
    Residuo in funzione della DISTANZA DAL CONTORNO del soggetto.

    residual_check misura bene l'interno ma, per costruzione, non guarda il
    bordo: la sua griglia sta a un 12% dai lati (o, con una maschera, a mezza
    finestra da 512 px). Sul Fra Galgario e' proprio li' che stava il difetto
    piu' visibile -- la mano sulla bottiglia, fuori di 5-14 px -- mentre la
    verifica riportava 0.045 px. Un numero vero, ma dell'interno soltanto.

    Qui le finestre sono piu' piccole (256 px) per poter stare vicino al
    contorno, e vengono raggruppate per fasce di distanza dal contorno. La
    distanza si calcola con una distance transform della maschera, quindi vale
    per qualunque forma -- rettangolo, ottagono, tondo, mandorla -- senza
    ragionare per righe e colonne.

    Senza maschera il "contorno" e' il bordo dell'immagine: su un soggetto che
    non riempie il fotogramma le fasce esterne cadono sullo sfondo e il
    risultato e' pessimista. Per un dipinto non rettangolare serve la maschera.

    Returns:
        dict con, per ogni fascia, numero di finestre affidabili, mediana, p90 e
        quote oltre 1/2/5 px, piu' le finestre peggiori con le coordinate.
    """
    H, W = fixed_arr.shape[:2]
    # La distance transform si fa su una versione ridotta: la distanza serve in
    # fasce di centinaia di pixel, non al pixel.
    rid = max(1, int(round(max(H, W) / 2000)))
    hs, ws = (H + rid - 1) // rid, (W + rid - 1) // rid
    if mask is not None and np.asarray(mask).any():
        mk = cv2.resize((np.asarray(mask) > 0).astype(np.uint8), (ws, hs),
                        interpolation=cv2.INTER_NEAREST)
    else:
        mk = np.ones((hs, ws), np.uint8)
    # Il bordo dell'immagine conta sempre come contorno: si circonda di zeri.
    dist = cv2.distanceTransform(np.pad(mk, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * rid

    half = lato // 2
    candidati = {f: [] for f in FASCE_BORDO}
    for cy in range(half, H - half + 1, passo):
        for cx in range(half, W - half + 1, passo):
            y0s, x0s = (cy - half) // rid, (cx - half) // rid
            y1s, x1s = (cy + half) // rid + 1, (cx + half) // rid + 1
            if mk[y0s:y1s, x0s:x1s].mean() < copertura_min:
                continue
            d = float(dist[min(cy // rid, hs - 1), min(cx // rid, ws - 1)])
            for f in FASCE_BORDO:
                if d >= f[0] and (f[1] is None or d < f[1]):
                    candidati[f].append((cx, cy, d))
                    break

    win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
    campioni = []
    for f, lista in candidati.items():
        # Se una fascia ha piu' finestre del necessario se ne prende una ogni k,
        # nell'ordine di scansione: restano distribuite lungo tutto il contorno.
        k = max(1, int(np.ceil(len(lista) / float(max_per_fascia))))
        for cx, cy, d in lista[::k]:
            y0, x0 = cy - half, cx - half
            fp = np.asarray(fixed_arr[y0:y0 + lato, x0:x0 + lato])
            if transform is None:
                mp = np.asarray(moving_arr[y0:y0 + lato, x0:x0 + lato])
                if mp.shape[:2] != fp.shape[:2]:
                    continue
            else:
                mp = transform.warp_tile_from(moving_arr, y0, x0, lato, lato)
            if fp.ndim == 3:
                fp = cv2.cvtColor(fp, cv2.COLOR_RGB2GRAY)
            if mp.ndim == 3:
                mp = cv2.cvtColor(mp, cv2.COLOR_RGB2GRAY)
            gf, gm = gradient_magnitude(fp), gradient_magnitude(mp)
            if gf.std() < 1e-4 or gm.std() < 1e-4:
                continue
            (dx, dy), c = cv2.phaseCorrelate(gf.astype(np.float64),
                                             gm.astype(np.float64), win)
            campioni.append((cx, cy, d, dx, dy, c))

    c = np.array(campioni, dtype=np.float64).reshape(-1, 6)
    buone = c[c[:, 5] >= min_conf]
    mag = np.hypot(buone[:, 3], buone[:, 4])
    fasce = []
    for lo, hi in FASCE_BORDO:
        sel = (buone[:, 2] >= lo) & ((buone[:, 2] < hi) if hi is not None else True)
        m = mag[sel]
        fasce.append({
            'da_px': lo, 'a_px': hi, 'n_affidabili': int(sel.sum()),
            'n_misurate': int(((c[:, 2] >= lo) & ((c[:, 2] < hi) if hi is not None
                                                  else True)).sum()),
            'mediana_px': float(np.median(m)) if len(m) else None,
            'p90_px': float(np.percentile(m, 90)) if len(m) else None,
            'quota_sopra_1px': float((m > 1).mean()) if len(m) else None,
            'quota_sopra_2px': float((m > 2).mean()) if len(m) else None,
            'quota_sopra_5px': float((m > 5).mean()) if len(m) else None,
        })
    ordine = np.argsort(mag)[::-1][:5]
    peggiori = [{'x': int(buone[i, 0]), 'y': int(buone[i, 1]),
                 'distanza_px': float(buone[i, 2]), 'residuo_px': float(mag[i]),
                 'confidenza': float(buone[i, 5])} for i in ordine]
    out = {'maschera': mask is not None, 'lato_finestra': lato, 'fasce': fasce,
           'peggiori': peggiori, 'campioni': buone.tolist(),
           'campioni_deboli': c[c[:, 5] < min_conf].tolist()}

    if verbose:
        rif = "contorno della maschera" if mask is not None else \
              "bordo dell'immagine (nessuna maschera: le fasce esterne possono cadere sullo sfondo)"
        print(f"  residuo per distanza dal {rif}, finestre da {lato} px:")
        for f in fasce:
            et = f"{f['da_px']}-{f['a_px']} px" if f['a_px'] else f"oltre {f['da_px']} px"
            if f['n_affidabili']:
                print(f"    {et:<14} n={f['n_affidabili']:3d}  mediana {f['mediana_px']:5.2f}  "
                      f"p90 {f['p90_px']:5.2f}   oltre 2 px {100 * f['quota_sopra_2px']:3.0f}%  "
                      f"oltre 5 px {100 * f['quota_sopra_5px']:3.0f}%")
            else:
                print(f"    {et:<14} nessuna finestra affidabile ({f['n_misurate']} misurate)")
        if peggiori:
            p = peggiori[0]
            print(f"    peggiore: ({p['x']},{p['y']}) a {p['distanza_px']:.0f} px dal contorno, "
                  f"{p['residuo_px']:.2f} px")
    return out


def zone_critiche(dopo, bordo=None, soglia_residuo=2.0,
                   conf_affidabile=0.30, distanza=900.0, min_finestre=2,
                   con_maschera=False, max_stampate=6, verbose=True):
    """
    Raggruppa le finestre problematiche in ZONE, distinguendo due casi che
    chiedono rimedi opposti.

    - "non misurabile": la correlazione non aggancia nulla, quindi li' non
      sappiamo se l'allineamento e' buono o pessimo. NON e' un elenco di posti
      dove mettere punti a mano: dove la correlazione non aggancia, di solito
      non aggancia neanche l'occhio, perche' le due riprese mostrano oggetti
      fisici diversi (nella radiografia biacca, tela e telaio al posto della
      pittura). Un punto messo li' senza riconoscere davvero lo stesso
      dettaglio non aggiunge informazione, la inventa, e peggiora il risultato.
    - "non seguito": la misura c'e' ed e' sicura, ma il residuo resta alto.
      Qui il problema non e' la misura, e' il modello: un punto manuale NON
      aiuta.

    E' il primo passo dello strumento di annotazione: elencare le zone, prima
    di costruire qualunque interfaccia per cliccarle.

    Returns:
        lista di zone, dalla piu' grave; ognuna con tipo, riquadro, numero di
        finestre e che cosa converrebbe farci.
    """
    punti = []          # (x, y, residuo|None, tipo)
    for c in (dopo or {}).get('campioni', []):
        r = float(np.hypot(c[2], c[3]))
        if c[4] >= conf_affidabile and r > soglia_residuo:
            punti.append((c[0], c[1], r, 'non seguito'))
    for c in (bordo or {}).get('campioni', []):
        r = float(np.hypot(c[3], c[4]))
        if c[5] >= conf_affidabile and r > soglia_residuo:
            punti.append((c[0], c[1], r, 'non seguito'))
    for c in (dopo or {}).get('campioni_deboli', []):
        punti.append((c[0], c[1], None, 'non misurabile'))
    for c in (bordo or {}).get('campioni_deboli', []):
        punti.append((c[0], c[1], None, 'non misurabile'))
    if not punti:
        if verbose:
            print("  nessuna zona critica: tutte le finestre sono misurate e in tolleranza")
        return []

    # Raggruppamento per vicinanza, separatamente per tipo: pochi punti, quindi
    # un confronto di tutti con tutti basta e avanza.
    zone = []
    for tipo in ('non seguito', 'non misurabile'):
        gruppo = [p for p in punti if p[3] == tipo]
        etichetta = [-1] * len(gruppo)
        n = 0
        for i in range(len(gruppo)):
            if etichetta[i] >= 0:
                continue
            etichetta[i] = n
            coda = [i]
            while coda:
                j = coda.pop()
                for k in range(len(gruppo)):
                    if etichetta[k] < 0 and np.hypot(gruppo[j][0] - gruppo[k][0],
                                                     gruppo[j][1] - gruppo[k][1]) <= distanza:
                        etichetta[k] = n
                        coda.append(k)
            n += 1
        for g in range(n):
            membri = [p for p, e in zip(gruppo, etichetta) if e == g]
            if len(membri) < min_finestre:
                continue
            xs = [m[0] for m in membri]
            ys = [m[1] for m in membri]
            residui = [m[2] for m in membri if m[2] is not None]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            if tipo == 'non misurabile':
                # Non e' un invito a metterci un punto, ed era un errore
                # presentarlo cosi'. Dove la correlazione non aggancia di
                # solito non aggancia neanche l'occhio, perche' le due immagini
                # mostrano cose fisicamente diverse: nella radiografia quella
                # zona e' biacca, tela e telaio, non la pittura che si vede nel
                # visibile. Un punto messo li' "a sentimento" non aggiunge
                # informazione, ne inventa: cinque punti di quel genere hanno
                # portato il residuo da 1.00 a 6.35 px, peggio che non
                # metterne affatto.
                nota = ("qui la verifica e' cieca: le due immagini non condividono "
                        "struttura, quindi non si sa se l'allineamento sia buono o "
                        "pessimo. NON metterci un punto se non riconosci con certezza "
                        "lo stesso dettaglio in entrambe: un punto inventato fa piu' "
                        "danno che nessun punto")
            else:
                nota = ("misura sicura che il campo non riesce a seguire: da guardare, "
                        "un punto manuale non aggiungerebbe nulla")
            zone.append({
                'tipo': tipo,
                'centro_px': [round(cx), round(cy)],
                'riquadro_px': [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))],
                'n_finestre': len(membri),
                'residuo_mediano_px': round(float(np.median(residui)), 2) if residui else None,
                'nota': nota,
            })
    zone.sort(key=lambda z: (z['tipo'] != 'non seguito',
                             -(z['residuo_mediano_px'] or 0), -z['n_finestre']))
    if verbose:
        n_seguito = sum(1 for z in zone if z['tipo'] == 'non seguito')
        print(f"  zone da rivedere: {len(zone)} ({n_seguito} misurate ma non seguite, "
              f"{len(zone) - n_seguito} non misurabili)")
        for z in zone[:max_stampate]:
            r = (f", residuo mediano {z['residuo_mediano_px']:.1f} px"
                 if z['residuo_mediano_px'] else "")
            print(f"    [{z['tipo']}] centro ({z['centro_px'][0]}, {z['centro_px'][1]}), "
                  f"{z['n_finestre']} finestre{r}")
            print(f"        {z['nota']}")
        if len(zone) > max_stampate:
            print(f"    ... e altre {len(zone) - max_stampate}, tutte nel report")
        if not con_maschera and any(z['tipo'] == 'non misurabile' for z in zone):
            print("    [nota] senza maschera molte zone 'non misurabile' sono "
                  "semplicemente sfondo: disegna la zona del dipinto per non contarle")
    return zone


def mind_quality(fixed_arr, moving_arr, transform, patch=768, n_side=3, verbose=True):
    """
    Costo MIND medio (metrica robusta cross-modale, invariante all'inversione di
    contrasto) su alcuni campioni a piena risoluzione, prima e dopo la
    trasformazione. Piu' basso = meglio allineato.

    Si valuta su campioni e non sull'immagine intera proprio per il motivo che
    faceva crashare la versione originale: un descriptor MIND a 15848x19850
    occuperebbe ~10 GB (8 canali float32), mentre su un campione da 768px sono
    18 MB.
    """
    H, W = fixed_arr.shape[:2]
    half = patch // 2
    ys = np.linspace(H * 0.2, H * 0.8, n_side).astype(int)
    xs = np.linspace(W * 0.2, W * 0.8, n_side).astype(int)

    # Il confronto "prima" ha senso solo se le due immagini hanno la stessa
    # dimensione: altrimenti la stessa finestra di coordinate inquadra zone
    # diverse del dipinto, e il valore di partenza non e' un termine di
    # paragone ma un numero a caso.
    confrontabile = tuple(moving_arr.shape[:2]) == tuple(fixed_arr.shape[:2])

    before, after = [], []
    for cy in ys:
        for cx in xs:
            y0, x0 = int(cy) - half, int(cx) - half
            if y0 < 0 or x0 < 0 or y0 + patch > H or x0 + patch > W:
                continue
            f_patch = np.asarray(fixed_arr[y0:y0 + patch, x0:x0 + patch])
            m_warp = transform.warp_tile_from(moving_arr, y0, x0, patch, patch)
            df = RobustMetrics.mind(f_patch)
            after.append(float(np.mean((df - RobustMetrics.mind(m_warp)) ** 2)))

            if confrontabile:
                m_raw = np.asarray(moving_arr[y0:y0 + patch, x0:x0 + patch])
                if m_raw.shape[:2] == f_patch.shape[:2]:
                    before.append(float(np.mean((df - RobustMetrics.mind(m_raw)) ** 2)))

    res = {'mind_prima': float(np.mean(before)) if before else None,
           'mind_dopo': float(np.mean(after)) if after else None,
           'n_campioni': len(after),
           'confrontabile_con_prima': bool(confrontabile and before)}
    if verbose and after:
        if res['mind_prima'] is not None:
            gain = 100.0 * (1 - res['mind_dopo'] / max(res['mind_prima'], 1e-12))
            res['guadagno_pct'] = float(gain)
            print(f"  costo MIND su {len(after)} campioni da {patch}px: "
                  f"{res['mind_prima']:.6f} -> {res['mind_dopo']:.6f} ({gain:+.1f}%)")
        else:
            print(f"  costo MIND dopo l'allineamento: {res['mind_dopo']:.6f} "
                  f"(nessun confronto col 'prima': le due immagini hanno dimensioni diverse)")
    return res


def make_overlay(fixed_small, moving_small_warped, path, mode='checker', tile=64,
                 max_side=1800):
    """
    Immagine di controllo visivo: scacchiera fra fixed e moving allineata
    (i bordi devono proseguire senza scalini fra un riquadro e l'altro), oppure
    falso colore (fixed nel canale rosso, moving nel verde: dove sono allineate
    l'immagine risulta grigia/gialla, dove non lo sono compaiono frange
    rosse e verdi).
    """
    # a_8bit e non .astype(np.uint8): su una ripresa a 16 bit il cast diretto
    # taglia i valori modulo 256 e l'immagine di controllo diventa illeggibile
    # (misurato: correlazione -0.01 con la versione corretta, cioe' rumore).
    a = image_io.a_8bit(fixed_small)
    b = image_io.a_8bit(moving_small_warped)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]

    # Queste immagini servono all'occhio, non alla misura: oltre ~1800 px di
    # lato non aggiungono informazione e costano memoria (a fattore 2
    # l'anteprima e' 79 Mpx: una sola maschera a scacchiera in int64 sarebbe
    # 630 MB, ed e' cosi' che il picco di memoria era salito a 3.8 GB).
    scale = max_side / max(h, w)
    if scale < 1.0:
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        tile = max(8, int(tile * scale))

    if mode == 'checker':
        mask = (((np.arange(h, dtype=np.int32) // tile)[:, None] +
                 (np.arange(w, dtype=np.int32) // tile)[None, :]) % 2).astype(bool)
        out = np.where(mask, a, b)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:, :, 2] = a   # rosso  (BGR)
        out[:, :, 1] = b   # verde
    image_io.scrivi_immagine(path, out)
    return path
