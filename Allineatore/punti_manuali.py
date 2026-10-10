"""
Punti di corrispondenza messi a mano.

A che cosa servono, e a che cosa NO. Il matching automatico misura lo
spostamento su decine di migliaia di blocchi con precisione di qualche decimo
di pixel; un clic umano arriva a 1-2 px. Quindi i punti manuali non migliorano
la precisione dove l'automatico funziona: la peggiorerebbero.

Servono dove l'automatico non parte. Sulla radiografia di questo progetto, dei
3300 blocchi misurati ne sopravvivono 61-83, cioe' il 2-3% (sulle altre bande
l'86%). Il motivo non e' che la radiografia sia incomprensibile, ma che i
blocchi cercano nel posto sbagliato: il pre-allineamento automatico sa fare
solo TRASLAZIONI, mentre fra visibile e radiografia resta una scala del 12% e
una rotazione di 0.3 gradi. Chi sopravvive e' chi per caso cadeva gia' vicino.

Pochi punti messi a mano danno la posa di partenza giusta; da li' in poi lavora
tutto l'automatico, che e' piu' preciso di qualunque clic. E' per questo che ne
bastano 4-8 e non servono decine.

Sul modello: con pochi punti un'affine completa (6 gradi di liberta') insegue
l'imprecisione dei clic, perche' con 3 punti ci passa esattamente dentro. Sotto
i 6 punti si adatta quindi una SIMILARITA' (scala, rotazione, traslazione: 4
gradi), che e' proprio cio' che manca al pre-allineamento; da 6 punti in su si
passa all'affine, che aggiunge anche shear e scala anisotropa.
"""

import json
import os

import numpy as np
import cv2


def affine_da_punti(punti, modello='auto', n_iter=3, sigma_reject=3.0):
    """
    Stima la trasformazione fixed -> moving dai punti di corrispondenza.

    Args:
        punti: (N, 4) con [x_fixed, y_fixed, x_moving, y_moving], in pixel a
            piena risoluzione di ciascuna immagine.
        modello: 'similarita', 'affine' oppure 'auto' (similarita' sotto i 6
            punti, affine da 6 in su).

    Returns:
        (M 2x3, residui per punto in px, modello usato)
    """
    p = np.asarray(punti, dtype=np.float64).reshape(-1, 4)
    if len(p) < 3:
        raise ValueError(f"Servono almeno 3 punti, ne sono stati dati {len(p)}.")
    if modello == 'auto':
        modello = 'affine' if len(p) >= 6 else 'similarita'

    tieni = np.ones(len(p), dtype=bool)
    for _ in range(max(1, n_iter)):
        M = _adatta(p[tieni], modello)
        res = _residui(p, M)
        if len(p) < 5:
            break
        # scarto robusto: un clic finito sul dettaglio sbagliato va escluso,
        # altrimenti trascina la posa di partenza di tutti gli altri
        med = np.median(res)
        mad = np.median(np.abs(res - med)) * 1.4826 + 1e-9
        nuovo = res <= med + sigma_reject * mad
        if nuovo.sum() < max(3, int(0.5 * len(p))) or (nuovo == tieni).all():
            break
        tieni = nuovo
    M = _adatta(p[tieni], modello)
    return M, _residui(p, M), modello


def _adatta(p, modello):
    x, y, u, v = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    if modello == 'affine':
        A = np.stack([x, y, np.ones_like(x)], axis=1)
        cu, *_ = np.linalg.lstsq(A, u, rcond=None)
        cv_, *_ = np.linalg.lstsq(A, v, rcond=None)
        return np.stack([cu, cv_], axis=0)
    # Similarita': u = a*x - b*y + c ; v = b*x + a*y + d
    n = len(p)
    A = np.zeros((2 * n, 4))
    A[0::2] = np.stack([x, -y, np.ones(n), np.zeros(n)], axis=1)
    A[1::2] = np.stack([y, x, np.zeros(n), np.ones(n)], axis=1)
    b = np.empty(2 * n)
    b[0::2], b[1::2] = u, v
    (a, bb, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[a, -bb, c], [bb, a, d]])


def _residui(p, M):
    pred = np.stack([p[:, 0], p[:, 1], np.ones(len(p))], axis=1) @ M.T
    return np.hypot(pred[:, 0] - p[:, 2], pred[:, 1] - p[:, 3])


def controlla_punti(punti, k=4, soglia_min=25.0, sigma=3.0):
    """
    Quali punti sono davvero sbagliati, e quali stanno solo descrivendo la
    deformazione della tela.

    Confrontare ogni punto con il modello globale non basta, e su questi dati
    induce in errore. Sulla radiografia di prova 47 punti messi a mano davano
    19.3 px di scarto mediano dall'affine, e la reazione naturale e' pensare che
    i clic siano imprecisi. Misurando si e' visto il contrario: lo scarto di un
    punto assomiglia a quello dei punti VICINI molto piu' che a quello di punti
    presi a caso (10.3 px di differenza contro 19.2), e il suo rms (26.4 px)
    coincide con quello del campo non rigido che il programma stima da solo su
    quella coppia (19.0 px). I punti stavano cioe' descrivendo una deformazione
    vera, che un'affine a sei parametri non puo' rappresentare: erano buoni, non
    cattivi, e insistere a "rifare i piu' incoerenti" non porta da nessuna parte.

    Un punto e' quindi sospetto quando litiga con i VICINI, non con il modello
    globale. E' lo stesso criterio con cui block_matching riammette i blocchi
    scartati (blocchi_coerenti): una deformazione e' liscia nello spazio, un
    errore no.

    Args:
        k: quanti vicini guardare.
        soglia_min: sotto questo scarto locale (px) non si segnala nulla, per
            non inseguire il rumore quando i punti sono gia' coerenti.

    Returns:
        dict con:
          M, modello, residui      come affine_da_punti
          locali                   scarto di ogni punto dai suoi vicini, px
          sospetti                 array di bool, True = da rivedere
          soglia                   la soglia usata sullo scarto locale
    """
    p = np.asarray(punti, dtype=np.float64).reshape(-1, 4)
    M, res, modello = affine_da_punti(p)
    fuori = dict(M=M, modello=modello, residui=res,
                 locali=np.zeros(len(p)), sospetti=np.zeros(len(p), bool),
                 soglia=float('inf'))
    if len(p) < k + 2:
        # Troppo pochi per avere dei "vicini": con quattro punti il vicino e'
        # mezza immagine piu' in la', e il confronto non vorrebbe dire niente.
        return fuori

    pred = np.stack([p[:, 0], p[:, 1], np.ones(len(p))], axis=1) @ M.T
    v = p[:, 2:4] - pred                       # scarto come vettore
    xy = p[:, :2]
    locali = np.empty(len(p))
    for i in range(len(p)):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        vicini = np.argsort(d)[1:k + 1]
        atteso = np.median(v[vicini], axis=0)
        locali[i] = np.hypot(*(v[i] - atteso))

    med = np.median(locali)
    mad = np.median(np.abs(locali - med)) * 1.4826
    soglia = max(soglia_min, med + sigma * mad)
    fuori.update(locali=locali, sospetti=locali > soglia, soglia=float(soglia))
    return fuori


def descrivi(M, res, modello):
    """Riga di riepilogo leggibile, con il residuo dei punti stessi.

    Il residuo dei punti NON e' la precisione dell'allineamento: misura solo
    quanto i clic sono coerenti fra loro. La precisione vera la da' la verifica
    a piena risoluzione, come sempre.
    """
    scala = float(np.hypot(M[0, 0], M[1, 0]))
    ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return (f"{len(res)} punti, modello {modello}: scala {scala:.5f}, "
            f"rotazione {ang:+.3f} gradi, traslazione ({M[0, 2]:+.0f}, {M[1, 2]:+.0f}) px; "
            f"scarto dei punti mediano {np.median(res):.1f} px, max {res.max():.1f} px")


def _tps_adatta(xy, v, lam=0.0):
    """Thin-plate spline: la superficie piu' liscia che passa per i punti dati.

    Con lam = 0 ci passa esattamente (il "punto fermo"); con lam > 0 si lascia
    smussare. Le coordinate vanno normalizzate prima di arrivare qui, altrimenti
    il sistema su un'immagine da 20000 px e' malcondizionato.
    """
    n = len(xy)
    d = np.hypot(xy[:, 0:1] - xy[None, :, 0], xy[:, 1:2] - xy[None, :, 1])
    K = np.where(d > 1e-9, d ** 2 * np.log(np.maximum(d, 1e-9)), 0.0) + lam * np.eye(n)
    Pm = np.column_stack([np.ones(n), xy])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, Pm, Pm.T
    B = np.zeros((n + 3, v.shape[1]))
    B[:n] = v
    return np.linalg.lstsq(A, B, rcond=None)[0]


def _tps_valuta(sol, base, xy):
    n = len(base)
    d = np.hypot(xy[:, 0:1] - base[None, :, 0], xy[:, 1:2] - base[None, :, 1])
    K = np.where(d > 1e-9, d ** 2 * np.log(np.maximum(d, 1e-9)), 0.0)
    return K @ sol[:n] + np.column_stack([np.ones(len(xy)), xy]) @ sol[n:]


def _distanza_da(posizioni, forma, passo):
    """Distanza, in px, dal piu' vicino fra `posizioni`, su una griglia."""
    gh, gw = forma
    m = np.ones((gh, gw), np.uint8)
    for x, y in posizioni:
        m[min(gh - 1, max(0, int(y / passo))), min(gw - 1, max(0, int(x / passo)))] = 0
    if m.all():
        return np.full((gh, gw), 1e9, np.float32)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5) * passo


def campo_dai_punti(punti, residui_vettore, forma, passo_griglia, misure,
                    dist_misure=1000.0, dist_punti=700.0, lam=0.0):
    """
    Il campo che fa passare la trasformazione ESATTAMENTE per i punti messi a
    mano, ma solo dove nessuno sta misurando.

    Fino a poco fa i punti servivano unicamente a dare la posa di partenza e poi
    venivano dimenticati: misurato su una corsa reale, la trasformazione finale
    cadeva a 10 px di mediana (p90 42 px) da dove l'utente aveva cliccato. Sul
    quarto destro di quella radiografia, dove NESSUNA delle 16 finestre di
    verifica e' misurabile, lo scarto arrivava a 31 px e il campo automatico non
    poteva saperlo: li' non ha dati e si limita a estendere quello del centro.

    Perche' interpolare e non smussare. Costruire dai punti un campo smussato
    (media pesata, come build_dense_field) non funziona: quello media i clic
    senza passarci dentro, e si porta dietro il loro errore senza ottenere il
    vincolo. Provato, peggiorava. L'interpolazione esatta invece regge: in
    leave-one-out, togliendo un punto e prevedendolo con gli altri, sbaglia
    8.7 px contro i 10.2 del non fare niente, e sul lato critico 17.7 contro
    31.3. Descrive quindi una deformazione vera, non insegue rumore.

    Perche' solo dove non si misura. Dove il matching automatico lavora, misura
    a frazioni di pixel, mentre un clic ne sbaglia 5-10: sovrapporgli i punti
    sarebbe un danno. Misurato su tre corse reali, con questo vincolo il residuo
    sulle finestre misurabili in entrambe le versioni resta IDENTICO alla
    seconda cifra (1.77 -> 1.77, 3.02 -> 3.02, 3.71 -> 3.71 px), nessuna
    finestra peggiora, e le finestre che diventano misurabili passano da 18 a
    20, da 19 a 22 e da 19 a 26. Quest'ultimo numero e' la prova piu' forte:
    sono misure indipendenti, che non sanno nulla dei punti, e agganciano dove
    prima non agganciavano.

    Perche' un limite STRETTO di distanza dai punti. Fra un punto e l'altro la
    spline non prevede niente: lo si e' misurato adattandola su una parte dei
    punti e guardando l'errore sugli altri, in funzione di quanto distavano dal
    punto usato piu' vicino (due serie reali, 41 e 50 punti; il confronto e' con
    il non applicare nessuna correzione, che vale 8.9 e 9.4 px):

        distanza dal punto      41 punti      50 punti
            0 -  700 px          6.6 px        4.6 px     <- meglio di niente
          700 - 1400 px          6.7 px       10.9 px
         1400 - 2500 px         16.6 px       12.1 px     <- peggio
         2500 - 4000 px         20.2 px       34.2 px
              oltre             51.1 px       98.6 px

    Oltre il raggio la correzione non aggiunge informazione, la inventa, e fa
    peggio dell'estrapolazione automatica che sostituisce. Il limite sta percio'
    a 700 px, con dissolvenza a zero entro 1400: si corregge l'intorno di ogni
    punto, dove c'e' una prova, e si lascia stare il resto.

    Serve anche a contenere la spline: fuori dall'inviluppo dei punti cresce
    come r^2 log r, e senza limite arrivava a 125 px nei quattro angoli del
    dipinto, dove non c'e' ne' una misura ne' un clic a sostenerla.

    Args:
        punti: (N, 4) coordinate a piena risoluzione [x_f, y_f, x_m, y_m].
        residui_vettore: (N, 2) quanto manca in ogni punto, come vettore, dopo
            la trasformazione stimata.
        forma: (H, W) dell'immagine di riferimento.
        passo_griglia: passo del campo di uscita, in px.
        misure: (M, 2) posizioni delle finestre di verifica RIUSCITE.
        dist_misure: sotto questa distanza da una misura il campo vale 0, e a
            due volte tanto vale 1.
        dist_punti: entro questa distanza da un punto il campo vale pieno, e a
            due volte tanto e' gia' spento. Va tenuto piccolo: vedi sopra.

    Returns:
        (campo (Gh, Gw, 2) float32, info dict)
    """
    P = np.asarray(punti, dtype=np.float64).reshape(-1, 4)
    V = np.asarray(residui_vettore, dtype=np.float64).reshape(-1, 2)
    H, W = forma
    gh = int(np.ceil(H / passo_griglia))
    gw = int(np.ceil(W / passo_griglia))
    if len(P) < 3:
        return np.zeros((gh, gw, 2), np.float32), {'n_usati': 0, 'motivo': 'troppo pochi punti'}

    # I punti incoerenti vanno esclusi, non solo segnalati: qui diventano
    # vincoli ESATTI, quindi un clic sbagliato verrebbe inciso nel risultato
    # invece di essere diluito fra gli altri come accadeva nell'affine.
    ctrl = controlla_punti(P)
    tieni = ~ctrl['sospetti']
    if tieni.sum() < 3:
        tieni = np.ones(len(P), bool)
    Pk, Vk = P[tieni], V[tieni]

    scala = 1000.0            # normalizzazione, per non malcondizionare il sistema
    sol = _tps_adatta(Pk[:, :2] / scala, Vk, lam)
    gy, gx = np.mgrid[0:gh, 0:gw]
    nodi = np.column_stack([(gx.ravel() * passo_griglia) / scala,
                            (gy.ravel() * passo_griglia) / scala])
    campo = _tps_valuta(sol, Pk[:, :2] / scala, nodi).reshape(gh, gw, 2)

    d_mis = _distanza_da(np.asarray(misure).reshape(-1, 2)[:, :2], (gh, gw), passo_griglia)
    d_pun = _distanza_da(Pk[:, :2], (gh, gw), passo_griglia)
    peso = (np.clip((d_mis - dist_misure) / dist_misure, 0.0, 1.0)
            * np.clip(1.0 - (d_pun - dist_punti) / dist_punti, 0.0, 1.0))
    campo = (campo * peso[:, :, None]).astype(np.float32)

    amp = np.hypot(campo[..., 0], campo[..., 1])
    attiva = peso > 0.05
    return campo, {
        'n_punti': int(len(P)),
        'n_usati': int(tieni.sum()),
        'n_esclusi': int((~tieni).sum()),
        'esclusi': [int(i) + 1 for i in np.flatnonzero(~tieni)],
        # serve alla diagnostica per disegnare in verde i vincoli usati e in
        # rosso quelli scartati; non va nel JSON, chi la usa la toglie
        '_usati': tieni,
        'quota_superficie': float(attiva.mean()),
        'correzione_mediana_px': float(np.median(amp[attiva])) if attiva.any() else 0.0,
        'correzione_max_px': float(amp.max()),
    }


def contorno_da_maschera(maschera, forma_piena, passo_px=20.0):
    """Il bordo di una maschera disegnata, in pixel dell'immagine piena.

    Si campiona un punto ogni `passo_px` invece di tenerli tutti: il contorno
    di un ottagono da 20000 px ha decine di migliaia di pixel di bordo, che non
    aggiungono niente a una curva liscia e fanno solo lavoro.
    """
    m = (np.asarray(maschera) > 127).astype(np.uint8)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not c:
        return None
    grande = max(c, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    H, W = forma_piena
    grande[:, 0] *= W / float(m.shape[1])
    grande[:, 1] *= H / float(m.shape[0])
    lung = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(grande, axis=0), axis=1))])
    if lung[-1] <= 0:
        return grande
    n = max(16, int(lung[-1] / max(passo_px, 1.0)))
    voluti = np.linspace(0, lung[-1], n, endpoint=False)
    idx = np.searchsorted(lung, voluti).clip(0, len(grande) - 1)
    return grande[np.unique(idx)]


def _ricampiona_chiuso(C, passo_px):
    """Un contorno chiuso con un punto ogni `passo_px` circa, lungo la curva."""
    C = np.asarray(C, dtype=np.float64).reshape(-1, 2)
    chiuso = np.vstack([C, C[:1]])
    lung = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(chiuso, axis=0), axis=1))])
    if lung[-1] <= 0:
        return C
    voluti = np.arange(0.0, lung[-1], max(passo_px, 0.5))
    return np.column_stack([np.interp(voluti, lung, chiuso[:, 0]),
                            np.interp(voluti, lung, chiuso[:, 1])])


def normali_esterne(G):
    """Normale di ogni punto di un contorno chiuso, rivolta verso l'esterno."""
    G = np.asarray(G, dtype=np.float64).reshape(-1, 2)
    k = max(3, len(G) // 160)
    t = np.roll(G, -k, axis=0) - np.roll(G, k, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    # Il verso si decide dall'area con segno: per un contorno percorso in un
    # senso la normale (-ty, tx) guarda fuori, per l'altro dentro (con y verso
    # il basso, come nelle immagini).
    x, y = G[:, 0], G[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return n if area < 0 else -n


def posa_dai_contorni(G, C, M0=None, iterazioni=50, scarta=0.10):
    """
    L'affine riferimento -> immagine da allineare che sovrappone i due contorni.

    E' il "bordo prima": la posa si ricava soltanto dalle due linee, il contorno
    della zona sul riferimento (G) e il bordo tracciato sull'immagine da
    allineare (C), senza guardare l'interno. Si itera il punto piu' vicino
    (ICP): ogni punto di G, portato con la posa corrente, si accoppia al punto
    di C piu' vicino, e si rifa' l'affine ai minimi quadrati; il 10% di coppie
    piu' lontane si scarta a ogni giro, per non farsi tirare da un tratto
    tracciato male.

    Args:
        M0: posa di partenza (2x3); None = quella che sovrappone i due
            rettangoli che contengono i contorni.

    Returns:
        (M 2x3, info dict)
    """
    from scipy.spatial import cKDTree
    G = np.asarray(G, dtype=np.float64).reshape(-1, 2)
    Cd = _ricampiona_chiuso(C, 5.0)
    partenza = 'data'
    if M0 is None:
        partenza = 'rettangoli'
        g0, g1 = G.min(axis=0), G.max(axis=0)
        c0, c1 = Cd.min(axis=0), Cd.max(axis=0)
        s = (c1 - c0) / np.maximum(g1 - g0, 1e-9)
        M0 = np.array([[s[0], 0.0, c0[0] - s[0] * g0[0]],
                       [0.0, s[1], c0[1] - s[1] * g0[1]]])
    M = np.asarray(M0, dtype=np.float64).copy()
    albero = cKDTree(Cd)
    A = np.column_stack([G, np.ones(len(G))])
    it = 0
    for it in range(iterazioni):
        d, j = albero.query(A @ M.T)
        tieni = d <= np.quantile(d, 1.0 - scarta)
        sol, *_ = np.linalg.lstsq(A[tieni], Cd[j[tieni]], rcond=None)
        spost = np.abs(A @ (sol.T - M).T).max()
        M = sol.T
        if spost < 0.05:
            break
    d = albero.query(A @ M.T)[0]
    return M, {'iterazioni': it + 1, 'partenza': partenza,
               'scarto_mediano_px': float(np.median(d)),
               'scarto_p90_px': float(np.percentile(d, 90))}


def campo_fascia_di_bordo(G, C, porta, forma, passo_griglia, larghezza=150.0,
                          liscia_px=300.0):
    """
    Il campo che porta il contorno del riferimento sul bordo dell'immagine da
    allineare, agendo solo in una fascia lungo il bordo.

    Sulla radiografia della Cappellochiaro i dettagli cliccati a piu' di 100 px
    dal bordo sono a posto (scarto perpendicolare al bordo fra -0.5 e 2.4 px),
    mentre quelli sul bordo stanno 59 px piu' fuori, quanto la distanza fra le
    due linee: la deformazione sta in una fascia sottile. Il vincolo quindi non
    si spalma sull'interno (il vincolo di bordo precedente, con 700 px di
    portata, spostava i dettagli a 250-1000 px dal bordo di 28-45 px), ma
    agisce per intero sul contorno e fuori, e si spegne entro `larghezza` px
    verso l'interno con un raccordo a coseno.

    Si usa la sola componente perpendicolare al contorno (lungo un lato dritto
    il punto piu' vicino scivola, e la componente tangente e' rumore), lisciata
    con una mediana mobile di `liscia_px` per parte lungo il contorno, cosi'
    che un tratto tracciato male non diventi un'onda nel campo.

    Quando il bordo dell'immagine da allineare sta fuori (il caso della
    radiografia) la corrispondenza lungo la normale resta monotona per
    qualsiasi larghezza: la fascia si comprime, ma non si ripiega. Quando sta
    dentro, lo spostamento si limita a 0.6 volte la larghezza, sotto il limite
    oltre il quale si ripiegherebbe.

    Args:
        G: (N, 2) contorno della zona sul riferimento, px a piena risoluzione.
        C: (M, 2) bordo tracciato sull'immagine da allineare, nei suoi px.
        porta: f(xy) -> xy secondo la trasformazione gia' stimata.
        passo_griglia: passo del campo in px; deve stare sotto un terzo della
            larghezza perche' il raccordo sia rappresentato.

    Returns:
        (campo (Gh, Gw, 2) float32 da sommare al campo della trasformazione,
         info dict)
    """
    from scipy.spatial import cKDTree
    H, W = forma
    G = _ricampiona_chiuso(G, 10.0)
    if len(G) < 16 or len(np.asarray(C).reshape(-1, 2)) < 8:
        return None, {'motivo': 'contorni troppo corti'}
    Cd = _ricampiona_chiuso(C, 5.0)
    n = normali_esterne(G)

    P = porta(G)
    j = cKDTree(Cd).query(P)[1]
    perp = ((Cd[j] - P) * n).sum(axis=1)
    k = max(1, int(round(liscia_px / 10.0)))
    giro = np.concatenate([perp[-k:], perp, perp[:k]])
    perp_l = np.array([np.median(giro[i:i + 2 * k + 1]) for i in range(len(perp))])
    perp_l = np.maximum(perp_l, -0.6 * larghezza)
    V = n * perp_l[:, None]

    gh = int(np.ceil(H / passo_griglia)) + 1
    gw = int(np.ceil(W / passo_griglia)) + 1
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float64) * passo_griglia
    jn = cKDTree(G).query(np.column_stack([xx.ravel(), yy.ravel()]))[1]
    # Distanza dal contorno con segno (positiva dentro), da un raster a 4 px:
    # piu' fine del passo del campo e molto piu' veloce di un test per nodo.
    r = 4.0
    rh, rw = int(np.ceil(H / r)) + 1, int(np.ceil(W / r)) + 1
    m = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(m, [np.round(G / r).astype(np.int32).reshape(-1, 1, 2)], 1)
    dentro = cv2.distanceTransform(m, cv2.DIST_L2, 5) * r
    fuori = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) * r
    iy = np.clip(np.round(yy / r).astype(int), 0, rh - 1)
    ix = np.clip(np.round(xx / r).astype(int), 0, rw - 1)
    dist = np.where(m[iy, ix] > 0, dentro[iy, ix], -fuori[iy, ix])
    peso = 0.5 * (1.0 + np.cos(np.pi * np.clip(dist, 0, larghezza) / larghezza))
    campo = (V[jn].reshape(gh, gw, 2) * peso[:, :, None]).astype(np.float32)
    amp = np.hypot(campo[..., 0], campo[..., 1])
    return campo, {
        'larghezza_px': float(larghezza),
        'n_punti_contorno': int(len(G)),
        'scarto_bordo_prima_px': float(np.median(np.abs(perp))),
        'scarto_bordo_con_segno_px': float(np.median(perp)),
        'scarto_bordo_p90_px': float(np.percentile(np.abs(perp), 90)),
        'correzione_max_px': float(amp.max()),
        'quota_superficie': float((peso > 0.05).mean()),
    }


def salva(punti, path, extra=None):
    """Salva i punti accanto ai parametri, cosi' la corsa e' ripetibile."""
    dati = {'commento': "punti di corrispondenza [x_fixed, y_fixed, x_moving, "
                        "y_moving] in pixel a piena risoluzione",
            'punti': np.asarray(punti, dtype=float).reshape(-1, 4).tolist()}
    if extra:
        dati.update(extra)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dati, fh, indent=2, ensure_ascii=False)


def carica(path):
    with open(path, encoding='utf-8-sig') as fh:
        dati = json.load(fh)
    return np.asarray(dati['punti'], dtype=float).reshape(-1, 4)


def punti_salvati(cartella_output, immagine):
    """Ripesca i punti dell'ultima corsa su QUESTA coppia, se ce ne sono.

    E' cio' che rende il lavoro riprendibile il giorno dopo: i punti stanno
    accanto ai parametri, con dentro il percorso dell'immagine su cui erano
    stati messi. Quelli di un'altra immagine non vengono raccolti, perche'
    le loro coordinate non c'entrano nulla con questa.
    """
    cartella = os.path.join(cartella_output or '', 'parametri')
    if not os.path.isdir(cartella):
        return None, 0
    atteso = os.path.normcase(os.path.abspath(immagine))
    candidati = sorted((n for n in os.listdir(cartella) if n.endswith('_punti.json')),
                       reverse=True)
    for n in candidati:
        try:
            with open(os.path.join(cartella, n), encoding='utf-8-sig') as fh:
                dati = json.load(fh)
            if os.path.normcase(os.path.abspath(dati.get('moving', ''))) == atteso:
                P = np.asarray(dati['punti'], dtype=float).reshape(-1, 4)
                return P, int(dati.get('n_maniglie') or 0)
        except Exception:
            continue
    return None, 0


def contorno_salvato(cartella_output, immagine):
    """Ripesca il bordo tracciato a mano sull'ultima corsa su QUESTA coppia.

    Stessa logica di punti_salvati, e per lo stesso motivo: tracciare il bordo
    della tela costa minuti di lavoro attento, e ricominciare da capo a ogni
    apertura del programma e' la ragione piu' banale per cui una funzione utile
    smette di essere usata.
    """
    cartella = os.path.join(cartella_output or '', 'parametri')
    if not os.path.isdir(cartella):
        return None
    atteso = os.path.normcase(os.path.abspath(immagine))
    for n in sorted((x for x in os.listdir(cartella) if x.endswith('_contorno.json')),
                    reverse=True):
        try:
            with open(os.path.join(cartella, n), encoding='utf-8-sig') as fh:
                dati = json.load(fh)
            if os.path.normcase(os.path.abspath(dati.get('moving', ''))) == atteso:
                return np.asarray(dati['contorno'], dtype=float).reshape(-1, 2)
        except Exception:
            continue
    return None
