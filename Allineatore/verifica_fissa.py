"""
La verifica a griglia fissa, per tutte le bande.

E' nata nella sezione delle radiografie, dove la mediana della verifica di
sempre nascondeva finestre fuori di 70-100 px. Poi si e' visto che anche le
bande ottiche possono averne bisogno: sull'UV di un affresco da 593 Mpx la
verifica di sempre dava 0.295 px con gli angoli fuori di 61-176 px, perche' le
finestre di bordo non misurabili non comparivano da nessuna parte. Ora si
aggiunge a ogni immagine del lotto, accanto alla verifica di sempre, che resta
il numero principale: vedi align_batch.allinea_lotto.
"""

import os

import numpy as np
import cv2

import image_io
import validation as val

FASCE = (('entro 1000 px dal bordo', 0, 1000), ('fra 1000 e 4000 px', 1000, 4000),
            ('oltre 4000 px', 4000, None))


def verifica_griglia_fissa(riferimento, immagine, T, maschera, rep, cartella_diag,
                           arr_fixed=None):
    """
    Sempre la stessa griglia, e cio' che la mediana nasconde.

    La verifica di sempre ha tre difetti, misurati per la prima volta sulla
    radiografia Cappellochiaro:

    - la griglia adattiva smette di infittirsi appena trova 40 finestre
      misurabili. Un tentativo migliore si ferma quindi prima, su una griglia
      piu' rada, e due tentativi finiscono per avere poche finestre in comune
      (28 su centinaia). Qui la griglia e' sempre la piu' fitta senza finestre
      sovrapposte: con la stessa zona del dipinto, sempre le stesse posizioni;
    - la mediana nasconde le finestre che sbagliano di molto quando sono poche:
      l'automatica aveva tre finestre nel quarto destro a 72, 104 e 97 px, e la
      mediana diceva 1.36 px. Qui le finestre oltre 10 px si elencano e si
      disegnano;
    - lungo il contorno la verifica a finestre piccole a volte non trova
      niente da misurare (0-1 finestre per fascia), e le fasce vuote si
      leggevano come un risultato. Qui si dice quante finestre si misurano
      vicino al bordo, e se il contorno e' cieco.

    Returns:
        dizionario con il riassunto, le finestre
        oltre 10 px, i conteggi per fascia di distanza dal bordo e il percorso
        della mappa.
    """
    arr_f = arr_fixed if arr_fixed is not None else image_io.apri(riferimento, verbose=False)[0]
    arr_m = image_io.apri(immagine, verbose=False)[0]
    v = val.residual_check(arr_f, arr_m, T, mask=maschera, target_affidabili=10 ** 9,
                           verbose=False)
    H, W = arr_f.shape[:2]
    lato = int(v.get('lato_finestra_px') or 512)
    buone = np.asarray(v.get('campioni') or [], np.float64).reshape(-1, 5)
    deboli = np.asarray(v.get('campioni_deboli') or [], np.float64).reshape(-1, 5)
    mag = np.hypot(buone[:, 2], buone[:, 3])

    con_maschera = maschera is not None and bool(np.asarray(maschera).any())
    d_buone = _distanza_dal_bordo(buone[:, :2], maschera if con_maschera else None, H, W)
    d_deboli = _distanza_dal_bordo(deboli[:, :2], maschera if con_maschera else None, H, W)
    fasce = []
    for nome, lo, hi in FASCE:
        sb = (d_buone >= lo) & ((d_buone < hi) if hi is not None else True)
        sd = (d_deboli >= lo) & ((d_deboli < hi) if hi is not None else True)
        fasce.append({'fascia': nome, 'n_campioni': int(sb.sum() + sd.sum()),
                      'n_affidabili': int(sb.sum()),
                      'residuo_mediano_px': float(np.median(mag[sb])) if sb.any() else None})

    # Una finestra fuori di molto da sola e' spesso un aggancio debole; piu'
    # finestre vicine fuori nello stesso verso sono un disallineamento vero.
    # La confidenza non basta a separarle: sulla Cappellochiaro il gruppo vero
    # del lato destro aveva confidenza 0.055-0.11, come gli agganci isolati.
    # E' lo stesso criterio dei blocchi e dei punti: una deformazione e'
    # liscia, un errore di misura no.
    alte = np.flatnonzero(mag > 10.0)
    in_gruppo = {}
    for i in alte:
        d = np.hypot(buone[alte, 0] - buone[i, 0], buone[alte, 1] - buone[i, 1])
        dv = np.hypot(buone[alte, 2] - buone[i, 2], buone[alte, 3] - buone[i, 3])
        in_gruppo[i] = bool(np.any((d > 0) & (d <= 1500.0) & (dv <= 0.5 * mag[i])))
    oltre = [{'x': int(buone[i, 0]), 'y': int(buone[i, 1]), 'residuo_px': float(mag[i]),
              'dx': float(buone[i, 2]), 'dy': float(buone[i, 3]),
              'confidenza': float(buone[i, 4]), 'in_gruppo': in_gruppo[i]}
             for i in sorted(alte, key=lambda k: -mag[k])]

    # Il contorno: la verifica a finestre piccole di align (verifica_bordo)
    # entro 1000 px dal bordo, che sulla RX di solito non misura niente.
    bordo = (rep.get('verifica') or {}).get('bordo') or {}
    vicine = [f for f in bordo.get('fasce') or [] if f.get('a_px') and f['a_px'] <= 1000]
    contorno_cieco = bool(vicine) and sum(f.get('n_affidabili') or 0 for f in vicine) == 0

    os.makedirs(cartella_diag, exist_ok=True)
    mappa = os.path.join(cartella_diag, 'verifica_finestre.png')
    try:
        _mappa_finestre(riferimento, buone, deboli, lato, maschera if con_maschera else None,
                        v, mappa, {(o['x'], o['y']) for o in oltre if o['in_gruppo']})
    except Exception as exc:
        print(f"  [attenzione] mappa delle finestre non disegnata: {exc}")
        mappa = None

    # Le finestre misurabili una per una, [x, y, residuo, confidenza]: con la
    # griglia fissa le stesse posizioni tornano a ogni tentativo, e due
    # tentativi si confrontano sulle finestre che misurano entrambi.
    finestre = [[int(r[0]), int(r[1]), round(float(m), 3), round(float(r[4]), 4)]
                for r, m in zip(buone, mag)]
    escludi = ('campioni', 'campioni_deboli')
    return dict({k: x for k, x in v.items() if k not in escludi},
                griglia='la piu\' fitta senza finestre sovrapposte',
                con_maschera=con_maschera, fasce=fasce, oltre_10px=oltre,
                contorno_cieco=contorno_cieco, mappa=mappa, finestre=finestre)


def _distanza_dal_bordo(xy, maschera, H, W):
    """Distanza di ogni punto dal contorno: della maschera se c'e', altrimenti
    dal bordo dell'immagine. Su una griglia ridotta, come in verifica_bordo."""
    if len(xy) == 0:
        return np.zeros(0)
    rid = max(1, int(round(max(H, W) / 2000)))
    hs, ws = (H + rid - 1) // rid, (W + rid - 1) // rid
    if maschera is not None:
        mk = cv2.resize((np.asarray(maschera) > 0).astype(np.uint8), (ws, hs),
                        interpolation=cv2.INTER_NEAREST)
    else:
        mk = np.ones((hs, ws), np.uint8)
    dist = cv2.distanceTransform(np.pad(mk, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * rid
    iy = np.clip((xy[:, 1] // rid).astype(int), 0, hs - 1)
    ix = np.clip((xy[:, 0] // rid).astype(int), 0, ws - 1)
    return dist[iy, ix]


# Colori della mappa (BGR): una sola tinta, dal chiaro allo scuro, per il residuo
# che resta in tolleranza; i due colori di stato per le finestre da guardare.
_COL_FINESTRE = ((1.0, (239, 182, 134)), (2.0, (229, 135, 57)), (5.0, (171, 92, 28)),
                 (10.0, (90, 131, 236)), (float('inf'), (59, 59, 208)))
_COL_NON_MISURABILE = (129, 135, 137)
_INK = (11, 11, 11)


def _mappa_finestre(riferimento, buone, deboli, lato, maschera, v, path, in_gruppo=()):
    """Le finestre della verifica sul riferimento: dove si misura, quanto, e le
    finestre oltre 10 px con il loro valore scritto accanto (bordo spesso se
    fanno gruppo con altre vicine fuori nello stesso verso, sottile se isolate)."""
    img, passo = image_io.anteprima_veloce(riferimento, 1600, in_grigio=True)
    img = image_io.a_8bit(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    base = (0.40 * img.astype(np.float32) + 0.60 * 247).astype(np.uint8)
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    h2 = lato / 2.0 / passo

    def angoli(r):
        return ((int(r[0] / passo - h2), int(r[1] / passo - h2)),
                (int(r[0] / passo + h2), int(r[1] / passo + h2)))

    riempite = out.copy()
    for r in buone:
        m = float(np.hypot(r[2], r[3]))
        col = next(c for soglia, c in _COL_FINESTRE if m < soglia)
        cv2.rectangle(riempite, *angoli(r), col, -1)
    out = cv2.addWeighted(riempite, 0.75, out, 0.25, 0)
    for r in deboli:
        cv2.rectangle(out, *angoli(r), _COL_NON_MISURABILE, 1)
    if maschera is not None:
        mk = cv2.resize((np.asarray(maschera) > 0).astype(np.uint8), (out.shape[1], out.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (78, 81, 82), 1, cv2.LINE_AA)

    def testo(s, xy, scala=0.5, spessore=1):
        cv2.putText(out, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scala, (255, 255, 255),
                    spessore + 3, cv2.LINE_AA)
        cv2.putText(out, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scala, _INK, spessore, cv2.LINE_AA)

    for r in buone:
        m = float(np.hypot(r[2], r[3]))
        if m > 10.0:
            (x0, y0), (x1, y1) = angoli(r)
            gruppo = (int(r[0]), int(r[1])) in in_gruppo
            cv2.rectangle(out, (x0, y0), (x1, y1), _COL_FINESTRE[-1][1], 3 if gruppo else 1)
            testo(f"{m:.0f}", (x0 + 3, y0 + 16), 0.5, 2 if gruppo else 1)
    med = v.get('residuo_mediano_px')
    testo(f"verifica a griglia fissa: {v.get('n_affidabili', 0)}/{v.get('n_campioni', 0)} finestre "
          f"misurabili, mediana " + (f"{med:.2f} px" if med is not None else "-"), (10, 24), 0.55, 1)
    x = 10
    for etichetta, (_, col) in zip(("< 1 px", "1-2", "2-5", "5-10", "> 10 px"), _COL_FINESTRE):
        cv2.rectangle(out, (x, 36), (x + 14, 50), col, -1)
        testo(etichetta, (x + 18, 49), 0.45, 1)
        x += 30 + 9 * len(etichetta)
    cv2.rectangle(out, (x, 36), (x + 14, 50), _COL_NON_MISURABILE, 1)
    testo("non misurabile", (x + 18, 49), 0.45, 1)
    image_io.scrivi_immagine(path, out)



def avvisi(voce):
    """Cio' che la mediana non dice, in frasi: finestre fuori di molto, bordo
    non verificato, TRE sui punti, fascia di bordo.

    `voce` e' la riga di un'immagine nel riepilogo del lotto. Il caso per cui
    esiste e' una mediana di 1.36 px con tre finestre a 72-104 px nel quarto
    destro: guardando solo la mediana andava tutto bene.
    """
    v = voce.get('verifica_fissa') or {}
    parti = []
    tre = voce.get('tre_punti') or {}
    if tre:
        parti.append(
            f"TRE sui {tre['n']} punti, ciascuno giudicato senza usarlo: mediana "
            f"{tre['mediana_px']:.1f} px, oltre 5 px il {100 * tre['quota_sopra_5px']:.0f}%, "
            f"oltre 10 px il {100 * tre['quota_sopra_10px']:.0f}%. Un clic sbaglia di "
            f"suo ~3 px, quindi sotto i 5 px si e' al limite di cio' che i punti misurano."
            + (" (*) Ottimista: i punti danno anche la posa di partenza."
               if tre.get('ottimistico') else ""))
    dist = v.get('distorsione_px')
    if dist and np.hypot(*dist) > 1.0:
        parti.append(f"Le finestre sono spostate in media di ({dist[0]:+.1f}, {dist[1]:+.1f}) px: "
                     f"un errore sistematico, tutto dalla stessa parte, non casuale.")
    fb = voce.get('fascia_di_bordo') or {}
    if fb.get('applicata'):
        parti.append(f"Bordo: la distanza fra la zona e il bordo tracciato sull'immagine e' "
                     f"passata da {fb['scarto_bordo_prima_px']:.0f} a "
                     f"{fb['scarto_bordo_dopo_px']:.1f} px (mediana lungo il contorno), agendo "
                     f"entro {fb['larghezza_px']:.0f} px dal bordo. Sono gli stessi contorni "
                     f"usati per correggere: non e' una misura indipendente.")
    if v:
        med = v.get('residuo_mediano_px')
        parti.append(f"Griglia fissa: {v.get('n_affidabili', 0)} finestre misurabili su "
                     f"{v.get('n_campioni', 0)}, mediana "
                     + (f"{med:.2f} px." if med is not None else "non misurabile."))
    alte = v.get('oltre_10px') or []
    if alte:
        gruppo = [o for o in alte if o.get('in_gruppo')]
        quante = ("1 finestra misura" if len(alte) == 1 else f"{len(alte)} finestre misurano")
        if gruppo:
            p = gruppo[0]
            parti.append(f"{quante} oltre 10 px, e {len(gruppo)} "
                         f"{'fa' if len(gruppo) == 1 else 'fanno'} gruppo con finestre "
                         f"vicine fuori nello stesso verso: li' l'allineamento e' sbagliato "
                         f"davvero (la peggiore {p['residuo_px']:.0f} px, in x={p['x']}, "
                         f"y={p['y']}). La mediana non lo mostra.")
        else:
            parti.append(f"{quante} oltre 10 px, tutte isolate: probabili agganci deboli, "
                         f"ma da guardare.")
        parti.append("Sono segnate in verifica_finestre.png, nella diagnostica (bordo spesso "
                     "= in gruppo).")
    vicino = next((f for f in v.get('fasce') or [] if f['fascia'].startswith('entro')), None)
    if v and not v.get('con_maschera'):
        parti.append("Senza una zona disegnata la verifica resta nel rettangolo centrale "
                     "e non arriva al bordo: disegnala per misurare anche li'.")
    elif vicino is not None:
        parti.append(f"Entro 1000 px dal bordo misurabili {vicino['n_affidabili']} finestre su "
                     f"{vicino['n_campioni']}"
                     + (", e la verifica lungo il contorno non ne misura nessuna"
                        if v.get('contorno_cieco') else "")
                     + ": li' l'allineamento e' verificato solo dove ci sono finestre.")
    largo = [l for l in (voce.get('livelli_grossolani') or [])[:-1] if l.get('applicato')]
    if largo:
        parti.append(f"Il livello grossolano largo e' stato applicato (deformazione fino a "
                     f"{largo[0].get('ampiezza_max_px_full', 0):.0f} px): e' un'immagine con "
                     f"deformazioni grandi, e se i bordi non tornano conviene tracciarne il "
                     f"bordo negli «Strumenti a mano» e rifarla.")
    return " ".join(parti)


def confronto(precedente, voce):
    """Il tentativo di adesso sulla stessa immagine e' meglio o peggio del
    precedente? Sulle finestre della griglia fissa misurate in ENTRAMBI: il
    residuo di due tentativi non e' confrontabile alla lettera, perche' le
    finestre misurabili cambiano dall'uno all'altro (misurato su una
    radiografia: 1.00 px su 11 finestre contro 2.66 px su 15, con quattro zone
    in piu' che il primo non vedeva)."""
    vp = (precedente or {}).get('verifica_fissa') or {}
    vo = voce.get('verifica_fissa') or {}
    if not (vp.get('finestre') and vo.get('finestre')
            and vp.get('n_campioni') == vo.get('n_campioni')
            and vp.get('con_maschera') == vo.get('con_maschera')):
        return ""
    a = {(f[0], f[1]): f[2] for f in vp['finestre']}
    b = {(f[0], f[1]): f[2] for f in vo['finestre']}
    comuni = sorted(set(a) & set(b))
    if len(comuni) < 5:
        return ""
    va = np.array([a[k] for k in comuni])
    vb = np.array([b[k] for k in comuni])
    return (f"Rispetto al tentativo precedente, sulle {len(comuni)} finestre misurate in "
            f"tutti e due: {np.median(va):.2f} -> {np.median(vb):.2f} px, "
            f"{int((vb < va - 0.2).sum())} migliorate e {int((vb > va + 0.2).sum())} "
            f"peggiorate; misurabili {len(a)} -> {len(b)}.")
