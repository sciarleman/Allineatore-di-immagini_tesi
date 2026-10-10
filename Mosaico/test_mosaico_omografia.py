"""
Test del mosaico da cavalletto su tasselli sintetici con pose NOTE.

    python Mosaico\\test_mosaico_omografia.py

Si fabbrica un "dipinto" sintetico con una cornice dritta e lo si riprende come
farebbe una camera che ruota su un cavalletto: ogni tassello e' il dipinto visto
attraverso una sua omografia (rotazione, scala e prospettiva diverse da scatto a
scatto), con le file sfalsate, uno scatto doppio, esposizione e rumore propri.
I file vengono salvati in ordine casuale: il programma non deve aver bisogno ne'
della griglia ne' del percorso.

Qui la verita' si conosce, quindi si misura cio' che sui tasselli veri non si
puo' misurare: l'errore di posizione VERO, senza togliere nessuna trasformazione
d'insieme. Il mosaico ha per sistema quello del tassello ancora, e la verita' in
quel sistema si calcola esattamente: cio' che resta e' tutto errore, compresa la
deriva che crescerebbe allontanandosi dall'ancora.

Le prove:
1. tasselli con omografie vere: il freno scelto in automatico deve allentarsi, e
   l'errore vero restare sotto il pixel fino ai tasselli piu' lontani;
2. gli stessi tasselli costretti a un'affine (freno infinito): l'errore deve
   salire, a conferma che la prova distingue i due modelli;
3. tasselli da carrello (solo traslazione e una piccola rotazione): il freno
   deve restare tirato, perche' l'omografia li' non guadagna nulla;
4. tasselli con la distorsione dell'obiettivo: va riconosciuta e stimata (e
   l'errore vero agli angoli lontani, pur con cuciture a 0.03 px, arriva a
   qualche pixel: e' il limite del metodo, e il test lo tiene sotto controllo);
5. la rettilineita': la cornice dritta deve risultare dritta con la soluzione
   giusta e storta con quella sbagliata;
6. l'immagine totale: riletta e confrontata col dipinto, finestra per finestra;
7. una camera vera (centro di proiezione, focale, rotazioni) messa fuori asse: il
   mosaico esce visto di sbieco, e dalle linee della cornice si devono ritrovare
   la focale, l'inclinazione dello scatto ancora e la forma vera del dipinto.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cv2

_QUI = os.path.dirname(os.path.abspath(__file__))
if _QUI not in sys.path:
    sys.path.insert(0, _QUI)

import mosaico
import mosaico_omografia as mo
import image_io
import tiff_memmap
import block_matching as bm
from test_mosaico import dipinto_sintetico

LATO = (1200, 800)          # larghezza e altezza di un tassello
L = max(LATO) / 2.0
CENTRO = np.array([LATO[0] / 2.0, LATO[1] / 2.0])
# x0, y0, x1, y1 della cornice nel dipinto: dentro la zona che i tasselli coprono,
# con il margine che serve a cercarne il bordo
CORNICE = (520.0, 420.0, 3440.0, 1920.0)


def dipinto_con_cornice(h, w, rng):
    """Il dipinto sintetico, con piu' grana fine (i punti caratteristici ne hanno
    bisogno) e una cornice scura dai bordi perfettamente dritti."""
    d = dipinto_sintetico(h, w)
    grana = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    d = d + 60.0 * grana[:, :, None]
    x0, y0, x1, y1 = (int(v) for v in CORNICE)
    fuori = np.ones((h, w), bool)
    fuori[y0:y1, x0:x1] = False
    d[fuori] *= 0.45
    return np.clip(d, 0, 255).astype(np.float32)


def omografia_vera(centro, rot_gradi, scala, prosp):
    """3x3 pixel (corretti) del tassello -> dipinto, scritta attorno al centro
    del tassello come nel programma: m = t + A d / (1 + q.d)."""
    th = np.radians(rot_gradi)
    A = scala * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = np.asarray(prosp, dtype=np.float64) / (L * L)
    B = A + np.outer(centro, q)
    H = np.eye(3)
    H[:2, :2] = B
    H[:2, 2] = centro - B @ CENTRO
    H[2, :2] = q
    H[2, 2] = 1.0 - q @ CENTRO
    return H


def fabbrica(cartella, dipinto, rng, n_righe=3, n_colonne=4, rot=2.5, scala=0.05, prosp=14.0,
             k=0.0, doppio=True, carrello=False):
    """
    I tasselli, salvati in ordine casuale.

    Args:
        prosp: ampiezza dei termini prospettici, in pixel sul bordo del tassello
        k: distorsione radiale dell'obiettivo (adimensionale: k * L px sul bordo)
        carrello: True = camera che trasla soltanto, con una piccola rotazione

    Returns:
        {nome del file: 3x3 vera, pixel corretti del tassello -> dipinto}
    """
    os.makedirs(cartella, exist_ok=True)
    W_t, H_t = LATO
    passo_x, passo_y = 0.66 * W_t, 0.66 * H_t
    pose = []
    for r in range(n_righe):
        sfalsa = rng.uniform(-120, 120)
        for c in range(n_colonne):
            centro = np.array([760 + c * passo_x + sfalsa, 640 + r * passo_y]) + rng.uniform(-30, 30, 2)
            if carrello:
                pose.append(omografia_vera(centro, rng.uniform(-0.4, 0.4), 1.0, (0.0, 0.0)))
            else:
                pose.append(omografia_vera(centro, rng.uniform(-rot, rot),
                                           1.0 + rng.uniform(-scala, scala),
                                           rng.uniform(-prosp, prosp, 2)))
    if doppio:
        # lo stesso scatto rifatto, spostato di poco
        H = pose[5].copy()
        H[:2, 2] += rng.uniform(-15, 15, 2)
        pose.append(H)
    ordine = rng.permutation(len(pose))
    vere = {}
    X, Y = np.meshgrid(np.arange(W_t, dtype=np.float64), np.arange(H_t, dtype=np.float64))
    for n_file, quale in enumerate(ordine):
        H = pose[quale]
        dx, dy = X - CENTRO[0], Y - CENTRO[1]
        f = 1.0 + k * (dx * dx + dy * dy) / (L * L)
        ux, uy = dx * f + CENTRO[0], dy * f + CENTRO[1]
        den = H[2, 0] * ux + H[2, 1] * uy + H[2, 2]
        mx = ((H[0, 0] * ux + H[0, 1] * uy + H[0, 2]) / den).astype(np.float32)
        my = ((H[1, 0] * ux + H[1, 1] * uy + H[1, 2]) / den).astype(np.float32)
        t = cv2.remap(dipinto, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        t = t * rng.uniform(0.9, 1.1) + rng.normal(0, 2.0, t.shape).astype(np.float32)
        nome = f"scatto_{n_file + 1:02d}.tif"
        mm = tiff_memmap.create_tiff_memmap(os.path.join(cartella, nome), H_t, W_t, samples=3,
                                            dtype=np.uint8)
        mm[:] = np.clip(t, 0, 255).astype(np.uint8)
        mm.flush()
        del mm
        vere[nome] = H
    return vere


def fabbrica_camera(cartella, dipinto, rng, focale=6000.0, n_righe=3, n_colonne=4,
                    fuori_asse=(-520.0, 330.0), rollio=1.5):
    """
    I tasselli di una camera VERA che ruota su un cavalletto davanti al dipinto.

    La camera sta a `focale` pixel dal piano del dipinto (un pixel del sensore
    vale circa un pixel del dipinto) ma NON davanti al suo centro: e' spostata
    di `fuori_asse`. Per inquadrare il centro deve quindi girarsi, e lo scatto
    centrale, che fara' da ancora, vede il dipinto di sbieco. E' il caso in cui
    il mosaico esce con i lati paralleli che convergono, e va raddrizzato.

    Returns:
        ({nome: 3x3 vera pixel del tassello -> dipinto}, inclinazione in gradi
        dell'asse di ogni scatto rispetto alla normale al dipinto, per nome)
    """
    os.makedirs(cartella, exist_ok=True)
    W_t, H_t = LATO
    passo_x, passo_y = 0.66 * W_t, 0.66 * H_t
    centro_griglia = np.array([760 + (n_colonne - 1) * passo_x / 2, 640 + (n_righe - 1) * passo_y / 2])
    C = np.array([centro_griglia[0] + fuori_asse[0], centro_griglia[1] + fuori_asse[1], -focale])
    K = np.diag([focale, focale, 1.0])
    Tc = np.array([[1.0, 0, CENTRO[0]], [0, 1.0, CENTRO[1]], [0, 0, 1.0]])
    pose, incl = [], []
    for r in range(n_righe):
        sfalsa = rng.uniform(-120, 120)
        for c in range(n_colonne):
            mira = np.array([760 + c * passo_x + sfalsa, 640 + r * passo_y, 0.0])
            mira[:2] += rng.uniform(-30, 30, 2)
            z = (mira - C) / np.linalg.norm(mira - C)
            x = np.cross([0.0, 1.0, 0.0], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            a = np.radians(rng.uniform(-rollio, rollio))
            R = np.array([np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y, z])
            # dal dipinto (X, Y, 1) ai pixel del tassello: K R [e1 e2 -C]
            M = Tc @ K @ R @ np.column_stack([[1.0, 0, 0], [0, 1.0, 0], -C])
            pose.append(np.linalg.inv(M))
            incl.append(float(np.degrees(np.arccos(z[2]))))
    ordine = rng.permutation(len(pose))
    vere, inclinazioni = {}, {}
    X, Y = np.meshgrid(np.arange(W_t, dtype=np.float64), np.arange(H_t, dtype=np.float64))
    for n_file, quale in enumerate(ordine):
        H = pose[quale]
        den = H[2, 0] * X + H[2, 1] * Y + H[2, 2]
        mx = ((H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / den).astype(np.float32)
        my = ((H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / den).astype(np.float32)
        t = cv2.remap(dipinto, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        t = t * rng.uniform(0.9, 1.1) + rng.normal(0, 2.0, t.shape).astype(np.float32)
        nome = f"scatto_{n_file + 1:02d}.tif"
        mm = tiff_memmap.create_tiff_memmap(os.path.join(cartella, nome), H_t, W_t, samples=3,
                                            dtype=np.uint8)
        mm[:] = np.clip(t, 0, 255).astype(np.uint8)
        mm.flush()
        del mm
        vere[nome] = H / H[2, 2]
        inclinazioni[nome] = incl[quale]
    return vere, inclinazioni


def errore_a_meno_di_similarita(pose_stimate, vere):
    """
    Quanto il mosaico differisce dal dipinto a meno di una similarita'
    (traslazione, rotazione, scala): e' l'errore di FORMA, quello che resta se
    il mosaico e' una vista di sbieco e che deve sparire dopo il raddrizzamento.
    """
    gx, gy = np.meshgrid(np.linspace(0, LATO[0] - 1, 5), np.linspace(0, LATO[1] - 1, 4))
    punti = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    st, ve = [], []
    for nome, H in vere.items():
        q = punti @ np.asarray(pose_stimate[nome]).T
        st.append(q[:, :2] / q[:, 2:3])
        q = punti @ H.T
        ve.append(q[:, :2] / q[:, 2:3])
    st, ve = np.concatenate(st), np.concatenate(ve)
    # similarita' ai minimi quadrati: [a -b tx; b a ty]
    A = np.zeros((2 * len(st), 4))
    A[0::2] = np.column_stack([st[:, 0], -st[:, 1], np.ones(len(st)), np.zeros(len(st))])
    A[1::2] = np.column_stack([st[:, 1], st[:, 0], np.zeros(len(st)), np.ones(len(st))])
    a, b, tx, ty = np.linalg.lstsq(A, ve.ravel(), rcond=None)[0]
    stima = np.column_stack([a * st[:, 0] - b * st[:, 1] + tx, b * st[:, 0] + a * st[:, 1] + ty])
    return np.hypot(*(stima - ve).T)


def errore_vero(report, vere, k_vero=0.0):
    """
    Errore di posizione VERO, in px, su una griglia di punti di ogni tassello.

    Il mosaico ha il sistema del tassello ancora (i suoi pixel corretti, a meno
    di una traslazione). La verita' in quel sistema e': dal tassello al dipinto
    con la posa vera, dal dipinto all'ancora con l'inversa della posa vera
    dell'ancora. Non si toglie nessuna trasformazione d'insieme.

    Returns:
        (errori (n_tasselli, n_punti), distanza di ogni tassello dall'ancora in px)
    """
    gx, gy = np.meshgrid(np.linspace(0, LATO[0] - 1, 5), np.linspace(0, LATO[1] - 1, 4))
    punti = np.column_stack([gx.ravel(), gy.ravel()])
    k_stimato = report['modello']['distorsione_k'] or 0.0

    def corretti(p, k):
        d = p - CENTRO
        return d * (1.0 + k * (d * d).sum(axis=1) / (L * L))[:, None] + CENTRO

    def applica(H, p):
        q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H).T
        return q[:, :2] / q[:, 2:3]

    per_nome = {os.path.basename(t['file']): t for t in report['tasselli']}
    anc = report['tasselli'][report['mosaico']['ancora'] - 1]
    H_anc_vera = vere[os.path.basename(anc['file'])]
    origine = applica(anc['tassello_verso_mosaico'], CENTRO[None, :])[0]
    errori, distanze = [], []
    for nome, H_vera in vere.items():
        t = per_nome[nome]
        if not t['nel_mosaico']:
            continue
        stimato = applica(t['tassello_verso_mosaico'], corretti(punti, k_stimato))
        nel_dipinto = applica(H_vera, corretti(punti, k_vero))
        vero = applica(np.linalg.inv(H_anc_vera), nel_dipinto) - CENTRO + origine
        errori.append(np.hypot(*(stimato - vero).T))
        distanze.append(float(np.hypot(*(applica(H_vera, CENTRO[None, :])[0]
                                         - applica(H_anc_vera, CENTRO[None, :])[0]))))
    return np.array(errori), np.array(distanze)


def scrivi_linee(path, vere, k=0.0):
    """Il file delle linee: i quattro lati della cornice, ognuno con gli estremi
    scritti nei pixel dei due tasselli che li vedono meglio."""
    x0, y0, x1, y1 = CORNICE
    m = 260.0
    lati = [("lato alto", "orizzontale", (x0 + m, y0), (x1 - m, y0)),
            ("lato basso", "orizzontale", (x0 + m, y1), (x1 - m, y1)),
            ("lato sinistro", "verticale", (x0, y0 + m), (x0, y1 - m)),
            ("lato destro", "verticale", (x1, y0 + m), (x1, y1 - m))]

    def nel_tassello(P):
        # il tassello che vede il punto piu' all'interno, e dove
        migliore = None
        for nome, H in vere.items():
            q = np.linalg.inv(H) @ np.array([P[0], P[1], 1.0])
            pu = q[:2] / q[2]
            d = pu - CENTRO
            for _ in range(8):
                d = (pu - CENTRO) / (1.0 + k * (d * d).sum() / (L * L))
            p = d + CENTRO
            interno = min(p[0], p[1], LATO[0] - 1 - p[0], LATO[1] - 1 - p[1])
            if migliore is None or interno > migliore[0]:
                migliore = (interno, nome, p)
        return {'tassello': migliore[1], 'x': float(migliore[2][0]), 'y': float(migliore[2][1])}

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'linee': [{'nome': n, 'direzione': d, 'estremi': [nel_tassello(a), nel_tassello(b)]}
                             for n, d, a, b in lati]}, fh, indent=1)


def controlla_totale(report, vere, dipinto, rng, lato=128, n=40):
    """
    L'immagine totale riletta e confrontata col dipinto: in finestre sparse sul
    mosaico si ricostruisce che cosa il dipinto dovrebbe mostrare li' (passando
    per la posa VERA dell'ancora) e si misura di quanto l'immagine scritta ne e'
    spostata. E' la prova che il ricampionamento usa le pose nel verso giusto.

    Returns:
        scarti in px
    """
    tot, _ = image_io.apri(report['immagine_totale'], verbose=False)
    Hm, Wm = tot.shape[:2]
    anc = report['tasselli'][report['mosaico']['ancora'] - 1]
    H_anc = vere[os.path.basename(anc['file'])]
    A = np.asarray(anc['tassello_verso_mosaico'])
    # mosaico -> pixel dell'ancora -> dipinto
    M = H_anc @ np.linalg.inv(A)
    win = cv2.createHanningWindow((lato, lato), cv2.CV_64F)
    scarti = []
    for _ in range(n * 6):
        if len(scarti) >= n:
            break
        X0, Y0 = int(rng.integers(0, Wm - lato)), int(rng.integers(0, Hm - lato))
        scritto = np.asarray(tot[Y0:Y0 + lato, X0:X0 + lato])
        if (scritto.sum(axis=2) == 0).any():
            continue                                    # tocca una zona senza tasselli
        X, Y = np.meshgrid(np.arange(X0, X0 + lato, dtype=np.float64),
                           np.arange(Y0, Y0 + lato, dtype=np.float64))
        den = M[2, 0] * X + M[2, 1] * Y + M[2, 2]
        mx = ((M[0, 0] * X + M[0, 1] * Y + M[0, 2]) / den).astype(np.float32)
        my = ((M[1, 0] * X + M[1, 1] * Y + M[1, 2]) / den).astype(np.float32)
        if mx.min() < 2 or my.min() < 2 or mx.max() > dipinto.shape[1] - 3 \
                or my.max() > dipinto.shape[0] - 3:
            continue
        atteso = cv2.remap(dipinto, mx, my, cv2.INTER_LINEAR)
        ga = bm.gradient_magnitude(cv2.cvtColor(atteso.astype(np.uint8), cv2.COLOR_RGB2GRAY))
        gb = bm.gradient_magnitude(cv2.cvtColor(np.ascontiguousarray(scritto), cv2.COLOR_RGB2GRAY))
        (dx, dy), conf = cv2.phaseCorrelate(ga.astype(np.float64), gb.astype(np.float64), win)
        if conf >= 0.05:
            scarti.append(float(np.hypot(dx, dy)))
    return np.array(scarti)


def riassumi(nome, rep, err, dist):
    lontani = dist >= 0.7 * dist.max()
    print("\n" + "=" * 72)
    print(nome)
    print(f"  freno scelto: {rep['modello']['freno_descrizione']}"
          + (f"; distorsione {rep['modello']['distorsione_px_sul_bordo']:+.2f} px sul bordo"
             if rep['modello']['distorsione_px_sul_bordo'] is not None else ""))
    print(f"  coppie misurate: {sum(c['stato'] == 'misurata' for c in rep['coppie'])}/"
          f"{len(rep['coppie'])}; cuciture {rep['verifica'].get('residuo_mediano_px', float('nan')):.3f} px "
          f"di mediana, max {rep['verifica'].get('residuo_max_px', float('nan')):.2f} px")
    print(f"  errore di posizione VERO: mediano {np.median(err):.3f} px, max {err.max():.3f} px; "
          f"nei tasselli piu' lontani dall'ancora mediano {np.median(err[lontani]):.3f} px, "
          f"max {err[lontani].max():.3f} px")


def main():
    lavoro = tempfile.mkdtemp(prefix='mosaico_om_test_')
    print(f"cartella di lavoro: {lavoro}\n")
    esiti = []
    try:
        rng = np.random.default_rng(11)
        dipinto = dipinto_con_cornice(2800, 4200, rng)
        comuni = dict(fattore=1, riduzione_sift=2, verbose=False)

        # ---- 1. omografie vere, freno automatico ---------------------------
        cart = os.path.join(lavoro, 'cavalletto')
        vere = fabbrica(cart, dipinto, rng)
        linee = os.path.join(lavoro, 'linee.json')
        scrivi_linee(linee, vere)
        out = os.path.join(lavoro, 'uscita')
        rep = mo.componi(mosaico.elenca_immagini(cart), out, distorsione='no', linee=linee,
                         formati=['TIFF'], **comuni)
        err, dist = errore_vero(rep, vere)
        riassumi("1. CAVALLETTO (rotazioni fino a 2.5 gradi, scale +-5%, prospettiva fino a "
                 "14 px, file sfalsate, uno scatto doppio)", rep, err, dist)
        tutti = all(t['nel_mosaico'] for t in rep['tasselli'])
        esiti.append(('cavalletto: tutti i tasselli nel mosaico, senza griglia', tutti))
        esiti.append(('cavalletto: freno allentato', rep['modello']['freno'] is not None
                      and rep['modello']['freno'] <= 1e-5))
        esiti.append(('cavalletto: errore vero sotto il pixel', np.median(err) < 0.3
                      and err.max() < 1.0))
        doppi = sum(not t['nell_immagine_totale'] for t in rep['tasselli'])
        esiti.append(('scatto doppio riconosciuto', doppi == 1))

        rl = rep['rettilineita']
        print(f"  rettilineita' della cornice: rms {rl['scarto_rms_px']:.3f} px, max "
              f"{rl['scarto_max_px']:.2f} px, freccia massima {rl['freccia_max_px']:.2f} px, "
              f"fuori squadra {rl.get('fuori_squadra_gradi')}")
        esiti.append(('cornice dritta con la soluzione giusta', rl['n_linee'] == 4
                      and rl['scarto_rms_px'] < 0.5 and rl['freccia_max_px'] < 1.0))
        affine = rl['con_gli_altri_freni']['infinito (affine)']
        print(f"  la stessa cornice con un'affine per tassello: rms {affine['scarto_rms_px']:.3f} px, "
              f"max {affine['scarto_max_px']:.2f} px, fuori squadra {affine['fuori_squadra_gradi']}, "
              f"aperture {[round(g['apertura_gradi'], 3) for g in affine['direzioni'].values()]} gradi "
              f"(con la soluzione giusta {[round(g['apertura_gradi'], 3) for g in rl['direzioni'].values()]})")
        esiti.append(('cornice storta con la soluzione sbagliata',
                      affine['scarto_rms_px'] > 2.0 * rl['scarto_rms_px']))

        scarti = controlla_totale(rep, vere, dipinto, rng)
        print(f"  immagine totale riletta e confrontata col dipinto su {len(scarti)} finestre: "
              f"scarto mediano {np.median(scarti):.3f} px, max {scarti.max():.3f} px; "
              f"foglio coperto al {100 * rep['copertura_del_foglio']:.0f}%")
        esiti.append(('immagine totale al posto giusto', len(scarti) >= 30
                      and np.median(scarti) < 0.3 and scarti.max() < 1.0))

        # ---- 2. gli stessi tasselli costretti a un'affine ------------------
        rep2 = mo.componi(mosaico.elenca_immagini(cart), os.path.join(lavoro, 'uscita_affine'),
                          freno=float('inf'), distorsione='no', scrivi_immagine=False, **comuni)
        err2, dist2 = errore_vero(rep2, vere)
        riassumi("2. GLI STESSI TASSELLI, FRENO INFINITO (un'affine per tassello)", rep2, err2, dist2)
        esiti.append(("con l'affine l'errore vero sale", np.median(err2) > 3.0 * np.median(err)))

        # ---- 3. tasselli da carrello ---------------------------------------
        cart3 = os.path.join(lavoro, 'carrello')
        vere3 = fabbrica(cart3, dipinto, rng, carrello=True, doppio=False)
        rep3 = mo.componi(mosaico.elenca_immagini(cart3), os.path.join(lavoro, 'uscita_carrello'),
                          distorsione='no', scrivi_immagine=False, **comuni)
        err3, dist3 = errore_vero(rep3, vere3)
        riassumi("3. CARRELLO (solo traslazione e rotazioni fino a 0.4 gradi)", rep3, err3, dist3)
        esiti.append(('carrello: il freno resta tirato', rep3['modello']['freno'] is None))
        esiti.append(('carrello: errore vero sotto il pixel', np.median(err3) < 0.3
                      and err3.max() < 1.0))

        # ---- 4. distorsione dell'obiettivo ---------------------------------
        cart4 = os.path.join(lavoro, 'distorsione')
        k_vero = 5.0 / L
        vere4 = fabbrica(cart4, dipinto, rng, k=k_vero, doppio=False)
        rep4 = mo.componi(mosaico.elenca_immagini(cart4), os.path.join(lavoro, 'uscita_dist'),
                          distorsione='auto', scrivi_immagine=False, **comuni)
        err4, dist4 = errore_vero(rep4, vere4, k_vero)
        riassumi("4. CAVALLETTO CON DISTORSIONE RADIALE (5 px sul bordo del tassello)",
                 rep4, err4, dist4)
        pk = rep4['modello']['distorsione_px_sul_bordo']
        esiti.append(('distorsione riconosciuta e stimata', pk is not None and abs(pk - 5.0) < 1.0))
        # Con la distorsione le corrispondenze sono tre volte meno precise (lo
        # spostamento cambia dentro il blocco), e l'errore agli angoli lontani
        # cresce con loro: e' la deriva che le cuciture non vedono, misurata.
        esiti.append(('con la distorsione: errore vero contenuto', np.median(err4) < 0.5
                      and err4.max() < 8.0))

        # ---- 5. camera vera fuori asse: focale implicita e raddrizzamento ----
        cart5 = os.path.join(lavoro, 'camera')
        vere5, incl5 = fabbrica_camera(cart5, dipinto, rng)
        linee5 = os.path.join(lavoro, 'linee_camera.json')
        scrivi_linee(linee5, vere5)
        rep5 = mo.componi(mosaico.elenca_immagini(cart5), os.path.join(lavoro, 'uscita_camera'),
                          distorsione='no', linee=linee5, raddrizza=True, scrivi_immagine=False,
                          **comuni)
        nomi5 = {t['numero']: os.path.basename(t['file']) for t in rep5['tasselli']}
        anc5 = nomi5[rep5['mosaico']['ancora']]
        prima = errore_a_meno_di_similarita(
            {nomi5[t['numero']]: t['tassello_verso_mosaico'] for t in rep5['tasselli']}, vere5)
        rad = rep5['raddrizzato']
        dopo = errore_a_meno_di_similarita(
            {nomi5[int(k)]: H for k, H in rad['tasselli'].items()}, vere5)
        foc = rep5['focale_implicita']
        print("\n" + "=" * 72)
        print("5. CAMERA VERA FUORI ASSE (focale 6000 px, lo scatto ancora vede il dipinto di sbieco)")
        print(f"  focale implicita: {foc['px']:.0f} px (vera 6000), correlazione {foc['correlazione']:+.4f}")
        print(f"  inclinazione dell'ancora rispetto al dipinto: stimata dalle linee "
              f"{rad['inclinazione_gradi']:.2f} gradi, vera {incl5[anc5]:.2f}; fuori squadra nello "
              f"spazio {rad['fuori_squadra_nello_spazio_gradi']:+.3f} gradi")
        print(f"  errore di FORMA (a meno di una similarita'): prima mediano {np.median(prima):.2f} px, "
              f"max {prima.max():.2f} px; dopo il raddrizzamento mediano {np.median(dopo):.2f} px, "
              f"max {dopo.max():.2f} px")
        esiti.append(('camera vera: focale implicita entro il 3%', abs(foc['px'] / 6000.0 - 1) < 0.03))
        esiti.append(("camera vera: inclinazione dell'ancora ritrovata",
                      abs(rad['inclinazione_gradi'] - incl5[anc5]) < 0.3))
        esiti.append(('camera vera: il raddrizzamento toglie la vista di sbieco',
                      prima.max() > 10.0 and dopo.max() < 0.25 * prima.max() and np.median(dopo) < 2.0))

        print("\n" + "=" * 72)
        for nome, ok in esiti:
            print(f"  {'ok     ' if ok else 'FALLITO'}  {nome}")
        tutto = all(ok for _, ok in esiti)
        print("TEST SUPERATO" if tutto else "TEST FALLITO")
        return 0 if tutto else 1
    finally:
        shutil.rmtree(lavoro, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
