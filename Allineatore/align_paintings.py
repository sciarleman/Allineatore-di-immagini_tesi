"""
Allineamento VIS/UV di dipinti ad altissima risoluzione, senza far esplodere la RAM.

    python align_paintings.py --fixed VIS_FraGalgario_TOT.tif \
                              --moving UVF_FraGalgario_TOT.tif \
                              --out UV_allineata.tif

Cosa fa, e perche' cosi':

1. STIMA su anteprime ridotte (default: fattore 2, ~7900x9900 px). La
   trasformazione cercata e' globale e liscia: stimarla a piena risoluzione non
   la renderebbe piu' precisa, costerebbe solo ordini di grandezza in piu' di
   memoria e tempo. La relazione geometrica fra anteprima e piena risoluzione e'
   esatta e nota, quindi la trasformazione si riporta a piena risoluzione senza
   perdita (vedi transform_model.py).

2. VERIFICA a piena risoluzione, su campioni indipendenti: si misura di quanti
   pixel le due immagini sono ancora disallineate dopo la trasformazione. E' il
   numero che conta davvero, e viene misurato sui dati veri, non sul costo
   interno di un ottimizzatore.

3. APPLICAZIONE a piena risoluzione in streaming, tile per tile: si legge dal
   TIFF sorgente (mappato in memoria) solo la regione che serve per ogni tile e
   si scrive direttamente nel TIFF di destinazione (anch'esso mappato). In RAM
   non c'e' mai piu' di un tile. Picco di memoria: qualche centinaio di MB,
   indipendente dalla dimensione dell'immagine.

4. SALVATAGGIO dei parametri (JSON + .npy del campo non rigido). Descrivono
   QUESTA coppia di esposizioni: servono a rigenerare il risultato senza
   ristimarlo e a warpare i raster derivati dalla stessa ripresa (maschere, ROI,
   ri-esportazioni) — vedi apply_transform.py. NON valgono per un'altra
   esposizione, nemmeno della stessa scena: per un canale IR, o per un secondo
   scatto UV, si rilancia questo script sulla nuova coppia. Il perche' e'
   spiegato in testa a apply_transform.py.

Confronto con la versione originale: robust_metrics.py caricava entrambi i TIFF
interi con cv2.imread (~1 GB l'uno, che diventano ~3.7 GB ciascuno appena
convertiti in float32) e ricalcolava un descriptor MIND da ~10 GB ad ogni
valutazione della funzione obiettivo, centinaia di volte. Da qui il crash.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import cv2

import image_io
import tiff_memmap
import block_matching as bm
import punti_manuali
import validation as val
from transform_model import Transform


# Sotto questa quota di blocchi buoni alla prima passata si controlla se
# l'aggancio dalla traslazione ha preso, e lo si considera fallito se la sua
# affine sbaglia piu' di RAPPORTO_AGGANCIO_FALLITO volte quella della posa
# grossolana. I numeri da cui vengono sono in _scegli_partenza.
SOGLIA_AGGANCIO = 0.25
RAPPORTO_AGGANCIO_FALLITO = 3.0


def human_time(sec):
    return f"{sec:.1f}s" if sec < 60 else f"{sec / 60:.1f} min"


def peak_memory_gb():
    """
    Picco di memoria del processo (Windows), senza dipendenze esterne.
    Serve a dare un riscontro oggettivo al motivo per cui esiste questa
    pipeline: il picco deve restare di poche centinaia di MB anche su
    immagini da 1 GB l'una.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PMC()
        counters.cb = ctypes.sizeof(PMC)
        kernel32 = ctypes.windll.kernel32
        # Senza argtypes/restype espliciti l'handle del processo (-1) viene
        # troncato a 32 bit e la chiamata fallisce silenziosamente.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        ok = get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        # Due numeri diversi e non intercambiabili:
        # - working set: include anche le pagine dei TIFF mappati in memoria che
        #   il processo ha toccato. Windows le conta qui, ma puo' liberarle in
        #   qualsiasi momento perche' sono gia' su disco: non sono un consumo
        #   reale e non possono causare un esaurimento della memoria.
        # - pagefile usage (memoria privata): quella davvero allocata dal
        #   processo. E' questo il numero da confrontare con la RAM disponibile.
        return (counters.PeakPagefileUsage / 1024 ** 3,
                counters.PeakWorkingSetSize / 1024 ** 3)
    except Exception:
        return None


def align(fixed_path, moving_path, out_path=None,
          params_path='transform_params.json',
          factor=2, nonrigid=True, tile_size=2048,
          block=256, step=128, refine_block=128, refine_step=64,
          min_conf=0.10, diagnostics_dir='diagnostica', mask=None,
          mask_info=None, punti=None, n_maniglie=0, contorno_moving=None, arr_fixed=None,
          tre_scale=False, punti_come_posa=True, punti_come_vincolo=True, margine=0,
          posa_dal_bordo=False,
          fascia_di_bordo=0, progress=None, verbose=True):
    """
    Esegue l'intera pipeline. Se out_path e' None si ferma dopo stima e verifica
    (utile per controllare il risultato prima di scrivere ~1 GB su disco).

    Args:
        mask: maschera opzionale (array uint8/bool a QUALSIASI risoluzione, viene
            ridimensionata): limita la stima alla zona non nulla. Serve solo se
            il fondo ha struttura in entrambe le immagini; vedi README.
        mask_info: descrizione della forma disegnata (dall'interfaccia), che
            finisce nel file dei parametri insieme al raster della maschera.
        punti: (N, 4) punti di corrispondenza messi a mano
            [x_fixed, y_fixed, x_moving, y_moving]. Servono solo dove
            l'automatico non aggancia, cioe' le radiografie: vedi punti_manuali.
            Lasciandoli a None la pipeline e' esattamente quella di sempre.
        contorno_moving: (M, 2) il bordo del soggetto tracciato a mano
            sull'immagine da allineare, in pixel di quell'immagine. Si usa con
            fascia_di_bordo (o posa_dal_bordo), insieme al contorno della
            maschera del riferimento; da solo viene soltanto salvato.
        arr_fixed: immagine di riferimento gia' aperta (per non riaprirla a ogni
            immagine quando si elabora un lotto).
        tre_scale: 'auto' calcola il campo grossolano e lo applica solo se la
            deformazione supera la portata della passata fine (64 px con i
            valori di sempre): e' l'algoritmo unico per tutte le bande, perche'
            sulle bande fotografiche si comporta come le due passate e sulle
            radiografie e lo SWIR come le tre scale.
            True aggiunge un campo a scala grossolana fra l'affine e il
            campo fine (block_matching.campo_grossolano), per le deformazioni
            oltre la portata della passata fine. Lo usa solo la sezione delle
            radiografie; con False (il default) la stima e' quella di sempre.
        punti_come_posa: True (il default, il comportamento di sempre) usa i
            punti anche per la posa di partenza, con la doppia posa. False li
            usa SOLO come vincolo locale dove la verifica non misura
            (_correggi_con_punti): la stima resta quella automatica. Con le tre
            scale la posa dai punti peggiorava il risultato (da 5.0 a 6.9 px sui
            dettagli interni della Cappellochiaro).
        punti_come_vincolo: True (il default di align) usa i punti come
            vincolo locale dove la verifica non misura. False non li usa per
            allineare (salvo punti_come_posa): servono solo a misurare il TRE,
            lo scarto fra dove la trasformazione porta ogni dettaglio e dove lo
            si e' cliccato. Sulla Cappellochiaro, con le tre scale, il vincolo
            non cambiava i dettagli indipendenti (5.0 px di mediana in entrambi
            i casi): la sezione RX parte quindi con False.
        margine: pixel di foglio in piu' per lato nell'immagine di uscita, o
            'auto' per calcolarli (margine_necessario). 0, il default, scrive
            come sempre un'immagine grande quanto il riferimento. Con un margine
            l'uscita non si sovrappone piu' al riferimento originale, ma a una
            sua copia con lo stesso margine (scrivi_con_margine).
        posa_dal_bordo: True ricava la posa di partenza dai soli due contorni
            (la zona sul riferimento e contorno_moving), prima di guardare
            l'interno: punti_manuali.posa_dai_contorni. Poi la stima procede
            come sempre, e la doppia posa la confronta con quella automatica.
        fascia_di_bordo: larghezza in px della fascia lungo il bordo in cui il
            contorno della zona viene portato su contorno_moving
            (punti_manuali.campo_fascia_di_bordo); 0, il default, non la usa.
        progress: callback opzionale f(testo, frazione 0..1) per l'interfaccia.

    Returns:
        (Transform, report dict)
    """
    t_start = time.time()
    os.makedirs(diagnostics_dir, exist_ok=True)

    def avanzamento(testo, frazione):
        if progress:
            progress(testo, frazione)

    info_f = image_io.info(fixed_path)
    info_m = image_io.info(moving_path)
    print("=" * 72)
    print("ALLINEAMENTO")
    print("=" * 72)
    print(f"riferimento:   {info_f}")
    print(f"da allineare:  {info_m}")

    H, W = info_f.height, info_f.width
    report = {'fixed': os.path.abspath(fixed_path), 'moving': os.path.abspath(moving_path)}
    if (info_m.height, info_m.width) != (H, W):
        print(f"  [nota] le due immagini hanno dimensioni diverse: l'output avra' "
              f"le dimensioni del riferimento ({W}x{H}) e il rapporto di scala "
              f"verra' assorbito dalla trasformazione.")

    # ---------------------------------------------------------------- 1. stima
    avanzamento("lettura anteprime", 0.05)
    t0 = time.time()
    target = max(H, W) // factor
    print(f"\n[1/4] Lettura anteprime (lato lungo ~{target} px) ...")
    prev_f, f_used = image_io.leggi_ridotta(fixed_path, lato_lungo_target=target, in_grigio=True)
    prev_m, f_used_m = image_io.leggi_ridotta(moving_path, lato_lungo_target=target, in_grigio=True)
    print(f"      riferimento {prev_f.shape[1]}x{prev_f.shape[0]} (fattore {f_used:g}), "
          f"da allineare {prev_m.shape[1]}x{prev_m.shape[0]} (fattore {f_used_m:g}) "
          f"in {human_time(time.time() - t0)}")

    # Normalizzazione di scala fra le due anteprime.
    #
    # Il matching confronta il blocco in (cx, cy) della fixed con quello nella
    # stessa posizione della moving: funziona finche' le due immagini sono
    # grosso modo sovrapposte. Se una ripresa ha un sensore o un ritaglio
    # diverso, il soggetto vi appare a una scala diversa e nessuna traslazione
    # puo' rimediare (un blocco a meta' immagine si troverebbe a centinaia di
    # pixel di distanza). Si porta quindi l'anteprima della moving alla stessa
    # scala di quella della fixed, e la differenza residua di inquadratura resta
    # a carico dell'affine, che e' fatta apposta.
    #
    # Il riscalamento e' ISOTROPO (stesso fattore sui due assi): un'eventuale
    # differenza di proporzioni e' una scala anisotropa vera, che deve comparire
    # nella trasformazione stimata e non essere nascosta qui.
    rapporto = float(np.sqrt((prev_f.shape[0] / prev_m.shape[0]) *
                             (prev_f.shape[1] / prev_m.shape[1])))
    #
    # Il fattore si passa a cv2.resize come fx = fy, non come dimensioni di
    # uscita. Con le dimensioni OpenCV usa il rapporto VERO di ciascun asse,
    # che dopo l'arrotondamento a numeri interi non e' piu' `rapporto`: sulla
    # radiografia 0.863479 in x e 0.863389 in y contro 0.863429. Il resto del
    # programma pero' torna a piena risoluzione con `rapporto`, e la differenza,
    # entrata dopo la stima, l'affine non la vedeva: la RX allineata scivolava
    # di ~0.5 px al centro e ~1 px verso il basso e verso destra (misurato, e
    # sparito correggendo con i rapporti veri). Con fx = fy la scala usata e'
    # esattamente `rapporto` su tutti e due gli assi. Le immagini con le stesse
    # dimensioni del riferimento non passano da qui.
    if abs(rapporto - 1.0) > 0.02:
        interp = cv2.INTER_AREA if rapporto < 1 else cv2.INTER_LINEAR
        prev_m = cv2.resize(prev_m, None, fx=rapporto, fy=rapporto, interpolation=interp)
        nuova = (prev_m.shape[1], prev_m.shape[0])
        f_used_m = f_used_m / rapporto
        print(f"      scala normalizzata: anteprima da allineare riportata a "
              f"{nuova[0]}x{nuova[1]} (fattore {rapporto:.4f}); il soggetto "
              f"appare {100 * (rapporto - 1):+.1f}% diverso fra le due riprese")

    mask_prev = None
    if mask is not None:
        mask_prev = cv2.resize(np.asarray(mask).astype(np.uint8),
                               (prev_f.shape[1], prev_f.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        print(f"      maschera attiva: {100.0 * mask_prev.mean():.0f}% "
              f"dell'inquadratura usata per la stima")

    print(f"\n[2/4] Stima della trasformazione")
    avanzamento("stima della trasformazione", 0.15)
    t0 = time.time()
    # Punti di corrispondenza messi a mano: non sostituiscono il matching, gli
    # danno la posa di partenza. Si pre-allinea la moving con l'affine dei punti
    # e da li' la pipeline prosegue identica; alla fine le due trasformazioni si
    # compongono. Serve dove il pre-allineamento automatico non basta, perche'
    # sa fare solo traslazioni: sulla radiografia restano 12% di scala e 0.3
    # gradi di rotazione, e il 97% dei blocchi finisce per cercare nel posto
    # sbagliato.
    T_punti = None
    # I punti hanno due ruoli distinti che conviene non confondere: dare la posa
    # di partenza, e fare da vincolo locale dove la verifica e' cieca. La doppia
    # posa piu' avanti puo' scartare il primo ruolo senza toccare il secondo.
    punti_vincolo = (punti if (punti_come_vincolo and punti is not None and len(punti) >= 3)
                     else None)
    prev_m_auto, f_used_m_auto = prev_m, f_used_m
    if punti is not None and len(punti) >= 3:
        ctrl = punti_manuali.controlla_punti(punti)
        M_punti, res_punti = ctrl['M'], ctrl['residui']
        n_sosp = int(ctrl['sospetti'].sum())
        print(f"\n      punti manuali: "
              f"{punti_manuali.descrivi(M_punti, res_punti, ctrl['modello'])}")
        # Lo scarto dal modello globale e quello dai vicini vanno letti insieme:
        # il primo e' alto anche con clic ottimi quando la tela e' deformata, il
        # secondo no. Vedi punti_manuali.controlla_punti.
        print(f"      scarto dai punti vicini: mediano "
              f"{np.median(ctrl['locali']):.1f} px; "
              + (f"{n_sosp} punti non concordano con i vicini" if n_sosp
                 else "nessun punto in contrasto con i vicini"))
        if punti_come_posa:
            T_punti = Transform(M_punti, ref_shape=(H, W))
            prev_m = T_punti.at_factor(f_used, f_used_m).apply_to_image(
                prev_m, prev_f.shape[:2])
            f_used_m = f_used          # la moving vive ora nel sistema della fixed
        else:
            print("      i punti si usano solo come vincolo locale, non per la posa")
        report['punti_manuali'] = {
            'come_posa': bool(punti_come_posa),
            'n': int(len(punti)), 'modello': ctrl['modello'],
            'scarto_mediano_px': float(np.median(res_punti)),
            'scarto_max_px': float(np.max(res_punti)),
            'scarto_locale_mediano_px': float(np.median(ctrl['locali'])),
            'n_sospetti': n_sosp,
        }

    # Il bordo prima di tutto: la posa di partenza viene dalle due linee, e
    # l'interno si stima a partire da li'. Dai punti, se ci sono, si prende solo
    # il punto di partenza dell'iterazione.
    if posa_dal_bordo and contorno_moving is not None and mask is not None:
        G_b = punti_manuali.contorno_da_maschera(mask, (H, W))
        if G_b is not None:
            M0 = M_punti if (punti is not None and len(punti) >= 3) else None
            M_b, info_pb = punti_manuali.posa_dai_contorni(G_b, contorno_moving, M0)
            T_punti = Transform(M_b, ref_shape=(H, W))
            prev_m = T_punti.at_factor(f_used, f_used_m_auto).apply_to_image(
                prev_m_auto, prev_f.shape[:2])
            f_used_m = f_used
            report['posa_dal_bordo'] = info_pb
            print(f"\n      posa dal bordo: le due linee coincidono a "
                  f"{info_pb['scarto_mediano_px']:.1f} px di mediana "
                  f"(p90 {info_pb['scarto_p90_px']:.1f} px), {info_pb['iterazioni']} giri")

    diag_stima = {}
    arr_f = arr_m = None
    partita_dalla_posa = False
    argomenti_stima = dict(
        block=block, step=step, min_conf=min_conf,
        nonrigid=nonrigid, refine_block=refine_block, refine_step=refine_step,
        mask=mask_prev, verbose=verbose, diagnostica=diag_stima,
        coarse_init=T_punti is None, campo_grossolano_tra_le_passate=tre_scale)
    try:
        # Il controllo dell'aggancio vale solo per la partenza automatica: con
        # i punti o con il bordo la posa di partenza l'ha data chi usa il
        # programma, e la stima prosegue come sempre.
        T, est_info = bm.estimate_transform(
            prev_f, prev_m, f_used, factor_moving=f_used_m,
            soglia_aggancio=SOGLIA_AGGANCIO if T_punti is None else None,
            **argomenti_stima)
        dubbio = None
    except bm.AggancioDubbio as exc:
        # Dell'eccezione si tengono solo l'affine e i numeri. Finche' resta
        # viva, la sua traccia tiene in memoria le variabili della stima
        # interrotta, cioe' le due mappe di gradiente (2.4 GB l'una a fattore 1
        # su 593 Mpx): lavorandoci dentro il picco saliva da 11.0 a 13.2 GB.
        dubbio = (exc.T, exc.info, str(exc))
    if dubbio is not None:
        gc.collect()

        def apri_le_due():
            return (arr_fixed if arr_fixed is not None else image_io.apri(fixed_path)[0],
                    image_io.apri(moving_path)[0])

        Tg, arr_f, arr_m = _scegli_partenza(dubbio, fixed_path, moving_path, H, W, mask,
                                            mask_prev, apri_le_due, report)
        if Tg is not None:
            T, est_info, diag_stima = _stima_dalla_posa(
                Tg, prev_f, prev_m, f_used, f_used_m, H, W, argomenti_stima)
            partita_dalla_posa = True
        else:
            T, est_info = bm.estimate_transform(
                prev_f, prev_m, f_used, factor_moving=f_used_m, **argomenti_stima)
    if T_punti is not None:
        # La moving era gia' nel sistema della fixed, quindi quanto stimato e'
        # un residuo su quel sistema: si compone con l'affine dei punti.
        T = T_punti.compose_residual(T.M, delta_field=T.field,
                                     delta_grid_step=T.grid_step)
    T.ref_shape = (H, W)
    T.meta = {'estimazione': est_info}
    if mask_prev is not None:
        # Senza questo, dai parametri non si capiva nemmeno SE una maschera fosse
        # stata usata, figuriamoci quale. Si salvano due cose: il raster alla
        # scala della stima (per riprodurre esattamente il risultato) e la forma
        # disegnata in frazioni (per sapere cosa e' stato escluso, e citarlo).
        file_maschera = params_path.rsplit('.', 1)[0] + '_maschera.png'
        image_io.scrivi_immagine(file_maschera, mask_prev.astype(np.uint8) * 255)
        T.meta['maschera'] = {
            'copertura_pct': round(100.0 * float(mask_prev.mean()), 2),
            'file': os.path.basename(file_maschera),
            'risoluzione_file': [int(mask_prev.shape[1]), int(mask_prev.shape[0])],
            **(mask_info or {'forma': 'non registrata (maschera passata come raster)'}),
        }
    else:
        T.meta['maschera'] = None
    if punti is not None and len(punti) >= 3:
        # I punti si salvano accanto ai parametri: la corsa resta ripetibile e
        # si possono riusare senza ricliccarli.
        file_punti = params_path.rsplit('.', 1)[0] + '_punti.json'
        # n_maniglie dice quante delle corrispondenze vengono dall'allineamento
        # a mano sul bordo e stanno in testa all'elenco. Senza, alla riapertura
        # non si saprebbe piu' quali rifare trascinando e quali sono punti messi
        # dentro il dipinto, che servono a tutt'altro.
        punti_manuali.salva(punti, file_punti,
                            extra={'fixed': os.path.abspath(fixed_path),
                                   'moving': os.path.abspath(moving_path),
                                   'n_maniglie': int(n_maniglie)})
        T.meta['punti_manuali'] = dict(report.get('punti_manuali', {}),
                                       file=os.path.basename(file_punti),
                                       n_maniglie=int(n_maniglie))
    if contorno_moving is not None and len(contorno_moving) >= 3:
        # Anche il contorno va salvato, per lo stesso motivo dei punti. Finche'
        # non lo era, una corsa col vincolo di bordo non si poteva rifare: la
        # maschera e i punti erano su disco, il contorno viveva solo nella
        # memoria della finestra e con la sessione spariva. Tracciarlo costa
        # all'utente parecchi minuti, quindi perderlo e' il genere di danno che
        # non si nota subito e si paga dopo.
        file_contorno = params_path.rsplit('.', 1)[0] + '_contorno.json'
        C = np.asarray(contorno_moving, dtype=float).reshape(-1, 2)
        with open(file_contorno, 'w', encoding='utf-8') as fh:
            json.dump({'commento': "bordo del soggetto tracciato a mano sulla "
                                   "moving, [x, y] in pixel a piena risoluzione",
                       'moving': os.path.abspath(moving_path),
                       'contorno': C.tolist()}, fh, indent=1, ensure_ascii=False)
        T.meta['contorno_moving'] = {'n_punti': int(len(C)),
                                     'file': os.path.basename(file_contorno)}
    print(f"\n      stima completata in {human_time(time.time() - t0)}")
    print("\nTrasformazione stimata (riferimento -> immagine, px a piena risoluzione):")
    print(T.describe())
    stima_iniziale = est_info
    report['stima'] = est_info

    # ----------------------------------------------------------- 2. verifica
    print(f"\n[3/4] Verifica a piena risoluzione (campioni indipendenti)")
    avanzamento("verifica", 0.4)
    if arr_f is None:
        arr_f = arr_fixed if arr_fixed is not None else image_io.apri(fixed_path)[0]
    if arr_m is None:
        arr_m, _ = image_io.apri(moving_path)

    # Quel che serve a rifare una stima da capo, raccolto una volta sola: lo
    # usano il ripiego e la doppia posa, che devono poter ripartire entrambi
    # dalle stesse anteprime NON pre-allineate dai punti.
    ctx = dict(fixed_path=fixed_path, moving_path=moving_path, H=H, W=W,
               mask=mask, mask_prev=mask_prev, prev_f=prev_f, prev_m=prev_m_auto,
               f_used=f_used, f_used_m=f_used_m_auto, block=block, step=step,
               min_conf=min_conf, nonrigid=nonrigid, refine_block=refine_block,
               refine_step=refine_step, arr_f=arr_f, arr_m=arr_m, verbose=verbose,
               tre_scale=tre_scale)

    # La stessa maschera usata per la stima vale anche qui: misurare il residuo
    # sullo sfondo non dice nulla sull'allineamento del dipinto, perche' lo
    # sfondo sta a un'altra distanza dalla camera e si sposta per parallasse.
    print("  prima dell'allineamento:")
    before = val.residual_check(arr_f, arr_m, None, mask=mask_prev)
    print("  dopo l'allineamento:")
    after = val.residual_check(arr_f, arr_m, T, mask=mask_prev)

    # Ripiego: se l'aggancio e' fallito del tutto si ricomincia da una posa
    # stimata su anteprime molto ridotte, dove la portata dei blocchi in pixel
    # veri e' molto piu' grande. Non scatta mai su una corsa riuscita - la
    # soglia e' a 10 px, e i fallimenti veri stanno sulle decine o centinaia -
    # quindi per le bande che gia' funzionano non cambia niente.
    # Se la stima e' gia' partita dalla posa grossolana il ripiego rifarebbe lo
    # stesso identico conto: si salta.
    if not partita_dalla_posa:
        T, est_info, diag_stima, after = _ripiego_se_serve(
            T, est_info, diag_stima, after, ctx, report, 'ripiego')

    # Doppia posa. Quando ci sono punti, la loro affine diventa la posa di
    # partenza e spegne il pre-allineamento automatico. Era la scelta giusta
    # quando l'automatico sapeva fare solo traslazioni, ma da quando c'e' il
    # ripiego non lo e' piu' sempre, e quale delle due pose sia migliore dipende
    # da quanto sono buoni i punti, cosa che non si sa prima. Quindi si stima
    # anche senza, si da' ANCHE a quel candidato la possibilita' del ripiego -
    # altrimenti si confronterebbe una posa soccorsa con una lasciata a se
    # stessa - e si tiene quella che misura meglio.
    if T_punti is not None:
        print("")
        print("  [doppia posa] si stima anche senza i punti, per confronto.")
        try:
            diag_auto = {}
            T_auto, est_auto = bm.estimate_transform(
                prev_f, prev_m_auto, f_used, factor_moving=f_used_m_auto,
                block=block, step=step, min_conf=min_conf,
                nonrigid=nonrigid, refine_block=refine_block,
                refine_step=refine_step, mask=mask_prev, verbose=False,
                diagnostica=diag_auto, coarse_init=True,
                campo_grossolano_tra_le_passate=tre_scale)
            T_auto.ref_shape = (H, W)
            # La meta si porta dietro: maschera, punti e contorno sono gia'
            # registrati li' e non devono sparire se vince questa posa.
            T_auto.meta = dict(T.meta, estimazione=est_auto)
            after_auto = val.residual_check(arr_f, arr_m, T_auto, mask=mask_prev,
                                            verbose=False)
            T_auto, est_auto, diag_auto, after_auto = _ripiego_se_serve(
                T_auto, est_auto, diag_auto, after_auto, ctx, report,
                'ripiego_senza_punti')
            m_p = after.get('residuo_mediano_px')
            m_a = after_auto.get('residuo_mediano_px')
            print("    coi tuoi punti contro senza:")
            # Sulle stesse finestre, sempre: le due pose rendono misurabili
            # finestre diverse, e le mediane grezze non sono confrontabili.
            conf = _confronta_stesse_finestre(after, after_auto)
            if conf is not None:
                vince_auto = conf['residuo_dopo_px'] < conf['residuo_prima_px'] - 0.05
            else:
                vince_auto = (m_a is not None and (m_p is None or m_a < m_p))
            report['doppia_posa'] = {
                'posa_dai_punti_tenuta': not vince_auto,
                'residuo_coi_punti_px': m_p,
                'residuo_senza_punti_px': m_a,
                'confronto_stesse_finestre': conf,
            }
            if vince_auto:
                print("    la posa automatica misura meglio: i punti non si usano "
                      "come posa di partenza (restano come vincolo locale).")
                T, est_info, diag_stima, after = T_auto, est_auto, diag_auto, after_auto
                T_punti = None
            else:
                print("    la posa dai tuoi punti misura meglio: si tiene quella.")
        except Exception as exc:
            print(f"    la seconda stima non e' riuscita: {exc}")
            report['doppia_posa'] = {'posa_dai_punti_tenuta': True, 'errore': str(exc)}

    # Il report deve descrivere la stima TENUTA. Il ripiego e la doppia posa
    # possono sostituirla, e finche' qui restava quella di partenza il report
    # raccontava il tentativo scartato: su un lotto a precisione massima diceva
    # 200 blocchi inlier su 22299 e "campo non rigido non applicato" per
    # un'immagine finita a 0.05 px con un campo da 81 px, e il riepilogo del
    # lotto ne ricavava campo_nonrigido: false. Il tentativo scartato si tiene
    # a parte, perche' dice perche' e' servito il ripiego.
    if est_info is not stima_iniziale:
        report['stima_primo_tentativo'] = stima_iniziale
        report['stima'] = est_info

    # I punti messi a mano diventano vincoli della MISURA, non solo della posa
    # di partenza: dove la verifica non riesce a misurare niente, l'unica
    # informazione disponibile sono loro. Vedi punti_manuali.campo_dai_punti per
    # i numeri e per il perche' l'interpolazione esatta funziona dove una media
    # pesata falliva.
    if punti_vincolo is not None:
        prima_punti = after
        T, info_corr, campo_punti, usati_punti = _correggi_con_punti(
            T, punti, arr_f, arr_m, after, mask_prev, report)
        tre = info_corr.get('tre')
        if tre:
            # Se la posa viene dai punti, la posa li ha visti tutti: il TRE
            # lascia fuori un punto solo dalla correzione locale.
            tre['ottimistico'] = T_punti is not None
            print(f"  [TRE sui punti, lasciandone fuori uno alla volta] mediana "
                  f"{tre['mediana_px']:.1f} px, oltre 5 px {100 * tre['quota_sopra_5px']:.0f}%, "
                  f"oltre 10 px {100 * tre['quota_sopra_10px']:.0f}% su {tre['n']} punti"
                  + (" (ottimista: i punti danno anche la posa)" if tre['ottimistico'] else ""))
        if info_corr.get('applicata'):
            print("  dopo la correzione con i punti:")
            after = val.residual_check(arr_f, arr_m, T, mask=mask_prev)
            # Il confronto finisce nel report e non solo a schermo: e' l'unico
            # modo onesto di dire se la correzione ha fatto bene o male, e
            # serve piu' tardi, non solo mentre gira.
            info_corr['confronto'] = _confronta_stesse_finestre(prima_punti, after)

    # Il bordo del dipinto: l'unica cosa che nelle due immagini significa la
    # stessa cosa dove non c'e' niente da misurare. Agisce solo in una fascia
    # lungo il contorno, quindi non tocca l'interno, che e' misurato.
    if fascia_di_bordo and contorno_moving is not None and mask is not None:
        prima_bordo = after
        T, info_bordo = _correggi_con_fascia(T, contorno_moving, mask, (H, W),
                                             float(fascia_di_bordo), report)
        if info_bordo.get('applicata'):
            print("  dopo il vincolo della fascia di bordo:")
            after = val.residual_check(arr_f, arr_m, T, mask=mask_prev)
            info_bordo['confronto'] = _confronta_stesse_finestre(prima_bordo, after)

    # Punti solo per verifica: la trasformazione non li ha visti (salvo la
    # posa), quindi lo scarto su ciascuno e' gia' un TRE, senza lasciarne fuori
    # nessuno. Si misura sulla trasformazione finale, fascia compresa.
    if punti_vincolo is None and punti is not None and len(punti) >= 3:
        P = np.asarray(punti, dtype=np.float64).reshape(-1, 4)
        tre = _riassunto_tre(P[:, 2:4] - _porta_punti(T, P[:, :2]), lascia_fuori=False)
        tre['ottimistico'] = T_punti is not None
        report['correzione_punti'] = {'applicata': False, 'motivo': 'punti solo per verifica',
                                      'tre': tre}
        print(f"  [TRE sui {tre['n']} punti, usati solo per verifica] mediana "
              f"{tre['mediana_px']:.1f} px, oltre 5 px {100 * tre['quota_sopra_5px']:.0f}%, "
              f"oltre 10 px {100 * tre['quota_sopra_10px']:.0f}%"
              + (" (ottimista: i punti danno la posa)" if tre['ottimistico'] else ""))

    # La verifica qui sopra, per costruzione, non arriva al bordo. Questa si',
    # e riporta il residuo per distanza dal contorno del soggetto.
    print("  lungo il contorno:")
    bordo = val.verifica_bordo(arr_f, arr_m, T, mask=mask_prev)
    quality = val.mind_quality(arr_f, arr_m, T)
    # Dove il risultato non convince, e perche': serve a sapere dove guardare,
    # e distingue le zone in cui un punto messo a mano servirebbe da quelle in
    # cui non servirebbe affatto.
    zone = val.zone_critiche(after, bordo, con_maschera=mask_prev is not None)
    escludi = ('campioni', 'campioni_deboli')
    report['verifica'] = {'prima': {k: v for k, v in before.items() if k not in escludi},
                          'dopo': {k: v for k, v in after.items() if k not in escludi},
                          'bordo': {k: v for k, v in bordo.items() if k not in escludi},
                          'zone_critiche': zone,
                          'mind': quality}

    # anteprime di controllo visivo
    prev_m_warp = T.at_factor(f_used, f_used_m).apply_to_image(prev_m, prev_f.shape[:2])
    val.make_overlay(prev_f, prev_m_warp, os.path.join(diagnostics_dir, 'overlay_scacchiera.png'),
                     mode='checker', tile=max(32, prev_f.shape[0] // 24))
    val.make_overlay(prev_f, prev_m_warp, os.path.join(diagnostics_dir, 'overlay_falsocolore.png'),
                     mode='color')
    val.make_overlay(prev_f, prev_m, os.path.join(diagnostics_dir, 'overlay_falsocolore_PRIMA.png'),
                     mode='color')
    _save_field_map(T, os.path.join(diagnostics_dir, 'campo_nonrigido.png'))
    # Dove sono finiti i vincoli e quanto correggono: e' l'unico modo di
    # guardare il lavoro dei punti, visto che agiscono proprio dove la verifica
    # e' cieca e nessun numero puo' raccontarlo.
    if punti_vincolo is not None and campo_punti is not None:
        _mappa_vincoli(campo_punti, punti, usati_punti, prev_f, (H, W),
                       os.path.join(diagnostics_dir, 'vincoli_dai_punti.png'))
    print(f"  immagini di controllo salvate in: {diagnostics_dir}/")

    # Margine dell'immagine di uscita. La trasformazione resta nel sistema del
    # riferimento; cambia solo il foglio su cui si scrive. Si registra nei
    # parametri perche' chi riapre l'immagine sappia di quanto e' spostata.
    if margine == 'auto':
        margine = margine_necessario(T, (info_m.height, info_m.width), (H, W))
    margine = int(margine or 0)
    report['margine_uscita_px'] = margine
    if margine:
        T.meta['margine_uscita_px'] = margine
        print(f"\n  margine attorno all'immagine di uscita: {margine} px per lato "
              f"(foglio {W + 2 * margine}x{H + 2 * margine}); il pixel (x, y) del "
              f"riferimento sta in (x+{margine}, y+{margine})")

    # ------------------------------------------------------------ 3. parametri
    T.save(params_path)
    with open(params_path.rsplit('.', 1)[0] + '_report.json', 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n  parametri salvati: {params_path}")

    # --------------------------------------------------------------- 4. warp
    if out_path is None:
        print(f"\n[4/4] Scrittura saltata (nessun --out): stima e verifica completate.")
        _print_footer(t_start, before, after)
        return T, report

    print(f"\n[4/4] Applicazione a piena risoluzione ({W}x{H}), tile da {tile_size}px, in streaming")
    scrivi_allineata(T, arr_m, out_path, H, W, info_m.samples, info_m.dtype,
                     tile_size=tile_size, progress=progress, margine=margine)

    _print_footer(t_start, before, after)
    return T, report


def _posa_da_rifare(after, soglia_px=10.0):
    """Il risultato e' cosi' sbagliato che conviene ricominciare da capo?

    Non e' la soglia del "sospetto", che segnala un risultato da guardare: qui
    si cerca il fallimento vero e proprio dell'aggancio, quello da decine o
    centinaia di pixel. Misurato sulla radiografia di questo progetto: a
    precisione massima il residuo era 161 px con 4 finestre misurabili su 62,
    mentre a precisione normale 1.00 px. Dieci pixel stanno comodamente in mezzo
    e non rischiano di far ripartire una corsa riuscita.
    """
    d = after or {}
    med = d.get('residuo_mediano_px')
    if med is None or not d.get('n_affidabili'):
        return True
    return bool(med > soglia_px)


def _posa_grossolana(fixed_path, moving_path, H, W, mask, lato=2400, verbose=False):
    """L'affine stimata su anteprime molto ridotte, da usare come posa di partenza.

    A che serve. Ogni blocco cerca il suo riscontro entro un quarto del proprio
    lato ALLA SCALA DI LAVORO, cioe' 64*fattore pixel veri. Una differenza di
    scala fra le due riprese produce uno spostamento che cresce verso il bordo,
    quindi solo un disco centrale resta in portata: con il 12% di scala della
    radiografia di prova, a fattore 1 quel disco copre lo 0.3% della superficie
    e l'aggancio fallisce (161 px di residuo). Guardando le immagini piu' in
    piccolo la portata cresce in proporzione: a fattore 8 diventa 512 px veri,
    il disco arriva all'11%, e la stessa coppia aggancia (3.44 px).

    Si e' provata anche la strada elegante, la correlazione log-polare, che
    ricava scala e rotazione globalmente e senza limiti di raggio: calibrata su
    trasformazioni note legge la scala entro lo 0.7% fino a +40%. Ma fra bande
    diverse fallisce proprio dove servirebbe - sulla radiografia leggeva 0.92 di
    scala invece di 0.99 e 13 gradi di rotazione invece di zero - perche' lo
    spettro di una radiografia e quello di una fotografia non si somigliano.
    """
    pf, ff = image_io.leggi_ridotta(fixed_path, lato_lungo_target=lato,
                                    in_grigio=True, verbose=False)
    pm, fm = image_io.leggi_ridotta(moving_path, lato_lungo_target=lato,
                                    in_grigio=True, verbose=False)
    mp = None
    if mask is not None:
        mp = cv2.resize(np.asarray(mask), (pf.shape[1], pf.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    Tg, info = bm.estimate_transform(pf, pm, ff, factor_moving=fm,
                                     nonrigid=False, mask=mp, verbose=verbose)
    Tg.ref_shape = (H, W)
    return Tg, info


def _stima_dalla_posa(Tg, prev_f, prev_m, f_used, f_used_m, H, W, argomenti):
    """La stima completa partendo dalla posa grossolana `Tg`.

    La moving si riporta nel sistema della fixed con `Tg`, la stima misura il
    residuo su quel sistema e le due trasformazioni si compongono. E' il conto
    del ripiego, tenuto in una funzione sola perche' lo fa anche la partenza
    quando l'aggancio dalla traslazione e' fallito in modo riconoscibile.

    Returns:
        (Transform composta, info della stima, dizionario di diagnostica)
    """
    prev_m2 = Tg.at_factor(f_used, f_used_m).apply_to_image(prev_m, prev_f.shape[:2])
    diag2 = {}
    arg = dict(argomenti, coarse_init=False, diagnostica=diag2)
    T2, est2 = bm.estimate_transform(prev_f, prev_m2, f_used, factor_moving=f_used, **arg)
    T2 = Tg.compose_residual(T2.M, delta_field=T2.field, delta_grid_step=T2.grid_step)
    T2.ref_shape = (H, W)
    return T2, est2, diag2


def _scegli_partenza(dubbio, fixed_path, moving_path, H, W, mask, mask_prev,
                     apri_le_due, report):
    """
    La prima passata ha trovato pochi blocchi buoni: l'aggancio dalla
    traslazione e' riuscito o no? Lo si chiede alla verifica, subito.

    Prima il fallimento si scopriva solo in fondo: si completava la stima, la si
    verificava, e sopra i 10 px di residuo scattava il ripiego. Due difetti,
    misurati su un affresco da 593 Mpx le cui bande hanno dimensioni diverse
    dal riferimento (e per le quali la traslazione grossolana esce sbagliata di
    ~200 px):
    - si buttava una stima intera, 5.8 minuti contro 3.0 a precisione massima;
    - un aggancio fallito poteva chiudere SOTTO i 10 px e venire tenuto: con i
      blocchi da 192 px la passata fine portava un primo tentativo da 8 blocchi
      buoni su 88 a 3.9 px di mediana, con il 48% delle finestre oltre 5 px.

    La quota di blocchi buoni da sola non basta a dire che e' fallito. Sulle
    radiografie e' bassa anche quando l'aggancio c'e': 1.3-1.5% a fattore 2,
    dove la stima riesce, e 1.3-1.6% a fattore 1, dove fallisce. Serve quindi
    solo a decidere QUANDO guardare (sotto il 25%; le bande fotografiche stanno
    fra il 28 e l'88%, e per loro non cambia niente).

    Quando si guarda, si mettono a confronto due affini sulle finestre di
    verifica a piena risoluzione: quella della prima passata e quella della
    posa grossolana. Se la prima sbaglia piu' di tre volte la seconda, o non e'
    misurabile affatto, l'aggancio e' fallito e si parte dalla posa. Misurato
    su 30 casi (due dipinti, otto bande, quattro precisioni): nei cinque
    falliti il rapporto va da 5.3 a 12.6 (131-209 px contro 16-33); negli altri
    da 0.9 a 1.9, comprese le radiografie a fattore 2 (1.2 e 1.9), che restano
    alla partenza di sempre.

    Le immagini a piena risoluzione si aprono qui, con `apri_le_due`, e solo
    DOPO aver stimato la posa grossolana: la posa rilegge i due file per
    ridurli, e con ingressi non mappabili (PNG, JPEG) tenerli gia' aperti per
    intero in quel momento alza il picco di memoria senza motivo.

    Returns:
        (posa, arr_f, arr_m): la posa grossolana (Transform) se si deve partire
        da li', oppure None se l'aggancio dalla traslazione regge e la stima va
        completata; e le due immagini aperte, da riusare per la verifica.
    """
    T_trasl, info_trasl, messaggio = dubbio
    p1 = info_trasl.get('pass1', {})
    print(f"\n  [aggancio] prima passata: {p1.get('inlier', 0)} blocchi buoni su "
          f"{p1.get('possibili', 0)} posizioni ({100 * p1.get('quota_inlier', 0):.1f}%): "
          f"si controlla se l'aggancio ha preso.")
    esito = {'quota_inlier': p1.get('quota_inlier'), 'inlier': p1.get('inlier'),
             'possibili': p1.get('possibili'), 'partenza': 'traslazione'}
    report['aggancio'] = esito
    try:
        Tg, _ = _posa_grossolana(fixed_path, moving_path, H, W, mask, verbose=False)
    except Exception as exc:
        esito['errore_posa'] = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if T_trasl is None:
            raise RuntimeError(messaggio) from exc
        print("      la posa grossolana non si e' potuta stimare: si prosegue.")
        return (None,) + tuple(apri_le_due())
    arr_f, arr_m = apri_le_due()

    def misura(T):
        T.ref_shape = (H, W)
        r = val.residual_check(arr_f, arr_m, T, mask=mask_prev, verbose=False, adattiva=False)
        return r.get('residuo_mediano_px'), int(r.get('n_affidabili') or 0)

    m_g, n_g = misura(Tg)
    m_t, n_t = misura(T_trasl) if T_trasl is not None else (None, 0)
    esito.update(residuo_affine_traslazione_px=m_t, finestre_traslazione=n_t,
                 residuo_affine_posa_px=m_g, finestre_posa=n_g)

    def testo(m, n):
        return "non misurabile" if m is None else f"{m:.0f} px su {n} finestre"

    print(f"      affine dalla traslazione: {testo(m_t, n_t)}; "
          f"affine della posa grossolana: {testo(m_g, n_g)}")
    fallito = (m_t is None) or (m_g is not None and m_t > RAPPORTO_AGGANCIO_FALLITO * m_g)
    if not fallito:
        print("      l'aggancio regge: si completa la stima.")
        return None, arr_f, arr_m
    esito['partenza'] = 'posa grossolana'
    print("      l'aggancio dalla traslazione e' fallito: si parte dalla posa grossolana.")
    return Tg, arr_f, arr_m


def _ripiego_se_serve(T, est_info, diag_stima, after, ctx, report, chiave='ripiego'):
    """Se l'aggancio e' fallito del tutto, riparte da una posa grossolana.

    La posa si stima su anteprime molto ridotte, dove la portata dei blocchi in
    pixel veri e' molto piu' grande. Non scatta mai su una corsa riuscita - la
    soglia e' a 10 px, e i fallimenti veri stanno sulle decine o centinaia -
    quindi per le bande che gia' funzionano non cambia niente. Misurato sulla
    radiografia a precisione grossolana: da 161.18 a 2.25 px.

    Sta in una funzione a se' perche' va offerto a OGNI candidato posa. Quando
    la doppia posa confronta la posa dai punti con quella automatica, dare il
    ripiego solo alla prima significherebbe mettere a confronto una posa
    soccorsa con una lasciata a se stessa, e il confronto direbbe piu' su quale
    delle due e' stata aiutata che su quale e' migliore.

    Torna sempre una quaterna utilizzabile: se il ripiego non serve, non
    riesce, o esce peggiore, torna quella di partenza intatta.
    """
    if not _posa_da_rifare(after):
        return T, est_info, diag_stima, after
    med = after.get('residuo_mediano_px')
    med_txt = "nessuna misura" if med is None else ("%.0f px" % med)
    print(f"\n  [{chiave}] residuo di {med_txt}: l'aggancio e' fallito, "
          "si ricomincia da una posa grossolana.")
    try:
        Tg, _ = _posa_grossolana(ctx['fixed_path'], ctx['moving_path'],
                                 ctx['H'], ctx['W'], ctx['mask'], verbose=False)
        T2, est2, diag2 = _stima_dalla_posa(
            Tg, ctx['prev_f'], ctx['prev_m'], ctx['f_used'], ctx['f_used_m'],
            ctx['H'], ctx['W'],
            dict(block=ctx['block'], step=ctx['step'], min_conf=ctx['min_conf'],
                 nonrigid=ctx['nonrigid'], refine_block=ctx['refine_block'],
                 refine_step=ctx['refine_step'], mask=ctx['mask_prev'],
                 verbose=ctx['verbose'],
                 campo_grossolano_tra_le_passate=ctx.get('tre_scale', False)))
        # compose_residual eredita la meta di Tg, che e' vuota: senza questo una
        # corsa salvata dal ripiego perdeva il fatto stesso di aver usato una
        # maschera.
        T2.meta = dict(T.meta, estimazione=est2)
        after2 = val.residual_check(ctx['arr_f'], ctx['arr_m'], T2,
                                    mask=ctx['mask_prev'], verbose=False)
        m1, m2 = after.get('residuo_mediano_px'), after2.get('residuo_mediano_px')
        meglio = (m2 is not None and (m1 is None or m2 < m1))
        print(f"      prima {('-' if m1 is None else f'{m1:.2f} px')}, "
              f"col ripiego {('-' if m2 is None else f'{m2:.2f} px')}: "
              + ("si tiene il ripiego." if meglio else "si tiene la prima."))
        report[chiave] = {
            'usato': bool(meglio),
            'residuo_prima_px': m1, 'residuo_dopo_px': m2,
            'scala_grossolana': float(np.hypot(Tg.M[0, 0], Tg.M[1, 0])),
        }
        if meglio:
            print(f"      dopo il ripiego: {m2:.2f} px su "
                  f"{after2.get('n_affidabili')}/{after2.get('n_campioni')} finestre")
            return T2, est2, diag2, after2
    except Exception as exc:
        print(f"      il ripiego non e' riuscito: {exc}")
        report[chiave] = {'usato': False, 'errore': str(exc)}
    return T, est_info, diag_stima, after


def _porta_punti(T, xy):
    """Dove la trasformazione manda i punti del riferimento, campo compreso."""
    p = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))]) @ T.M.T
    if T.field is None:
        return p
    # cv2.remap rifiuta mappe con piu' di 32767 righe: un contorno a piena
    # risoluzione ne ha decine di migliaia, quindi i punti si rimodellano in un
    # rettangolo con i lati sotto il limite.
    n = len(xy)
    larg = min(n, 16384)
    alt = int(np.ceil(n / larg))
    riempi = alt * larg - n
    gx = np.concatenate([(xy[:, 0] / T.grid_step).astype(np.float32),
                         np.zeros(riempi, np.float32)]).reshape(alt, larg)
    gy = np.concatenate([(xy[:, 1] / T.grid_step).astype(np.float32),
                         np.zeros(riempi, np.float32)]).reshape(alt, larg)
    fld = cv2.remap(T.field, gx, gy, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return p + fld.reshape(-1, 2)[:n]


def _riassunto_tre(errori, lascia_fuori):
    """Mediana, quote, media e distorsione di un insieme di errori ai punti."""
    e = np.asarray(errori, dtype=np.float64).reshape(-1, 2)
    m = np.hypot(e[:, 0], e[:, 1])
    return {'n': int(len(m)), 'lascia_fuori': bool(lascia_fuori),
            'mediana_px': float(np.median(m)), 'media_px': float(m.mean()),
            'quota_sopra_5px': float((m > 5).mean()),
            'quota_sopra_10px': float((m > 10).mean()),
            'distorsione_px': [float(e[:, 0].mean()), float(e[:, 1].mean())],
            'per_punto_px': [round(float(x), 2) for x in m]}


def _tre_lascia_fuori(P, residuo_v, forma, passo, misure, forma_campo):
    """
    Il TRE (target registration error) sui punti messi a mano, onesto.

    I punti usati come vincolo vengono attraversati esattamente dal campo,
    quindi lo scarto su di loro e' zero per costruzione e non giudica niente.
    Qui ogni punto si giudica con la correzione rifatta SENZA di lui: e'
    l'errore che la trasformazione fa in un posto dove non le si e' detto
    niente. Costa una correzione per punto, cioe' pochi secondi.

    Vale per la parte locale. Se i punti danno anche la posa di partenza, la
    posa li ha visti tutti, e il numero e' ottimista: chi lo usa lo segnala.
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 4)
    V = np.asarray(residuo_v, dtype=np.float64).reshape(-1, 2)
    errori = []
    for i in range(len(P)):
        k = np.arange(len(P)) != i
        campo_i, _ = punti_manuali.campo_dai_punti(P[k], V[k], forma, passo, misure)
        # Lo stesso adattamento alla griglia della trasformazione che si fa
        # applicando la correzione vera.
        if forma_campo is not None and campo_i.shape[:2] != tuple(forma_campo):
            campo_i = cv2.resize(campo_i, (forma_campo[1], forma_campo[0]),
                                 interpolation=cv2.INTER_LINEAR)
        g = np.array([[P[i, 0] / passo]], np.float32), np.array([[P[i, 1] / passo]], np.float32)
        c = cv2.remap(campo_i.astype(np.float32), g[0], g[1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE).reshape(2)
        errori.append(V[i] - c)
    return _riassunto_tre(errori, lascia_fuori=True)


def _correggi_con_punti(T, punti, arr_f, arr_m, after, mask_prev, report):
    """Compone sulla trasformazione il campo che la fa passare per i punti.

    Si applica solo dove la verifica non ha potuto misurare: dove misura, lo fa
    a frazioni di pixel e un clic da 5-10 px sarebbe un peggioramento.
    """
    P = np.asarray(punti, dtype=float).reshape(-1, 4)
    residuo_v = P[:, 2:4] - _porta_punti(T, P[:, :2])
    misure = np.asarray(after.get('campioni') or []).reshape(-1, 5)[:, :2]
    H, W = arr_f.shape[:2]
    passo = float(T.grid_step) if T.field is not None else 256.0
    campo, info = punti_manuali.campo_dai_punti(P, residuo_v, (H, W), passo, misure)
    # Il campo e la maschera dei punti usati servono alla diagnostica ma non
    # sanno stare in un JSON: restano fuori dal report, che viene salvato su
    # disco. Lasciarceli dentro faceva fallire il salvataggio dell'intera corsa
    # ogni volta che la correzione non veniva applicata.
    usati = info.pop('_usati', np.ones(len(P), bool))
    info['scarto_prima_px'] = float(np.median(np.hypot(*residuo_v.T)))

    if info.get('n_usati', 0) < 3 or info.get('correzione_max_px', 0) < 0.5:
        info['applicata'] = False
        # Senza correzione i punti non entrano nella trasformazione: lo scarto
        # che hanno e' gia' un errore misurato da un giudice indipendente.
        info['tre'] = _riassunto_tre(residuo_v, lascia_fuori=False)
        report['correzione_punti'] = info
        return T, info, None, usati
    info['tre'] = _tre_lascia_fuori(P, residuo_v, (H, W), passo, misure,
                                    T.field.shape[:2] if T.field is not None else None)

    if T.field is None:
        T.field, T.grid_step = campo, passo
    else:
        if campo.shape[:2] != T.field.shape[:2]:
            campo = cv2.resize(campo, (T.field.shape[1], T.field.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        T.field = (T.field + campo).astype(np.float32)
    info['applicata'] = True
    info['scarto_dopo_px'] = float(np.median(np.hypot(
        *(P[:, 2:4] - _porta_punti(T, P[:, :2])).T)))
    report['correzione_punti'] = info

    print(f"\n  [punti come vincolo] {info['n_usati']} punti su {info['n_punti']}"
          + (f" ({info['n_esclusi']} esclusi perche' incoerenti con i vicini: "
             f"{', '.join(map(str, info['esclusi']))})" if info['n_esclusi'] else "")
          + f"\n      correzione applicata sul {100 * info['quota_superficie']:.0f}% "
            f"della superficie, dove la verifica non misura: mediana "
            f"{info['correzione_mediana_px']:.1f} px, massima "
            f"{info['correzione_max_px']:.0f} px"
          + f"\n      scarto ai punti: {info['scarto_prima_px']:.1f} -> "
            f"{info['scarto_dopo_px']:.1f} px")
    # Sotto la precisione del clic non si scende: dirlo evita un giro in piu' a
    # inseguire pixel che l'occhio non sa mettere.
    if info['scarto_prima_px'] < 6.0:
        print(f"      [nota] lo scarto era gia' sotto i 6 px, cioe' dell'ordine "
              f"dell'imprecisione di un clic: un altro giro di punti non "
              f"migliorera' granche'.")
    return T, info, campo, usati


def _infittisci_campo(T, k, forma):
    """Porta il campo su una griglia k volte piu' fitta, senza cambiarlo.

    Il campo e' interpolato in modo bilineare fra i nodi, quindi ricampionarlo
    sui nodi di una griglia che contiene quelli vecchi descrive esattamente la
    stessa deformazione: serve solo a far posto a un vincolo piu' minuto.
    """
    from scipy.ndimage import map_coordinates
    H, W = forma
    passo = float(T.grid_step) / k
    gh, gw = int(np.ceil(H / passo)) + 1, int(np.ceil(W / passo)) + 1
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64) / k
    T.field = np.stack([map_coordinates(T.field[..., c].astype(np.float64), [gy, gx],
                                        order=1, mode='nearest')
                        for c in range(2)], axis=-1).astype(np.float32)
    T.grid_step = passo


def _correggi_con_fascia(T, contorno_moving, mask, forma, larghezza, report):
    """Porta il contorno della zona sul bordo della moving, in una fascia sottile."""
    H, W = forma
    G = punti_manuali.contorno_da_maschera(mask, (H, W), passo_px=10.0)
    if G is None or T.field is None:
        report['fascia_di_bordo'] = {'applicata': False,
                                     'motivo': 'maschera senza contorno o stima senza campo'}
        return T, report['fascia_di_bordo']
    # Il raccordo deve avere almeno quattro nodi di griglia per lato.
    k = int(np.ceil(float(T.grid_step) / (larghezza / 4.0)))
    if k > 1:
        _infittisci_campo(T, k, (H, W))
    campo, info = punti_manuali.campo_fascia_di_bordo(
        G, contorno_moving, lambda xy: _porta_punti(T, xy), (H, W), float(T.grid_step),
        larghezza=larghezza)
    if campo is None:
        info['applicata'] = False
        report['fascia_di_bordo'] = info
        return T, info
    T.field = (T.field + campo).astype(np.float32)
    # quanto resta fra le due linee, misurato come prima della correzione
    Gd = punti_manuali._ricampiona_chiuso(G, 10.0)
    n = punti_manuali.normali_esterne(Gd)
    Cd = punti_manuali._ricampiona_chiuso(contorno_moving, 5.0)
    from scipy.spatial import cKDTree
    P = _porta_punti(T, Gd)
    j = cKDTree(Cd).query(P)[1]
    info['scarto_bordo_dopo_px'] = float(np.median(np.abs(((Cd[j] - P) * n).sum(axis=1))))
    info['passo_campo_px'] = float(T.grid_step)
    info['applicata'] = True
    report['fascia_di_bordo'] = info
    T.meta['fascia_di_bordo'] = {k_: v for k_, v in info.items() if k_ != 'confronto'}
    print('')
    print(f"  [fascia di bordo] {larghezza:.0f} px verso l'interno; distanza fra le "
          f"due linee {info['scarto_bordo_prima_px']:.0f} -> "
          f"{info['scarto_bordo_dopo_px']:.1f} px (mediana); correzione massima "
          f"{info['correzione_max_px']:.0f} px; campo infittito a "
          f"{T.grid_step:.0f} px")
    return T, info


def _confronta_stesse_finestre(prima, dopo):
    """Il confronto onesto fra due verifiche: solo le finestre misurabili in
    entrambe.

    La mediana complessiva non e' confrontabile, perche' una correzione che
    funziona rende misurabili finestre prima cieche - le piu' difficili - e
    quelle alzano la mediana proprio mentre l'allineamento migliora.
    """
    a = {(int(r[0]), int(r[1])): float(np.hypot(r[2], r[3]))
         for r in (prima.get('campioni') or [])}
    b = {(int(r[0]), int(r[1])): float(np.hypot(r[2], r[3]))
         for r in (dopo.get('campioni') or [])}
    comuni = sorted(set(a) & set(b))
    if not comuni:
        return None
    va = np.array([a[k] for k in comuni])
    vb = np.array([b[k] for k in comuni])
    esito = {
        'n_finestre_comuni': len(comuni),
        'residuo_prima_px': float(np.median(va)),
        'residuo_dopo_px': float(np.median(vb)),
        'peggiorate': int((vb > va + 0.2).sum()),
        'migliorate': int((vb < va - 0.2).sum()),
        'misurabili_prima': len(a),
        'misurabili_dopo': len(b),
    }
    print(f"    confronto sulle {len(comuni)} finestre misurabili in entrambe: "
          f"{esito['residuo_prima_px']:.2f} -> {esito['residuo_dopo_px']:.2f} px, "
          f"{esito['peggiorate']} peggiorate, {esito['migliorate']} migliorate")
    print(f"    finestre misurabili in tutto: {len(a)} -> {len(b)}")
    return esito


def margine_necessario(T, forma_moving, forma_ref, passo=100, tetto=0.10):
    """Quanti pixel di foglio servono perche' l'immagine allineata non venga tagliata.

    Si guarda dove finisce, dopo la trasformazione, il FOTOGRAMMA dell'immagine
    da allineare: e' pura geometria, non serve riconoscere niente nell'immagine.
    E' abbondante per costruzione, perche' il fotogramma contiene anche lo
    sfondo: sulla radiografia Cappellochiaro esce fino a 458 px, mentre la tela
    tracciata a mano ne esce al piu' 56. Su una banda con le stesse dimensioni
    del riferimento (NIR850) esce di pochi pixel, e solo con sfondo.

    Si arrotonda per eccesso a `passo` px, cosi' tentativi successivi sulla
    stessa coppia danno di solito lo stesso margine (e la stessa copia del
    riferimento), con un tetto a `tetto` del lato lungo del riferimento.
    """
    Hm, Wm = forma_moving
    H, W = forma_ref
    t = np.linspace(0.0, 1.0, 400)
    q = np.vstack([np.c_[t * (Wm - 1), 0 * t], np.c_[t * (Wm - 1), 0 * t + Hm - 1],
                   np.c_[0 * t, t * (Hm - 1)], np.c_[0 * t + Wm - 1, t * (Hm - 1)]])
    # dal fotogramma della moving al sistema del riferimento: si inverte T per
    # iterazione (il campo non ha un'inversa in forma chiusa)
    Ai = np.linalg.inv(T.M[:, :2])
    p = (Ai @ (q - T.M[:, 2]).T).T
    for _ in range(30):
        p = p + (Ai @ (q - _porta_punti(T, p)).T).T
    fuori = max(0.0, -p[:, 0].min(), -p[:, 1].min(),
                p[:, 0].max() - (W - 1), p[:, 1].max() - (H - 1))
    if fuori <= 0:
        return 0
    m = int(np.ceil(fuori / passo) * passo)
    return int(min(m, np.ceil(tetto * max(H, W) / passo) * passo))


def scrivi_con_margine(path_in, out_path, margine, tile_size=2048):
    """Copia di un'immagine con `margine` pixel neri in piu' per lato.

    Serve a sovrapporre al riferimento un'immagine allineata scritta con un
    margine: le due devono avere lo stesso foglio. Scrive sempre un TIFF non
    compresso, a fasce di righe, senza caricare l'immagine intera.
    """
    arr = image_io.apri(path_in, verbose=False)[0]
    H, W = arr.shape[:2]
    m = int(margine)
    samples = 1 if arr.ndim == 2 else arr.shape[2]
    H2, W2 = H + 2 * m, W + 2 * m
    scrittore = tiff_memmap.create_tiff_memmap(out_path, H2, W2, samples=samples,
                                               dtype=arr.dtype, sequenziale=True)
    for y0 in range(0, H2, tile_size):
        th = min(tile_size, H2 - y0)
        fascia = np.zeros((th, W2) if samples == 1 else (th, W2, samples), dtype=arr.dtype)
        ys, ye = max(y0, m), min(y0 + th, m + H)
        if ye > ys:
            fascia[ys - y0:ye - y0, m:m + W] = np.asarray(arr[ys - m:ye - m])
        scrittore.scrivi(fascia)
    scrittore.chiudi()
    return out_path


def scrivi_allineata(T, arr_m, out_path, H, W, samples, dtype, tile_size=2048,
                     progress=None, margine=0):
    """
    Scrive l'immagine allineata a piena risoluzione, in streaming.

    Sta fuori da align() perche' serve anche da sola: una trasformazione gia'
    stimata si puo' riapplicare senza rifare la stima, per esempio quando si
    vuole scrivere il risultato di un tentativo che si era solo guardato.

    Con `margine` > 0 il foglio e' piu' grande del riferimento di `margine`
    pixel per lato: il pixel (x, y) del riferimento finisce in (x+m, y+m), e
    cio' che della moving cade fuori dal riferimento non viene tagliato.
    """
    t0 = time.time()
    m = int(margine)
    H, W = H + 2 * m, W + 2 * m          # il foglio di uscita

    def avanzamento(testo, frazione):
        if progress:
            progress(testo, frazione)

    avanzamento("scrittura dell'immagine allineata", 0.55)
    # I tile si producono per fasce di righe, dall'alto in basso: una fascia
    # completa si puo' scrivere in sequenza, senza pre-dimensionare il file.
    # Su NTFS pre-dimensionare 900 MB costa 14-42 s (il sistema li riempie di
    # zeri che poi riscriviamo), e la riscrittura via memmap sporca pagine che
    # vengono riversate in ordine sparso: misurato, 77-125 s contro 4-27 s.
    # Il prezzo e' una fascia in RAM, ~100 MB con tile da 2048.
    sequenziale = os.path.splitext(out_path)[1].lower() in ('.tif', '.tiff')
    if sequenziale:
        scrittore = tiff_memmap.create_tiff_memmap(out_path, H, W, samples=samples,
                                                   dtype=dtype, sequenziale=True)
    else:
        out, memmappato = image_io.crea_output(out_path, H, W, samples=samples,
                                               dtype=dtype)

    total = ((W + tile_size - 1) // tile_size) * ((H + tile_size - 1) // tile_size)
    done = 0
    for y0 in range(0, H, tile_size):
        th = min(tile_size, H - y0)
        forma = (th, W) if samples == 1 else (th, W, samples)
        fascia = np.empty(forma, dtype=dtype) if sequenziale else None
        for x0 in range(0, W, tile_size):
            tw = min(tile_size, W - x0)
            tile = T.warp_tile_from(arr_m, y0 - m, x0 - m, th, tw)
            if sequenziale:
                fascia[:, x0:x0 + tw] = tile
            else:
                out[y0:y0 + th, x0:x0 + tw] = tile
            done += 1
        if sequenziale:
            scrittore.scrivi(fascia)
        # le pagine gia' lette dell'immagine da allineare tornano al sistema
        image_io.rilascia_pagine()
        elapsed = time.time() - t0
        print(f"      tile {done}/{total}  ({100.0 * done / total:.0f}%)  "
              f"{human_time(elapsed)} trascorsi, "
              f"~{human_time(elapsed / done * (total - done))} rimanenti", end='\r', flush=True)
        avanzamento("scrittura dell'immagine allineata", 0.55 + 0.45 * done / total)
    if sequenziale:
        scrittore.chiudi()
    else:
        image_io.salva_output(out_path, out, memmappato)
        del out
    print(f"\n      scritto {out_path} ({os.path.getsize(out_path) / 1024 ** 3:.2f} GB) "
          f"in {human_time(time.time() - t0)}")


def _print_footer(t_start, before, after):
    print(f"\nTempo totale: {human_time(time.time() - t_start)}")
    mem = peak_memory_gb()
    if mem is not None:
        print(f"Picco di memoria privata: {mem[0]:.2f} GB "
              f"(working set, incluse le pagine dei file mappati: {mem[1]:.2f} GB)")
    print("=" * 72)
    _print_verdict(before, after)


def _mappa_vincoli(campo, punti, usati, prev_f, forma_piena, path):
    """Dove sono i punti messi a mano e quanto correggono, su un'anteprima.

    Serve perche' la correzione dai punti agisce proprio dove la verifica e'
    cieca: li' nessun numero puo' dire se ha funzionato, e l'unico giudice e'
    l'occhio. Senza questa mappa non si vede nemmeno DOVE il programma ha messo
    i vincoli, e non si puo' giudicare niente.

    Verde = punto usato come vincolo, rosso = scartato perche' in contrasto con
    i vicini. Il colore di sfondo e' l'ampiezza della correzione, su una scala
    FISSA 0-40 px: un colore che vuol dire sempre la stessa cosa fra una corsa e
    l'altra e' piu' utile di un contrasto massimizzato che cambia significato
    ogni volta.
    """
    if campo is None or prev_f is None:
        return
    H, W = forma_piena
    base = prev_f if prev_f.ndim == 3 else cv2.cvtColor(prev_f, cv2.COLOR_GRAY2RGB)
    # Ridotta a 1600 px di lato: e' un'immagine da guardare, e a piena
    # anteprima pesava 72 MB contro i 3-4 degli altri overlay.
    scala = min(1.0, 1600.0 / max(base.shape[:2]))
    if scala < 1.0:
        base = cv2.resize(base, (int(base.shape[1] * scala),
                                 int(base.shape[0] * scala)),
                          interpolation=cv2.INTER_AREA)
    h, w = base.shape[:2]
    amp = np.hypot(campo[:, :, 0], campo[:, :, 1])
    norm = np.clip(cv2.resize(amp, (w, h), interpolation=cv2.INTER_LINEAR)
                   / 40.0 * 255.0, 0, 255).astype(np.uint8)
    img = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.45,
                          cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), 0.55, 0)

    P = np.asarray(punti, dtype=float).reshape(-1, 4)
    for i, (x, y) in enumerate(P[:, :2]):
        px, py = int(round(x * w / W)), int(round(y * h / H))
        col = (80, 230, 80) if usati[i] else (60, 60, 240)
        cv2.drawMarker(img, (px, py), col, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, str(i + 1), (px + 7, py - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1, cv2.LINE_AA)
    for testo, y in ((f"correzione dai punti: max {amp.max():.0f} px "
                      f"(colore: 0-40 px)", 24),
                     (f"verde = vincolo usato ({int(np.sum(usati))}), "
                      f"rosso = scartato ({int(np.sum(~np.asarray(usati)))})", 48)):
        cv2.putText(img, testo, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, testo, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    image_io.scrivi_immagine(path, img)


def _save_field_map(T, path):
    """Mappa a colori dell'ampiezza del campo non rigido (diagnostica)."""
    if T.field is None:
        return
    mag = np.hypot(T.field[:, :, 0], T.field[:, :, 1])
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"max {mag.max():.1f} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    image_io.scrivi_immagine(path, img)


def _print_verdict(before, after):
    b = before.get('residuo_mediano_px')
    a = after.get('residuo_mediano_px')
    if b is None or a is None:
        return
    ic = after.get('residuo_mediano_ic90')
    intervallo = f" [{ic[0]:.2f}-{ic[1]:.2f}]" if ic else ""
    q5 = after.get('quota_sopra_5px')
    coda = f", {100 * q5:.0f}% delle finestre oltre 5 px" if q5 else ""
    print(f"ESITO: disallineamento mediano {b:.1f} px  ->  {a:.2f} px{intervallo}{coda}")
    if a < 1.5:
        print("       allineamento sub-pixel / al pixel: ottimo.")
    elif a < 4:
        print("       allineamento buono (pochi px). Per migliorare: --factor 2.")
    else:
        print("       residuo ancora sensibile: controlla gli overlay in diagnostica/ "
              "e valuta --factor 2 o blocchi piu' piccoli (--refine-block 96).")


def main():
    p = argparse.ArgumentParser(
        description="Allinea due immagini enormi dello stesso dipinto (es. VIS e UV).")
    p.add_argument('--fixed', required=True, help="immagine di riferimento (es. VIS)")
    p.add_argument('--moving', required=True, help="immagine da allineare (es. UV)")
    p.add_argument('--out', default=None,
                   help="TIFF di output allineato; se omesso, esegue solo stima e verifica")
    p.add_argument('--params', default='transform_params.json',
                   help="dove salvare i parametri della trasformazione")
    p.add_argument('--factor', type=int, default=2,
                   help="fattore di riduzione per la stima (2 = default, sub-pixel; 4 = piu' veloce)")
    p.add_argument('--no-nonrigid', action='store_true',
                   help="stima solo l'affine, niente campo di deformazione locale")
    p.add_argument('--tile', type=int, default=2048, help="dimensione dei tile in scrittura")
    p.add_argument('--block', type=int, default=256, help="lato dei blocchi, prima passata")
    p.add_argument('--refine-block', type=int, default=128, help="lato dei blocchi, seconda passata")
    p.add_argument('--min-conf', type=float, default=0.10,
                   help="confidenza minima di un blocco per essere usato")
    p.add_argument('--diagnostica', default='diagnostica', help="cartella per le immagini di controllo")
    args = p.parse_args()

    align(args.fixed, args.moving, args.out, args.params,
          factor=args.factor, nonrigid=not args.no_nonrigid, tile_size=args.tile,
          block=args.block, step=args.block // 2,
          refine_block=args.refine_block, refine_step=args.refine_block // 2,
          min_conf=args.min_conf, diagnostics_dir=args.diagnostica)


if __name__ == '__main__':
    main()

