"""
Allinea un gruppo di immagini sulla stessa immagine di riferimento.

    python align_batch.py --fixed immagini\\VIS.tif ^
                          --immagini immagini\\UVF.tif immagini\\IR.tif ^
                          --out-dir risultati_lotto

Ogni immagine riceve la SUA trasformazione, stimata contro il riferimento: non
si riusano i parametri di un'altra banda, perche' posa relativa, flessione della
tela e magnificazione dipendente dalla lunghezza d'onda cambiano da uno scatto
all'altro (vedi README, "Quando una trasformazione NON si riusa").

Tenendo lo stesso riferimento per tutte, i risultati finiscono tutti nello
stesso sistema di coordinate e sono quindi confrontabili pixel a pixel fra loro.

Il programma NON ha bisogno di sapere a che lunghezza d'onda corrisponde ogni
immagine: il matching lavora sulla struttura (gradienti e fase), non sui livelli
di grigio, quindi funziona allo stesso modo su UV, IR o qualsiasi altra banda.
I nomi servono solo a nominare i file di uscita.

Un'immagine che fallisce non ferma il lotto: viene segnalata nel riepilogo
finale con il motivo, e le altre proseguono.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import align_paintings
import verifica_fissa as vf
import formati as fmt
import image_io


def valuta_qualita(verifica, soglia):
    """
    Decide se un risultato e' affidabile, guardando piu' di un numero solo.

    La sola mediana del residuo non basta, e su una radiografia si e' visto
    perche': mediana sotto soglia, quindi "ok", ma con pochissime finestre
    affidabili e una coda di errori da decine di pixel. Cioe': preciso in quattro
    punti, sbagliato dappertutto altrove. Un risultato del genere va segnalato.

    I tre criteri, tutti sullo stato DOPO l'allineamento (che e' sempre
    interpretabile, a differenza del confronto col "prima" quando le immagini
    hanno dimensioni diverse):

    1. quante finestre danno una misura affidabile: se sono poche, le due
       immagini condividono poca struttura e la stima poggia sul nulla;
    2. la mediana del residuo;
    3. la QUOTA di finestre che sbagliano oltre 5 px. Prima si usava il p95
       rispetto alla mediana: si e' poi misurato che con poche finestre il p95
       non e' un percentile ma quasi il massimo, e oscilla del doppio spostando
       di poco la griglia. Una frequenza e' stabile.

    Returns:
        (esito, motivo) con esito in {'ok', 'sospetto'}
    """
    dopo = verifica.get('dopo', {})
    n_tot = dopo.get('n_campioni') or 0
    n_ok = dopo.get('n_affidabili') or 0
    mediano = dopo.get('residuo_mediano_px')
    quota5 = dopo.get('quota_sopra_5px')

    if not n_ok or mediano is None:
        return 'sospetto', ("nessun campione affidabile: le due immagini non condividono "
                            "abbastanza struttura perche' la verifica dia un risultato")
    if n_tot and n_ok < 0.5 * n_tot:
        # Va distinto il caso "allineamento probabilmente sbagliato" da quello
        # "non posso verificarlo": su una radiografia le finestre misurabili
        # sono poche per forza, perche' condivide poca struttura con il
        # visibile, e continuare a dire "sospetto" senza spiegare fa pensare a
        # un errore che non c'e'. Dove si misura, il residuo e' quello che e'.
        if mediano <= soglia:
            return 'sospetto', (
                f"verificabile solo in parte: misurabili {n_ok} finestre su {n_tot} "
                f"(le due immagini condividono poca struttura, tipico delle "
                f"radiografie). Dove si misura il residuo e' {mediano:.2f} px, "
                f"ma sul resto dell'immagine non c'e' prova: guarda "
                f"diagnostica/overlay_falsocolore.png")
        return 'sospetto', (f"solo {n_ok} finestre affidabili su {n_tot}, e dove si "
                            f"misura il residuo e' {mediano:.2f} px: la corrispondenza "
                            f"fra le due immagini regge solo in alcune zone")
    if mediano > soglia:
        return 'sospetto', f"residuo {mediano:.2f} px sopra la soglia di {soglia:.1f} px"
    # Coda: si usa la QUOTA di finestre molto sbagliate, non il p95. Con poche
    # finestre il p95 e' quasi il massimo e oscilla del doppio da una griglia
    # all'altra; una frequenza no.
    if quota5 is not None and quota5 > 0.05:
        return 'sospetto', (f"mediana buona ({mediano:.2f} px) ma il "
                            f"{100 * quota5:.0f}% delle finestre sbaglia oltre 5 px: "
                            f"allineato a macchie")
    return 'ok', ''


def _nome_output(path, cartella, suffisso, estensione=None, momento=None):
    """
    Nome del file allineato: <originale><suffisso> (data ora).<estensione>

    La data e l'ora servono a non sovrascrivere il risultato di un'elaborazione
    precedente sulla stessa immagine: rilanciando con impostazioni diverse si
    ottengono due file affiancati e confrontabili, invece di perdere il primo.

    L'ora usa il trattino e non i due punti, che Windows non ammette nei nomi
    di file.
    """
    base, est = os.path.splitext(os.path.basename(path))
    est = estensione or est
    if est.lower() not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
        est = '.tif'
    quando = (momento or datetime.now()).strftime('%Y-%m-%d %H-%M')
    return os.path.join(cartella, f"{base}{suffisso} ({quando}){est}")


def allinea_lotto(riferimento, immagini, cartella_output,
                  factor=2, nonrigid=True, mask=None, mask_info=None,
                  tile_size=2048,
                  min_conf=0.10, refine_block=128, suffisso='_allineata',
                  estensione=None, formati=None, soglia_sospetto=3.0,
                  scrivi_immagini=True,
                  margine=0, tre_scale='auto', strumenti=None, verifica_fissa=True,
                  copie_tiff=True, progress=None, verbose=True):
    """
    Allinea `immagini` sul `riferimento`, salvando tutto in `cartella_output`.

    Args:
        strumenti: gli strumenti a mano, immagine per immagine: un dizionario
            {percorso dell'immagine: {...}} con, a scelta, 'punti' ((N, 4)
            [x_rif, y_rif, x_imm, y_imm]), 'punti_come_vincolo',
            'punti_come_posa', 'contorno' (il bordo tracciato sull'immagine) e
            'fascia_di_bordo' (larghezza in px, 0 = spenta). Erano la sezione a
            parte delle radiografie; servono a qualunque banda che l'automatico
            non porta a posto ai bordi. Un'immagine che non compare qui va in
            automatico, con la stessa chiamata di sempre: i suoi numeri non
            cambiano.
        verifica_fissa: aggiunge a ogni immagine la verifica a griglia fissa
            (verifica_fissa.py), accanto a quella di sempre che resta il numero
            principale. Costa 10-40 s per immagine.
        copie_tiff: le immagini in un formato che non si legge a pezzi (PNG,
            JPEG, TIFF compresso) si leggono, per la durata dell'analisi, da
            una copia TIFF non compressa nella cartella temporanea del sistema
            (image_io.copia_tiff_temporanea). La copia di ogni immagine si
            toglie appena quell'immagine e' finita, quella del riferimento alla
            fine del lotto; i file prodotti restano nel formato chiesto.
        progress: callback f(indice, totale, testo, frazione) per l'interfaccia.
        soglia_sospetto: sopra questo residuo mediano (px) il risultato viene
            marcato come sospetto nel riepilogo. Il file viene comunque salvato:
            sta a te decidere se tenerlo.
        formati: formati in cui salvare le immagini allineate, piu' d'uno se si
            vuole ('TIFF', 'PNG', 'JPEG' o le estensioni). Vuoto o None = solo il
            formato dell'originale. Il primo si scrive a tile (il TIFF, se c'e'),
            gli altri si ricavano da quello: vedi formati.py.
        estensione: il modo di prima per chiedere un solo formato; con
            `formati` non serve.
        scrivi_immagini: False esegue solo stima e verifica (utile per provare
            le impostazioni su un lotto senza scrivere gigabyte).
        tre_scale: 'auto' (il default) stima anche il campo grossolano e lo
            applica solo dove la deformazione supera la portata della passata
            fine; False torna alle due passate. Vedi align_paintings.align.
        margine: pixel di foglio in piu' per lato nelle immagini allineate, uguale
            per tutto il lotto, o 'auto' per calcolarlo: il massimo, fra le
            immagini, di quanto il loro fotogramma esce da quello del riferimento
            (align_paintings.margine_necessario). La copia del riferimento con
            lo stesso margine non si scrive piu' (vedi sotto). Con 'auto' il
            lotto va in due tempi: prima stima e verifica di tutte, poi la
            scrittura, quando il margine e' noto. 0 (il default) = come sempre.

    Returns:
        dizionario di riepilogo, salvato anche come riepilogo.json.
    """
    t_start = time.time()
    # Un solo istante per tutto il lotto: le immagini di una stessa elaborazione
    # portano cosi' la stessa etichetta e si riconoscono a colpo d'occhio, invece
    # di differire di qualche minuto l'una dall'altra.
    momento = datetime.now()
    os.makedirs(cartella_output, exist_ok=True)
    cartella_par = os.path.join(cartella_output, 'parametri')
    cartella_diag = os.path.join(cartella_output, 'diagnostica')
    os.makedirs(cartella_par, exist_ok=True)
    os.makedirs(cartella_diag, exist_ok=True)

    if copie_tiff:
        image_io.copia_tiff_temporanea(riferimento)
    info_rif = image_io.info(riferimento)
    print("=" * 72)
    print(f"LOTTO: {len(immagini)} immagini da allineare")
    print(f"riferimento: {info_rif}")
    print("=" * 72)

    # Il riferimento si apre UNA volta sola e si riusa per tutte le immagini:
    # se non e' mappabile in memoria, ricaricarlo a ogni giro costerebbe la sua
    # dimensione piena moltiplicata per il numero di immagini.
    arr_rif, _ = image_io.apri(riferimento)

    # Con il margine automatico le immagini si scrivono dopo, tutte insieme:
    # qui si stima soltanto, e si ricorda dove andra' scritta ciascuna.
    in_due_tempi = scrivi_immagini and margine == 'auto'
    margine_fisso = 0 if margine == 'auto' else int(margine or 0)
    da_scrivere = []
    risultati = []
    for i, path in enumerate(immagini):
        nome = os.path.basename(path)
        print(f"\n{'#' * 72}\n# [{i + 1}/{len(immagini)}] {nome}\n{'#' * 72}")

        def prog(testo, frazione, _i=i):
            if progress:
                progress(_i, len(immagini), testo, frazione)

        prog("avvio", 0.0)
        voce = {'file': os.path.abspath(path), 'nome': nome}
        try:
            if copie_tiff:
                image_io.copia_tiff_temporanea(path)
            primo, altri = fmt.piano(formati or ([estensione] if estensione else []),
                                     os.path.splitext(path)[1])
            out_path = _nome_output(path, cartella_output, suffisso, primo,
                                    momento=momento) if scrivi_immagini else None
            voce['_altri'] = altri if scrivi_immagini else []
            # Stessa data e ora dell'immagine prodotta: parametri, diagnostica e
            # immagine di una stessa corsa restano appaiati anche dopo averla
            # rifatta con altre impostazioni.
            base = f"{os.path.splitext(nome)[0]} ({momento.strftime('%Y-%m-%d %H-%M')})"
            # Gli strumenti a mano di QUESTA immagine. Senza, `a_mano` resta
            # vuoto e la chiamata e' quella di sempre.
            st = (strumenti or {}).get(path) or {}
            a_mano = {}
            punti = st.get('punti')
            if punti is not None and len(punti) >= 3:
                a_mano.update(punti=punti,
                              punti_come_posa=bool(st.get('punti_come_posa')),
                              punti_come_vincolo=bool(st.get('punti_come_vincolo')))
            if st.get('fascia_di_bordo') and st.get('contorno') is not None and mask is not None:
                a_mano.update(contorno_moving=st['contorno'],
                              fascia_di_bordo=st['fascia_di_bordo'])
            T, rep = align_paintings.align(
                riferimento, path, None if in_due_tempi else out_path,
                params_path=os.path.join(cartella_par, f"{base}.json"),
                factor=factor, nonrigid=nonrigid, tile_size=tile_size,
                min_conf=min_conf, refine_block=refine_block,
                refine_step=max(8, refine_block // 2),
                diagnostics_dir=os.path.join(cartella_diag, base),
                mask=mask, mask_info=mask_info, margine=margine_fisso, tre_scale=tre_scale,
                arr_fixed=arr_rif, progress=prog, verbose=verbose, **a_mano)
            if in_due_tempi:
                info_m = image_io.info(path)
                da_scrivere.append((i, T, path, out_path, os.path.join(cartella_par, f"{base}.json"),
                                    align_paintings.margine_necessario(
                                        T, (info_m.height, info_m.width), arr_rif.shape[:2])))

            prima = rep['verifica']['prima'].get('residuo_mediano_px')
            dopo = rep['verifica']['dopo'].get('residuo_mediano_px')
            esito, motivo = valuta_qualita(rep['verifica'], soglia_sospetto)
            voce.update({
                'esito': esito,
                'residuo_prima_px': prima,
                'residuo_dopo_px': dopo,
                'residuo_p95_px': rep['verifica']['dopo'].get('residuo_p95_px'),
                'residuo_ic90': rep['verifica']['dopo'].get('residuo_mediano_ic90'),
                'quota_sopra_1px': rep['verifica']['dopo'].get('quota_sopra_1px'),
                'quota_sopra_2px': rep['verifica']['dopo'].get('quota_sopra_2px'),
                'quota_sopra_5px': rep['verifica']['dopo'].get('quota_sopra_5px'),
                'quota_sopra_10px': rep['verifica']['dopo'].get('quota_sopra_10px'),
                'distorsione_px': rep['verifica']['dopo'].get('distorsione_px'),
                'campioni_affidabili': rep['verifica']['dopo'].get('n_affidabili'),
                'campioni_totali': rep['verifica']['dopo'].get('n_campioni'),
                'campo_nonrigido': rep['stima'].get('nonrigid', {}).get('applicato', False),
                'moving': os.path.abspath(path),
                'output': out_path,
                'parametri': os.path.join(cartella_par, f"{base}.json"),
                'diagnostica': os.path.join(cartella_diag, base),
                'zone_critiche': rep['verifica'].get('zone_critiche') or [],
                'livelli_grossolani': [
                    {k: l.get(k) for k in ('riduzione', 'applicato', 'ampiezza_max_px_full')}
                    for l in (rep['stima'].get('campo_grossolano_livelli') or [])],
            })
            if a_mano:
                voce.update({
                    'n_punti': 0 if a_mano.get('punti') is None else int(len(a_mano['punti'])),
                    'tre_punti': (rep.get('correzione_punti') or {}).get('tre'),
                    'punti_manuali': rep.get('punti_manuali'),
                    'correzione_punti': rep.get('correzione_punti'),
                    'fascia_di_bordo': rep.get('fascia_di_bordo'),
                })
            if verifica_fissa:
                # Dopo la verifica di sempre, che resta il numero principale:
                # questa dice cio' che la mediana nasconde. Se fallisce,
                # l'immagine resta allineata e verificata come prima.
                prog("verifica sulla griglia fissa", 0.99)
                try:
                    ver = vf.verifica_griglia_fissa(riferimento, path, T, mask, rep,
                                                    os.path.join(cartella_diag, base),
                                                    arr_fixed=arr_rif)
                    voce['verifica_fissa'] = ver
                    voce['n_oltre_10px'] = len(ver.get('oltre_10px') or [])
                    rep_path = os.path.join(cartella_par, f"{base}_report.json")
                    with open(rep_path, encoding='utf-8') as fh:
                        salvato = json.load(fh)
                    salvato['verifica_griglia_fissa'] = ver
                    with open(rep_path, 'w', encoding='utf-8') as fh:
                        json.dump(salvato, fh, indent=2, ensure_ascii=False)
                except Exception as exc:
                    print(f"  [attenzione] verifica a griglia fissa non riuscita: {exc}")
            if motivo:
                voce['motivo'] = motivo
                print(f"\n  [DA CONTROLLARE] {nome}: {motivo}")
        except Exception as exc:
            voce.update({'esito': 'errore', 'motivo': str(exc)})
            print(f"\n[ERRORE] {nome}: {exc}")
            if verbose:
                traceback.print_exc()
        risultati.append(voce)
        if not in_due_tempi:
            # Con il margine automatico l'immagine si rilegge nel secondo tempo.
            T = rep = None
            image_io.rilascia_copia(path)
        prog("completata", 1.0)

    # Secondo tempo: il margine e' il massimo fra le immagini, uguale per tutte.
    m = margine_fisso
    if in_due_tempi and da_scrivere:
        m = max(d[5] for d in da_scrivere)
        H, W = arr_rif.shape[:2]
        print(f"\nmargine comune del lotto: {m} px per lato")
        for i, T, path, out_path, par, _ in da_scrivere:
            def prog2(testo, frazione, _i=i):
                if progress:
                    progress(_i, len(immagini), testo, frazione)
            try:
                arr_m, _ = image_io.apri(path)
                info_m = image_io.info(path)
                T.meta['margine_uscita_px'] = int(m)
                T.save(par)
                align_paintings.scrivi_allineata(T, arr_m, out_path, H, W, info_m.samples,
                                                 info_m.dtype, tile_size=tile_size,
                                                 progress=prog2, margine=m)
                del arr_m
            except Exception as exc:
                risultati[i].update({'esito': 'errore', 'motivo': f"scrittura: {exc}"})
                print(f"\n[ERRORE] {os.path.basename(path)}: {exc}")
            arr_m = None
            image_io.rilascia_copia(path)
    # La copia del riferimento con lo stesso margine non si scrive piu' (per ora:
    # la questione e' rimasta aperta). Non aggiungeva informazione - e' il
    # riferimento con una cornice nera - e costava 1.8 GB e 40 s su un visibile
    # da 593 Mpx. Il margine resta registrato nei parametri e nel riepilogo:
    # il pixel (x, y) del riferimento sta in (x+m, y+m) delle immagini allineate.
    # Chi la vuole la ottiene con align_paintings.scrivi_con_margine.
    riferimento_margine = None

    # Gli altri formati chiesti, ricavati dal file appena scritto.
    for v in risultati:
        altri = v.pop('_altri', [])
        if altri and v.get('esito') != 'errore' and v.get('output') and os.path.exists(v['output']):
            try:
                v['output_altri'] = fmt.scrivi_altri(v['output'], altri, verbose=True)
            except Exception as exc:
                v['motivo_formati'] = str(exc)
                print(f"\n[ERRORE] {v['nome']}: altri formati non scritti: {exc}")

    for v in risultati:
        v['margine_px'] = int(m)

    riepilogo = {
        'riferimento': os.path.abspath(riferimento),
        'margine_px': int(m),
        'riferimento_con_margine': riferimento_margine,
        'cartella_output': os.path.abspath(cartella_output),
        'durata_min': (time.time() - t_start) / 60.0,
        'impostazioni': {'factor': factor, 'nonrigid': nonrigid,
                         'maschera': (mask_info or True) if mask is not None else False,
                         'min_conf': min_conf,
                         'refine_block': refine_block},
        'immagini': risultati,
    }
    with open(os.path.join(cartella_output, 'riepilogo.json'), 'w', encoding='utf-8') as fh:
        json.dump(riepilogo, fh, indent=2, ensure_ascii=False)

    arr_rif = None
    for path in list(immagini) + [riferimento]:
        image_io.rilascia_copia(path)

    stampa_riepilogo(riepilogo)
    return riepilogo


def stampa_riepilogo(riepilogo):
    """Tabella finale: una riga per immagine, con il residuo prima e dopo."""
    print("\n" + "=" * 72)
    print("RIEPILOGO")
    print("=" * 72)
    print(f"{'immagine':<30}{'prima':>8}{'dopo':>8}{'IC 90%':>15}{'>2px':>7}{'>5px':>7}"
          f"{'finestre':>10}  esito")
    print("-" * 72)
    for v in riepilogo['immagini']:
        nome = v['nome'][:33]
        if v['esito'] == 'errore':
            print(f"{nome:<30}{'-':>8}{'-':>8}{'-':>15}{'-':>7}{'-':>7}{'-':>10}  "
                  f"ERRORE: {v.get('motivo', '')[:50]}")
            continue
        prima = f"{v['residuo_prima_px']:.1f}" if v.get('residuo_prima_px') is not None else '-'
        dopo = f"{v['residuo_dopo_px']:.2f}" if v.get('residuo_dopo_px') is not None else '-'
        ic = v.get('residuo_ic90')
        ic_s = f"[{ic[0]:.2f}-{ic[1]:.2f}]" if ic else '-'
        q2 = f"{100 * v['quota_sopra_2px']:.0f}%" if v.get('quota_sopra_2px') is not None else '-'
        q5 = f"{100 * v['quota_sopra_5px']:.0f}%" if v.get('quota_sopra_5px') is not None else '-'
        fin = (f"{v['campioni_affidabili']}/{v['campioni_totali']}"
               if v.get('campioni_totali') else '-')
        marchio = 'ok' if v['esito'] == 'ok' else f"SOSPETTO ({v.get('motivo', '')})"
        print(f"{nome:<30}{prima:>8}{dopo:>8}{ic_s:>15}{q2:>7}{q5:>7}{fin:>10}  {marchio}")
    print("-" * 72)
    n_ok = sum(1 for v in riepilogo['immagini'] if v['esito'] == 'ok')
    n_sosp = sum(1 for v in riepilogo['immagini'] if v['esito'] == 'sospetto')
    n_err = sum(1 for v in riepilogo['immagini'] if v['esito'] == 'errore')
    print(f"{n_ok} allineate, {n_sosp} da controllare, {n_err} fallite "
          f"in {riepilogo['durata_min']:.1f} minuti")
    print("Residui in pixel a piena risoluzione, su 64 finestre da 512 px in posizioni"
          " indipendenti dalla stima.")
    print("IC 90% = intervallo di confidenza della mediana; >2px e >5px = quota di finestre"
          " oltre quella soglia;")
    print("finestre = quante danno una misura su quante tentate (la mediana vale per quelle).")


def main():
    p = argparse.ArgumentParser(
        description="Allinea piu' immagini sulla stessa immagine di riferimento.")
    p.add_argument('--fixed', required=True, help="immagine di riferimento (es. il visibile)")
    p.add_argument('--immagini', required=True, nargs='+',
                   help="una o piu' immagini da allineare")
    p.add_argument('--out-dir', required=True, help="cartella dove salvare tutto")
    p.add_argument('--factor', type=int, default=2,
                   help="riduzione per la stima (2 = default, sub-pixel; 4 = piu' veloce)")
    p.add_argument('--no-nonrigid', action='store_true', help="solo affine")
    p.add_argument('--refine-block', type=int, default=128)
    p.add_argument('--min-conf', type=float, default=0.10)
    p.add_argument('--tile', type=int, default=2048)
    p.add_argument('--suffisso', default='_allineata')
    p.add_argument('--formati', nargs='+', default=None, metavar='FORMATO',
                   help="uno o piu' fra TIFF, PNG, JPEG; senza, solo il formato "
                        "dell'originale")
    p.add_argument('--soglia', type=float, default=3.0,
                   help="residuo oltre il quale segnalare il risultato come sospetto")
    p.add_argument('--solo-stima', action='store_true',
                   help="esegue stima e verifica senza scrivere le immagini")
    p.add_argument('--margine', default='0',
                   help="pixel di foglio in piu' per lato, uguale per tutto il lotto, "
                        "o 'auto' per calcolarlo (default 0: nessun margine)")
    p.add_argument('--due-passate', action='store_true',
                   help="senza il campo grossolano delle tre scale (com'era prima)")
    args = p.parse_args()

    riepilogo = allinea_lotto(
        args.fixed, args.immagini, args.out_dir,
        factor=args.factor, nonrigid=not args.no_nonrigid, tile_size=args.tile,
        min_conf=args.min_conf, refine_block=args.refine_block,
        suffisso=args.suffisso, formati=args.formati,
        soglia_sospetto=args.soglia, scrivi_immagini=not args.solo_stima,
        margine=args.margine if args.margine == 'auto' else int(args.margine),
        tre_scale=False if args.due_passate else 'auto')
    return 0 if all(v['esito'] != 'errore' for v in riepilogo['immagini']) else 1


if __name__ == '__main__':
    sys.exit(main())
