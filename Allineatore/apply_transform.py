"""
Riapplica una trasformazione gia' stimata e VERIFICATA, senza rifare la stima.

QUANDO SERVE
Una trasformazione salvata da align_paintings.py descrive **una specifica coppia
di esposizioni**, non "l'UV" in generale. Questo script serve quindi a portare in
coordinate VIS un raster che vive nel sistema di coordinate di **quella stessa
ripresa**:

- rigenerare l'immagine allineata partendo dai soli parametri (riproducibilita':
  chi legge la tesi deve poter ricostruire il risultato dal JSON, e tu puoi
  rigenerare l'output se lo cancelli o se cambi --tile);
- warpare qualunque cosa sia derivata da quello scatto: maschere, ROI tracciate
  sull'UV, mappe di segmentazione, falsi colori, una ri-esportazione a 16 bit o
  con un'altra curva. Tutte hanno la geometria identica per costruzione, quindi
  il warp e' esattamente lo stesso.

QUANDO *NON* SERVE - e questo e' il punto importante
Per una NUOVA esposizione (un canale IR, un secondo UV, un rifacimento) la
trasformazione va RI-STIMATA:

    python align_paintings.py --fixed VIS.tif --moving IR.tif --out IR_allineata.tif

Tre motivi indipendenti, tutti presenti nei dati di questo progetto:

1. Posa relativa. Fra VIS e UV sono stati misurati +0.129 gradi di rotazione e
   (-32.7, -14.9) px di traslazione: possono venire solo dal fatto che camera e
   dipinto si sono mossi l'uno rispetto all'altro FRA le due esposizioni. Una
   terza esposizione si e' mossa di nuovo, di una quantita' diversa.
2. Tempo. Il campo non rigido misurato (rms 6.8 px, max 15 px) e' la flessione
   della tela fra due ISTANTI. Un altro scatto, un altro istante, un'altra
   flessione.
3. Lunghezza d'onda. Aberrazione cromatica laterale e spessore/indice del filtro
   cambiano la magnificazione in funzione della banda. La correzione giusta per
   l'UV e' sbagliata per l'IR, e con segno opposto rispetto al visibile.

Per questo lo script, prima di scrivere alcunche', MISURA se la trasformazione si
applica davvero all'immagine che gli hai dato, e si ferma se non e' cosi'. Il
controllo costa un paio di secondi contro i due minuti e i ~900 MB di una
scrittura inutile.

Uso:
    python apply_transform.py --params risultati\\transform_params.json \\
                              --image immagini\\UVF_FraGalgario_TOT.tif \\
                              --out risultati\\UVF_allineata.tif \\
                              --reference immagini\\VIS_FraGalgario_TOT.tif
"""

import argparse
import os
import sys


import image_io
import validation as val
from transform_model import Transform
from align_paintings import peak_memory_gb, scrivi_allineata


class TrasformazioneNonApplicabile(RuntimeError):
    """La trasformazione salvata non allinea l'immagine fornita."""


def preflight_check(reference_path, image_path, transform, max_residuo=3.0, verbose=True):
    """
    Misura, PRIMA di scrivere, se la trasformazione allinea davvero questa
    immagine al riferimento.

    Usa validation.residual_check, lo stesso controllo della pipeline di stima:
    campiona alcune finestre a piena risoluzione, applica la trasformazione al
    volo e misura di quanti pixel le due immagini restano disallineate. Lavora
    sul modulo del gradiente, quindi funziona anche fra bande diverse.

    Costo: una ventina di finestre da 512 px, cioe' pochi MB letti da disco.

    Raises:
        TrasformazioneNonApplicabile: se il residuo mediano supera max_residuo.
    """
    if os.path.abspath(reference_path) == os.path.abspath(image_path):
        raise TrasformazioneNonApplicabile(
            f"--image e --reference sono lo stesso file ({os.path.basename(image_path)}).\n"
            "L'immagine da trasformare e' quella che sta nelle coordinate della 'moving' "
            "(l'UV), il riferimento e' la 'fixed' (il VIS): applicare a se stessa una "
            "trasformazione fra due immagini diverse non ha significato.")

    arr_ref, _ = image_io.apri(reference_path)
    arr_img, _ = image_io.apri(image_path)

    if verbose:
        print("\nControllo preliminare: la trasformazione si applica a questa immagine?")
    res = val.residual_check(arr_ref, arr_img, transform, verbose=verbose)

    if not res.get('n_affidabili'):
        raise TrasformazioneNonApplicabile(
            "Nessun campione affidabile: le due immagini non sembrano avere struttura "
            "in comune. Controlla di aver passato il riferimento giusto in --reference.")

    mediano = res['residuo_mediano_px']
    if mediano > max_residuo:
        raise TrasformazioneNonApplicabile(
            f"Residuo mediano {mediano:.2f} px, oltre la soglia di {max_residuo:.2f} px: "
            f"questa trasformazione NON allinea l'immagine indicata.\n\n"
            f"Causa quasi certa: e' un'esposizione diversa da quella su cui la "
            f"trasformazione e' stata stimata. Posa della camera, flessione della tela e "
            f"magnificazione dipendente dalla banda cambiano da uno scatto all'altro, "
            f"quindi i parametri non sono trasferibili.\n\n"
            f"Soluzione: stima la trasformazione per questa coppia (bastano ~2 minuti):\n"
            f"  python align_paintings.py --fixed {reference_path} "
            f"--moving {image_path} --out <output.tif>\n\n"
            f"Se sai quello che stai facendo, --force scrive comunque l'output.")

    if verbose:
        print(f"  -> ok: residuo {mediano:.2f} px, sotto la soglia di {max_residuo:.2f} px")
    return res


def apply_saved(params_path, image_path, out_path, reference_path=None,
                tile_size=2048, max_residuo=3.0, check=True, force=False,
                verbose=True):
    """
    Applica una trasformazione salvata a un'immagine, scrivendo il risultato a
    tile (memoria costante).

    Il controllo preliminare e' attivo per default: senza `reference_path` non e'
    possibile farlo, quindi in quel caso serve check=False insieme a force=True,
    per rendere esplicita la scelta di procedere senza verifica.

    Returns:
        dict con il residuo misurato prima e dopo la scrittura.
    """
    T = Transform.load(params_path)
    info = image_io.info(image_path)
    H, W = (T.ref_shape if T.ref_shape else (info.height, info.width))
    # Il margine con cui e' stata scritta l'immagine allineata originale: senza,
    # la nuova uscita non si sovrapporrebbe piu' a quella ne' alla copia del
    # riferimento con il margine.
    m = int((T.meta or {}).get('margine_uscita_px') or 0)

    print(f"Trasformazione da: {params_path}")
    print(T.describe())
    print(f"\nImmagine da trasformare: {os.path.basename(image_path)} "
          f"{info.width}x{info.height}")
    print(f"Output: {W + 2 * m}x{H + 2 * m} (dimensioni dell'immagine di riferimento"
          + (f", piu' {m} px di margine per lato)" if m else ")"))

    if (info.height, info.width) != (H, W):
        print("  [attenzione] l'immagine ha dimensioni diverse da quelle su cui e' stata\n"
              "  stimata la trasformazione: il risultato ha senso solo se l'inquadratura\n"
              "  e' comunque la stessa.")

    report = {}
    if check:
        if not reference_path:
            raise ValueError(
                "Serve --reference per verificare che la trasformazione sia applicabile. "
                "Per saltare la verifica (sconsigliato): --no-check --force.")
        try:
            report['prima'] = preflight_check(reference_path, image_path, T,
                                              max_residuo=max_residuo, verbose=verbose)
        except TrasformazioneNonApplicabile as err:
            if not force:
                raise
            print(f"\n[--force] verifica fallita, procedo lo stesso:\n{err}\n")
    else:
        print("\n[--no-check] verifica preliminare saltata.")

    # Il file di output viene creato solo DOPO che la verifica e' passata: cosi'
    # un tentativo rifiutato non lascia in giro un TIFF da 900 MB incompleto.
    # Lo stesso scrittore di align: stessa scrittura a tile e stesso margine.
    arr, _ = image_io.apri(image_path)
    scrivi_allineata(T, arr, out_path, H, W, info.samples, info.dtype,
                     tile_size=tile_size, margine=m)

    if reference_path:
        # Conferma sul file effettivamente scritto (non sulla trasformazione):
        # verifica anche che la scrittura a tile sia andata a buon fine.
        print("\nVerifica sul file scritto:")
        arr_ref, _ = image_io.apri(reference_path)
        arr_out, _ = image_io.apri(out_path)
        report['dopo'] = val.residual_check(arr_ref, arr_out[m:m + H, m:m + W], None,
                                            verbose=verbose)

    mem = peak_memory_gb()
    if mem is not None:
        print(f"Picco di memoria privata: {mem[0]:.2f} GB")
    return report


def main():
    p = argparse.ArgumentParser(
        description="Riapplica una trasformazione salvata a un raster della STESSA ripresa. "
                    "Per un'altra esposizione (es. un canale IR) usa align_paintings.py.")
    p.add_argument('--params', required=True, help="transform_params.json prodotto da align_paintings.py")
    p.add_argument('--image', required=True, help="immagine da trasformare")
    p.add_argument('--out', required=True, help="TIFF di output")
    p.add_argument('--reference', default=None,
                   help="immagine di riferimento (es. il VIS): obbligatoria salvo --no-check")
    p.add_argument('--max-residuo', type=float, default=3.0,
                   help="residuo mediano massimo accettato dal controllo preliminare, in px (default 3.0)")
    p.add_argument('--no-check', action='store_true',
                   help="salta il controllo preliminare (richiede --force)")
    p.add_argument('--force', action='store_true',
                   help="scrive l'output anche se il controllo preliminare fallisce")
    p.add_argument('--tile', type=int, default=2048)
    args = p.parse_args()

    if args.no_check and not args.force:
        p.error("--no-check richiede anche --force: applicare una trasformazione senza "
                "verificarla e' proprio l'errore che questo controllo esiste per evitare.")

    try:
        apply_saved(args.params, args.image, args.out, args.reference,
                    tile_size=args.tile, max_residuo=args.max_residuo,
                    check=not args.no_check, force=args.force)
    except (TrasformazioneNonApplicabile, ValueError) as err:
        print(f"\nOPERAZIONE ANNULLATA (nessun file scritto)\n\n{err}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

