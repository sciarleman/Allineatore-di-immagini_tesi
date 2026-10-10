"""
Script interattivo per calibrare la camera da immagini di scacchiera.

Workflow:
1. Acquisisci 15-20 foto della scacchiera da diverse angolazioni
   (assicurati che sia ben illuminata e tutta a fuoco)
2. Metti le foto in una cartella (es. calibration_images/vis_50mm/)
3. Esegui questo script
4. Salva i coefficienti (JSON) per uso futuro
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple


def create_checkerboard_template(rows: int = 9, cols: int = 6, 
                                 square_size_mm: int = 25,
                                 output_file: str = "checkerboard_template.png"):
    """
    Crea un'immagine con una scacchiera stampabile per la calibrazione.
    
    Args:
        rows, cols: numero di quadrati in verticale e orizzontale
        square_size_mm: dimensione del quadrato in mm
        output_file: dove salvare l'immagine
    
    Istruzioni di stampa:
    1. Stampa l'immagine su carta A4 a 100% scala (no scaling!)
    2. Incolla su cartone rigido
    3. Usala per acquisire foto da varie angolazioni
    """
    square_pixels = int((square_size_mm / 25.4) * 300)  # 300 DPI
    
    h = rows * square_pixels
    w = cols * square_pixels
    
    checkerboard = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y1, y2 = i * square_pixels, (i + 1) * square_pixels
                x1, x2 = j * square_pixels, (j + 1) * square_pixels
                checkerboard[y1:y2, x1:x2] = 255
    
    cv2.imwrite(output_file, checkerboard)
    print(f"✓ Scacchiera creata: {output_file}")
    print(f"  Dimensioni: {w}x{h} pixel ({w/square_pixels:.0f}x{h/square_pixels:.0f} quadrati)")
    print(f"  Stampa a 300 DPI per dimensione reale")
    return checkerboard


def collect_calibration_images_from_dir(image_dir: str, 
                                        board_size: Tuple[int, int] = (9, 6),
                                        verbose: bool = True) -> List[np.ndarray]:
    """
    Carica e verifica immagini di calibrazione da una cartella.
    
    Args:
        image_dir: cartella con immagini (.jpg, .png)
        board_size: (righe, colonne) della scacchiera
        verbose: stampa statistiche
    
    Returns:
        lista di immagini che contengono la scacchiera rilevabile
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Cartella non trovata: {image_dir}")
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    image_files = sorted(image_files)
    
    if not image_files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {image_dir}")
    
    print(f"\nCaricamento immagini da: {image_dir}")
    print(f"Trovate {len(image_files)} immagini")
    
    valid_images = []
    
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"  [{idx+1}] ✗ {img_path.name}: impossibile caricare")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Verifica che la scacchiera sia rilevabile
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            valid_images.append(img)
            print(f"  [{idx+1}] ✓ {img_path.name}: scacchiera rilevata")
        else:
            print(f"  [{idx+1}] ✗ {img_path.name}: scacchiera non rilevata")
    
    print(f"\nRisultato: {len(valid_images)}/{len(image_files)} immagini valide")
    
    if len(valid_images) < 3:
        raise ValueError(
            f"Necessarie almeno 3 immagini con scacchiera rilevabile. "
            f"Trovate {len(valid_images)}. "
            f"Verifica illuminazione, nitidezza e angolazione."
        )
    
    return valid_images


def calibrate_and_save(image_dir: str,
                       board_size: Tuple[int, int] = (9, 6),
                       square_size: float = 0.025,  # metri (25 mm)
                       name: str = "camera",
                       output_file: str = None):
    """
    Calibra la camera da una cartella di immagini e salva i coefficienti.
    
    Args:
        image_dir: cartella con immagini di scacchiera
        board_size: (righe, colonne) della scacchiera
        square_size: dimensione del quadrato in metri
        name: identificativo della calibrazione (es "vis_50mm", "uv_filter")
        output_file: percorso JSON di output (default: calibration_{name}.json)
    """
    
    # Carica immagini
    images = collect_calibration_images_from_dir(image_dir, board_size)
    
    # Calibra
    print(f"\n{'='*70}")
    print(f"CALIBRAZIONE: {name}")
    print(f"{'='*70}")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Coordinate 3D della scacchiera
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []
    imgpoints = []
    
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    # calibrateCamera
    gray_shape = (images[0].shape[1], images[0].shape[0])
    ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )
    
    # Errore di riproiezione
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distCoeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    
    # Salva
    calib_data = {
        'name': name,
        'K': K.tolist(),
        'distCoeffs': distCoeffs.flatten().tolist(),
        'image_size': list(gray_shape),
        'reprojection_error': float(mean_error),
        'num_images_used': len(objpoints),
        'board_size': list(board_size),
        'square_size_m': float(square_size)
    }
    
    if output_file is None:
        output_file = f"calibration_{name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    # Stampa risultati
    print(f"\n✓ Calibrazione completata")
    print(f"  Immagini utilizzate: {len(objpoints)}")
    print(f"  Errore riproiezione: {mean_error:.4f} pixel")
    print(f"\nMatrice di camera (K):")
    print(f"{K}")
    print(f"\nCoefficienti distorsione (Brown-Conrady):")
    print(f"  k1, k2, p1, p2, k3: {distCoeffs.flatten()}")
    print(f"\n✓ Coefficienti salvati in: {output_file}")
    
    return calib_data


def verify_calibration(calib_file: str, test_image_path: str):
    """
    Verifica la calibrazione applicando undistortion a un'immagine di test
    e visualizzando il risultato.
    """
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    K = np.array(calib['K'], dtype=np.float32)
    distCoeffs = np.array(calib['distCoeffs'], dtype=np.float32)
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Errore: impossibile caricare {test_image_path}")
        return
    
    undistorted = cv2.undistort(img, K, distCoeffs)
    
    # Side-by-side
    h, w = img.shape[:2]
    comparison = np.hstack([img, undistorted])
    
    output = "verification_undistort.png"
    cv2.imwrite(output, comparison)
    print(f"\n✓ Verifica completata: {output}")
    print(f"  Sinistra: immagine originale (distorta)")
    print(f"  Destra: immagine undistorted")
    print(f"  Osserva i bordi per vedere l'effetto della correzione")


# ============================================================================
# SCRIPT INTERATTIVO
# ============================================================================

if __name__ == "__main__":
    
    import sys
    
    print("="*70)
    print("CALIBRAZIONE CAMERA PER REGISTRAZIONE DIPINTI")
    print("="*70)
    
    # Step 1: Genera template (opzionale)
    generate_template = input(
        "\nVuoi generare un template di scacchiera stampabile? (s/n): "
    ).lower() == 's'
    
    if generate_template:
        print("\nGenerazione template scacchiera...")
        create_checkerboard_template(rows=9, cols=6, square_size_mm=25)
        print("Istruzioni:")
        print("  1. Stampa a 300 DPI a scala 100% (NO scaling)")
        print("  2. Incolla su cartone rigido")
        print("  3. Acquisisci 15-20 foto da varie angolazioni")
        print("  4. Metti le foto in una cartella (es. calib_vis/, calib_uv/)")
    
    # Step 2: Calibrazione
    print("\n" + "="*70)
    print("CALIBRAZIONE")
    print("="*70)
    
    calib_dir = input(
        "\nInserisci percorso della cartella con immagini di scacchiera: "
    ).strip()
    
    if not Path(calib_dir).exists():
        print(f"✗ Cartella non trovata: {calib_dir}")
        sys.exit(1)
    
    name = input("Nome della configurazione (es 'vis_50mm', 'uv_filter'): ").strip()
    
    if not name:
        name = "camera"
    
    square_size_mm = input("Dimensione quadrato in mm (default 25): ").strip()
    square_size_mm = int(square_size_mm) if square_size_mm else 25
    square_size_m = square_size_mm / 1000.0
    
    try:
        calib_data = calibrate_and_save(
            calib_dir,
            board_size=(9, 6),
            square_size=square_size_m,
            name=name
        )
    except Exception as e:
        print(f"✗ Errore durante calibrazione: {e}")
        sys.exit(1)
    
    # Step 3: Verifica (opzionale)
    print("\n" + "="*70)
    verify = input(
        "\nVuoi verificare la calibrazione su un'immagine di test? (s/n): "
    ).lower() == 's'
    
    if verify:
        test_img = input("Percorso dell'immagine di test: ").strip()
        calib_file = f"calibration_{name}.json"
        try:
            verify_calibration(calib_file, test_img)
        except Exception as e:
            print(f"✗ Errore: {e}")
    
    print("\n" + "="*70)
    print("COMPLETATO")
    print("="*70)
    print(f"\nCoefficienti salvati come: calibration_{name}.json")
    print("\nProssimo step: usa questi coefficienti nello script principale:")
    print(f"  reg.load_calibration('calibration_{name}.json')")
