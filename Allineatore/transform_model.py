"""
Modello della trasformazione fixed -> moving, con le conversioni di scala.

Convenzione unica in tutto il progetto (evita la maggior parte degli errori
geometrici): la trasformazione mappa le coordinate dell'immagine FIXED (VIS)
nelle coordinate dell'immagine MOVING (UV):

    s = M @ [x, y, 1] + F(x, y)

dove M e' una 2x3 affine e F un campo di spostamento residuo (la parte non
rigida), memorizzato su una griglia rada perche' e' liscio per costruzione.
Il pixel di output in posizione (x, y) si ottiene campionando la moving in s:
e' esattamente cio' che serve a cv2.remap, che lavora "all'indietro".

Tutte le quantita' sono in PIXEL A PIENA RISOLUZIONE. Quando si lavora su
un'anteprima ridotta di fattore intero f, si usa `at_factor(f)`, che applica
la relazione esatta fra le due griglie:

    coord_full = f * coord_preview + (f - 1) / 2

Il termine (f-1)/2 e' il centro del blocco di pixel mediati da INTER_AREA.
Trascurarlo introduce uno spostamento sistematico di f/2 pixel (con f=16 sono
8 px), cioe' dello stesso ordine dell'errore che stiamo correggendo.
"""

import json
import numpy as np
import cv2


class Transform:
    """Affine + campo di spostamento residuo opzionale."""

    def __init__(self, M, field=None, grid_step=None, ref_shape=None, meta=None):
        """
        Args:
            M: 2x3 affine, coordinate fixed -> coordinate moving
            field: (Gh, Gw, 2) float32, spostamento aggiuntivo in px (dx, dy),
                campionato su una griglia regolare di passo `grid_step` nelle
                coordinate dell'immagine FIXED. La cella (i, j) corrisponde
                esattamente alla coordinata fixed (j*grid_step, i*grid_step).
            grid_step: passo della griglia del campo, in px dell'immagine fixed
            ref_shape: (H, W) dell'immagine fixed per cui vale la trasformazione
            meta: dizionario libero di diagnostica (residui, n. blocchi, ecc.)
        """
        self.M = np.asarray(M, dtype=np.float64).reshape(2, 3)
        self.field = None if field is None else np.asarray(field, dtype=np.float32)
        self.grid_step = None if field is None else float(grid_step)
        self.ref_shape = None if ref_shape is None else (int(ref_shape[0]), int(ref_shape[1]))
        self.meta = meta or {}

    # ------------------------------------------------------------------
    # Scala
    # ------------------------------------------------------------------
    def at_factor(self, f, f_moving=None):
        """
        Restituisce la stessa trasformazione espressa nelle coordinate delle
        anteprime: quella FIXED ridotta di fattore `f`, quella MOVING ridotta di
        fattore `f_moving` (se None, si assume lo stesso fattore).

        I due fattori possono essere diversi perche' le due immagini possono
        avere dimensioni diverse (sensori o ritagli diversi): ciascuna viene
        ridotta al fattore intero che la porta vicino alla dimensione di lavoro
        voluta, e quel fattore non e' detto coincida.
        """
        f_f = float(f)
        f_m = float(f if f_moving is None else f_moving)
        c_f = np.array([(f_f - 1.0) / 2.0] * 2)
        c_m = np.array([(f_m - 1.0) / 2.0] * 2)

        A = self.M[:, :2]
        t = self.M[:, 2]
        A_new = (f_f / f_m) * A
        t_new = (t - c_m + A @ c_f) / f_m
        M_new = np.hstack([A_new, t_new.reshape(2, 1)])

        field_new = None
        grid_new = None
        if self.field is not None:
            # I valori del campo sono spostamenti in pixel della MOVING; la
            # griglia su cui e' campionato vive in coordinate della FIXED.
            field_new = self.field / f_m
            grid_new = self.grid_step / f_f
        ref_new = None if self.ref_shape is None else (int(self.ref_shape[0] // f_f),
                                                       int(self.ref_shape[1] // f_f))
        return Transform(M_new, field_new, grid_new, ref_new, dict(self.meta))

    @staticmethod
    def from_factor(M_prev, f, field=None, grid_step=None, ref_shape=None, meta=None,
                    f_moving=None):
        """
        Inverso di at_factor: da coordinate anteprima a piena risoluzione.

        `f` e' il fattore di riduzione dell'immagine FIXED, `f_moving` quello
        della MOVING (se None, lo stesso).
        """
        f_f = float(f)
        f_m = float(f if f_moving is None else f_moving)
        c_f = np.array([(f_f - 1.0) / 2.0] * 2)
        c_m = np.array([(f_m - 1.0) / 2.0] * 2)

        Mp = np.asarray(M_prev, dtype=np.float64).reshape(2, 3)
        A_prev, t_prev = Mp[:, :2], Mp[:, 2]
        A = (f_m / f_f) * A_prev
        t = f_m * t_prev + c_m - A @ c_f
        M = np.hstack([A, t.reshape(2, 1)])

        field_full = None if field is None else np.asarray(field, dtype=np.float32) * f_m
        grid_full = None if field is None else float(grid_step) * f_f
        return Transform(M, field_full, grid_full, ref_shape, meta)

    # ------------------------------------------------------------------
    # Applicazione
    # ------------------------------------------------------------------
    def maps_for_tile(self, y0, x0, h, w):
        """
        Mappe (map_x, map_y) float32 di shape (h, w): per ogni pixel del tile di
        output [y0:y0+h, x0:x0+w] (coordinate fixed) danno la coordinata
        corrispondente nell'immagine moving. Si passano direttamente a cv2.remap.
        """
        xs = (np.arange(x0, x0 + w, dtype=np.float32))
        ys = (np.arange(y0, y0 + h, dtype=np.float32))
        X, Y = np.meshgrid(xs, ys)

        a, b, tx = self.M[0]
        c, d, ty = self.M[1]
        map_x = (a * X + b * Y + tx).astype(np.float32)
        map_y = (c * X + d * Y + ty).astype(np.float32)

        if self.field is not None:
            gx = (X / self.grid_step).astype(np.float32)
            gy = (Y / self.grid_step).astype(np.float32)
            fld = cv2.remap(self.field, gx, gy, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            map_x += fld[:, :, 0]
            map_y += fld[:, :, 1]

        return map_x, map_y

    def apply_to_image(self, img_moving, output_shape, band=1024):
        """
        Applica la trasformazione a un'immagine moving GIA' IN RAM (anteprime,
        crop, test). Per i file a piena risoluzione si usa il warp a tile in
        align_paintings.py, mai questa funzione.

        Procede a bande di righe: le mappe di cv2.remap sono due array float32
        grandi quanto l'output, quindi su un'anteprima da 79 Mpx costerebbero
        630 MB solo loro. A bande restano poche decine di MB.
        """
        h, w = output_shape[:2]
        n_ch = img_moving.shape[2] if img_moving.ndim == 3 else 1
        shape = (h, w, n_ch) if img_moving.ndim == 3 else (h, w)
        out = np.empty(shape, dtype=img_moving.dtype)
        for y0 in range(0, h, band):
            bh = min(band, h - y0)
            out[y0:y0 + bh] = self.warp_tile_from(img_moving, y0, 0, bh, w)
        return out

    def warp_tile_from(self, moving_arr, y0, x0, h, w, margin=8, fill=0):
        """
        Produce il tile di output [y0:y0+h, x0:x0+w] (coordinate fixed) leggendo
        dall'immagine moving SOLO la regione sorgente che serve.

        `moving_arr` e' tipicamente un memmap (tiff_memmap.open_memmap): la regione
        viene letta dal disco al momento, quindi la RAM occupata e' quella di un
        tile, non dell'immagine intera. Affine e campo non rigido vengono
        applicati in UN SOLO cv2.remap: l'immagine viene quindi ricampionata una
        volta sola, senza la doppia interpolazione (e la conseguente perdita di
        nitidezza) di un warp affine seguito da un remap.
        """
        map_x, map_y = self.maps_for_tile(y0, x0, h, w)

        x_min = int(np.floor(map_x.min())) - margin
        x_max = int(np.ceil(map_x.max())) + margin
        y_min = int(np.floor(map_y.min())) - margin
        y_max = int(np.ceil(map_y.max())) + margin

        H, W = moving_arr.shape[:2]
        xs, xe = max(0, x_min), min(W, x_max)
        ys, ye = max(0, y_min), min(H, y_max)

        n_ch = moving_arr.shape[2] if moving_arr.ndim == 3 else 1
        out_shape = (h, w, n_ch) if moving_arr.ndim == 3 else (h, w)
        if xe <= xs or ye <= ys:
            return np.full(out_shape, fill, dtype=moving_arr.dtype)

        src = np.asarray(moving_arr[ys:ye, xs:xe])
        map_x -= xs
        map_y -= ys
        return cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    def compose_residual(self, delta_M, delta_field=None, delta_grid_step=None):
        """
        Aggiorna la trasformazione con un residuo misurato DOPO averla applicata.

        Il residuo si misura confrontando la fixed con la moving gia'
        pre-allineata con questa trasformazione, quindi e' una mappa
        D: coordinate fixed -> coordinate fixed (p -> p + spostamento residuo).
        La trasformazione aggiornata e' la COMPOSIZIONE, non la somma:

            s_new(p) = s_old(D(p))

        Per la parte affine la composizione e' esatta (prodotto delle matrici in
        coordinate omogenee). Per il campo vale

            F_new(p) = F_old(p) + A_old @ F_delta(p)

        dove `delta_field` e' misurato nel sistema della FIXED (e' un residuo
        misurato confrontando la fixed con la moving gia' pre-allineata) e
        A_old lo converte in pixel della MOVING. Quando le due immagini hanno la
        stessa dimensione A_old e' quasi l'identita' e la conversione cambia poco,
        ma se hanno dimensioni diverse A_old contiene il rapporto di scala e
        ignorarlo sarebbe un errore grossolano.

        Resta trascurato solo il termine F_old(p + delta) - F_old(p), di
        second'ordine: F_old e' liscio su centinaia di pixel e delta vale pochi
        pixel, quindi l'errore e' ben sotto il pixel, e in cambio si evita di
        reinterpolare il campo vecchio su una griglia deformata ad ogni passata.
        """
        dM = np.asarray(delta_M, dtype=np.float64).reshape(2, 3)
        old_h = np.vstack([self.M, [0.0, 0.0, 1.0]])
        delta_h = np.vstack([dM, [0.0, 0.0, 1.0]])
        M_new = (old_h @ delta_h)[:2, :]

        field_new, grid_new = self.field, self.grid_step
        if delta_field is not None:
            delta_field = np.asarray(delta_field, dtype=np.float32)
            # Da spostamento nel sistema della fixed a spostamento nel sistema
            # della moving (vedi docstring): F_delta_moving = A_old @ F_delta.
            A_old = self.M[:, :2]
            delta_field = np.einsum('ij,hwj->hwi', A_old, delta_field).astype(np.float32)
            if field_new is None:
                field_new, grid_new = delta_field, float(delta_grid_step)
            else:
                if abs(float(delta_grid_step) - grid_new) > 1e-6 or \
                        delta_field.shape != field_new.shape:
                    delta_res = cv2.resize(delta_field, (field_new.shape[1], field_new.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                    # riscalatura dei vettori non necessaria: sono gia' in px pieni
                    field_new = field_new + delta_res
                else:
                    field_new = field_new + delta_field
        return Transform(M_new, field_new, grid_new, self.ref_shape, dict(self.meta))

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------
    def save(self, json_path):
        """
        Salva la trasformazione: JSON leggibile + .npy per il campo non rigido.

        I parametri valgono per LA COPPIA DI ESPOSIZIONI su cui sono stati
        stimati: servono a rigenerare il risultato senza ristimarlo e a warpare
        i raster derivati dalla stessa ripresa (maschere, ROI, ri-esportazioni).
        Non sono trasferibili a un altro scatto — nemmeno della stessa scena —
        perche' posa relativa, flessione della tela e magnificazione dipendente
        dalla lunghezza d'onda cambiano da un'esposizione all'altra: vedi la
        spiegazione in testa a apply_transform.py.
        """
        data = {
            'convention': 'coord_moving = M @ [x_fixed, y_fixed, 1] + field(x_fixed, y_fixed)',
            'units': 'pixel a piena risoluzione',
            'M': self.M.tolist(),
            'has_field': self.field is not None,
            'grid_step': self.grid_step,
            'field_file': None,
            'ref_shape': list(self.ref_shape) if self.ref_shape else None,
            'meta': self.meta,
        }
        if self.field is not None:
            field_path = json_path.rsplit('.', 1)[0] + '_field.npy'
            np.save(field_path, self.field)
            data['field_file'] = field_path.replace('\\', '/').split('/')[-1]
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)

    @staticmethod
    def load(json_path):
        import os
        with open(json_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        field = None
        if data.get('has_field') and data.get('field_file'):
            field = np.load(os.path.join(os.path.dirname(os.path.abspath(json_path)),
                                         data['field_file']))
        return Transform(np.array(data['M']), field, data.get('grid_step'),
                         data.get('ref_shape'), data.get('meta', {}))

    # ------------------------------------------------------------------
    def describe(self):
        """Descrizione leggibile (scala, rotazione, shear, traslazione)."""
        A = self.M[:, :2]
        t = self.M[:, 2]
        sx = float(np.hypot(A[0, 0], A[1, 0]))
        sy = float(np.hypot(A[0, 1], A[1, 1]))
        rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        shear = float(np.degrees(np.arctan2(A[0, 1], A[1, 1])) + rot)
        lines = [
            f"  scala:       x {sx:.5f}   y {sy:.5f}  ({(sx - 1) * 100:+.3f}% / {(sy - 1) * 100:+.3f}%)",
            f"  rotazione:   {rot:+.4f} gradi",
            f"  shear:       {shear:+.4f} gradi",
            f"  traslazione: ({t[0]:+.1f}, {t[1]:+.1f}) px",
        ]
        if self.field is not None:
            mag = np.hypot(self.field[:, :, 0], self.field[:, :, 1])
            lines.append(f"  campo non rigido: rms {float(np.sqrt((mag ** 2).mean())):.2f} px, "
                         f"max {float(mag.max()):.2f} px (griglia {self.field.shape[1]}x{self.field.shape[0]}, "
                         f"passo {self.grid_step:.0f} px)")
        else:
            lines.append("  campo non rigido: assente (solo affine)")
        return "\n".join(lines)
