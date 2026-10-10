"""
Finestra per mettere a mano i punti di corrispondenza.

Serve dove il matching automatico non parte: sulla radiografia sopravvive il
2-3% dei blocchi, perche' il pre-allineamento automatico sa fare solo
traslazioni mentre fra visibile e radiografia restano il 12% di scala e 0.3
gradi di rotazione, e i blocchi finiscono per cercare nel posto sbagliato.
Bastano 4-8 punti per dare la posa di partenza: da li' lavora l'automatico, che
e' piu' preciso di qualunque clic.

Sulla precisione. Un clic vale quanto il pixel che stai guardando: se l'immagine
e' rimpicciolita venti volte per stare nello schermo, un pixel di schermo sono
venti pixel veri, e nessuna cura successiva li recupera. Per questo la vista non
lavora su un'anteprima fissa: a ogni ingrandimento i pixel vengono letti dal
file al passo giusto, uno per ogni pixel di schermo. Lo zoom arriva al 400%, e
in basso sono sempre scritte le coordinate vere sotto il puntatore.
"""

import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import cv2

import image_io
import punti_manuali

# Quanti campioni per lato si leggono, rispetto a quelli che servono a schermo,
# prima di mediarli. A 1 si campiona e basta: velocissimo ma scalettato. Misurato
# confrontando la resa con quella esatta (media d'area su tutti i pixel), in
# livelli su 255: a zoom 0.25 lo scarto e' 4.7 con 1 campione, 1.5 con 2 e 0.0
# con 3; a zoom 0.03, 4.6 / 2.6 / 1.5. Tre campioni costano 9-32 ms e danno una
# resa indistinguibile da quella esatta, che ne costerebbe qualche migliaio.
SOVRACAMPIONE = 3


class Riquadro(tk.Frame):
    """Una delle due viste: immagine, zoom, spostamento, punti segnati."""

    def __init__(self, padre, path, titolo, al_clic, al_movimento=None,
                 T_vista=None, forma_vista=None, arr_confronto=None):
        """
        Args:
            T_vista: se data, l'immagine viene mostrata GIA' PORTATA nel sistema
                del riferimento. Da quel momento le coordinate di questa vista
                sono quelle del riferimento, non piu' quelle del file: cliccare
                significa indicare un posto nel dipinto, e la posizione
                corrispondente nel file originale si ricava con `alla_sorgente`.
                E' cio' che permette di lavorare a giri successivi, ognuno sul
                risultato del precedente, senza mai ricampionare l'immagine.
            forma_vista: (H, W) del riferimento, obbligatoria con T_vista.
            arr_confronto: la seconda immagine, per la vista a scacchiera. Ha
                senso solo su una vista gia' portata nel sistema del
                riferimento, perche' li' le due immagini hanno le stesse
                coordinate e i riquadri si possono alternare.
        """
        super().__init__(padre)
        self.path = path
        self.al_clic = al_clic
        self.al_movimento = al_movimento
        self.arr, self.info = image_io.apri(path, verbose=False)
        self.Hm, self.Wm = self.arr.shape[:2]      # il file, sempre
        self.T_vista = T_vista
        if T_vista is not None:
            self.H, self.W = forma_vista
            self.scala_vista = float(np.hypot(T_vista.M[0, 0], T_vista.M[1, 0]))
        else:
            self.H, self.W = self.Hm, self.Wm
            self.scala_vista = 1.0
        # Un'anteprima piccola serve solo a decidere una volta per tutte la
        # scala dei toni (i percentili): per quello la qualita' non conta, e si
        # legge un pixel ogni sedici. La vista invece non la usa mai: legge
        # sempre dal file al passo giusto per l'ingrandimento corrente.
        prev, _f = image_io.anteprima_veloce(path, 1200, in_grigio=False)
        self._rgb(prev, calcola_scala=True)
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.zoom = 1.0
        self.punti = []          # (x, y) in coordinate immagine
        self.provvisorio = None
        self.suggerito = None
        self.raggio_sugg = 0.0   # di quanto la previsione puo' sbagliare, in px
        self.zoom_min_clic = 0.5   # sotto questo zoom il clic non registra

        self.arr_confronto = arr_confronto
        self.scacchiera = False
        self.lato_scacco = 96    # in pixel di schermo
        self._vista = None       # ultima vista disegnata, prima della luminosita'
        self._lut = None         # None = nessuna correzione, si disegna com'e'

        self.titolo = tk.Label(self, text=titolo, font=("Segoe UI", 10, "bold"))
        self.titolo.pack(anchor="w")
        # width piccolo per lo stesso motivo dell'etichetta di stato: la
        # larghezza richiesta non deve decidere come si spartiscono lo spazio le
        # due viste, che devono restare mezzo e mezzo.
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0,
                                cursor="tcross", width=50, height=50)
        self.canvas.pack(fill="both", expand=True)

        piede = tk.Frame(self)
        piede.pack(fill="x")
        # width=1: l'etichetta chiede un carattere di larghezza e poi si allarga
        # con fill. Senza, la larghezza RICHIESTA dipende dal testo, e siccome
        # le due viste si dividono lo spazio in proporzione a quanto chiedono,
        # il riquadro cambiava larghezza a ogni movimento del mouse - il testo
        # sotto cambia mentre si muove il puntatore.
        self.stato = tk.Label(piede, text="", fg="gray", anchor="w", width=1)
        self.stato.pack(side="left", fill="x", expand=True)
        # Luminosita' e contrasto sono per SINGOLA vista, non per la finestra:
        # visibile e radiografia hanno istogrammi che non c'entrano niente l'uno
        # con l'altro, e per riconoscere lo stesso dettaglio nelle due serve
        # regolarle in modo indipendente.
        # Si passa da variabili con trace invece che dal parametro `command`:
        # quello scatta solo quando e' l'utente a trascinare, non quando il
        # valore viene impostato dal programma, e il pulsante di azzeramento qui
        # accanto non avrebbe avuto alcun effetto.
        self.luce = tk.IntVar(value=0)
        self.contrasto = tk.IntVar(value=0)
        for testo, var in (("lum.", self.luce), ("contr.", self.contrasto)):
            tk.Label(piede, text=testo, fg="gray").pack(side="left", padx=(6, 1))
            tk.Scale(piede, variable=var, from_=-100, to=100, orient="horizontal",
                     length=100, showvalue=False, sliderlength=14).pack(side="left")
            var.trace_add('write', lambda *a: self._ricalcola_lut())
        tk.Button(piede, text="0", font=("Segoe UI", 7), padx=3, pady=0,
                  command=self._azzera_toni).pack(side="left", padx=(3, 0))

        self.canvas.bind("<Configure>", lambda e: self.adatta()
                         if self.zoom is None else self.disegna())
        self.canvas.bind("<Button-1>", self._clic)
        self.canvas.bind("<Motion>", self._muovi)
        self.canvas.bind("<Button-3>", self._inizio_trascino)
        self.canvas.bind("<B3-Motion>", self._trascina)
        self.canvas.bind("<MouseWheel>", self._rotella)
        self.zoom = None          # alla prima Configure si adatta

    # ------------------------------------------------------------- immagine
    def _rgb(self, a, calcola_scala=False):
        """Porta a RGB 8 bit con una scala FISSA, decisa una volta sull'anteprima.

        Ricalcolare i percentili su ogni ritaglio farebbe cambiare luminosita'
        mentre ci si sposta, il che confonde e basta.
        """
        a = np.asarray(a)
        if calcola_scala:
            f = a.astype(np.float32)
            self.lo = float(np.percentile(f, 0.5))
            self.hi = float(np.percentile(f, 99.5))
        f = (a.astype(np.float32) - self.lo) / max(self.hi - self.lo, 1e-6)
        out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        return out[:, :, :3]

    def adatta(self):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.zoom = min(cw / self.W, ch / self.H)
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.disegna()

    def disegna(self):
        if self.zoom is None:
            return self.adatta()
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        larg, alt = cw / self.zoom, ch / self.zoom        # zona inquadrata, in px immagine
        x0, y0 = self.cx - larg / 2.0, self.cy - alt / 2.0

        # Si legge sempre dal file, a un passo scelto per l'ingrandimento
        # corrente: SOVRACAMPIONE pixel del file per ogni pixel di schermo, mai
        # di piu'. Cosi' l'immagine e' nitida a QUALUNQUE zoom e il conto non
        # cresce mai. Prima si leggeva a piena risoluzione sotto una certa area
        # e si ripiegava su un'anteprima fissa da 1600 px sopra: a meta' strada
        # quell'anteprima veniva ingrandita due o tre volte, ed e' il motivo per
        # cui l'immagine andava a fuoco solo a certi ingrandimenti.
        xa, ya = int(np.floor(max(0, x0))), int(np.floor(max(0, y0)))
        xb = int(np.ceil(min(self.W, x0 + larg)))
        yb = int(np.ceil(min(self.H, y0 + alt)))
        if xb <= xa or yb <= ya:
            return
        passo_schermo = max(1, int((xb - xa) / max(cw, 1)))
        passo = max(1, passo_schermo // SOVRACAMPIONE)
        if self.T_vista is None:
            sorgente = self._rgb(np.ascontiguousarray(self.arr[ya:yb:passo, xa:xb:passo]))
        else:
            sorgente = self._ritaglio_portato(xa, ya, xb, yb, passo)
            if sorgente is None:
                return

        # cv2.warpAffine NON sa fare INTER_AREA: gliela si puo' passare, ma la
        # ignora e si comporta come INTER_LINEAR, che rimpicciolendo di tre
        # volte scaletta invece di mediare. (Valeva anche per il codice di
        # prima, che chiedeva INTER_AREA alla warp e non la otteneva.) Si
        # rimpicciolisce quindi prima con una resize vera, dove INTER_AREA
        # funziona davvero, e alla warp resta solo lo spostamento sub-pixel.
        k = max(1, int(round(passo_schermo / passo)))
        if k > 1 and min(sorgente.shape[:2]) >= 2 * k:
            sorgente = cv2.resize(sorgente, (max(1, sorgente.shape[1] // k),
                                             max(1, sorgente.shape[0] // k)),
                                  interpolation=cv2.INTER_AREA)
            passo *= k

        # da coordinate immagine a pixel della sorgente
        fx0, fy0 = (x0 - xa) / passo, (y0 - ya) / passo
        fw, fh = larg / passo, alt / passo
        M = np.float32([[cw / fw, 0, -fx0 * cw / fw], [0, ch / fh, -fy0 * ch / fh]])
        # Sopra il 100% si stanno ingrandendo pixel veri e conviene mostrarli
        # come sono; sotto, la sorgente e' gia' alla scala giusta e serve solo
        # l'interpolazione per il mezzo pixel di sfasamento.
        interp = cv2.INTER_NEAREST if self.zoom >= 1 else cv2.INTER_LINEAR
        self._vista = cv2.warpAffine(sorgente, M, (cw, ch), flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(32, 32, 32))
        if self.scacchiera and self.arr_confronto is not None:
            self._vista = self._mescola(self._vista, xa, ya, xb, yb, passo, cw, ch, M, interp)
        self._mostra_vista()
        self.stato.config(text=f"zoom {self.zoom * 100:.0f}%   "
                               f"{'pixel veri' if passo == 1 else f'1 px ogni {passo}'}")

    def _azzera_toni(self):
        self.luce.set(0)
        self.contrasto.set(0)

    def alla_sorgente(self, px, py):
        """Da coordinate di questa vista a coordinate del FILE originale.

        Senza T_vista sono la stessa cosa. Con T_vista, la vista vive nel
        sistema del riferimento e la posizione nel file si ottiene applicando
        la trasformazione: e' cosi' che i punti cliccati al secondo giro
        restano memorizzati in coordinate originali, e il loro file resta
        valido da solo invece di dipendere dai giri precedenti.
        """
        if self.T_vista is None:
            return float(px), float(py)
        m = self._porta(np.array([[px, py]], dtype=np.float64))[0]
        return float(m[0]), float(m[1])

    def dalla_vista(self, xm, ym):
        """Dove appare, in questa vista, un punto dato in coordinate del FILE.

        E' l'inverso di alla_sorgente, e serve a DISEGNARE i punti gia' messi
        sulla vista gia' corretta: senza, la radiografia resta senza croci e non
        si puo' controllare se i punti sono finiti dove si voleva. Lo scarto fra
        la croce e il dettaglio che si vede sotto e' esattamente quel che resta
        da correggere, quindi guardarlo dice piu' di qualunque numero.

        L'inverso non ha forma chiusa perche' c'e' il campo: si parte
        dall'inversa dell'affine e si aggiusta qualche volta. Il campo e' liscio
        e piccolo rispetto all'affine, quindi bastano tre giri.
        """
        if self.T_vista is None:
            return float(xm), float(ym)
        A = self.T_vista.M[:, :2]
        Ainv = np.linalg.inv(A)
        q = np.array([float(xm), float(ym)])
        p = Ainv @ (q - self.T_vista.M[:, 2])
        for _ in range(3):
            q_ora = self._porta(p.reshape(1, 2))[0]
            p = p + Ainv @ (q - q_ora)
        return float(p[0]), float(p[1])

    def _porta(self, xy):
        """Da coordinate del riferimento a coordinate del file, campo compreso.

        Il campo si interpola con cv2.remap, che pero' rifiuta mappe con piu' di
        32767 righe o colonne: passandogli i punti come un'unica colonna, una
        vista da 1800x1800 ne produce 3.2 milioni e la chiamata fallisce. Il
        disegno saltava e il riquadro restava nero. Si rimodellano quindi i
        punti in un rettangolo con i lati sotto il limite, si interpola, e si
        rimette tutto in fila.
        """
        T = self.T_vista
        p = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))]) @ T.M.T
        if T.field is None:
            return p
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

    def _ritaglio_portato(self, xa, ya, xb, yb, passo):
        """Legge dal file la zona che, portata, riempie il riquadro richiesto.

        Si campiona il file al passo corrispondente invece di leggerlo tutto:
        anche con la vista sull'intero dipinto si toccano pochi megabyte.
        """
        nx = max(1, int(np.ceil((xb - xa) / passo)))
        ny = max(1, int(np.ceil((yb - ya) / passo)))
        gx, gy = np.meshgrid(xa + passo * np.arange(nx, dtype=np.float64),
                             ya + passo * np.arange(ny, dtype=np.float64))
        m = self._porta(np.column_stack([gx.ravel(), gy.ravel()]))
        mx, my = m[:, 0].reshape(ny, nx), m[:, 1].reshape(ny, nx)
        bx0 = int(np.clip(np.floor(np.nanmin(mx)), 0, self.Wm - 1))
        bx1 = int(np.clip(np.ceil(np.nanmax(mx)) + 1, 1, self.Wm))
        by0 = int(np.clip(np.floor(np.nanmin(my)), 0, self.Hm - 1))
        by1 = int(np.clip(np.ceil(np.nanmax(my)) + 1, 1, self.Hm))
        if bx1 <= bx0 or by1 <= by0:
            return None
        pm = max(1, int(round(passo * self.scala_vista)))
        src = np.ascontiguousarray(self.arr[by0:by1:pm, bx0:bx1:pm])
        if src.size == 0:
            return None
        fuori = cv2.remap(src, ((mx - bx0) / pm).astype(np.float32),
                          ((my - by0) / pm).astype(np.float32),
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return self._rgb(fuori)

    def _mescola(self, vista, xa, ya, xb, yb, passo, cw, ch, M, interp):
        """Alterna a riquadri questa vista e quella dell'altra immagine.

        Serve a giudicare a occhio dove le due non combaciano: affiancate, lo
        scarto di venti pixel non si nota; alternate a scacchiera, un bordo che
        salta da un riquadro all'altro si vede subito.

        Ha senso solo quando questa vista e' gia' portata nel sistema del
        riferimento: li' le due immagini hanno le stesse coordinate, quindi i
        riquadri mostrano davvero lo stesso pezzo di dipinto.
        """
        altra = np.ascontiguousarray(self.arr_confronto[ya:yb:passo, xa:xb:passo])
        if altra.ndim == 3:
            altra = cv2.cvtColor(altra, cv2.COLOR_RGB2GRAY)
        f = altra.astype(np.float32)
        d = f[f > 0]
        if d.size:
            lo, hi = np.percentile(d, [1.0, 99.0])
            f = (f - lo) / max(hi - lo, 1e-6)
        alt8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        alt8 = cv2.cvtColor(alt8, cv2.COLOR_GRAY2RGB)
        alt = cv2.warpAffine(alt8, M, (cw, ch), flags=interp,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 32, 32))
        lato = max(16, int(self.lato_scacco))
        yy, xx = np.mgrid[0:ch, 0:cw]
        scacchi = (((yy // lato) + (xx // lato)) % 2).astype(bool)
        fuori = np.where(scacchi[:, :, None], alt, vista)
        # una griglia sottile: senza, non si capisce dove finisce un riquadro e
        # comincia l'altro, e un salto al confine sembra un dettaglio
        fuori[::lato, :] = (255, 200, 0)
        fuori[:, ::lato] = (255, 200, 0)
        return fuori

    def _mostra_vista(self):
        """Porta sul canvas l'ultima vista, applicando luminosita' e contrasto.

        La correzione si fa qui e non in _rgb perche' qui l'immagine e' grande
        quanto il canvas (meno di mezzo megapixel) invece che quanto il ritaglio
        letto dal file (fino a quattro): trascinare lo slider resta fluido, e
        non c'e' bisogno di rileggere i pixel dal disco a ogni scatto.
        """
        vista = self._vista
        if vista is None:
            return
        if self._lut is not None:
            vista = cv2.LUT(vista, self._lut)
        cw, ch = vista.shape[1], vista.shape[0]
        try:
            from PIL import Image as PILImage, ImageTk
            self.foto = ImageTk.PhotoImage(PILImage.fromarray(vista))
        except Exception:
            import base64
            testa = f"P6 {cw} {ch} 255 ".encode()
            self.foto = tk.PhotoImage(
                data=base64.b64encode(testa + vista.tobytes()).decode())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.foto)
        self._disegna_punti()

    def _ricalcola_lut(self):
        """Luminosita' e contrasto in una sola tabella da 256 valori.

        La luminosita' e' una GAMMA, non un'aggiunta costante: sommare un valore
        fisso spingerebbe i chiari fuori scala e li appiattirebbe tutti sul
        bianco, che su una radiografia vuol dire perdere proprio i dettagli piu'
        densi. La gamma sposta i mezzitoni lasciando stare il nero e il bianco.

        Il contrasto allarga o stringe attorno al mezzo tono. Sulla radiografia
        e' spesso piu' utile della luminosita': la densita' della biacca occupa
        una fetta stretta della scala, e allargarla fa saltare fuori i contorni
        che a occhio nudo non si distinguono.

        Con entrambi a 0 la tabella non si applica affatto, quindi l'immagine e'
        esattamente quella di prima che gli slider esistessero.
        """
        luce, contrasto = float(self.luce.get()), float(self.contrasto.get())
        if abs(luce) < 1e-6 and abs(contrasto) < 1e-6:
            self._lut = None
        else:
            x = np.arange(256, dtype=np.float32) / 255.0
            k = 2.0 ** (contrasto / 50.0)        # +100 -> 4x, -100 -> 1/4
            x = np.clip((x - 0.5) * k + 0.5, 0.0, 1.0)
            gamma = 2.0 ** (-luce / 50.0)        # +100 -> 0.25 (chiaro)
            self._lut = np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)
        self._mostra_vista()

    def _disegna_punti(self):
        for i, (px, py) in enumerate(self.punti, 1):
            x, y = self._a_schermo(px, py)
            self.canvas.create_line(x - 9, y, x + 9, y, fill="#00e5ff", width=2)
            self.canvas.create_line(x, y - 9, x, y + 9, fill="#00e5ff", width=2)
            self.canvas.create_text(x + 12, y - 10, text=str(i), fill="#00e5ff",
                                    anchor="w", font=("Segoe UI", 9, "bold"))
        if self.provvisorio is not None:
            x, y = self._a_schermo(*self.provvisorio)
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline="#ffd400", width=2)
        if self.suggerito is not None:
            x, y = self._a_schermo(*self.suggerito)
            self.canvas.create_line(x - 16, y, x + 16, y, fill="#ff5722", dash=(3, 3))
            self.canvas.create_line(x, y - 16, x, y + 16, fill="#ff5722", dash=(3, 3))
            # Il cerchio e' la parte onesta del suggerimento: dice che il
            # dettaglio sta "da qualche parte qui dentro", non esattamente li'.
            # Senza, una crocetta sola fa credere a una precisione che non c'e'.
            if self.raggio_sugg > 0:
                r = self.raggio_sugg * self.zoom
                if r > 6:
                    self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                            outline="#ff5722", dash=(4, 4))

    # ------------------------------------------------------------ coordinate
    def _a_schermo(self, px, py):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        return ((px - (self.cx - cw / self.zoom / 2)) * self.zoom,
                (py - (self.cy - ch / self.zoom / 2)) * self.zoom)

    def _a_immagine(self, ev):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        return (self.cx - cw / self.zoom / 2 + ev.x / self.zoom,
                self.cy - ch / self.zoom / 2 + ev.y / self.zoom)

    # ---------------------------------------------------------------- eventi
    def _clic(self, ev):
        px, py = self._a_immagine(ev)
        if not (0 <= px < self.W and 0 <= py < self.H):
            return
        # Un clic vale quanto il pixel che si sta guardando: rimpicciolita per
        # stare nello schermo, l'immagine ne mostra uno ogni 12-14 veri, e il
        # punto nasce gia' sbagliato di altrettanto. Misurato su venti clic
        # fatti cosi': 17 px di scarto, non riconducibili a nessuna
        # deformazione. Quindi un clic da lontano non registra il punto:
        # porta la vista al 100% li' sopra, e si riclicca preciso.
        if self.zoom < self.zoom_min_clic:
            self.centra(px, py, zoom=1.0)
            self.stato.config(text="ingrandito al 100%: adesso clicca il punto preciso")
            return
        self.provvisorio = (px, py)
        self.disegna()
        self.al_clic(self, px, py)

    def _muovi(self, ev):
        px, py = self._a_immagine(ev)
        self.stato.config(text=f"zoom {self.zoom * 100:.0f}%   x {px:.0f}  y {py:.0f}")
        if self.al_movimento:
            self.al_movimento(self, px, py)

    def _inizio_trascino(self, ev):
        self._da = (ev.x, ev.y, self.cx, self.cy)

    def _trascina(self, ev):
        if not hasattr(self, '_da'):
            return
        x0, y0, cx0, cy0 = self._da
        self.cx = cx0 - (ev.x - x0) / self.zoom
        self.cy = cy0 - (ev.y - y0) / self.zoom
        self.disegna()

    def _rotella(self, ev):
        px, py = self._a_immagine(ev)
        fattore = 1.25 if ev.delta > 0 else 1 / 1.25
        nuovo = float(np.clip(self.zoom * fattore, min(0.02, self.zoom), 4.0))
        if nuovo == self.zoom:
            return
        # si zooma attorno al puntatore: il punto sotto il cursore resta fermo
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.zoom = nuovo
        self.cx = px - (ev.x - cw / 2) / self.zoom
        self.cy = py - (ev.y - ch / 2) / self.zoom
        self.disegna()

    def centra(self, px, py, zoom=None):
        if zoom:
            self.zoom = float(np.clip(zoom, 0.02, 4.0))
        self.cx, self.cy = px, py
        self.disegna()

    def larghezza_vista(self):
        """Quanti pixel dell'immagine sono inquadrati in orizzontale."""
        if not self.zoom:
            return float(self.W)
        return max(1.0, self.canvas.winfo_width() / self.zoom)

    def mostra_zona(self, px, py, raggio, zoom=None):
        """Porta la vista su (px, py) senza toccare l'ingrandimento.

        Lo zoom resta quello scelto dall'utente: cambiarglielo sotto le mani,
        anche per inquadrare tutto il cerchio d'incertezza, fa perdere il
        riferimento visivo che si era appena costruito, e un ingrandimento
        piccolo e' anche quello che rende impreciso il clic. Se il cerchio non
        ci sta dentro si vede lo stesso il tratteggio ai bordi, e la barra di
        stato dice di quanti pixel si puo' dover cercare attorno.
        """
        self.suggerito = (px, py)
        self.raggio_sugg = float(raggio)
        if zoom:
            self.zoom = float(np.clip(zoom, 0.02, 4.0))
        self.cx, self.cy = px, py
        self.disegna()


class FinestraPunti(tk.Toplevel):
    """Le due viste affiancate, l'elenco delle coppie e il riepilogo."""

    def __init__(self, padre, path_fixed, path_moving, punti=None, zone=None,
                 trasformazione=None):
        """
        Args:
            trasformazione: risultato dell'ultimo giro. Se c'e', la vista di
                destra mostra l'immagine GIA' ALLINEATA, cosi' ogni giro si
                occupa solo di quello che resta da correggere e non si rivedono
                i punti gia' messi. I punti nuovi vengono comunque memorizzati
                in coordinate dell'immagine originale, quindi l'insieme resta
                uno solo, riutilizzabile e indipendente dai giri fatti.
        """
        super().__init__(padre)
        self.title(f"Punti di corrispondenza — {os.path.basename(path_moving)}")
        self.risultato = None
        # `punti` arriva come array numpy quando si riapre la finestra sugli
        # stessi punti: `punti or []` sarebbe una verifica di verita' su un
        # array, che numpy rifiuta con "truth value is ambiguous". E' l'errore
        # che compariva la SECONDA volta che si aprivano i punti, mai la prima.
        self.coppie = ([] if punti is None
                       else [list(map(float, p)) for p in punti])
        # Zone che una verifica precedente ha segnato come non misurabili: sono
        # i posti dove un punto messo a mano aggiunge davvero informazione.
        self.zone = [z for z in (zone or []) if z.get('tipo') == 'non misurabile']
        self.i_zona = 0
        self.geometry("1280x820")

        # La regola sta in cima e da sola, perche' e' l'unica che, se non
        # seguita, peggiora il risultato invece di lasciarlo com'e'.
        tk.Label(self, justify="left", fg="#b00020", font=("Segoe UI", 9, "bold"),
                 text="Metti un punto SOLO dove riconosci con certezza lo stesso dettaglio in "
                      "tutte e due le immagini.\nDove non lo riconosci, passa oltre: meglio "
                      "nessun punto che uno inventato — non c'è niente da coprire."
                 ).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(self, justify="left", fg="#333", text=(
            "Clicca lo stesso dettaglio prima a sinistra e poi a destra: la coppia si aggiunge da "
            "sola, e l'altra vista si sposta sulla zona corrispondente senza cambiarti "
            "l'ingrandimento.\n"
            "Zooma prima di cliccare: un clic vale quanto il pixel che stai guardando.\n"
            "Bastano 4-8 punti sparsi. Zoom: rotella. Sposta: tasto destro trascinato. Gli slider "
            "sotto ogni vista ne regolano luminosità e contrasto.")
                 ).pack(anchor="w", padx=10, pady=(0, 4))

        corpo = tk.Frame(self)
        corpo.pack(fill="both", expand=True, padx=8)
        self.sinistra = Riquadro(corpo, path_fixed, "Riferimento", self._clic)
        self.sinistra.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.trasformazione = trasformazione
        self.destra = Riquadro(
            corpo, path_moving,
            "Da allineare" if trasformazione is None else
            "Da allineare — già corretta dal giro precedente",
            self._clic, T_vista=trasformazione,
            forma_vista=(self.sinistra.H, self.sinistra.W),
            arr_confronto=(self.sinistra.arr if trasformazione is not None else None))
        self.destra.pack(side="left", fill="both", expand=True, padx=(4, 0))

        lato = tk.Frame(self)
        lato.pack(fill="x", padx=10, pady=6)
        # selezione multipla: togliere cinque punti sbagliati uno alla volta,
        # riaprendo ogni volta l'elenco, e' una piccola tortura
        self.elenco = tk.Listbox(lato, height=6, font=("Consolas", 9),
                                 selectmode="extended")
        self.elenco.pack(side="left", fill="x", expand=True)
        pulsanti = tk.Frame(lato)
        pulsanti.pack(side="left", padx=8)
        tk.Button(pulsanti, text="Togli selezionato", command=self._togli).pack(fill="x")
        self.btn_sospetti = tk.Button(pulsanti, text="Togli i sospetti",
                                      command=self._togli_sospetti, state="disabled")
        self.btn_sospetti.pack(fill="x", pady=4)
        tk.Button(pulsanti, text="Vai al punto", command=self._vai).pack(fill="x")
        tk.Button(pulsanti, text="Pulisci", command=self._pulisci).pack(fill="x", pady=4)
        tk.Button(pulsanti, text="Carica da file...", command=self._carica).pack(fill="x")
        # La scacchiera ha senso solo dal secondo giro: prima le due viste hanno
        # coordinate diverse e alternarle non direbbe niente.
        if trasformazione is not None:
            self.scacchi = tk.BooleanVar(value=False)
            tk.Checkbutton(pulsanti, variable=self.scacchi, justify="left",
                           text="Sovrapponi a\nscacchiera").pack(fill="x", pady=(4, 0))
            self.scacchi.trace_add('write', lambda *a: self._cambia_scacchi())
        if self.zone:
            # Non "zona da coprire": non c'e' niente da coprire, e chiamarla
            # cosi' faceva sembrare un compito da portare a termine una cosa
            # che nella maggior parte dei casi va lasciata stare.
            tk.Button(pulsanti, text=f"Vedi dov'e' cieca ({len(self.zone)})",
                      command=self._zona_successiva).pack(fill="x", pady=(4, 0))

        piede = tk.Frame(self)
        piede.pack(fill="x", padx=10, pady=(0, 10))
        self.riepilogo = tk.Label(piede, fg="#555", anchor="w", justify="left",
                                  text="servono almeno 3 punti")
        self.riepilogo.pack(side="left", fill="x", expand=True)
        tk.Button(piede, text="Annulla", command=self.destroy).pack(side="right", padx=4)
        tk.Button(piede, text="Usa questi punti", command=self._conferma,
                  bg="#2196F3", fg="white").pack(side="right")

        # I punti caricati vanno segnati sulle viste subito: finora venivano
        # disegnati solo dopo un clic o dopo aver tolto qualcosa, quindi
        # riaprendo la finestra su un insieme gia' salvato non si vedeva NIENTE
        # su nessuna delle due immagini, e non c'era modo di controllare dove
        # fossero finiti.
        self._segna_punti()
        self._aggiorna()
        self.transient(padre)
        self.grab_set()
        self.bind("<Escape>", lambda e: self.destroy())

    # ------------------------------------------------------------------ punti
    def _clic(self, riquadro, px, py):
        altro = self.destra if riquadro is self.sinistra else self.sinistra
        if altro.provvisorio is None:
            # Appena si segna il punto in una vista, l'altra si porta sulla zona
            # corrispondente: e' il motivo per cui la finestra ha due viste e non
            # due file aperti a parte. Funziona dal primo clic, non solo quando
            # ci sono gia' dei punti (vedi _previsione).
            sugg = self._previsione(px, py, riquadro is self.sinistra)
            if sugg is not None:
                x, y, raggio, come = sugg
                altro.mostra_zona(x, y, raggio)
                mezza = altro.larghezza_vista() / 2.0
                altro.stato.config(
                    text=f"{come}: ±{raggio:.0f} px"
                         + ("" if raggio <= mezza else
                            "  — piu' largo di quanto inquadri: se non lo vedi, "
                            "trascina col tasto destro qui attorno"))
        if self.sinistra.provvisorio and self.destra.provvisorio:
            xf, yf = self.sinistra.provvisorio
            # Se la vista di destra mostra l'immagine gia' corretta, il clic e'
            # in coordinate del riferimento: si riporta subito su quelle del
            # file originale, cosi' l'insieme dei punti resta uno solo e non
            # dipende da quanti giri si sono fatti.
            xm, ym = self.destra.alla_sorgente(*self.destra.provvisorio)
            self.coppie.append([xf, yf, xm, ym])
            self.sinistra.provvisorio = self.destra.provvisorio = None
            self.sinistra.suggerito = self.destra.suggerito = None
            self.sinistra.raggio_sugg = self.destra.raggio_sugg = 0.0
            self._segna_punti()
            self.sinistra.disegna()
            self.destra.disegna()
            self._aggiorna()

    def _cambia_scacchi(self):
        """Accende o spegne la vista a scacchiera sulla destra."""
        self.destra.scacchiera = bool(self.scacchi.get())
        self.destra.titolo.config(
            text="Visibile e radiografia alternati a riquadri"
            if self.destra.scacchiera else "Da allineare — già corretta dal giro precedente")
        self.destra.disegna()

    def _segna_punti(self):
        """Quali punti disegnare sulle due viste.

        Si disegnano su tutte e due, sempre. Quando la vista di destra e' gia'
        corretta i punti vanno riportati nelle sue coordinate: la croce finisce
        allora dove il programma CREDE che stia quel dettaglio, e lo scarto fra
        la croce e quel che si vede sotto e' cio' che resta da correggere. E'
        l'unico modo di controllare a occhio se i punti sono a posto.
        """
        self.sinistra.punti = [(c[0], c[1]) for c in self.coppie]
        if self.trasformazione is None:
            self.destra.punti = [(c[2], c[3]) for c in self.coppie]
        else:
            self.destra.punti = [self.destra.dalla_vista(c[2], c[3])
                                 for c in self.coppie]

    def _previsione(self, px, py, avanti=True):
        """Dove cade, nell'altra immagine, il punto appena cliccato?

        Returns:
            (x, y, raggio d'incertezza in px, come e' stata ottenuta) oppure None.

        Sotto i 3 punti non c'e' nessun modello da adattare, ma una previsione
        si puo' fare lo stesso: le due riprese inquadrano lo stesso dipinto, per
        cui la stessa frazione di larghezza e di altezza cade piu' o meno sullo
        stesso dettaglio. Misurato in leave-one-out sui 18 punti veri della
        radiografia (p90 dell'errore, immagine da 22824 px di lato):

            sole proporzioni       352 px      (1.5% del lato)
            proporzioni + 1 punto  653 px
            similarita' su 2 punti 1347 px, con casi oltre 4000
            modello su 3 punti     484 px
            modello su 8 punti     390 px, mediana 51

        Il risultato contro-intuitivo e' che UNO o DUE punti peggiorano la
        previsione invece di migliorarla: le proporzioni azzeccano gia' la scala
        d'insieme, mentre ancorarsi a un clic solo ci somma il suo errore
        locale, e una similarita' su due punti vicini fra loro e' malcondizionata.
        Quindi sotto i 3 punti si usano sempre e solo le proporzioni.
        """
        da = self.sinistra if avanti else self.destra
        a = self.destra if avanti else self.sinistra
        if self.trasformazione is not None:
            # Le due viste sono nello stesso sistema di coordinate: il dettaglio
            # sta nello stesso posto, a meno di quel che resta da correggere.
            # Il raggio e' l'ampiezza tipica di cio' che resta, misurata sui
            # punti gia' messi.
            raggio = 80.0
            if len(self.coppie) >= 3:
                try:
                    P = np.asarray(self.coppie, float)
                    d = P[:, 2:4] - self.destra._porta(P[:, :2])
                    raggio = float(np.clip(np.percentile(np.hypot(*d.T), 90),
                                           30.0, 400.0))
                except Exception:
                    pass
            return (px, py, raggio, "stessa posizione (la vista e' gia' corretta)")
        if len(self.coppie) >= 3:
            try:
                M, _, _ = punti_manuali.affine_da_punti(self.coppie)
                if not avanti:
                    M = cv2.invertAffineTransform(M)
                p = np.array([px, py, 1.0]) @ M.T
                if 0 <= p[0] < a.W and 0 <= p[1] < a.H:
                    return (float(p[0]), float(p[1]), self._incertezza(a),
                            f"previsione dai tuoi {len(self.coppie)} punti")
                # Fuori immagine: i punti sono incoerenti fra loro e il modello
                # manda la vista fuori dal dipinto. Meglio tornare alle
                # proporzioni, che almeno non sbagliano bersaglio.
            except Exception:
                pass
        # 3.5% del lato: e' il MASSIMO misurato sui punti veri (790 px su 22824),
        # non il tipico. Il primo punto e' l'unico per cui non c'e' modo di fare
        # meglio, e annunciare un cerchio troppo stretto proprio li' manderebbe
        # a cercare nel posto sbagliato.
        return (px * a.W / da.W, py * a.H / da.H,
                0.035 * max(a.W, a.H), "zona approssimativa")

    def _incertezza(self, bersaglio):
        """Di quanto sbaglia il modello adattato ai punti gia' messi.

        Non e' una tabella fissa: si misura sui punti dell'utente stesso,
        lasciandone fuori uno alla volta e guardando di quanto si sbaglia a
        prevederlo con gli altri. Se i clic sono coerenti il raggio si stringe da
        solo e la vista si chiude sul dettaglio; se litigano fra loro il raggio
        resta largo, ed e' giusto cosi'.
        """
        P = np.asarray(self.coppie, dtype=float)
        if len(P) < 5:
            # Con 3 o 4 punti il leave-one-out ne lascerebbe 2 o 3: una
            # similarita' ci passa esattamente dentro e l'errore misurato
            # sarebbe zero, cioe' una bugia. Si usa il valore misurato sui
            # punti veri della radiografia.
            return 480.0
        err = []
        for i in range(len(P)):
            try:
                M, _, _ = punti_manuali.affine_da_punti(np.delete(P, i, axis=0))
            except Exception:
                continue
            p = np.array([P[i, 0], P[i, 1], 1.0]) @ M.T
            err.append(np.hypot(p[0] - P[i, 2], p[1] - P[i, 3]))
        if not err:
            return 480.0
        return float(np.clip(np.percentile(err, 90), 25.0,
                             0.05 * max(bersaglio.W, bersaglio.H)))

    def _aggiorna(self):
        self.elenco.delete(0, "end")
        self.controllo = None
        if len(self.coppie) < 3:
            self.riepilogo.config(
                text=f"{len(self.coppie)} punti: ne servono almeno 3 (meglio 4-8)")
        else:
            try:
                self.controllo = punti_manuali.controlla_punti(self.coppie)
                self.riepilogo.config(text=self._riga_riepilogo(
                    self.controllo, int(self.controllo['sospetti'].sum())))
            except Exception as exc:
                self.riepilogo.config(text=f"punti non utilizzabili: {exc}")
        self.btn_sospetti.config(
            state="normal" if self.controllo is not None
            and self.controllo['sospetti'].any() else "disabled",
            text=(f"Togli i {int(self.controllo['sospetti'].sum())} sospetti"
                  if self.controllo is not None and self.controllo['sospetti'].any()
                  else "Togli i sospetti"))

        c = self.controllo
        for i, p in enumerate(self.coppie, 1):
            if c is None:
                coda = ""
            else:
                # Due numeri, non uno: "scarto" e' la distanza dal modello
                # globale, "vicini" quella dai punti intorno. Il primo su una
                # tela deformata e' alto anche quando il punto e' giusto; il
                # secondo no, ed e' quello che dice se il clic e' sbagliato.
                coda = (f"  scarto {c['residui'][i - 1]:5.1f}"
                        f"  vicini {c['locali'][i - 1]:5.1f}"
                        + ("  <- rivedi" if c['sospetti'][i - 1] else ""))
            self.elenco.insert("end", f"{i:2d}  ({p[0]:7.0f},{p[1]:7.0f}) -> "
                                      f"({p[2]:7.0f},{p[3]:7.0f}){coda}")
            if c is not None and c['sospetti'][i - 1]:
                self.elenco.itemconfig(i - 1, foreground="#c62828")

    def _riga_riepilogo(self, c, n_sosp):
        """Il riepilogo sotto l'elenco, scritto per non far inseguire un numero
        che non puo' scendere.

        Sulla radiografia di prova lo scarto dall'affine resta sui 19 px per
        quanti punti si aggiungano, perche' e' in gran parte deformazione vera
        della tela e non imprecisione dei clic (vedi punti_manuali.controlla_punti).
        Dirlo qui evita di far rifare all'infinito dei punti che vanno bene.
        """
        testo = punti_manuali.descrivi(c['M'], c['residui'], c['modello'])
        if n_sosp:
            testo += (f"\n{n_sosp} punt{'o' if n_sosp == 1 else 'i'} in rosso: "
                      f"non concord{'a' if n_sosp == 1 else 'ano'} con i punti "
                      f"vicini, e quello si' che e' un clic da rivedere.")
        else:
            testo += ("\nNessun punto in contrasto con i suoi vicini: vanno bene tutti. "
                      "Lo 'scarto' alto e' la deformazione della tela, che l'affine di "
                      "partenza non puo' seguire e che il campo non rigido correggera' "
                      "dopo: non e' un difetto dei tuoi clic e non scende aggiungendone.")
        return testo

    def _togli_sospetti(self):
        """Toglie i punti che litigano con i vicini, e li dice."""
        c = self.controllo
        if c is None or not c['sospetti'].any():
            return
        quali = [i + 1 for i in np.flatnonzero(c['sospetti'])]
        if not messagebox.askyesno(
                "Togliere i punti sospetti?",
                f"Verrebbero tolti i punti {', '.join(map(str, quali))}: sono quelli "
                f"il cui scarto non assomiglia a quello dei punti vicini, cioe' i "
                f"candidati a essere un clic finito sul dettaglio sbagliato.\n\n"
                f"Restano {len(self.coppie) - len(quali)} punti.", parent=self):
            return
        self.coppie = [p for i, p in enumerate(self.coppie) if not c['sospetti'][i]]
        self._ridisegna_tutto()

    def _togli(self):
        for i in reversed(self.elenco.curselection()):
            del self.coppie[i]
        self._ridisegna_tutto()

    def _pulisci(self):
        self.coppie = []
        self._ridisegna_tutto()

    def _carica(self):
        """Riprende i punti di una corsa precedente (il file accanto ai parametri).

        Serve soprattutto per tenere i punti buoni di un tentativo e rifare solo
        quelli incoerenti, invece di ricominciare da capo.
        """
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="Punti salvati", parent=self,
            filetypes=[("Punti", "*_punti.json"), ("JSON", "*.json"), ("Tutti", "*.*")])
        if not p:
            return
        try:
            import json
            with open(p, encoding='utf-8-sig') as fh:
                dati = json.load(fh)
            caricati = np.asarray(dati['punti'], dtype=float).reshape(-1, 4)
        except Exception as exc:
            messagebox.showerror("Non leggibile", str(exc), parent=self)
            return
        # I punti valgono per UNA coppia: quelli di un'altra immagine hanno
        # coordinate che non c'entrano nulla. E' successo davvero, caricando il
        # file di una radiografia mentre se ne allineava un'altra.
        atteso = os.path.normcase(os.path.abspath(self.destra.path))
        trovato = os.path.normcase(os.path.abspath(dati.get('moving', '')))
        if trovato and trovato != atteso:
            if not messagebox.askyesno(
                    "Punti di un'altra immagine",
                    f"Questi punti erano stati messi su:\n{os.path.basename(dati['moving'])}\n\n"
                    f"Adesso stai allineando:\n{os.path.basename(self.destra.path)}\n\n"
                    f"Le coordinate della seconda immagine non valgono per questa: "
                    f"userebbero dettagli che stanno altrove.\n\nCaricarli lo stesso?",
                    parent=self, default='no'):
                return
        self.coppie = [list(map(float, c)) for c in caricati]
        self._ridisegna_tutto()

    def _zona_successiva(self):
        """Porta la vista sulla prossima zona che l'automatico non sa misurare."""
        if not self.zone:
            return
        z = self.zone[self.i_zona % len(self.zone)]
        self.i_zona += 1
        x, y = z['centro_px']
        self.sinistra.centra(x, y, zoom=1.0)
        sugg = self._previsione(x, y)
        if sugg is not None:
            # Qui l'ingrandimento si imposta, ma perche' e' una navigazione
            # chiesta apposta ("portami sulla zona da coprire"), non un effetto
            # collaterale di un clic: la vista di sinistra va al 100% e la
            # destra la segue.
            self.destra.mostra_zona(sugg[0], sugg[1], sugg[2], zoom=1.0)
        self.riepilogo.config(
            text=f"zona {self.i_zona} di {len(self.zone)} in ({x}, {y}): qui la verifica "
                 f"e' cieca ({z['n_finestre']} finestre), non sbagliata.\n"
                 f"Se NON riconosci con certezza lo stesso dettaglio in tutte e due, "
                 f"passa oltre e non metterci niente: un punto inventato fa piu' danno "
                 f"che nessun punto.")

    def _vai(self):
        sel = self.elenco.curselection()
        if not sel:
            return
        c = self.coppie[sel[0]]
        self.sinistra.centra(c[0], c[1], zoom=1.0)
        self.destra.centra(c[2], c[3], zoom=1.0)

    def _ridisegna_tutto(self):
        self._segna_punti()
        self.sinistra.provvisorio = self.destra.provvisorio = None
        self.sinistra.suggerito = self.destra.suggerito = None
        self.sinistra.disegna()
        self.destra.disegna()
        self._aggiorna()

    def _conferma(self):
        if len(self.coppie) < 3:
            messagebox.showwarning("Troppo pochi punti",
                                   "Servono almeno 3 punti, meglio 4-8.", parent=self)
            return
        # Si avvisa solo per i punti che litigano con i VICINI. Il criterio di
        # prima guardava lo scarto dal modello globale e scattava sempre: su
        # questa radiografia quello scarto vale 19 px anche con clic ottimi,
        # perche' e' la deformazione della tela, e l'avviso diceva "rifai i piu'
        # incoerenti" mandando a rifare punti che andavano benissimo.
        try:
            c = punti_manuali.controlla_punti(self.coppie)
        except Exception as exc:
            messagebox.showerror("Punti non utilizzabili", str(exc), parent=self)
            return
        quali = [i + 1 for i in np.flatnonzero(c['sospetti'])]
        if quali:
            if not messagebox.askyesno(
                    "Qualche punto non concorda con i vicini",
                    f"I punti {', '.join(map(str, quali))} hanno uno scarto che non "
                    f"assomiglia a quello dei punti intorno: di solito vuol dire un clic "
                    f"finito sul dettaglio sbagliato.\n\n"
                    f"Il pulsante «Togli i sospetti» li leva tutti insieme.\n\n"
                    f"Attenzione ai punti sul BORDO del dipinto: nella radiografia "
                    f"quella zona mostra il telaio e la tela, non la pittura, quindi "
                    f"l'angolo che vedi nel visibile non e' lo stesso punto fisico.\n\n"
                    f"Usarli lo stesso?", parent=self, default='no'):
                return
        self.risultato = np.array(self.coppie, dtype=float)
        self.destroy()
