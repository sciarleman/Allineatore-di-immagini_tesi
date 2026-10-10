"""
Finestra per disegnare a mano una zona o un contorno su un'immagine.

Serve per due cose diverse, che condividono lo stesso disegno:

- la MASCHERA sul riferimento, che limita la stima e la verifica alla zona del
  dipinto;
- il CONTORNO della tela sulla radiografia, che diventa un vincolo di forma al
  bordo dove la verifica non riesce a misurare.

Entrambi si tracciano a mano, e non si ricavano automaticamente. Sul soggetto di
questo progetto qualunque criterio automatico sceglie la parte sbagliata: il
contorno piu' marcato dell'immagine e' lo spigolo tridimensionale della tela,
cioe' proprio cio' che si vuole escludere, e lo sfondo (una lavagna con scritte
a gesso) ha piu' contrasto della superficie dipinta. Provato anche sulla
radiografia, dove il nero attorno sembrava dare un contorno affidabile: ne
usciva una curva che in alcuni tratti seguiva il bordo dell'immagine invece
della tela, e che avrebbe spostato il dipinto di 84 px da un lato e 0
dall'altro.
"""

import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import cv2

import image_io


class FinestraMaschera(tk.Toplevel):
    """
    Disegno della zona da usare per la stima. Questo step aggiunge solamente una maschera per indicare dove lavorare

    Serve solo nel caso in cui il fondo abbia struttura in ENTRAMBE le immagini.
    Allineare sfondi diversi rischia di creare aberrazioni. Negli altri casi i filtri
    automatici bastano e questa finestra non serve.
    """

    def __init__(self, padre, percorso_riferimento, titolo=None, scopo=None):
        """
        Args:
            titolo: che cosa si sta disegnando, per la barra della finestra.
            scopo: una riga che spiega a che serve, mostrata in alto. Serve
                perche' la stessa finestra traccia due cose diverse - la zona
                da misurare sul riferimento e il bordo della tela sulla
                radiografia - e confonderle produrrebbe un risultato sbagliato
                senza nessun segnale.
        """
        super().__init__(padre)
        # Il nome del file nel titolo non e' un dettaglio: la zona si disegna
        # sempre sull'immagine di RIFERIMENTO, mai su quelle da allineare, e
        # vedendola in scala di grigi e' facile scambiarla per un'altra ripresa.
        self.title((titolo or "Selezione maschera")
                   + f" — {os.path.basename(percorso_riferimento)}")
        self.scopo = scopo
        self.maschera = None
        self.descrizione = None  # forma disegnata, in frazioni: va nei parametri
        self.punti = []          # SEMPRE in coordinate dell'immagine, mai della vista:
        self.modo = tk.StringVar(value="poly")   # cosi' lo zoom non sposta nulla
        self.trascino = None
        self.cerchio_esatto = False
        self.foto = None
        self.off_x = self.off_y = 0

        # A colori, non in scala di grigi: la stima userebbe comunque solo il
        # grigio, ma qui conta riconoscere a colpo d'occhio QUALE ripresa si sta
        # guardando. Un'immagine in bianco e nero si scambia facilmente per
        # un'altra banda.
        # Anteprima presa a passo, non mediata: qui serve solo da guardare, e
        # leggere l'intero gigabyte per disegnarci sopra un poligono costava
        # qualche secondo a ogni apertura della finestra.
        anteprima, _ = image_io.anteprima_veloce(percorso_riferimento, 1600,
                                                 in_grigio=False)
        # a_8bit: una ripresa a 16 bit va riportata a 8 bit sui percentili, altrimenti
        # non e' visualizzabile (e PIL non saprebbe nemmeno che farsene).
        anteprima = image_io.a_8bit(anteprima)
        self.h, self.w = anteprima.shape[:2]
        self.rgb = (anteprima if anteprima.ndim == 3
                    else cv2.cvtColor(anteprima, cv2.COLOR_GRAY2RGB))

        # La finestra non deve mai superare lo schermo: 
        # si calcola lo spazio davvero disponibile (togliendo barra delle applicazioni e i bordi) e ci si adatta. 
    
        schermo_w = self.winfo_screenwidth()
        schermo_h = self.winfo_screenheight()
        disp_w = max(320, schermo_w - 120)
        disp_h = max(240, schermo_h - 280)

        # Prima si decide lo zoom che fa entrare tutta l'immagine nello spazio
        # disponibile, POI si dimensiona il canvas sull'immagine cosi' ridotta.
        # Facendo il contrario (canvas largo quanto l'anteprima, zoom calcolato
        # dopo) un'immagine verticale lascia una enorme fascia vuota a destra:
        # lo zoom lo limita l'altezza, ma la larghezza resta quella piena.
        self.zoom_min = min(disp_w / self.w, disp_h / self.h, 1.0)
        self.zoom = self.zoom_min
        self.vista_w = max(320, min(disp_w, int(self.w * self.zoom_min) + 2))
        self.vista_h = max(240, min(disp_h, int(self.h * self.zoom_min) + 2))

        barra = tk.Frame(self, padx=8, pady=6)
        barra.pack(fill="x")
        if self.scopo:
            tk.Label(self, text=self.scopo, fg="#b00020", justify="left",
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4, 0))
        tk.Label(barra, text="Forma:").pack(side="left")
        # Etichette corte: il dettaglio su come si disegna ogni forma sta nel
        # piede della finestra. Una barra piu' stretta lascia decidere la
        # larghezza all'immagine, che e' cio' che conta guardare.
        tk.Radiobutton(barra, text="Poligono", variable=self.modo,
                       value="poly", command=self.pulisci).pack(side="left", padx=4)
        tk.Radiobutton(barra, text="Rettangolo", variable=self.modo,
                       value="rect", command=self.pulisci).pack(side="left", padx=4)
        tk.Radiobutton(barra, text="Cerchio / ellisse", variable=self.modo,
                       value="ellisse", command=self.pulisci).pack(side="left", padx=4)
        tk.Button(barra, text="-", width=3, command=lambda: self.zooma(1 / 1.25)).pack(side="left", padx=(14, 2))
        self.lbl_zoom = tk.Label(barra, text="100%", width=6)
        self.lbl_zoom.pack(side="left")
        tk.Button(barra, text="+", width=3, command=lambda: self.zooma(1.25)).pack(side="left", padx=2)
        tk.Button(barra, text="Adatta", command=self.adatta).pack(side="left", padx=(6, 0))
        tk.Button(barra, text="Annulla ultimo", command=self.annulla_ultimo).pack(side="right", padx=3)
        tk.Button(barra, text="Pulisci", command=self.pulisci).pack(side="right", padx=3)

        cornice = tk.Frame(self)
        cornice.pack(padx=8, fill="both", expand=True)
        self.canvas = tk.Canvas(cornice, width=self.vista_w, height=self.vista_h,
                                cursor="crosshair", highlightthickness=0, bg="#202020")
        sb_x = tk.Scrollbar(cornice, orient="horizontal", command=self.canvas.xview)
        sb_y = tk.Scrollbar(cornice, orient="vertical", command=self.canvas.yview)
        self.canvas.config(xscrollcommand=sb_x.set, yscrollcommand=sb_y.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")
        cornice.rowconfigure(0, weight=1)
        cornice.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.clic)
        self.canvas.bind("<B1-Motion>", self.trascina)
        self.canvas.bind("<ButtonRelease-1>", self.rilascia)
        # Se la finestra viene ridimensionata cambia lo spazio vuoto ai lati, e
        # con esso lo scostamento di centratura: va ricalcolato, altrimenti i
        # punti gia' disegnati finirebbero fuori posto.
        self.canvas.bind("<Configure>", self._mostra_immagine)
        # Trascinamento della vista col tasto destro, utile quando si e' zumati
        self.canvas.bind("<Button-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        # Ctrl + rotella = zoom; rotella da sola = scorrimento verticale
        self.canvas.bind("<Control-MouseWheel>", self.rotella_zoom)
        self.canvas.bind("<MouseWheel>",
                         lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>",
                         lambda e: self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        for sequenza in ("<Control-plus>", "<Control-KP_Add>", "<Control-equal>"):
            self.bind(sequenza, lambda e: self.zooma(1.25))
        for sequenza in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind(sequenza, lambda e: self.zooma(1 / 1.25))
        self.bind("<Control-0>", lambda e: self.adatta())
        self.bind("<Escape>", lambda e: self.annulla())

        piede = tk.Frame(self, padx=8, pady=8)
        piede.pack(fill="x")

        # I pulsanti per primi, ancorati a destra: cosi' hanno il loro spazio
        # riservato e nessun testo puo' spingerli fuori dalla finestra.
        tk.Button(piede, text="Annulla", command=self.annulla).pack(side="right", padx=4)
        tk.Button(piede, text="Conferma", command=self.conferma,
                  bg="#2196F3", fg="white").pack(side="right")

        # Le due spiegazioni vanno IMPILATE, non affiancate. Due Label con
        # side="left" nello stesso frame finiscono una accanto all'altra, e la
        # somma delle loro larghezze allarga la finestra a dismisura: per
        # metterle una sotto l'altra servono in un frame loro, impacchettate
        # con anchor="w". Cosi' si evitano anche gli "\n" iniziali usati per
        # spingerle in basso, che portavano con se' degli spazi e spostavano il
        # testo verso destra.
        testi = tk.Frame(piede)
        testi.pack(side="left", fill="x")
        tk.Label(testi, fg="gray", justify="left",
                 text="Racchiudi il dipinto: cio' che resta fuori non verra' usato per stimare la\n"
                      "trasformazione (ma verra' comunque trasformato e salvato nell'immagine finale).\n"
                      "Poligono: un clic per vertice.  Rettangolo ed ellisse: si trascinano "
                      "(l'ellisse dal centro).").pack(anchor="w")
        tk.Label(testi, fg="black", justify="left",
                 text="Shift mentre trascini l'ellisse = cerchio perfetto.\n"
                      "Zoom: Ctrl + rotella, oppure Ctrl +/-.  Ctrl+0 adatta.  "
                      "Tasto destro trascinato = sposta la vista.").pack(anchor="w", pady=(6, 0))

        self._mostra_immagine()
        self.transient(padre)
        self.grab_set()
        self.focus_set()

    # ------------------------------------------------------------ vista
    def _mostra_immagine(self, _evento=None):
        """Ridisegna l'immagine alla scala corrente e aggiorna l'area scorribile.

        L'immagine viene CENTRATA nel canvas. La larghezza minima della finestra
        la impongono la barra degli strumenti e il piede, quindi con un'immagine
        verticale avanza sempre dello spazio ai lati: centrata sembra voluta,
        appoggiata a sinistra sembra un errore di impaginazione.
        """
        nw, nh = max(1, int(self.w * self.zoom)), max(1, int(self.h * self.zoom))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.off_x = max(0, (cw - nw) // 2) if cw > 1 else 0
        self.off_y = max(0, (ch - nh) // 2) if ch > 1 else 0
        interp = cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_NEAREST
        vista = cv2.resize(self.rgb, (nw, nh), interpolation=interp)
        try:
            from PIL import Image as PILImage, ImageTk
            self.foto = ImageTk.PhotoImage(PILImage.fromarray(vista))
        except Exception:
            # Senza ImageTk si passa da un PPM in memoria, che tk sa leggere.
            import base64
            intestazione = f"P6 {nw} {nh} 255 ".encode()
            self.foto = tk.PhotoImage(
                data=base64.b64encode(intestazione + vista.tobytes()).decode())
        self.canvas.delete("sfondo")
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw",
                                 image=self.foto, tags="sfondo")
        self.canvas.tag_lower("sfondo")
        self.canvas.config(scrollregion=(0, 0, self.off_x + nw, self.off_y + nh))
        self.lbl_zoom.config(text=f"{self.zoom * 100:.0f}%")
        self.ridisegna()

    def zooma(self, fattore, ancora=None):
        """
        Cambia lo zoom mantenendo fermo il punto `ancora` (in coordinate vista).
        Senza ancora si tiene fermo il centro della vista.
        """
        nuovo = max(self.zoom_min * 0.5, min(8.0, self.zoom * fattore))
        if abs(nuovo - self.zoom) < 1e-6:
            return
        if ancora is None:
            ancora = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        # punto dell'immagine attualmente sotto l'ancora
        ix = (self.canvas.canvasx(ancora[0]) - self.off_x) / self.zoom
        iy = (self.canvas.canvasy(ancora[1]) - self.off_y) / self.zoom
        self.zoom = nuovo
        self._mostra_immagine()
        # rimetti quel punto sotto l'ancora
        nw, nh = max(1, self.w * self.zoom), max(1, self.h * self.zoom)
        self.canvas.xview_moveto(max(0.0, (ix * self.zoom + self.off_x - ancora[0]) / nw))
        self.canvas.yview_moveto(max(0.0, (iy * self.zoom + self.off_y - ancora[1]) / nh))

    def rotella_zoom(self, ev):
        self.zooma(1.25 if ev.delta > 0 else 1 / 1.25, ancora=(ev.x, ev.y))
        return "break"

    def adatta(self):
        self.zoom = self.zoom_min
        self._mostra_immagine()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _img(self, ev):
        """Coordinate immagine a partire da un evento del mouse.

        Si sottrae lo scostamento di centratura, altrimenti tutti i punti
        risulterebbero spostati della meta' dello spazio vuoto laterale.
        """
        return ((self.canvas.canvasx(ev.x) - self.off_x) / self.zoom,
                (self.canvas.canvasy(ev.y) - self.off_y) / self.zoom)

    # ---------------------------------------------------------- disegno
    def _semiassi(self):
        """Semiassi dell'ellisse in coordinate immagine, dai due punti memorizzati.

        Si trascina dal CENTRO: il primo punto e' il centro, il secondo dice
        quanto e' larga e quanto e' alta. Con Shift premuto i due semiassi
        vengono pareggiati, cioe' si ottiene un cerchio esatto.
        """
        (cx, cy), (px, py) = self.punti[0], self.punti[1]
        rx, ry = abs(px - cx), abs(py - cy)
        if self.cerchio_esatto:
            rx = ry = max(rx, ry)
        return (cx, cy), max(1.0, rx), max(1.0, ry)

    def ridisegna(self):
        self.canvas.delete("disegno")
        z = self.zoom
        ox, oy = self.off_x, self.off_y
        pt = [(x * z + ox, y * z + oy) for x, y in self.punti]
        if self.modo.get() == "ellisse":
            if len(self.punti) == 2:
                (cx, cy), rx, ry = self._semiassi()
                self.canvas.create_oval((cx - rx) * z + ox, (cy - ry) * z + oy,
                                        (cx + rx) * z + ox, (cy + ry) * z + oy,
                                        outline="#00e000", width=2, tags="disegno")
                self.canvas.create_line(cx * z + ox - 7, cy * z + oy, cx * z + ox + 7, cy * z + oy,
                                        fill="#00a000", tags="disegno")
                self.canvas.create_line(cx * z + ox, cy * z + oy - 7, cx * z + ox, cy * z + oy + 7,
                                        fill="#00a000", tags="disegno")
            return
        if self.modo.get() == "poly":
            for i, (x, y) in enumerate(pt):
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#00e000",
                                        outline="#004000", tags="disegno")
                if i:
                    self.canvas.create_line(*pt[i - 1], x, y, fill="#00e000",
                                            width=2, tags="disegno")
            if len(pt) > 2:
                self.canvas.create_line(*pt[-1], *pt[0], fill="#00a000",
                                        width=1, dash=(4, 3), tags="disegno")
        elif len(pt) == 2:
            (x0, y0), (x1, y1) = pt
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00e000", width=2,
                                         tags="disegno")

    def clic(self, ev):
        p = self._img(ev)
        if self.modo.get() == "poly":
            self.punti.append(p)
        else:
            self.punti = [p, p]
            self.trascino = True
            self.cerchio_esatto = bool(ev.state & 0x0001)   # Shift
        self.ridisegna()

    def trascina(self, ev):
        if self.modo.get() in ("rect", "ellisse") and self.trascino:
            self.punti[1] = self._img(ev)
            self.cerchio_esatto = bool(ev.state & 0x0001)
            self.ridisegna()

    def rilascia(self, _ev):
        self.trascino = False

    def annulla_ultimo(self):
        if self.punti:
            self.punti.pop()
            self.ridisegna()

    def pulisci(self):
        self.punti = []
        self.ridisegna()

    def conferma(self):
        m = np.zeros((self.h, self.w), np.uint8)
        punti = [(int(round(x)), int(round(y))) for x, y in self.punti]
        # Oltre al raster si conserva la FORMA disegnata, in frazioni di larghezza
        # e altezza: non dipende dalla risoluzione e finisce nel file dei
        # parametri, cosi' resta traccia di quale zona e' stata usata.
        fx = lambda p: [round(p[0] / self.w, 5), round(p[1] / self.h, 5)]
        if self.modo.get() == "poly" and len(punti) >= 3:
            cv2.fillPoly(m, [np.array(punti, np.int32)], 255)
            self.descrizione = {'forma': 'poligono',
                                'vertici_frazione': [fx(p) for p in self.punti]}
        elif self.modo.get() == "rect" and len(punti) == 2:
            (x0, y0), (x1, y1) = punti
            cv2.rectangle(m, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), 255, -1)
            self.descrizione = {'forma': 'rettangolo',
                                'vertici_frazione': [fx(p) for p in self.punti]}
        elif self.modo.get() == "ellisse" and len(self.punti) == 2:
            (cx, cy), rx, ry = self._semiassi()
            cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                        (int(round(rx)), int(round(ry))), 0, 0, 360, 255, -1)
            self.descrizione = {'forma': 'ellisse', 'centro_frazione': fx((cx, cy)),
                                'semiassi_frazione': [round(rx / self.w, 5),
                                                      round(ry / self.h, 5)]}
        else:
            messagebox.showwarning("Zona incompleta",
                                   "Servono almeno 3 vertici per un poligono, oppure un "
                                   "rettangolo o un'ellisse trascinati.", parent=self)
            return
        if m.mean() < 2:
            messagebox.showwarning("Zona troppo piccola",
                                   "La zona selezionata e' quasi vuota.", parent=self)
            return
        self.maschera = m
        self.destroy()

    def annulla(self):
        self.maschera = None
        self.descrizione = None
        self.destroy()


# Livelli di precisione offerti nella finestra principale. Il numero e' il
# fattore di riduzione usato per la STIMA: l'immagine finale viene comunque
# sempre scritta a piena risoluzione. Dimezzare il fattore quadruplica i pixel
# su cui si stima, quindi quadruplica all'incirca tempo e memoria.
