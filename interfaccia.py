"""
Interfaccia grafica per allineare un gruppo di immagini su un'immagine di
riferimento.

    python interfaccia.py

Impianto ripreso da TestUI.py (la vecchia interfaccia): riferimento, immagini da
allineare, cartella di salvataggio, opzioni, barra di avanzamento. Il motore
sotto e' pero' quello nuovo (align_batch/align_paintings): niente immagini intere
in RAM, una trasformazione stimata e VERIFICATA per ogni immagine, e un riepilogo
finale con l'errore residuo misurato a piena risoluzione.

L'elaborazione gira in un thread separato, cosi' la finestra resta reattiva;
tutto cio' che tocca l'interfaccia passa da una coda letta dal thread principale
(tkinter non e' utilizzabile da altri thread).

Nota: non serve dichiarare a che lunghezza d'onda corrisponde ogni immagine. Il
metodo lavora sulla struttura dell'immagine, non sui livelli di grigio, quindi
UV, IR e visibile si trattano allo stesso modo.
"""

import os
import queue
import subprocess
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Questo file sta nella cartella del progetto, accanto alle altre due finestre;
# il motore e le finestre secondarie (strumenti, punti, maschera) stanno in
# Allineatore/. Si mette quella cartella sul percorso di Python, e i moduli si
# importano per nome semplice da qualunque cartella si lanci il programma.
QUI = os.path.dirname(os.path.abspath(__file__))
_MOTORE = os.path.join(QUI, 'Allineatore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import align_batch
import finestra_strumenti
import formati
import image_io
import verifica_fissa
from finestra_maschera import FinestraMaschera


FORMATI = [("Immagini", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"), ("Tutti i file", "*.*")]


class CodaStampa:
    """Dirotta print() verso una coda, per mostrarlo nel riquadro di log."""

    def __init__(self, coda, originale):
        self.coda = coda
        self.originale = originale

    def write(self, testo):
        if testo:
            self.coda.put(('log', testo))
        try:
            self.originale.write(testo)
        except Exception:
            pass

    def flush(self):
        try:
            self.originale.flush()
        except Exception:
            pass


PRECISIONI = [
    ("massima (stima a piena risoluzione, lenta)", 1),
    ("precisa (sotto il pixel) - consigliata", 2),
    ("veloce (circa 1 pixel)", 4),
    ("molto veloce (per una prova rapida)", 8),
]

IMPOSTAZIONI_PREDEFINITE = {
    'refine_block': 128,      # lato dei blocchi nella seconda passata
    'min_conf': 0.10,         # confidenza minima perche' un blocco sia usato
    'soglia_sospetto': 3.0,   # oltre questo residuo il risultato e' segnalato
    'tile': 2048,             # lato dei tile in scrittura
    'solo_stima': False,      # True = non scrive le immagini allineate
    'formati': [],            # vuoto = solo il formato dell'originale
    'margine_auto': False,    # True = foglio allargato quanto serve, uguale per il lotto
}


def stima_costi(path_riferimento, fattore, n_immagini, tile, percorsi_immagini=None):
    """
    Stima grossolana della MEMORIA che servira', per scegliere la precisione
    con cognizione di causa.

    Ha una base fisica: durante la stima stanno in memoria due anteprime e tre
    mappe di gradiente (4 byte/px), cioe' ~14 byte per pixel di anteprima con
    sorgenti a 8 bit e ~20 con sorgenti a 16; un file non mappabile in memoria
    (PNG, JPEG, TIFF compresso) viene caricato per intero e il suo peso si
    somma. La scrittura non incide sul picco, perche' una fascia di righe costa
    ~100 MB. Misurato: 1.46 GB previsti e 1.5 misurati a fattore 2 su 314 Mpx;
    10.2 previsti e 11.0 misurati a fattore 1 su tre PNG da 593 Mpx.

    Qui c'era anche una stima del TEMPO, ed e' stata tolta: a precisione
    massima indicava 8-12 minuti per un lotto che ne ha chiesti 18-21. Il
    tempo dipende da cose che prima di partire non si sanno - se il primo
    aggancio prende o serve ripartire dalla posa grossolana, quanto sono lenti
    da decodificare gli ingressi, cosa sta facendo il disco - e un numero
    sbagliato del doppio e' peggio di nessun numero. I tempi misurati per ogni
    impostazione stanno in Documentazione/impostazioni/README.md.

    Returns:
        (testo_descrittivo, gb_stimati) oppure (messaggio, None) se non si puo'.
    """
    try:
        meta = info_immagine(path_riferimento)
    except Exception:
        return "scegli l'immagine di riferimento per vedere una stima", None

    mpx_sorgente = meta.width * meta.height / 1e6
    mpx_anteprima = mpx_sorgente / (fattore ** 2)

    # Byte per pixel: quelli della piu' pesante fra riferimento e immagini.
    byte_px = meta.samples * meta.dtype.itemsize
    for p in (percorsi_immagini or []):
        try:
            m = info_immagine(p)
            byte_px = max(byte_px, m.samples * m.dtype.itemsize)
        except Exception:
            pass

    per_px = 14.0 if byte_px <= 3 else 20.0
    gb = 0.30 + per_px * mpx_anteprima / 1024.0
    if not meta.memmappabile:
        # Un file non mappabile viene caricato per intero: si somma il suo peso.
        gb += mpx_sorgente * meta.samples * meta.dtype.itemsize / 1024.0

    return (f"stima: ~{gb:.1f} GB di memoria "
            f"(anteprime da {mpx_anteprima:.0f} Mpx)"), gb


def info_immagine(path):
    """Metadati dell'immagine (separata per poterla sostituire nei test)."""
    return image_io.info(path)


class FinestraImpostazioni(tk.Toplevel):
    """
    Impostazioni avanzate: le manopole che non servono nell'uso normale ma che
    contano quando si vuole scambiare tempo e memoria con qualita', o provare
    delle scelte senza scrivere gigabyte di immagini.
    """

    def __init__(self, padre, valori):
        super().__init__(padre)
        self.title("Impostazioni avanzate")
        self.resizable(False, False)
        self.risultato = None
        v = dict(valori)

        self.densita = tk.StringVar()
        self.conf = tk.StringVar()
        self.soglia = tk.StringVar(value=str(v['soglia_sospetto']))
        self.tile = tk.StringVar(value=str(v['tile']))
        self.solo_stima = tk.BooleanVar(value=v['solo_stima'])
        self.margine_auto = tk.BooleanVar(value=v.get('margine_auto', False))

        self.minsize(640, 560)
        # Il piede si impacchetta PRIMA del corpo e ancorato in basso: cosi' i
        # pulsanti hanno lo spazio riservato e non possono essere spinti fuori
        # dalla finestra dal contenuto che sta sopra.
        piede = tk.Frame(self, padx=16, pady=12)
        piede.pack(side="bottom", fill="x")

        cornice = tk.Frame(self, padx=16, pady=14)
        cornice.pack(side="top", fill="both", expand=True)
        riga = [0]

        def titolo(testo):
            tk.Label(cornice, text=testo, font=("Segoe UI", 10, "bold")).grid(
                row=riga[0], column=0, columnspan=2, sticky="w", pady=(10, 2))
            riga[0] += 1

        def campo(etichetta, widget, nota=None):
            tk.Label(cornice, text=etichetta).grid(row=riga[0], column=0, sticky="w", padx=(0, 10))
            widget.grid(row=riga[0], column=1, sticky="w")
            riga[0] += 1
            if nota:
                tk.Label(cornice, text=nota, fg="gray", justify="left",
                         wraplength=430).grid(row=riga[0], column=0, columnspan=2,
                                              sticky="w", pady=(0, 6))
                riga[0] += 1

        titolo("Qualita' della correzione locale")
        self.mappa_densita = {"fitta - blocchi 96 px": 96,
                              "media - blocchi 128 px": 128,
                              "larga - blocchi 192 px": 192}
        inv = {val: k for k, val in self.mappa_densita.items()}
        self.densita.set(inv.get(v['refine_block'], list(self.mappa_densita)[1]))
        campo("Densita' del campo:",
              ttk.Combobox(cornice, textvariable=self.densita, state="readonly", width=26,
                           values=list(self.mappa_densita)),
              "Blocchi piu' piccoli seguono deformazioni piu' minute, ma ogni blocco "
              "contiene meno struttura, quindi la singola misura e' meno sicura.")

        titolo("Robustezza")
        self.mappa_conf = {"bassa 0.05 - immagini difficili": 0.05,
                           "normale 0.10": 0.10,
                           "alta 0.20 - solo match sicuri": 0.20}
        inv = {val: k for k, val in self.mappa_conf.items()}
        self.conf.set(inv.get(v['min_conf'], list(self.mappa_conf)[1]))
        campo("Confidenza minima dei blocchi:",
              ttk.Combobox(cornice, textvariable=self.conf, state="readonly", width=26,
                           values=list(self.mappa_conf)),
              "Abbassala se la stima fallisce per mancanza di blocchi affidabili; "
              "alzala se sospetti che stia usando corrispondenze sbagliate.")
        campo("Segnala se il residuo supera (px):",
              tk.Spinbox(cornice, textvariable=self.soglia, from_=0.5, to=50.0,
                         increment=0.5, width=8),
              "Sopra questo valore il risultato viene marcato 'da controllare'. "
              "Non cambia l'allineamento, solo la severita' del giudizio.")

        titolo("Memoria e scrittura")
        campo("Lato dei tile in scrittura (px):",
              ttk.Combobox(cornice, textvariable=self.tile, state="readonly", width=10,
                           values=["1024", "2048", "4096"]),
              "Incide poco sul picco di memoria (un tile da 2048 px costa una "
              "decina di MB): il grosso lo occupa la fase di stima.")
        menu, self.leggi_formati = formati.menu_formati(cornice, v.get('formati'))
        campo("Formati delle immagini salvate:", menu,
              "Se ne possono spuntare piu' d'uno. Senza spunte si salva solo nel "
              "formato dell'originale. PNG e JPEG richiedono l'immagine intera in "
              "memoria (circa 1 GB per banda), il TIFF no.")

        tk.Checkbutton(cornice, variable=self.margine_auto,
                       text="Allarga il foglio se un'immagine esce dal riferimento").grid(
                           row=riga[0], column=0, columnspan=2, sticky="w", pady=(10, 0))
        riga[0] += 1
        tk.Label(cornice, fg="gray", justify="left", wraplength=430,
                 text="Se un'immagine inquadra piu' del riferimento, le parti che escono "
                      "dal suo fotogramma andrebbero tagliate. Con questa opzione si "
                      "aggiunge un bordo nero, uguale per tutto il lotto: le immagini "
                      "allineate restano sovrapponibili fra loro pixel a pixel, ma su un "
                      "foglio piu' grande del riferimento, che vi sta spostato del bordo "
                      "(il valore e' nel riepilogo).").grid(
                          row=riga[0], column=0, columnspan=2, sticky="w")
        riga[0] += 1

        tk.Checkbutton(cornice, variable=self.solo_stima,
                       text="Solo stima e verifica, senza salvare le immagini").grid(
                           row=riga[0], column=0, columnspan=2, sticky="w", pady=(10, 0))
        riga[0] += 1
        tk.Label(cornice, fg="gray", justify="left", wraplength=430,
                 text="Utile per provare delle impostazioni: ottieni i residui misurati "
                      "senza aspettare la scrittura ne' occupare spazio su disco.").grid(
                          row=riga[0], column=0, columnspan=2, sticky="w")
        riga[0] += 1

        tk.Button(piede, text="Ripristina i valori predefiniti",
                  command=self.ripristina).pack(side="left")
        tk.Button(piede, text="Annulla", command=self.destroy).pack(side="right", padx=4)
        tk.Button(piede, text="Applica", command=self.applica,
                  bg="#2196F3", fg="white").pack(side="right")

        self.transient(padre)
        self.grab_set()
        self.bind("<Escape>", lambda e: self.destroy())

    def ripristina(self):
        self.risultato = dict(IMPOSTAZIONI_PREDEFINITE)
        self.destroy()

    def applica(self):
        try:
            soglia = float(str(self.soglia.get()).replace(',', '.'))
        except ValueError:
            messagebox.showwarning("Valore non valido",
                                   "La soglia dev'essere un numero.", parent=self)
            return
        self.risultato = {
            'refine_block': self.mappa_densita[self.densita.get()],
            'min_conf': self.mappa_conf[self.conf.get()],
            'soglia_sospetto': max(0.1, soglia),
            'tile': int(self.tile.get()),
            'solo_stima': bool(self.solo_stima.get()),
            'formati': self.leggi_formati(),
            'margine_auto': bool(self.margine_auto.get()),
        }
        self.destroy()


class Applicazione:

    def __init__(self, root):
        self.root = root
        root.title("Allineatore")

        alt = min(880, max(620, root.winfo_screenheight() - 80))
        lar = min(1040, max(760, root.winfo_screenwidth() - 80))
        root.geometry(f"{lar}x{alt}")
        root.minsize(760, 600)
        root.configure(padx=18, pady=12)

        self.percorso_riferimento = ""
        self.percorsi = []
        self.cartella_output = ""
        self.maschera = None
        # Gli strumenti a mano, immagine per immagine (punti, bordo tracciato):
        # chi non compare qui va in automatico. E i risultati gia' ottenuti su
        # ogni immagine, per confrontare un tentativo con il precedente.
        self.strumenti = {}
        self.storico = {}
        self.voci = {}
        self.impostazioni = dict(IMPOSTAZIONI_PREDEFINITE)
        self.coda = queue.Queue()
        self.in_corso = False

        self._costruisci()
        self.root.after(120, self._svuota_coda)

    # ------------------------------------------------------------------ UI
    def _costruisci(self):
        r = self.root

        tk.Label(r, text="1. Immagine di riferimento",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        riga = tk.Frame(r)
        riga.pack(fill="x", pady=(2, 10))
        tk.Button(riga, text="Scegli...", width=12,
                  command=self.scegli_riferimento).pack(side="left")
        self.lbl_rif = tk.Label(riga, text="nessun file selezionato", fg="gray")
        self.lbl_rif.pack(side="left", padx=8)

        tk.Label(r, text="2. Immagini da allineare",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        riga = tk.Frame(r)
        riga.pack(fill="x", pady=(2, 2))
        tk.Button(riga, text="Aggiungi...", width=12, command=self.aggiungi).pack(side="left")
        tk.Button(riga, text="Rimuovi selezionate", command=self.rimuovi).pack(side="left", padx=4)
        tk.Button(riga, text="Svuota", command=self.svuota).pack(side="left")
        # Punti e bordo tracciato si assegnano a UNA immagine della lista. Erano
        # la sezione a parte delle radiografie; servono a qualunque banda che
        # l'automatico non porta a posto, e le altre non se ne accorgono.
        tk.Button(riga, text="Strumenti a mano...",
                  command=self.apri_strumenti).pack(side="left", padx=(12, 0))
        self.lbl_n = tk.Label(riga, text="0 immagini", fg="gray")
        self.lbl_n.pack(side="right")

        cornice = tk.Frame(r)
        cornice.pack(fill="x", pady=(0, 10))
        self.lista = tk.Listbox(cornice, height=4, selectmode="extended")
        self.lista.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(cornice, command=self.lista.yview)
        sb.pack(side="right", fill="y")
        self.lista.config(yscrollcommand=sb.set)

        tk.Label(r, text="3. Cartella di salvataggio",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        riga = tk.Frame(r)
        riga.pack(fill="x", pady=(2, 10))
        tk.Button(riga, text="Scegli...", width=12, command=self.scegli_output).pack(side="left")
        self.lbl_out = tk.Label(riga, text="nessuna cartella selezionata", fg="gray")
        self.lbl_out.pack(side="left", padx=8)

        tk.Label(r, text="4. Opzioni", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        opz = tk.Frame(r)
        opz.pack(fill="x", pady=(2, 10))

        tk.Label(opz, text="Precisione:").grid(row=0, column=0, sticky="w")
        self.qualita = tk.StringVar(value=PRECISIONI[1][0])
        combo = ttk.Combobox(opz, textvariable=self.qualita, state="readonly", width=42,
                             values=[e[0] for e in PRECISIONI])
        combo.grid(row=0, column=1, sticky="w", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda e: self.aggiorna_stima())
        tk.Button(opz, text="Impostazioni avanzate...",
                  command=self.apri_impostazioni).grid(row=0, column=2, sticky="w", padx=6)

        self.lbl_stima = tk.Label(opz, fg="#1565c0",
                                  text="scegli l'immagine di riferimento per vedere una stima")
        self.lbl_stima.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        self.nonrigida = tk.BooleanVar(value=True)
        tk.Checkbutton(opz, variable=self.nonrigida, text="Correggi anche le deformazioni locali "
                       "(La deformazione non è rigida)").grid(
                           row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.usa_maschera = tk.BooleanVar(value=False)
        tk.Checkbutton(opz, variable=self.usa_maschera, command=self._cambia_maschera,
                       text="Limita la stima a una zona che scelgo io").grid(
                           row=3, column=0, columnspan=2, sticky="w")

        self.btn_maschera = tk.Button(opz, text="Disegna zona...", state="disabled",
                                      command=self.disegna_maschera)
        self.btn_maschera.grid(row=3, column=2, sticky="w", padx=6)
        tk.Label(opz, fg="gray", text="   (serve se lo sfondo è diverso tra le varie immagini, o per "
                 "verificare fino al bordo e usare il bordo tracciato)").grid(row=4, column=0, columnspan=3,
                                                                sticky="w")

        azioni = tk.Frame(r)
        azioni.pack(pady=(6, 8))
        self.btn_avvia = tk.Button(azioni, text="AVVIA ALLINEAMENTO", command=self.avvia,
                                   bg="#127a12", fg="white", font=("Segoe UI", 11, "bold"),
                                   padx=14, pady=6)
        self.btn_avvia.pack(side="left")
        # Un'immagine difficile si fa a tentativi: si ritoccano punti o bordo e
        # si rifa' solo lei, senza rifare il lotto.
        self.btn_rifai = tk.Button(azioni, text="Rifai la selezionata",
                                   command=lambda: self.avvia(solo_selezionata=True),
                                   padx=10, pady=6)
        self.btn_rifai.pack(side="left", padx=(10, 0))

        # Il falso colore e' un programma a parte, ma lo si apre da qui: sempre,
        # anche senza un lotto appena fatto. Dopo un lotto la finestra arriva
        # gia' compilata con riferimento, bande e cartella; altrimenti vuota.
        # E' l'unica sezione a parte rimasta: le radiografie, che ne avevano
        # una loro, passano dal lotto come le altre bande, con gli strumenti a
        # mano dell'immagine.
        sez_fc = tk.LabelFrame(r, text=" Falso colore (IRFC e UVFC) ", fg="#555",
                               padx=10, pady=6)
        sez_fc.pack(fill="x", pady=(0, 8))
        self.btn_falsocolore = tk.Button(sez_fc, text="Componi falso colore...",
                                         command=self.apri_falsocolore)
        self.btn_falsocolore.pack(side="left")
        tk.Label(sez_fc, fg="gray", justify="left", anchor="w",
                 text="Da immagini gia' allineate; dopo un lotto, gia' compilata."
                 ).pack(side="left", padx=(8, 0))
        self.ultimo_riepilogo = None

        self.avanzamento = ttk.Progressbar(r, orient="horizontal", mode="determinate",
                                           maximum=1000)
        self.avanzamento.pack(fill="x")
        self.lbl_stato = tk.Label(r, text="pronto", fg="gray")
        self.lbl_stato.pack(anchor="w", pady=(2, 6))

        # Risultati e log in due schede che si prendono tutto lo spazio rimasto:
        # uno sotto l'altro non ci stavano, e a perdere era sempre il log.
        # Durante l'elaborazione sta davanti il log, alla fine i risultati.
        self.schede = ttk.Notebook(r)
        self.schede.pack(fill="both", expand=True)
        scheda_ris = tk.Frame(self.schede)
        scheda_log = tk.Frame(self.schede)
        self.schede.add(scheda_ris, text=" Risultati ")
        self.schede.add(scheda_log, text=" Log ")
        self.scheda_ris, self.scheda_log = scheda_ris, scheda_log

        # Sotto la tabella, cio' che la mediana non dice per la riga scelta:
        # si impacchetta prima della tabella e in basso, cosi' ha lo spazio
        # riservato anche su uno schermo piccolo.
        piede_ris = tk.Frame(scheda_ris)
        piede_ris.pack(side="bottom", fill="x")
        self.btn_diag = tk.Button(piede_ris, text="Apri la diagnostica", state="disabled",
                                  command=self._apri_diagnostica)
        self.btn_diag.pack(side="right", anchor="n", padx=(8, 0), pady=4)
        # Altezza fissa e barra di scorrimento: gli avvisi di una radiografia
        # sono dieci righe, e in un'etichetta che cresce schiacciavano la
        # tabella a una riga sola.
        self.txt_avvisi = ScrolledText(piede_ris, height=3, wrap="word", fg="#444",
                                       font=("Segoe UI", 9), bg="#f7f7f7", relief="flat")
        self.txt_avvisi.pack(side="left", fill="x", expand=True, pady=4)
        self._scrivi_avvisi("Seleziona una riga per leggere cio' che la mediana non dice: "
                            "finestre oltre 10 px, bordi, TRE sui punti.")

        self.tabella = ttk.Treeview(scheda_ris,
                                    columns=("prima", "dopo", "oltre5", "finestre", "fissa",
                                             "oltre10", "tre", "esito"),
                                    height=4)
        self.tabella.heading("#0", text="immagine")
        self.tabella.heading("prima", text="disallin. prima")
        self.tabella.heading("dopo", text="dopo")
        # La mediana da sola non basta: la quota oltre 5 px dice quanto e'
        # lunga la coda, le finestre misurabili per quanta parte del dipinto
        # vale il numero.
        self.tabella.heading("oltre5", text="oltre 5 px")
        self.tabella.heading("finestre", text="finestre")
        # La griglia fissa accanto alla verifica di sempre, che resta il numero
        # principale: mediana e finestre misurabili sulla griglia piu' fitta, e
        # le finestre oltre 10 px che la mediana nasconde. Il TRE c'e' solo per
        # le immagini a cui sono stati dati dei punti.
        self.tabella.heading("fissa", text="griglia fissa")
        self.tabella.heading("oltre10", text="oltre 10 px")
        self.tabella.heading("tre", text="TRE punti")
        self.tabella.heading("esito", text="esito")
        self.tabella.column("#0", width=190, stretch=False)
        self.tabella.column("prima", width=90, anchor="center", stretch=False)
        self.tabella.column("dopo", width=70, anchor="center", stretch=False)
        self.tabella.column("oltre5", width=65, anchor="center", stretch=False)
        self.tabella.column("finestre", width=60, anchor="center", stretch=False)
        self.tabella.column("fissa", width=125, anchor="center", stretch=False)
        self.tabella.column("oltre10", width=70, anchor="center", stretch=False)
        self.tabella.column("tre", width=70, anchor="center", stretch=False)
        self.tabella.column("esito", width=220)
        self.tabella.pack(fill="both", expand=True)
        self.tabella.bind("<<TreeviewSelect>>", lambda e: self._mostra_avvisi())
        self.tabella.tag_configure("ok", foreground="#127a12")
        self.tabella.tag_configure("sospetto", foreground="#b36b00")
        self.tabella.tag_configure("errore", foreground="#c00000")

        self.log = ScrolledText(scheda_log, height=5, font=("Consolas", 8), bg="#f7f7f7")
        self.log.pack(fill="both", expand=True)

    # ------------------------------------------------------------- comandi
    def scegli_riferimento(self):
        p = filedialog.askopenfilename(title="Immagine di riferimento", filetypes=FORMATI)
        if not p:
            return
        self.percorso_riferimento = p
        try:
            self.lbl_rif.config(text=str(image_io.info(p)), fg="black")
        except Exception as exc:
            self.lbl_rif.config(text=f"{os.path.basename(p)} (non leggibile: {exc})", fg="red")
        self.maschera = None
        self.aggiorna_stima()

    def aggiungi(self):
        percorsi = filedialog.askopenfilenames(title="Immagini da allineare", filetypes=FORMATI)
        for p in percorsi:
            if p not in self.percorsi:
                self.percorsi.append(p)
                self.lista.insert("end", self._testo_lista(p))
        self._aggiorna_conteggio()

    def rimuovi(self):
        for i in sorted(self.lista.curselection(), reverse=True):
            self.lista.delete(i)
            self.strumenti.pop(self.percorsi[i], None)
            del self.percorsi[i]
        self._aggiorna_conteggio()

    def svuota(self):
        self.lista.delete(0, "end")
        self.percorsi = []
        self.strumenti = {}
        self._aggiorna_conteggio()

    def _testo_lista(self, p):
        """Il nome dell'immagine e, se ne ha, i suoi strumenti a mano."""
        st = self.strumenti.get(p)
        return os.path.basename(p) + (f"     [{finestra_strumenti.descrivi(st)}]" if st else "")

    def _selezionata(self):
        """Il percorso dell'unica immagine selezionata nella lista, o None."""
        sel = self.lista.curselection()
        if len(sel) != 1:
            messagebox.showwarning("Scegli un'immagine",
                                   "Seleziona una sola immagine nella lista.")
            return None
        return self.percorsi[sel[0]]

    def apri_strumenti(self):
        """Punti e bordo tracciato per l'immagine selezionata."""
        if not self.percorso_riferimento:
            messagebox.showwarning("Manca il riferimento",
                                   "Scegli prima l'immagine di riferimento.")
            return
        p = self._selezionata()
        if p is None:
            return
        i = self.percorsi.index(p)
        ultimo = (self.storico.get(p) or [None])[-1]
        finestra = finestra_strumenti.FinestraStrumenti(
            self.root, self.percorso_riferimento, p, cartella_output=self.cartella_output,
            strumenti=self.strumenti.get(p), ultimo=ultimo,
            ha_zona=self.usa_maschera.get() and self.maschera is not None)
        self.root.wait_window(finestra)
        if finestra.risultato is None:
            return
        if finestra.risultato:
            self.strumenti[p] = finestra.risultato
        else:
            self.strumenti.pop(p, None)
        self.lista.delete(i)
        self.lista.insert(i, self._testo_lista(p))
        self.lista.selection_set(i)

    def _aggiorna_conteggio(self):
        n = len(self.percorsi)
        self.lbl_n.config(text=f"{n} immagine" if n == 1 else f"{n} immagini")
        self.aggiorna_stima()

    def scegli_output(self):
        p = filedialog.askdirectory(title="Cartella di salvataggio")
        if p:
            self.cartella_output = p
            self.lbl_out.config(text=p, fg="black")

    def fattore_scelto(self):
        """Fattore di riduzione corrispondente alla precisione selezionata."""
        for etichetta, fattore in PRECISIONI:
            if self.qualita.get() == etichetta:
                return fattore
        return 2

    def aggiorna_stima(self):
        """Ricalcola la stima di memoria con le scelte attuali."""
        if not self.percorso_riferimento:
            self.lbl_stima.config(
                text="scegli l'immagine di riferimento per vedere una stima", fg="gray")
            return
        testo, gb = stima_costi(self.percorso_riferimento, self.fattore_scelto(),
                                len(self.percorsi), self.impostazioni['tile'],
                                percorsi_immagini=self.percorsi)
        if self.impostazioni['solo_stima']:
            testo += "  |  solo stima: nessuna immagine verra' salvata"
        # Il colore e' l'unico avviso sensato qui: non conosco la RAM libera in
        # questo momento, ma oltre qualche GB vale la pena fermarsi a pensare.
        colore = "#1565c0" if (gb or 0) < 4 else "#b36b00"
        self.lbl_stima.config(text=testo, fg=colore)

    def apri_impostazioni(self):
        finestra = FinestraImpostazioni(self.root, self.impostazioni)
        self.root.wait_window(finestra)
        if finestra.risultato is not None:
            self.impostazioni = finestra.risultato
            self.aggiorna_stima()

    def _cambia_maschera(self):
        attiva = self.usa_maschera.get()
        self.btn_maschera.config(state="normal" if attiva else "disabled")
        if not attiva:
            self.maschera = None

    def disegna_maschera(self):
        if not self.percorso_riferimento:
            messagebox.showwarning("Manca il riferimento",
                                   "Scegli prima l'immagine di riferimento: la zona si disegna su quella.")
            return
        try:
            finestra = FinestraMaschera(self.root, self.percorso_riferimento)
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile aprire l'anteprima:\n{exc}")
            return
        self.root.wait_window(finestra)
        self.maschera = finestra.maschera
        self.descrizione_maschera = finestra.descrizione
        if self.maschera is not None:
            self.btn_maschera.config(text=f"Zona: {self.maschera.mean() / 2.55:.0f}% (ridisegna)")

    # ----------------------------------------------------------- esecuzione
    def avvia(self, solo_selezionata=False):
        if self.in_corso:
            return
        if not self.percorso_riferimento:
            messagebox.showwarning("Manca il riferimento", "Scegli l'immagine di riferimento.")
            return
        if not self.percorsi:
            messagebox.showwarning("Nessuna immagine", "Aggiungi almeno un'immagine da allineare.")
            return
        if not self.cartella_output:
            messagebox.showwarning("Manca la cartella", "Scegli dove salvare i risultati.")
            return
        rif = os.path.abspath(self.percorso_riferimento)
        if any(os.path.abspath(p) == rif for p in self.percorsi):
            messagebox.showwarning(
                "L'immagine di riferimento è fra le immagini da allineare.",
                )
            return
        if self.usa_maschera.get() and self.maschera is None:
            messagebox.showwarning("Zona non disegnata",
                                   "Maschera selezionata, ma non è stata disegnata."
                                   )
            return
        immagini = list(self.percorsi)
        if solo_selezionata:
            p = self._selezionata()
            if p is None:
                return
            immagini = [p]
        ha_zona = self.usa_maschera.get() and self.maschera is not None
        senza_zona = [os.path.basename(p) for p in immagini
                      if (self.strumenti.get(p) or {}).get('fascia_di_bordo') and not ha_zona]
        if senza_zona:
            messagebox.showwarning(
                "Manca la zona",
                "Per allineare il bordo servono la zona disegnata sul riferimento e il bordo "
                "tracciato sull'immagine. Manca la zona per: " + ", ".join(senza_zona))
            return

        # Le variabili di tkinter (StringVar, BooleanVar) possono essere lette
        # solo dal thread principale: leggerle dentro il thread di lavoro fa
        # fallire tutto con "main thread is not in main loop". Si raccoglie
        # quindi qui tutto cio' che serve, in normali oggetti Python.
        impostazioni = {
            'riferimento': self.percorso_riferimento,
            'immagini': immagini,
            'strumenti': {p: self.strumenti[p] for p in immagini if p in self.strumenti},
            'solo_selezionata': solo_selezionata,
            'output': self.cartella_output,
            'factor': self.fattore_scelto(),
            'nonrigida': bool(self.nonrigida.get()),
            'maschera': self.maschera if self.usa_maschera.get() else None,
            'maschera_info': (getattr(self, 'descrizione_maschera', None)
                              if self.usa_maschera.get() else None),
            'avanzate': dict(self.impostazioni),
        }

        self.in_corso = True
        self.btn_avvia.config(state="disabled", bg="gray", text="ELABORAZIONE IN CORSO...")
        self.btn_rifai.config(state="disabled")
        # Rifacendo una sola immagine le righe restano: il tentativo nuovo si
        # aggiunge sotto, e si confronta con i precedenti.
        if not solo_selezionata:
            for riga in self.tabella.get_children():
                self.tabella.delete(riga)
            self.voci = {}
            self._scrivi_avvisi("")
        self.log.delete("1.0", "end")
        self.schede.select(self.scheda_log)
        self.avanzamento['value'] = 0
        threading.Thread(target=self._lavora, args=(impostazioni,), daemon=True).start()

    def _lavora(self, imp):
        originale = sys.stdout
        sys.stdout = CodaStampa(self.coda, originale)
        try:
            nomi = [os.path.basename(p) for p in imp['immagini']]

            def progresso(i, n, testo, frazione):
                self.coda.put(('avanzamento',
                               (1000.0 * (i + max(0.0, min(1.0, frazione))) / max(n, 1),
                                f"[{i + 1}/{n}] {nomi[i]} - {testo}")))

            av = imp['avanzate']
            riepilogo = align_batch.allinea_lotto(
                imp['riferimento'], imp['immagini'], imp['output'],
                factor=imp['factor'], nonrigid=imp['nonrigida'],
                mask=imp['maschera'], mask_info=imp['maschera_info'],
                tile_size=av['tile'], min_conf=av['min_conf'],
                refine_block=av['refine_block'], formati=av.get('formati'),
                soglia_sospetto=av['soglia_sospetto'],
                scrivi_immagini=not av['solo_stima'],
                margine='auto' if av.get('margine_auto') else 0,
                strumenti=imp['strumenti'],
                progress=progresso, verbose=False)
            riepilogo['solo_selezionata'] = imp['solo_selezionata']
            self.coda.put(('fine', riepilogo))
        except Exception as exc:
            self.coda.put(('errore', f"{exc}\n\n{traceback.format_exc()}"))
        finally:
            sys.stdout = originale

    def _svuota_coda(self):
        try:
            while True:
                tipo, dato = self.coda.get_nowait()
                if tipo == 'log':
                    self.log.insert("end", dato)
                    self.log.see("end")
                elif tipo == 'avanzamento':
                    valore, testo = dato
                    self.avanzamento['value'] = valore
                    self.lbl_stato.config(text=testo, fg="#1565c0")
                elif tipo == 'fine':
                    self._concludi(dato)
                elif tipo == 'errore':
                    self._fallisci(dato)
        except queue.Empty:
            pass
        self.root.after(120, self._svuota_coda)

    def apri_falsocolore(self):
        """Apre la finestra del falso colore, gia' compilata se c'e' un lotto.

        Si lancia come processo separato invece di creare una seconda finestra
        qui dentro: due gerarchie tkinter nello stesso processo si ostacolano, e
        un blocco della composizione fermerebbe anche l'allineatore.
        """
        # Il falso colore e' un programma a parte, con la finestra accanto a
        # questa.
        comando = [sys.executable, os.path.join(QUI, 'interfaccia_falsocolore.py')]
        r = self.ultimo_riepilogo
        if not r:
            try:
                subprocess.Popen(comando)
            except Exception as exc:
                messagebox.showerror("Impossibile aprire", str(exc))
            return
        ir, uv = [], []
        for v in r['immagini']:
            out = v.get('output')
            if not out or v['esito'] == 'errore':
                continue
            n = os.path.basename(out).lower()
            if 'uvr' in n:
                uv.append(out)
            elif 'uvf' in n:
                continue          # la fluorescenza non entra in un falso colore
            elif any(k in n for k in ('nir', 'swir', 'irr', 'ir_')):
                ir.append(out)
        # Con un margine le bande allineate sono piu' grandi del riferimento: il
        # visibile da comporre con loro e' la sua copia con lo stesso margine.
        vis = r.get('riferimento_con_margine') or r['riferimento']
        comando += ['--vis', vis, '--out-dir',
                    os.path.join(r['cartella_output'], 'falsocolore')]
        if ir:
            comando += ['--ir'] + ir
        if uv:
            comando += ['--uv'] + uv
        try:
            subprocess.Popen(comando)
        except Exception as exc:
            messagebox.showerror("Impossibile aprire", str(exc))

    def _concludi(self, riepilogo):
        self.in_corso = False
        self.btn_avvia.config(state="normal", bg="#127a12", text="AVVIA ALLINEAMENTO")
        self.btn_rifai.config(state="normal")
        self.avanzamento['value'] = 1000
        solo = bool(riepilogo.get('solo_selezionata'))
        # Rifatta una sola immagine, nel riepilogo che serve al falso colore si
        # sostituisce la sua voce e le altre restano.
        if solo and self.ultimo_riepilogo:
            nuove = {v['file']: v for v in riepilogo['immagini']}
            self.ultimo_riepilogo['immagini'] = (
                [nuove.pop(v['file'], v) for v in self.ultimo_riepilogo['immagini']]
                + list(nuove.values()))
        else:
            self.ultimo_riepilogo = riepilogo
        self.schede.select(self.scheda_ris)

        def px(valore, cifre=2):
            return f"{valore:.{cifre}f} px" if valore else "-"

        n_ok = n_sosp = n_err = 0
        ultima_riga = None
        for v in riepilogo['immagini']:
            precedenti = self.storico.setdefault(v['file'], [])
            nome = v['nome'] + (f"  ({len(precedenti) + 1}°)" if precedenti else "")
            if v['esito'] == 'errore':
                n_err += 1
                riga = self.tabella.insert(
                    "", "end", text=nome, tags=("errore",),
                    values=("-", "-", "-", "-", "-", "-", "-",
                            "FALLITA: " + v.get('motivo', '')[:80]))
                self.voci[riga] = v
                ultima_riga = riga
                continue
            n_ok += v['esito'] == 'ok'
            n_sosp += v['esito'] == 'sospetto'
            q5 = v.get('quota_sopra_5px')
            vf = v.get('verifica_fissa') or {}
            tre = v.get('tre_punti') or {}
            fissa = (f"{vf['residuo_mediano_px']:.2f} px  {vf.get('n_affidabili', 0)}/"
                     f"{vf.get('n_campioni', 0)}" if vf.get('residuo_mediano_px') is not None
                     else ("non misurabile" if vf else "-"))
            riga = self.tabella.insert(
                "", "end", text=nome,
                tags=("ok" if v['esito'] == 'ok' else "sospetto",),
                values=(px(v.get('residuo_prima_px'), 1), px(v.get('residuo_dopo_px')),
                        f"{100 * q5:.0f}%" if q5 is not None else "-",
                        (f"{v['campioni_affidabili']}/{v['campioni_totali']}"
                         if v.get('campioni_totali') else "-"),
                        fissa, v.get('n_oltre_10px', '-') if vf else "-",
                        (f"{tre['mediana_px']:.1f} px" + ("*" if tre.get('ottimistico') else ""))
                        if tre else "-",
                        "allineata" if v['esito'] == 'ok'
                        else "DA CONTROLLARE: " + v.get('motivo', '')))
            v['_precedente'] = precedenti[-1] if precedenti else None
            precedenti.append(v)
            self.voci[riga] = v
            ultima_riga = riga
        if ultima_riga is not None and (solo or len(riepilogo['immagini']) == 1):
            self.tabella.selection_set(ultima_riga)
            self.tabella.see(ultima_riga)

        self.lbl_stato.config(
            text=f"finito in {riepilogo['durata_min']:.1f} min - "
                 f"{n_ok} allineate, {n_sosp} da controllare, {n_err} fallite",
            fg="#127a12" if not (n_sosp or n_err) else "#b36b00")
        if solo:
            return

        messaggio = (f"{n_ok} immagini allineate su {len(riepilogo['immagini'])}.\n\n"
                     f"Risultati in:\n{riepilogo['cartella_output']}")
        if n_sosp:
            messaggio += (f"\n\n{n_sosp} da controllare: guarda le immagini di confronto in "
                          f"diagnostica\\<nome>\\overlay_falsocolore.png prima di usarle.")
        if n_err:
            messaggio += f"\n\n{n_err} non sono riuscite: vedi il dettaglio nella tabella."
        messaggio += ("\n\nSeleziona una riga dei risultati per leggere cio' che la mediana "
                      "non dice: finestre oltre 10 px, bordi, TRE.")
        messagebox.showinfo("Elaborazione completata", messaggio)

    def _mostra_avvisi(self):
        """Sotto la tabella, gli avvisi della riga selezionata."""
        sel = self.tabella.selection()
        v = self.voci.get(sel[0]) if sel else None
        if not v:
            self._scrivi_avvisi("")
            self.btn_diag.config(state="disabled")
            return
        if v['esito'] == 'errore':
            testo = v.get('motivo', '')
        else:
            testo = " ".join(t for t in (verifica_fissa.confronto(v.get('_precedente'), v),
                                         verifica_fissa.avvisi(v)) if t)
            if v.get('margine_px'):
                testo += (f" Immagine scritta con {v['margine_px']} px di margine per lato: "
                          f"il riferimento vi sta spostato di {v['margine_px']} px in x e in y.")
        self._scrivi_avvisi(testo)
        d = v.get('diagnostica')
        self.btn_diag.config(state="normal" if d and os.path.isdir(d) else "disabled")

    def _scrivi_avvisi(self, testo):
        self.txt_avvisi.config(state="normal")
        self.txt_avvisi.delete("1.0", "end")
        self.txt_avvisi.insert("end", testo)
        self.txt_avvisi.config(state="disabled")

    def _apri_diagnostica(self):
        sel = self.tabella.selection()
        d = (self.voci.get(sel[0]) or {}).get('diagnostica') if sel else None
        if d and os.path.isdir(d):
            try:
                os.startfile(d)
            except Exception as exc:
                messagebox.showerror("Impossibile aprire", str(exc))

    def _fallisci(self, testo):
        self.in_corso = False
        self.btn_avvia.config(state="normal", bg="#127a12", text="AVVIA ALLINEAMENTO")
        self.btn_rifai.config(state="normal")
        self.lbl_stato.config(text="elaborazione interrotta da un errore", fg="#c00000")
        self.log.insert("end", "\n" + testo)
        self.log.see("end")
        self.schede.select(self.scheda_log)
        messagebox.showerror("Errore", testo.split("\n")[0])


def main():
    root = tk.Tk()
    Applicazione(root)
    root.mainloop()


if __name__ == '__main__':
    main()
