"""
Interfaccia grafica per comporre i falsi colori IRFC e UVFC.

    python interfaccia_falsocolore.py

Prende immagini GIA' ALLINEATE (l'uscita di interfaccia.py o di
Allineatore\\align_batch.py) e compone le due rappresentazioni standard
dell'imaging per i beni culturali:

    IRFC   IR -> R,  R(VIS) -> G,  G(VIS) -> B     (scarta il blu del visibile)
    UVFC   G(VIS) -> R,  B(VIS) -> G,  UV -> B     (scarta il rosso del visibile)

Il motore e' FalsoColore\\falso_colore.py: qui c'e' solo la scelta dei file e
l'avvio, per non doverli scrivere a mano sulla riga di comando.

Come per l'interfaccia dell'allineatore, il lavoro gira in un thread separato e
tutto cio' che tocca la finestra passa da una coda letta dal thread principale,
perche' tkinter non e' utilizzabile da altri thread.
"""

import os
import queue
import sys
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Questo file sta nella cartella del progetto: il motore e' in FalsoColore/, e
# le uscite vanno in risultati/, comune ai tre programmi.
RADICE = os.path.dirname(os.path.abspath(__file__))
_MOTORE = os.path.join(RADICE, 'FalsoColore')
if _MOTORE not in sys.path:
    sys.path.insert(0, _MOTORE)

import falso_colore as fc
import formati


FORMATI = [("Immagini", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
           ("Tutti i file", "*.*")]


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


class ElencoBande(ttk.LabelFrame):
    """Elenco di immagini da comporre: se ne possono aggiungere quante si vuole.

    Serve perche' le bande infrarosse sono tipicamente piu' d'una (NIR850,
    NIR950, SWIR...) e comporle in un colpo solo evita di rilanciare il
    programma per ognuna. Il visibile viene letto una volta sola per tutte.
    """

    def __init__(self, padre, titolo, suggerimento):
        super().__init__(padre, text=titolo, padding=8)
        self.percorsi = []
        ttk.Label(self, text=suggerimento, foreground='#666').grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        self.lista = tk.Listbox(self, height=3, activestyle='none')
        self.lista.grid(row=1, column=0, sticky='we')
        pulsanti = ttk.Frame(self)
        pulsanti.grid(row=1, column=1, sticky='ns', padx=(8, 0))
        ttk.Button(pulsanti, text="Aggiungi...", command=self._aggiungi).pack(fill='x')
        ttk.Button(pulsanti, text="Togli", command=self._togli).pack(fill='x', pady=(4, 0))
        self.columnconfigure(0, weight=1)

    def _aggiungi(self):
        for p in filedialog.askopenfilenames(title="Scegli le immagini",
                                             filetypes=FORMATI):
            p = os.path.normpath(p)
            if p not in self.percorsi:
                self.percorsi.append(p)
                self.lista.insert('end', os.path.basename(p))

    def _togli(self):
        for i in reversed(self.lista.curselection()):
            self.lista.delete(i)
            del self.percorsi[i]

    def get(self):
        return list(self.percorsi)


class SceltaFile(ttk.Frame):
    """Etichetta, casella con il percorso, pulsante Sfoglia. Nient'altro."""

    def __init__(self, padre, etichetta, suggerimento, cartella=False):
        super().__init__(padre)
        self.cartella = cartella
        self.valore = tk.StringVar()
        ttk.Label(self, text=etichetta, width=22).grid(row=0, column=0, sticky='w')
        ttk.Entry(self, textvariable=self.valore, width=62).grid(
            row=0, column=1, sticky='we', padx=(0, 6))
        ttk.Button(self, text="Sfoglia...", command=self._scegli).grid(row=0, column=2)
        ttk.Label(self, text=suggerimento, foreground='#666').grid(
            row=1, column=1, sticky='w', pady=(0, 8))
        self.columnconfigure(1, weight=1)

    def _scegli(self):
        if self.cartella:
            p = filedialog.askdirectory(title="Scegli la cartella")
        else:
            p = filedialog.askopenfilename(title="Scegli l'immagine", filetypes=FORMATI)
        if p:
            self.valore.set(os.path.normpath(p))

    def get(self):
        return self.valore.get().strip()


class Applicazione(tk.Tk):

    def __init__(self, iniziali=None):
        self.iniziali = iniziali or {}
        super().__init__()
        self.title("Falso colore - IRFC e UVFC")
        self.minsize(820, 640)
        self.coda = queue.Queue()
        self.al_lavoro = False
        self.prodotti = []
        self._costruisci()
        self.after(80, self._svuota_coda)

    # ------------------------------------------------------------- interfaccia
    def _costruisci(self):
        cornice = ttk.Frame(self, padding=12)
        cornice.pack(fill='both', expand=True)

        ttk.Label(cornice, text="Falso colore da immagini allineate",
                  font=('Segoe UI', 13, 'bold')).pack(anchor='w')
        ttk.Label(cornice, foreground='#666', text=(
            "Le immagini devono essere gia' allineate e avere le stesse "
            "dimensioni del visibile.")).pack(anchor='w', pady=(0, 10))

        self.vis = SceltaFile(cornice, "Visibile (VIS)",
                              "la ripresa a colori: e' il riferimento")
        self.vis.pack(fill='x')
        self.ir = ElencoBande(
            cornice, "Infrarosso  ->  un IRFC per ciascuna",
            "gia' allineate. Si possono aggiungere NIR850, NIR950, SWIR... "
            "Vuoto = nessun IRFC.")
        self.ir.pack(fill='x', pady=(0, 8))
        self.uv = ElencoBande(
            cornice, "Ultravioletto riflesso  ->  un UVFC per ciascuna",
            "gia' allineate. Serve l'UV RIFLESSO (UVR), non la fluorescenza "
            "(UVF). Vuoto = nessun UVFC.")
        self.uv.pack(fill='x', pady=(0, 8))
        self.dest = SceltaFile(cornice, "Cartella di uscita",
                               "vi finiscono i TIFF, le anteprime PNG e un JSON "
                               "con i parametri", cartella=True)
        self.dest.valore.set(os.path.join(RADICE, 'risultati', 'falsocolore'))
        self.dest.pack(fill='x')

        opzioni = ttk.LabelFrame(cornice, text="Opzioni", padding=10)
        opzioni.pack(fill='x', pady=(4, 10))

        self.vuole_anteprima = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opzioni, variable=self.vuole_anteprima,
            text="Salva anche un PNG ridotto (2000 px) accanto a ogni falso colore"
        ).grid(row=0, column=0, columnspan=3, sticky='w')

        self.forza = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opzioni, variable=self.forza,
            text="Componi anche se le immagini non sembrano allineate"
        ).grid(row=1, column=0, columnspan=3, sticky='w', pady=(6, 0))
        ttk.Label(opzioni, text="Formati:").grid(row=3, column=0, sticky='w', pady=(8, 0))
        menu, self.leggi_formati = formati.menu_formati(opzioni, [], testo_vuoto="come il visibile")
        menu.grid(row=3, column=1, sticky='w', padx=6, pady=(8, 0))
        ttk.Label(opzioni, foreground='#666', wraplength=740, justify='left', text=(
            "Prima di comporre, il programma misura se la banda e' davvero "
            "allineata al visibile. Le dimensioni uguali non bastano: le riprese "
            "originali hanno le stesse dimensioni di quelle allineate, e un falso "
            "colore fatto con le originali sembra giusto a colpo d'occhio ma ha "
            "frange ovunque.")
        ).grid(row=2, column=0, columnspan=3, sticky='w', pady=(2, 0))
        opzioni.columnconfigure(2, weight=1)

        barra = ttk.Frame(cornice)
        barra.pack(fill='x')
        self.avvia = ttk.Button(barra, text="Componi", command=self._avvia)
        self.avvia.pack(side='left')
        self.apri = ttk.Button(barra, text="Apri la cartella", state='disabled',
                               command=self._apri_cartella)
        self.apri.pack(side='left', padx=6)
        self.stato = ttk.Label(barra, text="pronto", foreground='#666')
        self.stato.pack(side='left', padx=12)

        self.log = ScrolledText(cornice, height=16, wrap='word',
                                font=('Consolas', 9))
        self.log.pack(fill='both', expand=True, pady=(10, 0))

        # Valori arrivati dall'allineatore: riferimento, bande appena allineate
        # e cartella. Restano modificabili, sono solo un punto di partenza.
        if self.iniziali.get('vis'):
            self.vis.valore.set(self.iniziali['vis'])
        if self.iniziali.get('out_dir'):
            self.dest.valore.set(self.iniziali['out_dir'])
        for elenco, chiave in ((self.ir, 'ir'), (self.uv, 'uv')):
            for p in self.iniziali.get(chiave) or []:
                elenco.percorsi.append(os.path.normpath(p))
                elenco.lista.insert('end', os.path.basename(p))
        if self.iniziali:
            self.log.insert('end', "Valori portati dall'allineamento: controllali e "
                                   "aggiungi o togli quello che serve.\n")

    # ------------------------------------------------------------------ avvio
    def _avvia(self):
        if self.al_lavoro:
            return
        # Le variabili tkinter si leggono SOLO qui, nel thread principale: il
        # thread di lavoro riceve valori gia' estratti.
        vis, ir, uv = self.vis.get(), self.ir.get(), self.uv.get()
        dest = self.dest.get()
        if not vis or not os.path.exists(vis):
            messagebox.showerror("Manca il visibile",
                                 "Scegli l'immagine nel visibile.")
            return
        if not ir and not uv:
            messagebox.showerror(
                "Manca una banda",
                "Aggiungi almeno un'infrarossa (per l'IRFC) o un'ultravioletta "
                "riflessa (per l'UVFC).")
            return
        for p in ir + uv:
            if not os.path.exists(p):
                messagebox.showerror("File assente",
                                     f"Questa immagine non esiste:\n{p}")
                return
        if not dest:
            messagebox.showerror("Manca la cartella", "Scegli dove salvare.")
            return

        impostazioni = {'vis': vis, 'ir': ir, 'uv': uv, 'dest': dest,
                        'canale': 'media',
                        'forza': bool(self.forza.get()),
                        'anteprima': 2000 if self.vuole_anteprima.get() else 0,
                        'formati': self.leggi_formati()}
        self.al_lavoro = True
        self.prodotti = []
        self.avvia.configure(state='disabled')
        self.apri.configure(state='disabled')
        self.stato.configure(text="in corso...", foreground='#a60')
        self.log.delete('1.0', 'end')
        threading.Thread(target=self._lavora, args=(impostazioni,),
                         daemon=True).start()

    def _lavora(self, imp):
        originale = sys.stdout
        sys.stdout = CodaStampa(self.coda, originale)
        try:
            os.makedirs(imp['dest'], exist_ok=True)
            momento = datetime.now()
            # Tutte le bande si compongono nello stesso ciclo di righe, cosi' il
            # visibile viene letto una volta sola.
            lavori = [(tipo, b, fc._nome(imp['dest'], tipo, b, momento))
                      for tipo, elenco in (('irfc', imp['ir']), ('uvfc', imp['uv']))
                      for b in elenco]
            fatti = fc.componi_molte(
                imp['vis'], lavori, canale=imp['canale'],
                anteprima_px=imp['anteprima'], forza=imp['forza'])
            fc.altri_formati([l[2] for l in lavori], imp['formati'],
                             originale=imp['vis'], report=fatti)
            percorso = os.path.join(
                imp['dest'],
                f"falso_colore ({momento.strftime('%Y-%m-%d %H-%M')}).json")
            import json
            with open(percorso, 'w', encoding='utf-8') as fh:
                json.dump(fatti, fh, indent=2, ensure_ascii=False)
            print(f"\nparametri salvati: {percorso}")
            self.coda.put(('fine', (imp['dest'], fatti)))
        except Exception as exc:
            self.coda.put(('errore', f"{type(exc).__name__}: {exc}\n\n"
                                     f"{traceback.format_exc()}"))
        finally:
            sys.stdout = originale

    # ------------------------------------------------------------------- coda
    def _svuota_coda(self):
        try:
            while True:
                genere, dato = self.coda.get_nowait()
                if genere == 'log':
                    # Il motore usa \r per l'avanzamento: a video si riscrive
                    # l'ultima riga invece di accumularne migliaia.
                    if dato.startswith('\r') or (dato and '\r' in dato):
                        ultima = dato.split('\r')[-1]
                        self.log.delete('end-2l linestart', 'end-1c')
                        self.log.insert('end', '\n' + ultima)
                    else:
                        self.log.insert('end', dato)
                    self.log.see('end')
                elif genere == 'fine':
                    cartella, fatti = dato
                    self.al_lavoro = False
                    self.ultima_cartella = cartella
                    self.avvia.configure(state='normal')
                    self.apri.configure(state='normal')
                    self.stato.configure(
                        text=f"fatto: {len(fatti)} immagine/i", foreground='#080')
                elif genere == 'errore':
                    self.al_lavoro = False
                    self.avvia.configure(state='normal')
                    self.stato.configure(text="errore", foreground='#a00')
                    self.log.insert('end', '\n' + dato)
                    self.log.see('end')
                    messagebox.showerror("Errore", dato.split('\n\n')[0])
        except queue.Empty:
            pass
        self.after(80, self._svuota_coda)

    def _apri_cartella(self):
        cartella = getattr(self, 'ultima_cartella', None)
        if cartella and os.path.isdir(cartella):
            os.startfile(cartella)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Finestra per comporre IRFC e UVFC. Gli argomenti servono solo "
                    "a precompilare i campi (li usa l'allineatore).")
    ap.add_argument('--vis')
    ap.add_argument('--ir', nargs='+')
    ap.add_argument('--uv', nargs='+')
    ap.add_argument('--out-dir')
    a = ap.parse_args()
    Applicazione({'vis': a.vis, 'ir': a.ir, 'uv': a.uv, 'out_dir': a.out_dir}).mainloop()
