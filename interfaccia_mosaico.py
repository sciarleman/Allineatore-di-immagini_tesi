"""
Interfaccia grafica per il mosaico di tasselli.

    python interfaccia_mosaico.py

E' un programma a parte, come quello del falso colore: il motore sta in
Mosaico\\mosaico.py, e qui c'e' solo la scelta dei tasselli, della griglia e
del percorso, piu' lo SCHEMA con le miniature.

Lo schema e' la parte che conta. Un percorso sbagliato (serpentina invece di
righe, partenza dall'angolo sbagliato) non da' errori: da' un mosaico con i
tasselli nelle caselle sbagliate, e il programma prova comunque ad agganciarli.
Mettendo le miniature vere nelle caselle, PRIMA di avviare, si vede subito se
i tasselli vicini continuano l'uno nell'altro.

Come nelle altre finestre, il lavoro gira in un thread separato e tutto cio'
che tocca la finestra passa da una coda letta dal thread principale, perche'
tkinter non e' utilizzabile da altri thread.
"""

import os
import queue
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from PIL import Image, ImageTk

# Questo file sta nella cartella del progetto: il motore del mosaico e' in
# Mosaico/, quello di lettura nella cartella dell'allineatore, e le uscite
# vanno in risultati/, comune ai tre programmi.
RADICE = os.path.dirname(os.path.abspath(__file__))
for _motore in (os.path.join(RADICE, 'Allineatore'), os.path.join(RADICE, 'Mosaico')):
    if _motore not in sys.path:
        sys.path.insert(0, _motore)

import image_io
import mosaico
import mosaico_omografia

# Due modi di riprendere a tasselli, e due motori. Da carrello la camera trasla
# parallela al dipinto: mosaico.py, griglia e percorso, livelli per Photoshop.
# Da cavalletto la camera ruota: mosaico_omografia.py, che i vicini li trova da
# solo, mette un'omografia per tassello e scrive l'immagine totale.
ETICHETTE_RIPRESA = {
    'carrello': "Da carrello: griglia regolare, livelli",
    'cavalletto': "Da cavalletto: la camera ruota, immagine totale",
}
ETICHETTE_PERCORSO = {
    'serpentina_righe': "Serpentina per righe",
    'serpentina_colonne': "Serpentina per colonne",
    'righe': "Per righe, sempre nello stesso verso",
    'colonne': "Per colonne, sempre nello stesso verso",
}
ETICHETTE_PARTENZA = {k: v[0].upper() + v[1:] for k, v in mosaico.PARTENZE.items()}
ETICHETTE_MODELLO = {
    'auto': "Automatico (scelto misurando)",
    'traslazione': "Traslazione",
    'rigida': "Rigida (traslazione + rotazione)",
    'similarita': "Similarita' (+ scala)",
    'affine': "Affine",
}
PRECISIONI = {"Piena risoluzione (lenta)": 1, "Normale": 2, "Rapida": 4}


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


def _scelta(padre, etichette, valore):
    """Menu a tendina che mostra un'etichetta leggibile e restituisce la chiave."""
    var = tk.StringVar(value=etichette[valore])
    menu = ttk.Combobox(padre, textvariable=var, values=list(etichette.values()),
                        state='readonly', width=34)
    inverso = {v: k for k, v in etichette.items()}
    return menu, (lambda: inverso[var.get()]), var


class Applicazione(tk.Tk):

    def __init__(self, iniziali=None):
        super().__init__()
        self.title("Mosaico di tasselli")
        self.minsize(900, 760)
        self.coda = queue.Queue()
        self.al_lavoro = False
        self.immagini = []
        self.miniature = {}          # percorso -> immagine PIL piccola
        self._foto = []              # riferimenti alle PhotoImage, se no tk le perde
        self.ultimo_report = None
        self._costruisci()
        iniziali = iniziali or {}
        if iniziali.get('cartella'):
            self._carica_cartella(iniziali['cartella'])
        self.after(80, self._svuota_coda)

    # ------------------------------------------------------------- interfaccia
    def _costruisci(self):
        cornice = ttk.Frame(self, padding=12)
        cornice.pack(fill='both', expand=True)
        ttk.Label(cornice, text="Mosaico di tasselli",
                  font=('Segoe UI', 13, 'bold')).pack(anchor='w')
        ttk.Label(cornice, foreground='#666', wraplength=860, justify='left', text=(
            "Da carrello: i tasselli vengono posizionati (al piu' con una trasformazione "
            "affine) e scritti come livelli con la trasparenza, da aprire in Photoshop. "
            "Da cavalletto: un'omografia per tassello, la distorsione dell'obiettivo, e "
            "un'immagine totale gia' composta. In nessuno dei due casi si tocca il colore.")
        ).pack(anchor='w', pady=(0, 10))

        alto = ttk.Frame(cornice)
        alto.pack(fill='x')
        sinistra = ttk.Frame(alto)
        sinistra.pack(side='left', fill='y', anchor='n')
        colonna_destra = ttk.Frame(alto)
        colonna_destra.pack(side='left', fill='both', expand=True, padx=(12, 0))
        destra = ttk.LabelFrame(colonna_destra, text="Schema del percorso", padding=6)
        destra.pack(side='top', fill='both', expand=True)

        # --- tasselli
        riga = ttk.Frame(sinistra)
        riga.pack(fill='x')
        ttk.Label(riga, text="Cartella dei tasselli", width=20).pack(side='left')
        self.cartella = tk.StringVar()
        ttk.Entry(riga, textvariable=self.cartella, width=34).pack(side='left', padx=(0, 6))
        ttk.Button(riga, text="Sfoglia...", command=self._scegli_cartella).pack(side='left')
        self.conteggio = ttk.Label(sinistra, foreground='#666', wraplength=400,
                                   justify='left', text=(
                                       "i file vengono presi in ordine naturale dei nomi "
                                       "(2 prima di 10): deve essere l'ordine di ripresa"))
        self.conteggio.pack(anchor='w', pady=(2, 10))

        # --- tipo di ripresa
        riga = ttk.Frame(sinistra)
        riga.pack(fill='x', pady=(0, 8))
        ttk.Label(riga, text="Ripresa", width=20).pack(side='left')
        menu, self.ripresa, v = _scelta(riga, ETICHETTE_RIPRESA, 'carrello')
        menu.configure(width=44)
        menu.pack(side='left')
        menu.bind('<<ComboboxSelected>>', lambda e: self._ridisegna())

        # --- griglia e percorso
        g = ttk.LabelFrame(sinistra, text="Griglia e percorso di ripresa (solo da carrello)",
                           padding=8)
        g.pack(fill='x')
        ttk.Label(g, text="Righe").grid(row=0, column=0, sticky='w')
        self.righe = tk.IntVar(value=3)
        ttk.Spinbox(g, from_=1, to=99, width=5, textvariable=self.righe,
                    command=self._ridisegna).grid(row=0, column=1, sticky='w', padx=(4, 16))
        ttk.Label(g, text="Colonne").grid(row=0, column=2, sticky='w')
        self.colonne = tk.IntVar(value=4)
        ttk.Spinbox(g, from_=1, to=99, width=5, textvariable=self.colonne,
                    command=self._ridisegna).grid(row=0, column=3, sticky='w', padx=(4, 0))
        ttk.Label(g, text="Percorso").grid(row=1, column=0, sticky='w', pady=(8, 0))
        menu, self.percorso, v = _scelta(g, ETICHETTE_PERCORSO, 'serpentina_righe')
        menu.grid(row=1, column=1, columnspan=3, sticky='w', pady=(8, 0))
        menu.bind('<<ComboboxSelected>>', lambda e: self._ridisegna())
        ttk.Label(g, text="Primo scatto").grid(row=2, column=0, sticky='w', pady=(4, 0))
        menu, self.partenza, v = _scelta(g, ETICHETTE_PARTENZA, 'alto_sinistra')
        menu.grid(row=2, column=1, columnspan=3, sticky='w', pady=(4, 0))
        menu.bind('<<ComboboxSelected>>', lambda e: self._ridisegna())
        ttk.Label(g, foreground='#666', wraplength=400, justify='left', text=(
            "l'angolo DEL DIPINTO in cui sta il primo scatto. Controlla lo schema: "
            "tasselli vicini devono continuare l'uno nell'altro.")
        ).grid(row=3, column=0, columnspan=4, sticky='w', pady=(4, 0))
        for var in (self.righe, self.colonne):
            var.trace_add('write', lambda *a: self.after_idle(self._ridisegna))

        # --- stima
        s = ttk.LabelFrame(sinistra, text="Stima", padding=8)
        s.pack(fill='x', pady=(10, 0))
        ttk.Label(s, text="Sovrapposizione attesa").grid(row=0, column=0, sticky='w')
        self.sovrapposizione = tk.IntVar(value=0)
        ttk.Spinbox(s, from_=0, to=90, width=5, textvariable=self.sovrapposizione
                    ).grid(row=0, column=1, sticky='w', padx=(4, 4))
        ttk.Label(s, text="%   (0 = non la so)", foreground='#666').grid(row=0, column=2, sticky='w')
        ttk.Label(s, text="Modello").grid(row=1, column=0, sticky='w', pady=(6, 0))
        menu, self.modello, v = _scelta(s, ETICHETTE_MODELLO, 'auto')
        menu.grid(row=1, column=1, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Label(s, text="Precisione").grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.precisione = tk.StringVar(value="Normale")
        ttk.Combobox(s, textvariable=self.precisione, values=list(PRECISIONI),
                     state='readonly', width=34).grid(row=2, column=1, columnspan=2,
                                                      sticky='w', pady=(6, 0))

        # --- uscita
        riga = ttk.Frame(sinistra)
        riga.pack(fill='x', pady=(10, 0))
        ttk.Label(riga, text="Cartella di uscita", width=20).pack(side='left')
        self.uscita = tk.StringVar(value=os.path.join(RADICE, 'risultati', 'mosaico'))
        ttk.Entry(riga, textvariable=self.uscita, width=34).pack(side='left', padx=(0, 6))
        ttk.Button(riga, text="Sfoglia...", command=self._scegli_uscita).pack(side='left')
        self.solo_stima = tk.BooleanVar(value=False)
        ttk.Checkbutton(sinistra, variable=self.solo_stima,
                        text="Solo stima e verifica (non scrive livelli ne' immagine totale)"
                        ).pack(anchor='w', pady=(8, 0))

        # --- schema
        self.tela = tk.Canvas(destra, width=440, height=250, background='#2a2a2a',
                              highlightthickness=0)
        self.tela.pack(fill='both', expand=True)
        self.tela.bind('<Configure>', lambda e: self._ridisegna())
        self.avviso_schema = ttk.Label(destra, foreground='#666', wraplength=420,
                                       justify='left', text="scegli la cartella dei tasselli")
        self.avviso_schema.pack(anchor='w', pady=(4, 0))

        # --- solo da cavalletto: le linee per il controllo di rettilineita'
        c = ttk.LabelFrame(colonna_destra, text="Da cavalletto: linee dritte (facoltative)",
                           padding=8)
        c.pack(fill='x', pady=(10, 0))
        riga = ttk.Frame(c)
        riga.pack(fill='x')
        self.linee = tk.StringVar()
        ttk.Entry(riga, textvariable=self.linee, width=44).pack(side='left', padx=(0, 6))
        ttk.Button(riga, text="Sfoglia...", command=self._scegli_linee).pack(side='left')
        self.raddrizza = tk.BooleanVar(value=False)
        ttk.Checkbutton(c, variable=self.raddrizza,
                        text="Scrivi anche la versione raddrizzata, di fronte al dipinto"
                        ).pack(anchor='w', pady=(4, 0))
        ttk.Label(c, foreground='#666', wraplength=430, justify='left', text=(
            "un file JSON con le linee che nella realta' sono dritte (bordi del supporto, "
            "cornici): servono a controllare la forma del mosaico, che le cuciture non "
            "vedono. Il formato e' descritto in Mosaico\\mosaico_omografia.py.")
        ).pack(anchor='w', pady=(4, 0))

        # --- comandi
        barra = ttk.Frame(cornice)
        barra.pack(fill='x', pady=(12, 0))
        self.avvia = ttk.Button(barra, text="Avvia", command=self._avvia)
        self.avvia.pack(side='left')
        self.apri = ttk.Button(barra, text="Apri la cartella", state='disabled',
                               command=self._apri_cartella)
        self.apri.pack(side='left', padx=6)
        self.apri_anteprima = ttk.Button(barra, text="Apri l'anteprima", state='disabled',
                                         command=self._apri_anteprima)
        self.apri_anteprima.pack(side='left')
        self.stato = ttk.Label(barra, text="pronto", foreground='#666')
        self.stato.pack(side='left', padx=12)
        self.barra = ttk.Progressbar(cornice, mode='determinate', maximum=1.0)
        self.barra.pack(fill='x', pady=(8, 0))

        self.log = ScrolledText(cornice, height=11, wrap='word', font=('Consolas', 9))
        self.log.pack(fill='both', expand=True, pady=(8, 0))

    # ---------------------------------------------------------------- tasselli
    def _scegli_cartella(self):
        p = filedialog.askdirectory(title="Cartella con i tasselli")
        if p:
            self._carica_cartella(p)

    def _carica_cartella(self, p):
        p = os.path.normpath(p)
        self.cartella.set(p)
        try:
            self.immagini = mosaico.elenca_immagini(p)
        except OSError as exc:
            messagebox.showerror("Cartella", str(exc))
            return
        if not self.immagini:
            self.conteggio.configure(text="nessuna immagine in questa cartella", foreground='#a00')
        else:
            primi = ", ".join(os.path.basename(x) for x in self.immagini[:3])
            ultimo = os.path.basename(self.immagini[-1])
            self.conteggio.configure(
                foreground='#666',
                text=f"{len(self.immagini)} immagini, in quest'ordine: {primi}"
                     f"{' ... ' + ultimo if len(self.immagini) > 3 else ''}")
            fuori = os.path.normpath(os.path.join(p, 'mosaico'))
            self.uscita.set(fuori)
        self._ridisegna()
        # Le miniature si leggono in un thread: su file grandi sono qualche
        # decimo di secondo l'una, e la finestra non deve bloccarsi.
        da_leggere = [x for x in self.immagini if x not in self.miniature]
        if da_leggere:
            threading.Thread(target=self._leggi_miniature, args=(da_leggere,),
                             daemon=True).start()

    def _leggi_miniature(self, percorsi):
        for p in percorsi:
            try:
                img, _ = image_io.anteprima_veloce(p, 240)
                img = image_io.a_8bit(img)
                if img.ndim == 3 and img.shape[2] > 3:
                    img = img[:, :, :3]
                self.coda.put(('miniatura', (p, Image.fromarray(np.ascontiguousarray(img)))))
            except Exception as exc:
                self.coda.put(('log', f"miniatura di {os.path.basename(p)} non leggibile: {exc}\n"))

    def _griglia(self):
        try:
            return max(1, int(self.righe.get())), max(1, int(self.colonne.get()))
        except (tk.TclError, ValueError):
            return None

    def _ridisegna(self):
        c = self.tela
        c.delete('all')
        self._foto = []
        g = self._griglia()
        if g is None:
            return
        R, C = g
        posizioni = mosaico.posizioni_griglia(R, C, self.percorso(), self.partenza())
        n = len(self.immagini)
        if self.ripresa() == 'cavalletto':
            self.avviso_schema.configure(
                foreground='#666',
                text="da cavalletto griglia e percorso non servono: i vicini si trovano da "
                     "soli, e l'ordine dei file non conta")
        elif n and n != R * C:
            self.avviso_schema.configure(
                foreground='#a00',
                text=f"{n} immagini, ma una griglia {R} x {C} ne vuole {R * C}: "
                     f"controlla righe e colonne")
        elif n:
            self.avviso_schema.configure(
                foreground='#666',
                text="il numero e' l'ordine di ripresa; la freccia segue il percorso")
        Wc, Hc = max(50, c.winfo_width()), max(50, c.winfo_height())
        rapporto = 4 / 3
        if self.immagini:
            try:
                info = image_io.info(self.immagini[0])
                rapporto = info.width / float(info.height)
            except Exception:
                pass
        margine = 8
        cw = min((Wc - 2 * margine) / C, (Hc - 2 * margine) / R * rapporto)
        ch = cw / rapporto
        ox = (Wc - cw * C) / 2
        oy = (Hc - ch * R) / 2
        centri = []
        for k, (r, col) in enumerate(posizioni):
            x0, y0 = ox + col * cw, oy + r * ch
            centri.append((x0 + cw / 2, y0 + ch / 2))
            p = self.immagini[k] if k < n else None
            if p in self.miniature and cw > 6 and ch > 6:
                mini = self.miniature[p].copy()
                mini.thumbnail((int(cw) - 2, int(ch) - 2))
                foto = ImageTk.PhotoImage(mini)
                self._foto.append(foto)
                c.create_image(x0 + cw / 2, y0 + ch / 2, image=foto)
            c.create_rectangle(x0 + 1, y0 + 1, x0 + cw - 1, y0 + ch - 1, outline='#bbbbbb')
        if len(centri) > 1:
            punti = [v for xy in centri for v in xy]
            c.create_line(*punti, fill='#3a8eff', width=3, arrow='last',
                          arrowshape=(14, 16, 6))
        corpo = max(8, min(22, int(ch / 4)))
        for k, (x, y) in enumerate(centri):
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                c.create_text(x + dx, y + dy, text=str(k + 1), fill='black',
                              font=('Segoe UI', corpo, 'bold'))
            c.create_text(x, y, text=str(k + 1), fill='white', font=('Segoe UI', corpo, 'bold'))

    def _scegli_linee(self):
        p = filedialog.askopenfilename(title="File delle linee dritte",
                                       filetypes=[("JSON", "*.json"), ("Tutti i file", "*.*")])
        if p:
            self.linee.set(os.path.normpath(p))

    def _scegli_uscita(self):
        p = filedialog.askdirectory(title="Cartella di uscita")
        if p:
            self.uscita.set(os.path.normpath(p))

    # ------------------------------------------------------------------ avvio
    def _avvia(self):
        if self.al_lavoro:
            return
        # Le variabili tkinter si leggono SOLO qui, nel thread principale.
        g = self._griglia()
        if not self.immagini:
            messagebox.showerror("Mancano i tasselli", "Scegli la cartella dei tasselli.")
            return
        cavalletto = self.ripresa() == 'cavalletto'
        linee = self.linee.get().strip()
        if cavalletto and linee and not os.path.isfile(linee):
            messagebox.showerror("Linee", "Il file delle linee non esiste.")
            return
        if cavalletto and self.raddrizza.get() and not linee:
            messagebox.showerror("Linee", "Per la versione raddrizzata serve il file delle linee.")
            return
        if not cavalletto and (g is None or g[0] * g[1] != len(self.immagini)):
            messagebox.showerror(
                "Griglia", f"{len(self.immagini)} immagini non riempiono una griglia di "
                           f"{g[0] if g else '?'} x {g[1] if g else '?'}.")
            return
        if not self.uscita.get().strip():
            messagebox.showerror("Manca la cartella", "Scegli dove salvare.")
            return
        sov = int(self.sovrapposizione.get() or 0)
        imp = {'immagini': list(self.immagini), 'righe': g[0] if g else 0,
               'colonne': g[1] if g else 0, 'cavalletto': cavalletto,
               'linee': linee or None, 'raddrizza': bool(self.raddrizza.get()),
               'percorso': self.percorso(), 'partenza': self.partenza(),
               'modello': self.modello(), 'sovrapposizione': sov if sov > 0 else None,
               'fattore': PRECISIONI[self.precisione.get()],
               'uscita': self.uscita.get().strip(), 'livelli': not self.solo_stima.get()}
        self.al_lavoro = True
        self.avvia.configure(state='disabled')
        self.apri.configure(state='disabled')
        self.apri_anteprima.configure(state='disabled')
        self.stato.configure(text="in corso...", foreground='#a60')
        self.barra['value'] = 0
        self.log.delete('1.0', 'end')
        threading.Thread(target=self._lavora, args=(imp,), daemon=True).start()

    def _lavora(self, imp):
        originale = sys.stdout
        sys.stdout = CodaStampa(self.coda, originale)
        try:
            avanza = lambda testo, f: self.coda.put(('avanzamento', (testo, f)))
            if imp['cavalletto']:
                report = mosaico_omografia.componi(
                    imp['immagini'], imp['uscita'], fattore=imp['fattore'], linee=imp['linee'],
                    raddrizza=imp['raddrizza'], scrivi_immagine=imp['livelli'], progress=avanza)
            else:
                report = mosaico.componi_mosaico(
                    imp['immagini'], imp['righe'], imp['colonne'], imp['uscita'],
                    percorso=imp['percorso'], partenza=imp['partenza'], modello=imp['modello'],
                    sovrapposizione=imp['sovrapposizione'], fattore=imp['fattore'],
                    scrivi_livelli=imp['livelli'], progress=avanza)
            self.coda.put(('fine', (imp['uscita'], report)))
        except Exception as exc:
            self.coda.put(('errore', f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"))
        finally:
            sys.stdout = originale

    # ------------------------------------------------------------------- coda
    def _svuota_coda(self):
        try:
            while True:
                genere, dato = self.coda.get_nowait()
                if genere == 'log':
                    self.log.insert('end', dato)
                    self.log.see('end')
                elif genere == 'miniatura':
                    p, img = dato
                    self.miniature[p] = img
                    self._ridisegna()
                elif genere == 'avanzamento':
                    testo, f = dato
                    self.barra['value'] = f
                    self.stato.configure(text=testo)
                elif genere == 'fine':
                    cartella, report = dato
                    self._concludi(cartella, report)
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

    def _concludi(self, cartella, report):
        self.al_lavoro = False
        self.ultima_cartella = cartella
        self.ultimo_report = report
        self.avvia.configure(state='normal')
        self.apri.configure(state='normal')
        self.apri_anteprima.configure(state='normal')
        self.barra['value'] = 1.0
        v = report.get('verifica', {})
        n_mis = sum(c['stato'] == 'misurata' for c in report['coppie'])
        if v.get('residuo_mediano_px') is None:
            testo, colore = f"fatto, ma nessuna cucitura verificabile ({n_mis} coppie misurate)", '#a60'
        else:
            testo = (f"fatto: {n_mis}/{len(report['coppie'])} coppie misurate, cuciture "
                     f"{v['residuo_mediano_px']:.2f} px di mediana, "
                     f"{100 * v['quota_sopra_2px']:.0f}% oltre 2 px")
            colore = '#080' if (n_mis == len(report['coppie'])
                                and v['quota_sopra_2px'] < 0.05) else '#a60'
        self.stato.configure(text=testo, foreground=colore)

    def _apri_cartella(self):
        cartella = getattr(self, 'ultima_cartella', None)
        if cartella and os.path.isdir(cartella):
            os.startfile(cartella)

    def _apri_anteprima(self):
        p = (self.ultimo_report or {}).get('anteprima')
        if p and os.path.exists(p):
            os.startfile(p)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Finestra per il mosaico di tasselli.")
    ap.add_argument('--cartella', help="precompila la cartella dei tasselli")
    a = ap.parse_args()
    Applicazione({'cartella': a.cartella}).mainloop()
