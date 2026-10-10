"""
Gli strumenti a mano per UNA immagine del lotto: punti di corrispondenza e
bordo tracciato.

Erano la sezione a parte delle radiografie. Poi si e' visto che non sono una
questione di banda: l'UV di un affresco, che in automatico restava fuori ai
bordi, con il bordo tracciato a mano e' passato da 28.7 a 1.5 px lungo il
contorno. Ora si assegnano a qualunque immagine della lista, e chi non ne ha
va in automatico come sempre.

La zona del dipinto non sta qui: si disegna una volta sul riferimento, nella
finestra principale, e vale per tutto il lotto.
"""

import os

import numpy as np

import tkinter as tk
from tkinter import messagebox

import finestra_punti
import image_io
import punti_manuali
from finestra_maschera import FinestraMaschera
from transform_model import Transform

FASCIA_PX = 150


def descrivi(strumenti):
    """Gli strumenti di un'immagine in poche parole, per la lista del lotto."""
    st = strumenti or {}
    parti = []
    n = 0 if st.get('punti') is None else len(st['punti'])
    if n:
        uso = [u for u, k in (("vincolo", 'punti_come_vincolo'), ("posa", 'punti_come_posa'))
               if st.get(k)]
        parti.append(f"{n} punti ({' e '.join(uso) if uso else 'solo TRE'})")
    if st.get('contorno') is not None:
        parti.append(f"bordo + fascia {st['fascia_di_bordo']} px" if st.get('fascia_di_bordo')
                     else "bordo tracciato (non usato)")
    return ", ".join(parti) if parti else "automatica"


class FinestraStrumenti(tk.Toplevel):
    """Punti e bordo per un'immagine; `risultato` e' il dizionario degli
    strumenti (vuoto = automatica), o None se si annulla."""

    def __init__(self, padre, riferimento, immagine, cartella_output="", strumenti=None,
                 ultimo=None, ha_zona=False):
        super().__init__(padre)
        self.title(f"Strumenti a mano — {os.path.basename(immagine)}")
        self.resizable(False, False)
        self.riferimento, self.immagine = riferimento, immagine
        self.ultimo = ultimo or {}
        self.ha_zona = ha_zona
        st = dict(strumenti or {})
        self.punti = st.get('punti')
        self.contorno = st.get('contorno')
        self.risultato = None

        corpo = tk.Frame(self, padx=16, pady=12)
        corpo.pack(fill="both", expand=True)
        tk.Label(corpo, justify="left", fg="#333", wraplength=620, text=(
            "Servono solo dove l'automatico non basta: di norma radiografie, o bande con "
            "deformazioni grandi ai bordi.\n"
            "I PUNTI, per impostazione predefinita, servono a MISURARE: il programma non li "
            "usa per allineare e riporta il TRE, lo scarto fra dove la trasformazione porta "
            "ogni dettaglio e dove l'hai cliccato. Mettili sparsi, solo dove riconosci con "
            "certezza lo stesso dettaglio.\n"
            "Il BORDO tracciato sull'immagine, insieme alla zona disegnata sul riferimento, "
            "porta l'uno sull'altra in una fascia lungo il contorno.")).pack(anchor="w")

        riga = tk.Frame(corpo)
        riga.pack(fill="x", pady=(10, 2))
        self.btn_punti = tk.Button(riga, command=self.apri_punti, padx=8, width=34)
        self.btn_punti.pack(side="left")
        # Spente per impostazione predefinita: sulla Cappellochiaro, con le tre
        # scale, il vincolo dai 17 punti lasciava i 115 dettagli indipendenti a
        # 5.0 px di mediana come senza, e la posa dai punti portava i dettagli
        # non usati in alto a sinistra da 3.9 a 41.7 px.
        self.punti_come_vincolo = tk.BooleanVar(value=bool(st.get('punti_come_vincolo')))
        tk.Checkbutton(corpo, variable=self.punti_come_vincolo,
                       text="Usa i punti come vincolo locale dove la verifica non misura"
                       ).pack(anchor="w")
        self.punti_come_posa = tk.BooleanVar(value=bool(st.get('punti_come_posa')))
        tk.Checkbutton(corpo, variable=self.punti_come_posa,
                       text="Usa i punti anche per la posa di partenza (solo se l'automatico "
                            "non aggancia)").pack(anchor="w")

        riga = tk.Frame(corpo)
        riga.pack(fill="x", pady=(12, 2))
        self.btn_bordo = tk.Button(riga, command=self.apri_bordo, padx=8, width=34)
        self.btn_bordo.pack(side="left")
        # Accesa appena c'e' un bordo: tracciarlo serve solo a questo.
        self.fascia = tk.BooleanVar(value=bool(st.get('fascia_di_bordo'))
                                    or st.get('contorno') is None)
        tk.Checkbutton(corpo, variable=self.fascia,
                       text=f"Allinea il bordo: porta il contorno della zona sul bordo "
                            f"tracciato (fascia di {FASCIA_PX} px)").pack(anchor="w")
        self.lbl_zona = tk.Label(corpo, fg="#b36b00", justify="left", wraplength=620, text="")
        self.lbl_zona.pack(anchor="w")

        piede = tk.Frame(self, padx=16, pady=12)
        piede.pack(fill="x")
        tk.Button(piede, text="Togli tutto (automatica)", command=self.togli).pack(side="left")
        tk.Button(piede, text="Annulla", command=self.destroy).pack(side="right", padx=4)
        tk.Button(piede, text="Applica", command=self.applica,
                  bg="#2196F3", fg="white").pack(side="right")

        self._aggiorna()
        self.transient(padre)
        self.grab_set()
        self.bind("<Escape>", lambda e: self.destroy())
        if strumenti is None:
            self.after(50, lambda: self._riprendi_salvati(cartella_output))

    def _aggiorna(self):
        n = 0 if self.punti is None else len(self.punti)
        self.btn_punti.config(
            text="Punti di corrispondenza..." if not n else f"Punti: {n} (rivedi)")
        nb = 0 if self.contorno is None else len(self.contorno)
        self.btn_bordo.config(
            text="Bordo tracciato sull'immagine..." if not nb
            else f"Bordo: {nb} punti (ridisegna)")
        self.lbl_zona.config(
            text="" if (self.ha_zona or self.contorno is None) else
            "La fascia di bordo ha bisogno anche della zona disegnata sul riferimento "
            "(«Limita la stima a una zona che scelgo io», nella finestra principale): "
            "senza, il bordo tracciato non viene usato.")

    def _riprendi_salvati(self, cartella_output):
        """Punti e bordo dell'ultima corsa su questa immagine tornano da soli:
        e' quello che rende il lavoro riprendibile senza rifare tutto."""
        recuperati, n_man = punti_manuali.punti_salvati(cartella_output, self.immagine)
        if recuperati is not None:
            # Le maniglie di una versione precedente erano corrispondenze prese
            # a occhio su un'anteprima: come vincoli esatti davano bordi a gobbe.
            recuperati = recuperati[n_man:] if n_man else recuperati
            if len(recuperati) and messagebox.askyesno(
                    "Ci sono gia' dei punti",
                    f"Per questa immagine ci sono gia' {len(recuperati)} punti di una "
                    f"corsa precedente.\n\nRiprenderli?", parent=self):
                self.punti = recuperati
        bordo = punti_manuali.contorno_salvato(cartella_output, self.immagine)
        if bordo is not None and messagebox.askyesno(
                "Bordo gia' tracciato",
                f"Per questa immagine c'e' gia' un bordo tracciato a mano ({len(bordo)} "
                f"punti) in una corsa precedente.\n\nRiprenderlo?", parent=self):
            self.contorno = bordo
        self._aggiorna()

    def apri_punti(self):
        # Le zone che l'ultima verifica non ha misurato dicono DOVE un punto
        # aggiunge informazione; e dal secondo giro la vista di destra mostra
        # l'immagine gia' corretta, cosi' si lavora solo su quello che resta.
        T_prec = None
        par = self.ultimo.get('parametri')
        if par and os.path.exists(par):
            try:
                T_prec = Transform.load(par)
            except Exception:
                T_prec = None
        try:
            f = finestra_punti.FinestraPunti(self, self.riferimento, self.immagine,
                                             punti=self.punti,
                                             zone=self.ultimo.get('zone_critiche') or [],
                                             trasformazione=T_prec)
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile aprire le immagini:\n{exc}",
                                 parent=self)
            return
        self.wait_window(f)
        if f.risultato is not None:
            self.punti = f.risultato
            self._aggiorna()

    def apri_bordo(self):
        """Il bordo va tracciato a mano: ricavato per soglia, in alcuni tratti
        seguiva il bordo dell'immagine invece di quello del dipinto."""
        f = FinestraMaschera(
            self, self.immagine, titolo="Bordo tracciato sull'immagine",
            scopo="Segui il BORDO DEL DIPINTO su questa immagine, lo stesso margine "
                  "che la zona segue sul riferimento.")
        self.wait_window(f)
        if f.maschera is None:
            return
        info = image_io.info(self.immagine)
        C = punti_manuali.contorno_da_maschera(f.maschera, (info.height, info.width))
        if C is None:
            messagebox.showwarning("Contorno non leggibile",
                                   "Non sono riuscito a ricavare un contorno.", parent=self)
            return
        self.contorno = C
        self.fascia.set(True)
        self._aggiorna()

    def togli(self):
        self.risultato = {}
        self.destroy()

    def applica(self):
        st = {}
        if self.punti is not None and len(self.punti):
            st.update(punti=np.asarray(self.punti, dtype=float),
                      punti_come_vincolo=bool(self.punti_come_vincolo.get()),
                      punti_come_posa=bool(self.punti_come_posa.get()))
        if self.contorno is not None:
            st.update(contorno=self.contorno,
                      fascia_di_bordo=FASCIA_PX if self.fascia.get() else 0)
        self.risultato = st
        self.destroy()
