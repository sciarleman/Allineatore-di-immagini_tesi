# Schemi riassuntivi per la tesi

Gli schemi descrivono il programma per passaggi: ogni riquadro dice cosa
succede e, in piccolo, quale funzione lo fa. Servono a spiegare il metodo senza
riportare il codice.

Ogni schema c'è in tre formati: `.png` (per Word), `.pdf` e `.svg` (vettoriali,
per LaTeX o per ritoccarli con Inkscape o Illustrator). Si rigenerano tutti con:

```powershell
python Documentazione\schemi\disegna_schemi.py
```

I testi dei riquadri stanno nello script: per cambiare una parola basta
modificarla lì e rilanciare.

| schema | cosa mostra |
|---|---|
| `schema_1_panoramica` | i tre programmi (allineatore, falso colore, mosaico), cosa ricevono e cosa producono |
| `schema_2_allineamento_di_una_coppia` | i sei passi di `align_paintings.align`, dalla lettura alla scrittura, con il ripiego |
| `schema_3_stima_a_tre_scale` | la posa affine, il campo grossolano e il campo fine, con le condizioni per applicarli |
| `schema_4_verifica_e_misure` | la verifica a finestre, le misure riportate e quelle in più per le radiografie |

Colori: grigio = dati, blu = elaborazione, arancio = controllo o misura,
tratteggio = passo facoltativo, verde = prodotto finale.

Da fare, se servono: la sezione delle radiografie (zona, bordo, fascia, punti),
il falso colore, il mosaico, la gestione dei file grandi e della memoria.
