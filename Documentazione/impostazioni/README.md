# Le impostazioni dell'Allineatore, misurate

Che cosa cambia, in precisione, tempo e memoria, muovendo ogni manopola del
programma. Misurato il 5 e 6 ottobre 2026 su due dipinti, 44 configurazioni
più 52 stime di confronto. I dati grezzi e gli script sono in `dati/` e
`script/`.

Il 7 ottobre le **impostazioni avanzate** (densità, confidenza, lato dei tile)
sono state rimisurate con il programma di oggi, su più coppie e con una griglia
di verifica che arriva ai bordi: 45 configurazioni. Dove le due misure non
concordano vale la seconda: vedi
[Le impostazioni avanzate, rimisurate](#le-impostazioni-avanzate-rimisurate-il-7-ottobre).

## In breve: che cosa scegliere

| richiesta | impostazione | che cosa aspettarsi |
|---|---|---|
| il risultato da tenere | **precisione massima** | 0.05–0.3 px di residuo mediano; 2–6 min per immagine di sola stima; 4–8 GB di memoria |
| un lotto di routine, o poca RAM | **precisa** (la predefinita) | 0.2–0.6 px sulle bande fotografiche, circa 1 px dove la banda condivide poca struttura; 35–140 s; 1.5–2.4 GB |
| vedere se aggancia, prima di impegnare tempo | **veloce** | 0.9–2 px; 15–25 s; 1 GB |
| solo capire se le immagini sono quelle giuste | **molto veloce** | 1.3–4 px; 10–15 s |
| densità del campo | lasciare **media**; **fitta** si può provare su una banda ottica con deformazioni forti, **mai sulle radiografie**; mai **larga** | sulle bande fotografiche differenze entro 0.03 px; la fitta guadagna 0.03–0.2 px su affresco e SWIR e porta la radiografia da 1.1 a 3.0 px |
| confidenza minima | lasciare **normale**, sempre | **bassa** non migliora nessuna coppia e porta la radiografia da 1.1 a 14.5 px e lo SWIR da 1.0 a 3.2; **alta** non ha dato vantaggi in nessuna misura |
| deformazioni locali | **sempre accese** | spente lasciano 6–35 px |
| lato dei tile | lasciare **2048** | l'immagine scritta è identica byte per byte con 1024, 2048 e 4096; cambia solo la memoria, 0.8–1.4 GB |
| formato | quello dell'ingresso; il **TIFF** è il più leggero da scrivere | il PNG raddoppia il tempo di scrittura e chiede 2.3–3.8 GB; il JPEG perde informazione |

La regola che vale ovunque: **dimezzare il fattore di precisione divide il
residuo per 3–4 e moltiplica per 3 tempo e memoria.** Le altre manopole
spostano il risultato di centesimi di pixel, tranne nei due casi segnalati
sotto.

## Come sono state fatte le misure

**Le immagini.**

| | dimensioni | bande usate |
|---|---|---|
| Fra Galgario (olio su tela) | 15848 × 19850 px, 314 Mpx | VIS di riferimento; NIR850, UVF, UVR; una radiografia (Cappellochiaro) solo per la precisione |
| affresco (cartella `checkà`) | 29049 × 20428 px, 593 Mpx | VIS di riferimento; NIR (29564 × 20412) e UV (28907 × 20108) |

Le due serie sono diverse per un aspetto che conta: le bande del Fra Galgario
hanno tutte le stesse dimensioni in pixel del riferimento e sono quasi
sovrapposte (campo di deformazione fino a ~30 px); quelle dell'affresco hanno
dimensioni diverse, 165–280 px di spostamento d'insieme e un campo che arriva a
160–200 px ai bordi.

**Che cosa si è fatto girare.** Ogni configurazione passa per
`align_paintings.align` con gli stessi argomenti che le dà il lotto
dell'interfaccia (tre scale in automatico), in un processo a sé, con stima,
verifica e immagini di controllo ma senza scrivere l'immagine allineata. La
scrittura è misurata a parte. Predefinite: precisione «precisa» (fattore 2),
densità media (blocchi da 128 px), confidenza 0.10, deformazioni locali accese,
nessuna maschera.

**La macchina.** Portatile con Intel Core i7-1065G7 (8 thread), 15.8 GB di RAM,
SSD, Windows 11.

**Che cosa vuol dire «residuo».** È quello della verifica del programma: 64
finestre da 512 px a piena risoluzione, in posizioni fisse, su cui si misura
di quanti pixel le due immagini restano fuori registro dopo l'allineamento. Si
riportano la mediana, la quota di finestre oltre 1 e 5 px e il massimo. È una
misura di coerenza fatta con lo stesso strumento della stima, non un confronto
con punti annotati a mano.

**Limiti di queste misure.**

- Due dipinti, una corsa per configurazione: i risultati sono ripetibili (la
  stima è deterministica), i tempi oscillano di qualche secondo.
- I tempi dell'affresco sono misurati su copie TIFF non compresse dei tre PNG
  (stessi pixel), per non pagare a ogni corsa la decodifica. Con i PNG veri
  ogni immagine costa circa 2 minuti in più: nella corsa reale del 5 ottobre,
  8.0–8.3 min di stima e verifica per immagine contro i 6.1–6.2 qui sotto.
  I residui sono quelli che oggi si ottengono anche dai PNG. Il confronto su
  partenza e portata, fatto sui PNG prima della correzione 5 in fondo, dava a
  massima 0.043–0.049 px per il NIR e 0.248 per l'UV invece di 0.051 e 0.295.
- La maschera («Limita la stima a una zona») e il margine automatico non sono
  stati misurati: la maschera si disegna a mano.
- Le radiografie hanno una sezione loro, con punti e bordo: qui la
  Cappellochiaro compare solo in automatico, come termine di paragone.

## Precisione

Tempo = stima, verifica e immagini di controllo, senza lettura dei PNG e senza
scrittura. Memoria = picco di memoria privata del processo.

### Fra Galgario, 314 Mpx

| banda | precisione | residuo mediano [IC 90%] | oltre 1 px | oltre 5 px | massimo | tempo | memoria |
|---|---|---|---|---|---|---|---|
| NIR850 | massima | 0.054 px [0.04–0.07] | 2% | 0% | 1.4 px | 106 s | 4.0 GB |
| | precisa | 0.214 px [0.18–0.26] | 5% | 0% | 2.4 px | 38 s | 1.5 GB |
| | veloce | 0.920 px [0.82–1.07] | 43% | 0% | 4.2 px | 16 s | 0.8 GB |
| | molto veloce | 1.346 px [1.16–1.66] | 74% | 3% | 7.6 px | 10 s | 0.8 GB |
| UVF | massima | 0.160 px [0.12–0.19] | 0% | 0% | 0.8 px | 104 s | 4.0 GB |
| | precisa | 0.574 px [0.50–0.62] | 21% | 0% | 4.3 px | 34 s | 1.5 GB |
| | veloce | 1.327 px [1.18–1.65] | 67% | 0% | 4.3 px | 15 s | 0.8 GB |
| | molto veloce | 1.832 px [1.57–3.06] | 82% | 16% | 9.0 px | 10 s | 0.8 GB |
| UVR | massima | 0.092 px [0.08–0.10] | 4% | 2% | 12.2 px | 109 s | 4.0 GB |
| | precisa | 0.323 px [0.30–0.39] | 8% | 0% | 1.8 px | 34 s | 1.5 GB |
| | veloce | 1.172 px [1.03–1.26] | 63% | 2% | 9.4 px | 15 s | 0.8 GB |
| | molto veloce | 2.397 px [1.85–3.28] | 91% | 11% | 7.4 px | 10 s | 0.8 GB |

Finestre misurabili: 60–61 su 64 per NIR850 e UVF, 52–55 su 64 per l'UVR.

Sull'UVR a massima una sola finestra, a bassa confidenza (0.08), sta a 12.2 px:
è la stessa che compare nel confronto sulla portata, più sotto.

### Affresco, 593 Mpx

| banda | precisione | residuo mediano [IC 90%] | oltre 1 px | oltre 5 px | massimo | tempo | memoria |
|---|---|---|---|---|---|---|---|
| NIR | massima | 0.051 px [0.04–0.06] | 0% | 0% | 0.8 px | 372 s | 7.7 GB |
| | precisa | 0.239 px [0.18–0.36] | 16% | 0% | 4.5 px | 136 s | 2.4 GB |
| | veloce | 0.912 px [0.71–1.11] | 44% | 0% | 5.0 px | 21 s | 1.0 GB |
| | molto veloce | 1.493 px [1.15–1.88] | 67% | 5% | 8.0 px | 11 s | 1.0 GB |
| UV | massima | 0.295 px [0.21–0.36] | 7% | 0% | 3.6 px | 363 s | 7.6 GB |
| | precisa | 1.094 px [0.94–1.29] | 60% | 5% | 19.0 px | 66 s | 2.3 GB |
| | veloce | 2.094 px [1.87–2.21] | 80% | 11% | 20.5 px | 25 s | 1.0 GB |
| | molto veloce | 2.775 px [2.31–3.49] | 93% | 25% | 25.3 px | 14 s | 1.0 GB |

Finestre misurabili: 64 su 64 per il NIR, 54–57 su 64 per l'UV.

I tempi a massima (e del NIR a precisa) comprendono un primo tentativo fallito
e il ripiego. Dal 6 ottobre il fallimento viene riconosciuto alla prima
passata e quei tempi scendono di circa un terzo: vedi «Difetti emersi», punto 1.

### Radiografia Cappellochiaro, in automatico

Senza punti, senza maschera, senza il bordo: è il termine di paragone, non il
modo in cui le radiografie vanno allineate.

| precisione | residuo mediano | finestre misurabili | oltre 1 px | oltre 5 px | massimo | tempo | memoria |
|---|---|---|---|---|---|---|---|
| massima | 0.421 px | 79 / 224 | 14% | 1% | 31.1 px | 225 s | 6.7 GB |
| precisa | 1.069 px | 81 / 225 | 51% | 4% | 23.7 px | 45 s | 2.1 GB |
| veloce | 2.136 px | 73 / 224 | 84% | 29% | 45.7 px | 23 s | 0.9 GB |
| molto veloce | 4.124 px | 82 / 224 | 94% | 41% | 47.1 px | 17 s | 0.7 GB |

Solo un terzo delle finestre è misurabile: il numero vale per quelle.

### Che cosa se ne ricava

- **La massima è sempre la più precisa**, su tutte e sei le immagini: il
  residuo è 3–4 volte più piccolo che a «precisa». Sulle stesse finestre, a
  massima migliorano 22–46 finestre nelle bande del Fra Galgario e 34–35
  nell'affresco; ne peggiorano 0–2.
- **Costa circa il triplo**: 104–109 s contro 34–38 sul Fra Galgario, 4.0 GB
  contro 1.5. Sull'affresco, 7.7 GB su una macchina da 15.8: ci sta, ma con i
  PNG veri il picco è stato di 11 GB.
- **«Veloce» e «molto veloce» non sono risultati da usare**: da 44% a 93%
  delle finestre oltre 1 px. Servono a sapere in 10–25 secondi se la coppia
  aggancia.
- **La stima di tempo che la finestra mostrava era bassa** a precisione
  massima: indicava 8–12 minuti per le due bande dell'affresco, e ne sono
  serviti 18–21. È stata tolta.

## Densità del campo

A precisione «precisa». È il lato dei blocchi della seconda passata: fitta 96
px, media 128, larga 192.

| immagine | densità | residuo mediano | oltre 1 px | oltre 5 px | massimo | tempo |
|---|---|---|---|---|---|---|
| Fra Galgario NIR850 | fitta | 0.202 px | 7% | 0% | 1.9 px | 38 s |
| | media | 0.214 px | 5% | 0% | 2.4 px | 38 s |
| | larga | 0.206 px | 7% | 0% | 2.3 px | 32 s |
| Fra Galgario UVF | fitta | 0.583 px | 28% | 0% | 4.5 px | 37 s |
| | media | 0.574 px | 21% | 0% | 4.3 px | 34 s |
| | larga | 0.529 px | 16% | 0% | 2.7 px | 31 s |
| affresco NIR | fitta | 0.242 px | 5% | 0% | 4.5 px | 122 s |
| | media | 0.239 px | 16% | 0% | 4.5 px | 136 s |
| | larga | **3.888 px** | 70% | **48%** | 129.5 px | 54 s |
| affresco UV | fitta | 1.005 px | 50% | 4% | 23.0 px | 66 s |
| | media | 1.094 px | 60% | 5% | 19.0 px | 66 s |
| | larga | 1.275 px | 67% | 5% | 20.3 px | 56 s |

- Sul Fra Galgario la densità non conta: le tre scelte stanno entro 0.06 px.
- Sull'affresco, dove il campo è grande e varia in fretta, la **fitta** aiuta
  un po': sul NIR le finestre oltre 1 px scendono dal 16% al 5% (12 finestre
  migliorate, 2 peggiorate).
- La **larga** non dà mai un vantaggio che valga i 5–10 secondi risparmiati, e
  sul NIR dell'affresco produce un risultato sbagliato. Non è la densità in sé:
  con quei blocchi il primo tentativo, pur fallito (8 blocchi buoni su 88 nella
  prima passata), chiudeva a 3.9 px, sotto la soglia di 10 px che fa scattare
  il ripiego, e il programma teneva quel risultato. Corretto il 6 ottobre: oggi
  dà 0.361 px. Vedi «Difetti emersi», punto 1.

## Confidenza minima dei blocchi

A precisione «precisa». Misure del 5 ottobre: il consiglio di usare la **bassa**
dove la banda ha poca struttura **non vale più**, vedi la sezione seguente.

| immagine | confidenza | residuo mediano | oltre 1 px | oltre 5 px | massimo |
|---|---|---|---|---|---|
| Fra Galgario NIR850 | bassa 0.05 | 0.210 px | 8% | 0% | 2.3 px |
| | normale 0.10 | 0.214 px | 5% | 0% | 2.4 px |
| | alta 0.20 | 0.210 px | 3% | 0% | 1.4 px |
| Fra Galgario UVF | bassa 0.05 | 0.595 px | 25% | 0% | 3.4 px |
| | normale 0.10 | 0.574 px | 21% | 0% | 4.3 px |
| | alta 0.20 | 0.550 px | 21% | 0% | 2.2 px |
| affresco NIR | bassa 0.05 | 0.240 px | 9% | 0% | 3.1 px |
| | normale 0.10 | 0.239 px | 16% | 0% | 4.5 px |
| | alta 0.20 | **errore**: «Solo 1 blocchi affidabili» | | | |
| affresco UV | bassa 0.05 | 1.001 px | 50% | **0%** | 3.0 px |
| | normale 0.10 | 1.094 px | 60% | 5% | 19.0 px |
| | alta 0.20 | 1.135 px | 59% | 14% | 87.4 px |

I tempi non cambiano (entro 3 secondi).

- Sul Fra Galgario non conta: tutto entro 0.05 px.
- Sull'affresco la **bassa** è la migliore: sull'UV porta le finestre oltre 5
  px dal 5% allo 0% e il massimo da 19 a 3 px. È coerente con ciò che la
  manopola fa: dove la banda condivide poca struttura i blocchi hanno
  confidenza bassa, e scartarli lascia zone senza misure.
- L'**alta** non ha dato vantaggi misurabili e sull'affresco fa danno: sull'UV
  lascia zone a 87 px, sul NIR la stima si interrompeva con un errore (oggi dà
  0.236 px: vedi «Difetti emersi», punto 2).

## Le impostazioni avanzate, rimisurate il 7 ottobre

La domanda era se le manopole della finestra «Impostazioni avanzate» servano
davvero. Rispetto alle misure qui sopra sono cambiate due cose. Il programma:
dal 6 ottobre ha due livelli grossolani e riconosce subito un aggancio fallito.
E la misura: oltre alla verifica del programma (64 finestre, che ai bordi non
arriva) c'è una griglia fitta di finestre da 512 px su tutto il foglio, bordi e
angoli compresi (450–940 finestre per immagine), su cui ogni configurazione si
confronta con la predefinita **finestra per finestra**. Le coppie sono nove:
cinque del Fra Galgario a «precisa» (NIR850, UVF, UVR, SWIR, radiografia
Cappellochiaro), due dell'affresco a «precisa» (NIR, UV) e due del Fra Galgario
a «massima» (NIR850, UVF). Ogni manopola si muove da sola attorno ai valori
predefiniti: blocchi da 128 px, confidenza 0.10.

Delle sette voci della finestra, solo due cambiano l'allineamento:

| voce | che cosa cambia |
|---|---|
| densità del campo | l'allineamento |
| confidenza minima dei blocchi | l'allineamento |
| «segnala se il residuo supera» | solo il giudizio scritto nella tabella |
| lato dei tile | solo la memoria in scrittura |
| formati, margine automatico, «solo stima» | che cosa viene scritto, non come |

**Residuo mediano della verifica del programma** (px a piena risoluzione):

| coppia | predefinita | densità fitta | densità larga | confidenza bassa | confidenza alta |
|---|---|---|---|---|---|
| Fra Galgario NIR850, precisa | 0.214 | 0.202 | 0.206 | 0.210 | 0.210 |
| Fra Galgario UVF, precisa | 0.574 | 0.583 | 0.529 | 0.595 | 0.550 |
| Fra Galgario UVR, precisa | 0.323 | 0.318 | 0.353 | 0.340 | 0.343 |
| Fra Galgario NIR850, massima | 0.054 | 0.056 | 0.077 | 0.056 | 0.054 |
| Fra Galgario UVF, massima | 0.160 | 0.182 | 0.172 | 0.188 | 0.171 |
| affresco NIR, precisa | 0.239 | 0.242 | 0.403 | 0.240 | 0.284 |
| affresco UV, precisa | 1.017 | 0.862 | 0.978 | 0.994 | 1.221 |
| Fra Galgario SWIR, precisa | 0.996 | 0.834 | 1.337 | **3.243** | 0.945 |
| Fra Galgario RX Cappellochiaro, precisa | 1.069 | **3.008** | 1.113 | **14.495** | 2.211 |

**Finestre della griglia fitta migliorate / peggiorate di oltre 0.5 px** rispetto
alla predefinita, fra quelle misurabili in entrambe:

| immagini | finestre | densità fitta | densità larga | confidenza bassa | confidenza alta |
|---|---|---|---|---|---|
| bande fotografiche del Fra Galgario (5 coppie) | 2200 | 58 / 96 | 67 / 113 | 30 / 81 | 52 / 103 |
| affresco, NIR e UV | 1720 | 187 / 80 | 116 / 275 | 92 / 94 | 77 / 191 |
| SWIR | 140 | 43 / 16 | 20 / 55 | 14 / 103 | 19 / 29 |
| radiografia | 128 | 9 / 95 | 15 / 64 | 1 / 125 | 12 / 69 |

Che cosa se ne ricava:

- **Sulle bande fotografiche con buona struttura le due manopole non servono.**
  NIR850, UVF e UVR del Fra Galgario si muovono di 0.03 px al più a «precisa»
  e di 0.02 a «massima», qualunque valore si scelga; cambia di oltre mezzo
  pixel il 3–5% delle finestre, più spesso in peggio che in meglio.
- **Nessun valore diverso dal predefinito migliora tutte le coppie, e ognuno ne
  peggiora almeno una.** La predefinita è la migliore o entro 0.05 px dalla
  migliore in sette coppie su nove; nelle altre due la fitta fa meglio (0.16 px
  sull'UV dell'affresco e sullo SWIR).
- **Densità fitta**: è l'unica che a volte aiuta. Sull'affresco migliora più
  finestre di quante ne peggiori (187 contro 80) e sullo SWIR porta la mediana
  da 1.00 a 0.83 px. Ma sulla radiografia la porta da 1.07 a 3.01 px: con i
  blocchi da 96 px la cross-validation non trova più un guadagno e il campo
  fine non viene applicato.
- **Densità larga**: mai meglio della predefinita oltre 0.05 px, peggio su
  affresco NIR (0.24 → 0.40 px), SWIR e radiografia. Fa risparmiare circa il
  10% del tempo di stima.
- **Confidenza bassa**: sulle bande fotografiche non cambia nulla, e dove la
  banda ha poca struttura fa danno: 14.5 px sulla radiografia e 3.2 sullo SWIR,
  perché con i blocchi deboli ammessi il campo fine non supera la
  cross-validation e resta la sola parte grossolana. Il 5 ottobre sembrava
  utile sull'UV dell'affresco (finestre oltre 5 px dal 5% allo 0%); con i due
  livelli grossolani quel vantaggio non c'è più (2% contro 3%).
- **Confidenza alta**: nessun vantaggio in nessuna coppia; peggio sull'UV
  dell'affresco (191 finestre peggiorate contro 77) e sulla radiografia
  (2.2 px).
- **Tempo e memoria**: la memoria non cambia. Il tempo di stima cambia di circa
  il 10% con la densità (a «massima»: 124 s la fitta, 111 la media, 101 la
  larga) e non cambia con la confidenza.
- **Lato dei tile**: con 1024, 2048 e 4096 il file scritto è identico byte per
  byte (stessa impronta MD5 sul NIR850 del Fra Galgario). Cambia il picco di
  memoria: 0.80, 0.83 e 1.38 GB.

In pratica: si lasciano i valori predefiniti. L'unica prova che può valere la
pena è la densità fitta su una banda ottica difficile, guardando poi il
risultato; sulle radiografie no. Le due scelte di confidenza diverse dalla
normale e la densità larga, nelle misure di oggi, non hanno un caso d'uso.

Limiti: una corsa per configurazione; SWIR e radiografia sono una sola immagine
ciascuno, con 130–150 finestre misurabili; i tempi a «precisa» della
configurazione predefinita, girata sempre per prima, comprendono la prima
lettura dei file e non si confrontano con gli altri. La griglia fitta misura
con lo stesso strumento della stima, come la verifica del programma: è un
controllo di coerenza, non un confronto con punti indipendenti.

## Deformazioni locali

A precisione «precisa», con e senza il campo non rigido.

| immagine | con il campo | solo affine | tempo con / senza |
|---|---|---|---|
| Fra Galgario NIR850 | 0.214 px | 6.25 px, 69% oltre 5 px | 38 s / 18 s |
| Fra Galgario UVF | 0.574 px | 5.64 px, 59% oltre 5 px | 34 s / 17 s |
| affresco NIR | 0.239 px | 15.8 px, 94% oltre 5 px | 136 s / 43 s |
| affresco UV | 1.094 px | 34.6 px, 100% oltre 5 px | 66 s / 47 s |

La sola affine non allinea nessuna di queste immagini. Il campo va spento solo
se si vuole proprio una trasformazione rigida, sapendo quanto si perde.

## Scrittura: lato dei tile e formato

Scrittura dell'immagine allineata a piena risoluzione, dalla trasformazione
già stimata.

**Lato dei tile**, Fra Galgario NIR850, TIFF da 0.88 GB:

| tile | tempo | picco di memoria |
|---|---|---|
| 1024 | 16 s | 0.59 GB |
| 2048 | 14 s | 0.74 GB |
| 4096 | 13 s | 1.29 GB |

**Formato:**

| immagine | formato | tempo | picco di memoria | peso su disco |
|---|---|---|---|---|
| Fra Galgario NIR850 (314 Mpx) | TIFF | 14 s | 0.74 GB | 0.88 GB |
| | PNG | 32 s | 2.26 GB | 0.28 GB |
| | JPEG | 16 s | 2.26 GB | 0.06 GB |
| | TIFF, poi anche PNG | 14 + 19 s | 1.38 GB | |
| | TIFF, poi anche JPEG | 14 + 3 s | 1.38 GB | |
| affresco NIR (593 Mpx) | TIFF | 32 s | 0.83 GB | 1.66 GB |
| | PNG | 71 s | 3.82 GB | 0.83 GB |
| | JPEG | 33 s | 3.82 GB | 0.16 GB |
| | TIFF, poi anche PNG | 32 + 43 s | 2.16 GB | |
| | TIFF, poi anche JPEG | 32 + 7 s | 2.16 GB | |

- Il lato dei tile non cambia il tempo; cambia solo la memoria, e di poco.
- Il **TIFF** si scrive a fasce e non tiene mai l'immagine intera in memoria.
  **PNG e JPEG** vanno composti interi: servono 2.3–3.8 GB, e il PNG costa più
  del doppio del tempo per via della compressione.
- Se servono due formati, chiedere TIFF più l'altro è più leggero in memoria
  che chiedere solo l'altro (1.4 contro 2.3 GB sul Fra Galgario), perché il
  secondo si ricava dal file già scritto.
- A questi tempi, con ingressi PNG, va aggiunta la decodifica: circa 25 secondi
  ogni volta che un PNG da 1.7 GB di pixel viene aperto.

## Come parte la stima, e la portata del campo grossolano

Due confronti fatti a parte, 52 stime sugli stessi dipinti (dati in
`dati/confronto_partenza_portata.jsonl`).

### La portata: cambiata nel programma

Il campo grossolano delle tre scale vedeva fino a 128 px a «precisa» ma solo
fino a 64 px a «massima»: la precisione più alta era anche quella che vedeva
meno lontano. Dal 6 ottobre la portata è di almeno 128 px a qualunque
precisione (`block_matching.riduzione_per_il_grossolano`). A «precisa» e sotto
la stima è identica a prima per costruzione.

Effetto a precisione massima:

| immagine | portata 64 px (prima) | portata 128 px (ora) |
|---|---|---|
| affresco UV | 0.289 px, 16% oltre 5 px, massimo 101.5 px | 0.248 px, 0% oltre 5 px, massimo 1.8 px |
| affresco NIR | 0.049 px | 0.043 px |
| Fra Galgario NIR850 | 0.054 px | identico finestra per finestra |
| Fra Galgario NIR950 | 0.068 px | identico finestra per finestra |
| Fra Galgario UVF | 0.171 px | 0.160 px |
| Fra Galgario UVR | 0.086 px, 13% oltre 1 px, massimo 4.8 px | 0.092 px, 4% oltre 1 px, massimo 12.2 px |

Sull'UV dell'affresco le 8 finestre sbagliate di 23–102 px stavano dove la
deformazione vera, rispetto all'affine, è di 80–117 px su un asse: oltre i 64
px. Con la portata a 128 px scendono a 0.1–0.5 px. Sull'UVR 10 finestre
migliorano e 3 peggiorano; una di queste, a confidenza 0.08, passa da 1.7 a
12.2 px.

### La partenza: non cambiata

Tre modi di partire, a confronto:

1. **oggi**: una traslazione misurata su una miniatura, e il ripiego (posa
   affine stimata su anteprime molto ridotte) solo se la verifica trova più di
   10 px;
2. **traslazione corretta**: come oggi, ma con le due miniature ridotte dello
   stesso fattore. Oggi quella da allineare viene ridimensionata alle
   dimensioni del riferimento, e quando le due immagini hanno dimensioni
   diverse la traslazione esce sbagliata: sull'affresco di circa 200 px;
3. **posa grossolana subito**, senza tentare la traslazione.

| caso | oggi | traslazione corretta | posa grossolana subito |
|---|---|---|---|
| affresco NIR, massima | 0.043 px, 5.8 min | 0.043 px, 2.8 min | 0.043 px, 3.0 min |
| affresco UV, massima | 0.248 px, 0% oltre 5 px, 5.7 min | 0.327 px, 10% oltre 5 px, 2.7 min | 0.248 px, 0% oltre 5 px, 3.0 min |
| affresco NIR, precisa | 0.243 px, 2.2 min | 0.254 px, 2% oltre 5 px, 0.8 min | 0.243 px, 1.3 min |
| Fra Galgario UVR, massima | 0.092 px | 0.088 px | 0.100 px, 11 finestre peggiorate su 54 |
| Fra Galgario NIR950, precisa | 0.199 px, massimo 1.4 px | identico | 0.192 px, una finestra a 15.2 px |
| RX Cappellochiaro, precisa | 81 finestre su 225 | 52 su 224 | 52 su 224 |
| RX Testa chiara, precisa | 80 su 224 | identico | 58 su 224 |

(Tempi di stima e verifica; a massima con la portata a 128 px.)

- La **traslazione corretta** dimezza il tempo sul NIR dell'affresco, ma
  sull'UV dà un primo tentativo che riesce a metà (non abbastanza male da far
  scattare il ripiego) e lascia zone a 100 px; sulla radiografia Cappellochiaro
  trova una traslazione assurda, (1179, −768) px.
- La **posa grossolana subito** dà sull'affresco lo stesso risultato di oggi
  in metà tempo, ma sul Fra Galgario e sulle radiografie è un po' peggio.
- **Oggi** dà il risultato migliore o pari in tutti i casi. Il suo difetto è
  solo il tempo: quando il primo tentativo fallisce si butta una stima intera,
  5.8 minuti invece di 3.0 sull'affresco a massima.

Per questo la partenza non è stata toccata.

## Difetti emersi dalle misure, e che cosa ne è stato fatto

Le tabelle qui sopra sono state misurate **prima** delle correzioni 1, 2 e 5,
fatte il 6 ottobre. Dove una correzione cambia un numero della guida è detto
sotto.

### 1. Un primo tentativo fallito poteva non essere riconosciuto — corretto

Il ripiego scattava solo se, a stima finita, il residuo mediano superava 10 px.
Sul NIR dell'affresco con densità larga il primo tentativo era fallito già alla
prima passata (8 blocchi buoni su 88), ma la passata fine lo portava a 3.9 px e
il programma lo teneva, con il 48% delle finestre oltre 5 px. E quando il
fallimento veniva riconosciuto, lo era solo dopo aver completato una stima
inutile.

Ora, se alla prima passata i blocchi buoni sono meno del 25% delle posizioni,
il programma controlla subito: verifica a piena risoluzione l'affine appena
trovata e quella della posa grossolana, e se la prima sbaglia più di tre volte
la seconda parte dalla posa grossolana senza completare la stima
(`align_paintings._scegli_partenza`).

La quota di blocchi buoni da sola non bastava a decidere: sulle radiografie è
dell'1.3–1.5% sia dove l'aggancio riesce (a «precisa») sia dove fallisce (a
«massima»). Il confronto fra le due affini invece separa i casi, misurato su
30 coppie (`dati/aggancio_criterio.jsonl`):

| caso | affine dalla traslazione | affine della posa grossolana | rapporto |
|---|---|---|---|
| affresco NIR, precisa (fallito) | 208.6 px | 16.5 px | 12.6 |
| affresco NIR, massima (fallito) | 196.5 px | 16.5 px | 11.9 |
| RX Cappellochiaro, massima (fallito) | 140.9 px su 5 finestre | 15.6 px su 26 | 9.0 |
| RX Testa chiara, massima (fallito) | 130.4 px su 6 finestre | 16.5 px su 20 | 7.9 |
| affresco UV, massima (fallito) | 171.8 px | 32.7 px | 5.3 |
| RX Cappellochiaro, precisa (riuscito) | 29.6 px | 15.6 px | 1.9 |
| RX Testa chiara, precisa (riuscito) | 19.9 px | 16.5 px | 1.2 |
| gli altri 23, bande fotografiche e SWIR | 5.5–35.7 px | 6.0–32.7 px | 0.9–1.5 |

Effetto, con i PNG veri dell'affresco e i TIFF del Fra Galgario:

| caso | prima | dopo |
|---|---|---|
| affresco NIR, precisa, densità larga | 3.888 px, 48% oltre 5 px | 0.361 px |
| affresco NIR, precisa, confidenza alta | errore | 0.236 px su 64/64 |
| affresco NIR e UV, massima | 8.0–8.3 min per immagine | 5.2–5.5 min, stesso risultato |
| RX Cappellochiaro, massima | 0.421 px in 225 s | 0.421 px in 158 s |
| RX Cappellochiaro e Testa chiara, precisa | 81/225 e 80/224 finestre, 45 s | identiche, 54–59 s |
| Fra Galgario NIR850 e UVR, precisa | 0.214 e 0.323 px | identici, stessi tempi |

Il costo è sulle radiografie a «precisa»: il controllo scatta, conferma la
partenza di sempre e la prima passata viene rifatta, 10–15 secondi in più.
Per le bande fotografiche (28–88% di blocchi buoni) non scatta e non cambia
niente. Con i punti o con il bordo la posa la dà chi usa il programma, e il
controllo non c'è.

Nella tabella della densità, quindi, la riga «affresco NIR, larga» oggi dà
0.361 px invece di 3.888; i tempi dell'affresco a massima scendono di circa un
terzo.

### 2. Con la confidenza alta la stima poteva interrompersi — corretto

Sul NIR dell'affresco a 0.20 il primo tentativo non trovava blocchi a
sufficienza e sollevava un errore («Solo 1 blocchi affidabili») che fermava
l'immagine prima di arrivare al ripiego. Ora quel caso passa dallo stesso
controllo del punto 1: 0.236 px su 64 finestre su 64. L'alta resta comunque
sconsigliata: non ha dato vantaggi in nessuna misura.

### 3. La traslazione grossolana è sbagliata con immagini di dimensioni diverse — lasciata

Vedi «La partenza». Corretta da sola fa più danno che altro; con il controllo
del punto 1 il suo errore non costa più una stima a vuoto né un risultato
sbagliato.

### 4. La stima di tempo nella finestra — tolta

Indicava circa la metà del vero a precisione massima. Resta la stima della
memoria, che torna (10.2 GB previsti, 11.0 misurati sull'affresco a massima).

### 5. Per PNG e JPEG il fattore di riduzione usciva non intero — corretto

`image_io.leggi_ridotta` riduce per un fattore intero scartando le righe e le
colonne che avanzano, ma restituiva altezza originale diviso altezza ridotta:
8003 / 4001 = 2.00025 invece di 2. Per i TIFF non compressi il fattore
restituito è sempre stato l'intero. Quando riferimento e immagine da allineare
hanno lo stesso resto l'errore si cancella; quando hanno resti diversi diventa
un errore di scala nella trasformazione finale.

Misurato sugli stessi pixel letti da TIFF e da PNG, riferimento 6001 × 8003 px
e immagine 6000 × 8000 (`dati/fattore_di_riduzione.jsonl`):

| precisione | da TIFF | da PNG, prima | da PNG, dopo | distanza fra le affini all'angolo opposto, prima → dopo |
|---|---|---|---|---|
| precisa | 0.222 px | 0.730 px | 0.222 px | 1.25 → 0.00 px |
| veloce | 0.564 px | 1.946 px | 0.564 px | 3.75 → 0.00 px |

Il JPEG, che perde informazione, sta entro 0.03–0.06 px dal TIFF a «precisa» e
«veloce».

Sull'affresco i tre PNG hanno lati multipli di 2 e di 4, quindi le corse a
«precisa» e «veloce» non erano toccate; lo era la posa grossolana (fattore
12.0024 invece di 12), che la stima successiva correggeva. Dopo la correzione
le corse sui PNG danno esattamente i numeri delle copie TIFF di questa guida:
0.239 px il NIR a «precisa», 0.295 px l'UV a «massima». I valori del confronto
su partenza e portata, fatto sui PNG prima della correzione (0.043 e 0.248 px a
massima), differiscono da questi per quel motivo.

## I file

- `dati/impostazioni.jsonl`: una riga per configurazione, con tempi, memoria,
  esito della verifica, le 64 finestre una per una.
- `dati/impostazioni_tabelle.txt`: le stesse in forma di tabella, con il
  confronto sulle finestre comuni rispetto alle impostazioni predefinite.
- `dati/scrittura.jsonl`: tempi, memoria e peso della scrittura.
- `dati/avanzate_2026-10-07.jsonl`, `…_tabelle.txt` e `…_tile.jsonl`: le 45
  configurazioni del 7 ottobre su densità e confidenza (con la griglia fitta
  finestra per finestra) e la prova sul lato dei tile. Script:
  `imp2_driver.py`, `imp2_worker.py`, `imp2_analisi.py`, `tile_worker.py`.
- `dati/confronto_partenza_portata.jsonl` e `…_tabelle.txt`: le 52 stime del
  confronto su partenza e portata.
- `dati/portata_UV.txt`: la misura sull'UV dell'affresco che ha portato a
  cambiare la portata.
- `dati/aggancio_criterio.jsonl`: le 30 coppie su cui è stato scelto il
  criterio per riconoscere un aggancio fallito.
- `dati/fattore_di_riduzione.jsonl`: TIFF contro PNG e JPEG, prima e dopo la
  correzione del fattore di riduzione.
- `script/`: gli script che hanno prodotto i dati. Sono stati lanciati da una
  cartella di lavoro temporanea e contengono i percorsi delle immagini di
  questa macchina: prima di rilanciarli vanno adattati. `imp_driver.py` lancia
  lo studio delle impostazioni, `confronto_generale.py` il confronto su
  partenza e portata, `scrittura_driver.py` le misure di scrittura.
