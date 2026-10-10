# Allineamento di immagini enormi — guida

## Installazione

Serve Python (provato con la versione 3.13) con le librerie elencate in
`requirements.txt`: NumPy, OpenCV, SciPy e Pillow.

```powershell
pip install -r requirements.txt
python interfaccia.py
```

Gli script delle misure e delle figure vogliono anche SimpleITK e Matplotlib
(`requirements-misure.txt`). Il programma è distribuito con licenza MIT: vedi
`LICENSE`.

## Struttura della cartella

Tre programmi, ognuno nella sua cartella, con le tre finestre fuori, nella
cartella del progetto, più i dati comuni:

```
Aligner_BETA/
├── interfaccia.py                FINESTRA: allinea (riferimento, immagini, cartella)
├── interfaccia_falsocolore.py    FINESTRA: componi IRFC e UVFC
├── interfaccia_mosaico.py        FINESTRA: ricomponi un dipinto ripreso a tasselli
├── Allineatore/
│   ├── align_batch.py            piu' immagini da riga di comando
│   ├── align_paintings.py        allineamento di una singola coppia
│   ├── apply_transform.py        riapplica i parametri alla stessa ripresa
│   ├── finestra_strumenti.py     punti e bordo tracciato per UNA immagine del lotto
│   ├── verifica_fissa.py         la verifica a griglia fissa, per tutte le bande
│   ├── finestra_punti.py         le due viste per mettere i punti a mano
│   ├── finestra_maschera.py      il disegno della maschera del soggetto
│   ├── punti_manuali.py          posa di partenza dai punti (similarita' / affine)
│   ├── block_matching.py         stima della trasformazione            ┐
│   ├── transform_model.py        modello della trasformazione          │ il MOTORE,
│   ├── image_io.py               lettura di qualsiasi formato          │ usato anche
│   ├── tiff_memmap.py            via veloce a memmap per i TIFF        │ dagli altri
│   ├── validation.py             verifica a piena risoluzione          │ due programmi
│   ├── robust_metrics.py         MIND / SSC / NGF / NMI                ┘
│   ├── formati.py                i formati di uscita (TIFF, PNG, JPEG)
│   ├── test_pipeline.py          test automatico su dati sintetici
│   ├── calibration_setup.py      calibrazione camera da scacchiera (indipendente)
│   └── benchmark/                le misure: script, risultati e tabelle (vedi «Le misure finali»)
├── FalsoColore/
│   ├── falso_colore.py           IRFC e UVFC da immagini gia' allineate
│   └── test_falsocolore.py       la composizione per canali contro quella in decimali
├── Mosaico/
│   ├── mosaico.py                motore del mosaico da carrello, anche da riga di comando
│   ├── mosaico_omografia.py      mosaico da cavalletto: omografie, immagine totale
│   ├── test_mosaico.py           test del mosaico su tasselli sintetici
│   └── test_mosaico_omografia.py test del mosaico da cavalletto, con le pose note
├── Documentazione/
│   ├── impostazioni/             guida alle impostazioni avanzate, con dati e script
│   └── schemi/                   schemi e figure, con gli script che li disegnano
├── immagini/                     le riprese originali, comuni a tutti   ┐ fuori da git:
└── risultati/                    uscite di tutti e tre i programmi      ┘ si creano all'uso
```

Le riprese non stanno nel repository. Gli script delle misure le cercano in
`immagini/` e, se manca, nella cartella scritta in testa a ciascuno script.

Le finestre stanno fuori dalle sottocartelle perché sono ciò che si apre: si
trovano tutte e tre a colpo d'occhio, senza dover ricordare in quale cartella
sta ciascuna. Il codice che fanno girare resta nella cartella del rispettivo
programma, e ogni finestra la mette da sola sul percorso di Python.

Il motore (lettura delle immagini, block matching, trasformazioni, verifica)
sta in `Allineatore/`, ed è lì che lo prendono anche il falso colore e il
mosaico: ciascuno mette la cartella accanto `Allineatore/` sul percorso di
Python, senza copiarne nulla. Una correzione al motore vale quindi per tutti e
tre, e cartelle e finestre vanno tenute insieme.

I comandi di questa guida si danno dalla cartella del progetto (per esempio
`python interfaccia.py`), ma i programmi trovano i loro moduli da qualunque
cartella li si lanci.

## Uso rapido — interfaccia

```powershell
python interfaccia.py
```

Finestra unica: si sceglie l'**immagine di riferimento**,
si aggiungono le **immagini da allineare (immagine target)** (quante si vuole, anche di formati e
dimensioni diversi), si indica la **cartella di salvataggio** e si avvia. Durante
l'elaborazione la barra mostra a che punto è; alla fine una tabella riporta, per
ogni immagine, il disallineamento **prima** e **dopo**, in verde se è andata
bene, in arancione se il residuo è sopra soglia e va controllata a occhio, in
rosso se è fallita.

Non serve dire al programma a che lunghezza d'onda corrisponde ogni immagine: il
metodo lavora sulla struttura (bordi, craquelure, pennellate) e non sui livelli
di grigio.

**Formati di uscita.** In *Impostazioni avanzate* il menu «Formati delle
immagini salvate» ha una crocetta per TIFF, PNG e JPEG, e se ne possono
spuntare più d'uno. Senza spunte si salva solo nel formato dell'originale. Il
primo file si scrive a tile, con la memoria costante: è il TIFF, se è fra quelli
scelti. Gli altri si ricavano poi da quel file. PNG e JPEG richiedono invece
l'immagine intera in memoria (circa 1 GB per banda a 8 bit), e il JPEG scende a
8 bit e comprime con perdita. Il falso colore ha lo stesso menu; lì, senza
spunte, si salva solo il TIFF. Da riga di comando: `--formati TIFF PNG`.

**Le radiografie passano dal lotto come le altre bande** (dal 6 ottobre 2026).
Prima avevano un riquadro loro, *Radiografie (RX) — a parte*; poi si è visto che
gli strumenti di quella sezione non sono una questione di banda: anche l'UV di un
affresco può restare fuori ai bordi. Ora:

- **Strumenti a mano…** (sopra la lista): per l'immagine selezionata si possono
  mettere dei **punti di corrispondenza** (di norma solo per misurare il TRE; a
  scelta come vincolo locale o come posa di partenza) e tracciare il **bordo**
  del dipinto su quell'immagine, che insieme alla zona disegnata sul riferimento
  dà la *fascia di bordo* (150 px). Punti e bordo di una corsa precedente si
  ricaricano da soli. Un'immagine senza strumenti va in automatico, con la stessa
  chiamata di sempre: i suoi numeri non cambiano (verificato: suite sintetica
  identica; NIR850 del Fra Galgario a «veloce» 0,920 px prima e dopo).
- **Rifai la selezionata**: rifà solo quell'immagine, per lavorare a tentativi.
  La riga nuova si aggiunge sotto le altre, e gli avvisi dicono se è meglio o
  peggio del tentativo precedente, sulle finestre misurate in entrambi.
- **Griglia fissa per tutte**: accanto alla verifica di sempre, che resta il
  numero principale, ogni immagine ha la verifica a griglia fissa
  (`verifica_fissa.py`): mediana, finestre misurabili, finestre oltre 10 px.
  Selezionando una riga dei risultati, sotto la tabella si legge ciò che la
  mediana non dice. Costa 10–40 s in più per immagine.

L'unica sezione a parte rimasta è il falso colore. Dove nel resto di questo
documento si legge «sezione RX», si intendono oggi gli strumenti a mano
dell'immagine: la storia delle misure è rimasta com'era. Vedi
[Il caso delle radiografie](#il-caso-delle-radiografie-e-delle-bande-difficili).

Misurato prima di unire (UV dell'affresco a «precisa», griglia fitta dentro la
zona): con i due livelli grossolani già in funzione, la fascia di bordo **non
migliora** le finestre vicine al contorno (entro 500 px: oltre 5 px dall'8% al
12%; entro 1500 px, sulle 194 finestre comuni, 3 migliorate e 7 peggiorate di
oltre 1 px), pur portando i due contorni da 22,5 a 1,2 px l'uno dall'altro. Per
questo resta spenta finché non la si chiede, immagine per immagine.

## Falso colore (IRFC e UVFC)

```powershell
python interfaccia_falsocolore.py
```

Compone le due rappresentazioni standard a partire da immagini **già allineate**:

| | R | G | B |
|---|---|---|---|
| **IRFC** | IR | R(VIS) | G(VIS) |
| **UVFC** | G(VIS) | B(VIS) | UV |

I canali scorrono in direzioni opposte: l'IRFC scarta il blu del visibile,
l'UVFC scarta il rosso. Per l'UVFC serve l'**ultravioletto riflesso (UVR)**, non
la fluorescenza (UVF): la fluorescenza è emissione del materiale, non
riflessione della banda, e si guarda com'è.

Serve anche da **controllo di registrazione**: un disallineamento residuo
appare come frangia colorata sui bordi, e l'occhio la coglie molto più in fretta
che in una scacchiera. Sul Fra Galgario il dipinto risulta pulito fino al bordo,
mentre lo sfondo (cavi, attrezzatura) mostra frange vistose — è la parallasse
già misurata, resa visibile in un colpo d'occhio.

**Sulla radiometria.** I canali vengono composti con i livelli come sono stati
ripresi, senza nessun aggiustamento, e il programma stampa quanto i canali di
ogni banda differiscono fra loro. C'era un'opzione *Bilancia* che stirava i
livelli fra due percentili: rendeva l'immagine più leggibile ma non
confrontabile con le tavole di riferimento dei pigmenti, e non veniva usata,
quindi è stata tolta.

**Come compone.** Un falso colore è solo uno spostamento di canali: l'IR va nel
rosso, il rosso del visibile nel verde, e così via. Il programma lo fa con
numeri interi, e dove serve porta i 16 bit a 8 con una tabella precalcolata,
invece di passare ogni valore per un decimale e tornare indietro. Il file è
**identico bit per bit** a quello della via decimale: verificato su tutti i
valori possibili, sulle riprese del Fra Galgario (NIR850, UVR, SWIR a 16 bit) e
contro la versione precedente del programma. Il tempo si dimezza: tre bande
insieme passano da 48–49 s a 21–25 s, e ciò che resta è quasi tutto lettura e
scrittura su disco. Il controllo si rifà con
`python FalsoColore\test_falsocolore.py`.

Da riga di comando:

```powershell
python FalsoColore\falso_colore.py --vis ...\VIS.tif ^
                                   --ir ...\NIR850_allineata.tif ^
                                   --uv ...\UVR_allineata.tif ^
                                   --out-dir risultati\falsocolore
```

## Mosaico di tasselli (in costruzione)

```powershell
python interfaccia_mosaico.py
```

Per i dipinti ripresi a tasselli: la camera (o il dipinto) si sposta su una
griglia di righe e colonne, e ogni scatto si sovrappone ai vicini. Si indicano
la cartella dei tasselli, righe e colonne, il **percorso** (serpentina o
righe/colonne sempre nello stesso verso) e l'angolo del primo scatto. I file
vengono presi in ordine naturale dei nomi (`scatto_2` prima di `scatto_10`).
Prima di avviare, lo schema della finestra mette le miniature vere nelle
caselle: un percorso sbagliato non dà errori, dà tasselli nel posto sbagliato,
e lì si vede subito.

Il programma **non fonde** i tasselli. Li posiziona, al più con una
trasformazione affine, e li scrive come **livelli**: un TIFF per tassello, già
ricampionato nel sistema del mosaico, con la trasparenza fuori dal suo contorno
e il profilo colore dell'originale. Lo script `apri_in_photoshop.jsx` li apre
tutti in un documento alle coordinate giuste (in Photoshop: *File > Script >
Sfoglia...*). Il documento prende profilo, modo e profondità dal primo livello,
così nessun livello viene convertito. Taglio, raccordo e colore restano da fare
in Photoshop.

Da riga di comando:

```powershell
python Mosaico\mosaico.py --cartella ...\tasselli --righe 3 --colonne 4 ^
                          --percorso serpentina_righe --partenza alto_sinistra ^
                          --out-dir risultati\mosaico
```

Nella cartella di uscita: `livelli/`, lo script per Photoshop, `mosaico.json`
(la matrice tassello → mosaico di ogni tassello, le coppie, la verifica) e
`anteprima_mosaico.png`, con i numeri, il percorso e l'esito di ogni cucitura.

**Come funziona.** Si misurano *tutte* le coppie di caselle adiacenti, non solo
quelle consecutive nello scatto. Con le sole consecutive si ottiene una catena, e
gli errori si sommano anello dopo anello. Con anche le coppie trasversali, ogni
riquadro di quattro tasselli chiude un anello: una stima globale ai minimi
quadrati distribuisce gli errori, e un aggancio sbagliato si riconosce perché
non torna con gli altri. È l'impostazione dello Stitching di Fiji e di MIST
(NIST). Per ogni coppia ci sono un aggancio grossolano su anteprime ridotte e
poi corrispondenze fini nella sola fascia comune, con lo stesso block matching
dell'allineatore. Una verifica a piena risoluzione misura, lungo ogni cucitura,
di quanti pixel due tasselli vicini restano fuori registro.

Il **modello** (traslazione, rigida, similarità, affine) si sceglie misurando: si
passa a uno più libero solo se riduce lo scarto su corrispondenze tenute da
parte di almeno il 10%, come per il campo non rigido dell'allineatore, **e** di
almeno 0.05 px. Il minimo assoluto viene da una misura. Su tasselli sintetici
con quattro verità diverse (traslazione, rigida, similarità, affine) e tre
combinazioni di riduzione, quando la scelta era sbagliata il guadagno era di
0.016–0.026 px, quando era giusta di 0.19 px e oltre. Con il solo 10% si
sbagliavano 3 casi su 12, con il minimo nessuno. La soglia va ricontrollata sui
primi mosaici veri.

Provato e scartato: tenere da parte coppie intere invece di punti. È una prova
più indipendente, ma non sceglie mai l'affine, nemmeno quando è vera (sceglie
la similarità, con 1.1 px di errore vero contro 0.14), perché da una fascia
stretta la deformazione di un tassello si estrapola male fino alla cucitura
opposta.

**Cosa è verificato, e solo su dati sintetici.** Il test (`python Mosaico\test_mosaico.py`) usa una griglia 3 × 4 con passo
sbagliato fino a 20 px, rotazioni fino a 0.4°, colore, esposizione e
vignettatura diversi per tassello. L'errore di posizione relativo agli angoli
dei tasselli è di 0.045 px mediano e 0.16 px massimo alla piena risoluzione, e
di 0.13 / 0.28 px al fattore 2. Il modello rigido viene riconosciuto da solo, e
i livelli riaperti cadono sul dipinto entro 0.09 px. Un tassello senza nessuna
struttura viene segnalato e messo al suo posto dal passo tipico del carrello.
**Non è verificato** lo script per Photoshop, che non ho potuto aprire in
Photoshop: la prima prova va fatta a mano.

Due cose emerse misurando, che valgono oltre questo programma:

- **Le cuciture perfette non garantiscono un mosaico fedele.** Con il modello
  affine le cuciture stanno a 0.04 px come con il rigido, ma l'errore vero sale
  da 0.045 a 0.47 px mediano e 1.5 px massimo: i gradi di libertà in più
  deformano il mosaico in modo coerente, senza lasciare traccia nelle
  sovrapposizioni. Per questo il modello più semplice è preferito. Per lo
  stesso motivo, un'affine scelta in automatico con scale lontane da 1 è quasi
  sempre il segno della distorsione dell'obiettivo, che va corretta a monte.
- **`cv2.phaseCorrelate` sbaglia di mezzo pixel su certi lati di finestra**,
  quelli per cui OpenCV allarga la FFT a una dimensione dispari (44 → 45, 74 →
  75, 402 → 405). Misurato: +0.3/+0.6 px di scarto sistematico contro −0.1/+0.1
  per i lati vicini. Nel mosaico una fascia stretta dava blocchi da 44 px e una
  cucitura sbagliata di 1.4 px; ora i lati vengono scelti fra quelli che OpenCV
  usa così come sono.

Le posizioni trovate valgono per quei file. Le riprese di un'altra banda sulla
stessa griglia si ricompongono per conto loro, e i mosaici delle bande si
allineano poi fra loro con l'allineatore.

### Da cavalletto: un'omografia per tassello (ottobre 2026)

Il primo mosaico vero ha mostrato che cosa `mosaico.py` dà per scontato. Un
affresco staccato, ripreso nel visibile con 27 scatti da 36 Mpx (Nikon D810,
170 mm), **dal cavalletto**: la camera ruota fra uno scatto e l'altro invece di
traslare. I tasselli sono ruotati fino a 5°, scalati fino al 9%, le righe sono
sfalsate fino a 2500 px e due scatti sono doppi. Su quei file `mosaico.py`
aggancia 27 coppie su 40 e lascia le cuciture a 6,3 px di mediana, 215 al
massimo.

Per quei casi c'è un modulo a parte, `Mosaico\mosaico_omografia.py`; il
programma da carrello resta com'è. Nella finestra si sceglie con **Ripresa: da
carrello / da cavalletto**. Da riga di comando:

```powershell
python Mosaico\mosaico_omografia.py --cartella ...\tasselli --out-dir ...\mosaico ^
                                    --linee linee.json --raddrizza
```

Che cosa cambia:

- **Niente griglia e niente percorso.** I vicini si trovano con punti
  caratteristici (SIFT) e RANSAC fra tutte le coppie di tasselli. L'ordine dei
  file non conta, uno scatto doppio è un tassello in più (nell'immagine totale
  entra solo il più nitido), e le coppie in diagonale chiudono altri anelli.
- **Misura fine sul vicino già riportato.** La correlazione di fase a blocchi è
  quella di sempre, ma dopo aver portato il vicino sul tassello con l'omografia
  dei punti caratteristici: resta da misurare qualche pixel anche fra scatti
  ruotati e scalati.
- **Un'omografia per tassello**, che per una superficie piana vista da punti
  diversi è la relazione esatta; l'affine vale solo per il carrello.
- **La distorsione dell'obiettivo**, un coefficiente radiale comune a tutti i
  tasselli, adottata solo se lo scarto fuori campione scende di almeno il 10%.
- **L'immagine totale**, scritta a bande con memoria costante, con il profilo
  colore dei tasselli. In ogni punto vale il tassello di cui quel punto è più
  interno, con una dissolvenza lungo i confini: 512 px se le cuciture sono
  sotto il pixel, 32 px altrimenti. Luminosità e colore non vengono raccordati.

Sul visibile dell'affresco (pixel a piena risoluzione):

| | `mosaico.py` (griglia 5 × 5) | `mosaico_omografia.py` |
|---|---|---|
| Coppie misurate | 27 su 40 | 77 su 77 |
| Scarto fuori campione, affine per tassello | — | 6,58 px |
| Scarto fuori campione, omografia | — | 3,40 px |
| Scarto fuori campione, omografia + distorsione | — | 0,29 px |
| Cuciture a piena risoluzione, mediana | 6,34 px | 0,17 px |
| Cuciture, 90° percentile | — | 0,43 px |
| Cuciture, massimo | 215 px | 1,21 px |

Il grosso dello scarto era la **distorsione dell'obiettivo**: −21 px sul bordo
del tassello (0,6%, a cuscinetto), che nessuna omografia segue. Stimarla porta
lo scarto da 3,4 a 0,29 px. La stima impiega 6 minuti, la scrittura di
un'immagine da 28 800 × 20 200 px altri 5–7.

Le altre due bande dello stesso affresco, ricomposte allo stesso modo:

| | Visibile | NIR | Fluorescenza UV |
|---|---|---|---|
| Scatti (di cui doppi, esclusi dall'immagine) | 27 (2) | 29 (4) | 29 (4) |
| Coppie misurate | 77 su 77 | 86 su 86 | 82 su 82 |
| Distorsione sul bordo del tassello | −21 px | −21 px | −93 px |
| Cuciture, mediana | 0,17 px | 0,10 px | 0,22 px |
| Cuciture, 90° percentile | 0,43 px | 0,34 px | 0,64 px |
| Cuciture, massimo | 1,21 px | 1,02 px | 2,23 px |
| Bordi del supporto misurati | 4 | 4 | 3 |
| Scarto dei bordi dalla retta (rms) | 4,7 px | 6,5 px | 6,6 px |
| Raddrizzato: scarto dall'angolo retto | 0,14° | 0,06° | non fatto |

Visibile e NIR danno la stessa distorsione, come ci si aspetta dallo stesso
obiettivo; la fluorescenza ne dà più di quattro volte tanto, e senza stimarla
lo scarto fuori campione resterebbe a 10,6 px invece di 0,33. Sulla
fluorescenza il bordo basso del supporto non si distingue (5 punti), quindi
manca una delle quattro linee e la versione raddrizzata non si fa. I tre
mosaici sono indipendenti, ma concordano: tolta un'omografia, il NIR differisce
dal visibile di 3,8 px in mediana e la fluorescenza di 6,4; e le due versioni
raddrizzate, ognuna dalle proprie linee, coincidono a meno di una similarità
entro 11 px di mediana.

**Il freno sui termini prospettici.** Si era pensato di frenare i due termini
prospettici di ogni omografia, che sono i meno vincolati e i più pericolosi da
propagare. Misurato, il freno fa il contrario di ciò che ci si aspettava. Su
tasselli sintetici con le pose note (errore di posizione vero nel punto
peggiore):

| Ripresa | Misure | Affine | Omografia libera |
|---|---|---|---|
| Cavalletto | esatte | 35 px | 2 px |
| Cavalletto | rumore 0,5 px | 37 px | 22 px |
| Carrello | esatte | 0,1 px | 0,4 px |
| Carrello | rumore 0,5 px | 2,4 px | 18 px |

Quando la camera ruota, i termini prospettici non sono zero e tirarli verso
zero deforma il mosaico, con cuciture che sembrano ancora buone (0,8 px). Il
freno aiuta solo quando la ripresa è in realtà da carrello. Sui bordi veri del
supporto dell'affresco, cercati tassello per tassello e confrontati con una
retta:

| Freno | Scarto dalla retta (rms) | Freccia massima |
|---|---|---|
| Infinito (affine per tassello) | 59,7 px | 385 px |
| 0,001 | 25,6 px | 137 px |
| 0,0001 | 6,9 px | 22 px |
| 0,00001 (scelto) | 4,7 px | 9 px |
| Nessuno (omografia libera) | 4,6 px | 8 px |

In automatico si usa quindi il freno più forte fra quelli che alle cuciture non
costano nulla (entro l'1% dal migliore): su una ripresa da cavalletto vuol dire
quasi nessuno. Ciò che il freno non toglie è la deriva dovuta al rumore delle
misure (seconda riga della prima tabella), che dalle cuciture non si vede.

**La rettilineità.** Le cuciture dicono se i vicini combaciano, non se il
mosaico ha la forma vera. Per quello serve qualcosa di dritto nella realtà: un
file JSON con delle linee, ognuna data da due estremi in pixel di un tassello.
Il programma cerca il bordo lungo ogni linea, tassello per tassello, e riporta
lo scarto dalla retta e la **freccia** (di quanto la linea si incurva). Le linee
si indicano a mano; per ora si scrivono nel file, una finestra per disegnarle
non c'è. Sull'affresco i quattro bordi del supporto restano dritti entro 5 px
su linee lunghe fino a 23 000 px (un bordo d'intonaco non è dritto al pixel
nemmeno nella realtà); il bordo basso, coperto in parte dai sostegni del
cavalletto, dà solo 19 punti ed è il meno affidabile. Vicino al margine di un
tassello la ricerca usa il tratto di profilo che cade dentro: scartare il
campione lasciava senza misura proprio i bordi del supporto, che stanno quasi
sempre vicino al margine degli scatti. Le linee indicate su una banda si portano
sulle altre con i punti caratteristici, tassello con tassello (visibile → NIR)
o mosaico con mosaico quando le due bande si somigliano poco (visibile →
fluorescenza). Il controllo ha un limite: una vista di sbieco lascia dritte le
rette, e non la vede.

**Il raddrizzamento.** Il mosaico nasce nel piano del sensore dello scatto
centrale: è il dipinto come lo vedrebbe una sola fotografia fatta da lì. Se in
quello scatto la camera non era perpendicolare al dipinto, i lati paralleli
convergono, e nessuna misura fra tasselli può accorgersene. Con `--raddrizza` si
scrive **anche** una seconda immagine, rimessa di fronte al dipinto: si assume
che le linee di una stessa direzione siano parallele, se ne ricavano i punti di
fuga e, con la focale, l'inclinazione del piano. La focale viene dai tasselli
stessi: per una camera che ruota il termine prospettico è proporzionale alla
posizione, e sull'affresco lo è con correlazione −0,999 (focale 39 440 px,
192 mm: i 170 nominali allungati dalla messa a fuoco ravvicinata).
Sull'affresco il dipinto risulta inclinato di 4,9° rispetto al sensore dello
scatto centrale, e dopo il raddrizzamento fra le due direzioni mancano 0,14°
all'angolo retto. Quest'ultimo numero non è imposto, è misurato: è piccolo, e
dice che il supporto è un rettangolo visto di sbieco. Se fosse grande, il
raddrizzamento non andrebbe creduto. La prima immagine resta comunque senza
nessuna ipotesi. Servono due linee per direzione: con tre linee in tutto,
assumendo l'angolo retto, il risultato si è rivelato inaffidabile (sulla
fluorescenza differiva dal visibile raddrizzato di 57 px in mediana e 320 al
massimo) e la scorciatoia è stata tolta.

**Cosa è verificato.** Il test (`python Mosaico\test_mosaico_omografia.py`)
fabbrica tasselli con pose note e misura l'errore di posizione vero, senza
togliere trasformazioni d'insieme: 0,035 px mediano e 0,24 px massimo su 13
tasselli da cavalletto; il carrello tiene il freno tirato; la distorsione viene
ritrovata (5,00 px su 5); l'immagine totale riletta cade sul dipinto entro
0,09 px. Con una camera vera messa fuori asse, la focale implicita torna entro
lo 0,3%, l'inclinazione entro 0,03°, e l'errore di forma scende da 30 a 0,4 px
dopo il raddrizzamento. Con la distorsione l'errore vero agli angoli lontani
arriva a 4,6 px pur con cuciture a 0,03 px: è il limite del metodo, e il motivo
per cui serve un controllo esterno come la rettilineità.

**Cosa non fa.** Non scrive i livelli per Photoshop e non tocca il colore: i
tasselli hanno esposizioni un po' diverse, e nell'immagine totale la differenza
resta, distesa dalla dissolvenza. Il confronto con i mosaici dello stesso
affresco cuciti con un altro programma dà 11–17 px di differenza mediana, banda
per banda, anche dopo aver tolto un'omografia. Anche quelli concordano fra loro
da una banda all'altra (5–6 px), quindi sono due famiglie coerenti al loro
interno e diverse fra loro: quale sia più fedele, senza misure prese
sull'opera, non si può dire.

Una misura che riguarda l'allineatore più che il mosaico: fra due bande dello
stesso affresco, riprese dal cavalletto, la differenza d'insieme è una
**prospettiva**, non un'affine. Sui totali già cuciti, tolta un'affine restano
23 px in mediana fra NIR e visibile (36 vicino ai bordi) e 46 fra fluorescenza
e visibile (114 vicino ai bordi); tolta un'omografia ne restano 5 e 6, bordi
compresi. È il motivo per cui su quelle immagini l'allineatore, che parte da
un'affine, deve recuperare ai bordi centinaia di pixel con il campo.

## Uso rapido — riga di comando

```powershell
# piu' immagini sullo stesso riferimento
python Allineatore\align_batch.py --fixed -percorso_immagine_riferimento ^
                                  --immagini -percorso_immagine_bersaglio ^
                                  --out-dir risultati_lotto

# una sola coppia (stima + verifica + immagine allineata)
python Allineatore\align_paintings.py --fixed -percorso_immagine_riferimento ^
                                      --moving -percorso_immagine_target ^
                                      --out -percorso_immagine_allineata ^
                                      --params -percorso_parametri ^
                                      --diagnostica -percorso_diagnostica

# solo stima e verifica, senza scrivere ~1 GB su disco (utile per provare opzioni)
python Allineatore\align_paintings.py --fixed -percorso_immagine_riferimento ^
                                      --moving -percorso_immagine_target

# riapplicare i parametri salvati a un raster della STESSA ripresa nella stessa lunghezza d'onda
# (una maschera, delle ROI, una ri-esportazione a 16 bit), o rigenerare l'output
python Allineatore\apply_transform.py --params ...\transform_params.json ^
                                      --image ...\maschera.tif ^
                                      --out ...\maschera.tif ^
                                      --reference -percorso_immagine_riferimento

# test automatico su dati sintetici con trasformazione nota
python Allineatore\test_pipeline.py
```

Il campo di deformazione sta in `transform_params_field.npy`, accanto al JSON:
se sposti i parametri, **sposta i due file insieme**.

Se è stata usata una maschera, il JSON la descrive in `meta.maschera`: la forma
disegnata (poligono, rettangolo o ellisse) con i vertici in frazioni di
larghezza e altezza, e la copertura. Accanto c'è anche il raster esatto usato
per la stima, `transform_params_maschera.png`. Senza maschera il campo vale
`null`, così "nessuna maschera" non si confonde con "maschera non registrata".

Opzioni utili di `align_paintings.py`:

| opzione | default | a cosa serve |
|---|---|---|
| `--factor` | 2 | riduzione usata per la stima. 2 = sub-pixel; 4 = circa metà tempo, ~1 px |
| `--no-nonrigid` | off | stima solo l'affine (nessun campo di deformazione locale) |
| `--refine-block` | 128 | lato dei blocchi nella seconda passata: più piccolo = campo più fitto |
| `--min-conf` | 0.10 | confidenza minima perché un blocco venga usato |
| `--tile` | 2048 | dimensione dei tile in scrittura |

## Quando una trasformazione NON si riusa

**Una trasformazione stimata non descrive "l'UV": descrive quella specifica
coppia di esposizioni.** Per ogni nuovo scatto — un canale IR, un secondo UV, un
rifacimento — va ristimata. Tre motivi indipendenti, tutti visibili nei numeri
misurati su questa coppia:

1. **Posa relativa.** Rotazione +0.129° e traslazione (−32.7, −14.9) px possono
   venire solo dal fatto che camera e dipinto si sono mossi l'uno rispetto
   all'altro *fra* le due esposizioni. Un terzo scatto si è mosso di nuovo, di
   una quantità diversa.
2. **Tempo.** Il campo non rigido (rms 6.8 px, picchi 15 px) è la flessione della
   tela fra due *istanti*. Un altro momento, un'altra flessione.
3. **Lunghezza d'onda.** Aberrazione cromatica laterale e spessore/indice del
   filtro cambiano la magnificazione in funzione della banda: la correzione
   giusta per l'UV è sbagliata per l'IR, e con segno opposto rispetto al
   visibile.

Ristimare costa ~2 minuti, quindi non c'è motivo di rischiare. È esattamente
quello che fanno l'interfaccia e `align_batch.py`: **una stima indipendente per
ogni immagine**, tutte contro lo stesso riferimento. Tenendo sempre la stessa
immagine come riferimento, i risultati finiscono tutti nello stesso sistema di
coordinate e sono quindi confrontabili pixel a pixel fra loro.

## Scegliere precisione, tempo e memoria

Il menu **Precisione** decide il fattore di riduzione usato per la *stima*
(l'immagine finale è comunque sempre scritta a piena risoluzione). Dimezzare il
fattore quadruplica i pixel su cui si stima, quindi quadruplica all'incirca
tempo e memoria. Sotto al menu compare una **stima della memoria** aggiornata
dal vivo, calcolata sulle dimensioni reali delle immagini scelte; il testo
diventa arancione oltre i 4 GB. Misurato: 1.4 GB previsti e 1.5 usati a fattore
2 su una coppia da 15848×19850 px; 10.2 previsti e 11.0 usati a fattore 1 su
tre PNG da 29049×20428 px.

La stima del **tempo** che compariva accanto è stata tolta: a precisione
massima indicava 8–12 minuti per un lotto che ne ha chiesti 18–21, perché il
tempo dipende da cose che prima di partire non si sanno (se il primo aggancio
prende, quanto costa decodificare gli ingressi, cosa sta facendo il disco). I
tempi misurati per ogni impostazione, su due dipinti, sono in
[Documentazione/impostazioni/README.md](Documentazione/impostazioni/README.md).

Il pulsante **Impostazioni avanzate...** apre il resto delle manopole:

| impostazione | a cosa serve |
|---|---|
| Densità del campo | blocchi più piccoli seguono deformazioni più minute, ma ogni misura è meno sicura |
| Confidenza minima | abbassala se la stima fallisce per mancanza di blocchi; alzala se sospetti corrispondenze sbagliate |
| Soglia di segnalazione | oltre questo residuo il risultato è marcato "da controllare"; non cambia l'allineamento |
| Lato dei tile | incide poco sul picco di memoria: il grosso lo occupa la fase di stima |
| Formato di output | come l'originale, oppure sempre TIFF o PNG |
| Solo stima e verifica | esegue tutto **senza salvare le immagini**: dà i residui misurati senza occupare disco |

L'ultima è la più utile per provare delle impostazioni: ottieni i numeri in una
frazione del tempo, e scrivi i gigabyte solo quando sei convinto.

## Come si misura il risultato

La verifica prende **64 finestre da 512 px** (griglia 8×8) a piena risoluzione,
in posizioni che non dipendono da come è andata la stima, applica la
trasformazione al volo e misura di quanti pixel le due immagini sono ancora
fuori registro.

Se c'è una maschera, le finestre vengono messe **solo dentro il soggetto** (ne
serve dentro almeno il 98%). Non è un dettaglio: sul Fra Galgario, che è
**ottagonale**, un quinto dell'immagine è sfondo — una lavagna con scritte a
gesso, attrezzatura, cavi. Le finestre a cavallo del bordo della tela davano
**3.47 px** di residuo mediano su NIR850 contro 0.18 px all'interno, e con
confidenza *alta*, perché lo spigolo della tela è un dettaglio nettissimo. Non
erano misure sbagliate: erano misure giuste di un oggetto sbagliato, che sta a
un'altra distanza dalla camera e si sposta per parallasse. Senza maschera si
resta col margine rettangolare, che su una forma non rettangolare è un ripiego.

Vengono riportati questi numeri, e servono tutti:

- la **mediana** del residuo, con il suo **intervallo di confidenza al 90%**
  (calcolato per ricampionamento): dice quanto del valore è sostanza e quanto è
  l'aver guardato solo alcune finestre;
- la **quota di finestre oltre 1, 2, 5 e 10 px**: descrive la coda in modo
  stabile, e vede ciò che la mediana nasconde;
- la **copertura**, cioè quante finestre danno una misura su quante tentate (nel
  riepilogo e nella tabella dell'interfaccia, colonna «finestre»): la mediana
  vale per quella parte del soggetto, e se crolla l'allineamento è da buttare;
- la **distorsione media**, lo spostamento medio come vettore: un errore
  sistematico, tutto dalla stessa parte, si vede qui e non nel modulo.

Nella sezione RX c'è in più il **TRE sui punti messi a mano** (colonna «TRE
punti»): ogni punto si giudica con la correzione rifatta senza di lui, perché
un punto usato come vincolo, giudicato su se stesso, darebbe zero per
costruzione. Se i punti danno anche la posa di partenza il valore è ottimista,
e compare con un asterisco. Come si comportano queste misure sulla radiografia,
e perché il costo MIND non serve a esprimere un errore, è nel
[benchmark](Allineatore/benchmark/README.md#come-si-comportano-le-misure).

Il p95 resta nei file ma non va usato come indicatore principale: con poche
finestre non è un percentile ma quasi il massimo, e oscilla del doppio
spostando di poco la griglia.

Il numero di finestre è stato scelto misurando. Con 25 finestre (la versione
precedente) la mediana sulla coppia VIS/UVF oscillava fra 0.35 e 0.67 px contro
un valore vero di 0.50, e il massimo riportato era ~3 px contro gli 11.3 px
realmente presenti. Costo: 25 finestre 1 s, 64 finestre 2 s, 256 finestre 9 s,
contro i ~16 s della stima.

### Lungo il contorno

La verifica qui sopra, per costruzione, **non guarda il bordo**: la sua griglia
resta lontana dai lati. Sul Fra Galgario era proprio lì il difetto più visibile —
la mano sulla bottiglia, fuori di 5–14 px e ben visibile in falso colore — mentre
la verifica riportava 0.045 px. Il numero era vero, ma dell'interno soltanto.

Per questo c'è una seconda verifica, che usa finestre più piccole (256 px) per
poter stare vicino al contorno e riporta il residuo **per fasce di distanza dal
contorno** (0–500, 500–1000, 1000–2000 px e oltre). La distanza viene dalla
distance transform della maschera, quindi vale per qualunque forma. Senza
maschera il contorno è il bordo dell'immagine, e le fasce esterne possono
cadere sullo sfondo.

### Perché il bordo era peggio, e cosa è cambiato

Il fit robusto scarta un blocco quando il suo residuo è grande rispetto a quello
di *tutti* gli altri. Per stimare l'affine va bene, ma quello stesso insieme
veniva usato anche per il campo. Vicino al bordo della tela la deformazione vera
è più grande della media, e così il criterio globale scartava come anomali
proprio i blocchi che la misuravano: 40 su 81 sulla mano, 65 su 87 al vertice
sinistro, con confidenza buona. Il campo non li vedeva.

Ora i blocchi scartati vengono **riammessi se concordano con i loro vicini**: una
deformazione vera è liscia, un falso aggancio no. Gli inlier dell'affine restano
tutti, quindi dove non veniva scartato nulla il risultato è identico a prima (sul
test con trasformazione nota non viene riammesso nessun blocco: 0.335 px,
invariato).

Dall'8 ottobre 2026 la soglia con cui un blocco «concorda» ha un minimo: non
scende sotto il 5% del passo dei blocchi (`SOGLIA_MINIMA_PASSO` in
`block_matching.py`). Dove il rumore è bassissimo la soglia diventava così
stretta da scartare una deformazione liscia ma curva, cioè i bordi e i
rigonfiamenti. Sulle immagini artificiali, alla precisione consigliata, i punti
oltre 5 px scendono dal 24% all'1% con una prospettiva da 200 px e dall'11% a
zero con un rigonfiamento da 30 px; sulle riprese vere l'effetto è piccolo,
perché lì il rumore tiene già larga la soglia.

Misurato su VIS/NIR850 a fattore 2:

| | interno | entro 500 px dal contorno | mano (mediana / max) |
|---|---|---|---|
| prima | 0.26 px | 0.30 px, 11% oltre 2 px | 3.6 / 15.6 px |
| dopo | 0.24 px | 0.22 px, **0% oltre 2 px** | **1.8 / 6.1 px** |

Con o senza maschera il risultato sul dipinto è lo stesso: la maschera serve a
misurare il soggetto giusto, non a migliorare l'allineamento.

### Cuciture di mosaico

Le riprese sono fusioni di più tasselli. Se due mosaici sono stati cuciti con
allineamenti interni diversi, lungo la cucitura lo spostamento fra le due
immagini **salta di colpo**: sul Fra Galgario, contro la VIS, il NIR850 salta
fino a ~20 px lungo una linea orizzontale a y ≈ 11330, il NIR950 ha anche una
cucitura verticale a x ≈ 8500, la UVF una orizzontale a y ≈ 11200. Il residuo
che restava visibile sulla mano sulla bottiglia era questo: l'angolo di un
tassello, non una deformazione della tela.

L'allineatore **non corregge** le cuciture, e non deve provarci: un campo liscio
non può seguire un taglio netto, e uno discontinuo taglierebbe nel posto
sbagliato, perché la cucitura segue i contorni e la conosciamo solo ogni
64–128 px. Il rimedio sta a monte, nella fusione dei tasselli.

Da un confronto a coppie **non si può dire quale** dei due mosaici abbia
l'errore: si misura la discontinuità fra i due.

Per un periodo il programma le cercava da solo: i salti fra blocchi adiacenti
allineati lungo una riga o una colonna, riportati nel report, disegnati in
`diagnostica/cuciture.png` e usati per marcare il risultato "da controllare".
Sui dati sintetici funzionava (una cucitura da 8.5 px ritrovata a 8.51 px,
nessuna falsa cucitura su deformazioni lisce). **È stato tolto**, per due
motivi. Non veniva usato: le cuciture si correggono nella fusione dei
tasselli, non qui. E a precisione massima era pericoloso: il disegno
lavorava sull'anteprima a piena risoluzione prima di ridurla, e chiedeva oltre
10 GB di memoria. Il 27 settembre ha fatto fallire NIR850, NIR950 e UVR, con
allineamenti già riusciti, mentre nello stesso processo girava anche la
finestra RX. Toglierlo non cambia nessun numero: il test con trasformazione
nota dà gli stessi risultati alla terza cifra.

### Zone da rivedere

Alla fine della verifica il programma raggruppa le finestre problematiche in
zone e le divide in due categorie, che chiedono rimedi opposti:

- **non misurabile** — la correlazione non aggancia nulla, quindi lì non
  sappiamo se l'allineamento è buono o pessimo. Sulla radiografia ne risultano
  31 con la maschera, 38–51 senza.
- **non seguito** — la misura c'è ed è sicura, ma il residuo resta alto. Qui il
  problema non è la misura ma il modello, e un punto manuale non aiuta.

**Non sono un elenco di cose da fare, e per un po' il programma le ha presentate
così sbagliando.** Dove la correlazione non aggancia, di solito non aggancia
neanche l'occhio: le due riprese mostrano oggetti fisici diversi, e nella
radiografia quella zona è biacca, tela e telaio al posto della pittura. Un punto
messo lì senza riconoscere davvero lo stesso dettaglio non aggiunge informazione,
la inventa: cinque punti di quel genere hanno portato il residuo da 1.00 a
6.35 px, peggio che non metterne affatto. La regola, adesso scritta anche
nell'interfaccia, è: *un punto solo dove riconosci con certezza lo stesso
dettaglio in tutte e due; altrimenti passa oltre*.

Il numero, inoltre, non risponde al lavoro fatto — su una radiografia le finestre
cieche sono 174 con zero punti e 165 con 47 — ed è in buona parte un effetto
della soglia di raggruppamento (900 px su un'immagine di 19850): riaggregando a
2500 px le 45 zone diventano 7–13, a 4000 px una sola. Misura quanto poco le due
riprese condividono, non quanti errori ci sono da correggere.

Senza maschera qualche zona "non misurabile" può cadere sullo sfondo, e il
programma lo segnala; sulla radiografia di prova, però, cadevano tutte e 45 sul
dipinto.

### L'ordine della media locale

Il campo denso si costruisce facendo la media pesata dei blocchi vicini. Con la
media di **primo ordine** si adatta localmente un piano (valore più pendenza)
invece di una costante: dove i blocchi stanno tutti da un lato — al bordo del
soggetto, o accanto a una zona senza misure — la media semplice è tirata verso
l'interno, una pendenza no. L'ordine viene **scelto dalla cross-validation**,
come il lisciamento, non deciso a priori. Su VIS/NIR850 sceglie l'ordine 1 e il
residuo interno passa da 0.240 a 0.216 px; sul test con trasformazione nota
l'errore mediano scende da 0.350 a 0.335 px (misurato il 4 ottobre 2026; i
valori scritti in origine, 0.338 e 0.326, non si riproducono con nessuna
versione del programma nel repository).

## Quando il risultato è marcato "da controllare"

Un risultato viene segnalato se **uno solo** di questi tre criteri non regge:

1. **poche finestre affidabili** (meno della metà): le due immagini condividono
   struttura solo in alcune zone, quindi la stima poggia su poco;
2. **residuo mediano sopra soglia** (default 3 px);
3. **oltre il 5% delle finestre sbaglia più di 5 px**: allineato a macchie, che
   è peggio che uniformemente mediocre, perché non te ne accorgi guardando una
   zona sola.

Il criterio 1 e il 3 non sono un lusso. Sulle radiografie di questo progetto la
sola mediana dice 0.99 e 3.07 px, ma le finestre affidabili sono 11 su 64 e
23 su 63, e su una delle due il 30% delle finestre sbaglia oltre 5 px.

## Il caso delle radiografie (e delle bande "difficili")

Una radiografia non è una fotografia con un'altra illuminazione: è
un'immagine di **trasmissione**. Mostra la distribuzione della biacca, la trama
della tela, il telaio, i chiodi, i ritocchi — e spesso una composizione
sottostante diversa da quella visibile. Non mostra le velature e i pigmenti
trasparenti ai raggi X, che sono invece quello che domina la foto in visibile.

La conseguenza pratica, misurata su questo dipinto:

| | UV / NIR | radiografia |
|---|---|---|
| confidenza mediana dei blocchi | 0.15 | **0.036** |
| finestre di verifica affidabili | 54–61 su 64 | **11 e 23 su 64** |
| blocchi spazialmente coerenti | oltre il 90% | **circa il 7%** |
| residuo finale | 0.19 – 0.57 px | 0.99 e 3.07 px di mediana, ma **fino al 30% delle finestre oltre 5 px** |

Le zone che corrispondono davvero sono quelle ricche di biacca (incarnati,
collo, maniche, mani). Ai bordi del dipinto non c'è quasi nessuna misura
affidabile, e lì la deformazione viene **estrapolata** invece che misurata: è
esattamente dove si vede il disallineamento.

Ho provato, e **non funzionano**: CLAHE e altre normalizzazioni di contrasto,
blocchi più grandi (peggiorano), soglie di confidenza più basse, filtri di
coerenza spaziale, RANSAC al posto della stima robusta, omografia invece
dell'affine, polinomi lisci di grado 2 e 3. Nessuno scende sotto i ~10 px
mediani. Non è un problema di modello geometrico: è che l'informazione di
corrispondenza, in quelle zone, non c'è.

Questa conclusione si è rivelata troppo netta. Quei tentativi erano giudicati
sulla verifica, che ai bordi è cieca. Giudicati invece sui punti messi a mano, il
lato destro della radiografia era fuori di ~100 px per un limite di **portata**
della stima, non per mancanza di informazione: vedi qui sotto.

### La stima a tre scale

**Ora è l'algoritmo di tutte le bande.** Il lotto e la sezione RX usano
`tre_scale='auto'`: il campo grossolano si stima sempre, ma si applica solo se
la sua ampiezza supera la portata della passata fine (64 px). Sulle bande
fotografiche (27–33 px) non si applica, e il risultato è identico alle due
passate; su SWIR e radiografie (109–131 px) si applica, e il risultato è
identico alle tre scale. Applicato sempre, invece, peggiorava un poco le bande
fotografiche (UVF da 0.57 a 0.65 px sulle finestre comuni). I numeri sono nel
[benchmark](Allineatore/benchmark/README.md#un-algoritmo-unico-per-tutte-le-bande-ottobre-2026);
dal lotto si torna alle due passate con `align_batch.py --due-passate`.

Fino al 4 ottobre la stima aveva due passate: l'affine da blocchi grandi, poi il
campo fine da blocchi piccoli sulla radiografia già riportata con l'affine. Sulla
Cappellochiaro questo lasciava il **lato destro fuori di circa 100 px**, mentre la
verifica diceva 1.36 px. Lo hanno mostrato tre riferimenti indipendenti dalla
stima:

- i punti messi a mano su dettagli interni, a oltre 250 px dal bordo: sul lato
  destro la radiografia vera stava 103 px più all'esterno di dove la metteva
  l'automatica, a sinistra circa 0;
- il bordo tracciato a mano sulle due immagini: l'automatica lo metteva 134 px
  più all'interno a destra e 13 px in alto a sinistra;
- le sole tre finestre di verifica misurabili nel quarto destro, a 72, 104 e
  97 px. Erano tre su 136, e la mediana le nascondeva.

**Perché.** La seconda passata scarta ogni spostamento oltre un quarto del
blocco: la sua portata è di **64 px** a piena risoluzione. La prima passata ne ha
il doppio, ma il fit affine robusto scarta come anomali i blocchi che misurano
una deformazione grande (sulla radiografia ne passa circa il 14%), e da quei
blocchi nessuno costruiva un campo. Una deformazione oltre i 64 px che l'affine
non rappresenta restava quindi invisibile a tutte e due. L'informazione però
c'era: blocchi da 512 px misurati sopra l'affine vedono +102 px sul lato destro,
come i punti.

**Cosa fa adesso la sezione RX.** Fra l'affine e il campo fine stima un **campo
grossolano** (`block_matching.campo_grossolano`): blocchi da 512 px a piena
risoluzione, con portata 128 px, tenuti solo se concordano con i vicini, e
applicati solo se la cross-validation mostra un guadagno di almeno il 10%, come
per il campo fine. La passata fine parte da lì e ricorregge ciò che il
grossolano lascia. L'ordine conta: in una prova con il campo grossolano aggiunto
*dopo* quello fine, il centro peggiorava da 1.56 a 1.72 px sulle stesse finestre,
perché nessuno ne toglieva il rumore.

Misurato con il programma (allora `finestra_rx.allinea_rx`) sulle due radiografie, con
la correzione del mezzo pixel descritta più sotto, e la stessa griglia di
verifica per le due versioni. Il campo grossolano passa la cross-validation con
+92% sulla Cappellochiaro e +88% sulla TestaChiara.

| | due passate | **tre scale** |
|---|---|---|
| Cappellochiaro, punti a mano su dettagli interni (137) | 13.2 px | **5.0 px** |
| … sul lato destro | 91.3 px | **7.2 px** |
| Cappellochiaro, stesse 162 finestre di verifica | 3.33 px | **1.02 px** (132 migliorate, 17 peggiorate) |
| Cappellochiaro, finestre misurabili su 671 | 188 | **226** |
| TestaChiara, stesse 133 finestre di verifica | 0.96 px | **0.91 px** (63 migliorate, 46 peggiorate) |
| TestaChiara, finestre misurabili su 671 | 150 | **216** |

Tutte le righe sono con la zona del dipinto. Sulla TestaChiara la differenza
nella mediana è piccola, ma a due passate restava un gruppo di finestre fuori di
59–65 px sul **lato destro**, lo stesso difetto della Cappellochiaro, che con le
tre scale sparisce. Senza zona i risultati sono simili (Cappellochiaro, punti
interni da 13.1 a 5.1 px), ma la zona permette di verificare anche vicino al
bordo e va disegnata.

Con i 16 punti della corsa del 27/9 le tre scale danno 6.9 px sui dettagli
interni, peggio che senza punti (5.0): in alto e in alto a sinistra i punti
tirano il risultato di 12–13 px. Il danno viene dall'usarli come **posa di
partenza**. Per questo adesso la finestra RX li usa per impostazione predefinita
**solo come vincolo locale**, dove la verifica non misura (opzione
`punti_come_posa` di `align`). Provato con i 17 punti del 27/9 18:45, giudicando
su 115 dettagli interni che la corsa non usa:

| | senza punti | solo vincolo | posa + vincolo |
|---|---|---|---|
| dettagli interni non usati | 5.0 px | 5.0 px | 4.9 px |
| … lato destro / alto a sinistra | 7.2 / 3.9 px | 5.6 / 3.9 px | 5.7 / **41.7** px |
| verifica, stesse finestre | 1.13 px | 1.19 px | 1.26 px (84 peggiorate) |

Come vincolo locale i punti agiscono sul 6% della superficie e sono quasi
neutri. Come posa fanno danni. Sopra le tre scale non servono.

Resta un gruppo di tre finestre fuori di 12–15 px sul bordo sinistro della
Cappellochiaro (x ≈ 900, y fra 7200 e 8600). Lo vedono anche i punti messi a mano
su quel lato (5.5 px contro i 2.8 della versione prima della correzione). È un
difetto locale ancora da capire.

Nella finestra delle radiografie la stima a tre scale è accesa per impostazione
predefinita, e si può spegnere per rifare un tentativo come prima. **Il lotto e
le altre bande non la usano**: con l'opzione spenta la stima è identica a prima
bit per bit, verificato su RX Cappellochiaro, NIR850 e RX TestaChiara (stessa
matrice e stesso campo, differenza 0.000000 px) e su `test_pipeline.py` (0.335 px,
invariato).

**Cosa non risolve.** I clic messi sul bordo stesso, entro 250 px, restano a
~65 px da qualunque trasformazione, mentre i dettagli 250 px più all'interno
stanno a 4–6 px. Sul bordo le due immagini mostrano cose fisicamente diverse
(nella radiografia la tela ripiegata e i chiodi), e quei clic non indicano lo
stesso punto. Per la stessa ragione il contorno della tela tracciato sulla
radiografia sta in mediana 63 px fuori dal bordo dipinto del visibile anche
quando l'interno è allineato a 4 px: i due contorni non sono la stessa linea.
Gli ultimi ~250 px dal bordo restano senza un riferimento affidabile.

### Il mezzo pixel delle radiografie

Fino al 4 ottobre ogni radiografia allineata era spostata di **~0.5 px al centro e
fino a ~1 px verso il basso e verso destra**, in tutte le versioni del programma.
La causa era nella normalizzazione di scala, cioè nel ramo che si attiva solo
quando l'immagine da allineare ha dimensioni diverse dal riferimento: la RX è
18488×22824 px contro 15848×19850 del visibile. L'anteprima veniva riscalata
con `cv2.resize` dando le dimensioni d'uscita, che devono essere intere, e quindi
con rapporti veri di 0.863479 in x e 0.863389 in y. Il resto del programma però
tornava a piena risoluzione con il rapporto unico 0.863429. La differenza, cinque
parti su centomila, entrava *dopo* la stima e l'affine non poteva assorbirla.

Si vedeva nel segno dei residui di verifica: mediane di dx −0.50 e dy +0.49 px,
uniformi su tutto il dipinto e crescenti verso destra e verso il basso come
previsto dai due rapporti. Non c'era sul NIR850, che non viene riscalato (dx
+0.003 px), e spariva a fattore 1. Correggendo a posteriori la trasformazione con
i rapporti veri, dx e dy scendevano a −0.04 e +0.02 px.

Adesso il fattore si passa a `cv2.resize` come `fx = fy`, e OpenCV usa
esattamente quello su entrambi gli assi. Verificato:

- su un'immagine sintetica lo scarto all'angolo dell'anteprima passa da 0.2 px a
  meno di 0.01;
- in `test_pipeline.py` l'unica immagine riscalata, la PNG piccola, passa da 0.27
  a 0.22 px, e il resto è invariato (0.335 px);
- il NIR850 resta identico bit per bit. Le altre bande hanno le stesse dimensioni
  del visibile e non passano da quel ramo;
- sulla Cappellochiaro a tre scale il residuo mediano della verifica passa da
  1.48 a 1.13 px, con 226 finestre misurabili invece di 211.

I numeri delle radiografie riportati più sopra in questo capitolo, prima della
sezione sulle tre scale, sono stati misurati con il difetto.

### I cinque pixel dei TIFF a un canale

Un secondo difetto, più grande e indipendente dal primo, stava nella
**scrittura** dei TIFF (`tiff_memmap.create_tiff_memmap`). Per le immagini a un
solo canale l'intestazione dichiarava 10 voci invece delle 11 scritte, quindi
diceva che i pixel cominciavano 10 byte prima di dove erano davvero. Chi apriva
il file (Photoshop, Pillow, il programma stesso) vedeva l'immagine **spostata a
destra di 10 byte**: 5 px per i TIFF a 16 bit, come tutte le radiografie
allineate, e 10 px per quelli a 8 bit in grigio. Misurato sulla RX allineata del
4/10 alle 14:38: +5.00 px in orizzontale e 0 in verticale, identico in tre punti
del dipinto. Le immagini a tre canali erano corrette, e su questo progetto lo
sono tutte le bande tranne le due radiografie.

C'era dal primo commit (24/9), e nessuna verifica lo vedeva. Le verifiche
misurano la trasformazione sulle immagini sorgenti, non il file scritto, quindi
**tutti i residui riportati restano validi**: era sbagliato solo il file. Ora
`test_pipeline.py` rilegge con Pillow l'immagine allineata e controlla che
coincida pixel per pixel con la trasformazione. Le RX allineate prodotte prima
del 4 ottobre vanno rigenerate.

### Il margine attorno alla radiografia

La RX è più grande del visibile, e la sua tela esce dal fotogramma del visibile
(sulla Cappellochiaro fino a 56 px in basso e a destra): un'uscita grande quanto
il riferimento la tagliava. Nella sezione RX c'è quindi l'opzione, accesa per
impostazione predefinita, di scrivere su un foglio più grande. Il margine si
calcola da solo (`align_paintings.margine_necessario`) guardando dove finisce,
dopo l'allineamento, il fotogramma dell'immagine da allineare. È pura geometria,
non serve riconoscere niente nell'immagine. È abbondante, perché il fotogramma
contiene anche sfondo, e viene arrotondato a 100 px.

Con un margine l'immagine allineata non si sovrappone più al visibile
originale. Per questo accanto si scrive una copia del visibile con lo stesso
margine nero (`<nome>_margine_<N>px.tif`). Il pixel (x, y) del riferimento sta
in (x+N, y+N), e i parametri lo registrano (`margine_uscita_px`). Verificato su
dati sintetici: la parte centrale è identica, bit per bit, all'uscita senza
margine.

Per le altre bande l'opzione sta nelle impostazioni avanzate («Allarga il
foglio se un'immagine esce dal riferimento»), ed è **spenta** per impostazione
predefinita. Da riga di comando è `align_batch.py --margine auto` oppure
`--margine <px>`. Serve per dipinti fotografati con poco bordo, dove una banda
inquadrata un po' di lato perde una striscia. Falso colore e sovrapposizioni
vogliono lo stesso foglio per tutte le immagini, quindi il margine è **uno solo
per tutto il lotto**: il massimo fra quelli che servono alle singole bande.
Per conoscerlo bisogna aver stimato tutte le bande, perciò con `auto` il lotto
procede in due tempi: prima stima e verifica di ogni immagine, poi la
scrittura. La copia del visibile con il margine si scrive una volta sola, e il
falso colore aperto dopo il lotto la usa al posto del visibile originale.
Verificato su un lotto sintetico (una banda in scala di grigi e una a colori,
spostate di 150–200 e di 40–60 px): margine comune di 200 px, la parte centrale
è identica bit per bit all'uscita senza margine, e la cornice contiene
esattamente le strisce che prima andavano tagliate. Con l'opzione spenta il
lotto si comporta come prima (test di `test_pipeline.py`: 0.335 px).

### Il bordo della tela: la fascia di bordo

Sulla Cappellochiaro la zona disegnata sul visibile e il bordo della tela
tracciato sulla RX segnano lo stesso margine della tela. Dopo la stima a tre
scale restano però distanti 65 px di mediana (px della RX), sempre con la tela
RX più fuori: da 27 px in alto a sinistra a 101 px in basso.

Prima di correggere si è guardato dove sta la deformazione, usando i 221
dettagli cliccati a mano e misurando lo scarto per distanza dal contorno della
zona:

| distanza dal contorno | dettagli | scarto | perpendicolare al bordo |
|---|---|---|---|
| 0–100 px | 82 | 67.0 px | 59.2 px |
| 100–250 px | 2 | 2.2 px | 0.3 px |
| 250–500 px | 17 | 5.7 px | −0.5 px |
| 500–1000 px | 18 | 4.3 px | 2.4 px |
| oltre 1000 px | 102 | 4.4–5.0 px | −0.1–0.9 px |

I dettagli sul bordo sono spostati verso l'esterno quanto le due linee, mentre
quelli a più di 100 px sono a posto: la deformazione fra le due riprese sta
tutta in una fascia sottile lungo il bordo.

Per questo la correzione (`punti_manuali.campo_fascia_di_bordo`) porta il
contorno della zona sul bordo della tela soltanto in una fascia. Sul contorno e
fuori agisce per intero, poi si spegne verso l'interno con un raccordo a
coseno. Usa la sola componente perpendicolare al bordo, lisciata con una
mediana mobile di 300 px per parte. Si applica per ultima, sopra la stima
normale, e per rappresentarla il campo viene infittito (da 128 a 32 px; il
ricampionamento di un campo bilineare sui nodi più fitti non lo cambia). La
corrispondenza lungo la normale resta monotona, quindi la fascia si comprime ma
non si ripiega: il determinante minimo dello jacobiano è 0.54.

Larghezza della fascia (dalla corsa del 4/10 17:12):

| | senza | 100 px | 150 px | 250 px | 400 px |
|---|---|---|---|---|---|
| distanza fra le linee, mediana | 64.8 | 1.7 | 1.6 | 0.9 | 0.7 |
| dettagli sul bordo (0–100 px) | 67.0 | 28.2 | 24.4 | 23.0 | 22.6 |
| di cui perpendicolare | 59.1 | 17.9 | 6.1 | −1.6 | −3.7 |
| dettagli a 100–250 px (2) | 2.2 | 2.2 | 2.2 | 14.5 | 49.6 |
| dettagli a 250–500 px (17) | 5.7 | 5.7 | 5.6 | 5.7 | 6.8 |
| verifica a griglia fissa | 1.19 | 1.19 | 1.19 | 1.19 | 1.19 |

Si usa 150 px: le linee coincidono, e l'interno resta dov'era. Oltre i 250 px
la fascia comincia a trascinare i dettagli interni. Lo scarto di 20–25 px che
resta sui dettagli del bordo è quasi tutto lungo il bordo, dove un clic su un
lato dritto non ha un punto preciso, e la correzione non agisce. La corsa
completa (4/10 17:41) dà gli stessi numeri: linee da 65 a 1.9 px, verifica 1.19
px su 222/671 finestre.

Il prezzo è visibile: la striscia della RX fra il bordo dipinto e il bordo
della tela (chiodi compresi) viene compressa nella fascia, e i chiodi appaiono
schiacciati verso l'interno.

È stato provato anche il «bordo prima» alla lettera, con la posa di partenza
ricavata dalle sole due linee (`punti_manuali.posa_dai_contorni`, punto più
vicino iterato) e poi tutta la stima. Le linee coincidono lo stesso, grazie
alla fascia, ma l'interno peggiora: dettagli a 500–1000 px dal bordo da 4.3 a
10.9 px, in alto a sinistra da 3.6 a 17.8 px, finestre misurabili da 222 a
205, oltre 10 px da 15 a 22. La posa delle due linee non è quella
dell'interno. Resta come parametro di `align` (`posa_dal_bordo`), fuori
dall'interfaccia.

Nella finestra RX: il pulsante «Bordo della tela (RX)...» traccia il poligono
sulla radiografia (e lo ripesca dall'ultima corsa sulla stessa RX), e la
casella «Allinea il bordo» applica la fascia. La casella è spenta per
impostazione predefinita e vuole sia la zona sia il bordo: ha senso solo se la
zona è disegnata proprio sul margine della tela. Spenta, la corsa è identica a
prima.

Il vincolo di bordo precedente (`campo_dal_contorno`, sul ramo `rx-maniglie`)
è stato tolto. Spalmava la correzione fino a 700 px verso l'interno e spostava
di 28–45 px i dettagli a 250–1000 px dal bordo. Inoltre si attivava da solo
appena `align` riceveva un contorno.

### Tutte le configurazioni a confronto

`Allineatore/benchmark/radiografie.py` mette in fila, sulla Cappellochiaro e
senza scrivere immagini, le configurazioni della sezione RX (dall'automatico del
lotto fino a tutto acceso), i metodi classici a punti di controllo e i metodi
concorrenti. Li giudica tutti con i dettagli cliccati a mano, con la distanza
fra le due linee del bordo e con la verifica a griglia fissa. In breve:

| | dettagli interni (mediana / p90) | linee del bordo | finestre misurabili |
|---|---|---|---|
| automatico del lotto | 15.4 / 87.3 px | 79.8 px | 188/671 |
| tre scale | 5.3 / 20.2 px | 62.5 px | 221/671 |
| tutto (zona, punti, fascia di bordo) | 5.0 / 20.3 px | 1.9 px | 222/671 |
| solo punti, thin-plate spline | 18.4 / 41.2 px | 67.6 px | 219/671 |
| il migliore dei concorrenti (NGF + Powell) | 33.8 / 70.2 px | 53.5 px | 216/671 |

![panoramica del benchmark RX](Allineatore/benchmark/figure/rx_1_panoramica.png)

La tabella completa, gli altri grafici e la discussione sono nel
[README del benchmark](Allineatore/benchmark/README.md#le-radiografie-giudicate-a-fondo-ottobre-2026).

### La verifica della sezione RX

Nella finestra delle radiografie, residuo e finestre misurabili vengono da una
verifica a **griglia fissa**: sempre la griglia più fitta senza finestre
sovrapposte, 29 × 29 con la zona del dipinto e 15 × 15 senza
(oggi `verifica_fissa.verifica_griglia_fissa`, per tutte le bande). Le ragioni:

- la verifica di sempre smette di infittire la griglia appena trova 40 finestre
  misurabili. Un tentativo migliore si ferma quindi prima, su una griglia più
  rada, e due tentativi finivano per avere 28 finestre in comune su centinaia.
  Con la griglia fissa lo storico confronta due tentativi **sulle finestre
  misurate da entrambi**;
- le finestre oltre 10 px compaiono nello storico (colonna «oltre 10 px») e nel
  consiglio, e sono disegnate in `verifica_finestre.png` nella diagnostica. Quelle
  che fanno **gruppo**, cioè con un'altra finestra entro 1500 px fuori nello
  stesso verso, sono un disallineamento vero; quelle isolate sono di solito
  agganci deboli. La confidenza non basta a distinguerle: il gruppo vero del lato
  destro della Cappellochiaro aveva confidenza 0.055–0.11, come gli agganci
  isolati;
- vicino al bordo si dice quante finestre si misurano, e se la verifica lungo il
  contorno è cieca. Sulla radiografia, con finestre da 256 px e confidenza
  minima 0.2, di solito non ne misura nessuna;
- senza la «Zona del dipinto» la griglia resta nel rettangolo centrale (12% di
  margine) e non arriva al bordo, e il consiglio lo ricorda.

La verifica di sempre resta nel report (`verifica`); quella a griglia fissa le si
aggiunge (`verifica_griglia_fissa`, con le finestre una per una). Costa 10–40 s in più per
tentativo, secondo la griglia.

### I punti oggi: per misurare, non per allineare

Con la stima a tre scale l'automatico aggancia la radiografia da solo, e i
punti messi a mano non migliorano più l'allineamento. Sulla Cappellochiaro,
giudicando sui 115 dettagli cliccati lontano dai punti usati
([benchmark RX](Allineatore/benchmark/README.md#le-radiografie-giudicate-a-fondo-ottobre-2026)):

| | dettagli interni (mediana / p90) | finestre oltre 10 px |
|---|---|---|
| tre scale + zona, senza punti | 5.0 / 20.2 px | 12 |
| + 17 punti come vincolo locale | 5.0 / 20.2 px | 15 |
| + gli stessi punti anche come posa | 4.9 / 41.8 px | 34 |

Il vincolo non aiuta neppure vicino ai punti. Rifacendo la correzione senza il
punto che si giudica, lo scarto è 4.6 px, lo stesso che quei punti hanno
senza nessuna correzione: il campo automatico li prevede già.

Per questo la sezione RX parte con i punti **solo per verifica**. Il programma
non li usa per allineare e riporta su di loro il TRE (colonna «TRE punti»),
che è l'unica misura che arriva dove le finestre non arrivano. Conviene
quindi metterli sparsi, anche nelle zone che la verifica non vede. La casella
«Usa i punti come vincolo locale» li rimette in gioco quando l'automatico non
aggancia, per esempio su un altro dipinto. La configurazione predefinita
(tre scale, zona, fascia di bordo, punti solo per verifica) dà 5.0 / 20.3 px
sui dettagli interni, 12 finestre oltre 10 px e 1.9 px fra le linee del bordo.

Le sezioni che seguono raccontano come ci si è arrivati: quando l'automatico
non agganciava, i punti erano indispensabili.

### La sezione RX, con i punti messi a mano

Le radiografie avevano una sezione tutta loro nell'interfaccia
(`finestra_rx.py`, oggi sostituita dagli «Strumenti a mano» del lotto,
`finestra_strumenti.py`), sotto il flusso normale. Non è una questione di ordine: sono
l'unico caso che l'automatico non aggancia, e l'unico che si fa **a tentativi**.
Il percorso che funziona per le altre bande non deve occuparsene.

Il motivo non è che la radiografia sia incomprensibile: il pre-allineamento
automatico sa fare solo **traslazioni**, mentre fra visibile e radiografia
restano una scala del 12% e una rotazione di 0.3 gradi. Quando quell'aggancio
iniziale non riesce, i blocchi cercano tutti nel posto sbagliato. Quattro–otto
punti messi a mano danno la posa di partenza e tolgono l'aggancio dal caso; da lì
in poi lavora tutto l'automatico, che è più preciso di qualunque clic (1–2 px).

Misurato sulla stessa coppia (`VIS` ↔ `RX_..._Cappellochiaro`), stessi dati,
stessa macchina, cambiando solo precisione e punti:

| precisione | punti a mano | residuo mediano | finestre misurabili | oltre 5 px |
|---|---|---|---|---|
| normale (fattore 2) | nessuno | 1.00 px | 11 su 64 | 18% |
| normale (fattore 2) | 5, incoerenti (scarto 87 px) | 6.35 px | 18 su 63 | 56% |
| normale (fattore 2) | 18 | 2.66 px | 15 su 63 | — |
| **massima (fattore 1)** | **nessuno** | **161.18 px** | 4 su 62 | 100% |
| massima (fattore 1) | 18 | 3.35 px | 20 su 63 | 35% |

Se ne leggono tre cose, tutte controintuitive:

1. **Sulla radiografia la precisione massima è la scelta peggiore, non la
   migliore.** Senza punti passa da 1.00 px a 161 px, stimando una scala del
   15.3% invece del 12.2% vero: la posa è semplicemente sbagliata. A piena
   risoluzione i blocchi coprono una porzione più piccola di dipinto, quindi
   vedono meno contesto, mentre lo scarto da recuperare in pixel è il doppio.
   È anche la spiegazione del `36.31 px` che compare nei risultati del lotto
   lanciato a precisione massima: non era un difetto della radiografia, era
   l'impostazione.
2. **I punti servono a salvare un aggancio fallito, non a migliorarne uno
   riuscito.** Dove l'automatico aveva fallito, 18 punti portano 161 px a 3.35
   px. Dove aveva già funzionato non aggiungono nulla di netto (1.00 px su 11
   finestre contro 2.66 px su 15: il secondo numero è più alto ma copre quattro
   zone in più, per cui i due non si confrontano alla lettera).
3. **Punti incoerenti fanno danno.** Cinque punti con 87 px di scarto reciproco
   portano il residuo da 1.00 a 6.35 px, peggio che non metterne affatto. Per
   questo la finestra mostra lo scarto di ogni punto e segna in rosso quelli che
   litigano con gli altri.

### Lo "scarto dei punti" non misura la bravura di chi clicca

È l'errore di lettura più facile da fare, e per un po' l'ha fatto anche questo
programma. Su una serie di 47 punti messi a mano lo scarto mediano dall'affine
era 19.3 px, e il messaggio diceva «i punti non concordano fra loro, rifai i più
incoerenti». Misurando si è visto che era il contrario:

| | |
|---|---|
| differenza dal vettore di scarto dei **4 punti più vicini** | 10.3 px |
| differenza dal vettore di scarto di **4 punti a caso** | 19.2 px |
| rms dello scarto dei punti dall'affine | 26.4 px |
| rms del **campo non rigido** stimato dal programma su quella coppia | 19.0 px |

Gli scarti sono **lisci nello spazio** — punti vicini sbagliano nella stessa
direzione — e la loro ampiezza coincide con quella della deformazione che il
programma stima per conto suo. Cioè i punti stavano descrivendo una deformazione
vera della tela, che un'affine a sei parametri non può rappresentare per
costruzione. Erano buoni, non cattivi, e rifarli non avrebbe abbassato quel
numero di un pixel.

Il criterio giusto è quindi lo stesso che `block_matching` usa per i blocchi
(`blocchi_coerenti`): **un punto è sospetto quando litiga con i vicini**, non con
il modello globale. Una deformazione è liscia, un clic sbagliato no.
`punti_manuali.controlla_punti` lo applica; su punti costruiti a tavolino con
20 px di deformazione liscia, 4 px di rumore dei clic e due errori piazzati
apposta, segnala esattamente i due errori. Sui set reali segnala fra 0 e 6 punti
su 9–47.

### Cliccare alla buona: il punto lo rifinisce il programma

Mettere i punti a mano fra visibile e radiografia è faticoso, e non per colpa
dell'interfaccia: in molte zone non c'è proprio nulla di riconoscibile in
entrambe. Dove invece qualcosa c'è, non serve azzeccare il pixel. Dopo il clic
sulla seconda immagine il programma cerca la posizione esatta lì intorno con una
correlazione di fase sul modulo del gradiente — la stessa del matching
automatico, ma con il punto di partenza dato da un umano.

Misurato sui punti veri di una radiografia, spostandoli apposta per simulare un
clic frettoloso:

| finestra | clic sbagliato di 20–50 px | 50–100 px | 100–200 px |
|---|---|---|---|
| 256 px | 5.1 px | 53.9 px | 161.3 px |
| **384 px** | **4.3 px** | **3.8 px** | 147.5 px |
| 768 px | 5.2 px | 5.9 px | 6.6 px |

Recupera un errore fino a circa un quarto della finestra. La finestra da 768
arriva più lontano ma può spostare di 132 px un punto già giusto, quando aggancia
la trama della tela che essendo periodica offre più picchi uguali; con 384 px il
caso peggiore misurato è 66 px. Si preferisce quindi non allargare.

**Su punti messi con cura non cambia niente**, ed è giusto dirlo: confrontando la
coerenza spaziale del gruppo prima e dopo su tre serie reali da 20, 26 e 47
punti, si passa da 10.3 a 9.5, da 7.8 a 8.5 e da 17.5 a 18.6 px. È rumore. Serve
a una cosa sola: permettere di cliccare **alla buona**, senza zumare al 100% e
cercare il pixel. Per questo, con la rifinitura accesa, la vista da allineare
accetta il clic anche da lontano, mentre quella di riferimento continua a
chiedere lo zoom (lì il clic decide *quale* dettaglio si sta indicando).

Quando la correlazione non è sicura il punto resta esattamente dov'è stato
cliccato — e sulla radiografia capita spesso, perché ci sono zone senza nulla da
agganciare. Il caso peggiore da 66 px resta possibile, ed è uno dei motivi per
cui il controllo dei punti guarda comunque la coerenza con i vicini: un punto
spostato a caso dalla rifinitura esce segnalato in rosso nell'elenco.

### I punti come vincolo della misura, non solo della posa

Per un po' i punti hanno avuto un difetto che li rendeva quasi inutili, e che
non si vedeva da nessun numero stampato: servivano **solo** a dare la posa di
partenza, e poi venivano dimenticati. Misurato su una corsa reale da 50 punti,
la trasformazione finale cadeva a **10.2 px di mediana** (p90 42 px, massimo
93 px) da dove i punti erano stati messi. Sul quarto destro di quella
radiografia, dove **nessuna delle 16 finestre di verifica è misurabile**, lo
scarto arrivava a 31 px e cresceva a rampa fino a 45 px sul bordo estremo: il
campo automatico lì non ha dati e si limita a estendere quello del centro.

Adesso i punti diventano **vincoli esatti** della trasformazione (thin-plate
spline: la superficie più liscia che passa per tutti i punti dati), applicati
però **solo dove la verifica non riesce a misurare nulla**. Dove il matching
automatico lavora, misura a frazioni di pixel e un clic ne sbaglia 5–10:
sovrapporglielo sarebbe un danno.

#### Che cosa garantisce, e che cosa no

Va detto con precisione, perché è facile rivendicare più di quanto ci sia.

**Ai punti**, la correzione è esatta per costruzione: lo scarto passa da 9.7 a
1.3 px. È vero ma è quasi una tautologia — la spline passa per i dati che le
sono stati dati.

**Attorno ai punti**, fino a circa 700 px, è meglio di niente. È stato misurato
adattando la spline su una parte dei punti e guardando l'errore sugli altri, mai
visti, in funzione della distanza dal punto usato più vicino (due serie reali di
41 e 50 punti; il confronto è con il non correggere affatto, 8.9 e 9.4 px):

| distanza dal punto più vicino | 41 punti | 50 punti | |
|---|---|---|---|
| 0 – 700 px | 6.6 px | 4.6 px | meglio di niente |
| 700 – 1400 px | 6.7 px | 10.9 px | al limite |
| 1400 – 2500 px | 16.6 px | 12.1 px | **peggio** |
| 2500 – 4000 px | 20.2 px | 34.2 px | peggio |
| oltre | 51.1 px | 98.6 px | molto peggio |

**Oltre**, la spline non prevede: inventa, e fa peggio dell'estrapolazione
automatica che sostituirebbe. Per questo la correzione si applica solo entro
700 px da un punto, con dissolvenza a zero entro 1400 — su una radiografia con
21 punti sono il **17% della superficie**, non il 75% a cui arrivava una prima
versione con il raggio troppo largo.

#### Una misura che sembrava una prova e non lo è

Il numero di finestre di verifica che diventano misurabili sale quando si
applica la correzione, e la tentazione è di prenderlo come prova che
l'allineamento è migliorato: sono misure indipendenti, che non sanno niente dei
punti. **È un ragionamento circolare, e si può dimostrare che lo è.** Ripetendo
tutto con i punti spostati a caso di ~40 px, quel numero sale **esattamente
uguale**:

| | finestre misurabili | residuo sulle 9 finestre comuni |
|---|---|---|
| senza punti | 11 | 1.00 px |
| con i punti veri | 15 (+4) | **0.87 px**, 1 peggiorata |
| con punti **sbagliati** | 15 (+4) | 1.13 px, 3 peggiorate |

Imporre un campo qualsiasi cambia la geometria locale quanto basta a far
agganciare qualche correlazione in più, giusto o sbagliato che sia. Il residuo
sulle finestre comuni invece **distingue**, ed è quello da guardare.

#### Perché interpolare e non smussare

Costruire dai punti un campo smussato (media pesata, come `build_dense_field`) è
stato provato e peggiorava: media i clic senza passarci dentro, quindi ne
eredita l'errore senza ottenere il vincolo. Serve anche a contenere la spline,
che fuori dall'inviluppo dei punti cresce come *r²·log r*: senza limite arrivava
a 125 px nei quattro angoli del dipinto, dove non c'è né una misura né un clic a
sostenerla.

Conseguenza importante: un punto sbagliato adesso viene **inciso** nel risultato
invece di essere diluito fra gli altri. Per questo i punti in contrasto con i
vicini non vengono più solo segnalati, ma **esclusi** dai vincoli, e il report
dice quali.

### Lavorare a giri successivi

Dal secondo giro in poi la finestra dei punti mostra a destra la radiografia
**già corretta** dal giro precedente. Le due viste vivono così nello stesso
sistema di coordinate: cliccare diventa «questo dettaglio è ancora spostato di
tanto così» invece di «ritrova lo stesso dettaglio in un'immagine più grande del
12% e ruotata». I punti già messi non vengono ridisegnati — sono stati assorbiti
dalla correzione — ma restano nell'elenco, da cui si possono togliere.

Sotto, però, **non si impila niente**: i punti vengono sempre memorizzati in
coordinate dell'immagine originale, e ogni giro rifà la stima da capo dalle
immagini originali con l'insieme completo. Impilare davvero avrebbe tre difetti:
gli errori di un giro resterebbero sotto quelli successivi proprio dove la
verifica è cieca; l'immagine verrebbe ricampionata a ogni giro e dopo tre sarebbe
visibilmente più molle; e i punti del terzo giro varrebbero solo dopo aver
rifatto esattamente i primi due, quindi il loro file non sarebbe più
riutilizzabile.

Il raffinamento ha un fondo: quando lo scarto ai punti scende sotto i 6 px si è
all'ordine dell'imprecisione di un clic, e il programma lo dice invece di
lasciare che si faccia un altro giro a inseguire rumore.

### Quanti punti servono davvero

I punti alimentano solo la posa di partenza, cioè un'affine a sei parametri:
tutto il resto lo misura il matching automatico. Quindi oltre una certa quantità
aggiungerne non cambia più niente. Prendendo i 47 punti di una sessione e
ricalcolando la posa con un sottoinsieme, la differenza rispetto alla posa
completa (misurata ai quattro angoli e al centro del dipinto) è:

| punti usati | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| differenza mediana | 133 px | 45 px | 43 px | 25 px | 23 px | 15 px | 9 px |

Da 12 punti in su si resta sotto i 25 px, che i blocchi da 512 px recuperano
senza fatica. E infatti i residui finali misurati lo confermano: 21 punti →
1.57 px, 26 punti → 1.16 px, 47 punti → 1.51 px. **Oltre i 12–16 punti il lavoro
in più non si traduce in precisione.**

Il lavoro è ripetibile in senso stretto, verificato: rifacendo lo stesso
tentativo con gli stessi punti si riottiene **la stessa trasformazione bit per
bit** (differenza massima 0 sull'affine e 0 sul campo denso, stesso residuo
verificato). I punti vengono salvati accanto ai parametri
(`<nome>_punti.json`, con dentro il percorso dell'immagine su cui erano stati
messi) e si ricaricano da soli riaprendo quella coppia; quelli di un'altra
radiografia non vengono raccolti, perché le loro coordinate non c'entrano nulla.

Nella finestra dei punti, appena si clicca un dettaglio nel visibile l'altra
vista si porta da sola sulla zona corrispondente, con un cerchio tratteggiato
che dice **di quanto la previsione può sbagliare**. Il raggio non è una costante
scelta a occhio: sopra i 4 punti si misura sui punti dell'utente stesso,
lasciandone fuori uno alla volta. Se i clic sono coerenti il cerchio si stringe
(±480 px con 3 punti, ±167 px con 18); se litigano fra loro resta largo
(±1141 px con due punti sbagliati apposta), che è l'avviso giusto. Sotto i 3
punti si usano le sole proporzioni fra le due immagini: misurando in
leave-one-out sui punti veri si vede che **uno o due punti peggiorano** la
previsione invece di migliorarla (p90 dell'errore 352 px con le sole
proporzioni, 653 px con un punto, 1347 px con due), perché le proporzioni
azzeccano già la scala d'insieme mentre un clic solo ci somma il proprio errore
locale.

Per il resto:

- usare comunque il residuo **dichiarato**: anche a 1.7 px il centro è allineato
  meglio dei bordi;
- non fidarsi del solo valore mediano: guardare gli overlay in
  `diagnostica/<nome>/`;
- evitare i punti sul **bordo del dipinto**: nella radiografia lì si vedono il
  telaio e la tela, non la pittura, quindi l'angolo che si riconosce nel visibile
  non è lo stesso punto fisico. Cinque punti messi così avevano portato lo scarto
  a 69 px.

## Le misure finali (10 ottobre 2026)

Tutte le misure sono state rifatte il 9–10 ottobre 2026 con la versione
`aligner-multisp-v7.0` del programma. Residui in pixel a piena risoluzione; dove
questa guida riporta numeri con una data precedente, quelli descrivono il
programma di allora e questi lo sostituiscono.

| misura | risultato | script e dati in `Allineatore/benchmark/` |
|---|---|---|
| Infrarosso vicino, tre riprese | 0.23 px alla precisione consigliata (0.20–0.29), 0.06 alla massima | `bande_precisioni.py`, `.jsonl` |
| Ultravioletto, tre riprese | 0.64 px alla consigliata (0.32–1.02), 0.16 alla massima | come sopra |
| Sola affine, stesse riprese | 10.0 px (infrarosso), 16.5 px (ultravioletto) | come sopra |
| SWIR | 1.19 px sul 39% delle finestre (1.83 px sul 35% senza campo grossolano) | `bande.py`, `bande_risultati.json` |
| Radiografie, due riprese | 1.13 px sul 33% delle finestre (2.60 px sul 25% senza) | come sopra |
| Radiografia, 98 dettagli individuati a mano | errore mediano 4.5 px, 90% entro 19.8 px (15.7 px senza campo grossolano) | `radiografie.py`, `rx_risultati.json`, poi `tre_per_dettaglio.py` |
| Sei metodi affini, quattro riprese fotografiche | 7–20 px, contro 0.45 px (ultravioletto) e 0.21 px (infrarosso) | `esegui.py`, `risultati.json` |
| Gli stessi sull'affresco (593 Mpx) | 17–80 px, contro 0.29 e 1.02 px; SIFT e Mattes non stanno in memoria | `esegui.py --affresco <cartella>` |
| Quattro metodi deformabili, griglia fissa | flusso ottico 0.53 e 0.23 px contro 0.44 e 0.20; sulle radiografie 12–17 px contro 1.13 | `deformabili.py`, `deformabili_reali.jsonl` |
| Immagini artificiali, 111 corse | posa entro 0.02 px; interno sotto 0.5 px salvo l'onda fitta (1.66 px); oltre 5 px il 6% dei punti con 200 px ai bordi, il 7 e il 18% con rigonfiamenti da 100 e 200 px | `sintetico.py`, `sintetico_tabelle.txt` |
| Riprese vere deformate in modo noto | 0.1–0.7 px oltre 500 px dal bordo, 0.3–11 px entro 150 px | `vere_deformate.py`, `vere_deformate_tabelle.txt` |
| Tempi e memoria | 0.11–0.14 s/Mpx e 4–5 byte/px alla consigliata; 0.35–0.5 s/Mpx e 14 byte/px alla massima | `bande_precisioni.jsonl` |

Ogni script scrive una riga per corsa e salta le corse già fatte: per rifarne
una va tolta la riga, o rinominato il file dei risultati. Le immagini di lavoro
finiscono in `benchmark/lavoro/`, fuori da git. Una prova pesante alla volta:
l'affresco alla precisione massima occupa quasi 9 GB.

## Rispetto a quello che esiste già

Esistono altri lavori che allineano immagini multibanda di dipinti, e alcuni
fanno anche correzione non rigida. Conviene saperlo prima di descrivere questo
programma come una novità.

### Sindel et al., registrazione non rigida su craquelure

Il lavoro più vicino è del gruppo di Erlangen ([CraquelureNet, 2022](https://arxiv.org/abs/2208.08836),
esteso al non rigido in [gennaio 2026](https://arxiv.org/abs/2601.16348)). Una
rete neurale trova punti corrispondenti sulle craquelure, una seconda rete li
accoppia, e la deformazione viene modellata con thin-plate-spline, con un
raffinamento a risoluzione crescente.

Errori dichiarati, su punti annotati a mano:

| dominio | errore medio | errore massimo |
|---|---|---|
| VIS–radiografia | 0.64 – 0.85 px | 2.35 – 2.51 px |
| radiografia–IR | 0.74 px | 2.65 px |
| VIS–IR | 1.24 px | 3.27 px |
| VIS–UV | 1.55 px | 3.84 px |

**Questi numeri non sono confrontabili con i nostri.** Loro misurano
l'accuratezza rispetto a punti annotati da un esperto; noi misuriamo la coerenza
fra le immagini con lo stesso strumento che usiamo per stimare. Sono due cose
diverse, e la seconda è più debole.

Quello che si può dire:

- **Sulla radiografia sono nettamente più avanti.** Loro riportano 0.64–0.85 px
  su punti annotati; noi, su dettagli individuati a mano, abbiamo un errore
  mediano di 4.5 px, e la verifica a finestre (1.1–1.2 px) misura solo un terzo
  della superficie. Il loro
  metodo è addestrato proprio sulle crepe, che nelle radiografie di tavole si
  vedono bene. È il nostro punto più debole e il loro più forte.
- **Il loro dominio è diverso**: dipinti su **tavola** tedeschi dal XIII al XVI
  secolo, e dichiarano che il rilevatore funziona meglio su strutture a crepa.
  Il tuo è su tela.
- **Reggono immagini enormi** (fino a 7939×42227 px, circa 335 Mpx) suddividendo
  il campo di spostamento in blocchi. Gestire file grandi non è quindi un tratto
  distintivo di questo programma.
- **Richiedono addestramento**: 20200 punti annotati per il rilevatore e 13595
  coppie per il descrittore. Questo programma non richiede nessun addestramento.
- **Girano su GPU** (Quadro RTX 8000 da 48 GB, 32 s per coppia). Questo gira su
  un portatile da 16 GB senza GPU, in ~25 s per coppia di stima e verifica (più
  47 s se si scrive anche il TIFF allineato da 0.88 GB).
- **Il codice non è pubblicato**, o almeno non l'ho trovato.

### Registrazione deformabile classica (elastix, ANTs)

B-spline più informazione mutua è lo standard della registrazione deformabile
multimodale. L'ho provata in SimpleITK con sette configurazioni, e il risultato
è interessante: **è potente ma instabile, e molto costosa**.

Sull'immagine ridotta non funziona (nessuna configurazione scende sotto i 5.8
px, quasi tutte peggiorano la sola affine), perché a quella scala la
deformazione da trovare vale meno di un pixel. Ma su ritagli a piena
risoluzione, dove vale diversi pixel, su un ritaglio ha dato **0.03 px** —
meglio del nostro campo — e su un altro è **divergita a 370 px**. Quel modo di
fallire è tipico di una B-spline senza penalità sull'energia di curvatura, che i
file di parametri di elastix includono e la mia configurazione no.

Tempi: da 6 a 58 minuti per un ritaglio da 1536 px, e l'immagine intera ne
conterrebbe 133.

Il dettaglio con tutte le configurazioni è in
[Allineatore/benchmark/README.md](Allineatore/benchmark/README.md).

L'8 ottobre 2026 il confronto è stato fatto sulle immagini intere, con quattro
metodi deformabili in SimpleITK e OpenCV che partono dalla posa del programma
(`benchmark/deformabili.py`; i numeri sono in «Le misure finali»). **Manca
ancora elastix con i suoi file di parametri**, cioè la B-spline con la penalità
sulla curvatura: quel confronto va considerato non fatto.

### Cosa resta di specifico, allora

- **nessun addestramento e nessuna GPU**: gira dove sta l'immagine;
- **dipendenze minime** (numpy, OpenCV, Pillow), nessuna libreria di
  registrazione;
- **verifica e giudizio automatici** su ogni risultato, con segnalazione
  esplicita quando non è affidabile — il lavoro di Sindel riporta l'accuratezza
  su un test set, non un giudizio per immagine al momento dell'uso;
- **funziona su tela** con struttura fine condivisa;
- un'**interfaccia** per chi non usa la riga di comando.

## Formati e dimensioni

Il riferimento e le immagini da allineare possono essere in formati diversi:

| formato | come viene letto |
|---|---|
| TIFF non compresso | mappato in memoria: si legge solo il pezzo che serve (via veloce) |
| TIFF compresso (LZW/ZIP), JPEG, PNG, BMP | nel lotto: letto da una copia TIFF temporanea, mappata in memoria (vedi sotto). Altrove: caricato per intero in RAM, con avviso se pesa oltre 0.25 GB |

**La copia TIFF temporanea (8 ottobre 2026).** Un formato compresso non si può
leggere a pezzi: restava per intero in memoria per tutta l'elaborazione,
riferimento e immagine insieme. Ora il lotto (`align_batch.allinea_lotto`,
quindi anche l'interfaccia) decomprime ogni immagine una volta sola, la scrive
in un TIFF non compresso nella cartella temporanea del sistema e la legge da
lì. La copia di un'immagine si toglie appena quell'immagine è finita, quella
del riferimento a fine lotto, e comunque alla chiusura del programma: su disco
non resta niente, e i file prodotti escono nel formato chiesto come prima.
Serve spazio temporaneo pari ai pixel non compressi (1,7 GB per un'immagine da
593 Mpx a colori); se non c'è, o se l'immagine supera i 4 GB, si legge in
memoria come prima. Si spegne con `copie_tiff=False`.

Misurato sull'ultravioletto di un affresco, PNG da 593 Mpx su un visibile in
PNG, precisione consigliata, uscita in PNG:

| | senza copia | con la copia |
|---|---|---|
| Picco di memoria del programma | 9,4 GB | 4,3 GB |
| Memoria libera minima del computer (15,8 GB) | 1,0 GB | 4,5 GB |
| Tempo totale | 26 min | 18,5 min |
| Residuo, finestre misurabili | 1,017 px, 55/64 | 1,017 px, 55/64 |
| File prodotto | — | identico byte per byte |

Il tempo scende perché il PNG si decomprime una volta invece di una per
l'anteprima e una per la piena risoluzione. Le finestre della zona e dei punti
aprono ancora l'originale.

Il file di uscita mantiene l'estensione dell'originale (salvo altri formati scelti), e ha sempre le
dimensioni dell'**immagine di riferimento**: è questo che rende i risultati
sovrapponibili.

Se una ripresa ha dimensioni diverse dal riferimento (sensore o ritaglio
diverso), la scala viene normalizzata prima della stima e la differenza residua
finisce nella trasformazione. Il limite: le immagini devono inquadrare
**grosso modo lo stesso soggetto**. Se una è un dettaglio ingrandito di una
porzione dell'altra, o è ruotata di 90°, la stima fallisce e lo dice.

`apply_transform.py` resta utile per due cose:

- **rigenerare** l'immagine allineata dai soli parametri (riproducibilità: chi
  legge la tesi ricostruisce il risultato dal JSON; e a te serve se cancelli
  l'output o cambi `--tile`);
- **warpare i raster derivati dalla stessa ripresa UV** — maschere, ROI tracciate
  sull'UV, mappe di segmentazione, falsi colori, ri-esportazioni — che hanno la
  geometria identica per costruzione.

Se i parametri sono stati salvati con un margine (`margine_uscita_px`), lo
script lo rispetta: la nuova uscita ha lo stesso foglio di quella originale e
si sovrappone alla stessa copia del riferimento con margine. Verificato sul
lotto sintetico: il file riscritto è identico bit per bit a quello del lotto.

Per evitare l'uso sbagliato, lo script **misura prima di scrivere**: campiona
alcune finestre a piena risoluzione, applica la trasformazione al volo e, se il
residuo mediano supera `--max-residuo` (default 3 px), si ferma senza creare
nulla e ti indica il comando giusto. `--force` lo scavalca, `--no-check` salta
la verifica ma richiede `--force`.

## Come funziona

### 1. I/O: i TIFF non vengono mai caricati interi (`tiff_memmap.py`)

I tuoi TIFF sono **non compressi**, quindi i pixel sono un blocco contiguo di
byte dentro il file: si possono mappare in memoria con `np.memmap` e leggerne
qualsiasi sotto-regione senza caricare il resto. Leggere un tile 2048×2048
costa ~12 MB, non 950 MB.

Nota: `tifffile` **non è installato** sul tuo PC (la pipeline in `Test_sonnet/`
non partiva proprio per questo). `tiff_memmap.py` usa solo `numpy`, `Pillow` e
`cv2`, che hai già. Se un giorno userai TIFF *compressi*, il modulo lo rileva e
lo dice con un errore esplicito invece di produrre risultati sbagliati.

### 1-bis. Lo sfondo dietro il dipinto

Lo sfondo (la stanza, il cavalletto, il muro) non va allineato e in genere si
esclude da solo, per tre filtri in cascata:

1. **blocchi piatti** — un blocco senza variazione, come un blocco nero,
   non contiene informazione di posizione e viene saltato;
2. **confidenza** — dove lo sfondo ha struttura in una sola banda, il picco di
   correlazione è debole e il blocco viene scartato;
3. **outlier** — quel che sopravvive ma dice una cosa diversa da tutti gli altri
   viene escluso dal fit robusto.

Lo si vede nel report della verifica: quando legge *"residuo su 22 campioni"* ma
i campioni erano 25, i tre mancanti erano finiti fuori dal dipinto.

L'unico caso in cui serve intervenire è uno sfondo **strutturato in tutte le
bande** (dipinto su cavalletto davanti a una parete con motivi visibili):
quei blocchi correlano bene, ma stando a una distanza diversa
dalla camera il loro spostamento apparente è genuinamente diverso da quello della
tela (parallasse) e falserebbe la stima. Per quel caso, nell'interfaccia c'è
*"Limita la stima a una zona che disegno io"* (poligono o rettangolo sul
riferimento), spenta di default. La finestra di disegno si adatta allo schermo e
ha lo zoom: **Ctrl + rotella** (zuma dove sta il puntatore), **Ctrl +/-**,
**Ctrl+0** per riadattare, e **tasto destro trascinato** per spostare la vista.
I vertici restano ancorati all'immagine, quindi si può zumare a metà disegno
senza spostare nulla.

Tre forme disponibili: **poligono** (un clic per vertice), **rettangolo**
(trascinato) ed **ellisse** (trascinata dal centro; tenendo **Shift** si ottiene
un cerchio esatto). L'ellisse è comoda per i dipinti ovali o tondi.

La maschera vive nel sistema di coordinate del **riferimento**, ed è lì che va
disegnata: i blocchi sono indicizzati sulle coordinate del riferimento, e nella
seconda passata l'immagine da allineare è già riportata in quel sistema. Un
blocco viene usato solo se **almeno il 75% della sua area** cade dentro la
maschera: guardare solo il centro lascerebbe entrare blocchi che sporgono per
metà nella zona che si voleva escludere.

Attenzione: la maschera limita **solo la stima**. L'immagine finale viene
trasformata e salvata per intero, sfondo compreso — sfondo che però resta
disallineato, perché ha una geometria sua, e non va usato per misure.

### 2. Stima: block matching a phase correlation (`block_matching.py`)

Su una coppia di anteprime ridotte (default: fattore 2) si misura lo
spostamento locale su una griglia di blocchi, con `cv2.phaseCorrelate`
applicata al **modulo del gradiente** delle due immagini.

- lavorare sul gradiente rende il confronto insensibile al fatto che una
  craquelure sia scura in VIS e chiara in UV, ad esempio;
- la phase correlation usa solo la fase dello spettro, quindi ignora le
  differenze di luminosità e contrasto fra le due modalità;
- ogni blocco restituisce anche una **confidenza**, quindi lo sfondo nero dell'immagine
  bersaglio e la parete della stanza visibile solo in VIS vengono scartati invece di
  inquinare la stima;
- non c'è alcuna finestra di ricerca da scegliere, e la stima è sub-pixel.

Poi:

1. **prima passata** (blocchi da 256 px): fit affine robusto, con scarto
   iterativo degli outlier oltre 3 MAD e pesi dati dalla confidenza;
2. **seconda passata** (blocchi da 128 px) sull'immagine *già pre-allineata*: gli
   spostamenti residui sono quasi nulli, quindi blocchi più piccoli restano
   affidabili e il campo risulta più fitto. Da qui escono una piccola
   correzione affine e il campo di deformazione locale.

Solo nella sezione delle radiografie, fra le due passate se ne inserisce una
terza, a scala grossolana: vedi [La stima a tre scale](#la-stima-a-tre-scale).

### 3. Il campo non rigido viene applicato solo se è segnale

Prima di usarlo, il campo viene validato in **cross-validation**: si costruisce
usando metà dei blocchi e si misura quanto riduce il residuo sull'altra metà,
cioè su blocchi mai visti. Se il guadagno fuori campione è sotto il 10%, il
campo è rumore di misura e viene scartato (resta la sola affine).

Con la stessa cross-validation viene scelta anche l'**ampiezza dello
smoothing** del campo, provando diversi sigma e tenendo quello che minimizza il
residuo fuori campione. Non è un dettaglio: sul test sintetico, uno smoothing
fisso di 144 px cancellava il 40% di una deformazione reale di periodo 900 px,
lasciando 2.4 px di errore invece di 0.37 px.

Sulle tue immagini il guadagno fuori campione è +80% (da 5.5 px a 1.1 px):
la deformazione è reale, non rumore.

### 4. Verifica indipendente (`validation.py`)

Il costo finale di un ottimizzatore dice solo quanto l'ottimizzatore è contento
di sé. Qui invece, **a piena risoluzione**, si prendono 64 finestre da 512 px
in posizioni scelte indipendentemente dalla stima, si applica la trasformazione
e si misura di quanti pixel le due immagini sono ancora disallineate (vedi
"Come si misura il risultato"). In più viene calcolato il costo MIND (la metrica
robusta del progetto) prima e dopo.

Una precisazione sul significato di "indipendente". La verifica è indipendente
per **risoluzione** (piena, contro la metà usata dalla stima), per **dimensione
delle finestre** e per **posizione**, che non dipende da come la stima è andata.
Non è indipendente per i **dati** — sono le stesse immagini, e ogni finestra da
512 px contiene circa 16 dei blocchi usati per stimare — né per il **principio**,
perché usa lo stesso strumento (gradiente più correlazione di fase). È quindi un
controllo di **coerenza**, non una misura di accuratezza rispetto a una verità
esterna. Per quella servono punti annotati a mano su dettagli riconoscibili:
sulla radiografia della Cappellochiaro ce ne sono 221, usati come giudice
indipendente nella sezione RX e nel benchmark delle radiografie.

In `diagnostica/` finiscono anche:

- `overlay_falsocolore.png` — riferimento in rosso, e immagine da modificare allineata in verde: dove
  l'allineamento è buono non si vedono frange rosse/verdi sui bordi;
- `overlay_falsocolore_PRIMA.png` — lo stesso prima dell'allineamento, per
  confronto;
- `overlay_scacchiera.png` — riquadri alternati: le strutture devono
  proseguire senza scalini da un riquadro all'altro;
- `campo_nonrigido.png` — mappa a colori dell'ampiezza della deformazione locale.

### 5. Applicazione a piena risoluzione, in streaming

L'immagine di output viene creata come TIFF non compresso e **mappata in
memoria in scrittura**: ogni tile di output viene calcolato leggendo dal TIFF
sorgente solo la regione che serve, e scritto direttamente al suo posto nel
file. In RAM non c'è mai più di un tile.

Affine e campo non rigido vengono applicati in **un solo `cv2.remap`**: una
sola interpolazione, quindi nessuna perdita di nitidezza aggiuntiva rispetto a
un warp affine seguito da un secondo warp.

## File

| file | contenuto |
|---|---|
| `interfaccia.py` | finestra grafica: riferimento, immagini, cartella, avanzamento, risultati |
| `interfaccia_falsocolore.py` | finestra grafica per comporre IRFC e UVFC |
| `falso_colore.py` | motore del falso colore, anche da riga di comando |
| `test_falsocolore.py` | test del falso colore: composizione per canali identica a quella in decimali |
| `align_batch.py` | stessa elaborazione da riga di comando, con riepilogo finale |
| `align_paintings.py` | allineamento di una coppia (stima → verifica → scrittura) |
| `finestra_strumenti.py` | gli strumenti a mano per un'immagine del lotto: punti di corrispondenza e bordo tracciato (erano la sezione a parte delle radiografie) |
| `verifica_fissa.py` | la verifica a griglia fissa per tutte le bande, con le finestre oltre 10 px, gli avvisi e il confronto fra tentativi |
| `finestra_punti.py` | le due viste affiancate per mettere i punti, con lettura a piena risoluzione e previsione della zona |
| `punti_manuali.py` | i punti messi a mano: posa di partenza (similarità sotto i 6 punti, affine sopra, con scarto robusto), vincolo locale dove la verifica non misura, fascia di bordo e posa dai contorni |
| `image_io.py` | apre qualsiasi formato scegliendo memmap o RAM |
| `apply_transform.py` | riapplica i parametri salvati a un raster della stessa ripresa, con verifica bloccante |
| `tiff_memmap.py` | lettura/scrittura di TIFF giganti via memmap, senza `tifffile` |
| `block_matching.py` | stima: phase correlation a blocchi, fit robusto, campo denso; il campo grossolano a due livelli, stimato per ogni banda e applicato solo dove serve |
| `formati.py` | i formati di uscita: di norma quello dell'ingresso, oppure a scelta |
| `transform_model.py` | modello della trasformazione (affine + campo) e conversioni di scala |
| `validation.py` | verifica a piena risoluzione e immagini di controllo |
| `robust_metrics.py` | MIND / SSC / NGF / NMI |
| `test_pipeline.py` | test end-to-end su dati sintetici con trasformazione nota |
| `calibration_setup.py` | calibrazione camera da scacchiera: non fa parte di questa pipeline, ma è autonomo e può servirti |
| `interfaccia_mosaico.py` | finestra grafica per il mosaico, con lo schema del percorso e le miniature |
| `mosaico.py` | mosaico di tasselli da carrello: coppie adiacenti, stima globale, verifica delle cuciture, livelli per Photoshop |
| `mosaico_omografia.py` | mosaico di tasselli da cavalletto: vicini senza griglia, un'omografia per tassello, distorsione dell'obiettivo, rettilineità, immagine totale e versione raddrizzata |
| `test_mosaico.py` | test del mosaico su tasselli sintetici con posizioni note |
| `test_mosaico_omografia.py` | test del mosaico da cavalletto: errore di posizione vero, freno, distorsione, rettilineità, raddrizzamento, immagine totale |


## Se un giorno non funziona

- **"TIFF compresso"**: ri-esporta senza compressione, oppure installa
  `tifffile` e adatta `tiff_memmap.py`.
- **Residuo finale alto (> 4 px)**: guarda `diagnostica/overlay_falsocolore.png`.
  Se le due immagini sono grossolanamente diverse (inquadrature molto diverse,
  un'immagine ruotata di 90°), nessuna stima locale può funzionare: serve un
  allineamento grossolano preliminare. Se invece sono simili, prova
  `--factor 2 --refine-block 96 --min-conf 0.05`.
- **"Solo N blocchi affidabili"**: le due immagini condividono poca struttura.
  Abbassa `--min-conf`, oppure verifica di non aver invertito `--fixed` e
  `--moving` con immagini di dimensioni diverse.
- **Memoria**: se la macchina ha meno RAM, usa `--factor 4` (picco ~0.6 GB).
