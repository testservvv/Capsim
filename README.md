# Capsim — Simulation einer Kondensatormikrofonkapsel

Lumped-Element-Simulation (elektroakustisches Ersatzschaltbild) einer
Kondensatormikrofonkapsel mit Streamlit-Oberfläche.

## Komponenten

| Datei | Inhalt |
|---|---|
| `microphone_capsule.py` | Physik-Klasse `MicrophoneCapsule` (ABCD-Kettenmatrizen, Zwikker–Kosten-Lochimpedanzen, Škvor-Squeeze-Film **oder** 2D-Reynolds-Feldmodell, elektrostatische Wandlung mit Pull-in, Gehäusebeugung) — eigenständig lauffähig mit Testlauf |
| `app.py` | Streamlit-GUI: Parameter-Seitenleiste, Bode-Plot, Polardiagramm, Projekt speichern/laden (JSON), CSV-Export |

## Spaltfilm-Modell: 1D vs. 2D

Der Luftspalt zwischen Membran und Backplate kann auf zwei Arten
gerechnet werden (umschaltbar per `squeeze_model` bzw. GUI-Schalter):

- **1D (Standard):** ein Lumped-Element (Škvor-Widerstand + Nachgiebigkeit
  + Lochimpedanz). Schnell; für dichte, gleichmäßige Lochmuster
  ausreichend und für die validierten Beispiele verwendet.
- **2D:** das Druckfeld im Spalt wird als *modifizierte Reynolds-Gleichung*
  (Homentcovschi & Miles, JASA 2004; Bao) axialsymmetrisch als
  Feldgleichung gelöst — inklusive Zell-Engstellenwiderstand je Bohrung
  (im dichten Grenzfall wird Škvor reproduziert, verifiziert im Test),
  viskoser Trägheit des Spaltfilms (Schlitz-Zwikker–Kosten; bei 25 kHz
  bis Faktor 4 mit −73° Phase) und polytroper Kompressibilität
  (isotherm→adiabatisch). Trennt den Nachgiebigkeits-Rückweg (Blind-
  löcher, lokal) vom Rückkopplungsweg (Durchgangslöcher) und nutzt bei
  Einzel-Backplate-Architekturen die Lochkreis-Radien (PCD) — je Lochtyp
  auch MEHRERE Lochkreise (in der GUI per ➕-Button, in der Klasse über
  `through_hole_rings`/`blind_hole_rings`), um reale Bohrbilder
  nachzubilden. Für die Doppelmembran-Bauform (K67) ist das Feldmodell
  die EMPFOHLENE Berechnung: Oberhalb ~8 kHz ist die Wellenlänge kleiner
  als die Platte — das interne Luftpolster ist dann kein Lumped-Element
  mehr, und das 1D-Kettenmodell erzeugt eine unphysikalisch scharfe
  Absorber-Kerbe (Rückmembran gegen inneres Polster bei
  f ≈ f_res/√δ); das Feldmodell löst den radialen Druckverlauf auf und
  liefert den glatten Präsenzpeak der echten Kapsel — bei gleicher
  Niere (Null bei 180°). Reziprok und passiv (im Test geprüft).
  Braucht SciPy.

## Installation & Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Nur das Physikmodell (ohne GUI) testen:

```bash
python microphone_capsule.py
```

## Beispielprojekte

`examples/u87_k67_projekt.json` — recherchierter und modellvalidierter
Parametersatz einer Neumann-K67/K870-Kapsel (U87Ai, Nierenmodus) in der
echten Doppelmembran-Bauform: zwei 26-mm-Membranen (6 µm Mylar) außen,
zwei innenliegende Backplate-Hälften (je ~4 mm) mit 50-µm-Spacer, 60 V.
Die passive Rückmembran bildet das Phasenschiebernetzwerk, und das
Lochbild ist die verifizierte **Stufenbohr-Geometrie**: je Seite 120
Bohrungen ⌀1,3 mm × 3,7 mm tief, jede zweite mit konzentrischem
0,6-mm-Durchbruch am Grund (GUI-Schalter „Stufenbohrung", Klasse
`through_holes_stepped` — enges Rohr nur über die Restdicke, Senkung
als Sackvolumen, korrekte Stirnporosität, eine Škvor-Senke je Bohrung).
Das Lochbild wird als **120 Sacklöcher je Seite** eingegeben
(`n_blind = 120` = Gesamtzahl), davon **60 durchgebohrt**
(`n_through = 60`, Stufenbohrung an) — genau wie die reale K67 (jedes
zweite Sackloch mit 0,6-mm-Durchbruch). Diese Zählweise gilt **nur bei
aktiver Stufenbohrung**; sonst bleiben beide Lochtypen unabhängig.
Vor beiden Membranen sitzen **Klemmringe** (je 2 mm dick, 2 mm breit,
Parameter `clamp_ring_thickness`/`clamp_ring_width`) — sie versenken
die Membranen und definieren die ehrliche externe Distanz
`d_ext = axial + 2·Ringdicke` (s. u.). Gerechnet wird mit dem
**2D-Feldmodell** (`squeeze_2d = true`) und dem vermessenen
Kapselkopf-Durchmesser als Beugungskörper. Ergebnis: Ruhekapazität
C₀ = 50 pF (trifft den nachgemessenen Wert), Niere mit **Null bei 180°
von 100 Hz bis 2 kHz** (−14…−16 dB mit den nominellen inneren Maßen),
glatter Präsenzpeak +4,0 dB @ 11,4 kHz (roh; Korb und Elektronik —
nicht modelliert — glätten auf die veröffentlichten ~+2..3 dB). Die
Nierentiefe reagiert empfindlich auf die nicht sicher verifizierten
inneren Maße (Spacer-Höhe, Sacklochvolumen) — `examples/cardiodtest.json`
zeigt eine Variante (Spacer 40 µm, kleinere Sacklöcher), die **−23…−31 dB
@ 180°** erreicht. Über „Projekt laden" importierbar.

`examples/cardiodtest.json` — die K67-Bauform mit auf tiefe
180°-Auslöschung abgestimmten **inneren** Maßen (Spacer 40 µm,
Sacklöcher ⌀1,0 mm × 2,0 mm — alles innerhalb der dokumentierten
Unsicherheit der echten Kapsel): Null bei 180° von 100 Hz bis 4 kHz,
−31 dB @ 500 Hz, 90° = −5,9 dB (Lehrbuch-Niere), Frequenzgang flach.
Die Niere entsteht dabei vollständig aus den internen akustischen
Parametern (s. Abschnitt „Nierenbildung").

### Nierenbildung: interne Laufzeit aus den akustischen Parametern

Die Niere der Doppelmembran-Bauform entsteht, wenn die **interne**
akustische Laufzeit des Phasenschieber-Netzwerks (Bohrungen, Spaltfilme,
Spacer — die Reibungs-/Nachgiebigkeits-Verzögerung des Rückschalls auf
dem Weg zur Frontmembran) die **externe** geometrische Wegdifferenz
`d_ext = axialer Membranabstand + 2·Klemmringdicke` trifft. Beide Größen
fallen **aus den physikalischen Parametern** — es gibt keinen
Fit-Koeffizienten. Möglich macht das die Korrektur des
**Rückwärts-Durchlaufs** in der 2D-Kette (`_abcd_reverse`): der
Rückspalt wird als Port-Tausch `[D B; C A]` durchlaufen (identisch zur
umgekehrten Elementreihenfolge des 1D-Pfads), **nicht** als
Matrix-Inverse — deren negative (aktive) Elemente löschten zuvor
Laufzeit und Dämpfung des Hinwegs exakt aus, sodass die Niere nur über
eine f_res-abhängige Spaltasymmetrie-Krücke entstand und der Rückzweig
eine ungedämpfte Resonanzüberhöhung zeigte. Mit der Korrektur ist die
Nullstelle bei 180° **unabhängig von Membranresonanz und
Polarisationsspannung** (im Testlauf verankert), und die Tiefe wird
allein von den inneren Maßen bestimmt: Spacer-Höhe und Sackloch-Volumen
laden den internen Phasenschieber — `examples/cardiodtest.json` zeigt
eine darauf abgestimmte K67-Variante. Der aktive Membrandurchmesser
(26 mm) und der Außendurchmesser inklusive Klemmring (34 mm K67 /
32 mm Debenham) bleiben getrennte Größen: `membrane_diameter` bzw.
`body_diameter` + `clamp_ring_width`.

### Nierenform der dünnen Doppelmembran-Scheibe

Front- und Rückmembran der K67 sitzen auf den zwei Flächen einer nur
~8 mm dünnen Scheibe. Für den Front-Rück-Gradienten (der die Niere
erzeugt) ist die **geometrische axiale Laufzeit** dieser Scheibendicke
maßgeblich, nicht der viel größere Umweg um das Kugel-Ersatzgehäuse der
Beugungsrechnung. Würde man den rückwärtigen Einlass wie bei den
Einzelmembran-Bauformen als Ring/Kalotte auf der Beugungskugel
platzieren, zöge das die Nullstelle vor 180° und machte aus der Niere
eine Superniere. Das Modell nutzt deshalb bei `dual_diaphragm` die
Scheiben-Laufzeit für die Front-Rück-Phase und die Kugelbeugung nur
noch für die gemeinsame HF-Bündelung/Druckstau — so bleibt die
Nullstelle im Grundton-/Mittenbereich korrekt bei 180°, und erst zu
hohen Frequenzen bündelt die Niere (wie real).

## Verlustmechanismen (vollständig erfasst)

Neben Zwikker–Kosten-Rohrreibung und Škvor-Spaltfilm rechnet das
Modell: **viskose Mündungswiderstände** (Sampson/Roscoe-Kriechströmung,
Weissberg-Zusatzlänge 3πr/16 je Mündung, thermoviskos ausgewertet —
bei kurzen engen Bohrungen vergleichbar mit dem Rohrwiderstand selbst;
nur an Mündungen in große Volumina, im Spaltfilm deckt die
Škvor-/Zell-Ausbreitung die Zuströmung ab), **Sacklöcher/Senkungen als
endseitig geschlossene thermoviskose Leitungsstubs** (verteilte
Reibung, LF-Grenzfall R/3, Nachgiebigkeit isotherm→adiabatisch mit
Relaxationsdämpfung, λ/4-Verhalten) und die **laterale Trägheit der
Spaltluft** im 1D-Modell (Schlitz-Zwikker–Kosten-Korrektur Φ(ω),
identisch zum Filmleitwert des 2D-Feldmodells). Alle Grenzfälle sind
im Testlauf verifiziert (Gegenprobe 12).

`examples/debenham_stereo_condenser.json` — Braunmühl-Weber-Kapsel aus
Debenham/Robinson/Stebbings, *A Stereo Condenser Microphone* (Hi-Fi
News): einteilige durchbohrte Mittelelektrode (`center_gap = 0`), 1"-
Membranen, je Seite 12 Durchgangs- + 46 Dämpfungslöcher auf den echten
Lochkreisen der Konstruktionszeichnung (0.860/0.688/0.516/0.344/0.172"),
50 V. Nach der Port-Tausch-Korrektur liefert das Modell die Nierenform
(Null bei 180°, flacher Frequenzgang, HF-Bündelung wie Fig. 9), aber
unterhalb 2 kHz nur −2…−3 dB Tiefe statt der im Artikel gemessenen
−23 dB: die interne Phasenschieber-Laufzeit dieser einteiligen
Elektrode fällt im Modell zu lang aus (die 46 großvolumigen
Dämpfungs-Sacklöcher belasten den Rückkopplungsweg zu den nur 12 engen
Durchgangslöchern). Ein ehrlich offener Punkt — nicht mehr durch
Fit-Parameter kaschiert. Dieser Parametersatz ist zugleich die
Voreinstellung beim App-Start.

`examples/k103_bauform_demo.json` — Demonstration der **K103-Bauform**
(Neumann TLM 103): Einzelmembran-Niere auf K87-Basis, deren Rückseite
statt einer Rückmembran durch einen engen **Spacer** und eine massive,
gelochte **Rückplatte** abgeschlossen ist; die Plattenlöcher münden
direkt ins rückwärtige Schallfeld. Kein validierter K103-Parametersatz
(die inneren Maße sind nicht veröffentlicht), sondern eine plausible
Vorlage zum Abstimmen: der Spacer ist mit R ∝ 1/h³ das Stellglied des
Nieren-Phasenschiebers. Eine Rückplatte ohne Löcher verschließt die
Kapsel (Druckempfänger). Beide Elemente sitzen im Abschnitt
„Rückseite & Laufzeitglied".

Im Nierenmodus ist nur die Frontmembran polarisiert; die Leerlauf-
Empfindlichkeit der Kapsel liegt im niedrigen mV/Pa-Bereich. Datenblatt-
Empfindlichkeiten gelten am Verstärkerausgang (Gain nicht modelliert),
die absolute Empfindlichkeit ist daher nur näherungsweise.

## Projektdateien

Projekte werden als menschenlesbares JSON gespeichert
(`capsim_projekt.json`) und können über die Seitenleiste wieder geladen
werden. Die CSV-Exporte (Frequenzgang, Richtdiagramm) sind wahlweise im
internationalen Format (Komma/Punkt) oder Excel-DE-Format
(Semikolon/Dezimalkomma) verfügbar.
