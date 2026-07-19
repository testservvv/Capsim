# Capsim — Simulation einer Kondensatormikrofonkapsel

Lumped-Element-Simulation (elektroakustisches Ersatzschaltbild) einer
Kondensatormikrofonkapsel mit Streamlit-Oberfläche.

## Komponenten

| Datei | Inhalt |
|---|---|
| `microphone_capsule.py` | Physik-Klasse `MicrophoneCapsule` (ABCD-Kettenmatrizen, Zwikker–Kosten-Lochimpedanzen, Škvor-Squeeze-Film **oder** 2D-Reynolds-Feldmodell, elektrostatische Wandlung mit Pull-in, Gehäusebeugung) — eigenständig lauffähig mit Testlauf |
| `app.py` | Streamlit-GUI: Parameter-Seitenleiste, Bode-Plot, Polardiagramm, Projekt speichern/laden (JSON), CSV-Export |
| `translations.py` | Übersetzungstabelle der GUI (Englisch/Deutsch) |

## Sprache / Language

Die Oberfläche ist zweisprachig (**Englisch** als Standard, Deutsch
umschaltbar) — die Sprachwahl steht oben in der Seitenleiste. Sie ist eine
reine Anzeige-Einstellung und wandert **nicht** in die Projektdateien: die
kanonischen Auswahl-Werte (Architektur, Material, axialer Körper …) bleiben
sprachunabhängig gespeichert, sodass Projekte zwischen beiden Sprachen
voll austauschbar sind. Auch der Diagnose-Summary (`MicrophoneCapsule.
summary(lang=…)`) folgt der Sprachwahl.

The interface is bilingual (**English** default, German selectable via the
language switch at the top of the sidebar). The language is a display-only
setting and is **not** written to project files, so projects stay fully
interchangeable between both languages.

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
- **3D:** volles (r, φ)-Sandwich der durchbohrten Elektrode(n): alle
  Spaltfilme UND die Membran(en) als Felder, Durchgangs- und Sacklöcher
  **diskret** an ihren Positionen (Azimutwinkel als dokumentierte
  Konvention, da nicht gezeichnet). Bauformen der Doppelmembran:
  die **einteilige** Elektrode (Debenham-Typ, `center_gap = 0`, zwei
  Filme) und seit Gegenprobe 22 die **zweiteilige** Elektrode (K67-Typ,
  `center_gap > 0`): der Zwischenspalt wird als dritter Reynolds-Film
  gerechnet, Stufenbohrungen als Zweitor-Kette je Loch (Senkung als
  Leitungsstück + Karal-Stufe + enger Kern), und die Elektrodenhälften
  sind gegeneinander **verdreht** (`half_rotation_deg`, Standard eine
  halbe Lochteilung 180°/n wie an der realen K67). Seit Gegenprobe 23
  rechnet der Löser auch **single/dual**: ein Membranfeld, ein Film je
  Backplate; die Durchgangslöcher münden als Zweitor-Ketten in einen
  **Sammelknoten**, dessen Abschluss die baugleiche Lumped-Kette des
  1D/2D-Pfads bildet (Spacer/Rückplatte, Gewebe, Laufzeitglied,
  Hohlraum; vorn Strahlung + Gewebe). Verankert über exakte
  Port-Reziprozität (Maschinengenauigkeit), den K103-dicht-Grenzfall
  (3D ≡ 1D auf wenige %), die geschlossene Rückseite (exakte Kugel,
  Betrag UND Phase == 1D — die 3D-Ausgänge folgen seither der
  Ketten-Vorzeichenkonvention aller Modelle) und die
  D_r-Übereinstimmung mit 2D (~1 %); die absolute Empfindlichkeit
  trägt die dokumentierte Membranfeld-Klasse (±2–3 dB), das
  PfadVERHÄLTNIS (Pattern) ist robust. Der Löser löst die
  azimutale Zuströmung zu den einzelnen Bohrungen und die dadurch
  teilentkoppelten Sacklöcher auf — bedämpft die interne
  Helmholtz-Resonanz realistisch und macht die Mündungs-Engstellen der
  Löcher (und ihre Entlastung durch Freistiche) explizit sichtbar.
  **Befund zur Verdrehung** (beantwortet die alte Frage, was der
  Versatz der Bohrbilder bewirkt): zeigen die Durchgangslöcher beider
  Hälften aufeinander (0°), kurzschließen sie den Nieren-Phasenschieber
  durch den Zwischenspalt — flache 180°-Auslöschung (−7 dB) bei hoher
  Empfindlichkeit (34 mV/Pa); schon die halbe Teilung (3° bei 60
  Löchern) zwingt den Pfad durch den Zwischenspalt-Film und liefert
  −20 dB bei 22 mV/Pa, nahe am homogenisierten 2D-Modell (−26 dB,
  21 mV/Pa), das versetzte Arrays stillschweigend annimmt.
  Verifiziert über Reziprozität (±1 %), Gitterkonvergenz, die
  Grenzfälle einteilig ≡ zweiteilig-ausgerichtet (5-µm-Spalt, 2 %) und
  Stufenbohrung → glatte Bohrung (winzige Senkung, 0,8 %) sowie die
  Gültigkeits-Gatter im Testlauf. DEUTLICH langsamer (LU-Faktorisierung
  je Frequenzpunkt: einteilig ~24 000 Unbekannte, ~1–2 s; K67-Typ mit
  drittem Film und feinerer Azimut-Auflösung bis ~10 s bei 60 Löchern)
  — in der GUI die Frequenzpunkte reduzieren. Absolute Empfindlichkeit
  weicht modellbedingt ≤ 2–3 dB von 1D/2D ab (Membran als Feld statt
  Grundmode).

### Position des rückwärtigen Gewebes (`fabric_rear_position`)

Das Gewebe hinter der Backplate kann an zwei Orten sitzen (GUI-Option
im Gewebe-Panel, Gegenprobe 24): **an der Backplate** (Bestand — das
Tuch überspannt die volle Zylinderbohrung, Z = Rayl/S_Bohrung) oder
**über den Einlassöffnungen** (außen auf den Hohlraum-Einlasslöchern,
K103-Rückplattenlöchern bzw. bei Direktmündung den Durchgangslöchern).
Am Einlass wird nur die **Lochfläche** durchströmt — dasselbe Tuch ist
um den Faktor S_Bohrung/S_Löcher hochohmiger — und der Widerstand liegt
**hinter** den Shunt-Volumina von Laufzeitrohr/Hohlraum, in Serie mit
der Einlassloch-Masse (bedämpft deren Helmholtz-Resonator direkt).
Referenzfall (Nieren-Single, 25 Rayl, 60 Einlässe ⌀0,6 mm, Faktor 26):
an der Backplate praktisch transparent, am Einlass steigt die interne
Laufzeit von 0,64 auf 3,0 des externen Wegs und die Empfindlichkeit um
+38 % (die rückwärtige Auslöschung bricht ein). Ohne Gewebe (0 Rayl)
ist die Position exakt wirkungslos; die Doppelmembran-Bauform hat
keinen rückwärtigen Einlass (Gatter). 1D/2D/3D teilen die Kette.

Die GUI rechnet mit **Fortschrittsbalken** (steht ab Sekunde null, auch
während des Modellaufbaus) und **zweistufigem Cache** im Session-State:
das Kapsel-Objekt je Bau-Parametersatz (Konstruktor mit Elektrostatik,
Pull-in und 3D-Gitteraufbau) und die Ergebnisse (Frequenzgang,
Richtdiagramme, Diagnose, Summary) je vollem Parametersatz. Reruns ohne
Parameteränderung — Widget-Interaktionen, Umschalten der Normierung —
rechnen gar nichts mehr; auch der Diagnose-Summary-Text kommt aus dem
Cache (Streamlit führt eingeklappte Expander-Inhalte bei jedem Rerun
aus, und summary() enthält bei 3D eine volle LU-Lösung). Gerade beim
3D-Modell macht erst das die Bedienung auf Streamlit Cloud praktikabel.
Im Projekt-Panel setzt ein Reset-Button nach einem Bestätigungsschritt
alle Kapselparameter auf null (bzw. auf den kleinsten baubaren Wert) —
eine leere Leinwand für Entwürfe von Grund auf.

### Laufzeit-Indikator (GUI) / `delay_diagnostics()`

Unter den Kennwerten zeigt die GUI die beiden Laufzeiten des
Nieren-Phasenschiebers: **extern** (Front→Rück um den Kapselkörper, aus
der axialen Körperbeugung) und **intern** (Phasensteigung der
Netzwerk-Rückübertragung D_r), beide als Phase bei 1 kHz ausgewertet,
samt Verhältnis intern/extern. Deutung (Gegenprobe 17, numerisch
verifiziert): **≈ 1** — angepasst, tiefste Auslöschung bei 180°
(opti.json: 1,00); **< 1** — interne Laufzeit zu kurz, das
Pattern-Minimum wandert vor 180° (K67 nominal: 0,96, Minimum ~165°);
**> 1** — zu lang, das Minimum bleibt bei 180° gepinnt, wird aber
flacher (cardiodtest 45-µm-Spacer: 1,24). Die interne Laufzeit ist
frequenzabhängig (RC-Phasenschieber mit Filmträgheit, kein reines
Laufzeitglied) — der Klassen-Methode `delay_diagnostics(f_probe_hz=…)`
kann eine andere Sondenfrequenz übergeben werden.

Weil eine einzelne Zahl täuschen kann, wenn die **interne
Helmholtz-Resonanz** (Durchgangsloch-Trägheit gegen Spalt- und
Blindloch-Nachgiebigkeit) im Band liegt, zeigt die GUI zusätzlich:

- **Richtwirkung über die Frequenz**: Pegel bei 90° und 180° relativ zu
  0° als Bode-Kurven (mit −6-dB-Referenz der idealen Niere),
- **Phasenschieber-Plot**: Betrag und Phase von D_r gegen das externe
  Ziel G(180°) — die Kreuzung der Phasen und der Betragseinbruch an der
  Resonanz sind direkt sichtbar,
- **f_H als Kennwert** (90°-Phasendurchgang von D_r, Klassen-Methode
  `helmholtz_resonance_hz`), mit Warnhinweis, wenn sie im
  Übertragungsband liegt: dann morpht die Pattern-Form über die
  Frequenz (Superniere → breite Niere → Kugel); Abhilfe sind
  Stufenbohrungen, dünnere Platten oder größere/mehr Durchgangslöcher.

Alle Kurven entstammen EINEM Durchlauf je Frequenz
(`angle_responses`: Superposition q = a·p_front + b·p_rück mit
winkelunabhängigen a, b — beim 3D-Modell keine Mehrkosten).

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
Dieser Parametersatz (im 2D-Feldmodell) ist zugleich die
**Voreinstellung beim App-Start**.
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
Kapselkopf-Durchmesser als Beugungskörper. Ergebnis mit den nominellen
inneren Maßen: Ruhekapazität C₀ = 50,3 pF (trifft den nachgemessenen
Wert), Niere **−6,4/−17,2/−25,7 dB @ 90/135/180° (1 kHz)** — praktisch
die publizierten U87-Werte —, glatter Präsenzpeak +3,6 dB @ 11,4 kHz
(roh; Korb und Elektronik — nicht modelliert — glätten auf die
veröffentlichten ~+2..3 dB), Empfindlichkeit 20,9 mV/Pa. Das tiefste
Minimum liegt in den Mitten knapp vor 180° (~160°);
`examples/cardiodtest.json` pinnt es mit Spacer 45 µm exakt auf 180°.
Über „Projekt laden" importierbar.

`examples/cardiodtest.json` — identisch zum nominellen K67-Datensatz
bis auf **ein einziges Maß**: den Spacer zwischen den Elektrodenhälften
(45 statt 50 µm — das am wenigsten sicher verifizierte innere Maß,
plausibel 40–65 µm). Damit trifft die interne Phasenschieber-Laufzeit
die externe exakt, und die Null sitzt bei **allen** Frequenzen im
Übertragungsband auf 180° (−17/−19/−19/−17 dB @ 250/500/1k/2k,
90° = −5,3 dB, C₀ = 50,3 pF, Frequenzgang flach). Die Niere entsteht
dabei vollständig aus den akustischen Parametern (s. Abschnitt
„Nierenbildung").

### Nierenbildung: interne trifft externe Laufzeit — beides hergeleitet

Die Niere der Doppelmembran-Bauform entsteht, wenn die **interne**
akustische Laufzeit des Phasenschieber-Netzwerks (Bohrungen, Spaltfilme,
Spacer — die Reibungs-/Nachgiebigkeits-Verzögerung des Rückschalls auf
dem Weg zur Frontmembran) die **externe** Laufzeit des Schalls um den
Kapselkörper trifft. Beide Größen fallen **aus den physikalischen
Parametern** — es gibt keinen Fit-Koeffizienten:

- **Intern:** die Korrektur des **Rückwärts-Durchlaufs** in der 2D-Kette
  (`_abcd_reverse`): der Rückspalt wird als Port-Tausch `[D B; C A]`
  durchlaufen (identisch zur umgekehrten Elementreihenfolge des
  1D-Pfads), **nicht** als Matrix-Inverse — deren negative (aktive)
  Elemente löschten zuvor Laufzeit und Dämpfung des Hinwegs exakt aus,
  sodass die Niere nur über eine f_res-abhängige Spaltasymmetrie-Krücke
  entstand und der Rückzweig eine ungedämpfte Resonanzüberhöhung zeigte.
- **Extern:** die Beugung um die **axiale Körperausdehnung**
  (`_axial_body_transfer`): der Pol-zu-Pol-Transfer der exakten
  Morse-Streureihe an der Kugel mit Durchmesser
  `d_ext = axialer Membranabstand + 2·Klemmringdicke`. Für ka → 0 ergibt
  das automatisch den bekannten 3/2-Dipolfaktor (effektiv 1,5·d_ext,
  K67: 18,3 mm) samt konsistenter Amplituden-Asymmetrie. Die
  Ring-/Kalotten-Platzierung auf der breiten R_body-Kugel wäre falsch
  (gemessen: ~0,6·d_ext bzw. ~2,5·R_body — die Breitenkugel überzeichnet
  die axiale Ausdehnung der dünnen Scheibe).

Mit beiden Korrekturen ist die Nullstelle bei 180° **unabhängig von
Membranresonanz und Polarisationsspannung** (im Testlauf verankert), und
die nominelle K67 erreicht die publizierte Tiefe (−26 dB @ 180°/1 kHz)
ohne jede Kalibrierung. Der aktive Membrandurchmesser (26 mm) und der
Außendurchmesser inklusive Klemmring (34 mm K67 / 32 mm Debenham)
bleiben getrennte Größen: `membrane_diameter` bzw.
`body_diameter` + `clamp_ring_width`.

**Sphäroid als Referenzkörper (`axial_body_model="spheroid"`):** Für die
frei stehende Scheibe ist die Streuung am **starren oblaten Sphäroid**
(radiale Halbachse R_body, axiale d_ext/2) exakt implementiert — mit
eigenen Spezialfunktionen, weil scipys `obl_rad2` für ξ₀ < 1 versagt:
Flammer-Rekursion als Tridiagonal-Eigenproblem (Querprobe `obl_cv`,
1e-14), R³ per Hankel-Reihen-Start und Einwärts-Integration der
Radial-ODE, Wronski-Selbstprüfung je Mode (< 1e-5 übers Band). An den
Polen verschwinden alle azimutalen Ordnungen m > 0 — es bleibt eine
m=0-Reihe analog zur Morse-Kugel. Verifiziert am Kugel-Grenzfall
(b → a: Morse-Reihe auf 2e-3) und am Dünne-Scheiben-Grenzwert (LF-Distanz
am Pol → 4a/π, klassisches Resultat). **Befund (Gegenprobe 20):** die
freie K67-Scheibe hätte d_eff ≈ 32 mm — gegen die interne Laufzeit
(~17 mm) ergäbe das eine Superniere mit Minimum bei ~123°, im
Widerspruch zur realen U87. Der dahinterliegende **Mikrofonkörper
unterbindet den Scheibenrand-Umweg** — deshalb bleibt die d_ext-Kugel
(montierte Kapsel) Standard; das Sphäroid ist der dokumentierte
Referenzfall der freien Scheibe (GUI: „Axialer Körper").

**Montagetreues BEM (`axial_body_model="bem"`):** Die dritte Stufe
rechnet den Front-Rück-Transfer per **axisymmetrischem
Randelementverfahren** (m = 0) auf der tatsächlichen Kontur
*Kapselkopf-Scheibe + Mikrofonkörper-Zylinder* (⌀/Luftspalt/Länge als
Parameter `bem_body_*`, GUI-Felder; ⌀ 0 = freier Kopf). Direkte
Kirchhoff-Helmholtz-Kollokation mit verrundeten Kanten, Diagonale aus
der statischen Raumwinkel-Identität, CHIEF-Punkte gegen irreguläre
Frequenzen; Membranmittelwerte sind exakte m=0-Projektionen.
**Validiert gegen beide exakten Referenzen** (Gegenprobe 21):
Kugelkontur trifft die Morse-Reihe und Sphäroidkontur die
Sphäroid-Reihe auf < 2·10⁻⁴. Ergebnis für die K67 (56-mm-Körper):
d_eff läuft mit dem Montagespalt von 16,9 mm (5 mm Spalt) über
23,3 mm (15 mm) bis 33,3 mm (frei) — **die d_ext-Kugel (18,3 mm)
entspricht ~7 mm Spalt, genau der realen Sattelmontage**. Damit ist
der Kugel-Standard quantitativ begründet, und abweichende Aufbauten
(Messabstand zum Body, Grenzflächen-Montagen) sind ehrlich rechenbar.
Kosten: ~1–2 s je Frequenzpunkt (~200 Elemente), Ergebnisse werden
gecacht.

### Nierenform der dünnen Doppelmembran-Scheibe

Front- und Rückmembran der K67 sitzen auf den zwei Flächen einer nur
~12 mm dünnen Scheibe. Für den Front-Rück-Gradienten (der die Niere
erzeugt) ist die **axiale Ausdehnung** dieser Scheibe maßgeblich, nicht
die viel größere Breite des Kugel-Ersatzgehäuses der Beugungsrechnung.
Würde man den rückwärtigen Einlass wie bei den Einzelmembran-Bauformen
als Ring/Kalotte auf der breiten Beugungskugel platzieren, zöge das die
Nullstelle weit vor 180° (Superniere) bzw. ließe sie zu flach werden.
Das Modell trennt deshalb bei `dual_diaphragm` die beiden Skalen: die
R_body-Kugel liefert die gemeinsame HF-Bündelung/Druckstau, der
Front-Rück-Gradient kommt aus dem Pol-zu-Pol-Transfer der Kugel mit der
**korrekten axialen Ausdehnung** `d_ext` (s. `_axial_body_transfer`) —
so bleibt die Nullstelle im Grundton-/Mittenbereich nahe 180°, und erst
zu hohen Frequenzen bündelt die Niere (wie real).

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

Drei Verfeinerungen (Gegenprobe 19, jeweils fit-frei):

- **Fok/Melling-Mündungswechselwirkung:** die Flanschkorrektur 0,85·r
  gilt für die einsame Mündung; im Locharray überlappen die Nahfelder
  und die mitschwingende Masse sinkt um den Fok-Faktor F(ξ),
  ξ = r/r_Zelle aus der Fläche je Loch. Angewandt auf die array-
  seitigen Mündungen der Durchgangs- und Rückplattenlöcher (K67:
  F ≈ 0,74) — die filmseitigen Mündungen behalten ihre Konvention.
- **Lokales Spaltprofil h(r) im 2D-Film:** die polarisierte Membran ist
  statisch durchgebogen; das Feldmodell rechnet die Zell-Leitwerte mit
  dem örtlichen h(r) = h − w₀·φ(r) (h³-Wirkung!) statt des
  Flächenmittels — die Bias-Kopplung an die Richtcharakteristik ist
  damit quantitativ.
- **Exakte Leitungen für alle Radien:** Laufzeitglied, Hohlraum und
  Reststücke rechnen mit der vollen Zwikker–Kosten-Form statt der
  Kirchhoff-Asymptotik (numerisch robust über eine
  Grenzschicht-Asymptotik oberhalb der Schubzahl 600); summary() nennt
  zusätzlich die erste azimutale Quermode des Hohlraums als ehrliche
  1D-Gültigkeitsgrenze und die exakte J₀-Modalfrequenz der Membran
  (der Lumped-Kolbenfaktor 4/3 liegt ~1,9 % darüber).

`examples/debenham_stereo_condenser.json` — Braunmühl-Weber-Kapsel aus
Debenham/Robinson/Stebbings, *A Stereo Condenser Microphone* (Hi-Fi
News): einteilige durchbohrte Mittelelektrode (`center_gap = 0`), 1"-
Membranen, je Seite 12 Durchgangs- + 46 Dämpfungslöcher auf den echten
Lochkreisen der Konstruktionszeichnung (0.860/0.688/0.516/0.344/0.172"),
50 V — und der **Clearance-Ring** der Zeichnung: ein
Stirnflächen-Freistich am Elektrodenrand (0,038 mm Abtrag über die
äußeren 1,27 mm, GUI-Felder „Clearance-Ring", Klasse
`clearance_ring_*`). Dieser Freistich entlastet die
Mündungs-Engstellen der wenigen engen Durchgangslöcher im 38-µm-Spalt —
sie waren der begrenzende Widerstand des Nieren-Phasenschiebers. Damit
trifft das Modell Fig. 9 bei 100 Hz fast exakt (−1,3/−5,4/−11,7/−13,6
@ 45/90/135/180° vs. −1/−3/−10/−12), liefert −30 dB @ 250 Hz und die
gemessene HF-Bündelung (10 kHz: Null 143° vs. 142°); bei 1–2 kHz bleibt
es ~10 dB flacher als der Artikel (die axialsymmetrische
Homogenisierung der 12 diskreten Löcher erfasst dort nur einen Teil der
Mündungs-Entlastung — offener Rest, dokumentiert). Der strengere
**3D-Löser** ordnet das ein: mit *nur* dem Rand-Freistich der Zeichnung
bleibt die Niere flach (−4 dB @ 1 kHz — die inneren Lochmündungen
bleiben verengt); deckt der Freistich dagegen alle Lochkreise ab (im
GUI-Clearance-Ring einstellbar, physikalisch ≈ angesenkte/entgratete
Mündungen), wird sie breitbandig tief (−13…−15 dB @ 250 Hz–2 kHz, Null
exakt 180°). Die reale Kapsel dürfte solche Mündungs-Fasen haben (in
Zeichnungen selten bemaßt).

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
