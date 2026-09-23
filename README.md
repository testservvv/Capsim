# Capsim — Simulation einer Kondensatormikrofonkapsel

Lumped-Element-Simulation (elektroakustisches Ersatzschaltbild) einer
Kondensatormikrofonkapsel mit Streamlit-Oberfläche.

## Komponenten

| Datei | Inhalt |
|---|---|
| `microphone_capsule.py` | Physik-Klasse `MicrophoneCapsule` (ABCD-Kettenmatrizen, Zwikker–Kosten-Lochimpedanzen, Škvor-Squeeze-Film **oder** 2D-Reynolds-Feldmodell **oder** 3D-(r,φ)-Feldlöser mit diskreten Löchern, elektrostatische Wandlung mit Pull-in, Gehäusebeugung) — eigenständig lauffähig mit Testlauf |
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
- **Gültigkeitsgrenze von 1D und 2D:** beide verschmieren die Löcher.
  Bei spärlichen Lochbildern und weicher Membran beult sich die Membran
  zwischen den Löchern örtlich aus, was nur der 3D-Löser abbildet. Die
  Grenzfrequenz f_hom (`homogenization_limit()`, Gegenprobe 48) steht in
  `summary()`; liegt sie im Hörband, warnen Modell und GUI.
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
  sind gegeneinander **verdreht** (`half_rotation_deg`; Standard seit
  Gegenprobe 48 kreisweise so, dass die Durchgangslöcher der
  Gegenseite über den Sacksenkungen liegen — die frühere pauschale
  halbe Teilung 180°/n gilt nur für einen einzigen Lochkreis). Seit
  Gegenprobe 23
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
  durch den Zwischenspalt — das Minimum wandert auf ~106°. Versetzte
  Hälften legen es auf 180°; wie TIEF es wird, hängt an der nicht
  dokumentierten Kernlage im Zwischenspalt (vollständig versetzt
  −11 dB, teilweise fluchtend −29 dB wie im 2D-Modell, s. Gegenprobe
  48; auf dem feinen Gitter −14,6 bzw. −28,5 dB, s. Gegenprobe 50). Die früher hier genannten −20 dB bei „3°" stammten aus einem
  3D-Stand, in dem die Mündungen abgeschnitten und nicht äquipotential
  waren und die Kerne sich überlappten.
  Verifiziert über Reziprozität (±1 %), Gitterkonvergenz, die
  Grenzfälle einteilig ≡ zweiteilig-ausgerichtet (5-µm-Spalt, 2 %) und
  Stufenbohrung → glatte Bohrung (winzige Senkung, 0,8 %) sowie die
  Gültigkeits-Gatter im Testlauf. DEUTLICH langsamer (LU-Faktorisierung
  je Frequenzpunkt: einteilig ~24 000 Unbekannte, ~1–2 s; K67-Typ mit
  drittem Film und feinerer Azimut-Auflösung bis ~10 s bei 60 Löchern)
  — in der GUI die Frequenzpunkte reduzieren. Absolute Empfindlichkeit
  weicht modellbedingt ≤ 2–3 dB von 1D/2D ab (Membran als Feld statt
  Grundmode). **Gitter:** grob (Standard) oder fein (`grid_3d="fine"`,
  GUI-Schalter „Feines 3D-Gitter"), s. Gegenprobe 50.

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

### Eigenrauschen (`self_noise()` / `noise_spectrum()`)

Aus dem **Fluktuations-Dissipations-Theorem** (verallgemeinertes
Nyquist-Theorem, Twiss 1955) folgt das thermisch-akustische
Eigenrauschen der Kapsel ohne jeden Fit-Koeffizienten: Die
Kurzschluss-Rauschstromdichte am Membranzweig eines passiven Netzwerks
ist S_qq = 4·k_B·T·Re{1/Z_tot}, wobei Z_tot die Treibpunkt-Impedanz
über dem Membran-Serienzweig ist (Front-Ausgangswiderstand +
Membranimpedanz + Rück-Eingangsimpedanz). Dieses eine Ergebnis wichtet
**jeden** dissipativen Widerstand automatisch korrekt — Spaltfilm,
Bohrungen, Gewebe, Strahlung, inklusive der Stromaufteilung an allen
Shunt-Zweigen. Auf den freien Feld-Schalldruck zurückgerechnet
(S_p,eq = S_v,out/|H|², Θ und ω kürzen sich) ergibt sich die
äquivalente Eingangs-Druckrauschdichte, integriert und A-bewertet
(IEC 61672) der **Ersatzgeräuschpegel** in dB(A). Da Z_tot eine
Serien-Summe dreier Anteile ist, liefert das Modell zugleich die
**Pfad-Zerlegung** (Front / Membranfilm / Rückpfad, Anteil ∝ Re{Z_i}):
Bei der Niere dominiert der Rückpfad — genau die Widerstände, die den
Phasenschieber und damit die Richtcharakteristik bilden, sind die
Rauschquelle (Zielkonflikt Richtwirkung ↔ Rauschen).

Es ist das reine **Kapsel**rauschen (physikalische Untergrenze der
Geometrie); der FET/Verstärker, der das Datenblatt-Eigenrauschen realer
Mikrofone meist dominiert, ist bewusst nicht enthalten. Nur für die
ABCD-Modelle 1D/2D — der 3D-Feldlöser hat keinen konzentrierten
Membranzweig. Validiert (Gegenprobe 25): das Nyquist-Ergebnis trifft an
einem Mini-Netzwerk mit Shunt-Zweigen die Brute-Force-Superposition
über jeden Einzelwiderstand auf Maschinengenauigkeit; T → 2T ergibt
exakt +3 dB; die Pfad-Anteile summieren zu 1.

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
Wert), Niere **−6,4/−17,3/−26,2 dB @ 90/135/180° (1 kHz)** — praktisch
die publizierten U87-Werte —, glatter Präsenzpeak +3,3 dB @ 13,5 kHz,
Empfindlichkeit 21,0 mV/Pa. **Wichtig für Datenblatt-Vergleiche:** das
Modell rechnet die **nackte Kapsel**. Die K67 ist bewusst hell ausgelegt
(„Pre-Emphasis"), und die U87-Elektronik nimmt das über Gegenkopplung
wieder heraus („De-Emphasis"); dazu kommt der nicht modellierte Korb.
Ein Kapselmodell **muss** im Hochton also ÜBER der veröffentlichten
Gesamtkurve liegen — Übereinstimmung mit dem Datenblatt oberhalb
~10 kHz wäre ein Warnzeichen, kein Gütesiegel. Das tiefste
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

**BEM-Frontfaktor (Gegenprobe 26):** Im BEM-Modus kommt auch der
**Antrieb der Frontmembran** aus demselben Lösungsgang: das exakte
Flächenmittel des Drucks auf der realen **flachen Stirnfläche** ersetzt
die Kugelkalotten-Näherung der Morse-Reihe. Physikalischer Kern: an der
flachen Stirnfläche steht die Membran senkrecht zur einlaufenden Welle
— der Druckstau erreicht die Verdopplung (+6 dB, mit
Randbeugungs-Überschwingen bis ~+8 dB) schon bei ka ≈ 2…3, während die
bei der K67 um ±50° gekrümmte Kugelkalotte dort erst +3…4 dB liefert.
Diese systematische **3–4-dB-Unterschätzung des frontalen Druckstaus
war die künstliche Vertiefung der ~7-kHz-Senke** des Kalottenmodells.
Dass das exakte Physik und kein Fit ist, sichern vier Anker ab:
(1) auf einer **Kugelkontur** reproduziert das BEM-Frontmittel das
Kalottenmittel der Morse-Reihe (< 2·10⁻⁴); (2) am **oblaten Sphäroid**
(flacher Ersatzkörper) trifft es die *absolute* Flammer-Reihe
p(η) = 2i/(c(ξ₀²+1)) Σ (−i)ⁿ Sₙ(cos θ)Sₙ(η)/(Nₙ R³′ₙ), deren
Vorfaktor aus der ebenen-Wellen-Expansion folgt (Identität numerisch
auf Maschinengenauigkeit geprüft) — Abweichung < 10⁻⁴; (3) Grenzfall
ka → 0 ⇒ F → 1; (4) der Frontfaktor geht exakt multiplikativ in die
Übertragung ein (H ∝ F·(a + b·G), Netzwerk unberührt, Konsistenz auf
Maschinengenauigkeit). Ehrlicher Befund der Validierung: im BEM-Modus
hängt die Mittenband-Form nun sichtbar von der **Montagegeometrie**
ab — der starr montierte Körper bildet unter dem Kopf einen
ungedämpften **Ringspalt-Resonator** (Welligkeit ~4–6 kHz, in der
Realität durch die elastische Halterung bedämpft), der freie Kopf
(`bem_body_diameter=0`) liefert die glatte Referenz mit dem längeren
Randumweg der freien Scheibe. Beides ist die exakte Lösung seiner
Geometrie; geglättet wird nichts.

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

### Strahlungsimpedanz: exakter Kolben statt Asymptote (Gegenprobe 27)

Die Membranaußenseite koppelt über ihre **Strahlungsimpedanz** ans
Freifeld — im Ersatzschaltbild ein Serienglied zwischen dem (geblockten)
Beugungsdruck und der Membran. Gerechnet wird jetzt die geschlossene
Form des Kolbens in unendlicher Schallwand,

    Z_rad = ρ₀c/S · [R₁(2ka) + j·X₁(2ka)],
    R₁(x) = 1 − 2·J₁(x)/x,   X₁(x) = 2·H₁(x)/x

mit Bessel J₁ und Struve H₁ — die geschlossene Lösung des
Rayleigh-Integrals, ohne Fit. Vorher stand dort die
**Kleinargument-Asymptote** R ≈ (ka)²/2, X ≈ 8ka/(3π). Sie ist für
ka → 0 exakt, oberhalb ka ≈ 1 aber grob falsch: die echte Reaktanz X₁
hat ein **Maximum** bei 2ka ≈ 2 und fällt danach wie 4/(π·2ka) ab,
während die Asymptote linear weiterwächst. Die mitschwingende Luftmasse
war deshalb **konstant 25 kg/m⁴** statt zusammenzubrechen:

| f (26-mm-Membran) | 1 k | 4 k | 8 k | 12 k | 16 k |
|---|---|---|---|---|---|
| M Asymptote | 25,0 | 25,0 | 25,0 | 25,0 | 25,0 |
| M exakt | 24,6 | 19,6 | 8,9 | 2,0 | **0,8** |

Da die Membran selbst nur M_A = 20,9 kg/m⁴ hat, **verdoppelte** die
Asymptote die bewegte Masse über das ganze Band und drückte den Hochton
künstlich (bei 16 kHz um ~7 dB). Die Strahlungslast ist damit alles
andere als vernachlässigbar — |Z_rad| liegt in der Größenordnung von
|Z_mem| selbst. Die 1-kHz-Anker (Empfindlichkeit, Nierendämpfung, C₀)
bleiben unberührt; es ändert sich der Hochton, wo die Asymptote nie
gültig war. Verbleibende, bewusst dokumentierte Näherung: die
**unendliche Schallwand** — exakt lieferte das der BEM über die
Reziprozität von Streu- und Strahlungsproblem.

### 3D: Außenknoten der Doppelmembran-Bauform (Gegenprobe 27)

Der 3D-Löser trieb die Membranaußenseiten der K67-Bauform **direkt** aus
der Quelle: Strahlungsimpedanz und Gewebe fehlten ersatzlos —
`fabric_front_rayl`/`fabric_rear_rayl` blieben im 3D-Modus **exakt
wirkungslos**, ohne Hinweis (`single`/`dual` hatten ihren Frontknoten
bereits). Jetzt tragen zwei Sammelknoten vor den Membranaußenseiten
dieselbe Kette wie der 1D/2D-Pfad (vorn Z_rad + rayl_front/S_mem, hinten
rayl_rear/S_mem + Z_rad). Verankert (Gegenprobe 27): der Grenzfall
Z_außen → 0 reproduziert den Direktantrieb mit **erster Ordnung**
(zehnfach kleineres Z ⇒ zehnfach kleinerer Abstand — beweist Vorzeichen
und Struktur), Gewebe dämpft **monoton** (Passivität), die Dämpfung
stimmt mit der 1D/2D-Kette überein (−8,4 vs. −7,7 dB bei 10⁵ Rayl), und
die Reziprozität X_r = −B_f bleibt erhalten.

### Spaltmündung ohne Doppelzählung (Gegenprobe 28)

Eine Bohrung, die in den **engen Spalt** mündet, strahlt nicht in einen
Halbraum — es gibt dort kein halbkugeliges Nahfeld, die Strömung wird
sofort radial gequetscht. Diese laterale Ausbreitungsmasse steckt
bereits **vollständig** im Škvor-Term; nachgewiesen als Identität mit
der Baird/Zuckerwar-Spaltmasse:

    M_gap = ρ₀·B(q)/(n·π·h)  ≡  R_Škvor·ρ₀h²/(12μ)   (auf 10⁻¹⁵)

und die Frequenzkorrektur Φ(ω) realisiert sie auch wirklich: im Tiefton
ist Im(R·Φ)/ω = **(6/5)·M_gap** — der Faktor 6/5 ist der kinetische
Profilfaktor der Poiseuille-Verteilung, den eine reine Lumped-Masse gar
nicht kennt. Eine **zusätzliche** Freifeld-Flanschmasse 0,85·r auf der
Spaltseite wäre daher Doppelzählung. Das 2D-Feldmodell (Zell-Engstelle)
und der 3D-Feldlöser (Filmfeld) führten sie ohnehin nie; der **1D-Pfad**
tat es noch und ist jetzt auf dieselbe Konvention gebracht. Wirkung nur
im 1D-Pfad: bei ungestuften Bohrungen bis ~1 dB im Hochton, bei der
gestuften K67 ~0,1 dB; Voreinstellung (2D) und 3D bleiben unberührt.

### Mehrmoden-Membran (`membrane_modes`, Gegenprobe 28)

Das Lumped-Modell führt die Membran als EINEN Freiheitsgrad
(Grundmode). Oberhalb weniger kHz schwingt eine reale Membran aber
längst nicht mehr kolbenförmig, sondern bildet Knotenringe. Mit
`membrane_modes > 1` treten die höheren axialsymmetrischen
(0,m)-Bessel-Moden ψ_m = J₀(x_m·r/a) hinzu. Bei gleichförmiger
Drucklast folgt aus der Modalzerlegung, dass die Moden **parallel**
liegen:

    Y_ak = Σ_m 1/Z_m,   M_A,m = M_A,1·(x_m/x₁)²,   ω_m = ω₁·x_m/x₁

Die **Grundmode bleibt exakt die kalibrierte** (M_A_mem, C_A_eff) — alle
bestehenden Anker (f_res, Empfindlichkeit, Pull-in, Nierendämpfung) sind
unberührt, `membrane_modes = 1` (Voreinstellung) ist bit-für-bit der
bisherige Stand. Die drei ersten Moden machen die Membran akustisch um
den Faktor 0,789 leichter; im Hochton hebt das den Pegel:

| K67 (2D) | 7 kHz | 12 kHz | 16 kHz |
|---|---|---|---|
| `membrane_modes=1` | −3,6 | +4,1 | +1,3 |
| `membrane_modes=3` | −3,6 | +3,1 | +3,7 |
| `membrane_modes=5` | −3,9 | +2,8 | +4,2 |

**Wichtiges Negativergebnis:** Der ~7-kHz-Sattel ist **kein** Modeneffekt
— er bleibt unverändert. Das deckt sich mit dem 3D-Löser, der die
Membranen ohnehin als Felder ohne Modenabschneidung führt und bei
ausgerichteten Löchern sogar −7,7 dB zeigt. Ursache des Sattels bleibt
die interne Antiresonanz, die die reale K67 über die **Verdrehung der
Lochbilder** bedämpft (3D, 3°: −1,2 dB). Bewusste Näherungen: die
elektrostatische Feder-Erweichung wird nicht auf die höheren Moden
übertragen, und deren Filmdämpfung wird gleich der Grundmode gesetzt
(konservativ — real ist sie kleiner).

### Durchgehender Randspalt (`ring_vent_width`, Gegenprobe 29)

Bei vielen Messmikrofon-Kapseln (B&K-Bauart) ist der Luftspalt am
Plattenumfang **nicht dicht**: ein umlaufender Ringkanal verbindet ihn
mit der Rückkammer. Bis dahin war der Filmrand im Modell hermetisch
(Neumann, kein Fluss) — die Luft konnte den Spalt ausschließlich durch
Bohrungen verlassen. **Der Clearance-Ring kann das nicht ersetzen:** der
ist eine Nut *in* der Elektrodenfläche (Freistich bzw. Sack-Stub) und
öffnet nie einen Weg nach hinten.

Jetzt wird der Filmrand bei r = a_bp zum **Port**, und die Randströmung
läuft durch eine thermoviskose **Schlitzleitung** (LRF, Schlitz-Pendant
der Zwikker–Kosten-Rohrleitung, abgewickelte Breite b = 2π·a_bp) zum
selben rückwärtigen Port wie die Bohrungen. Eine Platte **ohne**
Bohrungen bekommt den analytisch herleitbaren Randwiderstand

    R_edge = 3μ/(2πh³)

aus der radialen Poiseuille-Strömung zum offenen Rand bei gleichförmigem
Kolbenantrieb (flächengemittelt; wie bei Škvor unabhängig vom
Plattenradius, aber **ohne** den 1/n-Faktor der Lochplatte — deshalb
deutlich größer: die Luft muss den ganzen Weg nach außen).

Verankert (Gegenprobe 29), alles fit-frei: die Schlitzleitung trifft im
Tiefton exakt R = 12μL/(b·w³), die kurze Leitung exakt die Masse
(6/5)·ρ₀L/(b·w) — derselbe kinetische Profilfaktor 6/5 wie oben — und
die isotherme Nachgiebigkeit V/P_atm, bei det T = 1 auf 10⁻¹⁶; ein
dichter Rand **entkoppelt** Membran und Rückport mindestens wie w³
(sehr schmale Spalte sogar exponentiell, weil die Leitung dann ins
Wellenleiter-Regime wechselt); die Wirkung ist monoton und sättigt,
sobald nicht mehr der Kanal, sondern der Film selbst begrenzt
(10 → 200 µm: +15,8 dB); das 2D-Zweitor bleibt mit Randknoten reziprok,
auch mit Bohrungen **und** Randspalt gleichzeitig; 1D und 2D liegen im
Tiefton 1,9 dB auseinander.

**Auch im 3D-Löser** hängt derselbe Ringkanal über den Randflächen-
Leitwert an der äußersten Filmzellreihe. Verankert am **Kolben-
Grenzfall**: nur wenn die Membran sich *nicht* verformen kann,
beschreiben 1D/2D (Grundmode φ erzwungen) und 3D (freies Membranfeld)
dasselbe Problem — mit steifer Membran im quasistatischen Tiefton fallen
beide auf **0,4 dB** zusammen (seit Gegenprobe 50 mit exaktem
Membranrand), und das Ergebnis ist von der azimutalen
Auflösung unabhängig (< 10⁻⁶ zwischen Np = 96 und 192).

**Dokumentierter Modellunterschied, kein Fehler:** Bei *weicher* Membran
liegt der 3D-Wert systematisch höher — bei der Referenzkapsel bis 8 dB.
Grund ist die Einmoden-Grenze des homogenisierten Modells: Der
Randwiderstand `R_edge` ist groß und der Strömungsweg lang, und eine
*freie* Membran umgeht ihn, indem sie bevorzugt außen arbeitet, wo der
Weg kurz ist. Das 2D-Modell zwingt sie dagegen in die Kolbenform. Der
Effekt verschwindet sauber mit steigender Membransteifigkeit
(f_res 2,1 → 8 → 20 → 50 kHz ergibt 8,0 → 0,4 → −0,2 → −0,3 dB) — es ist
dieselbe Effektklasse wie beim K67-Sattel. Für randbelüftete Bauformen
mit weicher Membran ist der **3D-Modus daher der belastbarere**.
Gegenprobe 48 hat daraus das allgemeine Kriterium f_hom gemacht (der
Randspalt zählt dort als Senke am Plattenrand).

**Nebenbefund (nicht Teil dieses Features):** Der Kettenpfad führt den
Škvor-Widerstand zweimal — einmal in `_membrane_impedance`, einmal im
Backplate-Zweitor. Bei gelochten Platten macht das wenige dB (der
3D-Wert liegt zwischen den beiden Varianten), bei randbelüfteten ~4 dB.
Das ist eine bestehende Modellkonvention, die eigens zu klären wäre.

**Gatter (statt stiller Zahlen):** nur `single`/`dual` (bei der
K67-Bauform versiegeln Spacer und Klemmringe den Rand); nicht mit
Spacer/Rückplatte (K103) kombinierbar, die sich denselben Rand teilen;
im 1D-Pfad nur *ohne* Bohrungen (Bohrungen **und** Randspalt brauchen
die Stromaufteilung eines Feldmodells).

### Durchfluss-Zellfunktion: Sackgassen zählen nicht (Gegenprobe 30)

Die azimutale Zuströmung im Spaltfilm hat **zwei verschiedene Ziele**,
und sie brauchen verschiedene Zellgrößen:

* **Aufnahme** (Verdrängungsströmung): jede Bohrung ist eine Senke — die
  Luft läuft zur nächstgelegenen, gleich welcher Art.
* **Durchfluss zur Rückseite**: nur **Durchgangslöcher** zählen;
  Blindlöcher sind Sackgassen, die Luft aufnehmen, aber keinen Weg nach
  hinten bieten.

Bis dahin nutzte auch der Durchfluss die Zellfunktion *aller* Bohrungen.
Der Zugang zur Rückseite war dadurch zu leicht und die interne Laufzeit
des Nieren-Phasenschiebers zu kurz — bei der Debenham-Platte 12 statt
58 Senken, also ein rund fünffach längerer Weg als modelliert. Die
Škvor-**Dämpfung** (`R_A_gap`) bleibt unverändert über alle Senken
gebildet; dort ist jede Bohrung ein gültiges Ziel.

**Wirkung — die K67-Niere sitzt jetzt richtig:**

| K67 @ 1 kHz | Minimum | 90° | 135° | 180° | Empfindlichkeit |
|---|---|---|---|---|---|
| publiziert (U87) | 180° | −6 | −17 | −26 | ~20 mV/Pa |
| vorher | 164° | −6,4 | −17,3 | −26,2 | 21,0 |
| **jetzt** | **180°** | −5,9 | −15,1 | **−26,6** | **19,8** |

Das Pattern-Minimum liegt jetzt bei 180°, wie bei der realen K67, und
Rückwärtsdämpfung wie Empfindlichkeit treffen die publizierten Werte
besser. Der 135°-Wert wird dabei um ~2 dB ungenauer — das ist der
ehrliche Preis. Auch gegenüber dem **3D-Feldlöser**, der die diskreten
Löcher auflöst und deshalb Referenz ist, rückt das 2D-Modell näher:
RMS-Abweichung des Richtdiagramms 0,83 → 0,59 dB (Debenham) und
1,13 → 0,99 dB (Nieren-Single).

**Drei Gegenproben mussten angepasst werden** (17a, 17c, 24) — alle drei
kodierten abgelesene Werte des alten Simulationszustands, keine
Messreferenzen. Gegenprobe 17 verlangte ausdrücklich ein Laufzeit-
Verhältnis knapp *unter* 1 mit Minimum *vor* 180°; das war eine
Selbstbestätigung der Simulation und widersprach der realen Kapsel. Die
**Deutung** des Verhältnisses (< 1 → Minimum wandert vor 180°; > 1 →
gepinnt, aber flacher) ist unverändert gültig — die K67 wechselt nur die
Kategorie.

### Filmdämpfung genau einmal (Gegenprobe 31)

Der Škvor-Widerstand stand **zweimal** in derselben Kette: in
`_membrane_impedance` *und* im Backplate/Spalt-Zweitor, das in Serie
folgt. Physikalisch ist es ein Weg — die Piston-Bewegung drückt die
Spaltluft lateral zu den Senken —, also einmal zu zählen. Der
Strukturbeweis ist die Eingangsimpedanz des Zweitors bei
kurzgeschlossenem Port und widerstandsarmen Bohrungen: Z_in = R_A_gap
(Verhältnis 1,0004). Die Membranimpedanz trägt jetzt nur noch die
Materialdämpfung der Folie.

**Warum das lange unentdeckt blieb:** Der Fehler ist an
Gradientenbauformen nicht messbar. Deren Ausgangsgröße hängt an einer
rückwärtigen Auslöschung und reagiert auf jede Phasenänderung
überempfindlich — dort schien die Korrektur mehrfach zu *scheitern*. Nur
der **Druckempfänger** misst die Dämpfung unverfälscht. Empfindlichkeit
bei 4 kHz gegen den 3D-Feldlöser (der die Löcher diskret auflöst), bei
konstanter Lochfläche:

| n_th | 12 | 24 | 48 | 96 | 192 |
|---|---|---|---|---|---|
| einmal gezählt | −5,5 | −1,3 | **+0,4** | **+0,4** | +2,5 |
| doppelt (vorher) | −10,8 | −6,5 | −4,5 | −3,7 | −0,5 |

Im Gültigkeitsbereich der Homogenisierung (48–96 Bohrungen) trifft das
2D-Modell den Feldlöser jetzt auf **0,4 dB**; vorher lag es 4 dB daneben.
Bei sehr spärlichem Raster (12 Bohrungen) bleibt eine Abweichung — dort
ist die axialsymmetrische Homogenisierung am Ende und der 3D-Löser
nötig. Das ist als Grenze dokumentiert, nicht wegkalibriert.

Die K67 bleibt dabei auf ihren publizierten Werten: Minimum bei 180°,
−6,0/−15,5/−27,9 dB bei 90/135/180°, 20,1 mV/Pa (publiziert ~20).

*Nachtrag (Gegenprobe 48):* die Grenze „48–96 Bohrungen" war zu grob —
sie hängt nicht an der Lochzahl allein, sondern an der Frequenz f_hom
(s. u.). Mit dem korrigierten 3D-Löser und dem exakten Arbeitspunkt
(Gegenprobe 49; der Prüfling läuft jetzt mit 45 V, denn 50 V liegen
genau auf seinem Pull-in) lautet die Zeile „einmal gezählt" −5,8 /
−1,2 / +0,2 / +0,3 / −0,5 dB; die Entscheidung bleibt dieselbe, und
die frühere Unstimmigkeit bei 192 Bohrungen (+2,5 dB) ist verschwunden.

### Externe Referenzen: FEM und Messung (Gegenproben 32, 38)

Zwei fremde, in sich vollständige Quellen verankern das Modell von
außen:

* **FEM (Gegenprobe 32):** Šimonová/Honzík, JASA 159, 4512 (2026),
  COMSOL 3D thermoviskos, ~6·10⁶ Freiheitsgrade, alle Parameter in
  derselben Arbeit. Der Prüfling ist bewusst extrem (R = 18 mm,
  230-µm-Spalt, nur **vier** Bohrungen). Getroffen: der Tiefton
  (< 1 dB) und — die eigentliche Dämpfungsprobe — die
  Resonanzüberhöhung (+6,4 gegen +6,7 dB). Das **Dublett** der FEM im
  Kerbenband (3500/4200 Hz) kann der homogenisierende 2D-Pfad
  prinzipiell nicht haben; der 3D-Löser zeigt es (3378/4127 Hz).
  **Offen** bleibt die Resonanzlage: 2D und 3D liegen beide 13 % unter
  der FEM (477/482 gegen 550 Hz). Früher stand hier, das sei die
  Homogenisierungsgrenze — ein Fehler, den das diskret rechnende Modell
  genauso macht, kann das nicht sein (korrigiert in Gegenprobe 48).
* **Messung (Gegenprobe 38):** Zuckerwar, JASA 64, 1278 (1978), B&K
  4134 und 4146 — Tabelle I vollständig, Tabelle II die Ersatzelemente,
  Fig. 6/7 Amplitude **und** Phase gegen Messwerte. M und C_M treffen
  analytisch (< 0,2 %), der Frequenzgang des 4134 liegt 0,3 dB RMS neben
  der Messung. Beim 4146 bleibt die Streuung der Škvor-Zellregel
  q = n·r²/a_bp² als dokumentierter Rest (Schranke 1,6 dB RMS).

### Modenweise Anregung (Gegenproben 33, 34, 42, 43)

* **Modengewicht der Frontmittelung (33):** eine Membranmode wird von der
  Galerkin-Projektion ⟨p·ψ⟩/⟨ψ⟩ getrieben, nicht vom Flächenmittel. Das
  alte Flächenmittel erzeugte bei k·a·sin θ = 3,83 eine Auslöschung, die
  die Grundmode gar nicht hat. Geprüft gegen die geschlossene
  Freifeldform D(u) = z₀₁²·J₀(u)/(z₀₁² − u²).
* **Modenabhängiger Quelldruck (`modal_source`, 34):** jede Mode bekommt
  ihre eigene Projektion; weil die Moden in der Kette parallel am selben
  Spaltknoten liegen, ist die Zusammenfassung zu einer Ersatzquelle
  exakt. Bei streifendem Einfall fehlten der uniformen Anregung gegen die
  FEM oberhalb 5 kHz 14 dB.
* **Flächenmittel der Modenreihe (42):** die zwei Rayleigh-Summen
  Σ1/z_m² = 1/4 (Hochton: freier Kolben) und Σ1/z_m⁴ = 1/32 (exakte
  Statik) verankern die parallelen Zweige; die Grundmode allein trägt
  95,7 % der statischen Nachgiebigkeit, deshalb die Normierung der
  höheren Zweige.
* **Modenfaktoren aus demselben Körper (43):** mit BEM kommen alle
  Modenfaktoren aus EINEM Lösungsgang (vorher Grundmode aus dem BEM,
  höhere aus der Kugelkalotte), und Gewichtung und Kette benutzen
  dieselben gedämpften Zweige. Gegen Grinnips Messung: 90° von 3,7 auf
  2,4 dB RMS, 180° von 3,0 auf 1,7 dB.

### Spaltfilm gegen die Literatur (Gegenproben 35, 36, 37)

* **Filmträgheit Φ(ω) (35):** unsere selbst hergeleitete Frequenz-
  korrektur trifft die Reihe von Homentcovschi & Miles (JASA 124, 175,
  2008) im ersten **und** zweiten Glied (1 + jK²/10 + K⁴/8400). Deren
  Aussage „M = 1 genügt unter 100 kHz" gilt für MEMS-Spalte; bei
  Kapselspalten ist K = O(1…10) und die Korrektur wesentlich.
* **Reaktivanteil der Zell-Engstelle (36):** der Imaginärteil des
  Zellterms ist exakt die kinetische Energie der Schmierfilmströmung —
  zwei unabhängige Wege, ein Ergebnis. Die Stokes-Zelle von
  Homentcovschi/Murray/Miles (2010) wurde nachgebaut (ihre Tabelle 3 auf
  0,0 %) und liefert MEHR, nicht weniger — der Zellterm ist kein
  Massenüberschuss.
* **Randumgehung (37):** der Film liegt nur unter der Backplate, die
  Membran erzeugt Fluss aber über ihre ganze Fläche. Für die lochfreie
  Platte ist die Filmimpedanz geschlossen integrierbar,
  Z = (12μ/πh³)·(u²/2 − u³/3 + u⁴/16) mit u = (a_bp/a_mem)²; ohne die
  Umgehung wäre Z um 1/[u(2−u)]² zu groß (B&K: +28/+56 %).

### Pull-in der Gegentakt-Bauform (Gegenprobe 39)

„Zwei Backplates → doppelte Pull-in-Spannung" stimmt **nicht**. Bei
`dual` heben sich die statischen Kräfte auf (w₀ = 0), der Wandler-
koeffizient verdoppelt sich — aber auch die Feder-Erweichung addiert
sich. Pull-in ist dort das Eigenwertkriterium am Ruhespalt, für die
volle, lochfreie Elektrode geschlossen **Ā = j₀₁²/4 = 1,4458**; die
Einzel-Backplate kollabiert erst, nachdem die Membran ein gutes Stück
gekrochen ist (Warren: Ā = 0,789). Das Verhältnis ist deshalb
U_PI(dual)/U_PI(single) = √(1,4458/0,789) = **1,3537** (Ein-Moden-Bild
√(1,5·A₃(x*)) = 1,3464, starrer Kolben √(27/16) = 1,299).

### BEM: flache Stirnfläche und Rückeinlass (Gegenproben 41, 44)

* **Frontfaktor der flachen Stirnfläche (41):** eine Ein-Membran-Kapsel
  ist kein Ball. Die reale flache Stirnfläche staut stärker als die
  Kugelkalotte (die bei ~+5 dB sättigt); gegen Grinnip (JAES 54, 157,
  2006, Fig. 5) 1,0 dB RMS auf Achse statt 3,1 dB mit der Kugel. Die
  mehrdeutige Tabellenangabe „h_c = 5,955e-7/b²" entscheidet die Physik
  eindeutig (als Volumen gelesen 1,6 dB RMS, als Höhe 10,4 dB). Offen:
  ein Rest außerhalb der Achse (90°: 2,5 dB RMS mit modenweiser
  Anregung).
* **Rückpatch des Gradientenempfängers (44):** der rückwärtige Einlass
  sitzt als Ring auf der realen Kontur — im Mantel bei seiner
  Einbautiefe oder (`cavity_hole_position='end'`) in der hinteren
  Stirnfläche. Absolut geprüft gegen die exakte Morse-Reihe auf der
  Kugelkontur (2·10⁻⁴), unabhängig von der Elementteilung. Bei zwei
  symmetrischen Backplates ist der vordere Einlass die Außenseite der
  vorderen Platte; der Abstand zum Rückeinlass ist d_ext. Das frühere
  Gatter ist eine **Warnung** (glatter Zylinder, kein Korb/Gitter).

### Mittenterminierung / Ringmembran (Gegenproben 45, 47)

Eine in der Mitte festgelegte Membran (Kontaktstift, Mittenbolzen,
`center_post_diameter`) ist eine **Ringmembran**. Die statische Form
enthält einen Logarithmus — schon r_i/a = 1 % nimmt 22 % der
Nachgiebigkeit weg. Verankert in 1D/2D (45): r_i = 0 exakt der Bestand;
die statische Form gegen eine unabhängige Finite-Volumen-Lösung; die
Modenintegrale gegen Quadratur; die Ringvariante der Rayleigh-Summe
(ΣC_m = Ring-Nachgiebigkeit) und die Massensummenregel; Pull-in gegen
Warren (JASA 58, 733, 1975): kritisches Ā = 0,789 (Kreis) bzw. 1,548
(Ring, ρ = 0,1) — seit Gegenprobe 49 exakt getroffen (vorher lag der
Ein-Moden-Galerkin +5,0 % bzw. +2,6 % darüber). Im 3D-Löser
(47) beginnt das Gitter am Pfostenrand; derselbe Flächenleitwert-Term
ist bei r₀ = 0 die Achsenbedingung und bei r₀ > 0 die eingespannte Wand.
Das reine Membranfeld trifft die geschlossene Ring-Nachgiebigkeit
gitterkonvergent, 3D und 2D passen mit Pfosten so gut zusammen wie
ohne.

### Laufzeit und Rauschintegral (Gegenprobe 46)

Die K67 mit BEM-Kopf und -Körper lief 24 Minuten, davon 19 still bei
100 %. Ursachen: jeder BEM-Lösungsgang wurde für das Eigenrauschen ein
zweites Mal gerechnet, und der frequenzunabhängige statische Kern wurde
je Frequenz neu gebaut — jetzt ein Ergebnisspeicher je (ω, θ) und ein
Kern je Geometrie (bitgleiches Ergebnis, ~5 statt 24 min). Die
„divide by zero"-Warnungen waren stehengebliebene LAPACK-Flags; der
Lösungsgang prüft jetzt sein Ergebnis. Unabhängig davon war das
Rauschintegral zu grob: S_p = S_v/|H|² hat Spitzen, wo die Kapsel taub
ist; eine lokale Nachverfeinerung bringt den Fehler von 0,53 auf
0,0002 dB.

### Exakter statischer Arbeitspunkt (Gegenprobe 49)

Die Membran unter Polarisationsspannung ist eine nichtlineare
Randwertaufgabe, kein Ein-Freiheitsgrad-Problem:

    T·∇²w = −(ε₀U²/2)·[c_s/(h − w)² + c_b/(h + d − w)²]

(c_s, c_b: Anteile solider Elektrode bzw. über Sacklöchern, radial wie
im Feldmodell; w = 0 am Rand, bei der Ringmembran auch am Pfosten). In
der Koordinate u = r²/a² ist der Operator an der Achse regulär und das
Finite-Volumen-System tridiagonal. **Pull-in ist der Faltpunkt** des
Lösungsasts — parametrisiert über das verdrängte Volumen, das auch über
die Falte hinweg monoton ist. Die **Feder-Erweichung** kommt aus dem
linearisierten Operator am Arbeitspunkt und divergiert genau dort;
Wandlerkoeffizient, Ruhekapazität C₀ und das Spaltprofil der Feldmodelle
(2D und 3D) sehen das exakte Profil h − w(r); der 3D-Löser erhält die
Erweichung **örtlich** statt als gleichförmige, an der Grundmode
kalibrierte Konstante. Bei `dual` (w₀ = 0) ist Pull-in das
Eigenwertkriterium am Ruhespalt.

Bis dahin stand ein Ein-Moden-Galerkin mit der statischen Form φ da —
für kleine Lasten exakt, zum Pull-in hin aber zu steif. Geprüft mit
unabhängigen Methoden:

| Prüfung | Ergebnis |
|---|---|
| Form gegen Schießverfahren (solve_ivp in r) | 3·10⁻⁶ |
| Kleinsignal-Nachgiebigkeit gegen ∂V/∂p des nichtlinearen Asts | 5·10⁻⁹ |
| Warren, Kreis | Ā = 0,7892 gegen 0,789 (Ein-Moden: 0,8274, +4,8 %) |
| Warren, Ring ρ = 0,1 | 1,549 gegen 1,548 (Ein-Moden: +2,6 %) |
| Gegentakt, volle Elektrode | Ā = j₀₁²/4 = 1,4458 geschlossen |
| Nachgiebigkeit 0,5 → 0,999·U_PI | 1,08 → 10,9 × (divergiert) |

Die kleinen Restabstände zu Warren sind die Biegesteife der Folie
(0,1 % der Spannung), die Warrens reine Membran nicht hat.

**Wirkung:** der Pull-in sinkt um 1,3…2,5 % — bei der K67 von 74,4 auf
**72,6 V** (Arbeitspunkt bei 60 V: w₀ = 11,7 µm, C₀ = 50,3 pF,
Erweichung 28 %). Vier Prüflinge des Selbsttests saßen genau auf oder
knapp über ihrem exakten Pull-in — die weiche 1"-Kapsel der
Gegenproben 29/31 und die Debenham-Variante ohne Sacklöcher in 22
(48,9…50,0 V gegen 50 V Bias) sowie die fast volle K67-Elektrode in 19
(59,7 V gegen 60 V) — und laufen jetzt mit 45 bzw. 50 V; sie prüfen
Filmphysik bzw. Struktur, nicht die Nähe zum Kollaps. Die Beispiel-
projekte bleiben stabil (Debenham 52,0 V bei 50 V Betrieb, K67 72,6 V
bei 60 V).

### Homogenisierungsgrenze der 1D/2D-Modelle (Gegenprobe 48)

1D und 2D verschmieren die Bohrungen zu einer Senkendichte und zwingen
der Membran über jeder Lochzelle die **globale Modenform** auf. Über
einem großen lochfreien Bereich staut sich aber der Film, und die
gespannte Membran weicht ihm örtlich aus — sie beult sich zwischen den
Löchern. Das kann nur der 3D-Löser. Maßgeblich ist das Verhältnis der
viskosen Filmkraft zur Spannungssteifigkeit der Membran über dem
größten lochfreien Bereich (Überdeckungsradius ρ = größter Abstand
eines Elektrodenpunkts zur nächsten Durchgangsbohrung):

    Π(ω) = ω · 12μ·ρ⁴ / (h³ · T · j₀₁²)

Der Atmosphärendruck kürzt sich heraus — es zählt die Viskosität,
nicht die Kompressibilität. Gegen den (korrigierten, konvergierten)
3D-Löser setzt die 1-dB-Abweichung bei Π = 10…42 ein (zwei Kapseln,
T = 40 und 109 N/m, Spalte 20/38/65 µm). Mit dem vorsichtigen Rand
Π = 10 folgt die Grenzfrequenz

    f_hom = 10 / (2π · 12μ·ρ⁴ / (h³·T·j₀₁²)).

Liegt sie im Hörband (< 20 kHz), rechnet das Modell trotzdem, **warnt**
aber (`UserWarning`, in der GUI als Hinweis, in `summary()` als Zeile
„Loch-Homogenisierung bis"). `homogenization_limit()` liefert ρ, T und
f_hom. Beispiele:

| Kapsel | ρ | T | f_hom |
|---|---|---|---|
| K67 (60 Durchgangs-Senkungen) | 3,2 mm | 13 N/m | 1,1 kHz |
| Debenham (12 Bohrungen) | 6,0 mm | 41 N/m | 43 Hz |
| B&K 4134 (6 Bohrungen + Randspalt) | 2,0 mm | 3300 N/m | 73 kHz |
| Standardkapsel (60 Durchgangs- + 30 Sacklöcher, 8 kHz) | 2,1 mm | 440 N/m | 58 kHz |

Großmembran-Kapseln mit weicher Folie liegen damit fast immer im
Warnbereich. Π = 10 ist bewusst der vorsichtige Rand; bei der K67
liegen 2D und 3D auf Achse bis zu 4 dB auseinander (−4,0 dB bei
8 kHz), tragen aber schon im Tiefton einen Versatz von ~2 dB, den
f_hom nicht erklärt (Zwischenspalt-Geometrie, s. u.).

**Geprüft und verworfen** (mit Beleg in der Gegenprobe): die
Kompressibilität *in* der Škvor-Zelle (exakte Lösung mit modifizierten
Besselfunktionen, gegen eine FD-Zelle auf 10⁻⁹ — ändert B bis zur
Zell-Squeeze-Zahl 1 um < 1 %), der Modenabbruch der Membran
(`membrane_modes = 3` ändert die 2D-Rechnung um < 0,01 dB) und
Sacklöcher als Entlastung (36 tiefe Sacklöcher zwischen 12
Durchgangslöchern senken die Abweichung nur von 10 auf 7,6 dB — es
zählen die Durchgangslöcher).

**Der 3D-Löser musste dafür erst selbst belastbar werden.** Vier Fehler
fielen auf, jeder physikalisch begründet behoben:

1. **Fußabdruck:** das Suchfenster der Mündungszellen war fest ±4
   Zellen. Größere Mündungen wurden abgeschnitten (12 × ⌀1,4 mm auf
   3 mm Radius wirkten wie ⌀0,8 mm), das Ergebnis **wanderte** mit der
   Gitterfeinheit statt zu konvergieren. Jetzt exakter Abstand, Fenster
   nach Lochgröße: konvergent (0,97 → 0,34 dB je Halbierung).
2. **Äquipotentiale Mündung:** über dem Lochquerschnitt gibt es keinen
   Film. Die Mündungszellen waren gewöhnliche Filmzellen mit
   gleichverteiltem Zufluss — ein Aufschlag von +1/8 auf Škvors B
   (+28 % Zellwiderstand bei q = 0,04). Jetzt kurzgeschlossen
   (G_s·(I − 11ᵀ/k); das Ergebnis hängt vom numerischen Leitwert nicht
   ab, 10⁻⁴ dB).
3. **Lochlage:** gleichverteilte Löcher lagen auf flächengleichen
   Hilfskreisen mit gleicher Lochzahl (außen Zellen von 0,9 × 10 mm) —
   jetzt ein isotropes Raster. Gleichverteilte Durchgangs- und
   Sacklöcher waren zwei getrennte Raster, 15° versetzt; an der K67
   überlappten die Senkungen. Jetzt ein gemeinsames Raster mit
   abwechselnder Belegung (wie die reale K67: 120 Senkungen, jede
   zweite durchgebohrt), und die Gegenelektrode ist kreisweise
   verdreht. Die frühere Voreinstellung 180°/n_th ist nur für EINEN
   Lochkreis eine halbe Teilung; auf dem Mehrkreis-Raster legte sie die
   Kerne beider Hälften im Zwischenspalt übereinander.
4. **Spaltprofil:** der 3D-Film sieht jetzt wie das 2D-Feld das örtliche
   h(r) = h − w₀·φ(r) statt des Flächenmittels.

**Offene Punkte, die dabei sichtbar wurden:**

* **K67-Rückdämpfung im 3D hängt an der Kernlage.** Wie die Kerne beider
  Hälften im 50-µm-Zwischenspalt zueinander liegen, ist nirgends
  dokumentiert — und genau das bestimmt die Tiefe der Auslöschung:
  vollständig versetzt (automatisch, jeder Kern über einer Sacksenkung
  der Gegenseite, ~2 mm Querweg) −11 dB bei 180°/1 kHz; teilweise
  fluchtend (global 9°) −29 dB wie im 2D-Modell (−28 dB), dessen
  Škvor-Zelle einen Querweg von etwa einem Zellradius annimmt; fluchtend
  (0°) wandert das Minimum auf 106°. Das ist eine Geometriefrage an der
  realen Kapsel, kein Modellfehler — `half_rotation_deg` stellt sie ein.
  (Werte auf dem groben Gitter; fein: versetzt −14,6 dB, 9° −28,5 dB —
  der Befund bleibt, der Abstand schrumpft von 18 auf 14 dB.)
* **Standardgitter des 3D-Lösers:** das grobe Gitter löst kleine
  Mündungen nur mit rund einer Zelle je Radius auf (48 × ⌀0,7 mm auf
  1": rund 1 dB neben feineren Gittern). Seit Gegenprobe 50 gibt es
  dafür das feine Gitter (Schalter, s. u.); Gegenprobe 48 rechnet ihre
  Trennprobe damit.
* **B&K 4134 im 3D:** bei 13…20 kHz liegt der 3D-Löser 1,9…3,1 dB über
  der Messung, das 2D-Modell höchstens 0,6 dB. Die im 3D fehlende
  Randumgehung ist es nicht (mit a_bp = a_mem wird die Differenz eher
  größer — nach der Randkorrektur aus Gegenprobe 50 nachgerechnet),
  und das Gitter auch nicht (fein weicht von grob um höchstens 0,2 dB
  ab).

### 3D-Gitter grob/fein (Gegenprobe 50)

Der 3D-Löser rechnet standardmäßig auf dem **groben** Gitter (60
Radialzellen, 96…320 azimutal). Der GUI-Schalter **„Feines 3D-Gitter"**
im Abschnitt Spaltfilm-Modell (in der Klasse `grid_3d="fine"`) schaltet
auf das **feine** Gitter um; er ist nur mit dem 3D-Modell aktiv, die
Wahl wandert in die Projektdatei, und `summary()` nennt das Gitter.

**Regel.** Der Gitterfehler kommt von der Treppenkontur der Mündungen:
eine Zelle gehört zur Mündung, wenn ihre Mitte darin liegt. Wie stark
das wirkt, hängt am Verhältnis Zellweite zu Mündungsradius — und
welche Richtung es bestimmt, an der Bauform (K67: radial, das 96er-
Umfangsraster einteiliger Elektroden: azimutal). Das feine Gitter
richtet sich deshalb nach der kleinsten Mündung, die ein Film sieht
(membranseitig Loch bzw. weite Senkung, Sacklöcher, im Zwischenspalt
der K67 der enge Kern): mindestens **2 Zellen je Mündungsradius**,
radial und azimutal (dort am äußersten Lochmittenkreis), und in beiden
Richtungen mindestens 1,5-mal feiner als grob. Obergrenze 50 000
Zellen je Feld; greift sie (winzige Löcher), warnen Modell und GUI.

**Wirkung** (1 kHz, Abweichung gegen das feinste gerechnete Gitter;
Zeit je Frequenzpunkt auf einem Kern, allein gemessen):

| Kapsel | grob | fein | Referenz | Abw. grob | Abw. fein | Zeit grob → fein |
|---|---|---|---|---|---|---|
| 12 × ⌀1,4 mm auf 1" | 60 × 96 | 90 × 162 | 180 × 324 | 1,2 dB | 0,25 dB | 0,5 → 3 s |
| 24 × ⌀0,46 mm auf ½" (1/4/12 kHz) | 60 × 96 | 90 × 254 | 135 × 380 | 0,9…1,7 dB | 0,1…0,2 dB | 0,4 → 5 s |
| 48 × ⌀0,7 mm auf 1" | 60 × 192 | 90 × 376 | 135 × 564 | 1,4 dB | 0,4 dB | 2 → 11 s |
| Debenham (einteilig), Pegel / 180° | 60 × 96 | 90 × 388 | 120 × 512 | 0,5 / 1,2 dB | < 0,06 dB | 0,8 → 20 s |
| K67, Pegel / 180° | 60 × 240 | 90 × 480 | 135 × 480, 90 × 720 | 0,6 / 3,3…3,8 dB | 0,05 / 0,2…0,3 dB | 22 → 34 s |

Speicher: fein bis etwa 3 GB (K67-Typ), grob etwa 1 GB. Für die K67
heißt das: die 180°-Dämpfung des groben Gitters (−11 dB bei 1 kHz mit
automatischer Verdrehung) ist nicht konvergiert — fein liegt sie bei
−14,6 dB.

Das feine Gitter drückt den Gitterfehler etwa auf ein Fünftel,
beseitigt ihn aber nicht: die Treppenkontur konvergiert langsam und
unregelmäßig (bei 12 × ⌀1,4 mm liegt 180 × 324 noch 0,15 dB neben
135 × 244). Eine konturtreue Mündung (Flächenleitwerte am Rand mit dem
wahren Abstand zur Kreiskontur, Shortley–Weller) würde die Konvergenz
auf zweite Ordnung heben — noch nicht umgesetzt.

**Dabei behobene Gitterfehler:**

* **Membranrand.** Die Einspannung rastete auf das nächste Vielfache der
  Zellweite ein — mit a_bp = a_mem sogar eine ganze Zelle zu weit. Die
  Membran war je nach Gitter bis 0,6 % zu groß oder zu klein, ihre
  Nachgiebigkeit (∝ a⁴) sprang mit der Auflösung; bei der K67 zwischen
  60 und 90 Radialzellen um 4 %. Der lochfreie Kolben-Grenzfall streute
  zwischen den Gittern um bis zu 0,3 dB. Jetzt hat der Membranring außerhalb
  der Elektrode eine eigene Zellweite, die Einspannung liegt auf jedem
  Gitter exakt bei a_mem, und der Grenzfall konvergiert monoton (über
  30…135 Radialzellen innerhalb 0,015 dB).
* **Clearance-Ring:** liegt jetzt auf dem 3D-Gitter selbst (fein wird
  ein schmaler Ring als Relief aufgelöst, wo das 2D-Feld einen Stub
  braucht); die Stub-Zelle wird ab dem Pfostenrand gezählt (vorher ab
  der Achse — mit Mittenterminierung saß der Stub um den Pfostenradius
  zu weit außen).
* **Randspalt mit Mittenpfosten:** der Randleitwert im 3D nimmt den
  Randradius (q0 + Nr)·dr wie das 2D-Feld (Wirkung < 0,01 dB).

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
