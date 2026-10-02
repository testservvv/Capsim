# Capsim — Simulation einer Kondensatormikrofonkapsel

Lumped-Element-Simulation (elektroakustisches Ersatzschaltbild) einer
Kondensatormikrofonkapsel mit Streamlit-Oberfläche.

## Komponenten

| Datei | Inhalt |
|---|---|
| `microphone_capsule.py` | Physik-Klasse `MicrophoneCapsule` (ABCD-Kettenmatrizen, Zwikker–Kosten-Lochimpedanzen, Škvor-Squeeze-Film **oder** 2D-Reynolds-Feldmodell **oder** 3D-(r,φ)-Feldlöser mit diskreten Löchern, elektrostatische Wandlung mit Pull-in, Gehäusebeugung); `python microphone_capsule.py` startet den Selbsttest |
| `app.py` | Streamlit-GUI: Parameter-Seitenleiste, Bode-Plot, Polardiagramm, Projekt speichern/laden (JSON), CSV-Export |
| `translations.py` | Übersetzungstabelle der GUI (Englisch/Deutsch) |
| `tests/` | Die Gegenproben (pytest), thematisch gruppiert; `basis.py` hält die gemeinsamen Referenzkapseln und Messdaten, `conftest.py` die Fixtures, `stand.py` und `basis_werte.json` die Stand-Werte und Sperrklinken, `stokes_zelle.py` und `stokes_eben.py` die Referenzlöser der Spalt-Mündung (Gegenprobe 67) |

## Sprache / Language

Die Oberfläche ist zweisprachig (**Englisch** als Standard, Deutsch
umschaltbar) — die Sprachwahl steht oben in der Seitenleiste. Sie ist eine
reine Anzeige-Einstellung und wandert **nicht** in die Projektdateien: die
kanonischen Auswahl-Werte (Architektur, Material, axialer Körper …) bleiben
sprachunabhängig gespeichert, sodass Projekte zwischen beiden Sprachen
voll austauschbar sind. Auch der Diagnose-Summary (`MicrophoneCapsule.
summary(lang=…)`) und die Meldungen des Modells bei ungültigen Parametern
(`ParameterFehler.text(…)`) folgen der Sprachwahl.

The interface is bilingual (**English** default, German selectable via the
language switch at the top of the sidebar). The language is a display-only
setting and is **not** written to project files, so projects stay fully
interchangeable between both languages. The model's messages for invalid
parameters follow the selected language as well.

## Spaltfilm-Modell: 1D vs. 2D

Der Luftspalt zwischen Membran und Backplate kann auf zwei Arten
gerechnet werden (umschaltbar per `squeeze_model` bzw. GUI-Schalter):

- **1D (Standard):** ein Lumped-Element (Škvor-Widerstand + Nachgiebigkeit
  + Lochimpedanz). Schnell; für dichte, gleichmäßige Lochmuster
  ausreichend und für die validierten Beispiele verwendet.
- **Gültigkeitsgrenze von 1D und 2D:** beide verschmieren die Löcher.
  Bei spärlichen Lochbildern und weicher Membran beult sich die Membran
  zwischen den Löchern örtlich aus, was nur der 3D-Löser abbildet
  (beim weiten Spalt verstärkt durch die Trägheit der Spaltluft und die
  Eigenresonanz der Beule); bei Lochkreisen ist zudem die Darstellung
  als verschmiertes Band nicht eindeutig. Die Grenzfrequenz
  (`homogenization_limit()`, Gegenproben 48/53) steht in `summary()`;
  liegt sie im Hörband, warnen Modell und GUI.
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
  −17 dB, global 9°/12° −19/−37 dB, dazwischen eine Lage mit −28 dB wie
  im 2D-Modell; s. Gegenproben 48, 51 und 54). Die früher hier genannten −20 dB bei „3°" stammten aus einem
  3D-Stand, in dem die Mündungen abgeschnitten und nicht äquipotential
  waren und die Kerne sich überlappten.
  Verifiziert über Reziprozität (±1 %), Gitterkonvergenz, die
  Grenzfälle einteilig ≡ zweiteilig-ausgerichtet (5-µm-Spalt, 2 %) und
  Stufenbohrung → glatte Bohrung (winzige Senkung, 0,8 %) sowie die
  Gültigkeits-Gatter im Testlauf. DEUTLICH langsamer als 1D/2D (eine
  LU-Zerlegung je Frequenzpunkt, auf einem Kern gemessen: einteilig
  ~24 000 Unbekannte, ~0,5 s; Doppel-Backplate und K67-Typ mit zweitem
  bzw. drittem Film ~1,5–2 s; feines Gitter bis ~2 s; vor Gegenprobe 57
  das 4- bis 100-Fache) — in der GUI die Frequenzpunkte reduzieren. Jede Frequenz wird je
  Kapsel nur einmal gelöst (Gegenprobe 56): Richtdiagramm,
  Empfindlichkeit, Laufzeit-Diagnose und `summary()` teilen sich die
  Lösung, und in der GUI kostet eine zusätzliche Richtfrequenz nur ihre
  eigene. Absolute Empfindlichkeit
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
(opti.json: 1,00; K67 nominal: 1,05); **< 1** — interne Laufzeit zu
kurz, das Pattern-Minimum wandert vor 180° (K103-Demo: 0,42, Minimum
bei 114°); **> 1** — zu lang, das Minimum bleibt bei 180° gepinnt,
wird aber flacher (cardiodtest 45-µm-Spacer: 1,32; Debenham-Beispiel:
3,52, s. „Offene Punkte" 6). Die interne Laufzeit ist
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

Die Gegenproben in `tests/` laufen lassen (das Physikmodell;
Gegenprobe 63 startet zusätzlich die App ohne Browser):

```bash
pip install -r requirements-dev.txt
python microphone_capsule.py                  # alle, parallel über alle Kerne
python microphone_capsule.py -m "not slow"    # schnelle Stufe
python microphone_capsule.py -k gp48          # eine Gegenprobe
python microphone_capsule.py --lf             # nur die zuletzt gescheiterten
python microphone_capsule.py --basis-uebernehmen   # Stand-Werte bewusst übernehmen
```

`python microphone_capsule.py` ruft pytest auf (mit pytest-xdist auf
allen Kernen); weitere Argumente gehen an pytest, `pytest` direkt geht
ebenso. Jede Gegenprobe ist eine Testfunktion `test_gpNN_…`; ein Lauf
meldet alle Fehler, nicht nur den ersten. Am Ende steht das Protokoll
der OK-Zeilen in Nummernfolge (`--kein-protokoll` blendet es aus).
Markierungen: `slow` (über etwa 10 s), `feld3d` (3D-Feldlöser), `bem`
(BEM-Körpermodell). Unerwartete `UserWarning`s gelten als Fehler, und
Klassenschalter wie `MicrophoneCapsule._MASS_EXACT` werden nach jedem
Test zurückgesetzt, auch wenn er scheitert.

Laufzeit auf 4 Kernen: alle Gegenproben knapp 4 Minuten, die schnelle
Stufe ohne die `slow`-Proben (32, 41, 43, 57, 60, 64, 66b, 67, 68 und
70 im 3D-Löser) gut 2 Minuten (gemessen in einer Cloud-Sitzung). Vor Gegenprobe 57 (LU-Zerlegung des 3D-Lösers) waren es knapp
3 Minuten. Die Gegenproben 22, 23 und 48 sind
in unabhängige Teilprüfungen aufgeteilt (22a–e, 23a–f, 48a–g3), die
parallel laufen; einzeln aufgerufen sind die meisten Teile in Sekunden
fertig.
Damit die Worker gleich lange rechnen, merkt sich pytest die
Laufzeiten jedes Laufs (`.pytest_cache`) und ordnet danach die Tests
für die Verteilung `--dist worksteal`, die `python microphone_capsule.py`
einstellt (direkt: `pytest -n auto --dist worksteal`). Parallel rechnet
jeder Worker mit einem BLAS-Thread; seit der symmetrischen Zerlegung
(Gegenprobe 57) hängen die Ergebnisse davon nur noch im Bereich von
10⁻⁸ ab.

### Drei Arten von Prüfungen

Jede Aussage einer Gegenprobe gehört zu einer von drei Arten; nur die
ersten beiden sind `assert`s.

- **Invariante**: exakte Identität, Grenzfall, Reziprozität,
  Konvergenz, Literaturformel, Richtung einer Wirkung,
  Programmverhalten. Hartes `assert`; die Toleranz ist die des
  Grenzfalls (Rundung, Gitter), kein Fenster um den heutigen Wert.
- **Fremdreferenz**: Messung, FEM, digitalisierte Kurve, Datenblatt.
  `assert` mit einer Toleranz aus der Unsicherheit der Referenz.
- **Stand-Wert**: eine eigene Rechengröße ohne äußeren Grund. Bisher
  stand hier ein Zahlenfenster um das, was das Modell gerade lieferte
  (etwa „K67-Fok-Faktor zwischen 0,70 und 0,80“). Jetzt meldet der Test
  sie mit `stand.wert(...)`, ohne `assert`. Der Lauf vergleicht sie mit
  `tests/basis_werte.json`, und der Abschnitt „Stand-Werte gegenüber
  der Basis“ am Ende nennt jede Verschiebung mit altem Wert und Prozent.
  Eine Modelländerung bricht damit keine Probe mehr, die nur ihren
  eigenen alten Wert festhielt; sie zeigt ihre Wirkung als Liste.

Sonderfall **Sperrklinke** (`stand.sperrklinke(...)`): ein bekannter
Restfehler, der nur kleiner werden darf. Das sind Resonanzlage und
Verlauf gegen die FEM (Gegenprobe 32, 2D und 3D), die Amplitude des 2D-Modells
gegen Zuckerwars 4146 (38), die Restlücke der modenweisen Anregung
(34), der Off-Axis-Rest gegen Fig. 6 (41, RMS und 14 kHz), die
Obergrenze der Aktuatorlast (52), der Abstand des 2D- und des
3D-Modells zur COMSOL-FEM der B&K 4134 (59) und die Freifeldkorrektur
der 4134 gegen die NBS-Messung (64). Der Spaltwiderstand
gegen Zuckerwars Tabelle II war bis Gegenprobe 59 eine Sperrklinke
(Streuung der Škvor-Zellregel); seit dem Makroelement rechnet das
2D-Modell den Film am Lochkreis exakt, und die Tabelle ist Zuckerwars
eigene Näherung — bewusst gelockert zum Stand-Wert (Gegenprobe 60).
Neu festgelegt wurde mit Gegenprobe 65 die 2D-Amplitude gegen
Zuckerwars 4146 (0,63 → 0,87 dB): der alte Wert enthielt die
Strahlungslast im Druckgang, die den Hochtonüberschuss des 2D-Modells
verdeckte. Ebenso mit Gegenprobe 67 der Abstand des 3D-Modells zur
COMSOL-FEM der 4134 (0,066 → 0,112 dB): die Spalt-Mündung bringt den
Widerstand an die FEM (Re Z 3D/FEM 0,96 → 1,01, an der ganzen Membran
gemessen, Gegenprobe 68), der Pegel fällt aber bei 20 kHz um 0,45 dB
unter die FEM; gegen das B&K-Messmittel wird 3D besser (0,36 →
0,21 dB). (Die erste Begründung, 3D überschätze den Filmwiderstand am
Lochkreis, war ein Messfehler der Probe, s. Gegenprobe 68.) Wird ein
solcher Wert schlechter als
die Basis (über eine kleine Toleranz hinaus), scheitert der Test; wird
er besser, meldet es der Bericht.

Die Basis ändert sich nur bewusst: `--basis-uebernehmen` schreibt die
Werte aller bestandenen Tests hinein. Auch mit `-k` bleiben die Werte
nicht gelaufener Tests stehen, und Werte verschwundener Tests fallen
heraus. Eine gescheiterte Sperrklinke kommt so nicht hinein; sie zu
lockern heißt, ihren Eintrag von Hand aus der Datei zu entfernen. Als
gleich gilt ein Wert bis 10⁻⁵ relativ, weit über dem Rundungsrauschen.

Wo eine Aussage beides enthielt, ist sie geteilt: die Richtung bleibt
Invariante, der Betrag wird Stand-Wert. Beispiel Gegenprobe 24: dass
das Gewebe am Einlass die Laufzeit verlängert, wird geprüft, um wie
viel, wird gemeldet (ebenso 17, 18, 19, 52 und 53). Modellvergleiche
(2D gegen 3D, 1D gegen 2D) sind Stand-Werte, außer im Grenzfall, in
dem beide dasselbe Problem beschreiben. Dort sind sie eng geprüft:
geschlossene Rückseite und Doppel-Backplate im quasistatischen Tiefton
(23b, 23f: 1 %) und K103 dicht im Tiefton (23c: 2 %). Festgehaltene
Vorher-Werte (Gegenproben 54, 55) bleiben `assert`s, denn sie belegen,
dass eine Korrektur genau den früheren Befund erklärt.

Die Basis umfasst 115 Werte aus 33 Gegenproben, davon 12 Sperrklinken.
Der Prüfrahmen selbst hat eine Gegenprobe (`tests/test_stand_werte.py`).

## Offene Punkte

Alle bekannten offenen Punkte an einer Stelle; die Abschnitte weiter
unten erzählen, wie sie entstanden sind. Zahlen in `Code-Schrift` sind
Stand-Werte oder Sperrklinken aus `tests/basis_werte.json` — dort steht
jeweils der aktuelle Wert, die Zahlen hier sind der Stand beim
Schreiben.

### Physik: Abweichungen gegen Referenzen

1. **Lochkreise: 2D gelöst, Rest im 3D-Löser (Gegenproben 38, 52, 58,
   59, 60).** Das 2D-Feld rechnet Lochkreise seit Gegenprobe 60 als
   exaktes Makroelement: statisch auf 0,22 % gegen eine unabhängige
   Lösung, gegen die COMSOL-FEM der 4134 0,17 dB RMS
   (`gp59.rms_2d_gegen_fem_makro`; vor der Spalt-Mündung, Gegenprobe
   67, 0,31 dB, vor dem Makroelement 1,80 dB), Re Z 2 % unter der FEM
   (`gp59.widerstand_2d_zu_fem`, vorher 6 % darunter, vor dem
   Makroelement 61 % darüber). Der 3D-Löser trifft den Film an
   Lochkreisen und Mittellöchern auf 0,1–0,5 % (Gegenprobe 68; das
   Mittelloch lag bis dahin durch einen Fehler im Fußabdruck 6–55 %
   daneben, die übrigen „1–4 % zu viel" waren Messgrößen der Proben).
   Offen bleibt:
   - **3D gegen die COMSOL-FEM der 4134:** 0,11 dB RMS
     (`gp59.rms_3d_gegen_fem`, vor Gegenprobe 67 0,07 dB). Der
     Widerstand stimmt (Re Z 3D/FEM 1,01, an der ganzen Membran
     gemessen; vor der Spalt-Mündung 0,96), aber bei 20 kHz liegt 3D
     jetzt 0,39 dB UNTER der FEM (vorher +0,06 dB) — eine Frage der
     Reaktanz. Gegen das B&K-Messmittel liegt 3D besser: 0,21 statt
     0,36 dB. Die Sperrklinke ist mit Gegenprobe 67 bewusst neu
     festgelegt (0,066 → 0,112 dB).
   - **Zuckerwars 4134 von 1978** war stärker gedämpft als heutige 4134
     (20 kHz: −3,1 gegen −1,2 dB); 2D und 3D liegen gleichermaßen
     darüber (2,0 bzw. 1,6 dB RMS, `gp38.rms_2d_4134_db`; vor
     Gegenprobe 67 2,2/1,7 dB). Warum, bleibt
     offen (Tabelle I weicht von B&Ks Geometrie ab: Spalt 20,77 statt
     18,6 µm, Lochkreis 2,03 statt 1,70 mm). Am 4146 liegen 2D 0,70
     und 3D 0,83 dB daneben (`gp38.rms_2d_4146_db`; vor Gegenprobe 67
     0,87/0,73 dB, mit der Spalt-Mündung und vor dem Mittelloch-Fix
     0,70/0,90 dB, s. Punkt 8).

2. **Resonanzlage gegen die FEM (Gegenprobe 32): aufgeklärt, Rest
   offen.** Die 10 % Abstand waren zum größten Teil ein Fehler im
   Prüfaufbau: der Klassen-Standard `delay_length = 3e-3` hängte
   unbemerkt ein Laufzeitglied an (+39 % Rückvolumen; die FEM-Kapsel hat
   keins). Ohne es lag das Modell 6 % zu hoch (2D 584, 3D 581 Hz) und
   war 1,1–1,3 dB zu schwach bedämpft. Den Rest trägt die Spalt-Mündung
   (Gegenprobe 67): 2D 561, 3D 558 Hz (`gp32.verstimmung_2d` 0,021,
   `gp32.verstimmung_3d` 0,014; vorher 0,091/0,095), Überhöhung
   +7,32/+7,18 gegen +6,74 dB, RMS gegen die sechs FEM-Punkte 0,42/0,23
   dB (`gp32.rms_fem_2d`, `gp32.rms_fem_3d`; vorher 0,95/0,72 dB).
   Offen: der Film ist um rund 0,5 dB zu schwach bedämpft, und das
   Dublett der vier Bohrungen (FEM 3500/4200 Hz) liegt im 3D-Löser bei
   3350/4110 Hz — mit der Mündungsmasse 2 % tiefer als vorher
   (3421/4127 Hz). Die Mündungsmasse selbst ist im Trägheitsgrenzfall
   die der Potentialströmung (Gegenprobe 67); im Kerbenband fehlt also
   anderswo Nachgiebigkeit, oder es ist Masse zu viel. Dasselbe Band
   zeigt Gegenprobe 34 (Punkt 4).
   *Untersucht:* die 3D-Minima sind gitterkonvergent (fein 3358/4116 Hz,
   Nr 90 × Np 288 3359/4117 Hz). Nötig wären d ln f = +0,043 / +0,022.
   Empfindlichkeiten (d ln f je Änderung, erstes/zweites Minimum):
   Spaltnachgiebigkeit +10 % −0,031/−0,004, Filmleitwert +10 %
   (Filmmasse und -widerstand −9 %) +0,015/+0,010, portseitige
   Flanschmasse ganz weg +0,038/+0,014, Lochrohr −10 % +0,017/+0,004,
   Spalt adiabat statt polytrop +0,021/−0,000, Spalt-Mündung aus
   +0,020/+0,005, Membranmasse +10 % (gleiche Resonanz) +0,005/+0,010,
   Rückvolumen +20 % −0,002/−0,000. Keine einzelne Größe trifft das
   Muster; das zweite Minimum hängt auch an der Membran. Die FEM-Werte
   (3500/4200 Hz, Maximum 3860 Hz) sind auf 100 Hz gerundete Ablesungen
   aus Fig. 4 der Arbeit (±1–2 %); weiter eingrenzen lässt sich das erst
   mit genaueren FEM-Daten.

3. **Kolben- statt Modenkonvention im 1D-Pfad (Gegenproben 8, 29).**
   Der 1D-Pfad rechnet den Filmwiderstand für gleichförmigen
   Kolbenantrieb; das 2D-Feld projiziert auf die Membranform. Symptome:
   rein randbelüftet liegen 1D und 2D im Tiefton 2,8 dB auseinander
   (`gp29.abstand_1d_2d`), und im dichten Grenzfall von Gegenprobe 8
   ist 2D/Škvor 1,13 statt 1 (die Probe lässt 0,8–1,6 zu). Probe: der
   geschlossene Ausdruck der Randumgehung (Gegenprobe 37) für die
   Membranform ist bei der Gegenprobe-29-Kapsel das 1,59-Fache des
   Kolbenwerts; damit liegt 1D bei 200 Hz −0,39 dB und bei 1 kHz
   −0,68 dB neben 2D, bei 5 kHz aber −5,2 statt −2,1 dB. Einzeln
   austauschen geht also nicht; die Konvention muss im ganzen 1D-Pfad
   stimmen.

4. **Modenweise Anregung bei streifendem Einfall (Gegenprobe 34).**
   Gegen COMSOL oberhalb 5 kHz 13,4 dB mit uniformer Anregung, 6,5 dB
   mit drei Moden (`gp34.restluecke_3_moden`; mit dem Gaußband 14,1 und
   7,3 dB). Seit Gegenprobe 67 ohne das versehentliche Laufzeitglied
   und mit der Spalt-Mündung; mit dem Volumenfix allein wären es 5,5 dB
   — die Mündungsmasse verschlechtert dieses Band, wie das Dublett in
   Punkt 2. Die Reihe über 1…5 Moden (8,2/7,1/6,5/6,0/5,6 dB) flacht
   ab; mehr Moden allein schließen die Lücke nicht.

5. **Richtwirkung gegen Grinnip (Gegenproben 41, 43).** Auf Achse trifft
   der BEM-Frontfaktor Grinnips Rechnung auf 1,0 dB RMS; gegen die
   Messung (9–15 kHz) sind es 4,0 dB, Grinnips eigene Rechnung 3,1 dB —
   beide liegen darüber. Bei 90° bleiben mit einer Mode 3,5 dB RMS
   (`gp41.rest_90grad_rms`), bei 14 kHz 4,3 dB (`gp41.rest_90grad_14k`;
   vor Gegenprobe 67 3,9/4,9 dB);
   modenweise 2,4 dB (90°) und 1,7 dB (180°) gegen Grinnips Rechnung,
   3,2 dB gegen die Messung bei 90°.

6. **Debenham (Beispielprojekt).** Die gemessene Null (−13,6 dB bei
   317 Hz, −12/−8 dB bei 1–2 kHz) treffen mit der Zeichnungsgeometrie
   weder 3D noch — seit Gegenprobe 60 — 2D: mit nur dem Rand-Freistich
   bleibt die Niere flach (317 Hz: 2D −3,1, 3D −3,7 dB), mit Freistich
   über allen Lochkreisen wird sie tief. Den früheren 2D-Treffer gab ein
   Schalter, der allen Durchgangslöchern die entlastete Engstelle gab,
   sobald eines im Freistich lag (seit Gegenprobe 69 ganz ersetzt: jede
   Mündung sieht den Spalt an ihrem Ort). Vermutet sind angefaste
   Mündungen, die die Zeichnung nicht bemaßt (s. Geometriefragen).
   Zurückgestellt: im Debenham-Modell selbst sind erst Ungereimtheiten zu
   klären.
7. **Freifeldkorrektur der 4134 im Hochton (Gegenprobe 64).** Gegen die
   NBS-Messung (4134 ohne Gitter) trifft das BEM der flachen Stirnfläche
   den Tiefton auf 0,12 dB, liegt aber im Hochton bis 0,8 dB darüber
   (`gp64.rms_bem_gegen_nbs` 0,53 dB, vor Gegenprobe 65 0,58 dB).
   Ausgeschlossen: BEM-Netz und Winkelquadratur (≤ 0,03 dB), die
   Gewichtung mit der Grundmode (das reziproke Gewicht des 3D-Felds
   weicht 0,02 dB ab), die Stablänge ab 30 mm (≤ 0,3 dB); die
   Strahlungslast steckt seit Gegenprobe 65 richtig nur im Freifeldgang.
   Offen sind die reale Stirnform ohne Gitter (Gewinde, Fase,
   Klemmring) und der Messstab (Vorverstärker ⌀12,7 mm).
8. **Aktuatorlast (Gegenprobe 65).** Zuckerwars Fig. 6/7 und
   vermutlich auch B&Ks Messmittel sind Aktuatormessungen. Der Aktuator
   belastet die Membran (Luft zwischen Gitter und Membran, Strömung
   durch die Schlitze); Capsim vergleicht dagegen den lastfreien
   Druckgang. Bis Gegenprobe 65 wirkte die Freifeld-Strahlungslast dort
   als ungefährer Ersatz; ohne sie liegen 2D/3D gegen Fig. 6 bei
   2,17/1,71 dB (vorher 2,11/1,67), gegen das Messmittel bei 0,61/0,36 dB
   (vorher 0,56/0,32), und die Sperrklinke 2D gegen die 4146 ist neu
   festgelegt (0,63 → 0,87 dB; 3D 0,744 → 0,734 dB). Seit der
   Spalt-Mündung (Gegenprobe 67): Fig. 6 2,01/1,56 dB, Messmittel
   0,46/0,21 dB, 4146 2D 0,70 dB. Gegenprobe 52 d
   grenzt eine reine Luftmasse vor der Membran ein (höchstens +0,5 dB);
   eine hergeleitete Aktuatorlast (Abstand, Schlitzgeometrie) fehlt.
9. **K67 mit realem Bohrbild: Laufzeitverhältnis 1,49 — Ursache ist
   die externe Bezugslaufzeit (Gegenprobe 71).** Mit dem aus dem Foto
   vermessenen Bohrbild (108 Senkungen auf dem 2-mm-Raster, Mitte frei,
   Hälften um 90° verdreht), Randnut und Mittenaussparung liegt die
   Auslöschung bei 1 kHz im 3D-Löser bei −14,2 dB statt der
   publizierten −26 dB; intern/extern 1,49 (26,6 gegen 18,3 mm).
   *Untersucht:*
   - **Die interne Laufzeit stimmt.** Die Rückkette zerfällt exakt in
     Blöcke (D_r = T_rück[1,1]); über 90 % trägt C_vorn·R_Zwischenspalt
     — das Volumen der Frontsenkungen mal dem Film zwischen den
     versetzten Kernen. Den Film prüft eine unabhängige Lösung: das
     unendliche Schachbrett aus Quellen und Senken im 2-mm-Raster gibt
     (1/πK)·[ln(p/a) − 0,617] je Kernpaar; die zwei Škvor-Halbzellen des
     2D-Modells liegen 19 % darunter. Damit korrigiert käme 2D auf
     ≈ 27,5 mm — der 3D-Löser rechnet 27,2 mm (fein 26,6 mm).
   - **Die externe Bezugslaufzeit passt nicht zur U87.** Die d_ext-Kugel
     (18,3 mm) entspricht im BEM einem 56-mm-Körper 7 mm KOAXIAL hinter
     der Kapsel. Die U87 ist seitenbesprochen; hinter der Rückmembran
     sitzt nur der Korb. Die freie Scheibe gibt 32,2 mm (Sphäroid,
     exakt), der freie Kopf im BEM 34,9 mm; dagegen ist die interne
     Laufzeit 18–22 % zu KURZ (0,82/0,78): 1 kHz −6,9/−22,3 dB bei
     90/135°, tiefste Stelle −27 dB bei 141–145°, 180° −17 bis −19 dB.
     Gewählt wurde die Kugel, als die interne Laufzeit noch ~17 mm war
     (Gegenprobe 20: „freie Scheibe ergäbe eine Superniere bei 123°") —
     zwei Fehler hoben sich auf; mit Spalt-Mündung, exaktem Zwischenspalt
     und realem Bohrbild ist die interne Seite vollständig, und die
     Kompensation kippt.
   - **Tiefe der Null: auch der Betrag.** Der Außenweg ist eine reine
     Laufzeit (|G(180°)| = 1,00), das innere RC-Netzwerk hebt den Betrag
     (|D_r| = 1,05; bei reinem RC 1/cos φ). Das allein begrenzt die Null
     auf −20…−26 dB: mit freier Scheibe und 45-µm-Spacer passt die Phase
     (1,04), die Null bleibt bei −19,5 dB.
   Offen: welche externe Laufzeit der Korb und die Halterung der U87
   wirklich ergeben, und ob dem inneren Netzwerk Trägheit fehlt, die es
   von einem RC-Glied zu einer Laufzeit macht. Ein passender Spacer allein
   wäre ein Fit.
### Modellgrenzen (dokumentiert, nicht behoben)

- **Einmodenbild von 1D/2D:** die Membran kann dem Filmdruck nicht
  ausweichen. 3D liegt deshalb oberhalb der Filmgrenze höher
  (Randschlitz 20 kHz: `gp52.formanpassung_20khz` 1,6 dB; K103 dicht bei
  60 V und 1 kHz: Spannung 3D/1D `gp23c.spannung_1khz` 0,92). Mehrere
  2D-Moden helfen nicht, weil sie sich einen Spaltknoten teilen. Bei
  weicher Membran mit Randspalt liegt 3D bis 8 dB über 2D (Gegenprobe
  29).
- **Homogenisierung der Löcher:** gilt bis f_hom bzw. bis zur
  Lochkreis-Grenze f_ring (Gegenproben 48, 53); darüber warnen Modell
  und GUI (beide Grenzen sind vorsichtige Schätzungen und stehen nur als
  zugeklappter „Hinweis“ am Seitenende). Weiche Großmembran-Kapseln
  liegen fast immer im
  Warnbereich. Bei der K67 liegen 2D und 3D auf Achse bis 3,2 dB
  (8 kHz) auseinander, mit einem Tieftonversatz von 1,3 dB, den f_hom
  nicht erklärt (`gp22e.empf_3d_zu_2d`, Zwischenspalt-Geometrie).
- **Lochkreise im 2D-Feld (Makroelement, Gegenprobe 60):** statisch
  exakt; dynamisch gilt die Speicherung je Zelle über das Zellmittel des
  Drucks. Freistich-Zellen im Band gehen mit dem statischen
  Leitwertverhältnis ein (Debenham: höchstens 0,2 dB bei 10–20 kHz), die
  Durchbiegung der polarisierten Membran über den mittleren Leitwert des
  Bandes. Kreise mit teilerfremden Lochzahlen und nicht abgeklungenen
  Harmonischen werden getrennt, wenn das gemeinsame Sektorgitter zu
  groß würde (Debenham).
- **Folie als Membran mit Randschicht (Gegenprobe 62):** die
  Biegesteifigkeit wirkt als um √(D/T) versetzte Einspannung. Die
  Plattendispersion im Innern (ω² um k²·D/T höher; an der Grundmode
  ≈ z₁²λ²/2, B&K-Nickel 1·10⁻⁴, höhere Moden mehr) ist nicht gerechnet.
  Ab 1 % Abweichung der Grundfrequenz gegen die Platte warnt das Modell
  (ohne Pfosten λ ≈ 0,06), ab einer Randschicht über ein Viertel der
  Membranbreite bricht es ab.
- **Spalt-Mündung (Gegenproben 67, 70):** an allen MEMBRANseitigen
  Mündungen (Durchgangs-, Sack-, Stufenbohrung; 1D, 2D, 3D) und seit
  Gegenprobe 70 auch in den membranlosen Spalten (K67-Zwischenspalt,
  K103-Spacer beidseitig, q = 0). Gerechnet für die von der Membran
  getriebene Zelle; strömt Luft bei ruhender Membran von hinten durch
  die Löcher, ist der Fehler von der Ordnung q·ΔZ (q
  Lochflächenanteil). Die Stufenbohrung trägt sie wie den Zellterm in
  beiden Zweigen der Senkung. Über h/a = 2 geklemmt. Im
  K67-Zwischenspalt legt 1D/2D sie je Loch in Serie und setzt damit
  versetzte Kerne voraus (wie schon die Škvor-Halbzelle); fluchtende
  Kerne kann nur der 3D-Löser, der sie auf die Filmflächen legt.
- **Elektrode und Einmodenbild (Gegenprobe 68):** die Kette wandelt die
  Modenamplitude mit Θ in Spannung; im 3D-Feld sieht die Spannung die
  Auslenkung nur über der Elektrode, die über den Löchern fehlt. Unter
  Filmlast weicht die Form ab, und die 2D-Spannung sieht bei großen
  Löchern 2–3 % weniger Verlust als die 3D-Spannung
  (`gp68.elektrodengewicht_lochkreis` 1,03), bei gleichem Film.
- **Höhere Membranmoden (`membrane_modes`):** die elektrostatische
  Feder-Erweichung wird nicht auf sie übertragen, und ihre
  Filmdämpfung wird gleich der Grundmode gesetzt (konservativ).
- **Strahlungsimpedanz:** Kolben in unendlicher Schallwand
  (Gegenprobe 27).
- **BEM:** der starr montierte Körper bildet einen ungedämpften
  Ringspalt-Resonator (Welligkeit 4–6 kHz, real durch die elastische
  Halterung bedämpft); offener Rückeinlass nur am glatten Zylinder
  (Warnung, Gegenprobe 44).
- **Gewebe im 3D-Doppelmembran-Pfad:** 3D und 2D rechnen den inneren
  Gleichtaktpfad verschieden (3D bei 10 Hz 34 % größer); die
  Gewebedämpfung liegt bei −5,3 gegen −6,0 dB (Gegenprobe 27, nur
  ausgegeben). Welches Bild näher an der realen Kapsel liegt, ist
  nicht geprüft.
- **3D-Gitter:** grob meist auf ~0,2 dB; Mündungen, die azimutal
  kleiner als eine Zelle sind (Debenham), brauchen das feine Gitter;
  über 50 000 Zellen je Feld wird gewarnt (Gegenproben 50, 51).
- **Gatter:** Randspalt nicht mit K103-Spacer/Rückplatte und im
  1D-Pfad nur ohne Bohrungen; Eigenrauschen nur in 1D/2D; die
  Empfindlichkeit gilt im Leerlauf an der Kapsel, ohne Verstärker und
  Korb.

### Geometriefragen an realen Kapseln

Diese Punkte kann das Modell nicht entscheiden; sie brauchen Maße oder
Messungen der realen Kapsel.

- **K67-Kernlage (Gegenprobe 22e):** bestimmt die Tiefe der
  Auslöschung: global 9°/10,5°/12° verdreht −23,2/−25,0/−22,6 dB bei
  180° und 1 kHz (`gp22e.ausloeschung_9grad`, `…_12grad`), vollständig
  versetzt −14,6 dB, das 2D-Modell −21,7 dB. Seit der Spalt-Mündung im
  Zwischenspalt (Gegenprobe 70) ist die Abhängigkeit flacher (vorher
  9°/12° −20,4/−31,0 dB, 2D −26,4 dB). *Seit Gegenprobe 71 aus einem Foto
  bekannt:* quadratisches 2-mm-Raster mit Schachbrett-Durchbohrung, die
  zweite Hälfte um 90° verdreht — kein Kern fluchtet, jeder sieht den
  nächsten Kern der Gegenseite 2 mm entfernt. Das ist der vollständig
  versetzte Fall; mit exakter Rasterlage im 3D-Löser −13,7 dB
  (`examples/k67_experimentell_bohrbild.json`). Offen bleibt damit nicht
  mehr die Kernlage, sondern warum die Laufzeit zu lang ist (s. „Offene
  Punkte“ 9).
- **Debenham-Mündungsfasen:** s. Punkt 6 oben.
- **K103:** die inneren Maße sind nicht veröffentlicht; die Demo ist
  eine Vorlage zum Abstimmen und zurzeit keine Niere (Laufzeitverhältnis
  0,42, Minimum bei 114°).

### Werkzeug

- Die GUI (`app.py`) hat seit Gegenprobe 63 einen Test: jedes
  Beispielprojekt läuft durch die echte App, liegt in den Feldbereichen
  und ergibt die Kurve des Modells; seit Gegenprobe 66 in beiden
  Sprachen, mit vollständigen Übersetzungstabellen. Es fehlt ein Test
  des Uploaders selbst (der AppTest kann ihn nicht bedienen; die Probe
  schreibt den Session-State wie `_load_project()`). Die Meldungen des
  Modells bei ungültigen Parametern sind seit Gegenprobe 66 c
  zweisprachig (`ParameterFehler`); seine Warnungen (`UserWarning`)
  bleiben deutsch, die App zeigt sie aber nicht (nur im
  Konsolenprotokoll).
- Kein automatischer Testlauf bei jedem Push (kein GitHub-Workflow).
- Laufzeit: die BEM-Proben 41, 43, 44, 26 und 21 brauchen etwa 117 der
  316 s Rechenzeit; Gegenprobe 41 allein (54 s) ist die Untergrenze des
  parallelen Laufs. Eine wiederverwendete BEM-Körperlösung (wie beim
  3D-Löser) würde helfen.
- Zwei weite Fenster sind noch als Invariante formuliert: Gegenprobe 8
  (2D/Škvor 0,8–1,6, s. Punkt 3) und Gegenprobe 48 (f_hom 1–12 kHz bzw.
  4–20 kHz).
- Die Zahlen in den Abschnitten unten sind Schnappschüsse; den
  aktuellen Stand nennen die Basis und der Bericht am Ende jedes
  Testlaufs.
- `microphone_capsule.py` hat rund 8600 Zeilen; eine Aufteilung in
  Module (Spaltfilm, 3D-Löser, BEM, Elektrostatik) steht aus.

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
inneren Maßen: Ruhekapazität C₀ = 50,0 pF (trifft den nachgemessenen
Wert), Niere **−5,5/−13,4/−21,7 dB @ 90/135/180° (1 kHz)**, Minimum bei
180° von 250 Hz bis 2 kHz, glatter Präsenzpeak +5,2 dB @ 13,5 kHz (gegen
1 kHz, roh), Empfindlichkeit 19,1 mV/Pa. Bis Gegenprobe 69 waren es
−5,9/−15,0/−26,4 dB — praktisch die publizierten U87-Werte
(−6/−17/−26 dB) — und 19,9 mV/Pa; seither trägt auch die Mündung der
Kerne in den Zwischenspalt ihre Umlenkung (Gegenprobe 70). Wie tief die
Niere der realen Kapsel ist, hängt an der nicht dokumentierten Lage der
Kerne (s. „Geometriefragen“). **Wichtig für Datenblatt-Vergleiche:** das
Modell rechnet die **nackte Kapsel**. Die K67 ist bewusst hell ausgelegt
(„Pre-Emphasis"), und die U87-Elektronik nimmt das über Gegenkopplung
wieder heraus („De-Emphasis"); dazu kommt der nicht modellierte Korb.
Ein Kapselmodell **muss** im Hochton also ÜBER der veröffentlichten
Gesamtkurve liegen — Übereinstimmung mit dem Datenblatt oberhalb
~10 kHz wäre ein Warnzeichen, kein Gütesiegel. (Bis Gegenprobe 30 lag
das Minimum in den Mitten knapp vor 180°, ~160°.)
Über „Projekt laden" importierbar.

`examples/cardiodtest.json` — identisch zum nominellen K67-Datensatz
bis auf **ein einziges Maß**: den Spacer zwischen den Elektrodenhälften
(45 statt 50 µm — das am wenigsten sicher verifizierte innere Maß,
plausibel 40–65 µm). Gedacht war er, um die Null auf 180° zu pinnen;
das tut seit Gegenprobe 30 schon die nominelle K67 (Laufzeitverhältnis
1,05). Der engere Spacer verzögert jetzt über (1,32): die Null bleibt
bei 180°, wird aber flacher (−15,6/−17,0/−17,1/−15,6 dB @
250/500/1k/2k, 90° = −5,1 dB, C₀ = 50,0 pF, 18,1 mV/Pa). Die Datei
zeigt damit die Über-Verzögerung; die Niere entsteht auch hier
vollständig aus den akustischen Parametern (s. Abschnitt
„Nierenbildung").

`examples/k67_experimentell_bohrbild.json` — **experimentell**: die K67
mit dem realen Bohrbild, der Randnut und der Mittenaussparung, vermessen
aus einem Foto zweier Backplate-Hälften (s. „K67-Bohrbild aus dem Foto
und Freistich in der Elektrostatik (Gegenprobe 71)“). Gegenüber
`u87_k67_projekt.json` geändert: 108 Senkungen auf dem quadratischen
2-mm-Raster als 13 Lochkreise (54 durchgebohrt, Mitte frei), Elektrode
25,4 mm, Membran 27,2 mm, Mittenterminierung 1 mm und eine 5-mm-
Aussparung in der Mitte. Auslöschung bei 1 kHz (tiefste Stelle, jeweils
bei 180°): 3D −14,2 dB (feines Gitter, Referenz), 2D −17,4 dB — das
2D-Modell warnt hier selbst (lochfreie Mitte). Voreingestellt ist 2D;
den Referenzwert liefert das 3D-Feldmodell (in der App ~6 min bei 100
Punkten).

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
ohne jede Kalibrierung (bis Gegenprobe 69; mit der Spalt-Mündung im
Zwischenspalt −21,7 dB, s. Gegenprobe 70). Der aktive Membrandurchmesser (26 mm) und der
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
und Struktur), Gewebe dämpft **monoton** (Passivität), und die
Reziprozität X_r = −B_f bleibt erhalten. Die Dämpfung der 1D/2D-Kette
wird nur noch zum Vergleich ausgegeben (seit Gegenprobe 54: −5,3 gegen
−6,0 dB bei 10⁵ Rayl und 100 Hz; nach Gegenprobe 52 −4,9 dB). Beim Druckgradientenempfänger ist die
Gewebedämpfung im Tiefton die Differenz aus Gradientenantrieb und
Gleichtakt-Druckabfall am Gewebe; beide sind vergleichbar groß, und die
Differenz verstärkt jeden Unterschied der inneren Gleichtaktantwort.
Die bestimmt an diesem Prüfling (12 gestufte Löcher, Zwischenspalt) der
innere Widerstandspfad, den 2D und 3D verschieden rechnen (3D bei 10 Hz
34 % größer, bei dichter einteiliger Platte 4 %). Die frühere
Übereinstimmung auf 0,04 dB war Zufall: der unbelastete Membranring
glich den Unterschied aus.

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
Was weder Film noch Rohr trägt, ist die Umlenkung zwischen beiden; sie
steht seit Gegenprobe 67 als eigene Mündung im Modell, mit dem
Dünnspalt-Grenzfall, in dem die Aussage hier exakt wird.

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
Lochbilder** bedämpft (3D-Stand vor Gegenprobe 48: 3°, −1,2 dB; wie
stark, hängt heute an der undokumentierten Kernlage, s. „Offene
Punkte"). Bewusste Näherungen: die
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
(10 → 200 µm: +22,1 dB); das 2D-Zweitor bleibt mit Randknoten reziprok,
auch mit Bohrungen **und** Randspalt gleichzeitig; 1D und 2D liegen im
Tiefton 2,8 dB auseinander (Stand-Wert; zur vermuteten Ursache s.
„Offene Punkte").

**Auch im 3D-Löser** hängt derselbe Ringkanal über den Randflächen-
Leitwert an der äußersten Filmzellreihe. Verankert am **Kolben-
Grenzfall**: nur wenn die Membran sich *nicht* verformen kann,
beschreiben 1D/2D (Grundmode φ erzwungen) und 3D (freies Membranfeld)
dasselbe Problem — mit steifer Membran im quasistatischen Tiefton fallen
beide auf **0,01 dB** zusammen (seit dem Massenfaktor 8/j₀₁²,
Gegenprobe 55; davor 0,3 dB), und das Ergebnis ist von der azimutalen
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

**Nebenbefund (erledigt mit Gegenprobe 31):** Der Kettenpfad führte den
Škvor-Widerstand zweimal — einmal in `_membrane_impedance`, einmal im
Backplate-Zweitor. Er zählt jetzt genau einmal (s. u.).

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

*Nachtrag Gegenprobe 70:* mit der Spalt-Mündung auch im membranlosen
Zwischenspalt liegt die K67 bei −5,5/−13,4/**−21,7** dB und 19,1 mV/Pa,
das Minimum weiter bei 180°. Geprüft bleibt das Minimum bei 180°; die
Tiefe ist seither ein Stand-Wert (`gp30.k67_180grad`) statt
„< −25 dB“ — die alte Grenze lehnte sich an die publizierten −26 dB an
und hielt nur, solange die Umlenkung im Zwischenspalt fehlte. Wie tief
die reale Kapsel auslöscht, hängt an der nicht dokumentierten Kernlage
(Gegenprobe 22e).

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

*Nachtrag (Gegenproben 51, 52):* mit konturtreuen Mündungen und
gekoppeltem Membranring liegt 3D bei 4 kHz für 48 und 96 Bohrungen
1,0 bzw. 0,8 dB über 2D — auch bei 96 Bohrungen, deren f_hom (7,5 kHz)
weit darüber liegt. Das ist die lochbildunabhängige Formanpassung der
Membran oberhalb der Resonanz, die das 2D-Einmodenbild nicht kann
(Gegenprobe 52), nicht die Filmdämpfung. Verglichen wird deshalb bei
1 kHz (die stark erweichte 45-V-Kapsel hat ihre Resonanz unter 300 Hz):
−2,2 / −0,2 / +0,3 dB bei 12 / 48 / 96 Bohrungen. Die Entscheidung
„einmal gezählt" bleibt.

*Nachtrag (Gegenprobe 54):* mit der Θ-konsistenten 3D-Wandlung
verschwindet der Tieftonversatz (20 Hz: +0,02 / −0,02 dB bei 48 / 96
Bohrungen). Bei 4 kHz liegt 3D nur noch 0,5 bzw. 0,3 dB über 2D, bei
1 kHz sind es −1,8 / +0,3 / +0,8 dB bei 12 / 48 / 96 Bohrungen. Die
Kapsel läuft bei 45 V mit 24 % Durchbiegung; die Spannung gewichtet die
Membranmitte (1/g²), wo der Film am steifsten ist und die Mitte
zurückbleibt. Für 12 Bohrungen prüft die Gegenprobe jetzt die
1-dB-Toleranz der Homogenisierungswarnung (f_hom 133 Hz) statt einer
frei gewählten 2-dB-Grenze. Mit dem Massenfaktor 8/j₀₁² (Gegenprobe
55) sind es bei 1 kHz −1,8 / +0,3 / +0,7 dB, bei 4 kHz 0,6 / 0,4 dB.

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
  prinzipiell nicht haben; der 3D-Löser zeigt es (3406/4127 Hz).
  Die Resonanzlage lag lange 10–13 % zu tief; das war ein Fehler im
  Prüfaufbau (ein versehentliches Laufzeitglied) plus die fehlende
  Spalt-Mündung. Seit Gegenprobe 67: 2D 561, 3D 558 Hz gegen 550 Hz
  (s. „Offene Punkte" 2).
* **Messung (Gegenprobe 38):** Zuckerwar, JASA 64, 1278 (1978), B&K
  4134 und 4146 — Tabelle I vollständig, Tabelle II die Ersatzelemente,
  Fig. 6/7 Amplitude **und** Phase gegen Messwerte. M und C_M treffen
  analytisch (< 0,2 %), der Frequenzgang des 4134 liegt 0,3 dB RMS neben
  der Messung. Beim 4146 bleibt die Streuung der Škvor-Zellregel
  q = n·r²/a_bp² als dokumentierter Rest (Schranke 1,6 dB RMS).
* **Membranmodell der Referenzen (Gegenprobe 62):** FEM, COMSOL-Modell
  der 4134 und Zuckerwars Tabelle II rechnen die Folie als reine
  Membran; verglichen wird dort ohne Randschicht. Gemessene Prüflinge
  bekommen die Vakuumresonanz vorgegeben, weil ihre Spannung aus ihr
  zurückgerechnet ist.

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
  höheren Zweige. Mit dem Massenfaktor 8/j₀₁² (Gegenprobe 55) läuft die
  normierte Reihe für viele Moden exakt auf den freien Kolben.
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

*Nachtrag (Gegenprobe 54):* die Spannung einer Ringmembran lag in 1D/2D
um den Faktor 2·m₁ zu hoch — der Wandlerkoeffizient rechnete auch mit
Pfosten mit der Parabel (mittlere Auslenkung 1/2), die Ringform hat
m₁ = 0,62…0,63 (1–3 mm Pfosten auf 25 mm): +1,9…2,0 dB. Der frühere
3D-Pfad nutzte denselben Koeffizienten und verdeckte den Fehler; die
Mechanik (Nachgiebigkeit, Moden, Pull-in) war davon nicht betroffen.

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

Die Probe rechnet wie Warren eine reine Membran. Mit der Randschicht
der Folie (Gegenprobe 62) gilt Warrens Ā für den wirksamen Radius
a − √(D/T) (0,7892; mit a gerechnet 0,7998).

**Wirkung:** der Pull-in sinkt um 1,3…2,5 % — bei der K67 von 74,4 auf
**72,6 V** (Arbeitspunkt bei 60 V: w₀ = 11,7 µm, C₀ = 50,3 pF,
Erweichung 28 %; seit dem Massenfaktor 8/j₀₁², Gegenprobe 55, ist die
Membran bei vorgegebener Resonanz 3,6 % steifer: 73,9 V, 11,1 µm,
50,0 pF, 26,5 %). Vier Prüflinge des Selbsttests saßen genau auf oder
knapp über ihrem exakten Pull-in — die weiche 1"-Kapsel der
Gegenproben 29/31 und die Debenham-Variante ohne Sacklöcher in 22
(48,9…50,0 V gegen 50 V Bias) sowie die fast volle K67-Elektrode in 19
(59,7 V gegen 60 V) — und laufen jetzt mit 45 bzw. 50 V; sie prüfen
Filmphysik bzw. Struktur, nicht die Nähe zum Kollaps. Die Beispiel-
projekte bleiben stabil (Debenham 53,0 V bei 50 V Betrieb, K67 73,9 V
bei 60 V; Stand Gegenprobe 55).

### Homogenisierungsgrenze der 1D/2D-Modelle (Gegenprobe 48)

1D und 2D verschmieren die Bohrungen zu einer Senkendichte und zwingen
der Membran über jeder Lochzelle die **globale Modenform** auf. Über
einem großen lochfreien Bereich staut sich aber der Film, und die
gespannte Membran weicht ihm örtlich aus — sie beult sich zwischen den
Löchern. Das kann nur der 3D-Löser. Maßgeblich ist das Verhältnis der
Filmkraft zur Steifigkeit der Beule über dem größten lochfreien Bereich
(Überdeckungsradius ρ = größter Abstand eines Elektrodenpunkts zur
nächsten Durchgangsbohrung). Im Tiefton ist die Filmkraft rein viskos:

    Π(ω) = ω · 12μ·ρ⁴ / (h³ · T · j₀₁²)

Der Atmosphärendruck kürzt sich heraus. Allgemein (Gegenprobe 53) zählt
die **volle Filmleitfähigkeit** K(ω) — beim weiten Spalt wird die
Spaltluft träge, die Filmkraft wächst um |K₀/K| — und die **Masse der
Membran**: die Beule hat eine eigene Resonanz f_ρ = f_res·a_mem/ρ, bei
der ihre Steifigkeit verschwindet. Π wächst deshalb um den Faktor
|K₀/K(ω)| / |1 − (f/f_ρ)²|. Gegen den (korrigierten, konvergierten)
3D-Löser setzt die 1-dB-Mehrabweichung oberhalb Π = 10 ein (zwei
Kapseln, T = 45 und 109 N/m, Spalte 20/25/38/65 µm). Die Grenzfrequenz
f_hom ist die Frequenz mit Π = 10 (im Tiefton unverändert
10 / (2π · 12μ·ρ⁴ / (h³·T·j₀₁²)), sonst numerisch, stets unter f_ρ).

**Lochkreise** haben eine zweite Grenze: das Radialfeld verschmiert
jeden Lochkreis zu einem Band, dessen Breite keinen eindeutigen
physikalischen Wert hat. Wo das Ergebnis mit der schmalsten
darstellbaren Liniensenke um mehr als 0,5 dB anders ausfällt, ist die
Darstellung nicht belastbar (s. Gegenprobe 53).

Liegt eine der beiden Grenzen im Hörband (< 20 kHz), rechnet das Modell
trotzdem, **warnt** aber (`UserWarning`, in der GUI als Hinweis, in
`summary()` als Zeile „Loch-Homogenisierung bis", mit der Ursache).
`homogenization_limit()` liefert ρ, T, f_ρ, f_hom, die Lochkreis-Grenze
f_ring und die kleinere f_limit. Beispiele:

| Kapsel | ρ | T | f_ρ | f_hom | f_ring |
|---|---|---|---|---|---|
| K67 (60 Durchgangs-Senkungen) | 3,2 mm | 13 N/m | 4,7 kHz | 1,0 kHz | — |
| Debenham (12 Bohrungen auf Lochkreisen) | 6,0 mm | 40 N/m | 4,4 kHz | 43 Hz | ≤ 20 Hz |
| B&K 4134 (6 Bohrungen auf einem Kreis + Randspalt) | 2,0 mm | 3160 N/m | 50 kHz | 33 kHz | 6,0 kHz |
| Standardkapsel (60 Durchgangs- + 30 Sacklöcher, 8 kHz) | 2,1 mm | 440 N/m | 42 kHz | 23 kHz | — |

Vorher (rein viskos, ohne Lochkreis-Grenze) lagen K67, B&K und
Standardkapsel bei 1,1 / 73 / 58 kHz. Die B&K 4134 bekommt damit eine
Warnung ab 6 kHz — vorsichtig: 2D und 3D liegen dort 0,3 dB
auseinander, bei 10 kHz 0,8 dB und bei 20 kHz 3,0 dB (im Tiefton
gleich, s. Gegenproben 54/55).

Großmembran-Kapseln mit weicher Folie liegen damit fast immer im
Warnbereich. Π = 10 ist bewusst der vorsichtige Rand; bei der K67
liegen 2D und 3D auf Achse bis zu 3,2 dB auseinander (−3,2 dB bei 8 kHz), tragen
aber schon im Tiefton einen Versatz von ~1,4 dB, den f_hom nicht erklärt
(Zwischenspalt-Geometrie, s. u.; vor den konturtreuen Mündungen −4,0
bzw. ~2 dB, vor dem gekoppelten Membranring −3,5 bzw. ~1,2 dB).

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
   nach Lochgröße: konvergent (0,97 → 0,34 dB je Halbierung; mit
   konturtreuen Mündungen, Gegenprobe 51, 0,12 → 0,00 dB).
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
  der Gegenseite, ~2 mm Querweg) −17 dB bei 180°/1 kHz; global
  gedreht 6°/9°/12°/15°/18° −10/−19/−37/−31/−14 dB — zwischen 9° und
  12° liegt eine Lage, die das 2D-Modell (−28 dB) trifft, dessen
  Škvor-Zelle einen Querweg von etwa einem Zellradius annimmt;
  fluchtend (0°) wandert das Minimum auf ~106°. Das ist eine
  Geometriefrage an der realen Kapsel, kein Modellfehler —
  `half_rotation_deg` stellt sie ein. (Mit konturtreuen Mündungen,
  Gegenprobe 51, grob und fein auf ~0,5 dB gleich, und Θ-konsistenter
  Wandlung, Gegenprobe 54; vorher traf 12° mit −29 dB, vor der
  Konturkorrektur versetzt −11 dB, 9° −29 dB.)
* **Standardgitter des 3D-Lösers:** das grobe Gitter löst kleine
  Mündungen nur mit rund einer Zelle je Radius auf. Seit Gegenprobe 50
  gibt es das feine Gitter (Schalter, s. u.), seit Gegenprobe 51
  konturtreue Mündungen — das grobe Gitter liegt damit meist auf
  ~0,2 dB. Gegenprobe 48 rechnet ihre Trennprobe fein.
* **B&K 4134 im 3D:** bei 13…20 kHz liegt der 3D-Löser 2,2…3,5 dB über
  der Messung (mit konturtreuen Mündungen, gekoppeltem Membranring und
  physikalischer Membranspannung, grob wie fein; mit der aus der
  Kettenresonanz zurückgerechneten Spannung 2,2…3,8 dB), das 2D-Modell
  höchstens 0,6 dB. Aufgeklärt in
  Gegenprobe 52: der 3D-Löser rechnet die Modellgleichungen richtig
  (unabhängige Referenz auf 0,001 dB); das 2D-Modell überschätzt für
  den Lochkreis der 4134 den Filmwiderstand 1,7-fach, was den Hochton
  senkt. Dass es damit die Messung trifft, heißt: die reale Kapsel
  dämpft stärker als der Reynolds-Film. Die Aktuatormessung als
  Erklärung ist eingegrenzt und scheidet aus (s. Gegenprobe 52); die
  Ursache der stärkeren Dämpfung ist offen. *Nachtrag (Gegenprobe 60):*
  mit dem Makroelement liegt auch 2D über der Messung, wie 3D.

### 3D-Gitter grob/fein (Gegenprobe 50)

Der 3D-Löser rechnet standardmäßig auf dem **groben** Gitter (60
Radialzellen, 96…320 azimutal). Der GUI-Schalter **„Feines 3D-Gitter"**
im Abschnitt Spaltfilm-Modell (in der Klasse `grid_3d="fine"`) schaltet
auf das **feine** Gitter um; er ist nur mit dem 3D-Modell aktiv, die
Wahl wandert in die Projektdatei, und `summary()` nennt das Gitter.

**Regel.** Der Gitterfehler sitzt an den Mündungen: eine Zelle gehört
zur Mündung, wenn ihre Mitte darin liegt. Seit Gegenprobe 51 rechnen
die Randflächen mit dem wahren Abstand zur Kreiskontur (s. u.); eine
Mündung, die kleiner als eine Zelle ist, bleibt aber unkorrigiert. Wie
stark das wirkt, hängt am Verhältnis Zellweite zu Mündungsradius — und
welche Richtung es bestimmt, an der Bauform (K67: radial, das 96er-
Umfangsraster einteiliger Elektroden: azimutal). Das feine Gitter
richtet sich deshalb nach der kleinsten Mündung, die ein Film sieht
(membranseitig Loch bzw. weite Senkung, Sacklöcher, im Zwischenspalt
der K67 der enge Kern): mindestens **2 Zellen je Mündungsradius**,
radial und azimutal (dort am äußersten Lochmittenkreis), und in beiden
Richtungen mindestens 1,5-mal feiner als grob. Obergrenze 50 000
Zellen je Feld; greift sie (winzige Löcher), warnen Modell und GUI.

**Wirkung** (1 kHz, mit konturtreuen Mündungen, Abweichung gegen das
feinste gerechnete Gitter; Zeit je Frequenzpunkt auf einem Kern, allein
gemessen):

| Kapsel | grob | fein | Referenz | Abw. grob | Abw. fein | Zeit grob → fein |
|---|---|---|---|---|---|---|
| 12 × ⌀1,4 mm auf 1" | 60 × 96 | 90 × 162 | 180 × 324 | 0,10 dB | 0,03 dB | 0,5 → 3 s |
| 24 × ⌀0,46 mm auf ½" (1/4/12 kHz) | 60 × 96 | 90 × 254 | 135 × 380 | 0,09…0,23 dB | < 0,02 dB | 0,4 → 5 s |
| 48 × ⌀0,7 mm auf 1" | 60 × 192 | 90 × 376 | 135 × 564 | 0,16 dB | 0,03 dB | 2 → 11 s |
| Debenham (einteilig), Pegel / 180° | 60 × 96 | 90 × 388 | 120 × 512 | 0,45 / 1,0 dB | 0,04 / 0,1 dB | 0,8 → 20 s |
| K67, Pegel / 180° (gegen fein) | 60 × 240 | 90 × 480 | — | 0,05 / 0,4 dB | — | 22 → 34 s |

Speicher: fein bis etwa 3 GB (K67-Typ), grob etwa 1 GB. Das grobe
Gitter reicht damit meist auf ~0,2 dB; wo die Mündungen azimutal
kleiner als eine Zelle sind (Debenham: 12 Löcher ⌀0,71 mm auf dem
96er-Raster), braucht es das feine. Vor der Konturkorrektur lag grob
bei 1–1,6 dB, die K67-Rückdämpfung bei 3–4 dB daneben, und selbst
sehr feine Gitter wanderten noch um einige Zehntel dB.

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

### Konturtreue Mündungen im 3D-Löser (Gegenprobe 51)

Jede Lochmündung ist im 3D-Gitter die Menge der Zellen, deren Mitte in
ihr liegt — eine Treppe, deren wirksamer Rand je nach Lage um einen
Bruchteil der Zellweite neben der Kreiskontur liegt. Auf den
Filmflächen zwischen einer Mündungs- und einer Filmzelle strömt die
Luft aber nicht über den vollen Mittenabstand Δ durch Film, sondern nur
über das Stück ℓ außerhalb der Kontur. Nach Shortley–Weller wird der
Flächenleitwert dort mit **Δ/ℓ** skaliert; den Schnittpunkt liefert
radial der Strahl, azimutal der Bogen durch die Zellmitten (zwei
Mündungen beiderseits einer Fläche: ℓ = Steg dazwischen). Abseits der
Mündungsränder bleibt der Faktor exakt 1; er ist auf 20 begrenzt, und
der Mündungskurzschluss wird mit dem größten Faktor mitskaliert.

**Exakte Referenz:** eine Äquipotentialscheibe exzentrisch in einer
Kreisscheibe mit festem Randdruck. Der Leitwert ist in bipolaren
Koordinaten geschlossen, G = 2πK / arcosh((a² + r² − e²)/(2ar)).
Gerechnet mit den Fußabdrücken und Faktoren des Modells selbst:

| Mündung | ohne Korrektur (60 × 96) | mit (60 × 96) | mit (120 × 192) |
|---|---|---|---|
| ⌀1,4 mm bei e = 6 mm | −6,0 % | −0,30 % | −0,06 % |
| ⌀2,0 mm bei e = 3 mm | −3,2 % | −0,08 % | −0,02 % |

Der Fehler viertelt sich jetzt mit halbierter Zellweite (zweite
Ordnung), vorher fiel er linear und unregelmäßig. Akustisch (Gegenprobe
50 b): das grobe Gitter liegt bei 12 × ⌀1,4 mm auf 1" 0,1 dB neben dem
Referenzgitter, ohne Korrektur 1,6 dB. **Grenze:** eine Mündung, die
kleiner als eine Zelle ist (die Mitte der einzigen Zelle liegt außerhalb
der Kontur), bleibt unkorrigiert — dafür das feine Gitter.

**Was sich dadurch verschoben hat** (jeweils gitterkonvergent, grob und
fein stimmen überein):

* **K67-Rückdämpfung** (1 kHz): automatische Verdrehung −16,5 dB statt
  −11 dB; global 6°/9°/12° −11/−22/−28 dB. Das 2D-Modell (−28 dB)
  entspricht jetzt der 12°-Lage (vorher 9°; seit Gegenprobe 54 einer
  Lage zwischen 9° und 12°). Der Befund bleibt: die Tiefe der
  Auslöschung hängt an der undokumentierten Kernlage.
* **FEM-Referenz (Gegenprobe 32):** die 3D-Resonanz liegt bei 495 Hz
  statt 482 Hz — 4 % über 2D (478 Hz), 10 % unter der FEM (550 Hz). Ohne
  Korrektur wanderte sie mit dem Gitter (482…488 Hz). Die diskreten
  Bohrungen erklären damit rund ein Viertel der Verstimmung; der Rest
  bleibt offen.
* **Gewebe am 3D-Außenknoten (Gegenprobe 27):** der Vergleich mit der
  1D/2D-Kette galt bei 100 Hz auf 0,04 dB. Seit Gegenprobe 52 zeigt
  sich das als Zufall (heute −5,3 gegen −6,0 dB); er wird nur noch
  ausgegeben, s. Gegenprobe 27.
* **B&K 4134:** 3D liegt bei 20 kHz jetzt 3,7 dB über der Messung (vorher
  3,1 dB), grob wie fein. Der Hochtonüberschuss des 3D-Lösers ist damit
  kein Gitterfehler mehr, sondern ein Modellbefund — und er zeigt sich
  jetzt auch bei **dichten** Lochbildern: bei der steifen ½"-Kapsel liegt
  3D oberhalb der Membranresonanz 1–1,4 dB über 2D, gleich ob 24, 48
  oder 96 Löcher. Die Treppenkontur hatte das bisher im Mittelband
  teilweise überdeckt. (Aufgeklärt in Gegenprobe 52: bis auf den
  unbelasteten Membranring kein Fehler des 3D-Lösers.)
* **Homogenisierungsgrenze (Gegenprobe 48):** nachgeprüft über
  gleichverteilte Lochbilder, gemessen gegen das dichte Raster gleicher
  Lochfläche (so fällt die gemeinsame Formanpassung heraus), mit
  konturtreuen Mündungen und gekoppeltem Membranring. Bei üblichen
  Spalten (20…38 µm) setzt die 1-dB-Abweichung bei Π ≥ 15 ein, für die
  weiche 1"-Kapsel (T ≈ 45 N/m) **und** die steife ½"-Kapsel
  (T = 400 N/m) — Π = 10 ist dort der vorsichtige Rand. Die frühere
  Streuung der steifen Kapsel (Π ≈ 3…9) war der unbelastete Membranring
  (Gegenprobe 52). Beim **weiten Spalt** und bei **Lochkreisen** war die
  Grenze zu optimistisch, zwei Fälle blieben ganz ohne Warnung — die
  Lücke ist geschlossen (Gegenprobe 53, s. u.).

### Hochtonüberschuss des 3D-Lösers aufgeklärt (Gegenprobe 52)

Oberhalb der Membranresonanz lag der 3D-Löser 1–2 dB über dem 2D-
Modell, auch bei dichten Lochbildern, und an der B&K 4134 2–4 dB über
der Messung. Zerlegt in drei Teile:

**1. Ein Fehler des 3D-Lösers — behoben.** Der Membranring außerhalb
der Backplate (a_bp < r < a_mem) war hinten **unbelastet**, als läge
Vakuum hinter ihm. Er liegt aber über dem tiefen Ringraum zwischen
Plattenrand und Einspannung, dessen Druck der Randdruck ist. Jetzt spürt
er diesen Druck und speist seinen Volumenfluss dort ein — mit Randspalt
über einen eigenen **Ringknoten** zwischen Filmrand und Schlitzleitung,
sonst in die äußerste Filmzelle. Das ist dieselbe Physik wie die
Randumgehung des 2D-Felds (Gegenprobe 37). Wirkung: bei geschlossenem
Rand bis 2 dB (B&K-Geometrie ohne Randschlitz), bei der dichten
1"-Kapsel 0,4 dB.

**Belegt mit einem unabhängigen Löser:** axialsymmetrisch (B&K-4134-
Geometrie nur mit Randschlitz), knotenzentrierte Differenzen statt
Zell-FV, eigene Assemblierung von Membranfeld, Reynolds-Film,
Ringknoten, Schlitzleitung, Rückkette und Frontknoten. 3D trifft ihn
über 20 Hz…20 kHz auf **0,001 dB** (ohne Ringkopplung 0,08 dB daneben),
die geschlossene Tieftonform V = f_in·C_m/(1 + C_m/C_b) auf 0,6 %.

**2. Kein Fehler: die Formanpassung der Membran.** Wo die Filmkraft
gegen die Membranspannung zählt, weicht die Membran dem Filmdruck aus.
Beim Randschlitz ist der Druck in der Mitte am größten; die Membran
arbeitet dann bevorzugt außen, bei 20 kHz bis zu einem Ringbuckel mit
dem 2,25-Fachen der Mittenauslenkung. 3D und unabhängiger Löser zeigen
das übereinstimmend. Das 2D-Modell hält die Grundmodenform fest und
liegt bei 20 kHz 1,8 dB tiefer. Mehrere 2D-Moden helfen nicht, weil sie
sich einen Spaltknoten teilen. Bei der dichten, weich gespannten 1"-
Kapsel (96 Löcher) ist das der ganze verbleibende Unterschied (1,2–1,5 dB
zwischen 5 und 16 kHz): bei erzwungener Grundmodenform, also sehr
steifer Membran, ist der Filmwiderstand von 3D und 2D **identisch**
(Verhältnis 1,000).

**3. Kein Fehler: Löcher auf einem Lochkreis.** Bei erzwungener Form
ist der Filmwiderstand der B&K 4134 im 3D nur das **0,59-Fache** des
2D-Werts (Stand-Wert; vor dem Massenfaktor 8/j₀₁² 0,57). Das 2D-Feld löst die radiale Zuströmung zum Lochring selbst
auf und addiert zusätzlich die volle Škvor-Zelle, die diese Konvergenz
nochmals enthält. *Nachtrag (Gegenprobe 60):* der Faktor enthielt
zusätzlich den verschieden angesetzten Folienverlust (sauber 0,65);
mit dem Makroelement liegt 3D/2D bei 1,04.

**Offen:** Die B&K-Messung (Zuckerwar 1978, Aktuatorverfahren) folgt dem
2D-Modell. Da der 3D-Löser die Modellgleichungen nachweislich richtig
rechnet, kamen zwei Erklärungen in Frage: die reale Kapsel dämpft
stärker als der Reynolds-Film (um etwa den Faktor 1,7 im Widerstand),
oder der Aktuator-Frequenzgang weicht im Hochton vom Druckfrequenzgang
ab.

**Aktuator eingegrenzt:** Der Aktuator treibt die Membran mit einem
gleichmäßigen elektrostatischen Druck. Die bewegte Membran erzeugt aber
unter der Platte einen kleinen Zusatzdruck, der bei steifen Membranen
kleiner ist (Frederiksen 2013, Int. J. Metrol. Qual. Eng. 4, 97–107,
Abschn. 11; Zahlen für die 4134 nennt der Artikel nicht). Im Modell ist
das eine Luftmasse vor der Membran statt der Abstrahlung. Die
Plattengeometrie liegt nicht vor; selbst mit großzügigen 10 mm
wirksamer Luftsäule hebt diese Last die 4134 bei 13–20 kHz nur um
höchstens 0,6 dB an (3D; 2D höchstens 0,3 dB). Die Aktuator-Antwort
läge damit sogar leicht **über** der Druckantwort. Das ist zu klein und
hat das falsche Vorzeichen, um die 2,2–3,5 dB zu erklären, um die 3D
über der Messung liegt (Gegenprobe 52 d). Es bleibt die stärkere
Dämpfung der realen Kapsel; ihre Ursache ist offen.

Das 2D-Modell bleibt Standard: es trifft die Messung, ist aber im
Hochton aus zwei ausgleichenden Näherungen zusammengesetzt.

### Warnlücke beim weiten Spalt und bei Lochkreisen geschlossen (Gegenprobe 53)

Die Homogenisierungsgrenze war beim weiten Spalt (65 µm) und bei
Lochkreisen zu optimistisch; zwei Fälle mit mehr als 1 dB blieben ganz
ohne Warnung. Neu vermessen gegen den 3D-Löser: 114 Fälle (42
gleichverteilt, 72 auf ein oder zwei Lochkreisen), zwei Kapseln (1" mit
T ≈ 45 N/m, ½" mit 109 N/m), Spalte 20/25/38/65 µm. Nach der
Korrektur der 3D-Wandlung (Gegenprobe 54) und mit dem Massenfaktor
8/j₀₁² (Gegenprobe 55) ist alles neu gerechnet; die Befunde bleiben.
Drei Befunde:

**1. Das Messmaß war schief.** Verglichen wurde die vorzeichenrichtige
Differenz zum dichten Raster gleicher Lochfläche. Beim weiten Spalt hat
das dichte Raster aber eine scharfe Resonanz, und dort liegen 2D und 3D
selbst ±1,5 dB auseinander. Diese Abweichung landete im Befund: bei der
1"-Kapsel mit 16 Löchern und 1,2 kHz weicht das spärliche Lochbild
selbst nur um 0,1 dB ab, das dichte um 1,3 dB. Gemessen wird jetzt die
**Mehrabweichung** |2D/3D| − |2D/3D dicht|, also was das Ausdünnen
verschlechtert.

**2. Weiter Spalt: Trägheit und Beulresonanz.** Die Kennzahl Π kannte
nur die viskose Filmkraft. Beim weiten Spalt wird die Spaltluft träge,
und die Membranfläche zwischen den Löchern hat eine eigene Resonanz
f_ρ = f_res·a_mem/ρ, bei der ihre Steifigkeit verschwindet. Mit der
vollen Filmleitfähigkeit K(ω) und der dynamischen Steifigkeit der Beule
liegt die Grenze der ½"-Kapsel mit 65 µm und 16 Löchern bei 17,6 statt
107 kHz; 2D weicht dort ab etwa 19 kHz um mehr als 1 dB von 3D ab. Die
Schwelle Π = 10
bleibt, im Tiefton ändert sich nichts. Alle 42 gleichverteilten Fälle
werden jetzt vor dem 1-dB-Einsatz gewarnt.

**3. Lochkreise: die Darstellung selbst.** Bei erzwungener Membranform
stimmt der Filmwiderstand von 2D und 3D für alle Lochkreise auf 3 %
überein; die frühe Abweichung ist Formanpassung an das radial stark
gegliederte Druckfeld. Sie hängt an der Lochzahl je Kreis: Kreise mit
vielen Löchern wirken als Liniensenke, und das 2D-Feld verschmiert sie
zu einem Band von 0,1·a_bp Breite. Diese Breite hat keinen eindeutigen
Wert. Rechnet man den Kreis als schmalste darstellbare Liniensenke,
ändert sich das Ergebnis um bis zu 7 dB (Kreis nahe der Mitte, 48
Löcher). Eine physikalisch korrekte Lochreihen-Darstellung (Liniensenke
plus Konvergenzwiderstand ln(s/(2π·r))/(2πK) je Loch) trifft 3D nicht
besser. Die Warnung prüft deshalb die **Selbstkonsistenz**: sie gilt ab
der Frequenz, bei der Band und Liniensenke um mehr als 0,5 dB
auseinanderliegen (halbe Toleranz, weil die 3D-Abweichung bis zum
Doppelten dieser Spanne erreichte). Damit kam die Warnung in allen 72
Lochkreis-Fällen vor dem 1-dB-Einsatz; im knappsten Fall (½"-Kapsel,
65 µm, 48 Löcher auf einem Kreis) beträgt die Mehrabweichung an der
Warnfrequenz 2,23 kHz 0,95 dB; 1 dB erreicht sie erst bei 2,26 kHz
(direkt nachgerechnet; vor dem Massenfaktor 8/j₀₁², Gegenprobe 55,
waren es 2,20 kHz und 0,91 dB). Der Abstand ist knapp. Sonst ist die
Prüfung vorsichtig: bei zwei Lochkreisen warnt sie bis zu 40-fach zu
früh.

Die Gegenprobe hält die tragenden Stichproben fest: das Messmaß
(+0,10 dB spärlich gegen +1,26 dB dicht), die weite Spalte (Grenze
17,6 kHz, eigene Abweichung dort 0,67 dB, bei 20 kHz 1,12 dB), den
Lochkreis mit 48 Löchern (Grenze 139 Hz statt 390 Hz, Mehrabweichung
0,71 gegen 2,27 dB, Filmwiderstand 3D/2D 0,987) und die Spanne der
Darstellungen (7,3 dB). Bei der weiten Spalte prüft sie die eigene
Abweichung: das dichte Raster kreuzt dort die Null, die Mehrabweichung
bleibt bei 20 kHz mit 0,93 dB knapp unter 1 dB. Die Prüfung kostet
beim Aufbau einer Kapsel mit Lochkreisen höchstens 0,12 s.

**Offen:** Die physikalische Verbesserung wäre eine Lochkreis-
Darstellung im 2D-Feld, die die Formanpassung an das radiale Druckfeld
mitnimmt. Das würde validierte Ergebnisse (B&K 4134, Debenham) ändern
und ist deshalb nicht Teil dieser Änderung.

**Nachtrag (Gegenprobe 60):** Mit dem Makroelement rechnet das 2D-Feld
die Lochkreise exakt; die frühe Abweichung von 3D ist seitdem allein die
Formanpassung der Membran. Neu vermessen an 80 Lochkreis-Fällen (beide
Kapseln, 20/25/38/65 µm, ein und zwei Kreise): die Warnung kommt
weiterhin in allen vor dem 1-dB-Einsatz. Im knappsten Fall (½"-Kapsel,
65 µm, 48 Löcher) warnt sie ab 2,23 kHz, die Mehrabweichung erreicht
1 dB erst zwischen 2,6 und 2,9 kHz. Ohne die Lochkreis-Prüfung blieben
acht Fälle (½"-Kapsel, 65 µm) ungewarnt; sie bleibt also. Beim alten
Stichprobenfall (25 µm) reicht inzwischen die lokale Grenze, die
Gegenprobe prüft deshalb den 65-µm-Fall: Warnung 2225 Hz, lokale Grenze
4630 Hz, dazwischen steigt die Mehrabweichung über 1 dB (1,36 dB).

### Statischer Versatz 2D/3D aufgeklärt (Gegenprobe 54)

An der B&K 4134 lagen 2D und 3D schon im Tiefton 1,38 dB auseinander,
über alle Frequenzen gleich. Das waren zwei Fehler im 3D-Pfad, keine
Physik der Kapsel:

**1. Spannungsbildung (1,08 dB).** Die Kette rechnet e = Θ·V. V ist die
Volumenverschiebung ihrer Grundmode über der **ganzen** Membran; das
Elektrodenintegral (Spaltprofil, Porosität, Elektrodenrand) steckt in Θ.
Der 3D-Löser setzte stattdessen die Volumenverschiebung **über der
Elektrode** ein. Bei kleinerer Elektrode (a_bp < a_mem) war er damit um
den Faktor u·(2 − u), u = (a_bp/a_mem)², zu leise: an der B&K
(a_bp/a_mem = 0,81) genau 1,08 dB, bei der Standardkapsel 0,27 dB. Jetzt
wandelt der 3D-Löser sein eigenes Auslenkungsfeld mit demselben
Elektrodenintegral wie Θ in Spannung; für die Grundmodenform ist das
exakt die Kette.

**2. Membranspannung (0,31 dB).** Bei vorgegebener Vorspannung (B&K:
3162 N/m) rechnete der 3D-Löser die Spannung aus der Resonanz der Kette
zurück. Die liegt mit dem Kolbenfaktor 4/3 der statischen Form 1,9 % zu
hoch (23,4 statt 23,0 kHz), die Spannung damit 3,75 %. Jetzt nimmt der
3D-Löser die physikalische Spannung, über die exakte Modalfrequenz
(Vorspannung plus Biegeanteil der Folie). Seit Gegenprobe 62 ist das
die Vorspannung selbst: die Biegung steckt in der Randschicht, in Kette
und Feld gleich.

Ergebnis an der B&K: **−0,003 dB** im Tiefton. Im Vergleich mit der
Messung (auf 250 Hz normiert) fällt der konstante Faktor heraus; die
etwas tiefere Resonanz senkt die Abweichung bei 13/16/20 kHz auf
2,2/2,6/3,5 dB (vorher 2,2/2,7/3,8 dB).

**Rest bei vorgegebener Resonanz.** Ist statt der Vorspannung die
Resonanz vorgegeben, blieb ein kleiner Versatz (½"-Kapsel 0,12 dB,
Standardkapsel 0,19 dB). Das war keine 3D-Frage, sondern die
Ein-Moden-Kalibrierung der Kette (Kolbenfaktor 4/3). Behoben mit dem
Massenfaktor 8/j₀₁², s. Gegenprobe 55.

**3. Dabei aufgedeckt: Θ der Ringmembran (2D, +1,9…2,0 dB).** Die
Kette rechnete auch mit Mittelpfosten dw = 2V/S, also mit der
mittleren Auslenkung der Parabel (1/2). Die Ringform hat m₁ = 0,62…0,63
(1–3 mm Pfosten auf 25 mm); die 2D-Spannung lag um 2·m₁ zu hoch. Jetzt
dw = V/S_eff mit der schon vorhandenen wirksamen Fläche S_eff = S·m₁.
Der alte 3D-Pfad nutzte denselben Koeffizienten und verdeckte den
Fehler. Probe: bei 1 V und 20 Hz ändert ein 3-mm-Pfosten das Verhältnis
3D/2D nicht (0,997 → 0,998); mit dem alten Koeffizienten wären es 0,80.

**Folgen für andere Gegenproben.** Wo die Membran bei hoher Vorspannung
stark durchgebogen ist, gewichtet die physikalisch richtige Wandlung die
Mitte (∝ 1/g²). Die Formanpassung wird dort sichtbar, die die reine
Volumenverschiebung verdeckte. Drei Prüfungen sind deshalb ehrlich
umformuliert, keine Schwelle ist angepasst:
- **K67 (Gegenprobe 22):** Die 2D-Auslöschung von −28 dB liegt jetzt
  zwischen den Kernlagen 9° und 12° statt genau bei 12°.
- **K103 (Gegenprobe 23):** Die Struktur wird an der Volumenverschiebung
  geprüft. Bei 1 kHz liegt die 3D-Spannung wegen 26 % Durchbiegung 9 %
  unter der Kette.
- **Ringmembran-Prüfling (Gegenprobe 45):** 60 V, 32 % Durchbiegung,
  weit über f_hom. Die Mechanik wird an der Volumenverschiebung geprüft,
  die Spannung ausgegeben.

Außerdem gilt in Gegenprobe 31 für 12 Bohrungen die 1-dB-Toleranz der
Warnung statt einer frei gewählten 2-dB-Grenze.

Die Gegenprobe prüft das Ausgangsgewicht an der Grundmode (Einzel- und
Doppel-Backplate, mit Arbeitspunkt, Ringmembran), die Zerlegung an der
B&K (1,38 = 1,08 aus u·(2 − u) + 0,31 → −0,003 dB), die ½"-Kapsel einmal
über die Resonanz, einmal über die Vorspannung vorgegeben, und Θ der
Ringmembran gegen das 3D-Feld.

### Massenfaktor 8/j₀₁² (Gegenprobe 55)

Ein Freiheitsgrad trifft nur zwei Größen exakt. Die Kette nimmt die
statische Nachgiebigkeit der Membran (exakt) und wählte die Masse
bisher mit dem Rayleigh-Wert der statischen Form, ⟨φ²⟩/⟨φ⟩² = 4/3 für
die Parabel. Der Rayleigh-Wert ist eine obere Schranke der Frequenz:
bei vorgegebener Vorspannung lag die Resonanz 1,9 % zu hoch, bei
vorgegebener Resonanz war die Membran statisch 3,75 % zu nachgiebig
(+0,32 dB).

Jetzt ist der Massenfaktor **μ = 8/(z₁²·g)**, ohne Pfosten
8/j₀₁² = 1,383, mit Pfosten mit dem Ring-Eigenwert z₁ und dem
Ringfaktor g der Nachgiebigkeit (1 mm Pfosten auf 22 mm: 1,277 statt
1,247). Statik und Grundresonanz sind damit beide exakt; „Resonanz
vorgeben" und „Vorspannung vorgeben" beschreiben dieselbe Membran.
Die Modenmasse der J₀-Form, j₀₁²/4 = 1,446, wäre ebenso falsch: sie
gehört zur Modennachgiebigkeit, die nur 95,7 % der statischen ist.

**Unabhängige Bestätigung:** Die normierte Mehrmoden-Kette
(`membrane_modes` > 1) läuft für viele Moden genau dann auf den freien
Kolben, wenn μ·g·z₁²/8 = 1 ist. Mit 8/(z₁²·g) trägt dann jeder Zweig
exakt seine Modenmasse und -nachgiebigkeit; mit 4/3 lief die Reihe auf
0,964·σ/S, also 0,32 dB über den Kolben hinaus. Bei der Ringmembran
ist der Grenzwert die Ringfläche S·(1 − ρ²).

**Wirkung:**

| | vorher (4/3) | jetzt (8/j₀₁²) |
|---|---|---|
| ½"-Kapsel über f_res, 2D gegen 3D (20 Hz) | +0,12 dB | 0,00 dB |
| 3D/Kette bei 100 Hz, einfach / Gegentakt (Gegenprobe 23) | 0,974 / 0,961 | 1,000 / 0,998 |
| lochfreier Kolben-Grenzfall 3D/2D (Gegenprobe 29) | 0,31 dB | 0,01 dB |
| Standardkapsel (8 kHz vorgegeben), 1 kHz | 69,8 mV/Pa | 67,9 mV/Pa (−0,25 dB) |
| K67 (1150 Hz vorgegeben): Empfindlichkeit, Pull-in | 20,3 mV/Pa, 72,6 V | 20,1 mV/Pa, 73,9 V |
| Debenham (2100 Hz vorgegeben): Empfindlichkeit, Pull-in | 9,93 mV/Pa, 52,0 V | 9,77 mV/Pa, 53,0 V |
| B&K 4134 gegen Zuckerwar Fig. 6 (2D) | 0,27 dB / 0,94° RMS | 0,30 dB / 0,98° RMS |
| B&K 4146 gegen Zuckerwar Fig. 7 (2D) | 1,09 dB / 7,5° RMS | 1,09 dB / 7,2° RMS |
| FEM (Šimonová/Honzík): Resonanz, Überhöhung | 477 Hz, +6,41 dB | 478 Hz, +6,44 dB (FEM 550 Hz, +6,74 dB) |
| Grinnip VC 0/90/180° (RMS) | 1,04/2,48/1,79 dB | 1,03/2,40/1,71 dB |

Bei vorgegebener Resonanz wird die Membran 3,6 % steifer. Der Tiefton
sinkt dadurch höchstens um 0,32 dB, weniger, wo die geringere
Feder-Erweichung einen Teil zurückgibt (K67 −0,07 dB bei 26,5 statt
28,2 % Erweichung). Bei vorgegebener Vorspannung ändert sich der
Tiefton nicht, nur die Resonanz liegt 1,8 % tiefer. Die K67-Niere
bleibt bei −28,1 dB (180°, 1 kHz). Der B&K 4134 rückt gegen die
Messung um 0,03 dB RMS ab, innerhalb der Ableseunsicherheit der Kurve
(±0,15 dB); Zuckerwar selbst rechnet in Tabelle II mit 4/3.

Die Gegenprobe prüft die Resonanz bei vorgegebener Vorspannung
(Vollkreis und Ringmembran auf 4·10⁻⁵, B&K-Nickelfolie 1·10⁻⁴ über die
Biegesteifigkeit), die Gleichheit beider Vorgaben (Rest genau der
Biegeanteil der Folie, 3,6·10⁻⁴), die ½"-Kapsel gegen das 3D-Feld mit
altem und neuem Faktor und den Hochtongrenzwert der Modenreihe über
400 Moden für Vollkreis und Ring (ρ = 0,1). Der alte Wert bleibt über
den Klassenschalter `_MASS_EXACT` für Vergleiche erreichbar;
Gegenprobe 54 stellt damit ihre historische Zerlegung nach.
NACHTRAG (Gegenprobe 62): z₁ und g sind jetzt die der wirksamen
Membran (Randschicht); die Kette trifft deren Eigenwert exakt, gegen
die Platte unter Zug bleibt die Plattendispersion. „f_res vorgeben"
und „Spannung vorgeben" sind seither ohne Rest dieselbe Kapsel.

### Wiederverwendung der 3D-Lösung (Gegenprobe 56)

Richtdiagramm, Übertragungsfunktion, Laufzeit-Diagnose und `summary()`
lösen bei gleicher Frequenz dasselbe 3D-System. Bisher wurde es jedes
Mal neu aufgestellt und LU-faktorisiert, bei der K67 im 3D-Modus rund
17 s je Lösung. Jetzt bewahrt die Kapsel ihre Lösungen je Frequenz auf.
Gespeichert sind nur die Ausgaben (für beide Gewichte, `output` und
`volume`, mit den Rückmembran-Antworten und der Reziprozitäts-
Diagnose), nicht die Feldvektoren; das sind etwa 1 kB je Frequenz.

Der Speicher hängt am 3D-Gitter und ist nach einem Neubau des Gitters
leer. Außerdem wird er verworfen, sobald sich etwas ändert, das die
Lösung nach dem Bau der Kapsel noch beeinflussen kann: Schalter und
Methoden der Klasse (auch ausgetauschte, wie in den Gegenproben 27 und
52), auf der Kapsel ersetzte Methoden und die Stoffwerte der Luft. Die
übrigen Parameter liegen mit dem Bau fest; nachträglich geändert würden
auch ihre abgeleiteten Größen nicht nachgeführt.

Gegenprobe 56 prüft:
- dieselbe Aufruffolge mit und ohne Speicher ergibt bitgleiche
  Ergebnisse (Pattern, Übertragung, Rückmembran, Laufzeit);
- jede Frequenz wird genau einmal faktorisiert (6 → 2), eine neue
  Frequenz in einem gemischten Aufruf genau einmal mehr;
- nach dem Tausch einer Klassen- oder Instanzmethode wird neu gelöst,
  bitgleich mit einer frisch gebauten Kapsel, und nach dem Zurücktausch
  ergibt sich wieder das Original;
- nach einem neuen Gitter wird neu gelöst, bitgleich mit einer frisch
  gebauten Kapsel dieses Gitters;
- die Reziprozitäts-Diagnose der Einzel-Backplate kommt auch aus dem
  Speicher.

**Wirkung:** In der GUI (Debenham-Beispiel im 3D-Modus, 40 Punkte, 5
Richtfrequenzen) braucht die erste Berechnung 43 statt 45 Lösungen.
`summary()` kostet in beiden Sprachen nichts mehr (vorher je eine
Lösung), und eine zusätzliche Richtfrequenz braucht 1 statt 46 Lösungen
(1 statt 38 s). Im Selbsttest sinkt die Summe der Testzeiten um etwa
5 %; die Wandzeit bleibt bei rund 3 Minuten, weil die Gegenproben 23
und 48 fast nur verschiedene Systeme lösen (48: 20 von 20).

### LU-Zerlegung des 3D-Lösers (Gegenprobe 57)

Die Doppel-Backplate brauchte im 3D-Löser 137 s je Frequenz, die
Einzel-Backplate gleicher Größe 2 s, bei nur 1,5-mal so vielen
Unbekannten. Die Ursache war nicht die Struktur, sondern die
Pivotisierung. SuperLU ordnete die Spalten füllungsarm (COLAMD) und
tauschte dann Zeilen nach Betrag; die Tausche zerstörten die Ordnung.
L+U hatte bei der Doppel-Backplate 101 Mio. Einträge (114-fach).
Nachgewiesen an derselben Matrix:

| Zerlegung (Doppel-Backplate) | Zeit | Einträge L+U |
|---|---|---|
| COLAMD, Zeilentausch (bisher) | 137 s | 101 Mio. |
| COLAMD, Schwelle 0,1 | 97 s | 94 Mio. |
| COLAMD, Diagonal-Pivots | 8,9 s | 23 Mio. |
| symmetrische Ordnung (MMD auf A+Aᵀ), Diagonal-Pivots | 1,4 s | 8,6 Mio. |

Das System ist strukturell symmetrisch: Film, Membran, Löcher und
Knoten koppeln wechselseitig. Der Löser zerlegt es jetzt mit
symmetrischer Ordnung und Diagonal-Pivots. Diagonal-Pivots sind nur so
stabil wie die Diagonale. Deshalb wird jede Lösung am
komponentenweisen Rückwärtsfehler max|S·x − b| / (|S|·|x| + |b|)
geprüft. Liegt er über 10⁻¹¹ oder scheitert die Zerlegung, rechnet
der Löser wie bisher mit Pivotisierung.

Die symmetrische Zerlegung ist dabei nicht nur schneller, sondern
auch genauer. Ihr Rückwärtsfehler liegt bei 10⁻¹⁵ bis 10⁻¹⁴, der
der pivotisierenden bei 10⁻¹² bis 5·10⁻⁸. Im ganzen Selbsttest (374
Zerlegungen) gab es keinen Rückfall; der größte Rückwärtsfehler war
6,9·10⁻¹⁴. Die Ausgänge ändern sich um höchstens 3·10⁻⁸ relativ. Im
Protokoll sichtbar ist das nur an einer Stelle: Der Konvergenz-
exponent des Grenzfalls Z → 0 (Gegenprobe 27) liegt jetzt bei 0,091,
am theoretischen Wert 0,0909. Vorher hing er vom Rundungsrauschen und
damit von der Thread-Zahl ab (0,090 bzw. 0,093).

Zeit je Frequenz auf einem Kern, vorher → jetzt: Einzel-Backplate
2 → 0,5 s, Doppel-Backplate 137 → 1,4 s, K67 17 → 1,5 s (verdreht)
bzw. 52 → 1,9 s (ausgerichtet), Debenham 0,6 → 0,3 s, B&K 4134
0,4 → 0,1 s, feines Gitter (A48) 25 → 1,7 s. In der GUI (40 Punkte,
5 Richtfrequenzen) rechnet das Debenham-Beispiel im 3D-Modus 13 statt
37 s und das K67-Beispiel 82 s statt hochgerechnet rund 13 Minuten.
Der Selbsttest läuft auf 4 Kernen in etwa 1 statt knapp 3 Minuten.

Die Gegenprobe prüft den Rückwärtsfehler an allen Bauformen (Einzel-
und Doppel-Backplate, einteilige Doppelmembran, K67 mit Zwischenspalt,
B&K mit Randspalt) bei 20 Hz, 1 kHz und 20 kHz; die Übereinstimmung
mit der pivotisierenden Zerlegung; die Auffüllung der Doppel-Backplate
(22 %); den erzwungenen Rückfall (bitgleich) und eine Matrix mit
winziger Diagonale, an der die Prüfung die instabile Zerlegung
erkennen muss.

### Dämpfung am Lochkreis: Recherche (Gegenprobe 58)

Ausgangspunkt ist der offene Punkt aus Gegenprobe 52: an der B&K 4134
liegt der 3D-Löser bei 13–20 kHz 2,2–3,5 dB über Zuckerwars Messung,
das 2D-Modell trifft sie nur, weil es den Filmwiderstand am Lochkreis
1,7-fach überschätzt. Gefragt war: welche Physik dämpft die reale
Kapsel zusätzlich?

**Was fehlt, ist Widerstand.** Zuckerwars Fig. 6 enthält auch die
Phase. Schon weit unter der Resonanz (2–10 kHz, Resonanz ~23 kHz), wo
Masse und Steife die Phase nicht bewegen, eilt das 3D-Modell nur zu
rund 70 % so stark nach wie die Messung (`gp58.phase_3d_anteil_4134`).
Eine Resonanzverschiebung sähe anders aus. Ein einziger
frequenzunabhängiger Serienwiderstand vor der Membran, 5000 Rayl bzw.
8,1·10⁷ Pa·s/m³ (rund zwei Drittel des exakten Filmwiderstands), bringt
Amplitude **und** Phase zugleich zur Messung: 0,13 dB / 1,3° RMS statt
1,67 dB / 7,6° (2D: 0,30 dB / 0,98°). 3D-Film (etwa 0,59 × 2D, also
rund 1,2·10⁸) plus Zusatz ergibt etwa 2·10⁸ Pa·s/m³, die Größe von
Zuckerwars Tabellenwert R.

**Aber nicht am 4146.** Eine echte physikalische Ursache müsste auch die
zweite Kapsel derselben Arbeit treffen. Dort ist der unveränderte
3D-Löser schon besser als 2D (0,75 gegen 1,09 dB RMS), und jede
zusätzliche Dämpfung verschlechtert ihn (+1000 Rayl: 1,38 dB). Dasselbe
gilt für die zweite Deutung, die am 4134 passt: ein fast dichter
Randschlitz (wirksam 15 µm statt 0,838 mm; 4134 0,28 dB, 4146
0,75 → 4,1 dB).

**Keine fehlende Filmphysik.** Gegen die volle thermoviskose FEM der
Gegenprobe 32 (Navier–Stokes statt Reynolds, vier Löcher auf einem
Kreis) liegt die 3D-Überhöhung nur 0,23 dB über der FEM, rund 3 % zu
wenig Dämpfung. *(Nachtrag Gegenprobe 67: dieser Vergleich litt am
versehentlichen Laufzeitglied im Prüfaufbau; ohne es und mit der
Spalt-Mündung sind es +0,44 dB.)* Beide Feldmodelle führen im Film ohnehin die volle
Dünnschicht-Physik (viskose Trägheit, polytrope Kompressibilität mit
thermischer Relaxation). Nur abgeschätzt, nicht eigens gerechnet:
Gasverdünnung wirkt in die falsche Richtung (Knudsen-Zahl 0,003,
Schlupf senkt die Dämpfung); Eintrittsverluste an Löchern und Schlitz
sind bei Mündungsradien vom 25-Fachen des Spalts klein. *(Seit
Gegenprobe 67 für die Löcher gerechnet: die Spalt-Mündung erhöht den
Filmwiderstand der 4134 um rund 5 %. Die Randschlitz-Mündung ist eine
andere Geometrie und nicht gerechnet.)*

**Der Aktuator scheidet aus** (belegt): nach B&Ks Microphone Handbook
(BE 1447, Abschn. 2.7) steht die perforierte Aktuatorplatte 0,4–0,8 mm
vor der Membran, und Aktuator- und Druckfrequenzgang unterscheiden sich
um 0,1 bis etwa 1 dB; für ½″-Kapseln sind keine Korrekturen nötig. Eine
Luftschicht lieferte 8·10⁷ Pa·s/m³ erst bei etwa 47 µm Abstand
(randbelüftet, 3μ/(2πd³)), und die Massenlast wirkt mit falschem
Vorzeichen (Gegenprobe 52).

**Der Spalt erklärt die Hälfte.** Das COMSOL-Anwendungsmodell der 4134
rechnet mit B&Ks Originalgeometrie und einem Spalt von „around 19 µm"
(Zuckerwars Tabelle I: 20,77 µm; B&K fertigt typisch 20 µm ± 0,5 µm) und
trifft damit gemessene Kurven. Mit 19 µm halbiert sich die
Amplitudenabweichung des 3D-Modells (1,67 → 0,77 dB RMS), und das
Phasendefizit bis 5 kHz verschwindet weitgehend; zur Resonanz hin
bleibt es (6,3° RMS gegen 1,3° mit Serienwiderstand). Der Rest muss
aus der übrigen Geometrie kommen, die COMSOL von B&K hat und Tabelle I
nicht wiedergibt.

**Geometrie-Deutungen verworfen:** eine geschlossene Ringnut statt des
Schlitzes (10,2° Phase); die Lesart des Lochkreises als Durchmesser
(1,19 dB / 4,9°). Nebenbefund: die Dämpfung hängt stark an der Lage
des Lochkreises; bei etwa 1,5 mm Radius ist sie am kleinsten, weil sich
Loch- und Randentlüftung dort die Wege teilen.

**Zwischenfolgerung:** dem 3D-Film fehlt keine allgemeine Physik; die
Abweichung am 4134 ist kapselspezifisch. Der COMSOL-Vergleich unten
bestätigt das und benennt die Ursache.

**Zuckerwars Schlitz.** In Zuckerwars Theorie (ausführlich im
NASA-Bericht zum Hochtemperatur-Wandler, NTRS 19770013461, Kap. II und
Anhang A) ist der Randschlitz eine Öffnung *in* der Gegenelektrode, über
deren Mündung — wie über jedem Loch — eine gleichförmige
Ausströmgeschwindigkeit angesetzt wird (Petritskayas Randbedingung);
der Film läuft über die Mündung weiter. Er beschränkt das selbst auf
Öffnungen mit kleiner radialer Ausdehnung; sein Prototyp hatte einen
schmalen Schlitz (95 µm breit, 0,66 mm tief). Der 4134-Schlitz aus
Tabelle I füllt dagegen den ganzen Ring zwischen Plattenrand und
Einspannung (0,838 mm breit, 19 % des Radius). Für eine so breite
Mündung dürfte die Randbedingung Strömung durch den Film über der
Mündung erzwingen, also mehr Widerstand ergeben als eine offene
Ringmündung mit gleichförmigem Druck, wie Capsim sie rechnet (Überlegung,
nicht nachgerechnet). Dass Zuckerwars Tabellenwert R die
Messung trifft, ließ einen engeren Randweg in der realen 4134 vermuten.
**B&Ks Originalgeometrie widerlegt das** (s. u.). Ebenfalls in Zuckerwars
Bericht: die Kolbenform der Membran unterschätzt die Dämpfung gegenüber
Parabel- und Besselform (vgl. „Offene Punkte" 3).

**B&Ks Originalgeometrie.** Das COMSOL-Anwendungsmodell rechnet mit der
Geometrie von B&K; aus ihr (als Text exportiert, 1/12-Sektor) abgelesen:
Spalt 18,6 µm; sechs Löcher mit r = 0,5 mm auf dem Kreis r = 1,70 mm
(Tabelle I: 2,03 mm); Platte r = 3,6 mm, am Rand 0,3 mm dick, Unterseite
kegelig (an der Lochmitte 1,03 mm); Membran r = 4,5 mm; der Ring zwischen
Plattenrand und Einspannung ist 0,9 mm breit **offen** zur Rückkammer,
deren Außenwand ein Kegel von (4,5 mm; 0) nach (3,75 mm; −2,95 mm) ist
(131 mm³, mit Löchern 136 mm³; Tabelle I: 126 mm³); dazu die
Druckausgleichs-Öffnung (11 µm breit, 1,5 mm lang, für das Hörband
bedeutungslos). Zuckerwars „slit" (0,838 × 0,3 mm) ist also dieser
offene Ring mit der 0,3-mm-Plattenkante — einen gedrosselten Randweg
gibt es nicht. Mit dieser Geometrie bleibt das 3D-Modell gegen Fig. 6
gleich unterdämpft (1,73 dB / 5,8° RMS, +3,5 dB bei 20 kHz;
`gp58.rms_4134_bk_geometrie_db`): der kleinere Spalt dämpft mehr (allein
0,58 dB), der weiter innen liegende Lochkreis weniger — die Dämpfung ist
bei r ≈ 1,5 mm am kleinsten. Mit 200 V Vorspannung (wie im COMSOL-
Modell; Membran in der Mitte 0,68 µm durchgebogen) liegt 3D bei 1,29 dB /
3,7°, das 2D-Modell wird überdämpft (0,83 dB). Das 2D-Modell trifft die
Messung mit beiden Geometrien bei 28 V (0,30 dB).

**Homentcovschi & Miles (JASA 130, 3698, 2011)** lösen den Luftraum der
4134 — Spalt, sechs Löcher, Randschlitz (0,3 mm lang, in ihrer Fig. 1
als schräger Ringkanal vom Plattenrand in die Rückkammer), Rückkammer —
mit den linearisierten Stokes-Gleichungen, isotherm (c_T = 290 m/s),
mit Zuckerwars Geometrie und μ = 1,89·10⁻⁵. Sie schreiben selbst, ihr
Ergebnis sei gegen die Messung überdämpft, ohne Begründung. Aus ihrer
Fig. 3 abgelesen: Amplitude −2,3 dB bei 10 kHz, −7,5 dB bei 15 kHz,
unter −20 dB bei 19,5 kHz (Messung +0,6 / 0 / etwa −1 bis −3 dB). Die
Phase dagegen liegt bis 10 kHz fast auf dem 3D-Modell von Capsim (28°
gegen 26° bei 10 kHz; Messung 32–38°) — beide Rechnungen eilen dort
weniger nach als die Messung. Amplitude und Phase ihrer Kurve passen
nicht zu *einem* gedämpften Resonator; ihre Amplitude taugt deshalb
nicht als Referenz, ihre Phase stützt den Phasenbefund oben. Damit
treffen zwei unabhängige Rechnungen der Zuckerwar-Geometrie die Messung
nicht, in entgegengesetzter Richtung der Amplitude — der
Beschreibung des Randwegs kommt das größte Gewicht zu.

**Gelesene Quellen:** Homentcovschi & Miles, JASA 130, 3698 (2011);
COMSOL Application Library, „The Brüel & Kjær 4134 Condenser Microphone"
(Modelldokumentation 6.4); B&K Microphone Handbook Vol. 1 (BE 1447);
B&K Technical Review 1959-1; Zuckerwar, NASA-Bericht NTRS 19770013461;
Honzík et al., JASA 134, 3573 (2013) (Randkavität ohne Löcher, für die
4134 nur mittelbar). Entscheidend bleibt die reale Geometrie von
Plattenrand und Schlitz der 4134; B&Ks Geometrie steckt im COMSOL-Modell
(`bk_4134_microphone.mphbin`, nur mit COMSOL zu öffnen).

Die Gegenprobe prüft die Richtungen gegen Zuckerwars Messungen:
Phasendefizit unter der Resonanz, Verbesserung des 4134 durch
Serienwiderstand bzw. gedrosselten Schlitz, Verschlechterung des 4146
durch beides, und dass Spalt und Ringnut die Phase schlechter treffen
als der Serienwiderstand. Die Beträge sind Stand-Werte.

**Der COMSOL-Vergleich entscheidet (Gegenprobe 59).** Mit dem
gerechneten Ergebnis des COMSOL-Anwendungsmodells (volle thermoviskose
FEM auf B&Ks Originalgeometrie, 200 V) und den drei B&K-Messkurven
desselben Modells (heutige 4134) steht erstmals eine Referenz zur
Verfügung, die Geometrie und Physik zugleich festlegt. Capsim rechnet
dieselbe Konfiguration (Maße s. o., 200 V, Rückseite geschlossen; FEM
mit abgeschirmter Belüftung):

| auf 251 Hz bezogen | 10 kHz | 14,1 kHz | 17,8 kHz | 20 kHz | RMS 1–20 kHz gegen FEM | gegen Messmittel |
|---|---|---|---|---|---|---|
| COMSOL-FEM | +0,79 | +1,04 | +0,52 | −0,25 | — | 0,32 dB |
| **Capsim 3D** | +0,93 | +1,14 | +0,41 | −0,51 | **0,10 dB** | **0,31 dB** |
| Capsim 2D, Gaußband (bis Gegenprobe 59) | −0,78 | −1,96 | −3,47 | −4,48 | 1,80 dB | 1,51 dB |
| **Capsim 2D, Makroelement (Gegenprobe 60)** | +1,12 | +1,54 | +0,94 | +0,03 | **0,26 dB** | **0,54 dB** |
| B&K-Messungen (Mittel) | +0,74 | +0,57 | −0,26 | −1,20 | | |
| Zuckerwar 1978 (Fig. 6) | +0,14 | −0,71¹ | −1,11² | −3,06 | | |

¹ 13 kHz, ² 16 kHz. Das 3D-Ergebnis ist gitterkonvergent (grob/fein
höchstens 0,02 dB auseinander). Bis 12,6 kHz liegt es im Streuband der
drei Messkurven; darüber liegen FEM und 3D gleichermaßen etwas über der
Messung. Der äquivalente akustische Widerstand Re(p/Q) ohne
Strahlungslast (die FEM hat keine) liegt im 3D-Modell 13 % über der
FEM, im 2D-Modell mit Gaußband 61 % darüber, mit dem Makroelement 6 %
darunter (bei 2–5 kHz FEM 1,43·10⁸, 3D 1,63·10⁸, 2D 2,34·10⁸ bzw.
1,37·10⁸ Pa·s/m³).

**Folgerung:** Der 3D-Löser ist für den Lochkreis richtig, das 2D-Modell
überschätzt dort den Filmwiderstand um rund 60 %. Zuckerwars Prüfling
von 1978 war deutlich stärker gedämpft als heutige 4134; dass das
2D-Modell ihn trifft, ist das Zusammentreffen beider Abweichungen. Die
Aufgabe ist damit eine bessere Lochkreis-Darstellung im 2D-Feld, mit
Gegenprobe 59 als Anker — erledigt mit dem Makroelement (Gegenprobe 60,
nächster Abschnitt).

**Referenzdaten lokal:** Die COMSOL-Daten stehen unter COMSOLs Lizenz
und liegen nicht im Repo. Gegenprobe 59 liest sie aus `tests/extern/`
(von git ignoriert) und überspringt sich ohne sie. Erzeugen im
COMSOL-Anwendungsmodell `bk_4134_microphone` (Acoustics Module,
Electroacoustic Transducers): unter *Results* die Empfindlichkeit
(Modell mit offener Belüftung und die drei Messkurven), die
Empfindlichkeit mit abgeschirmter Belüftung („vent unexposed", die
Referenz — Capsim rechnet die Rückseite geschlossen) und „Equivalent
Acoustic Resistance" über *Add Plot Data to Export* als Text
exportieren und als `tests/extern/comsol_4134_sens.txt`,
`comsol_4134_unexposed.txt` bzw. `comsol_4134_resis.txt` ablegen. Ab
1 kHz unterscheiden sich offene und abgeschirmte Belüftung um höchstens
0,0013 dB (die Gegenprobe prüft das).

### Lochkreise als Makroelement (Gegenprobe 60)

Das 2D-Feld verschmierte jeden Lochkreis zu einem Gaußband der Breite
0,1·a_bp und gab jeder Bohrung die Škvor-Zelle in Serie. Für kleine
Löcher glichen sich die beiden Näherungen aus; für große Löcher (B&K
4134: r = 0,14·a) überschätzte das den Filmwiderstand um 60–70 %
(Gegenproben 52, 58, 59). Zwei Zwischenstufen wurden geprüft und
verworfen:

- **Liniensenke** mit dem Zusammenlaufwiderstand ln(R/(n·r))/(2πK) je
  Loch (aus dem Potential ln|zⁿ − Rⁿ|): trifft kleine Löcher auf 0,6 %,
  wird aber negativ, sobald n·r > R. Große, äquipotentiale Löcher
  schließen das Feld zwischen innen und außen teilweise kurz
  (Dipolanteil); die Liniensenke kennt das nicht (B&K: 2,5-fach daneben).
- **Exakter Dreipol** (innen, außen, Loch) aus einer Sektorrechnung:
  behebt den Kurzschluss, verliert aber die Quellen im Band, also die
  Druckbögen zwischen den Löchern und die Lochfläche ohne Film (±3 %).

**Das Makroelement** (`_lochband_makro`) nimmt alles mit. Je Lochkreis
rechnet eine kleine statische Sektorlösung die vollständige Kopplung
zwischen den FV-Zellen des Bandes, den Nachbarzellen innen und außen
und einem Lochknoten je Lochgruppe. Gelöst wird in der log-Ebene
ζ = ln z, in der die Reynolds-Gleichung konform invariant bleibt: der
Kreisring wird zum Streifen, das Lochmuster periodisch mit 2π/g (g =
ggT der Lochzahlen). Das Tensorgitter ist um die Löcher verdichtet, am
Lochrand gilt die symmetrische Shortley–Weller-Form (Gibou et al. 2002,
zweite Ordnung) mit den analytischen Schnittpunkten der Kreiskontur.
Knoten sind die Zellmittel des Drucks über der FILMfläche jeder Zelle,
gegen eine über dieselbe Fläche gleichverteilte Quelle; beides ist
zueinander reziprok, die Leitwertmatrix symmetrisch mit Zeilensumme 0.
Im Feld skaliert sie mit der Filmleitfähigkeit K(ω) (Reibung und
Trägheit). Speicherung und Membranquelle der Zellen laufen über die
Filmfläche, die Membran über den Mündungen pumpt in den Lochknoten und
spürt dessen Druck. Durchgangslöcher führen von dort mit n/Z_Loch zum
Rückport, Sacklöcher mit ihrem Stub gegen Masse. Zellen ganz in einem
Mittelloch oder in einem geschlossenen Ringschlitz aus sich
überlappenden Löchern hängen am Lochknoten. Das Ergebnis ist
maßstabsfrei und wird zwischengespeichert; der Aufbau dauert für die
B&K 4134 rund 60 ms, für die Debenham-Platte (acht Lochkreise) 200 ms.

**Bänder.** Kreise, deren Lochspannen [R − r, R + r] sich berühren,
teilen sich ein Band. Jedes Band reicht drei Abklinglängen R/g der
Lochharmonischen über die Spanne hinaus (Störung am Rand e⁻³), höchstens
bis zur logarithmischen Mitte zum Nachbarband. Klingen die Harmonischen
bis dorthin nicht ab, teilen sich die Kreise ein Band, solange das
Sektorgitter höchstens 30 000 Knoten hat. Getrennt verlöre die B&K 4146
1,5 % Filmwiderstand; die Debenham-Platte mit teilerfremden Lochzahlen
bleibt an zwei Stellen getrennt (dokumentierte Näherung).

**Gegenproben:**

| Prüfung | Gaußband + Škvor | Makroelement |
|---|---|---|
| statisch gegen unabhängige exakte Lösung (4 Lochkreise, B&K, Tab. I, r = 0,15·a, 24 kleine) | B&K 1,71-fach | höchstens 0,22 % |
| Mittelloch gegen geschlossene Form | — | 0,1 % |
| 3D/2D bei erzwungener Form, 26 Lochkreis-Fälle der Testbank (zwei Kapseln) | 0,956…1,019 | 0,999…1,000 |
| dasselbe, B&K-Fälle ohne Folienverlust (s. u.) | 0,59…0,72 | 0,99…1,04 |
| COMSOL-FEM B&K 4134, RMS 1–20 kHz | 1,80 dB | 0,26 dB |
| Re Z gegen FEM | 1,61 | 0,94 |
| Zuckerwar 4146 (Fig. 7), Amplitude | 1,09 dB | 0,63 dB |

Die unabhängige Lösung (`_exakt60`) rechnet die GANZE Platte auf dem
Halbsektor einer Lochteilung, mit anderer Diskretisierung als das
Makroelement; ihre eigene Konvergenz liegt bei 0,1 %. Das Zweitor bleibt
mit Bändern, Mittelloch, Ringschlitz und Freistich reziprok (det T = 1
auf 10⁻⁹).

**Was es verändert hat:**

- **B&K 4134 (Zuckerwar 1978):** 2D liegt jetzt wie 3D über der Messung
  (2,1 gegen 1,7 dB RMS, vorher 0,30 dB). Der alte Treffer war der
  Ausgleich zweier Fehler (Gegenprobe 59). Die Messschranken gelten dort
  nicht mehr dem 2D-Modell; geprüft wird, dass 2D und 3D gleich liegen.
- **Debenham mit Rand-Freistich:** ein Schalter gab bisher ALLEN
  Durchgangslöchern die entlastete Engstelle, sobald eines im Freistich
  lag. Unter dem Rand-Freistich der Zeichnung liegen aber nur die sechs
  äußeren. Mit dem Makroelement entlastet er nur diese: die Null bei
  500 Hz bleibt flach (2D −3,2 dB, 3D −4,1 dB, vorher 2D −17,5 dB),
  die Niere ist stark über-verzögert (Verhältnis 5,2; 3D 3,9; vorher
  1,2), die interne Resonanz steigt von 2,3 auf 3,1 kHz. Erst ein
  Freistich über allen Mündungen vertieft die Null (2D −22,3, 3D
  −25,6 dB). Gegenprobe 15 prüft jetzt das.
- **Lochkreis-Warnung (Gegenprobe 53):** neu vermessen an 80
  Lochkreis-Fällen. Die Warnung kommt weiter in allen vor dem
  1-dB-Einsatz; ohne die Lochkreis-Prüfung blieben 8 Fälle (½"-Kapsel,
  65 µm) ungewarnt. Die Ursache ist jetzt allein die Formanpassung der
  Membran. Die Stichprobe der Gegenprobe ist deshalb der 65-µm-Fall.

**Grenzen:** Freistich-Zellen gehen mit dem statischen Verhältnis
((h + t)/h)³ ins Band ein. Oberhalb einiger kHz fällt das wahre
Verhältnis durch die Trägheit (Debenham: 7,97 statisch, 3,6 bei 10 kHz);
das ändert die Debenham-Kurve bei 10–20 kHz um höchstens 0,2 dB. Die
Durchbiegung der polarisierten Membran nimmt das Band über den mittleren
Leitwert mit, nicht als Profil.

**Vorsicht bei der Phasenmethode** (erzwungene Form, Gegenproben 52,
53, 60): der Materialverlust der Folie war im 2D ein fester Widerstand
ω₀·M/Q (an der Resonanz definiert), im 3D wuchs er mit ω. Mit
f_res = 300 kHz blähte das den 2D-Wert bei 1 kHz 300-fach auf; bei
Nickelfolie waren das bis 8 % des Filmwiderstands (B&K-Geometrie: 3D/2D
scheinbar 0,94 statt 1,02, mit 40 µm Spalt 0,49 statt 1,03). Seit
Gegenprobe 61 tragen beide Modelle denselben hysteretischen
Folienverlust, und er fällt aus dem Phasenverhältnis heraus. Hier
stand danach, 3D liege bei großen Mündungen 1–3 % über der exakten
Lösung und am Mittelloch der 4146 um 5–54 %. Gegenprobe 68 hat das
aufgelöst: das Mittelloch war ein Fehler im Fußabdruck, der Rest kam
von zwei weiteren Fallen der Methode — gemessen an der Spannung (die
über den Löchern keine Elektrode sieht) und bei Kapseln mit Körper mit
Beugung (deren Phase die Filmphase überdeckt). Seitdem misst
`filmwiderstand_3d_zu_2d` (tests/basis.py) an der Volumenverschiebung
der ganzen Membran ohne Beugung; 3D trifft das exakte 2D auf 0,1–0,5 %
(`gp60.r3d_zu_exakt_bk` 1,004).

### Folienverlust (Gegenprobe 61)

Bis Gegenprobe 60 stand in der Membran eine Materialgüte Q = 100, als
„numerischer Boden" gedacht und in beiden Modellen verschieden
umgesetzt: die 2D-Kette als fester Widerstand R = √(M/C)/Q (an der
Resonanz definiert), der 3D-Löser als Masse σ·(1 − j/Q) (mit ω
wachsend). Keines der beiden Gesetze war begründet, und bei erzwungener
Form verfälschte der Unterschied den Vergleich der Filmwiderstände um
bis zu 8 % (Gegenprobe 60).

**Physik.** Die Folie verliert über den komplexen E-Modul E·(1 + j·η)
ihres Werkstoffs. Bei einer vorgespannten Folie ist aber die Arbeit
gegen die Vorspannung in erster Ordnung verlustfrei: die Dehnung, die
eine Auslenkung erzeugt (w'²/2), ist quadratisch, die Vorspannung
statisch. Verlustbehaftet ist nur die Biegeenergie — die
Dissipationsverdünnung der Nanomechanik (Fedorov et al., Phys. Rev. B
99, 054107, 2019). Je Mode gilt

    η_m = η · U_Biegung/U_gesamt = η · ∂ln ω_m²/∂ln D,

ausgewertet an der exakten Eigenwertgleichung der eingespannten Platte
unter Zug (J0/I0-Ansatz, mit Pfosten zusätzlich Y0/K0;
`_platten_verduennung`). Die Biegeenergie sitzt fast ganz in der
Randschicht der Breite √(D/T) an der Einspannung; ohne Pfosten ist
η_m ≈ η·(λ + z_m²·λ²), λ = √(D/(T a²)). Für reale Folien sind das
0,1–0,7 % des Werkstoffwerts: K67 (PET 6 µm) Güte 6900, B&K 4134
(Nickel) rund 150 000 mit dem Nickelwert.

**Form, in 2D und 3D gleich.** Hysteretisch, also frequenzunabhängig
auf der mechanischen Steifigkeit (die elektrostatische Erweichung ist
verlustfrei): in der 2D-Kette R_m = η_m/(ω·C_m) je Mode, im 3D-Feld die
Spannung T·(1 + j·η_1). Das 3D-Feld hat keinen Biegeoperator; alle
seine Formen tragen den Verlustfaktor der Grundmode.

**Werkstoffwerte** (`MATERIALS["eta"]`, Größenordnungen bei
Raumtemperatur und kleiner Amplitude): PET 0,02 (glasig zwischen
β-Relaxation und Glasübergang 0,01–0,06), Nickel, Titan, Gold 10⁻³,
Aluminium 10⁻⁴ (Blanter et al., *Internal Friction in Metallic
Materials*, Springer 2007). Eigene Material-dicts ohne `eta` erhalten
den Polymerwert 0,02 — für Metalle eher zu hoch. Die genaue Zahl ist
belanglos: auch das Zehnfache ändert den Frequenzgang der K67 um
0,002 dB, einer Nickelkapsel um 0,01 dB.

**Gegenproben:**

- Eigenwertgleichung gegen die Randschicht-Asymptotik (Kreis und Ring
  mit Pfosten): Abweichung O(λ), mit λ fallend.
- Komplexe Eigenwerte mit D·(1 + jη): Im(ω²)/Re(ω²) = η·q auf 1 %.
- 2D und 3D tragen denselben Verlust: ein großer Verlustfaktor
  verschiebt die Phase im steifigkeitsbestimmten Tiefton gleich (5 Hz:
  −0,947 gegen −0,946 mrad).
- Wirkung an realen Kapseln: η_eff ist 0,1–0,7 % von η, ×10 ändert
  höchstens 0,05 dB.

**Was es verändert hat:** der alte Boden Q = 100 trug an
Hochton-Resonanzen spürbar zur Dämpfung bei. An der B&K 4134 (Resonanz
um 20 kHz) liegt 2D jetzt 0,28 statt 0,26 dB RMS neben der COMSOL-FEM,
3D 0,097 statt 0,099 dB. Bei der K67 ändern sich die Stand-Werte um
höchstens 0,1 %.

**Nebenbefund:** dieselbe Randschicht hebt die Grundfrequenz um den
Faktor ≈ 1 + λ, den die „exakte Modalfrequenz" der Kette nicht enthielt
— umgesetzt in Gegenprobe 62.

### Randschicht der Folie (Gegenprobe 62)

Eine eingespannte Folie ist eine Platte unter Zug, D·∇⁴w − T·∇²w = p,
mit w = w' = 0 am Rand. Für λ = √(D/(T a²)) ≪ 1 ist sie im Innern
Membran; nur in einer Randschicht der Breite ℓ = √(D/T) biegt sie sich
in die Einspannung. Außerhalb der Schicht erfüllt die Membranlösung
w = ℓ·∂w/∂r, verschwindet also um ℓ vor dem Rand. In erster Ordnung ist
die Folie eine **Membran mit dem wirksamen Radius a − ℓ** (am Pfosten
r_i + ℓ).

**Was fehlte.** Bis Gegenprobe 61 stand die Biegung als parallele
Plattenfeder C_B = πa⁶/(192D) in der Kette (C_T/C_B = 24λ²), und die
„exakte Modalfrequenz" addierte Membran- und Platten-Eigenwert
quadratisch — beides Terme der Ordnung λ². Die Randschicht ist ein Term
der Ordnung λ: sie senkt die statische Nachgiebigkeit um ≈ 4λ und hebt
die Grundfrequenz um ≈ λ. Bei der B&K-Nickelfolie (5 µm, 3162 N/m,
λ = 0,0061) sind das −2,4 % und +0,61 %, im Tiefton −0,15 dB. Bei
vorgegebener Resonanz — so sind die meisten Kapseln beschrieben —
bleibt die Resonanz, und die Nachgiebigkeit sinkt um ≈ 2λ (PET 6 µm:
λ = 0,001…0,007).

**Umsetzung.** `_wirk_membran` liefert die wirksame Membran in der
Bezugsgröße a_mem: statische Form (Parabel bzw. Ringform zwischen den
wirksamen Rändern, `_form`), ihre Momente, den Nachgiebigkeitsfaktor,
die Eigenwerte und Modenintegrale. Kette, Moden, Quellprojektion,
Elektrostatik, exakter Arbeitspunkt und 2D-Feld rechnen damit. Der
3D-Löser spannt seine Membranfelder an denselben Rändern ein: endet das
Gitter dort (Ring außerhalb der Elektrode), exakt; liegt die
Einspannung im Elektrodengitter oder am Pfosten, als Schnittzelle mit
Elementgewicht (exakt für stückweise lineare Auslenkung) und
logarithmischem Leitwert (exakt für die radiale Laplace-Lösung) — am
Pfosten 10–20-mal genauer als die frühere Halbzelle vor der Wand. Bei
vorgegebener Resonanz ist die Spannung die, deren wirksame Membran
f_res trifft (Fixpunkt, weil ℓ von T abhängt). Die „exakte
Modalfrequenz" in `summary()` ist jetzt der Eigenwert der Platte unter
Zug (`_platten_verduennung`); die Kette liegt um die Plattendispersion
im Innern darunter (B&K −0,01 %). Der Klassenschalter `_RANDSCHICHT`
schaltet die Randschicht für Vergleiche mit reinen Membranmodellen ab.

**Gegenprobe 62:**

| Prüfung | Ergebnis |
|---|---|
| Statik gegen eine unabhängige Lösung der Plattengleichung (Kreis, Ring) | 10⁻⁷…10⁻⁶ (die Randschicht selbst: −2,4 % bei B&K) |
| unabhängige Lösung gegen die geschlossene Form 1 − 4λ(I₀/I₁ − 2λ) | < 10⁻⁶ |
| Frequenz gegen den exakten Eigenwert der Platte | Rest = Plattendispersion ≈ −z₁²λ²/2 (B&K −1,1·10⁻⁴) |
| Form über der Elektrode gegen die Platte | 5·10⁻⁵ (ohne Randschicht 8·10⁻³) |
| 3D-Membranfeld gegen die Kette (Gitterende, Schnittzelle, Pfosten) | ≤ 2·10⁻⁴ |
| Randschicht verschiebt 2D und 3D gleich | auf 0,002 dB |
| Warren mit a − ℓ | Ā = 0,7892 (mit a: 0,7998) |
| Gatter | Warnung ab 1 % gegen die Platte, Abbruch ab ℓ > ¼ Membranbreite |

**Referenzen ohne Biegesteifigkeit.** Die COMSOL-FEM der 4134 rechnet
die Folie mit dem Membraninterface („Model the diaphragm using the
Membrane interface … add an initial stress equal to the membrane
tension", Modelldokumentation), Šimonová/Honzík geben Spannung und
Resonanz nach der Membranformel (1040 Hz aus 116,27 N/m; mit Biegung
wären es 1053 Hz), und Zuckerwars Tabelle II ist sein Membranmodell.
Gegen diese Modelle rechnet Capsim ohne Randschicht (Gegenproben 32,
34, 38 a, 59 a/c). Die Filmproben 37 und 60 setzen die Parabel bis
a_mem an und laufen ebenfalls ohne.

**Gemessene Prüflinge.** Zuckerwar bestimmt die Spannung aus der
gemessenen ersten Vakuumresonanz mit der Membranformel (NASA-Bericht
PGSTR-PH77-48, 1977, Gl. 2-34: T = 6,825·a²·f_R1²·ρt); Grinnip gibt die
Vakuumresonanz direkt an (3500 Hz). Solche Spannungen sind
membranäquivalent, die Randschicht steckt schon darin. Gegen Messungen
bekommt der Prüfling deshalb die Resonanz vorgegeben (`_messpruefling`
in `tests/test_referenzen.py`, Gegenproben 38, 41, 43, 58, 59 b);
mit der Spannung zählte das Modell die Randschicht doppelt (Resonanz
0,6 % zu hoch). Gegen die Messungen ändert die Wahl wenig, weniger als
die Ableseunsicherheit, und sie hat die Wahl nicht bestimmt: 4134
2,13 → 2,11 dB RMS (2D), 4146 0,64 → 0,63 dB, B&K-Messmittel 3D
0,33 → 0,32 dB.

**Was es verändert hat.** Bei vorgegebener Resonanz wenig: K67 im
Tiefton −0,002 dB, die Niere bei 1 kHz −28,1 → −27,8 dB (2D), die
interne Resonanz der Debenham-Kapsel −1,8 %; die übrigen Stand-Werte
meist unter 1 %, größere relative Änderungen nur bei kleinen
Beträgen (2D − 3D bei 16 Löchern 0,10 → 0,06 dB). Bei vorgegebener
Spannung hebt sich die Resonanz um ≈ λ und der Tiefton sinkt um ≈ 4λ
(B&K −0,15 dB, `gp62.bk_tiefton_db`).

### Eingabebereiche der GUI (Gegenprobe 63)

**Befund.** In der App fiel der Frequenzgang beider B&K-4134-Beispiele
ab etwa 700 Hz (COMSOL-Geometrie: −0,3 dB bei 1 kHz, −7,6 dB bei
10 kHz), obwohl das Modell aus denselben Dateien flach bis 10 kHz
rechnet. Ursache war nicht die Physik: das GUI-Feld „Randspalt“ ging
bis 500 µm, die Dateien haben 860 bzw. 838 µm. Streamlit (1.64) setzt
einen Session-Wert außerhalb [min, max] beim Aufbau des Zahlenfelds
**still auf das Minimum**, ohne Fehler. Der Randspalt war damit
geschlossen, die Luft unter dem Membranrand musste durch den 18,6-µm-
Spalt zu den sechs Löchern, und die Membran war überdämpft. Der Lader
meldete trotzdem „58 Parameter übernommen“. Im Browser (Chromium, Upload
der Datei) ist das mit dem App-Stand vor `b795a8a` genau nachgestellt.
Schieberegler verhalten sich anders: sie lassen Werte außerhalb
ungeprüft durch.

**Korrektur.** Die Feldgrenzen stehen an EINER Stelle (`_BEREICH`,
abhängige Grenzen in `_bereich()`: Mittenstift ≤ Membranradius,
Lochkreis ≤ Backplate, Sacklochtiefe < Plattendicke, Bohrungsposition
≤ Zylinderlänge, Körper ≥ Kapsel). Jedes Zahlenfeld entsteht über
`_zahl()`, das einen Wert außerhalb vorher auf die **nächste** Grenze
klemmt und im Hauptbereich meldet („Wert außerhalb seines
Eingabebereichs … 2500 → 2000“). Das gilt auch für die bisher still
geklemmten abhängigen Felder und die Zahl der Frequenzpunkte. Der
Lader ist als `_projekt_params()` herausgelöst, damit der Test genau
ihn benutzt.

**Gegenprobe 63** (`tests/test_app_projekte.py`):

- a) Jede Projektdatei unter `examples/`, die Voreinstellung und der
  Null-Zustand liegen in den Feldbereichen; kein Projektschlüssel wird
  vom Lader verworfen; jeder Zahlenschlüssel hat einen Bereich. Mit dem
  alten Maximum 500 µm meldet die Prüfung genau den Randspalt.
- b) Jedes Beispiel läuft durch die echte App (Streamlit-AppTest): kein
  Fehler, keine Klemmung, jeder Wert unverändert im Session-State, und
  die gezeichnete Kurve ist der Frequenzgang des Modells aus derselben
  Datei (Abweichung unter 10⁻¹¹ dB).
- c) Randspalt 2500 µm, Lochkreis 9 mm auf 7,2-mm-Backplate und
  Sacklochtiefe 5 mm werden zu 2000 µm, 7,2 mm und 0,929 mm, mit einer
  Meldung, die im nächsten Lauf entfällt. Vorher-Befund: mit Randspalt 0
  liegt 10 kHz bei −7,6 statt +1,1 dB (re 100 Hz) — das ist der
  gemeldete Abfall.

### Freifeldkorrektur der B&K 4134 (Gegenprobe 64)

**Befund.** Mit „Gehäuse & Beugung“ lag der Frequenzgang der
B&K-Beispiele weit neben der COMSOL-Rechnung. Das ist zum größten Teil
richtig so: COMSOL rechnet den **Druckfrequenzgang** (gleichförmiger
Druck auf der Membran, kein Körper), mit Beugung rechnet Capsim den
**Freifeldfrequenzgang** bei 0°. Dazwischen liegt die
Freifeldkorrektur, der Druckstau vor der Stirnfläche — bei einem
½-Zoll-Mikrofon gemessen +4,0 dB bei 10 kHz und +8,0 dB bei 20 kHz.
Falsch war dagegen ihr Betrag: die Voreinstellung „Kugel“ legt die
Membran auf eine gekrümmte Kalotte und liefert nur +3,4 / +4,1 dB.
Und „BEM (Kopf + Körper)“ hätte den 56-mm-Körper der U87-Voreinstellung
mitgerechnet (bei 4 kHz −10,8 dB).

**Referenz.** E. D. Burnett, V. Nedzelnitsky, „Free-Field Reciprocity
Calibration of Microphones“, J. Res. NBS 92(2), 129–151 (1987),
Fig. 23: die gemessene Freifeldkorrektur einer 4134 ohne Schutzgitter
(Membran bündig in der Stirnfläche), dazu Matsuis Theorie für den
halbunendlichen Stab bis 3 kHz. Gemeinfrei; aus dem PDF digitalisiert
(Ring-Mustervergleich, Achsen aus den Teilstrichen; Gegenprobe der
Digitalisierung: Tabelle 6 des Berichts auf ≤ 0,07 dB), die Werte stehen
in `tests/basis.py`.

**Gegenprobe 64** (`tests/test_beugung_bem.py`; Prüfling: der
Messprüfling der COMSOL-Geometrie als ½-Zoll-Stab, flache Stirnfläche
⌀13,2 mm, 50 mm lang, ohne Körper; Stand nach Gegenprobe 65, also
Druckgang ohne Strahlungslast — vorher 0,58 dB RMS mit dem BEM):

| | 1,25–4 kHz | 10 kHz | 16 kHz | 20 kHz | RMS 4–20 kHz |
|---|---|---|---|---|---|
| NBS-Messung | −0,03…+0,99 | +4,01 | +7,04 | +7,96 | — |
| BEM, flache Stirnfläche | ±0,12 dB daneben | +4,51 | +7,57 | +8,42 | 0,53 dB |
| Kugel (bisher Voreinstellung) | bis −0,45 dB daneben | +3,36 | +3,91 | +4,14 | 2,25 dB |

- a) Tiefton: BEM trifft die Messung innerhalb ihrer Unsicherheit
  (0,16 dB) und bis 2,5 kHz Matsuis Theorie auf 0,05 dB.
- b) Hochton: BEM liegt bis 0,8 dB darüber; Sperrklinke
  `gp64.rms_bem_gegen_nbs` (Ursachen offen, s. „Offene Punkte“ 7).
- c) Kugel: Stand-Werte; geprüft wird die Richtung (BEM mindestens
  doppelt so gut, Kugel ab 10 kHz zu niedrig).
- d) `pressure_response()` der Freifeldkapsel ist exakt der Druckgang
  der Kapsel ohne Beugung (Abweichung ≤ 10⁻¹²; bis Gegenprobe 65 war
  es der Freifeldgang geteilt durch den Frontfaktor).
- e) **Gewicht des Frontdrucks.** Die Kette projiziert den
  Oberflächendruck auf die Grundmode J₀. Physikalisch richtig ist das
  reziproke Gewicht, c = S⁻ᵀ·w_out aus dem 3D-Feld (Membran im
  verteilten Spaltfilm). Gegenprobe der Methode: gleichförmiger Druck
  ergibt X_f auf 10⁻⁸. Ergebnis: −0,02 dB gegen J₀ — der Film flacht
  die Membranform nicht ab. Das Flächenmittel träfe die NBS-Messung
  zufällig besser (0,29 dB RMS); es ist physikalisch nicht begründet
  und wird nicht benutzt.

**Was sich geändert hat.**
- Das Bode-Diagramm zeigt mit Beugung bei dichter Rückseite zusätzlich
  den **Druckfrequenzgang** gestrichelt („wie COMSOL/Kuppler“), nach d)
  aus derselben Rechnung; der Abstand beider Kurven ist die
  Freifeldkorrektur. Der CSV-Export hat dafür die Spalte
  `pressure_response_db`.
- Bei einer Ein-Membran-Kapsel mit dem Körpermodell „Kugel“ warnt die
  App und verweist auf das BEM.
- Die B&K-Beispiele rechnen mit Beugung den ½-Zoll-Stab im BEM
  (Kapsellänge 30 mm, Körper-Ø 0; 30 mm liegen höchstens 0,26 dB neben
  100 mm, bei 0,2 s statt 2 s je Frequenzpunkt). Ohne Beugung, also für
  den Vergleich mit COMSOL, ändert sich nichts.
- Die Phase des Freifeldgangs bezieht sich, wie bei der Kugel, auf den
  ungestörten Druck im Körpermittelpunkt; beim 30-mm-Stab enthält sie
  deshalb die Laufzeit über 15 mm (+157° bei 10 kHz).

### Strahlungslast im Druckgang (Gegenprobe 65)

**Befund.** Capsim legte die Strahlungsimpedanz der Membran (Kolben in
unendlicher Schallwand) immer vor die Membran, auch ohne Beugung. Bei
einer dichten Kapsel ist das aber der **Druckfrequenzgang**:
gleichförmiger Druck an der Membran wie in der COMSOL-FEM, im Kuppler
oder am Aktuator. Dort strahlt die Membran nicht ins freie Feld. Bei der
4134 senkte die Last den Hochton um 0,30 dB bei 20 kHz
(`gp65.last_20khz_2d`).

**Korrektur.** Die Last liegt jetzt nur im **Freifeld** an: mit
Beugung, bei offener Rückseite (auch im Modell ohne Körper strahlt die
Membranvorderseite) und bei der Doppelmembran-Bauform. Im Druckfeld
fehlt sie standardmäßig; `pressure_radiation_load=True` bzw. der
GUI-Schalter „Strahlungslast im Druckgang“ (Gehäuse & Beugung) legt sie
wieder dazu. Im 3D-Löser wird der Frontknoten ohne Last und ohne
Gewebe auf den Quelldruck gesetzt (Dirichlet statt 1/Z). Neu ist
`pressure_response()`: der Druckgang einer dichten Kapsel, auch wenn
sie mit Beugung gebaut ist. Im 3D-Modell kommt er aus derselben
Feldlösung (Knotendruck p_n und Netzwerkfluss q_n:
X_p = (X_f/p_n)/(1 + Z_Gewebe·q_n/p_n)), die GUI zeichnet ihn als
gestrichelte Vergleichskurve.

**Gegenprobe 65** (`tests/test_beugung_bem.py`):
- a) Im Freifeld (Beugung, offene Rückseite, K67) ändert der Schalter
  nichts, bitgleich.
- b) Im Druckfeld ist die Last exakt ein Serienglied:
  Θ/(jω·H_mit) − Θ/(jω·H_ohne) = Z_rad auf 10⁻¹⁰.
- c) Der Dirichlet-Knoten des 3D-Lösers ist der stetige Grenzfall
  Z → 0 (4·10⁻¹¹); `pressure_response()` der Freifeldkapsel ist in 2D
  und 3D, mit und ohne Frontgewebe, der Druckgang ohne Beugung. Mit
  gesetztem Schalter rechnet das Modell bitgleich wie vor der Änderung
  (gegen den alten Code geprüft, 10⁻¹⁶).
- d) Gegen die COMSOL-FEM (falls die Daten vorliegen) trifft 3D ohne
  Last besser: 0,098 → 0,066 dB RMS.

**Was sich verschoben hat** (bewusst in die Basis übernommen):

| Vergleich | Referenz | mit Last | ohne Last |
|---|---|---|---|
| 3D gegen COMSOL-FEM | ohne Last | 0,098 dB | 0,066 dB |
| 2D gegen COMSOL-FEM | ohne Last | 0,28 dB | 0,31 dB (max. 0,68) |
| Resonanzlage gegen die FEM (Gegenprobe 32), 2D / 3D | FEM | 9,6 / 10,0 % | 9,1 / 9,5 % |
| BEM-Freifeldkorrektur gegen NBS | Freifeld | 0,58 dB | 0,53 dB |
| 3D gegen Zuckerwar 4146 (Aktuator) | Aktuatorlast | 0,744 dB | 0,734 dB |
| 2D gegen Zuckerwar 4146 (Aktuator) | Aktuatorlast | 0,63 dB | 0,87 dB |
| 2D / 3D gegen Zuckerwar 4134 (Aktuator) | Aktuatorlast | 2,11 / 1,67 dB | 2,17 / 1,71 dB |
| 2D / 3D gegen B&Ks Messmittel | Aktuator (vermutlich) | 0,56 / 0,32 dB | 0,61 / 0,36 dB |

Gegen alle lastfreien Referenzen wird das Modell besser, gegen die
Aktuatormessungen meist etwas schlechter. Der Aktuator belastet die
Membran selbst (Luft zwischen Gitter und Membran, Strömung durch die
Schlitze); die Freifeld-Strahlungslast wirkte dort als ungefährer
Ersatz. Zwei Grenzen sind deshalb bewusst neu festgelegt (Entscheidung
des Projekts): die 2D-Abweichung gegen die FEM in Gegenprobe 59
(0,6 → 0,7 dB) und die Sperrklinke `gp38.rms_2d_4146_db`
(0,63 → 0,87 dB). Beide alten Werte enthielten die Last, die den
Hochtonüberschuss des 2D-Modells verdeckte; das 3D-Modell trifft beide
Referenzen ohne sie besser. Die Aktuatorlast selbst ist offen („Offene
Punkte“ 8).

### Sprachen der Oberfläche (Gegenprobe 66)

`tests/test_app_projekte.py`:
- a) Jeder Eintrag in `TR` und `LABEL_TR` hat Englisch und Deutsch, mit
  denselben Platzhaltern, und lässt sich in beiden füllen. Jeder
  `tr()`-Aufruf in `app.py` nennt einen vorhandenen Schlüssel (aus dem
  Quelltext gelesen), und kein Schlüssel ist unbenutzt. Gefunden hat das
  den unbenutzten Schlüssel `prog_di`: der Fortschrittstext beim
  Richtdiagramm war fest „Richtdiagramm … Hz“, auch auf Englisch. Ebenso
  waren die Meldungen des Projekt-Laders („ungültiger Wert …“, „kein
  Capsim-Projektformat“) fest deutsch; alle drei laufen jetzt über `tr()`.
- b) Voreinstellung und alle Beispielprojekte laufen auf Englisch und
  auf Deutsch durch die App (16 Läufe): kein Fehler, und kein sichtbarer
  Text — Überschriften, Meldungen, Feldbeschriftungen und Hilfen,
  angezeigte Optionen, Diagrammtitel und Spurnamen — ist ein fester Text
  der anderen Sprache.
- c) Meldungen des Modells: bis hier waren sie nur deutsch und
  erschienen so auch in der englischen Oberfläche. Jede Parameterprüfung
  wirft jetzt einen `ParameterFehler` — ein `ValueError` mit der
  deutschen Meldung wie bisher (`str(exc)`, für Skripte unverändert) —,
  dessen Text `exc.text("en")` aus der Tabelle `_MELDUNGEN` neben dem
  Modell bildet (54 Meldungen, Deutsch und Englisch mit denselben
  Platzhaltern). Geprüft: jeder Eintrag zweisprachig und füllbar, jeder
  `ParameterFehler(...)` im Quelltext mit vorhandenem Schlüssel und
  genau dessen Platzhaltern, kein Eintrag unbenutzt, und in den
  prüfenden Methoden (Bau, abgeleitete Größen, Randschicht, Ringmoden,
  BEM, 3D-Gitter) kein einfacher `ValueError` mehr. Einfache
  `ValueError` bleiben für falsche Aufrufe aus eigenem Code (interne
  Löser, Rauschspektrum, Druckgang bei offener Rückseite).
- d) In der App: die Voreinstellung mit 400 V kollabiert (Pull-in); die
  Meldung erscheint auf Englisch und auf Deutsch in der jeweiligen
  Sprache. Scheitert eine Rechnung an den Maßen (etwa eine nicht lösbare
  BEM-Randintegralgleichung), zeigt die App dieselbe Meldung statt eines
  Tracebacks.

### Mündung Spaltfilm → Bohrung (Gegenprobe 67)

**Was fehlte.** Der Reynolds-Film (Škvor-Zelle, Makroelement,
3D-Filmgitter) führt die Strömung bis an den Lochrand und setzt dort den
Lochdruck; das Zwikker–Kosten-Rohr beginnt voll entwickelt an der
Plattenoberfläche. Beides ist für sich exakt — die **Umlenkung**
dazwischen trug keines. Vor der Kante verliert der Film auf einer
Strecke ~h die Schmierfilmform, hinter ihr ist das Rohr auf ~a nicht
entwickelt, und die Membran über der Öffnung drückt unmittelbar in die
Bohrung. Die bisherige Annahme, die Škvor-Ausbreitung decke die
filmseitige Mündung ab (Gegenprobe 28), ist der Dünnspalt-Grenzfall.
Bei der FEM-Kapsel der Gegenprobe 32 (h/a = 0,46) fehlten dem Lochpfad
13 % Widerstand und 7 % Masse.

**Referenzlöser** (`tests/stokes_zelle.py`): linearisierte,
inkompressible, instationäre Stokes-Gleichung in der axialsymmetrischen
Lochzelle (Spalt über der Zelle, Bohrung darunter, Membran gleichförmig
bewegt), MAC-Gitter von der Ecke aus geometrisch gestuft. Film und Rohr
werden auf **demselben** Gitter abgezogen; so bleibt nur der Gitterfehler
der Ecke (< 0,2 % von ΔZ). Kein Abgleich mit der FEM.

**Dreitor.** Die Ecke koppelt Filmzufluss, Membranfluss über der Öffnung
und Rohrstrom; für eine Zelle mit Lochflächenanteil q gilt
ΔZ = (1−q)²·z_ff + 2q(1−q)·z_fm + q²·z_mm. Das ist bis auf Rundung
exakt (aus drei Zellen bestimmt, sagt es fremde auf 10⁻⁹ voraus), weil
die Schmierfilmlösung mit gleichförmigem Quetschen im Ring selbst eine
exakte Stokes-Lösung ist: Abweichungen entstehen nur an der Ecke.
Jedes Element steht als **Widerstands- und Massen-Zusatzlänge** auf
einem Belag des Modells — z_ff in Spalthöhen auf dem Filmbelag am
Lochrand, z_fm und z_mm auf dem Rohrbelag. Getrennt wie 3π/16 und 0,85
am Rohrende, und aus demselben Grund: viskose und träge Umlenkung sind
verschiedene Strömungen; die getrennten Längen laufen über die Frequenz
glatt. Tabelliert über h/a = 0,01…2 und a/δ = 0,3…128, kubisch
interpoliert (gegen Direktlösungen zwischen den Stützstellen höchstens
1,2 %). `python tests/stokes_zelle.py` rechnet die Tabelle zeichengleich
neu.

**Grenzfälle, geprüft:**
- **Dünnspalt:** die Film-Zusatzlänge läuft gegen 0,637·h — die
  Eckkonstante der *ebenen* Umlenkung, unabhängig mit einem kartesischen
  Löser gerechnet (`tests/stokes_eben.py`), der vorher Hasimotos exakten
  Schlitz (32μq'/πh²) auf 0,5 % trifft. Relativ zum Film verschwindet die
  Korrektur linear mit h/a (0,40/0,81 % bei h/a = 0,01/0,02): der
  Škvor-Grenzfall bleibt exakt. Genau daran scheiterte die Stokes-Zelle
  von Homentcovschi/Murray/Miles (Gegenprobe 36).
- **Trägheit:** für a/δ → ∞ laufen die Massenlängen gegen die der
  Potentialströmung (dieselbe Zelle mit μ = 0) — die Mündungsmasse ist
  die kinetische Energie der wirbelfreien Umlenkung.

**Wirkung.** An allen membranseitigen Mündungen, in 1D, 2D und 3D an
derselben Stelle (in Serie zum Lochzweig). Gegenprobe 32: Resonanz 2D
584 → 561 Hz, 3D 581 → 558 Hz gegen die FEM bei 550 Hz, Überhöhung
+8,04 → +7,32 bzw. +7,87 → +7,18 dB gegen +6,74 dB, RMS gegen die
FEM 0,95 → 0,42 bzw. 0,72 → 0,23 dB. Gegen Messungen: B&K-Messmittel
3D 0,36 → 0,21 dB, 2D 0,61 → 0,46 dB; Zuckerwar 4134 2D 2,17 → 2,01 dB,
4146 2D 0,87 → 0,70 dB. Schlechter: 4146 3D (0,73 → 0,90 dB), 3D gegen
die COMSOL-FEM der 4134 (0,07 → 0,11 dB) und im Kerbenband der FEM-Kapsel
das Dublett und die streifende Anregung (s. „Offene Punkte" 1, 2, 4).
Die Sperrklinke 3D gegen die COMSOL-FEM ist dafür bewusst neu
festgelegt (0,066 → 0,112 dB).

**Prüfaufbau.** Gegenprobe 32 und 34 rechneten bis hier mit dem
Klassen-Standard `delay_length = 3e-3`, also mit einem Laufzeitglied,
das die FEM-Kapsel nicht hat (+39 % Rückvolumen). Es verdeckte, dass
das Modell ohne die Mündung 6 % zu hoch und zu schwach bedämpft lag.

### Mittelloch und Messgrößen im 3D-Löser (Gegenprobe 68)

Offen stand, der 3D-Löser überschätze den Filmwiderstand am Mittelloch
um 5–54 %, an großen Lochkreis-Mündungen um 1–3 %, und sein Widerstand
liege gegen die COMSOL-FEM der 4134 12–18 % hoch. Eine eigene Messbank
(Druckfeld und Membranauslenkung je Zelle aus der Lösung) hat das
getrennt:

- **Fußabdruck (Fehler, behoben):** die Zellen einer Mündung wurden
  ringweise in einem azimutalen Fenster der halben Breite r/r_i
  gesucht. Fern der Achse reicht das; ein Mittelloch — oder eines, das
  die Achse überdeckt — umfasst auf seinen äußeren Ringen aber den
  ganzen Umfang, und das Fenster schnitt es zum Keil zu (Druckfeld
  azimutal 17 % ungleich bei einer achsensymmetrischen Aufgabe). Jetzt
  mit dem exakten halben Öffnungswinkel; der Film am Mittelloch trifft
  die geschlossene Form auf 0,1 % (r/a = 0,06/0,14/0,28: vorher
  +6/+19/+55 %). Eine Kapsel nur mit Mittelloch brach im 3D-Löser sogar
  ab (Division durch null bei der Gitterauflösung). Betroffen war die
  4146: gegen Zuckerwars Fig. 7 jetzt 0,83 statt 0,90 dB.
- **Phasenmethode an der Spannung (Messgröße):** im Steifigkeitsbereich
  ist tan(Phase) ∝ Filmwiderstand — aber nur, wenn Ein- und Ausgang
  dieselbe Form sehen. Die Spannung sieht die Auslenkung nur über der
  Elektrode, die über den Löchern fehlt; das 3D-Feld zeigt damit bei
  großen Löchern 2–3 % mehr Verlust als die Einmoden-Kette, bei gleichem
  Film. An der Volumenverschiebung der ganzen Membran trifft 3D das
  exakte 2D auf 0,1–0,5 %.
- **Beugung (Messgröße):** bei Kapseln mit Körper rechnete die Methode
  mit Beugung, deren Phase (0,3 rad gegen 0,002 rad Filmphase) das
  Verhältnis trivial 1 machte. Die Prüfungen „gleichverteilte Löcher
  3D = 2D" (Gegenprobe 52) und die ½″-Lochkreise (Gegenprobe 60) prüften
  so nichts; jetzt ohne Beugung gemessen. Dabei zeigte sich: das grobe
  3D-Gitter hat bei sechs Mündungen mit 1,9 Zellen je Radius
  (azimutal) 1 % Filmfehler, das feine 0,7 % — Gegenprobe 60 rechnet
  diesen Fall seither fein.
- **Re Z gegen die FEM (Messgröße):** Gegenprobe 59 bildete den
  3D-Widerstand mit der Volumenverschiebung nur über der Elektrode
  (a_bp = 0,8·a_mem), FEM und 2D meinen die der ganzen Membran. Mit
  `weight='membrane'` (neu) liegt 3D bei 1,01 statt 1,18.

Die Filmvergleiche laufen jetzt über `filmwiderstand_3d_zu_2d` in
`tests/basis.py` (Volumen der ganzen Membran, ohne Beugung) und geben
die Spannung getrennt aus.

### Freistich am Ort der Mündung (Gegenprobe 69)

Ein Freistich (Clearance-Ring) vertieft den Spalt nur in einem Ring.
Das 2D-Feld setzte bis hier ein Flag, sobald irgendeine Mündung im
Freistich lag, und gab dann ALLEN Durchgangslöchern die entlastete
Engstelle (Zellterm) und seit Gegenprobe 67 auch die entlastete
Spalt-Mündung — bei gleichverteilten Löchern allen, sobald der
Freistich eine Zelle berührte, bei Lochkreisen auch denen weit weg vom
Freistich (den Film am Lochkreis rechnete das Makroelement schon
örtlich, die Mündung nicht).

Jetzt sieht jede Mündung den Spalt an ihrem Ort: gleichverteilte
Löcher je Zelle (Freistich der Zelle, Durchbiegung wie bisher über das
dichtegewichtete Mittel), Lochkreise am Kreisradius (Freistich und
Durchbiegung dort). Geprüft:

- **Grenzfall:** ein Freistich über die ganze Platte ist ein größerer
  Spalt — das Zweitor gleicht dem der Kapsel mit h + Tiefe auf 10⁻⁹,
  gleichverteilt und am Lochkreis.
- **Örtlich:** mit einem Freistich nur am Rand sehen gleichverteilte
  Mündungen h und h + t, ein Lochkreis innen nur h (die Spalt-Mündung
  wird dafür im Test protokolliert).

Wirkung: Debenham (Lochkreise, Freistich am Rand) interne Resonanz
2957 → 3062 Hz (`gp18.f_h_debenham`), Laufzeitverhältnis mit Freistich
5,32 → 5,40; an den B&K-Kapseln (Lochkreis, polarisiert) zählt nur die
Durchbiegung am Kreisradius statt im Mittel (Pegel ≤ 0,02 dB). Damit ist der
frühere offene Punkt „Freistich bei gleichverteilten Löchern“ erledigt.

### Portseitige Mündung nach Bauform (Gegenprobe 70)

Bis hier trug jede Durchgangsbohrung portseitig eine
Freifeld-Flanschmündung (0,85·r mit Fok-Faktor und Sampson-Widerstand).
Das stimmt nur, wo die Bohrung in ein großes Volumen (Hohlraum,
Laufzeitglied, Gewebe) oder ins Schallfeld mündet. `_portmuendung`
entscheidet jetzt nach Bauform:

- **großes Volumen oder Freifeld:** Flansch wie bisher — die neue
  Aufteilung (Rohr + Portmündung) gleicht der alten Form auf 10⁻¹⁶,
  gerade und gestuft;
- **dünner Spalt ohne Membran** (K67-Zwischenspalt, K103-Spacer): die
  Spalt-Mündung der Gegenprobe 67 mit q = 0 (über der Öffnung liegt
  keine Membran). Im Spacer gilt sie auch für die Löcher der
  Rückplatte, die von der anderen Seite in denselben Film münden;
- **einteilige Mittelelektrode** (Doppelmembran ohne Zwischenspalt):
  keine — bis hier saßen zwei Freifeld-Mündungen mitten im Rohr. Jetzt
  sind zwei Plattenhälften exakt das Rohr durch die volle Dicke, wie im
  3D-Löser.

**Im 3D-Löser** ist der Zwischenspalt aufgelöst. Fluchten die Kerne,
strömt die Luft von Loch zu Loch, ohne durch den Film umzulenken; die
Mündung je Loch in Serie (wie in 1D/2D) würde diesen Durchgang sperren.
Sie sitzt deshalb auf den **Filmflächen** um jede Mündung: jede Fläche
zwischen Mündungs- und Filmzelle bekommt die Zusatzlänge Δ = z_ff/Z′_f,
gewichtet mit W/(2π·r) (W die Summe der Flächenbreiten um die Mündung
auf der Treppe), so dass alle Flächen zusammen genau z_ff tragen.
Geprüft (`_MUENDUNG_ZWISCHEN_SERIE` schaltet zum Vergleich die
Serienform ein):

- **fluchtend** (Grenzfall der Gegenprobe 22a, 5 µm): Flächenform
  0,13 % vom einteiligen Löser, Serienform 68 % — der Grenzfall trennt
  beide scharf;
- **versetzt** (alle Luft durch den Film): Flächen- und Serienform auf
  0,5 % gleich (fein 0,8 %) bei 7–14 % Wirkung der Mündung. Der Rest
  ist die Verteilung selbst: auf Flächen zählt jede ihren Fluss im
  Quadrat, Σ z_k·q_k² ≥ z_ff·Q² (Cauchy-Schwarz, Gleichheit bei
  gleichmäßigem Abfluss); die Flächenform wirkt in allen Größen etwas
  stärker (`gp70.flaeche_zu_serie_versetzt` 0,07 der Wirkung).

**Wirkung.** K67 (2D, 1 kHz, 180°) −26,4 → −21,7 dB
(`gp07.niere_60v`; 20 V −26,8 → −22,2 dB), interne/externe Laufzeit
1,075 → 1,175 (`gp17.k67_verhaeltnis`; mit 45-µm-Spacer 1,344 →
1,459), Präsenzanhebung 4,9 → 5,2 dB, Empfindlichkeit 19,9 →
19,1 mV/Pa. Im 3D-Löser wird die Abhängigkeit der Auslöschung von der
Kernlage flacher (s. „Geometriefragen“). K103 und die übrigen
Doppelmembran-Proben ändern sich um ≤ 0,2 %, Einmembran-Kapseln ohne
Spacer gar nicht.

**Bewusst geändert (Entscheidung des Projekts):** Gegenprobe 30 prüft
weiter das Minimum bei 180°, die Tiefe ist Stand-Wert statt
„< −25 dB“. Gegenprobe 22e prüft weiter, DASS die Auslöschung von der
Kernlage abhängt: 12° muss jetzt 3 dB statt 8 dB tiefer liegen als
vollständig versetzt — ein Vielfaches der Gitterunsicherheit (grob/fein
≤ 0,5 dB); gemessen sind 7,9 dB (`gp22e.abstand_12grad_versetzt`). Beide
alten Grenzen hielten nur ohne die Umlenkung im Zwischenspalt.

### K67-Bohrbild aus dem Foto und Freistich in der Elektrostatik (Gegenprobe 71)

**Vermessung.** Ein Foto zweier K67-Backplate-Hälften (Messing, 16
Flanschbohrungen je Seite, davon 14 zum Spannen der Membran) zeigt das
Bohrbild, eine Ringnut am Rand der Elektrodenfläche und eine lochfreie,
abgesetzte Mitte mit einem kleinen Mittelloch. Eine veröffentlichte
Zeichnung gibt es nicht; die Literatur nennt nur das Raster (12 × 12,
2 mm, Zeilen 6, 8, 10, 12, …, 6) und dass jedes zweite Loch durchgebohrt
und die zweite Hälfte um 90° verdreht ist. Auf dem Foto wurden die
Lochmitten automatisch erkannt und ein 2-mm-Raster samt Perspektive
(Homographie) angepasst: Restfehler median 0,04 mm, beide Hälften
gleich. Der Maßstab aus der Teilung ist unabhängig bestätigt — Senkungen
1,28/1,39 mm (Projekt 1,3 mm), Plattenaußen-Ø 34,3 mm (K67: 34 mm).
Gemessen:

| Merkmal | linke / rechte Hälfte | im Modell |
|---|---|---|
| fehlende Rasterpunkte in der Mitte | 12 / 12 (r = 1,41 und 3,16 mm) | 108 Senkungen, 54 durch |
| Mittenaussparung Ø | 4,7 / 5,2 mm | 5,0 mm |
| Mittelloch Ø | ~1,0 mm | Mittenterminierung 1,0 mm (Annahme) |
| Elektrodenfläche Ø (Nut innen) | 25,8 / 25,1 mm | 25,4 mm |
| Nut außen Ø | 27,2 / 27,0 mm | Membran 27,2 mm |

**Einbettung.** Das Raster geht als 13 Lochkreise nach Radius ein; je
Radius ist genau die Hälfte durchgebohrt (die Spiegelung a → −a kehrt
die Schachbrett-Parität um), beide Hälften haben also dieselben Listen,
und um 90° gedreht fluchtet kein einziger Kern. Im 3D-Löser ändert die
exakte Rasterlage (Gegenhälfte global um 90° gedreht, Fußabdrücke auf
0,07 mm an den Rasterpunkten) die Auslöschung gegenüber gleichverteilten
Kreisen um 0,1 dB — ein eigenes Raster-Bohrbild braucht das Modell
nicht. Die Randnut ist der tiefe Ringraum zwischen Plattenrand und
Einspannung, den das Modell schon kennt (Gegenprobe 52). Die Mitte trägt
die Mittenterminierung (die K67 ist mittenterminiert) und einen
Freistich von r = 0,5 bis 2,5 mm, 0,2 mm tief (Tiefe nicht messbar;
0,05/0,5 mm ändern die 3D-Auslöschung um ≤ 0,2 dB).

**Freistich in der Elektrostatik (Gegenprobe 71).** Bis hier sah nur der
Film einen Freistich; Kraft, Erweichung, Wandlerkoeffizient und C0
rechneten dort mit dem vollen Feld des Spalts h — für eine Aussparung,
über der fast kein Feld steht, falsch. Jetzt gilt der örtliche Spalt
h + t (über Blindlöchern h + t + d) in Statik, Pull-in, Elektroden-
integralen und dem 3D-Ausgangsgewicht. Geprüft:
- **Grenzfall:** ein Freistich über die ganze Platte ist elektrostatisch
  ein größerer Spalt — C0, Pull-in, Θ, Nachgiebigkeit und Ruhe-
  auslenkung gleich auf 7·10⁻¹⁴;
- **örtlich:** ein schmaler Ring senkt C0 um ε0·A·c·(1/g − 1/(g + t))
  (1308 gegen 1331 fF, 1,7 % bei 5 % Abtastgrenze der Ringkanten);
- **Richtung:** der Freistich nimmt Feld weg, der Pull-in steigt
  (`gp71.pullin_anstieg_ring`).

Ohne Freistich ändert sich nichts. Mit Freistich: Debenham (Rand-
Freistich) interne Resonanz 3062 → 3016 Hz (`gp18.f_h_debenham`),
Laufzeitverhältnis 2D 5,40 → 5,33, 3D 2,56 → 2,88 (`gp17.debenham_3d`);
die B&K 4134 mit geschlossener Ringnut trifft die Phase von Fig. 6
besser (10,1° → 9,0°, `gp58.rms_4134_ringnut_grad`).

**Lochkreise beim Laden (Gegenprobe 63 d).** Der Projekt-Lader übernahm
höchstens 8 Lochkreise je Lochtyp und verwarf den Rest still — das
K67-Raster kam mit 32 statt 54 Durchgangslöchern an (2D-Auslöschung
−11,6 statt −13,4 dB für das Raster allein). Jetzt sind 16 Kreise
erlaubt, und ein Kürzen meldet der Lader als Warnung.

**Ergebnis bei 1 kHz** (tiefste Stelle, jeweils bei 180°):

| Schritt | 2D | 3D |
|---|---|---|
| `u87_k67_projekt.json` (gleichverteilt 120/60) | −21,7 dB | −14,6 dB |
| + reales Raster (108/54, Mitte frei) | −13,4 dB | |
| + Randnut (Elektrode 25,4, Membran 27,2 mm) | −14,2 dB | |
| + Mittenterminierung 1 mm | −17,8 dB | |
| + Mittenaussparung = `k67_experimentell_bohrbild.json` | −17,4 dB | −13,6 dB (fein −14,2 dB) |

Die 2D-Werte nach dem Raster liegen außerhalb der Homogenisierung (das
Modell warnt: lochfreie Mitte bis 4,3 mm); maßgeblich ist 3D. Dort ändert
das reale Bohrbild die Auslöschung gegenüber der gleichverteilten Basis
nur um rund 1 dB. Die publizierten −26 dB erreicht keine der beiden
Rechnungen; das Laufzeitverhältnis intern/extern ist 1,49 (s. „Offene
Punkte“ 9).

## Verlustmechanismen (vollständig erfasst)

Neben Zwikker–Kosten-Rohrreibung und Škvor-Spaltfilm rechnet das
Modell: **viskose Mündungswiderstände** (Sampson/Roscoe-Kriechströmung,
Weissberg-Zusatzlänge 3πr/16 je Mündung, thermoviskos ausgewertet —
bei kurzen engen Bohrungen vergleichbar mit dem Rohrwiderstand selbst;
nur an Mündungen in große Volumina), die **Spalt-Mündung** an jeder
Öffnung in den Membranfilm (Umlenkung Film → Bohrung als Dreitor aus
der instationären Stokes-Zelle, Widerstand und Masse, Gegenprobe 67),
**Sacklöcher/Senkungen als
endseitig geschlossene thermoviskose Leitungsstubs** (verteilte
Reibung, LF-Grenzfall R/3, Nachgiebigkeit isotherm→adiabatisch mit
Relaxationsdämpfung, λ/4-Verhalten) und die **laterale Trägheit der
Spaltluft** im 1D-Modell (Schlitz-Zwikker–Kosten-Korrektur Φ(ω),
identisch zum Filmleitwert des 2D-Feldmodells). Alle Grenzfälle sind
im Testlauf verifiziert (Gegenprobe 12). Dazu der **Folienverlust** der
Membran: hysteretisch, durch die Vorspannung auf die Biegeenergie
verdünnt, in 2D und 3D gleich (Gegenprobe 61).

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
  (seit dem Massenfaktor 8/j₀₁², Gegenprobe 55, trifft die Kette sie;
  der Rayleigh-Wert 4/3 lag ~1,9 % darüber).

`examples/debenham_stereo_condenser.json` — Braunmühl-Weber-Kapsel aus
Debenham/Robinson/Stebbings, *A Stereo Condenser Microphone* (Hi-Fi
News): einteilige durchbohrte Mittelelektrode (`center_gap = 0`), 1"-
Membranen, je Seite 12 Durchgangs- + 46 Dämpfungslöcher auf den echten
Lochkreisen der Konstruktionszeichnung (0.860/0.688/0.516/0.344/0.172"),
50 V — und der **Clearance-Ring** der Zeichnung: ein
Stirnflächen-Freistich am Elektrodenrand (0,038 mm Abtrag über die
äußeren 1,27 mm, GUI-Felder „Clearance-Ring", Klasse
`clearance_ring_*`). Er liegt nur über dem äußeren Lochkreis: die
sechs Durchgangslöcher dort werden entlastet, die sechs inneren bleiben
im 38-µm-Spalt verengt und begrenzen den Nieren-Phasenschieber. Seit
Gegenprobe 60 rechnet das 2D-Modell das örtlich und ist sich darin mit
dem 3D-Löser einig: die Niere bleibt flach (180°: −2,8 / −3,1 / −3,3 /
−3,9 dB bei 100 / 250 / 1000 / 2000 Hz, Laufzeitverhältnis 3,5), die
gemessene HF-Bündelung trifft es weiter (10 kHz: Null 143° vs. 142°).
Der Artikel zeigt eine tiefe Niere (Fig. 9, 100 Hz: −1/−3/−10/−12 dB
@ 45/90/135/180°). Bis Gegenprobe 59 traf das 2D-Modell sie fast exakt
(−1,2/−4,8/−10,2/−12,7) — aber nur, weil ein Schalter ALLEN
Durchgangslöchern die entlastete Engstelle gab, sobald eines im
Freistich lag (s. „Offene Punkte" 6). Das Beispiel legt die Stufenbohrung ausdrücklich ab
(`th_stepped: false`): bis zur Korrektur des Projekt-Laders erbte es
sie von der K67-Voreinstellung, 12 der 46 Sacklöcher wurden zu
Senkungen, und das Minimum lag bei 250 Hz–1 kHz auf 145–153°. Der
strengere
**3D-Löser** ordnet das ein: mit *nur* dem Rand-Freistich der Zeichnung
bleibt die Niere flach (−4 dB @ 1 kHz — die inneren Lochmündungen
bleiben verengt); deckt der Freistich dagegen alle Lochkreise ab (im
GUI-Clearance-Ring einstellbar, physikalisch ≈ angesenkte/entgratete
Mündungen), wird sie breitbandig tief (−13…−15 dB @ 250 Hz–2 kHz, Null
exakt 180°; 2D bei 500 Hz −22 dB). Die reale Kapsel dürfte solche
Mündungs-Fasen haben (in Zeichnungen selten bemaßt); das Beispiel
bleibt bei der Zeichnung, statt die Fasen an die Messung anzupassen.

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

`examples/bk4134_comsol_geometrie.json` und
`examples/bk4134_zuckerwar_1978.json` — das **B&K 4134**
(½-Zoll-Druckmikrofon), genau so, wie es die Gegenproben rechnen. Es
sind zwei Dateien, weil die Referenzen zwei verschiedene Kapseln
beschreiben:

- *COMSOL-Geometrie* (`_BK4134_COMSOL`, Gegenproben 58 e, 59, 61): die
  Originalgeometrie aus dem COMSOL-Anwendungsmodell (Spalt 18,6 µm,
  Platte 1,029 mm, Lochkreis 3,4 mm, offener Ring 0,86 × 0,30 mm zur
  Rückkammer, 200 V). Gegen die FEM: 2D 0,31 dB, 3D 0,06 dB RMS über
  1–20 kHz; gegen B&Ks Messmittel 3D 0,37 dB RMS (Aktuatorlast nicht
  modelliert, s. „Offene Punkte“ 8).
- *Zuckerwar 1978, Tab. I* (`_BK38["4134"]`, Gegenproben 38, 52, 54,
  58, 62): der Prüfling seiner Messung (Spalt 20,77 µm, Platte
  0,843 mm, Lochkreis 4,064 mm, Randschlitz 0,838 × 0,3048 mm, 28 V).
  Gegen Fig. 6: 2D 2,2 dB / 7,6° RMS — der Prüfling von 1978 war
  stärker gedämpft (s. Gegenproben 58, 59).

Beide geben die **Vakuumresonanz** vor (22 664,9 bzw. 22 953,7 Hz)
statt der Spannung: die Literaturspannung ist membranäquivalent aus
der gemessenen Resonanz bestimmt, die Randschicht der Biegesteifigkeit
steckt also schon darin (Gegenprobe 62). Gerechnet wird der
**Druckfrequenzgang ohne Beugung** wie in FEM und Messung. Mit
„Gehäuse & Beugung“ rechnen beide den Freifeldgang eines ½-Zoll-Stabs
im BEM (flache Stirnfläche ⌀13,2 mm, 30 mm, ohne Körper); er liegt um
die Freifeldkorrektur über dem Druckgang (+4,5 dB bei 10 kHz, +8,3 dB
bei 20 kHz; gemessen +4,0 / +8,0 dB, Gegenprobe 64), der gestrichelt
zum Vergleich mit COMSOL stehen bleibt. Der Druckgang ist ohne
Strahlungslast, wie in der FEM (Gegenprobe 65; zuschaltbar unter
„Gehäuse & Beugung“). Werkstoff ist
das GUI-Nickel (COMSOL: 8900 kg/m³, 221 GPa; Unterschied < 0,01 dB).
Gegen die Testmodelle weichen die geladenen Dateien höchstens
0,007 dB ab. Für die 0,84–0,86 mm breiten Randspalte reicht das
GUI-Feld „Randspalt" bis 2000 µm. Bis `b795a8a` endete es bei 500 µm,
und Streamlit setzte den Randspalt beim Laden still auf 0 — der
Frequenzgang fiel dann ab ~700 Hz (s. Gegenprobe 63).

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
