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
  nachzubilden. Bei der
  spiegelsymmetrischen Doppelmembran-Bauform kürzt sich die radiale
  Anordnung heraus — das Feldmodell zeigt dort die zugrunde liegende
  Acht; die im 1D-Modell entstehende Niere hängt an dessen Leiter-
  Topologie. Reziprok und passiv (im Test geprüft). Braucht SciPy.

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
zwei innenliegende Backplates mit 50-µm-Spacer, 60 V, Membranresonanz
≈ 1,15 kHz. Die passive Rückmembran bildet das Phasenschiebernetzwerk,
und das Lochbild ist als echte **Stufenbohrung** erfasst: 120 weite
Bohrungen je Seite, jede zweite mit konzentrischem Durchbruch am Grund
(GUI-Schalter „Stufenbohrung", Klasse `through_holes_stepped` — enges
Rohr nur über die Restdicke, Senkung als Sackvolumen, korrekte
Stirnporosität, eine Škvor-Senke je Bohrung). Validierung gegen
Herstellerdaten: Niere (−26 dB @ 180°/1 kHz), flacher Frequenzgang
100 Hz–5 kHz, Präsenzanhebung ≈ +3 dB bei ~12 kHz (die U87Ai-Elektronik
glättet den Kapselpeak auf die veröffentlichten ~+2 dB).
Über „Projekt laden" importierbar.

`examples/debenham_stereo_condenser.json` — Braunmühl-Weber-Kapsel aus
Debenham/Robinson/Stebbings, *A Stereo Condenser Microphone* (Hi-Fi
News): einteilige durchbohrte Mittelelektrode (`center_gap = 0`), 1"-
Membranen, je Seite 12 Durchgangs- + 46 Dämpfungslöcher auf den echten
Lochkreisen der Konstruktionszeichnung (0.860/0.688/0.516/0.344/0.172"),
50 V. Validiert gegen die im Artikel gemessenen Richtdiagramme (Fig. 9)
und den Frequenzgang (Fig. 10). Dieser Parametersatz ist zugleich die
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
