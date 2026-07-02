# Capsim — Simulation einer Kondensatormikrofonkapsel

Lumped-Element-Simulation (elektroakustisches Ersatzschaltbild) einer
Kondensatormikrofonkapsel mit Streamlit-Oberfläche.

## Komponenten

| Datei | Inhalt |
|---|---|
| `microphone_capsule.py` | Physik-Klasse `MicrophoneCapsule` (ABCD-Kettenmatrizen, Zwikker–Kosten-Lochimpedanzen, Škvor-Squeeze-Film, elektrostatische Wandlung) — eigenständig lauffähig mit Testlauf |
| `app.py` | Streamlit-GUI: Parameter-Seitenleiste, Bode-Plot, Polardiagramm, Projekt speichern/laden (JSON), CSV-Export |

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
Parametersatz einer Neumann-K67/K870-Kapsel (U87Ai, Nierenmodus): 26-mm-
Membran (6 µm Mylar), 60 µm Spalt, 60 V, Membranresonanz ≈ 1,15 kHz.
Über „Projekt laden" in der Seitenleiste importierbar. Das rückseitige
Netzwerk ist eine Ersatz-Näherung für die passive Rückmembran der
Doppelmembran-Kapsel.

## Projektdateien

Projekte werden als menschenlesbares JSON gespeichert
(`capsim_projekt.json`) und können über die Seitenleiste wieder geladen
werden. Die CSV-Exporte (Frequenzgang, Richtdiagramm) sind wahlweise im
internationalen Format (Komma/Punkt) oder Excel-DE-Format
(Semikolon/Dezimalkomma) verfügbar.
