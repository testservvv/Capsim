# -*- coding: utf-8 -*-
"""
Capsim — Streamlit-GUI für die Kondensatormikrofonkapsel-Simulation
====================================================================

Schritt 2 des Projekts: grafische Oberfläche auf Basis der Physik-Klasse
:class:`microphone_capsule.MicrophoneCapsule` (Schritt 1).

- Seitenleiste: alle Kapselparameter, direkt editierbar, mit den in
  Schritt 1 ermittelten Voreinstellungen als Startwerte
- Hauptbereich: Bode-Plot (Amplitude + Phase) und Polardiagramm,
  aktualisieren sich bei jeder Parameteränderung
- Projekte speichern/laden als JSON (menschenlesbar, versioniert,
  diff-freundlich — daher das geeignetste Format)
- CSV-Export der berechneten Daten (Frequenzgang und Richtdiagramm)

Start:  streamlit run app.py
"""

import datetime as _dt
import glob
import io
import json
import math
import os


def _cgroup_pfade(proc, regler_name):
    """Die cgroup des Prozesses für den Regler ``regler_name`` und alle
    darüber (v2: Zeile ohne Regler, v1: Zeile mit dem Regler)."""
    pfade = {"/"}
    try:
        with open(proc) as fh:
            for zeile in fh:
                _, regler, pfad = zeile.strip().split(":", 2)
                if regler == "" or regler_name in regler.split(","):
                    while True:
                        pfade.add(pfad or "/")
                        if pfad in ("", "/"):
                            break
                        pfad = os.path.dirname(pfad)
    except (OSError, ValueError):
        pass
    return pfade


def _erlaubte_kerne(wurzel="/sys/fs/cgroup", proc="/proc/self/cgroup"):
    """CPU-Quote des Prozesses in Kernen, None ohne Quote. In der Cloud
    sieht die App mehr Kerne, als sie nutzen darf. Gelesen aus der cgroup
    des Prozesses und allen darüber (v2: cpu.max, v1: cpu.cfs_quota_us /
    cpu.cfs_period_us); es gilt die engste."""
    pfade = _cgroup_pfade(proc, "cpu")
    quoten = []
    for pfad in pfade:
        rel = pfad.lstrip("/")
        try:                                             # cgroup v2
            with open(os.path.join(wurzel, rel, "cpu.max")) as fh:
                quote, periode = fh.read().split()[:2]
            if quote != "max":
                quoten.append(int(quote) / int(periode))
        except (OSError, ValueError):
            pass
        for regler in ("cpu", "cpu,cpuacct"):            # cgroup v1
            ordner = os.path.join(wurzel, regler, rel)
            try:
                with open(os.path.join(ordner, "cpu.cfs_quota_us")) as fq, \
                        open(os.path.join(ordner, "cpu.cfs_period_us")) as fp:
                    quote, periode = int(fq.read()), int(fp.read())
                if quote > 0:
                    quoten.append(quote / periode)
            except (OSError, ValueError):
                pass
    return min(quoten) if quoten else None


def _freier_speicher(wurzel="/sys/fs/cgroup", proc="/proc/self/cgroup",
                     meminfo="/proc/meminfo"):
    """Freier Arbeitsspeicher in Byte, den der Prozess noch belegen darf
    (Gegenprobe 75): das Kleinste aus MemAvailable und, für die cgroup des
    Prozesses und alle darüber, Grenze minus Belegung (v2: memory.max /
    memory.current, v1: memory.limit_in_bytes / memory.usage_in_bytes).
    None, wenn nichts lesbar ist."""
    frei = []
    try:
        with open(meminfo) as fh:
            for zeile in fh:
                if zeile.startswith("MemAvailable:"):
                    frei.append(int(zeile.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    for pfad in _cgroup_pfade(proc, "memory"):
        rel = pfad.lstrip("/")
        for ordner, grenze, belegt in (
                (os.path.join(wurzel, rel), "memory.max", "memory.current"),
                (os.path.join(wurzel, "memory", rel),
                 "memory.limit_in_bytes", "memory.usage_in_bytes")):
            try:
                with open(os.path.join(ordner, grenze)) as fg, \
                        open(os.path.join(ordner, belegt)) as fb:
                    g, b = fg.read().strip(), int(fb.read())
                if g != "max" and int(g) < 1 << 60:   # v1: "keine Grenze"
                    frei.append(int(g) - b)
            except (OSError, ValueError):
                pass
    return max(0, min(frei)) if frei else None


def _thread_vorgabe(kerne, sichtbar):
    """BLAS-Threads für eine CPU-Quote ``kerne`` bei ``sichtbar`` Kernen:
    None = Bibliotheksstandard lassen."""
    if kerne is None or kerne >= sichtbar:
        return None
    return max(1, int(kerne))


# RECHEN-THREADS NACH DER CPU-QUOTE (Gegenprobe 63 i). Darf der Prozess
# weniger rechnen, als er Kerne sieht (Streamlit Cloud: höchstens 2), dann
# treten sich die BLAS-Threads gegenseitig auf die Füße: gemessen brauchte
# eine Frequenz im feinen 3D bei einem erlaubten von vier sichtbaren Kernen
# 47,8 s mit vier Threads und 15,8 s mit einem (bei 0,5 Kernen 120,8 gegen
# 33,9 s). OHNE Quote bleibt der Bibliotheksstandard: dort bringt die
# Drossel nichts (13,0 gegen 15,2 s), und vier gleichzeitige Sitzungen im
# selben Prozess rechneten mit einem Thread sogar langsamer (25 gegen
# 11 s). Eine gesetzte Umgebungsvariable gewinnt immer. Wirkt nur vor dem
# ersten numpy-Import; ist numpy schon geladen, drosselt threadpoolctl
# (falls installiert) die laufenden Bibliotheken.
_THREADS = _thread_vorgabe(_erlaubte_kerne(), os.cpu_count() or 1)
if _THREADS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, str(_THREADS))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit.runtime.scriptrunner import get_script_run_ctx

from microphone_capsule import MicrophoneCapsule, ParameterFehler
from translations import TR, LABEL_TR

if "OMP_NUM_THREADS" in os.environ:
    try:
        from threadpoolctl import threadpool_limits as _threadpool_limits
        _threadpool_limits(limits=int(os.environ["OMP_NUM_THREADS"]))
    except Exception:       # ohne threadpoolctl gilt nur die Umgebung oben
        pass

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
_BEISPIEL_ORDNER = os.path.join(os.path.dirname(__file__), "examples")

# ---------------------------------------------------------------------------
# Sprache (GUI-Einstellung, KEIN Kapselparameter — bleibt außerhalb der
# Projektdateien). Standard: Englisch; umschaltbar in der Seitenleiste.
# ---------------------------------------------------------------------------
def _lang():
    try:
        return st.session_state.get("ui_lang", "en")
    except Exception:      # ohne Streamlit-Runtime (bare Tests)
        return "en"


def tr(schluessel, /, **kw):
    """Übersetzten GUI-Text holen; Platzhalter per str.format füllen. Der
    Schlüssel ist rein positionell: Texte wie err_value haben selbst einen
    Platzhalter {key} — mit dem alten Parameternamen ``key`` scheiterte
    genau diese Fehlermeldung an einem TypeError."""
    txt = TR[schluessel][_lang()]
    return txt.format(**kw) if kw else txt


def meldung(exc):
    """Fehlermeldung des Modells in der Sprache der Oberfläche: ein
    ParameterFehler trägt seine Meldung zweisprachig (s.
    microphone_capsule._MELDUNGEN), jeder andere Fehler zeigt seinen
    Text wie gemeldet."""
    if isinstance(exc, ParameterFehler):
        return exc.text(_lang())
    return str(exc)


def tr_label(canonical):
    """Anzeige eines kanonischen Auswahl-Werts in der aktuellen Sprache."""
    return LABEL_TR.get(canonical, {}).get(_lang(), canonical)


def _label_formatter():
    """format_func für Auswahl-Widgets mit EINGEFRORENER Sprache.

    Streamlit ruft format_func auch außerhalb des Skriptlaufs auf (Serde,
    AppTest-Serialisierung); dort ist die Sprachwahl nicht zugreifbar und
    ein dynamisches tr_label() fiele auf die falsche Sprache zurück. Die
    zur Renderzeit gültige Sprache gehört fest zu den in diesem Lauf
    erzeugten Optionen.
    """
    lang = _lang()
    return lambda c: LABEL_TR.get(c, {}).get(lang, c)


# ---------------------------------------------------------------------------
# Seiten-Setup & Designsprache
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=tr("page_title"),
    page_icon=_LOGO_PATH,
    layout="wide",
    # "auto": am Rechner offen, am Handy zugeklappt (sonst verdeckt die
    # Seitenleiste dort den ganzen Inhalt)
    initial_sidebar_state="auto",
)
# Logo oben links im Hauptbereich und in der Seitenleiste (ersetzt das
# frühere Emoji-Icon); dasselbe Bild dient auch als eingeklapptes Icon.
st.logo(_LOGO_PATH, icon_image=_LOGO_PATH, size="large")

# Farbrollen (validierte Referenzpalette, hell)
INK = "#0b0b0b"          # primäre Schrift
INK_2 = "#52514e"        # sekundäre Schrift
MUTED = "#898781"        # Achsen-/Tickbeschriftung
GRID = "#e1e0d9"         # Haarlinien-Raster
AXIS = "#c3c2b7"         # Achsenlinie
SURFACE = "#fcfcfb"      # Diagrammfläche
# Kategoriale Serienfarben in fester Reihenfolge (nie rotieren)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
      /* Nur die Entwickler-/Deploy-Bedienelemente ausblenden — NICHT die
         ganze Toolbar, denn darin sitzt der Pfeil zum Wiedereinblenden der
         Seitenleiste. Ohne diese Ausnahme ließe sich die einmal einge-
         klappte Seitenleiste auf Mobil/Cloud nicht mehr öffnen. */
      [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
      #MainMenu, footer { display: none !important; }
      [data-testid="stExpandSidebarButton"],
      [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important; opacity: 1 !important;
      }
      h1 { font-weight: 650; letter-spacing: -0.02em; }
      [data-testid="stSidebar"] h2 { font-size: 1.0rem; }
      [data-testid="stMetric"] {
        background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
        border-radius: 10px; padding: 0.6rem 0.9rem;
      }
      [data-testid="stMetricLabel"] { color: #52514e; }
      /* Kennzahl nie abschneiden ("20.9 mV…" auf Laptopbreite): kleiner
         setzen und umbrechen lassen */
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
      [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] *,
      [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {
        white-space: normal; overflow: visible; text-overflow: clip;
      }
      @media (max-width: 1150px) {
        [data-testid="stMetric"] { padding: 0.5rem 0.55rem; }
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        [data-testid="stMetricDelta"] { font-size: 0.75rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Parameterschema
# ---------------------------------------------------------------------------
# Anzeigeeinheiten sind bedienfreundlich (mm/µm); die Umrechnung nach SI
# geschieht zentral in build_capsule(). "key" ist zugleich der Feldname im
# Projekt-JSON. Voreinstellungen = die in Schritt 1 ermittelten Dummy-Werte.
MATERIAL_LABELS = {
    "PET (Mylar)": "pet",
    "Nickel": "nickel",
    "Titan": "titan",
    "Aluminium": "aluminium",
    "Gold": "gold",
}
ARCH_LABELS = {
    "Single Backplate": "single",
    "Dual Symmetrical Backplates": "dual",
    "Doppelmembran (K67-Bauform)": "dual_diaphragm",
}
K67_LABEL = "Doppelmembran (K67-Bauform)"
POS_LABELS = {"Umfang": "circumference", "Ende (Stirnfläche)": "end"}
AX_LABELS = {"Kugel (d_ext, montiert)": "sphere",
             "Sphäroid (freie Scheibe)": "spheroid",
             "BEM (Kopf + Körper)": "bem"}
# Position des rückwärtigen Gewebes (kanonische Werte wie die übrigen
# Auswahl-Widgets; Anzeige übersetzt LABEL_TR)
FAB_POS_LABELS = {"An der Backplate": "backplate",
                  "Über den Einlassöffnungen": "inlet"}

DIRECTIVITY_OPTIONS = [50, 100, 125, 250, 500, 1000, 2000, 4000,
                       5000, 8000, 10000, 12500, 16000, 20000]

# Fehlwerte beim Projekt-Laden: fehlt einem Projekt ein Schlüssel, gilt
# dieser Wert (s. _projekt_params). Das ist die Voreinstellung der App bis
# zur 3D-K67 (identisch zu examples/u87_k67_projekt.json, 2D, d_ext-Kugel)
# und bleibt eingefroren: ältere Projekte kennen z. B. Mittenterminierung,
# BEM-Körper oder 3D-Schalter nicht und meinten deren damaligen Wert —
# die heutige Voreinstellung (3D, freier Kopf, Pfosten 1 mm) würde sie
# still zu einer anderen Kapsel machen.
_PROJEKT_FEHLWERTE = {
    # Membran
    "material": "PET (Mylar)",
    "use_f_res": True,
    "f_res_hz": 1150.0,
    "mem_diameter_mm": 26.0,
    "mem_thickness_um": 6.0,
    "mem_tension_npm": 13.7,
    "mem_modes": 1,
    "modal_source": False,
    # Mittenterminierung (Kontaktstift/Bolzen); 0 = randgespannte
    # Kreismembran. Die K67 hat eine — ihr Wert ist Sache des Projekts.
    "center_post_mm": 0.0,
    # Backplate
    "air_gap_um": 65.0,
    "bp_diameter_mm": 25.0,
    "bp_thickness_mm": 4.0,
    "bias_v": 60.0,
    "architecture": "Doppelmembran (K67-Bauform)",
    "center_gap_um": 50.0,
    # Lochmuster als Lochkreis-Listen [Anzahl, Lochkreis-Ø in mm];
    # Lochkreis-Ø 0 = gleichmäßig verteilt. K67 (Stufenbohrungs-
    # Zählweise): 120 Sacklöcher 1.3 x 3.7 mm GESAMT, davon 60 mit
    # 0.6-mm-Kern durchgebohrt.
    "d_through_mm": 0.6,
    "th_rings": [[60, 0.0]],
    "th_stepped": True,
    "d_blind_mm": 1.3,
    "blind_depth_mm": 3.7,
    "bh_rings": [[120, 0.0]],
    # Klemmringe vor den Membranen (nur K67-Bauform); 0 = keine.
    # K67: je ~2 mm dick, Außen-Ø 34 mm -> 4 mm breit.
    "clamp_ring_mm": 2.0,
    "clamp_width_mm": 4.0,
    # Clearance-Ring (Stirnflächen-Freistich): bei der K67 keiner.
    "clr_dia_mm": 0.0,
    "clr_width_mm": 0.0,
    "clr_depth_mm": 0.0,
    "ring_vent_um": 0.0,
    "ring_vent_len_mm": 0.0,
    # Rückseite / akustische Netzwerke (bei K67-Bauform inaktiv)
    "rear_enabled": True,
    # Spacer + massive gelochte Rückplatte (K103-Bauform); 0 = nicht vorhanden
    "spacer_um": 0.0,
    "rearplate_mm": 0.0,
    "n_rearplate": 0,
    "d_rearplate_mm": 1.0,
    "delay_mm": 1.0,
    "cavity_length_mm": 5.0,
    "cavity_wall_mm": 1.5,
    "hole_position": "Umfang",
    "n_cavity": 200,
    "d_cavity_mm": 0.25,
    "cavity_axial_mm": 2.5,
    "fabric_front_rayl": 0.0,
    "fabric_rear_rayl": 0.0,
    # Gewebe-Position hinten: an der Backplate (im Zylinder, Bestand)
    # oder außen über den Einlassöffnungen (nur Lochfläche durchströmt,
    # hinter den Shunt-Volumina — Gegenprobe 24; nicht bei K67-Bauform)
    "fabric_rear_pos": "An der Backplate",
    # Gehäuse & Beugung (34 mm = Kapselkopf-Außen-Ø inkl. Klemmring)
    "diffraction_on": True,
    # Strahlungslast der Membran im Druckgang (ohne Beugung, dichte
    # Rückseite): Standard aus wie COMSOL/Kuppler/Aktuator (Gegenprobe 65)
    "rad_load": False,
    "body_diameter_mm": 34.0,
    # Axialer Körper für den Front-Rück-Transfer der Doppelmembran:
    # Kugel (d_ext) = montierte Kapsel (Standard); Sphäroid = frei
    # stehende Scheibe (Referenzfall, Gegenprobe 20); BEM = montagetreue
    # Kontur Kopf + Mikrofonkörper (Gegenprobe 21).
    "axial_body": "Kugel (d_ext, montiert)",
    # Axiale Körperlänge — nur für BEM bei EIN-Membran-Bauformen
    # (Druckempfänger); die Doppelmembran nimmt dort d_ext.
    "body_length_mm": 12.0,
    "bem_body_dia_mm": 56.0,
    "bem_body_gap_mm": 15.0,
    "bem_body_len_mm": 80.0,
    # Spaltfilm-Modell (Debenham braucht 2D für die tiefe Niere;
    # 3D = diskrete Löcher, alle Architekturen: dual_diaphragm mit
    # center_gap > 0 als K67-Modus (Gegenprobe 22), single/dual über
    # Sammelknoten + Lumped-Rückbaugruppe (Gegenprobe 23))
    "squeeze_2d": True,
    "squeeze_3d": False,
    # Verdrehung der Elektrodenhälften (nur 3D-K67-Modus): automatisch =
    # halbe Teilung des Durchgangs-Lochbilds (180°/n_th, reale K67)
    "half_rot_auto": True,
    "half_rot_deg": 3.0,
    # Auflösung des 3D-Gitters: grob (Standard) oder fein (Gegenprobe 50;
    # ein Vielfaches an Rechenzeit je Frequenzpunkt)
    "grid_3d_fine": False,
    # Simulation
    "n_points": 400,
    "normalize_1khz": True,
    "dir_freqs": [125, 1000, 4000, 8000, 16000],
}

# (key, Widget-Art, Default) — Gruppen s. Sidebar-Aufbau weiter unten
# Voreinstellung beim Start: die experimentelle K67 (U87 Ai, Nierenmodus)
# mit dem Bohrbild aus dem Foto und 45 µm wirksamem Zwischenspalt, im
# FEINEN 3D-Feldmodell mit dem freien Kopf als Außenweg — identisch zu
# examples/k67_experimentell_zwischenspalt45.json (dort Maße, Quellen und
# die ANPASSUNG: 45 statt 40 µm ist kein Messwert), nur mit 3D. Gerechnet
# ~1,5 h; die App setzt deshalb gespeicherte Ergebnisse ein, solange
# Modell und Parameter dazu passen (voreinstellung.py, Gegenprobe 63 f).
# Die übrigen Kapseln liegen unter examples/.
DEFAULTS = {
    **_PROJEKT_FEHLWERTE,
    "f_res_hz": 1700.0,
    "mem_diameter_mm": 27.2,
    "center_post_mm": 1.0,
    "air_gap_um": 50.0,
    "bp_diameter_mm": 25.4,
    "bp_thickness_mm": 4.07,
    "center_gap_um": 45.0,
    # 2-mm-Raster der K67 exakt (s. Projektdatei, Gegenprobe 73): je
    # Zeile ein 180°-Lochpaar mit dem Winkel des ersten Lochs; 54 Kerne
    # mit Senkung, dazu 54 rein blinde Senkungen 1.3 x 3.75 mm
    "th_rings": [[2, 8.485, 45.0], [2, 10.198, 11.31], [2, 10.198, 78.69],
                 [2, 11.662, 120.96], [2, 11.662, 149.04], [2, 14.142, 45.0],
                 [2, 14.142, 98.13], [2, 14.142, 171.87], [2, 15.232, 23.2],
                 [2, 15.232, 66.8], [2, 17.205, 125.54], [2, 17.205, 144.46],
                 [2, 18.111, 6.34], [2, 18.111, 83.66], [2, 18.974, 108.43],
                 [2, 18.974, 161.57], [2, 19.799, 45.0], [2, 20.591, 29.05],
                 [2, 20.591, 60.95], [2, 22.091, 95.19], [2, 22.091, 174.81],
                 [2, 22.804, 15.26], [2, 22.804, 74.74], [2, 22.804, 127.87],
                 [2, 22.804, 142.13], [2, 24.166, 114.44],
                 [2, 24.166, 155.56]],
    "blind_depth_mm": 3.75,
    "bh_rings": [[2, 8.485, 135.0], [2, 10.198, 101.31], [2, 10.198, 168.69],
                 [2, 11.662, 30.96], [2, 11.662, 59.04], [2, 14.142, 8.13],
                 [2, 14.142, 81.87], [2, 14.142, 135.0], [2, 15.232, 113.2],
                 [2, 15.232, 156.8], [2, 17.205, 35.54], [2, 17.205, 54.46],
                 [2, 18.111, 96.34], [2, 18.111, 173.66], [2, 18.974, 18.43],
                 [2, 18.974, 71.57], [2, 19.799, 135.0], [2, 20.591, 119.05],
                 [2, 20.591, 150.95], [2, 22.091, 5.19], [2, 22.091, 84.81],
                 [2, 22.804, 37.87], [2, 22.804, 52.13], [2, 22.804, 105.26],
                 [2, 22.804, 164.74], [2, 24.166, 24.44], [2, 24.166, 65.56]],
    "clamp_width_mm": 3.4,
    "clr_dia_mm": 3.0,
    "clr_width_mm": 2.0,
    "clr_depth_mm": 0.2,
    "axial_body": "BEM (Kopf + Körper)",
    "bem_body_dia_mm": 0.0,
    "squeeze_3d": True,
    "grid_3d_fine": True,
    "dir_freqs": [125, 250, 500, 1000, 2000, 4000],
}

_FLOAT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, float)}
_INT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, int)
             and not isinstance(v, bool)}
_BOOL_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, bool)}

# Lochkreis-Listen: Projekt-/DEFAULTS-Schlüssel -> Session-Key-Präfix.
# Die Ringzeilen leben als einzelne Widget-Keys p_<prefix>_ring_n_<i> /
# p_<prefix>_ring_pcd_<i> / p_<prefix>_ring_phi_<i> (Winkel, None = Lage-
# konvention) plus Zeilenzähler <prefix>_ring_count, weil Streamlit-
# Widgets nur skalare Zustände tragen.
_RING_PREFIX = {"th_rings": "th", "bh_rings": "bh"}
# Höchstzahl der Lochkreise je Lochtyp. Das quadratische 2-mm-Raster der
# K67 (12 × 12) liegt auf 13 Radien; bis Gegenprobe 71 waren es 8, und
# ein Projekt mit mehr Kreisen wurde beim Laden STILL gekürzt. Jetzt
# meldet der Lader jedes Kürzen (s. _projekt_params). Exakt (mit
# Winkeln, Gegenprobe 73) sind es 27 Lochpaare je Lochtyp.
MAX_RINGS = 32

# Fehlt ein Schlüssel im Projekt, gilt _PROJEKT_FEHLWERTE (die frühere
# 2D-K67). Für Merkmale, deren K67-Wert nicht der neutrale ist, wäre das
# falsch: ein Projekt ohne den Schlüssel stammt aus der Zeit vor dem
# Merkmal und kann es nicht gemeint haben. Ohne diese Regel bekam das
# Debenham-Beispiel beim Laden Stufenbohrungen (12 der 46 Sacklöcher
# wurden zu Senkungen der Durchgangslöcher).
_FEHLT_IM_PROJEKT = {"th_stepped": False}

# Eingabebereiche der Zahlenfelder — EINE Quelle für Widgets und Tests.
# Streamlit (1.64) setzt einen Session-Wert außerhalb [min, max] beim
# Aufbau des Zahlenfelds STILL auf das Minimum, ohne Fehler. Das
# B&K-4134-Beispiel (Randspalt 860 µm) lief bei einem Feldmaximum von
# 500 µm so mit geschlossenem Randspalt, und sein Frequenzgang fiel ab
# ~700 Hz statt flach bis 10 kHz zu laufen. _zahl() klemmt deshalb vorher
# auf die NÄCHSTE Grenze und meldet es; Gegenprobe 63 prüft alle
# Beispielprojekte gegen diese Tabelle.
_BEREICH = {
    "f_res_hz": (100.0, 50000.0),
    "mem_diameter_mm": (3.0, 60.0),
    "mem_thickness_um": (0.5, 100.0),
    "mem_tension_npm": (1.0, 5000.0),
    "mem_modes": (1, 5),
    "air_gap_um": (5.0, 500.0),
    "bp_diameter_mm": (2.0, 60.0),
    "bp_thickness_mm": (0.2, 20.0),
    "bias_v": (0.5, 400.0),
    "center_gap_um": (0.0, 500.0),
    "ring_n": (0, 2000),
    "d_through_mm": (0.05, 5.0),
    "d_blind_mm": (0.05, 5.0),
    "clr_dia_mm": (0.0, 60.0),
    "clr_width_mm": (0.0, 10.0),
    "clr_depth_mm": (0.0, 5.0),
    "ring_vent_um": (0.0, 2000.0),
    "ring_vent_len_mm": (0.0, 20.0),
    "clamp_ring_mm": (0.0, 10.0),
    "clamp_width_mm": (0.0, 10.0),
    "spacer_um": (0.0, 1000.0),
    "rearplate_mm": (0.0, 20.0),
    "n_rearplate": (0, 2000),
    "d_rearplate_mm": (0.05, 5.0),
    "delay_mm": (0.0, 100.0),
    "cavity_length_mm": (0.0, 100.0),
    "cavity_wall_mm": (0.0, 10.0),
    "n_cavity": (0, 5000),
    "d_cavity_mm": (0.0, 5.0),
    "fabric_front_rayl": (0.0, 100000.0),
    "fabric_rear_rayl": (0.0, 100000.0),
    "body_length_mm": (4.5, 200.0),
    "bem_body_dia_mm": (0.0, 200.0),
    "bem_body_gap_mm": (3.0, 100.0),
    "bem_body_len_mm": (10.0, 300.0),
    "half_rot_deg": (0.0, 180.0),
    "n_points": (100, 1500),
    "ring_phi_deg": (0.0, 360.0),
}


def _bereich(name, wert):
    """(min, max) des Zahlenfelds ``name``. Abhängige Grenzen lesen die
    übrigen Werte über ``wert(schlüssel)`` (Session oder Projekt)."""
    if name == "center_post_mm":          # Stift höchstens Membranradius
        return 0.0, 0.5 * wert("mem_diameter_mm")
    if name == "ring_pcd_mm":             # Lochkreis auf der Backplate
        return 0.0, wert("bp_diameter_mm")
    if name == "blind_depth_mm":          # Sackloch nicht durchgebohrt
        return 0.05, max(0.1, wert("bp_thickness_mm") - 0.1)
    if name == "cavity_axial_mm":         # Bohrung innerhalb des Zylinders
        return 0.0, wert("cavity_length_mm")
    if name == "body_diameter_mm":        # Körper umschließt die Kapsel
        return (max(wert("mem_diameter_mm"), wert("bp_diameter_mm")),
                100.0)
    return _BEREICH[name]


def _bereichsverletzungen(p):
    """Alle Werte eines Parametersatzes (kanonische Projektschlüssel)
    außerhalb ihres Feldbereichs: Liste (Feld, Wert, min, max). Jeder
    Zahlenschlüssel braucht einen Bereich (sonst KeyError)."""
    aus = []

    def pruefe(feld, name, val):
        lo, hi = _bereich(name, p.__getitem__)
        if not lo <= val <= hi:
            aus.append((feld, val, lo, hi))

    for key, val in p.items():
        if key in _RING_PREFIX:
            for i, zeile in enumerate(val):
                pruefe(f"{key}[{i}].n", "ring_n", zeile[0])
                pruefe(f"{key}[{i}].pcd", "ring_pcd_mm", zeile[1])
                if len(zeile) > 2:
                    pruefe(f"{key}[{i}].phi", "ring_phi_deg", zeile[2])
        elif key in _FLOAT_KEYS or key in _INT_KEYS:
            pruefe(key, key, val)
    return aus


def _set_ring_state(prefix, rings):
    """Ringliste [[Anzahl, Lochkreis-Ø mm(, Winkel °)], ...] in die
    Widget-Session-Keys schreiben (Winkel None = Lagekonvention);
    überzählige Zeilen eines früheren Zustands werden entfernt."""
    old = st.session_state.get(f"{prefix}_ring_count", 0)
    for i in range(len(rings), old):
        for feld in ("n", "pcd", "phi"):
            st.session_state.pop(f"p_{prefix}_ring_{feld}_{i}", None)
    for i, zeile in enumerate(rings):
        st.session_state[f"p_{prefix}_ring_n_{i}"] = int(zeile[0])
        st.session_state[f"p_{prefix}_ring_pcd_{i}"] = float(zeile[1])
        phi = zeile[2] if len(zeile) > 2 else None
        st.session_state[f"p_{prefix}_ring_phi_{i}"] = (
            None if phi is None else float(phi))
    st.session_state[f"{prefix}_ring_count"] = len(rings)


def _ringzeile(cnt, pcd, phi=None):
    """Kanonische Ringzeile: [Anzahl, Ø] ohne Winkel (wie in Projekten vor
    Gegenprobe 73), [Anzahl, Ø, Winkel] mit. Ein Winkel gilt nur für einen
    Lochkreis; bei Ø 0 (gleichmäßig verteilt) fällt er weg — das Feld ist
    dann gesperrt."""
    zeile = [int(cnt), float(pcd)]
    if phi is not None and float(pcd) > 0.0:
        zeile.append(float(phi) % 360.0)
    return zeile


def _rings_from_state(prefix):
    n = st.session_state.get(f"{prefix}_ring_count", 1)
    return [_ringzeile(st.session_state[f"p_{prefix}_ring_n_{i}"],
                       st.session_state[f"p_{prefix}_ring_pcd_{i}"],
                       st.session_state.get(f"p_{prefix}_ring_phi_{i}"))
            for i in range(n)]


# Rückübersetzung Anzeigetext -> kanonischer Wert (alle Sprachen). Nötig
# zur Selbstheilung nach einem Sprachwechsel: Radio/Selectbox senden vom
# Browser den ANZEIGETEXT der zuletzt gerenderten Sprache zurück; findet
# Streamlits Serde ihn in den aktuellen Optionen nicht, schreibt er ihn
# WÖRTLICH in den Session-State (accept_new_options-Verhalten) — dann
# stünde z. B. "Dual diaphragm (K67 design)" statt des kanonischen Werts
# in p_architecture.
_DISPLAY_TO_CANON = {disp: canon for canon, langs in LABEL_TR.items()
                     for disp in langs.values()}
_CSV_TO_CANON = {TR["csv_intl"][lg]: "intl" for lg in ("en", "de")}
_CSV_TO_CANON.update({TR["csv_excel_de"][lg]: "excel_de"
                      for lg in ("en", "de")})


def _heal_canonical_state():
    """Fremdsprachige Anzeigetexte im Session-State auf kanonische Werte
    zurückübersetzen — VOR dem Widget-Aufbau aufrufen (Sprachwechsel)."""
    for key, valid, dflt in (
            ("p_material", MATERIAL_LABELS, DEFAULTS["material"]),
            ("p_architecture", ARCH_LABELS, DEFAULTS["architecture"]),
            ("p_hole_position", POS_LABELS, DEFAULTS["hole_position"]),
            ("p_axial_body", AX_LABELS, DEFAULTS["axial_body"]),
            ("p_fabric_rear_pos", FAB_POS_LABELS,
             DEFAULTS["fabric_rear_pos"])):
        v = st.session_state.get(key)
        if v is not None and v not in valid:
            st.session_state[key] = _DISPLAY_TO_CANON.get(v, dflt)
    v = st.session_state.get("csv_format")
    if v is not None and v not in ("intl", "excel_de"):
        st.session_state["csv_format"] = _CSV_TO_CANON.get(v, "intl")
    v = st.session_state.get("ui_lang")
    if v is not None and v not in ("en", "de"):
        st.session_state["ui_lang"] = {"English": "en",
                                       "Deutsch": "de"}.get(v, "en")


def _init_state():
    st.session_state.setdefault("ui_lang", "en")   # Standard: Englisch
    st.session_state.setdefault("projekt_notizen", "")
    for key, val in DEFAULTS.items():
        if key in _RING_PREFIX:
            if f"{_RING_PREFIX[key]}_ring_count" not in st.session_state:
                _set_ring_state(_RING_PREFIX[key], val)
        else:
            st.session_state.setdefault("p_" + key, val)


def _halte_ausgeblendete():
    """Ausgeblendete Felder behalten ihren Wert (Gegenprobe 63 e).

    Streamlit räumt den Zustand eines Widgets ab, das in einem Lauf nicht
    gezeichnet wird; _init_state setzte dann die Voreinstellung ein. So
    kam nach BEM → Kugel → BEM der Körper Ø 56 mm zurück, obwohl das
    Projekt den freien Kopf (0 mm) wollte — mit drei Feldern und einem
    vielfach teureren BEM. Zu Beginn jedes Laufs zurückgeschrieben, sind
    die Werte Sitzungswerte und überstehen das Aufräumen (dokumentierter
    Weg, gilt für alle unterstützten Streamlit-Versionen)."""
    for key in DEFAULTS:
        if key not in _RING_PREFIX and "p_" + key in st.session_state:
            st.session_state["p_" + key] = st.session_state["p_" + key]


def _wirksam(p):
    """Was die Oberfläche bei der gewählten Bauform AUSBLENDET, wirkt nicht
    (Gegenprobe 63 g). Seit ausgeblendete Felder ihren Wert behalten
    (Gegenprobe 63 e), blieb ein bei 'Single' gesetzter Randspalt nach dem
    Wechsel zur K67-Bauform stehen: das Modell lehnte ihn ab, das Feld war
    aber nicht mehr zu sehen — eine Sackgasse. Der Wert bleibt in der
    Sitzung (für den Rückweg), geht aber nicht ins Modell und nicht in die
    gespeicherte Projektdatei."""
    p = dict(p)
    if p["architecture"] == K67_LABEL:
        p["ring_vent_um"] = 0.0
        p["ring_vent_len_mm"] = 0.0
    return p


def _current_params():
    return _wirksam({k: (_rings_from_state(_RING_PREFIX[k])
                         if k in _RING_PREFIX
                         else st.session_state["p_" + k])
                     for k in DEFAULTS})


def _coerce(key, val):
    """Robuste Typkonvertierung beim Projekt-Laden."""
    def fehler():
        return ValueError(tr("err_value", key=key, val=repr(val)))
    if key in _RING_PREFIX:
        rings = []
        for r in list(val)[:MAX_RINGS]:
            if not 2 <= len(r) <= 3:
                raise fehler()
            phi = r[2] if len(r) > 2 else None
            if phi is not None and not math.isfinite(float(phi)):
                raise fehler()
            rings.append(_ringzeile(max(0, int(r[0])), max(0.0, float(r[1])),
                                    phi))
        if not all(math.isfinite(r[1]) for r in rings):
            raise fehler()
        return rings or [[0, 0.0]]
    if key in _BOOL_KEYS:
        # nur echte Wahrheitswerte: bool("false") wäre True
        if isinstance(val, bool) or val in (0, 1):
            return bool(val)
        raise fehler()
    if key in _FLOAT_KEYS:
        v = float(val)
        if not math.isfinite(v):            # json liest auch NaN/Infinity
            raise fehler()
        return v
    if key in _INT_KEYS:
        if isinstance(val, float) and not math.isfinite(val):
            raise fehler()
        return int(val)
    if key == "dir_freqs":
        return [f for f in DIRECTIVITY_OPTIONS if f in set(int(x) for x in val)]
    if key == "material" and val in MATERIAL_LABELS:
        return val
    if key == "axial_body" and val in AX_LABELS:
        return val
    if key == "architecture" and val in ARCH_LABELS:
        return val
    if key == "hole_position" and val in POS_LABELS:
        return val
    if key == "fabric_rear_pos" and val in FAB_POS_LABELS:
        return val
    raise fehler()


# Schlüssel des alten Projektformats (Version 1), s. _projekt_params
_ALT_SCHLUESSEL = ("n_through", "th_pcd_mm", "n_blind", "bh_pcd_mm")
PROJEKT_VERSION = 2


def _projekt_params(data):
    """Projekt-JSON (dict) -> (vollständiger Parametersatz, Anzahl der
    übernommenen Projektwerte, gekürzte Lochkreis-Listen als
    [(Schlüssel, Anzahl in der Datei)], unbekannte Schlüssel der Datei).
    ValueError bei defekter Datei — auch bei einer Datei ganz ohne
    Parameter: sie überschrieb bis Gegenprobe 63 h still alles mit den
    Fehlwerten und meldete Erfolg."""
    if not isinstance(data, dict) or data.get("format") != "capsim-project":
        raise ValueError(tr("err_format"))
    params_in = data.get("params")
    if not isinstance(params_in, dict) or not params_in:
        raise ValueError(tr("err_no_params"))
    # Unbekannte Schlüssel (Tippfehler, neuere Version) verwirft der Lader
    # — aber nie still: ein vertippter Luftspalt lud sonst als Fehlwert.
    unbekannt = sorted(k for k in params_in
                       if k not in DEFAULTS and k not in _ALT_SCHLUESSEL)
    gekuerzt = [(key, len(params_in[key])) for key in _RING_PREFIX
                if isinstance(params_in.get(key), list)
                and len(params_in[key]) > MAX_RINGS]
    # Erst alles validieren (staged), dann atomar anwenden — eine
    # defekte Datei lässt den aktuellen Zustand unangetastet.
    staged = {key: _coerce(key, val) for key, val in params_in.items()
              if key in DEFAULTS}
    # Altes Projektformat (Version 1, EIN Lochkreis je Lochtyp):
    # n_through/th_pcd_mm bzw. n_blind/bh_pcd_mm -> eine Ringzeile.
    if "th_rings" not in staged and "n_through" in params_in:
        staged["th_rings"] = [[int(params_in["n_through"]),
                               float(params_in.get("th_pcd_mm", 0.0))]]
    if "bh_rings" not in staged and "n_blind" in params_in:
        staged["bh_rings"] = [[int(params_in["n_blind"]),
                               float(params_in.get("bh_pcd_mm", 0.0))]]
    # Ein Projekt beschreibt die KOMPLETTE Kapsel: im Projekt nicht
    # enthaltene Parameter fallen auf die eingefrorenen Fehlwerte zurück
    # (ältere Projekte kennen z. B. Spacer/Rückplatte noch nicht), NICHT
    # auf die heutige Voreinstellung (s. _PROJEKT_FEHLWERTE).
    return ({key: staged.get(key, _FEHLT_IM_PROJEKT.get(
                key, _PROJEKT_FEHLWERTE[key]))
             for key in DEFAULTS}, len(staged), gekuerzt, unbekannt)


def _notizen_aus(data):
    """Notizen eines Projekts (Feld ``_kommentar``: Text oder Zeilenliste)
    als Text. Bis Gegenprobe 63 h gingen sie beim Laden und erneuten
    Speichern verloren."""
    k = data.get("_kommentar", "")
    if isinstance(k, (list, tuple)):
        return "\n".join(str(z) for z in k)
    return str(k)


def _projekt_json(params, notizen=""):
    """Projektdatei als JSON-Text (Version 2: Lochmuster als Lochkreis-
    Listen th_rings/bh_rings; Version-1-Projekte mit n_through/th_pcd_mm
    … werden beim Laden konvertiert). Notizen als Zeilenliste, wie in den
    Beispielprojekten."""
    data = {"format": "capsim-project", "version": PROJEKT_VERSION,
            "saved": _dt.date.today().isoformat()}
    if notizen.strip():
        data["_kommentar"] = notizen.rstrip().split("\n")
    data["params"] = params
    return json.dumps(data, indent=2, ensure_ascii=False)


def _projekt_anwenden(data):
    """Projekt in den Session-State schreiben (Callback-Kontext: läuft vor
    dem Widget-Aufbau). Werte außerhalb ihres Feldbereichs klemmt und
    meldet danach _zahl(). Rückgabe: (Art, Meldung)."""
    params, n, gekuerzt, unbekannt = _projekt_params(data)
    for key, val in params.items():
        if key in _RING_PREFIX:
            _set_ring_state(_RING_PREFIX[key], val)
        else:
            st.session_state["p_" + key] = val
    st.session_state["projekt_notizen"] = _notizen_aus(data)
    msg = tr("load_ok", n=n)
    # gekürzte Lochkreis-Listen, unbekannte Schlüssel und eine neuere
    # Formatversion nie still übergehen
    for key, n_datei in gekuerzt:
        msg += " " + tr("load_rings_cut", max=MAX_RINGS, n=n_datei,
                        typ=tr("md_through" if key == "th_rings"
                               else "md_blind"))
    if unbekannt:
        msg += " " + tr("load_unknown", keys=", ".join(unbekannt))
    version = data.get("version")
    neuer = isinstance(version, (int, float)) and version > PROJEKT_VERSION
    if neuer:
        msg += " " + tr("load_newer", v=version, eigene=PROJEKT_VERSION)
    return ("warning" if (gekuerzt or unbekannt or neuer) else "success",
            msg)


def _load_project():
    """on_change-Callback des Uploaders."""
    up = st.session_state.get("project_upload")
    if up is None:
        return
    try:
        st.session_state["_load_msg"] = _projekt_anwenden(json.load(up))
    except Exception as exc:  # defekte Datei darf die App nicht stoppen
        st.session_state["_load_msg"] = (
            "error", tr("load_fail", exc=exc))


def _beispiele():
    """Die mitgelieferten Beispielprojekte (Dateinamen). In der Cloud sind
    sie nur so erreichbar — der Uploader sieht nur den Rechner des
    Nutzers."""
    return sorted(os.path.basename(f) for f in
                  glob.glob(os.path.join(_BEISPIEL_ORDNER, "*.json")))


def _load_example():
    """on_click-Callback: gewähltes Beispielprojekt laden."""
    name = st.session_state.get("beispiel_wahl")
    if name not in _beispiele():
        return
    try:
        with open(os.path.join(_BEISPIEL_ORDNER, name),
                  encoding="utf-8") as fh:
            st.session_state["_load_msg"] = _projekt_anwenden(json.load(fh))
    except Exception as exc:
        st.session_state["_load_msg"] = (
            "error", tr("load_fail", exc=exc))


def _restore_defaults():
    """on_click-Callback: zurück zur Voreinstellung (deren Ergebnisse
    gespeichert vorliegen, s. _voreinstellung_einsetzen). Bis Gegenprobe
    63 h ging das nur über ein Neuladen der Seite."""
    for key, val in DEFAULTS.items():
        if key in _RING_PREFIX:
            _set_ring_state(_RING_PREFIX[key], val)
        else:
            st.session_state["p_" + key] = (list(val)
                                            if isinstance(val, list)
                                            else val)
    st.session_state["projekt_notizen"] = ""
    st.session_state["_confirm_reset"] = False
    st.session_state["_load_msg"] = ("success", tr("default_done"))


# Null-Zustand für den Reset-Button: 0 überall, wo die Widgets 0 zulassen,
# sonst der KLEINSTE zulässige Widget-Wert (die Kapsel bleibt so baubar
# und die App rechenfähig — s. _zero_all_params für die zwei Modell-
# optionen, die dafür mitgehen). Auswahl- (Material, Architektur,
# Position), Modell- (1D/2D, Beugung, Verdrehung) und Simulations-
# einstellungen (Frequenzpunkte, Normierung, Sprache) bleiben unberührt.
_ZERO_STATE = {
    # air_gap 50 statt Widget-Minimum 5 µm: mit einer weichen Membran
    # läge die Pull-in-Spannung sonst unter dem kleinsten Bias (0.5 V) —
    # der Null-Zustand muss BAUBAR bleiben. f_res 1000 statt 100 Hz: bei
    # 100 Hz ist die 3-mm-Folie so schwach gespannt, dass ihre Randschicht
    # über ein Viertel der Membran reicht; das Modell bricht dann seit
    # Gegenprobe 62 ab, und der Null-Reset rechnete seither für KEINE
    # Bauform mehr (Gegenprobe 63 g).
    "f_res_hz": 1000.0, "mem_diameter_mm": 3.0, "mem_thickness_um": 0.5,
    "mem_tension_npm": 1.0, "mem_modes": 1, "modal_source": False,
    "center_post_mm": 0.0,
    "air_gap_um": 50.0,
    "bp_diameter_mm": 2.0,
    "bp_thickness_mm": 0.2, "bias_v": 0.5, "center_gap_um": 0.0,
    "d_through_mm": 0.05, "th_stepped": False, "d_blind_mm": 0.05,
    "blind_depth_mm": 0.05, "clamp_ring_mm": 0.0, "clamp_width_mm": 0.0,
    "clr_dia_mm": 0.0, "clr_width_mm": 0.0, "clr_depth_mm": 0.0,
    "ring_vent_um": 0.0, "ring_vent_len_mm": 0.0,
    "rear_enabled": False, "spacer_um": 0.0, "rearplate_mm": 0.0,
    "n_rearplate": 0, "d_rearplate_mm": 0.05, "delay_mm": 0.0,
    "cavity_length_mm": 0.0, "cavity_wall_mm": 0.0, "n_cavity": 0,
    "d_cavity_mm": 0.0, "cavity_axial_mm": 0.0,
    "fabric_front_rayl": 0.0, "fabric_rear_rayl": 0.0,
    "body_diameter_mm": 3.0,
}


def _zero_all_params():
    """Bestätigter Null-Reset (on_click-Callback: läuft VOR dem
    Widget-Aufbau und darf deren Session-Keys setzen)."""
    for key, val in _ZERO_STATE.items():
        st.session_state["p_" + key] = val
    # Die Gegentakt-Bauform braucht Löcher in der vorderen Backplate,
    # sonst erreicht kein Schall die Membran: dort bleibt EIN Loch.
    dual = ARCH_LABELS.get(st.session_state["p_architecture"]) == "dual"
    _set_ring_state("th", [[1 if dual else 0, 0.0]])
    _set_ring_state("bh", [[0, 0.0]])
    # Ohne Löcher gibt es kein 3D-Feldmodell, und der fast leere Kopf ist
    # für die BEM-Kontur zu dünn: 3D aus, Kugel als Körper. (Bis
    # Gegenprobe 63 g blieben beide stehen; seit die Voreinstellung 3D mit
    # BEM ist, endete der Reset in einer Fehlermeldung.)
    st.session_state["p_squeeze_3d"] = False
    st.session_state["p_axial_body"] = "Kugel (d_ext, montiert)"
    st.session_state["projekt_notizen"] = ""
    st.session_state["_confirm_reset"] = False
    st.session_state["_load_msg"] = ("success", tr("reset_done"))


def _ask_reset():
    st.session_state["_confirm_reset"] = True


def _cancel_reset():
    st.session_state["_confirm_reset"] = False


def _add_ring(prefix):
    """+‑Button: hängt einen weiteren (leeren) Lochkreis an."""
    n = st.session_state[f"{prefix}_ring_count"]
    if n >= MAX_RINGS:
        return
    st.session_state[f"p_{prefix}_ring_n_{n}"] = 0
    st.session_state[f"p_{prefix}_ring_pcd_{n}"] = 0.0
    st.session_state[f"p_{prefix}_ring_phi_{n}"] = None
    st.session_state[f"{prefix}_ring_count"] = n + 1


def _remove_ring(prefix):
    """−‑Button: entfernt den letzten Lochkreis (mindestens einer bleibt)."""
    n = st.session_state[f"{prefix}_ring_count"]
    if n <= 1:
        return
    for feld in ("n", "pcd", "phi"):
        st.session_state.pop(f"p_{prefix}_ring_{feld}_{n - 1}", None)
    st.session_state[f"{prefix}_ring_count"] = n - 1


# Meldungen der Bereichsklemmung dieses Laufs (Modul-Global: jeder Rerun
# führt das Skript neu aus und beginnt mit einer leeren Liste).
_KLEMM_MELDUNGEN = []


def _ss_wert(name):
    return st.session_state["p_" + name]


def _klemme(key, lo, hi, titel):
    """Session-Wert ``key`` vor dem Widget-Aufbau auf [lo, hi] klemmen
    (nächste Grenze, nicht Streamlits stilles Minimum) und melden."""
    val = st.session_state[key]
    if val is None:                      # leeres Feld (Winkel: Konvention)
        return
    neu = min(max(val, lo), hi)
    if neu != val:
        st.session_state[key] = type(val)(neu)
        _KLEMM_MELDUNGEN.append(tr("clamp_item", lbl=titel, alt=val,
                                   neu=neu, lo=lo, hi=hi))


def _zahl(name, label, ort=None, key=None, titel=None, **kw):
    """Zahlenfeld mit dem Bereich aus _bereich(); ein Wert außerhalb wird
    vorher geklemmt und im Hauptbereich gemeldet (s. _BEREICH)."""
    key = key or "p_" + name
    lo, hi = _bereich(name, _ss_wert)
    _klemme(key, lo, hi, titel or label)
    return (ort or st).number_input(label, lo, hi, key=key, **kw)


def _ring_rows(prefix):
    """Dynamische Lochkreis-Zeilen eines Lochtyps (Anzahl + Lochkreis-Ø)
    mit ➕/➖-Buttons; gibt die Gesamt-Lochzahl zurück."""
    n_rings = st.session_state[f"{prefix}_ring_count"]
    # Anzahl schmaler, damit Winkel bis 359.9° nicht abgeschnitten werden
    breiten = [0.8, 1.1, 1.1]
    h1, h2, h3 = st.columns(breiten)
    h1.caption(tr("cap_ring_n"))
    h2.caption(tr("cap_ring_pcd"))
    h3.caption(tr("cap_ring_phi"), help=tr("help_ring_phi"))
    art = tr("md_through" if prefix == "th" else "md_blind")
    total = 0
    for i in range(n_rings):
        c1, c2, c3 = st.columns(breiten)
        # Beschriftung nur für Screenreader (kompakte Tabellenoptik;
        # die sichtbare Kopfzeile liefern die Captions darüber). Schrumpft
        # die Backplate, klemmt _zahl den Lochkreis auf ihren Durchmesser.
        _zahl("ring_n", tr("lbl_ring_n", i=i + 1), ort=c1,
              key=f"p_{prefix}_ring_n_{i}",
              titel=f"{art} · {tr('lbl_ring_n', i=i + 1)}",
              step=1, label_visibility="collapsed")
        _zahl("ring_pcd_mm", tr("lbl_ring_pcd", i=i + 1), ort=c2,
              key=f"p_{prefix}_ring_pcd_{i}",
              titel=f"{art} · {tr('lbl_ring_pcd', i=i + 1)}",
              step=0.5, label_visibility="collapsed")
        # Winkel des ersten Lochs (Gegenprobe 73); leer = Lagekonvention,
        # gesperrt bei Ø 0 (gleichmäßig verteilt, dort ohne Bedeutung)
        st.session_state.setdefault(f"p_{prefix}_ring_phi_{i}", None)
        _zahl("ring_phi_deg", tr("lbl_ring_phi", i=i + 1), ort=c3,
              key=f"p_{prefix}_ring_phi_{i}",
              titel=f"{art} · {tr('lbl_ring_phi', i=i + 1)}",
              step=1.0, format="%.1f", placeholder=tr("ph_ring_phi"),
              disabled=st.session_state[f"p_{prefix}_ring_pcd_{i}"] <= 0.0,
              label_visibility="collapsed")
        total += st.session_state[f"p_{prefix}_ring_n_{i}"]
    b1, b2 = st.columns(2)
    b1.button(tr("btn_ring_add"), key=f"btn_add_{prefix}",
              on_click=_add_ring, args=(prefix,),
              disabled=n_rings >= MAX_RINGS, width="stretch",
              help=tr("help_ring_add"))
    b2.button(tr("btn_ring_del"), key=f"btn_del_{prefix}",
              on_click=_remove_ring, args=(prefix,),
              disabled=n_rings <= 1, width="stretch",
              help=tr("help_ring_del"))
    if n_rings > 1:
        st.caption(tr("cap_ring_total", total=total, n=n_rings))
    return total


def _modell_ring(zeile):
    """Ringzeile der Oberfläche [Anzahl, Ø mm(, Winkel °)] -> Modell-Eintrag
    (Anzahl, Ø m oder None(, Winkel))."""
    n, d = int(zeile[0]), float(zeile[1])
    if d <= 0.0:
        return (n, None)
    if len(zeile) > 2 and zeile[2] is not None:
        return (n, d * 1e-3, float(zeile[2]))
    return (n, d * 1e-3)


def _capsule_kwargs(p):
    """Anzeigewerte -> SI: die Konstruktor-Argumente der Kapsel (reine
    JSON-Typen, damit Arbeiterprozesse sie bekommen, Gegenprobe 75)."""
    return dict(
        membrane_material=MATERIAL_LABELS[p["material"]],
        membrane_resonance_hz=p["f_res_hz"] if p["use_f_res"] else None,
        membrane_diameter=p["mem_diameter_mm"] * 1e-3,
        membrane_thickness=p["mem_thickness_um"] * 1e-6,
        membrane_tension=p["mem_tension_npm"],
        membrane_modes=int(p.get("mem_modes", 1)),
        modal_source=1 if p.get("modal_source", False) else 0,
        center_post_diameter=p.get("center_post_mm", 0.0) * 1e-3,
        air_gap=p["air_gap_um"] * 1e-6,
        backplate_diameter=p["bp_diameter_mm"] * 1e-3,
        backplate_thickness=p["bp_thickness_mm"] * 1e-3,
        bias_voltage=p["bias_v"],
        architecture=ARCH_LABELS[p["architecture"]],
        center_gap=p["center_gap_um"] * 1e-6,
        # Lochkreis-Ø 0 in der GUI = dieser Anteil gleichmäßig verteilt;
        # ein dritter Wert ist der Winkel des ersten Lochs (Gegenprobe 73)
        through_hole_rings=[_modell_ring(z) for z in p["th_rings"]],
        through_hole_diameter=p["d_through_mm"] * 1e-3,
        blind_hole_rings=[_modell_ring(z) for z in p["bh_rings"]],
        blind_hole_diameter=p["d_blind_mm"] * 1e-3,
        blind_hole_depth=p["blind_depth_mm"] * 1e-3,
        through_holes_stepped=p["th_stepped"],
        clamp_ring_thickness=p["clamp_ring_mm"] * 1e-3,
        clamp_ring_width=p["clamp_width_mm"] * 1e-3,
        clearance_ring_diameter=p["clr_dia_mm"] * 1e-3,
        clearance_ring_width=p["clr_width_mm"] * 1e-3,
        clearance_ring_depth=p["clr_depth_mm"] * 1e-3,
        ring_vent_width=p.get("ring_vent_um", 0.0) * 1e-6,
        ring_vent_length=(p.get("ring_vent_len_mm", 0.0) * 1e-3
                          if p.get("ring_vent_len_mm", 0.0) > 0.0
                          else None),
        # Rückseite deaktiviert -> keine rückwärtige Baugruppe: die
        # Durchgangslöcher der Backplate münden (durch das rückwärtige
        # Gewebe) direkt ins Schallfeld. Hermetisch dicht ist die Kapsel
        # nur bei 0 Durchgangslöchern — das entscheidet das Physikmodell.
        rear_network_enabled=p["rear_enabled"],
        rear_spacer_height=p["spacer_um"] * 1e-6,
        rear_plate_thickness=p["rearplate_mm"] * 1e-3,
        n_rear_plate_holes=p["n_rearplate"],
        rear_plate_hole_diameter=p["d_rearplate_mm"] * 1e-3,
        delay_length=p["delay_mm"] * 1e-3,
        cavity_length=p["cavity_length_mm"] * 1e-3,
        cavity_wall_thickness=p["cavity_wall_mm"] * 1e-3,
        cavity_hole_position=POS_LABELS[p["hole_position"]],
        n_cavity_holes=p["n_cavity"],
        cavity_hole_diameter=p["d_cavity_mm"] * 1e-3,
        cavity_hole_axial_position=p["cavity_axial_mm"] * 1e-3,
        fabric_front_rayl=p["fabric_front_rayl"],
        fabric_rear_rayl=p["fabric_rear_rayl"],
        # Doppelmembran hat keinen rückwärtigen Einlass — dort bleibt die
        # Position fest "backplate" (Gewebe über der Rückmembran), damit
        # ein Architekturwechsel nie am Gatter scheitert.
        fabric_rear_position=(
            FAB_POS_LABELS.get(p.get("fabric_rear_pos"), "backplate")
            if ARCH_LABELS[p["architecture"]] != "dual_diaphragm"
            else "backplate"),
        body_diameter=p["body_diameter_mm"] * 1e-3,
        # body_length gilt NUR für BEM bei Ein-Membran-Bauformen; bei der
        # Doppelmembran spannen die Membranen die Stirnflächen auf (d_ext),
        # dort weist das Physikmodell den Wert zurück.
        body_length=(p.get("body_length_mm", 12.0) * 1e-3
                     if (AX_LABELS.get(p.get("axial_body")) == "bem"
                         and ARCH_LABELS[p["architecture"]]
                         != "dual_diaphragm")
                     else None),
        include_diffraction=p["diffraction_on"],
        pressure_radiation_load=p.get("rad_load", False),
        axial_body_model=AX_LABELS.get(p.get("axial_body",
                                             "Kugel (d_ext, montiert)"),
                                       "sphere"),
        bem_body_diameter=p.get("bem_body_dia_mm", 56.0) * 1e-3,
        bem_body_gap=p.get("bem_body_gap_mm", 15.0) * 1e-3,
        bem_body_length=p.get("bem_body_len_mm", 80.0) * 1e-3,
        squeeze_model=("3d" if p.get("squeeze_3d") else
                       ("2d" if p["squeeze_2d"] else "1d")),
        # None = automatisch eine halbe Teilung (180°/n_th)
        half_rotation_deg=(None if p.get("half_rot_auto", True)
                           else p.get("half_rot_deg", 3.0)),
        grid_3d=("fine" if p.get("grid_3d_fine") else "coarse"),
    )


def build_capsule(p):
    """Anzeigewerte -> SI -> MicrophoneCapsule."""
    return MicrophoneCapsule(**_capsule_kwargs(p))


# ---------------------------------------------------------------------------
# Berechnung (gecacht) mit Fortschrittsanzeige
# ---------------------------------------------------------------------------
# Eigene Caches in st.session_state statt st.cache_data: (a) der
# Fortschritts-Callback zeichnet auf einen AUSSEN angelegten Platzhalter
# (st.empty), was Streamlits Cache-Replay beim Hit nicht wiederherstellen
# kann (CacheReplayClosureError); (b) Modul-Globals taugen nicht als
# Ablage, weil jeder Rerun das Skript — samt Initialisierung — neu
# ausführt. st.session_state überlebt Reruns genau zu diesem Zweck.
# Zwei Ebenen:
#   _capsule_cache — das MicrophoneCapsule-Objekt je Bau-Parametersatz.
#     Der Konstruktor rechnet Elektrostatik, Pull-in und (bei 3D) den
#     Gitter-/Geometrieaufbau — das darf nicht bei jeder Widget-
#     Interaktion erneut anfallen.
#   _results_cache — Frequenzgang/Richtdiagramme/Diagnose je vollem
#     Parametersatz (wichtig vor allem für das 3D-Modell mit einer
#     LU-Faktorisierung je Frequenzpunkt).
_RESULTS_CACHE_MAX = 24
_CAPSULE_CACHE_MAX = 6
_BARE_STORES = {}          # Fallback ohne Streamlit-Runtime (Tests)


def _session_store(name):
    try:
        return st.session_state.setdefault(name, {})
    except Exception:
        return _BARE_STORES.setdefault(name, {})


def get_capsule(params, progress=None):
    """Kapsel aus dem Session-Cache oder neu bauen (ValueError reicht durch).

    Schlüssel sind nur die BAU-Parameter (ohne n_points/dir_freqs/
    Normierung) — Simulationseinstellungen erzwingen keinen Neubau.
    """
    build_params = {k: v for k, v in params.items()
                    if k not in ("n_points", "dir_freqs", "normalize_1khz")}
    key = json.dumps(build_params, sort_keys=True)
    store = _session_store("_capsule_cache")
    capsule = store.get(key)
    if capsule is None:
        if progress is not None:
            progress(0.0, tr("prog_build"))
        capsule = build_capsule(params)
        store[key] = capsule
        while len(store) > _CAPSULE_CACHE_MAX:
            store.pop(next(iter(store)))
    return capsule


def _fr_raster(n_pts):
    """Frequenzraster des Frequenzgangs [Hz] (compute_results)."""
    return np.logspace(np.log10(10.0), np.log10(25000.0), int(n_pts))


def _frequenzplan(p):
    """Alle Frequenzen [Hz], die compute_results für den Parametersatz ``p``
    löst (Gegenprobe 75): das Raster des Frequenzgangs, die Richt-
    frequenzen und 1 kHz (Empfindlichkeit, Laufzeit, Summary). Fehlt hier
    eine, rechnet compute_results sie selbst — nur langsamer."""
    extra = [float(fd) for fd in (sorted(p["dir_freqs"]) or [1000])]
    return np.unique(np.concatenate([_fr_raster(p["n_points"]),
                                     np.array(extra + [1000.0])]))


# Spitzenspeicher eines Arbeiterprozesses im 3D-Modell [Byte]: gemessen an
# der K67 2,1 GB (feines Gitter, ~1 GB grob), mit Reserve
_SPEICHER_3D = {"fine": 2.5e9, "coarse": 1.2e9}
# Frequenzen je Auftrag an einen Arbeiter (Lastverteilung)
_AUFTRAG_3D = 2


def _prozesse_3d(capsule, kerne=None, frei=None):
    """Wie viele Arbeiterprozesse die 3D-Frequenzen rechnen dürfen
    (Gegenprobe 75): so viele, wie Kerne (CPU-Quote) UND freier Speicher
    (je Prozess _SPEICHER_3D, dazu 0,5 GB für den Hauptprozess) erlauben.
    In der Streamlit Cloud (höchstens 2 Kerne, 2,7 GB) bleibt es bei
    einem — dort rechnet die App wie bisher im eigenen Prozess."""
    if kerne is None:
        kerne = _erlaubte_kerne() or float(os.cpu_count() or 1)
    if frei is None:
        frei = _freier_speicher()
    if frei is None:
        return 1
    je = _SPEICHER_3D.get(getattr(capsule, "grid_3d", "fine"),
                          _SPEICHER_3D["fine"])
    return max(1, min(int(kerne), int((frei - 0.5e9) // je)))


def _vorrechnen_3d(params, capsule, progress=None, prozesse=None):
    """Die fehlenden 3D-Lösungen des Frequenzplans parallel in Arbeiter-
    prozessen rechnen und in die Kapsel einsetzen (Gegenprobe 75), bevor
    compute_results sie braucht. Nur für das unveränderte 3D-Modell (eine
    Probe mit ausgetauschten Methoden rechnet selbst) und nur, wenn sich
    mindestens zwei Prozesse lohnen. Rückgabe: Zahl der parallel
    gerechneten Frequenzen (0 = nichts, compute_results rechnet wie
    bisher) und der Prozesse. Scheitert ein Arbeiter, bleibt der Rest
    compute_results."""
    if (type(capsule) is not MicrophoneCapsule
            or capsule.squeeze_model != "3d"
            or any(callable(v) for v in vars(capsule).values())):
        return 0, 1
    vorhanden = capsule.loesungen_3d()
    fehlend = [float(f) for f in _frequenzplan(params)
               if float(2.0 * np.pi * f) not in vorhanden]
    w = prozesse if prozesse is not None else _prozesse_3d(capsule)
    if w < 2 or len(fehlend) < 2 * w:
        return 0, w
    import concurrent.futures as _cf
    import multiprocessing as _mp
    import microphone_capsule as _M
    bau = json.dumps(_capsule_kwargs(params), sort_keys=True)
    teile = [fehlend[i:i + _AUFTRAG_3D]
             for i in range(0, len(fehlend), _AUFTRAG_3D)]
    fertig = 0
    # Kein with-Block: dessen shutdown(wait=True) wartete bei einem Abbruch
    # (Streamlit bricht das Skript bei jeder neuen Eingabe mit einer
    # BaseException ab) auf ALLE noch wartenden Aufträge. Fertige Lösungen
    # sind dann schon in der Kapsel und bleiben für den nächsten Lauf.
    ex = _cf.ProcessPoolExecutor(max_workers=w,
                                 mp_context=_mp.get_context("spawn"),
                                 initializer=_M._arbeiter_init)
    sauber = False
    try:
        auftraege = [ex.submit(_M._arbeiter_loese_3d, bau, t) for t in teile]
        for fu in _cf.as_completed(auftraege):
            capsule.loesungen_3d_einsetzen(fu.result())
            fertig += 1
            if progress is not None:
                progress(fertig / len(teile),
                         tr("prog_par", n=len(fehlend), w=w))
        sauber = True
    except Exception:           # Arbeiter gescheitert: der Rest seriell
        pass
    finally:
        ex.shutdown(wait=sauber, cancel_futures=not sauber)
    vorhanden = capsule.loesungen_3d()
    return sum(1 for f in fehlend if float(2.0 * np.pi * f) in vorhanden), w


def compute_results(cache_key, capsule, progress=None):
    """Frequenzgang, Richtdiagramme, Diagnose und Summary — gecacht.

    ``cache_key`` ist das JSON der physikrelevanten Parameter. Der
    Frequenzgang wird blockweise gerechnet, damit ``progress`` (Callback
    frac, label) einen echten Fortschritt melden kann; die Blöcke sind
    numerisch identisch zum Gesamtaufruf, weil alle Modelle je Frequenz
    unabhängig rechnen (frequency_response-Logik gespiegelt). Auch der
    summary()-Text gehört hierher: er enthält eine Netzwerkauswertung
    und würde sonst bei JEDEM Rerun im Diagnose-Expander mitgerechnet —
    Streamlit führt auch eingeklappte Expander-Inhalte aus. (Im 3D-Modus
    ist seine 1-kHz-Lösung seit Gegenprobe 56 frei: sens_1k hat dieselbe
    Frequenz schon gelöst, und die Kapsel bewahrt ihre Feldlösungen auf.
    Deshalb kostet auch eine zusätzliche Richtfrequenz nur ihre eigene
    Lösung, solange die Kapsel im Session-Cache bleibt.)
    """
    store = _session_store("_results_cache")
    hit = store.get(cache_key)
    if hit is not None:
        return hit
    p = json.loads(cache_key)
    n_pts = int(p["n_points"])
    dir_freqs = sorted(p["dir_freqs"]) or [1000]
    n_work = n_pts + len(dir_freqs) + 1        # + Eigenrauschen
    done = 0

    def _tick(k, label):
        nonlocal done
        done += k
        if progress is not None:
            progress(min(done / n_work, 1.0), label)

    # Frequenzgang — Achse/Normierung identisch zu frequency_response();
    # sofortiger 0%-Tick, damit der Balken ab der ersten Sekunde steht.
    # angle_responses liefert je Frequenz in EINEM Netzwerk-/Feldaufbau
    # die Übertragung bei 0/90/180° UND die Rück-Übertragung D_r des
    # Phasenschiebers (Superposition q = a·p_front + b·p_rück).
    _tick(0, tr("prog_fr"))
    f = _fr_raster(n_pts)
    H = np.empty(n_pts, dtype=complex)
    H90 = np.empty(n_pts, dtype=complex)
    H180 = np.empty(n_pts, dtype=complex)
    G180 = np.empty(n_pts, dtype=complex)
    D_r = np.empty(n_pts, dtype=complex)
    has_dr = True
    chunk = 2 if capsule.squeeze_model == "3d" else 100
    for i in range(0, n_pts, chunk):
        j = min(i + chunk, n_pts)
        r = capsule.angle_responses(f[i:j])
        H[i:j] = r["H"][0.0]
        H90[i:j] = r["H"][90.0]
        H180[i:j] = r["H"][180.0]
        G180[i:j] = r["G180"]
        if r["D_r"] is None:
            has_dr = False
        else:
            D_r[i:j] = r["D_r"]
        _tick(j - i, tr("prog_fr"))
    amp_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-30))
    ref_db = np.interp(np.log10(1000.0), np.log10(f), amp_db)
    fr = {
        "frequency_hz": f,
        "sensitivity_v_pa": H,
        "amplitude_db": amp_db,
        "amplitude_db_norm": amp_db - ref_db,
        "phase_deg": np.rad2deg(np.unwrap(np.angle(H))),
    }
    # Mit Beugung zusätzlich der DRUCKFREQUENZGANG als Vergleichskurve —
    # das, was COMSOL, Kuppler und Aktuator liefern: ohne Körper und ohne
    # Strahlungslast (Gegenprobe 65), im 3D-Modell aus derselben
    # Feldlösung. Der Abstand beider Kurven ist die Freifeldkorrektur.
    # Gleiche Bezugsgröße (H bei 1 kHz) für beide.
    if (capsule.include_diffraction and not capsule.rear_open
            and capsule.architecture != "dual_diaphragm"):
        p_db = 20.0 * np.log10(np.maximum(
            np.abs(capsule.pressure_response(f)), 1e-30))
        fr["pressure_db"] = p_db
        fr["pressure_db_norm"] = p_db - ref_db
    habs = np.maximum(np.abs(H), 1e-30)
    aux = {
        "level_90_db": 20.0 * np.log10(np.maximum(np.abs(H90), 1e-30)
                                       / habs),
        "level_180_db": 20.0 * np.log10(np.maximum(np.abs(H180), 1e-30)
                                        / habs),
        "D_r": D_r if has_dr else None,
        "G180": G180,
        "f_helmholtz_hz": (MicrophoneCapsule.helmholtz_resonance_hz(f, D_r)
                           if has_dr else None),
    }
    # Richtdiagramme — je Frequenz ein Netzwerk-/Feldaufbau
    di = {"angles_deg": None, "patterns": {}}
    for fd in dir_freqs:
        d1 = capsule.directivity(frequencies_hz=(float(fd),))
        di["angles_deg"] = d1["angles_deg"]
        di["patterns"].update(d1["patterns"])
        _tick(1, tr("prog_di", f=fd))
    sens_1k = float(abs(capsule.transfer_function(np.array([1000.0]))[0]))
    delay = capsule.delay_diagnostics()
    # Eigenrauschen (thermisch-akustisch, FDT/Nyquist) — nur 1D/2D; der
    # 3D-Feldlöser hat keinen konzentrierten Membranzweig.
    noise = None
    if capsule.squeeze_model != "3d":
        _tick(0, tr("prog_noise"))
        try:
            # DASSELBE Raster fuer Spektrum und Kennzahl: noise_spectrum
            # ruft transfer_function, und im BEM-Modus steckt darin je
            # Frequenz ein Randelementsystem. Ein zweites Raster hat den
            # Lauf frueher verdoppelt — unsichtbar hinter dem 100 %-Balken.
            f_n = f              # dasselbe Raster wie der Frequenzgang
            spn = capsule.noise_spectrum(f_n)
            noise = {"self": capsule.self_noise(spectrum=spn),
                     "f": f_n, "asd": spn["asd_pa_shz"]}
        except Exception:            # Rauschen darf nie die App stoppen
            noise = None
    _tick(1, tr("prog_noise"))
    result = {"fr": fr, "di": di, "aux": aux, "sens_1k": sens_1k,
              "delay": delay, "noise": noise}
    store[cache_key] = result
    while len(store) > _RESULTS_CACHE_MAX:
        store.pop(next(iter(store)))
    return result


def _ergebnis_schluessel(params):
    """Cache-Schlüssel: alle Parameter, die Physik oder berechnete Daten
    ändern. Reine Darstellungsoptionen (Normierungs-Toggle) bleiben
    draußen, damit ihr Umschalten keine Neuberechnung auslöst. Die Typen
    folgen DEFAULTS: liefert ein Widget derselben Einstellung einmal 400
    und einmal 400.0 (je nach Streamlit-Version), bleibt der Schlüssel
    gleich — sonst träfe die gespeicherte Voreinstellung nicht."""
    key_params = {}
    for k, v in params.items():
        if k == "normalize_1khz":
            continue
        if k in _RING_PREFIX:
            v = [_ringzeile(*z) for z in v]
        elif k == "dir_freqs":
            v = sorted(int(f) for f in v)
        elif k in _BOOL_KEYS:
            v = bool(v)
        elif k in _FLOAT_KEYS:
            v = float(v)
        elif k in _INT_KEYS:
            v = int(v)
        key_params[k] = v
    return json.dumps(key_params, sort_keys=True)


def _kennwerte(capsule):
    """Die Kennwerte der Kapsel, die der Hauptbereich zeigt — mit der
    gespeicherten Voreinstellung abgelegt, damit der App-Start die Kapsel
    gar nicht bauen muss (3D fein: ~6 s und ~400 MB, in der gedrosselten
    Cloud ein Vielfaches)."""
    g3d = getattr(capsule, "_g3d", None) or {}
    return {
        "f_res": float(capsule.f_res),
        "f_res_from_tension": float(capsule.f_res_from_tension),
        "C_elec_0": float(capsule.C_elec_0),
        "n_bp": int(capsule.n_bp),
        "U_pullin": float(capsule.U_pullin),
        "softening_ratio": float(capsule.softening_ratio),
        "w0_static": float(capsule.w0_static),
        "h_min_static": float(capsule.h_min_static),
        "squeeze_model": capsule.squeeze_model,
        "architecture": capsule.architecture,
        "axial_body_model": capsule.axial_body_model,
        "include_diffraction": bool(capsule.include_diffraction),
        "rear_open": bool(capsule.rear_open),
        "d_ext": float(capsule.d_ext),
        "cavity_hole_position": capsule.cavity_hole_position,
        "_g3d": {"fine_capped": bool(g3d.get("fine_capped", False)),
                 "cells_rm": float(g3d.get("cells_rm", 0.0))},
        # Plausibilität der Lochlage (Gegenprobe 73)
        "loch_befund": {k: [int(v[0]), float(v[1])] for k, v in
                        (getattr(capsule, "loch_befund", None) or {}).items()},
    }


class _GespeicherteKapsel:
    """Stellvertreter der Kapsel für die gespeicherte Voreinstellung: die
    Kennwerte aus der Datei, die Klassenkonstanten vom Modell."""
    _F_BAND_TOP = MicrophoneCapsule._F_BAND_TOP
    _GRID_FINE_MAX = MicrophoneCapsule._GRID_FINE_MAX
    _GRID_FINE_CELLS = MicrophoneCapsule._GRID_FINE_CELLS
    _RING_REPR_DB = MicrophoneCapsule._RING_REPR_DB

    def __init__(self, kennwerte):
        self.__dict__.update(kennwerte)


# Was die gespeicherten Ergebnisse der Voreinstellung erzeugt hat — ihr
# Fingerabdruck enthält diese Funktionen (s. voreinstellung.py)
_BAU_FUNKTIONEN = (build_capsule, _capsule_kwargs, _modell_ring)
_VOREINSTELLUNG_FUNKTIONEN = _BAU_FUNKTIONEN + (compute_results, _kennwerte,
                                                _fr_raster)


def _physik_schluessel(cache_key):
    """Ergebnisschlüssel ohne die Simulationseinstellungen (Punktzahl,
    Richtfrequenzen): was die 3D-Lösung je Frequenz bestimmt."""
    p = json.loads(cache_key)
    for k in ("n_points", "dir_freqs"):
        p.pop(k, None)
    return json.dumps(p, sort_keys=True)


def _prozess_loesungen(cache_key, capsule, merken=False):
    """3D-Lösungen über Sitzungen hinweg (Gegenprobe 75): ein Neuladen der
    Seite oder eine zweite Sitzung mit derselben Kapsel findet die schon
    gelösten Frequenzen im Serverprozess wieder (voreinstellung.
    prozess_holen; Schlüssel: Physik-Schlüssel, Modellquelle und
    Baufunktionen). Bis hier lagen sie nur an der Kapsel der Sitzung.
    ``merken``: die Lösungen der Kapsel dort ablegen. Rückgabe: Zahl der
    eingesetzten Frequenzen."""
    if (type(capsule) is not MicrophoneCapsule
            or capsule.squeeze_model != "3d"):
        return 0
    import voreinstellung
    physik = _physik_schluessel(cache_key)
    abdruck = voreinstellung.fingerabdruck(physik, _BAU_FUNKTIONEN)
    if merken:
        voreinstellung.prozess_merken(abdruck, capsule.loesungen_3d())
        return 0
    return capsule.loesungen_3d_einsetzen(voreinstellung.prozess_holen(abdruck))


def _voreinstellung_loesungen(cache_key, capsule):
    """Gespeicherte 3D-Lösungen der Voreinstellung in die Kapsel einsetzen
    (Gegenprobe 75), wenn sie physikalisch die Voreinstellung ist — nur
    Punktzahl oder Richtfrequenzen anders — und Modell und Baufunktionen
    zum Fingerabdruck der Lösungen passen. Dann rechnet eine weitere
    Richtfrequenz nur sich selbst (~12 s) statt der ganzen Voreinstellung
    (~85 min). Rückgabe: Zahl der eingesetzten Frequenzen."""
    if (getattr(capsule, "squeeze_model", None) != "3d"
            or type(capsule) is not MicrophoneCapsule
            or getattr(capsule, "_voreinstellung_geprueft", False)):
        return 0
    capsule._voreinstellung_geprueft = True
    physik = _physik_schluessel(cache_key)
    if physik != _physik_schluessel(_ergebnis_schluessel(DEFAULTS)):
        return 0
    import voreinstellung
    loes = voreinstellung.laden_loesungen(
        physik, voreinstellung.fingerabdruck(physik, _BAU_FUNKTIONEN))
    return capsule.loesungen_3d_einsetzen(loes) if loes else 0


def _voreinstellung_einsetzen(cache_key):
    """Gespeicherte Ergebnisse der Voreinstellung in die Session-Caches
    setzen (Gegenprobe 63 f) — NUR für genau ihren Parametersatz.

    Rückgabe: None (andere Parameter), die Metadaten der Datei (Treffer)
    oder der Grund, warum sie nicht passt ('veraltet', 'fehlt',
    'parameter'). Geprüft wird einmal je Sitzung; die geladenen Daten
    bleiben in der Sitzung und werden wieder eingesetzt, falls der
    begrenzte Ergebnis-Cache sie verdrängt hat."""
    if cache_key != _ergebnis_schluessel(DEFAULTS):
        return None
    store = _session_store("_voreinstellung")
    if "status" not in store:
        import voreinstellung
        abdruck = voreinstellung.fingerabdruck(cache_key,
                                               _VOREINSTELLUNG_FUNKTIONEN)
        erg, zus, meta = voreinstellung.laden(cache_key, abdruck)
        store.update(status=meta, ergebnis=erg, zusammenfassung=zus)
    if store["ergebnis"] is not None:
        results = _session_store("_results_cache")
        if cache_key not in results:
            results[cache_key] = store["ergebnis"]
        summaries = _session_store("_summary_cache")
        for lang, text in store["zusammenfassung"].items():
            summaries.setdefault(f"{lang}|{cache_key}", text)
    return store["status"]


def get_summary(cache_key, capsule, lang):
    """Diagnose-Summary getrennt gecacht (Schlüssel inkl. Sprache).

    So bleibt der teure Ergebnis-Cache sprachunabhängig gültig, und ein
    Sprachwechsel formatiert nur den Text neu (die 1-kHz-Lösung bewahrt
    im 3D-Modus die Kapsel selbst auf, Gegenprobe 56).
    """
    store = _session_store("_summary_cache")
    key = f"{lang}|{cache_key}"
    hit = store.get(key)
    if hit is None:
        hit = capsule.summary(lang=lang)
        store[key] = hit
        while len(store) > _RESULTS_CACHE_MAX:
            store.pop(next(iter(store)))
    return hit


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------
def _base_layout(fig, height):
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=10, r=16, t=36, b=10),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS,
                        font=dict(family=FONT, color=INK)),
    )
    return fig


def bode_figure(fr, normalized):
    amp = fr["amplitude_db_norm"] if normalized else fr["amplitude_db"]
    amp_title = tr("fig_amp_norm") if normalized else tr("fig_amp_abs")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.58, 0.42])
    druck = "pressure_db" in fr
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=amp, mode="lines",
        line=dict(color=SERIES[0], width=2),
        name=tr("name_freefield") if druck else tr("fig_amp_abs"),
        showlegend=druck,
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
    ), row=1, col=1)
    if druck:
        # Druckfrequenzgang (ohne Körper) zum Vergleich mit COMSOL/Kuppler
        fig.add_trace(go.Scatter(
            x=fr["frequency_hz"],
            y=fr["pressure_db_norm"] if normalized else fr["pressure_db"],
            mode="lines", line=dict(color=MUTED, width=1.5, dash="dash"),
            name=tr("name_pressure"), showlegend=True,
            hovertemplate=("%{x:.0f} Hz · %{y:.1f} dB<extra>"
                           + tr("name_pressure_short") + "</extra>"),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=fr["phase_deg"], mode="lines",
        line=dict(color=SERIES[4], width=2), name=tr("fig_phase"),
        showlegend=False,
        hovertemplate="%{x:.0f} Hz · %{y:.0f}°<extra></extra>",
    ), row=2, col=1)

    for row in (1, 2):
        fig.update_xaxes(
            type="log", row=row, col=1, gridcolor=GRID, griddash="dot",
            linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED),
            zeroline=False,
            tickvals=[10, 100, 1000, 10000],
            ticktext=["10", "100", "1k", "10k"],
        )
    fig.update_xaxes(title_text=tr("fig_freq"), title_font=dict(color=INK_2),
                     row=2, col=1)
    fig.update_yaxes(title_text=amp_title, row=1, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2),
                     zeroline=False)
    fig.update_yaxes(title_text=tr("fig_phase"), row=2, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2),
                     zeroline=False)
    fig.update_layout(showlegend=druck, hovermode="x unified",
                      title=dict(text=tr("fig_bode_title"),
                                 font=dict(color=INK, size=16)),
                      legend=dict(x=0.01, y=0.99, xanchor="left",
                                  yanchor="top", bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=INK_2, size=12)))
    return _base_layout(fig, 560)


def rear_bode_figure(fr, aux):
    """Richtwirkung über die Frequenz: Pegel bei 90°/180° relativ zu 0°."""
    f = fr["frequency_hz"]
    fig = go.Figure()
    # −6-dB-Marke als Legendeneintrag: als Text im Diagramm lag sie je
    # nach Kapsel mitten in den Kurven
    fig.add_trace(go.Scatter(
        x=[f[0], f[-1]], y=[-6.0, -6.0], mode="lines",
        name=tr("fig_rear_ann"), hoverinfo="skip",
        line=dict(color=MUTED, width=1, dash="dot")))
    fig.add_trace(go.Scatter(
        x=f, y=aux["level_90_db"], mode="lines", name=tr("name_90"),
        line=dict(color=SERIES[1], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra>90°</extra>"))
    fig.add_trace(go.Scatter(
        x=f, y=aux["level_180_db"], mode="lines", name=tr("name_180"),
        line=dict(color=SERIES[5], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra>180°</extra>"))
    fig.update_xaxes(type="log", gridcolor=GRID, griddash="dot",
                     linecolor=AXIS, tickcolor=AXIS,
                     tickfont=dict(color=MUTED), zeroline=False,
                     tickvals=[10, 100, 1000, 10000],
                     ticktext=["10", "100", "1k", "10k"],
                     title_text=tr("fig_freq"),
                     title_font=dict(color=INK_2))
    y_min = float(min(np.min(aux["level_180_db"]), -20.0))
    fig.update_yaxes(title_text=tr("fig_lvl"),
                     range=[max(y_min - 3.0, -45.0), 3.0],
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top",
                                  y=-0.22, xanchor="center", x=0.5,
                                  font=dict(color=INK_2)),
                      title=dict(text=tr("fig_rear_title"),
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 460)


def noise_figure(noise):
    """Äquivalente Eingangs-Rauschdichte über die Frequenz [dB re 20 µPa/√Hz]."""
    f = noise["f"]
    asd_db = 20.0 * np.log10(np.maximum(noise["asd"], 1e-30) / 2.0e-5)
    sn = noise["self"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, y=asd_db, mode="lines", name=tr("noise_asd_name"),
        line=dict(color=SERIES[7], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>"))
    fig.add_hline(
        y=sn["spl_a_db"], line=dict(color=SERIES[5], width=1, dash="dot"),
        annotation_text=tr("noise_asd_aline", v=sn["spl_a_db"]),
        annotation_font=dict(color=SERIES[5], size=11))
    fig.update_xaxes(type="log", gridcolor=GRID, griddash="dot",
                     linecolor=AXIS, tickcolor=AXIS,
                     tickfont=dict(color=MUTED), zeroline=False,
                     tickvals=[10, 100, 1000, 10000],
                     ticktext=["10", "100", "1k", "10k"],
                     title_text=tr("fig_freq"),
                     title_font=dict(color=INK_2))
    fig.update_yaxes(title_text=tr("noise_asd_axis"),
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_layout(hovermode="x unified", showlegend=False,
                      title=dict(text=tr("fig_noise_title"),
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 420)


def dr_figure(fr, aux):
    """Phasenschieber-Diagnose: |D_r| und Phase vs. externes Ziel G(180°)."""
    f = fr["frequency_hz"]
    D_r, G180 = aux["D_r"], aux["G180"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=f, y=np.abs(D_r), mode="lines", name=tr("name_dr"),
        line=dict(color=SERIES[0], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.2f}<extra>|D_r|</extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=np.abs(G180), mode="lines", name=tr("name_g"),
        line=dict(color=SERIES[2], width=2, dash="dash"),
        hovertemplate="%{x:.0f} Hz · %{y:.2f}<extra>|G|</extra>"),
        row=1, col=1)
    ph_d = np.rad2deg(np.unwrap(np.angle(D_r)))
    ph_g = np.rad2deg(np.unwrap(np.angle(G180)))
    fig.add_trace(go.Scatter(
        x=f, y=ph_d, mode="lines", name=tr("name_arg_dr"),
        line=dict(color=SERIES[4], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f}°<extra>arg D_r</extra>"),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=ph_g, mode="lines", name=tr("name_arg_g"),
        line=dict(color=SERIES[2], width=2, dash="dash"),
        hovertemplate="%{x:.0f} Hz · %{y:.1f}°<extra>arg G</extra>"),
        row=2, col=1)
    f_h = aux["f_helmholtz_hz"]
    if f_h is not None:
        for row in (1, 2):
            fig.add_vline(x=f_h, row=row, col=1,
                          line=dict(color=SERIES[5], width=1, dash="dot"))
        fig.add_annotation(x=np.log10(f_h), y=1, yref="y domain", row=1,
                           col=1, text=f"f_H ≈ {f_h / 1000:.2f} kHz",
                           showarrow=False, yanchor="bottom",
                           font=dict(color=SERIES[5], size=11))
    for row in (1, 2):
        fig.update_xaxes(type="log", row=row, col=1, gridcolor=GRID,
                         griddash="dot", linecolor=AXIS, tickcolor=AXIS,
                         tickfont=dict(color=MUTED), zeroline=False,
                         tickvals=[10, 100, 1000, 10000],
                         ticktext=["10", "100", "1k", "10k"])
    fig.update_xaxes(title_text=tr("fig_freq"),
                     title_font=dict(color=INK_2), row=2, col=1)
    # Betragsachse deckeln: oberhalb der Resonanz explodiert |D_r|
    dmax = float(np.max(np.abs(D_r)))
    fig.update_yaxes(title_text=tr("fig_mag"), row=1, col=1,
                     range=[0.0, min(max(1.6, 1.1 * dmax), 4.0)],
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_yaxes(title_text=tr("fig_phase"), row=2, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    # Legende UNTER dem Diagramm: vier Einträge neben dem langen Titel
    # überdeckten ihn (und die f_H-Marke)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top",
                                  y=-0.22, xanchor="center", x=0.5,
                                  font=dict(color=INK_2)),
                      title=dict(text=tr("fig_dr_title"),
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 460)


def _freq_label(f):
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


def polar_figure(di):
    angles = di["angles_deg"]
    fig = go.Figure()
    r_max = 0.0
    for i, (f, pat) in enumerate(sorted(di["patterns"].items())):
        r_max = max(r_max, float(np.max(pat["db"])))
        # Bei mehr als 8 Kurven wiederholen sich die Serienfarben — die
        # Wiederholungen werden gestrichelt, damit sie eindeutig bleiben.
        dash = "solid" if i < len(SERIES) else "dash"
        fig.add_trace(go.Scatterpolar(
            theta=angles, r=pat["db"], mode="lines",
            name=_freq_label(f),
            line=dict(color=SERIES[i % len(SERIES)], width=2, dash=dash),
            hovertemplate="%{theta:.0f}° · %{r:.1f} dB<extra>"
                          + _freq_label(f) + "</extra>",
        ))
    fig.update_layout(
        title=dict(text=tr("fig_polar_title"),
                   font=dict(color=INK, size=16)),
        polar=dict(
            bgcolor=SURFACE,
            angularaxis=dict(rotation=90, direction="clockwise", dtick=30,
                             gridcolor=GRID, linecolor=AXIS,
                             tickfont=dict(color=MUTED)),
            radialaxis=dict(range=[-40, max(2.0, np.ceil(r_max))], dtick=10,
                            ticksuffix=" dB", angle=90, tickangle=90,
                            gridcolor=GRID, linecolor=AXIS,
                            tickfont=dict(color=MUTED)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5,
                    font=dict(color=INK_2)),
    )
    _base_layout(fig, 560)
    # breitere Ränder als die übrigen Diagramme: die Winkelbeschriftungen
    # bei 270° und 90° waren abgeschnitten
    fig.update_layout(margin=dict(l=46, r=46, t=44, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Projekt + Parameter
# ---------------------------------------------------------------------------
_init_state()
_halte_ausgeblendete()
_heal_canonical_state()   # Selbstheilung nach Sprachwechsel (s. oben)

with st.sidebar:
    st.title("Capsim")
    st.caption(tr("sidebar_caption"))

    # ---------------- Sprache / Language -------------------------------
    # GUI-Einstellung, kein Kapselparameter: die Sprachwahl wandert NICHT
    # in Projektdateien. Umschalten löst nur einen Rerun aus.
    st.radio(tr("lang_label"), options=["en", "de"],
             format_func=lambda c: {"en": "English",
                                    "de": "Deutsch"}.get(c, c),
             key="ui_lang", horizontal=True)

    # ---------------- Projekt speichern / laden ------------------------
    with st.expander(tr("exp_project"), expanded=False):
        st.file_uploader(
            tr("upload_label"), type=["json"],
            key="project_upload", on_change=_load_project,
        )
        # Mitgelieferte Beispielprojekte (in der Cloud der einzige Weg zu
        # ihnen) und der Rückweg zur Voreinstellung
        _bsp = _beispiele()
        if _bsp:
            st.selectbox(tr("lbl_example"), _bsp, index=None,
                         key="beispiel_wahl",
                         placeholder=tr("example_placeholder"))
            st.button(tr("btn_example"), key="btn_example",
                      width="stretch", on_click=_load_example,
                      disabled=st.session_state.get("beispiel_wahl")
                      is None)
        st.button(tr("btn_default"), key="btn_default", width="stretch",
                  on_click=_restore_defaults, help=tr("help_default"))
        # Notizen: wandern als _kommentar in die Projektdatei und zurück
        st.text_area(tr("lbl_notes"), key="projekt_notizen", height=110,
                     help=tr("help_notes"))
        st.download_button(
            tr("save_btn"),
            data=_projekt_json(_current_params(),
                               st.session_state["projekt_notizen"]),
            file_name=tr("project_filename"), mime="application/json",
            width="stretch",
        )
        # Null-Reset mit Bestätigungsschritt (zweistufig, s. _ZERO_STATE)
        if not st.session_state.get("_confirm_reset", False):
            st.button(tr("btn_reset"), key="btn_reset_ask",
                      width="stretch", on_click=_ask_reset,
                      help=tr("help_reset"))
        else:
            st.warning(tr("reset_confirm"))
            rc1, rc2 = st.columns(2)
            rc1.button(tr("btn_reset_yes"), key="btn_reset_yes",
                       width="stretch", on_click=_zero_all_params)
            rc2.button(tr("btn_reset_no"), key="btn_reset_no",
                       width="stretch", on_click=_cancel_reset)
    if "_load_msg" in st.session_state:
        kind, msg = st.session_state.pop("_load_msg")
        {"success": st.success, "warning": st.warning}.get(
            kind, st.error)(msg)

    # ---------------- Membran ------------------------------------------
    with st.expander(tr("exp_membrane"), expanded=True):
        st.selectbox(tr("lbl_material"), list(MATERIAL_LABELS),
                     format_func=_label_formatter(), key="p_material")
        st.checkbox(tr("lbl_use_fres"), key="p_use_f_res",
                    help=tr("help_use_fres"))
        _zahl("f_res_hz", tr("lbl_fres"), step=100.0, format="%.0f",
              disabled=not st.session_state["p_use_f_res"])
        _zahl("mem_diameter_mm", tr("lbl_mem_dia"), step=0.5,
              titel=f"{tr('exp_membrane')} · {tr('lbl_mem_dia')}")
        _zahl("mem_thickness_um", tr("lbl_mem_thick"), step=0.5,
              titel=f"{tr('exp_membrane')} · {tr('lbl_mem_thick')}")
        _zahl("mem_tension_npm", tr("lbl_mem_tension"), step=10.0)
        _zahl("center_post_mm", tr("lbl_center_post"), step=0.1,
              format="%.2f", help=tr("help_center_post"))
        _zahl("mem_modes", tr("lbl_mem_modes"), step=1,
              help=tr("help_mem_modes"))
        st.checkbox(tr("lbl_modal_source"), key="p_modal_source",
                    help=tr("help_modal_source"))

    # ---------------- Backplate ----------------------------------------
    with st.expander(tr("exp_backplate"), expanded=True):
        _zahl("air_gap_um", tr("lbl_air_gap"), step=1.0)
        _zahl("bp_diameter_mm", tr("lbl_bp_dia"), step=0.5,
              help=tr("help_bp_dia"),
              titel=f"{tr('exp_backplate')} · {tr('lbl_bp_dia')}")
        _zahl("bp_thickness_mm", tr("lbl_bp_thick"), step=0.1,
              titel=f"{tr('exp_backplate')} · {tr('lbl_bp_thick')}")
        _zahl("bias_v", tr("lbl_bias"), step=1.0, help=tr("help_bias"))
        st.radio(tr("lbl_arch"), list(ARCH_LABELS),
                 format_func=_label_formatter(), key="p_architecture",
                 help=tr("help_arch"))
        _zahl("center_gap_um", tr("lbl_center_gap"), step=5.0,
              disabled=st.session_state["p_architecture"] != K67_LABEL,
              help=tr("help_center_gap"))

        st.markdown(tr("hd_holes"))
        st.caption(tr("cap_holes"))

        st.markdown(tr("md_through"), help=tr("help_through"))
        _ring_rows("th")
        _zahl("d_through_mm", tr("lbl_th_dia"), step=0.05,
              help=tr("help_hole_dia"))
        st.toggle(tr("lbl_stepped"), key="p_th_stepped",
                  help=tr("help_stepped"))

        st.markdown(tr("md_blind"), help=tr("help_blind"))
        _ring_rows("bh")
        _zahl("d_blind_mm", tr("lbl_bh_dia"), step=0.05,
              help=tr("help_hole_dia"))
        _zahl("blind_depth_mm", tr("lbl_bh_depth"), step=0.05)

        st.markdown(tr("hd_clr"), help=tr("help_clr"))
        _zahl("clr_dia_mm", tr("lbl_clr_dia"), step=0.5,
              help=tr("help_clr_dia"))
        _zahl("clr_width_mm", tr("lbl_clr_w"), step=0.1,
              help=tr("help_clr_w"))
        _zahl("clr_depth_mm", tr("lbl_clr_d"), step=0.01, format="%.3f",
              help=tr("help_clr_d"))

        # Durchgehender Randspalt (B&K) — nicht bei der K67-Bauform
        if st.session_state["p_architecture"] != K67_LABEL:
            _zahl("ring_vent_um", tr("lbl_ring_vent"), step=5.0,
                  help=tr("help_ring_vent"))
            _zahl("ring_vent_len_mm", tr("lbl_ring_vent_len"), step=0.1,
                  format="%.2f", help=tr("help_ring_vent_len"))

        # Klemmringe vor den Membranen — nur bei K67-Bauform relevant
        if st.session_state["p_architecture"] == K67_LABEL:
            st.markdown(tr("hd_clamp"), help=tr("help_clamp"))
            _zahl("clamp_ring_mm", tr("lbl_clamp_t"), step=0.5,
                  help=tr("help_clamp_t"))
            _zahl("clamp_width_mm", tr("lbl_clamp_w"), step=0.5,
                  help=tr("help_clamp_w"))

    # ---------------- Rückseite / Laufzeitglied -------------------------
    _is_k67 = st.session_state["p_architecture"] == K67_LABEL
    with st.expander(tr("exp_rear"), expanded=not _is_k67):
        if _is_k67:
            st.caption(tr("cap_rear_k67"))
        st.toggle(tr("lbl_rear_on"), key="p_rear_enabled",
                  disabled=_is_k67, help=tr("help_rear_on"))
        _rear_on = st.session_state["p_rear_enabled"] and not _is_k67
        st.markdown(tr("hd_spacer"), help=tr("help_spacer_hd"))
        _zahl("spacer_um", tr("lbl_spacer"), step=5.0,
              disabled=not _rear_on, help=tr("help_spacer"))
        _zahl("rearplate_mm", tr("lbl_rp_t"), step=0.1,
              disabled=not _rear_on, help=tr("help_rp_t"))
        _rp_on = _rear_on and st.session_state["p_rearplate_mm"] > 0.0
        _zahl("n_rearplate", tr("lbl_rp_n"), step=1, disabled=not _rp_on,
              help=tr("help_rp_n"))
        _zahl("d_rearplate_mm", tr("lbl_rp_d"), step=0.05,
              disabled=not _rp_on)

        st.markdown(tr("hd_delay"))
        _zahl("delay_mm", tr("lbl_delay"), step=0.5, disabled=not _rear_on,
              help=tr("help_delay"))
        _zahl("cavity_length_mm", tr("lbl_cav_len"), step=0.5,
              disabled=not _rear_on)
        _zahl("cavity_wall_mm", tr("lbl_cav_wall"), step=0.1,
              disabled=not _rear_on)
        st.radio(tr("lbl_hole_pos"), list(POS_LABELS),
                 format_func=_label_formatter(),
                 key="p_hole_position", disabled=not _rear_on)
        _zahl("n_cavity", tr("lbl_cav_n"), step=1, disabled=not _rear_on,
              help=tr("help_cav_n"))
        _zahl("d_cavity_mm", tr("lbl_cav_d"), step=0.05,
              disabled=not _rear_on, help=tr("help_cav_d"))
        # Bohrungsposition höchstens Zylinderlänge (Ober-Grenze 0 bei
        # fehlendem Zylinder: Streamlit erlaubt min == max)
        _zahl("cavity_axial_mm", tr("lbl_cav_ax"), step=0.5,
              disabled=(not _rear_on)
              or st.session_state["p_hole_position"] != "Umfang"
              or st.session_state["p_cavity_length_mm"] <= 0.0,
              help=tr("help_cav_ax"))

    # ---------------- Akustisches Gewebe --------------------------------
    with st.expander(tr("exp_fabric"), expanded=True):
        _zahl("fabric_front_rayl", tr("lbl_fab_front"), step=5.0)
        _zahl("fabric_rear_rayl", tr("lbl_fab_rear"), step=5.0)
        st.radio(tr("lbl_fab_pos"), list(FAB_POS_LABELS),
                 format_func=_label_formatter(), key="p_fabric_rear_pos",
                 disabled=_is_k67, help=tr("help_fab_pos"))
        if _is_k67:
            st.caption(tr("cap_fab_pos_k67"))

    # ---------------- Gehäuse & Beugung ----------------------------------
    with st.expander(tr("exp_body"), expanded=False):
        st.toggle(tr("lbl_diffr"), key="p_diffraction_on",
                  help=tr("help_diffr"))
        # Strahlungslast im Druckgang: nur ohne Beugung wählbar (mit
        # Beugung liegt sie immer an), nicht bei der K67-Bauform
        st.toggle(tr("lbl_rad_load"), key="p_rad_load",
                  disabled=(st.session_state["p_diffraction_on"]
                            or st.session_state["p_architecture"]
                            == K67_LABEL),
                  help=tr("help_rad_load"))
        _zahl("body_diameter_mm", tr("lbl_body_dia"), step=0.5,
              disabled=not st.session_state["p_diffraction_on"],
              help=tr("help_body_dia"))
        _k67 = st.session_state["p_architecture"] == K67_LABEL
        # Sphäroid ist ein reines Front-Rück-Transfermodell und bleibt der
        # Doppelmembran vorbehalten; BEM liefert bei Ein-Membran-Bauformen
        # den Frontfaktor der flachen Stirnfläche (Druckempfänger).
        _ax_opts = [k for k, v in AX_LABELS.items()
                    if _k67 or v != "spheroid"]
        if st.session_state["p_axial_body"] not in _ax_opts:
            st.session_state["p_axial_body"] = _ax_opts[0]
        st.selectbox(tr("lbl_ax_body"), _ax_opts,
                     format_func=_label_formatter(), key="p_axial_body",
                     disabled=not st.session_state["p_diffraction_on"],
                     help=tr("help_ax_body"))
        if AX_LABELS.get(st.session_state["p_axial_body"]) == "bem":
            if not _k67:
                _zahl("body_length_mm", tr("lbl_body_len"), step=0.5,
                      help=tr("help_body_len"))
            _zahl("bem_body_dia_mm", tr("lbl_bem_dia"), step=1.0,
                  help=tr("help_bem_dia"))
            if st.session_state["p_bem_body_dia_mm"] > 0.0:
                _zahl("bem_body_gap_mm", tr("lbl_bem_gap"), step=1.0,
                      help=tr("help_bem_gap"))
                _zahl("bem_body_len_mm", tr("lbl_bem_len"), step=5.0,
                      help=tr("help_bem_len"))
            st.caption(tr("cap_bem") if _k67 else tr("cap_bem_front"))

    # ---------------- Spaltfilm-Modell -----------------------------------
    with st.expander(tr("exp_squeeze"), expanded=False):
        st.toggle(tr("lbl_2d"), key="p_squeeze_2d", help=tr("help_2d"))
        if st.session_state["p_squeeze_2d"]:
            st.caption(tr("cap_2d"))
        st.toggle(tr("lbl_3d"), key="p_squeeze_3d", help=tr("help_3d"))
        # Immer sichtbar (ohne 3D gesperrt), damit die Wahl beim Ab- und
        # Wiedereinschalten von 3D erhalten bleibt
        st.toggle(tr("lbl_grid_fine"), key="p_grid_3d_fine",
                  help=tr("help_grid_fine"),
                  disabled=not st.session_state["p_squeeze_3d"])
        if st.session_state["p_squeeze_3d"]:
            if st.session_state["p_grid_3d_fine"]:
                st.caption(tr("cap_grid_fine"))
            st.toggle(tr("lbl_rot_auto"), key="p_half_rot_auto",
                      help=tr("help_rot_auto"))
            if not st.session_state["p_half_rot_auto"]:
                _zahl("half_rot_deg", tr("lbl_rot_deg"), step=0.5,
                      help=tr("help_rot_deg"))
            st.caption(tr("cap_3d"))

    # ---------------- Simulation ---------------------------------------
    with st.expander(tr("exp_sim"), expanded=False):
        # Schieberegler lassen Werte außerhalb ungeprüft durch: klemmen
        _npts = _bereich("n_points", _ss_wert)
        _klemme("p_n_points", *_npts, tr("lbl_npts"))
        st.slider(tr("lbl_npts"), *_npts, step=50, key="p_n_points")
        st.toggle(tr("lbl_norm"), key="p_normalize_1khz")
        st.multiselect(tr("lbl_dirf"), DIRECTIVITY_OPTIONS,
                       key="p_dir_freqs", max_selections=10)

# ---------------------------------------------------------------------------
# Berechnung
# ---------------------------------------------------------------------------
def _hauptbereich():
    """Berechnung und Hauptbereich. Läuft nur mit Streamlit-Laufzeit
    (App, AppTest): als Modul importiert (Proben, voreinstellung.py)
    liefert das Skript nur seine Definitionen und rechnet nichts — die
    Voreinstellung ist ein 3D-Lauf von über einer Stunde."""
    params = _current_params()

    # Platzhalter VOR dem Modellaufbau: schon der Konstruktor (Elektrostatik,
    # bei 3D der Gitteraufbau) soll als Fortschritt sichtbar sein.
    _prog_slot = st.empty()


    def _show_progress(frac, label):
        _prog_slot.progress(frac, text=tr("prog_fmt", label=label,
                                          pct=100 * frac))


    _cache_key = _ergebnis_schluessel(params)
    # Voreinstellung: gespeicherte Ergebnisse statt einer Stunde 3D-Rechnung
    # (nur genau für ihren Parametersatz und nur, solange sie zum Modell
    # passen, s. _voreinstellung_einsetzen). Dann wird auch die Kapsel
    # nicht gebaut — ihre Kennwerte stehen in der Datei.
    _gespeichert = _voreinstellung_einsetzen(_cache_key)
    if isinstance(_gespeichert, str) and not st.session_state.get(
            "_voreinstellung_rechnen"):
        # Passt die Datei nicht, startet die Stundenrechnung NIE von selbst
        # (in der Streamlit Cloud drosselte das die App): Grund zeigen, auf
        # Knopfdruck rechnen; jede andere Einstellung rechnet wie immer.
        st.info(tr("info_default_stale" if _gespeichert == "veraltet"
                   else "info_default_missing"))
        st.button(tr("btn_default_compute"),
                  on_click=lambda: st.session_state.update(
                      _voreinstellung_rechnen=True))
        st.stop()
    _uebernommen, _aus_prozess, (_parallel, _prozesse) = 0, 0, (0, 1)
    if isinstance(_gespeichert, dict):
        capsule = _GespeicherteKapsel(_gespeichert["kennwerte"])
    else:
        try:
            capsule = get_capsule(params, _show_progress)
        except ValueError as exc:
            _prog_slot.empty()
            st.error(tr("err_params", exc=meldung(exc)))
            st.stop()
        # 3D (Gegenprobe 75): ist die Kapsel physikalisch die Voreinstellung
        # (nur Punktzahl oder Richtfrequenzen anders), kommen ihre
        # gespeicherten Lösungen je Frequenz in die Kapsel; was dann noch
        # fehlt, rechnen Arbeiterprozesse parallel, wenn Kerne und Speicher
        # reichen (in der Streamlit Cloud nicht)
        if _cache_key not in _session_store("_results_cache"):
            _aus_prozess = _prozess_loesungen(_cache_key, capsule)
            _uebernommen = _voreinstellung_loesungen(_cache_key, capsule)
            _parallel, _prozesse = _vorrechnen_3d(params, capsule,
                                                  _show_progress)

    # Auch eine Rechnung kann an den Maßen scheitern (etwa eine nicht lösbare
    # BEM-Randintegralgleichung) — dann dieselbe Meldung statt eines
    # Tracebacks. Andere Fehler bleiben sichtbar, wie sie sind.
    try:
        _res = compute_results(_cache_key, capsule, _show_progress)
    except ParameterFehler as exc:
        _prog_slot.empty()
        st.error(tr("err_params", exc=meldung(exc)))
        st.stop()
    _prog_slot.empty()
    _prozess_loesungen(_cache_key, capsule, merken=True)
    fr, di, aux = _res["fr"], _res["di"], _res["aux"]
    sens_1k, delay = _res["sens_1k"], _res["delay"]
    noise = _res.get("noise")
    summary_text = get_summary(_cache_key, capsule, _lang())

    # ---------------------------------------------------------------------------
    # Hauptbereich
    # ---------------------------------------------------------------------------
    st.title(tr("app_title"))
    st.caption(tr("app_caption"))
    if isinstance(_gespeichert, dict):
        st.caption(tr("cap_default_stored",
                      datum=_gespeichert["erzeugt"][:10],
                      min=_gespeichert["rechenzeit_s"] / 60.0))
    if _aus_prozess:
        st.caption(tr("cap_3d_process", n=_aus_prozess))
    if _uebernommen:
        st.caption(tr("cap_3d_reused", n=_uebernommen))
    if _parallel:
        st.caption(tr("cap_3d_parallel", n=_parallel, w=_prozesse))

    # Werte außerhalb ihres Feldbereichs (geladenes Projekt, abhängige
    # Grenze): geklemmt und gerechnet, aber nie stillschweigend (s. _BEREICH).
    if _KLEMM_MELDUNGEN:
        st.warning(tr("warn_clamp", items="; ".join(_KLEMM_MELDUNGEN)))

    # BEM mit offenem Rückeinlass: gerechnet, aber mit Modellgrenze — der
    # Bohrungskranz sitzt als idealer Ring bei seiner Einbautiefe auf einer
    # glatten Zylinderkontur (s. Gegenprobe 44).
    if capsule.axial_body_model == "bem" and capsule.rear_open \
            and capsule.architecture != "dual_diaphragm":
        st.warning(tr("warn_bem_gradient", d=capsule.d_ext * 1e3,
                      wo=tr("warn_bem_end"
                            if capsule.cavity_hole_position == "end"
                            else "warn_bem_circ")))

    # Kugel als Körper einer Ein-Membran-Kapsel: der Druckstau der flachen
    # Stirnfläche fehlt bis zu 3.8 dB (B&K 4134 gegen NBS, Gegenprobe 64).
    if (capsule.include_diffraction and capsule.axial_body_model == "sphere"
            and capsule.architecture != "dual_diaphragm"):
        st.warning(tr("warn_sphere_flat"))

    # Homogenisierungsgrenze der 1D/2D-Modelle: gerechnet wird trotzdem, aber
    # oberhalb f_hom bzw. f_ring ist das Ergebnis nicht mehr gegen den
    # 3D-Löser abgesichert (Gegenproben 48, 53, 60). Beide Grenzen sind
    # vorsichtige Schätzungen und stehen deshalb nur als zugeklappter
    # Hinweis am Seitenende.
    _hinweis_hom = None
    if capsule.squeeze_model in ("1d", "2d"):
        _hom = capsule.homogenization_limit()
        if _hom["f_limit"] < capsule._F_BAND_TOP and _hom["cause"] == "ring":
            _hinweis_hom = tr("warn_ring_repr",
                              model=capsule.squeeze_model.upper(),
                              f=_hom["f_ring"] / 1e3, db=capsule._RING_REPR_DB)
        elif _hom["f_hom"] < capsule._F_BAND_TOP:
            _hinweis_hom = tr("warn_sparse_holes",
                              model=capsule.squeeze_model.upper(),
                              f=_hom["f_hom"] / 1e3, rho=_hom["rho"] * 1e3)

    # Biegesteife Folie: das Modell warnt, wenn die Membran mit Randschicht
    # die Grundfrequenz der eingespannten Platte um mehr als 1 % verfehlt
    # (Gegenprobe 62). Bis Gegenprobe 63 i stand das nur im Konsolen-
    # protokoll. (Der Stellvertreter der gespeicherten Voreinstellung trägt
    # den Wert nicht — sie liegt unter der Grenze, s. Gegenprobe 63 f.)
    _rand = getattr(capsule, "rand_abweichung", 0.0)
    if (MicrophoneCapsule._RANDSCHICHT
            and abs(_rand) > MicrophoneCapsule._RAND_GRENZE):
        st.warning(tr("warn_stiff_foil", pct=100.0 * _rand))

    # Lochlage (Gegenprobe 73): Mündungen verschiedener Lochkreise, die sich
    # schneiden, rechnet das Modell als getrennte Löcher — eine Warnung.
    # Überstand über Plattenrand oder Mittenpfosten ist eine Notiz: das
    # Modell rechnet den Teil auf der Platte (die Voreinstellung: die
    # äußeren Senkungen des Fotorasters berühren den Rand um 0,03 mm).
    _lb = getattr(capsule, "loch_befund", None) or {}
    if "ueberlappung" in _lb:
        st.warning(tr("warn_holes_overlap", n=_lb["ueberlappung"][0],
                      mm=1e3 * _lb["ueberlappung"][1]))
    _notiz = []
    if "rand" in _lb:
        _notiz.append(tr("info_holes_edge", n=_lb["rand"][0],
                         mm=1e3 * _lb["rand"][1]))
    if "pfosten" in _lb:
        _notiz.append(tr("info_holes_post", n=_lb["pfosten"][0],
                         mm=1e3 * _lb["pfosten"][1]))
    if _notiz:
        st.info(" ".join(_notiz))

    # Feines 3D-Gitter an der Obergrenze: die kleinste Mündung ist dann
    # schlechter aufgelöst als die Regel verlangt (s. Gegenprobe 50).
    if capsule.squeeze_model == "3d" and capsule._g3d["fine_capped"]:
        st.warning(tr("warn_grid_capped", n=capsule._GRID_FINE_MAX,
                      c=capsule._g3d["cells_rm"], k=capsule._GRID_FINE_CELLS))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr("met_sens"), f"{sens_1k * 1e3:.1f} mV/Pa",
              help=tr("help_met_sens", db=20 * np.log10(max(sens_1k, 1e-12))))
    m2.metric(tr("met_fres"), f"{capsule.f_res:.0f} Hz",
              help=tr("help_met_fres", f=capsule.f_res_from_tension))
    m3.metric(tr("met_c0"), f"{capsule.C_elec_0 * 1e12:.1f} pF",
              help=tr("help_met_c0", n=capsule.n_bp))
    _upi = (f"{capsule.U_pullin:.0f} V" if np.isfinite(capsule.U_pullin)
            else "> 20 kV")
    m4.metric(tr("met_soft"),
              f"{capsule.softening_ratio * 100:.1f} %",
              help=tr("help_met_soft", upi=_upi,
                      w0=capsule.w0_static * 1e6,
                      hmin=capsule.h_min_static * 1e6))

    # Laufzeit-Anpassung des Nieren-Phasenschiebers: die 180°-Null entsteht,
    # wenn die interne Rück-Übertragung D_r des Netzwerks die externe
    # Front-Rück-Übertragung (Körperbeugung) trifft. Beide als äquivalente
    # Laufzeit bei 1 kHz ausgewertet (s. delay_diagnostics; die interne
    # Laufzeit ist frequenzabhängig — RC-Glied, kein reines Laufzeitglied).
    if delay is not None:
        l1, l2, l3, l4 = st.columns(4)
        l1.metric(tr("met_tau_ext"),
                  f"{delay['tau_ext_s'] * 1e6:.1f} µs",
                  help=tr("help_tau_ext", f=delay['f_probe_hz'],
                          d=delay['dist_ext_m'] * 1e3))
        l2.metric(tr("met_tau_int"),
                  f"{delay['tau_int_s'] * 1e6:.1f} µs",
                  help=tr("help_tau_int", f=delay['f_probe_hz'],
                          d=delay['dist_int_m'] * 1e3))
        _ratio = delay["ratio"]
        l3.metric(tr("met_ratio"), f"{_ratio:.2f}",
                  delta=tr("delta_ratio", d=(_ratio - 1.0) * 100),
                  delta_color="off",
                  help=tr("help_ratio"))
        _fh = aux["f_helmholtz_hz"]
        l4.metric(tr("met_fh"),
                  f"{_fh / 1000:.2f} kHz" if _fh is not None else "> 25 kHz",
                  delta=None if _fh is None or _fh > 8000.0
                  else tr("fh_band"),
                  delta_color="off",
                  help=tr("help_fh"))

    # Eigenrauschen: thermisch-akustischer Ersatzgeräuschpegel + Pfad-Anteile
    if noise is not None:
        sn = noise["self"]
        n1, n2, n3, n4 = st.columns(4)
        # dominanter Rauschpfad (Front/Membranfilm/Rückpfad)
        _paths = [(sn["spl_a_front_db"], tr("noise_path_front")),
                  (sn["spl_a_mem_db"], tr("noise_path_mem")),
                  (sn["spl_a_rear_db"], tr("noise_path_rear"))]
        _dom = max(_paths, key=lambda t: t[0])
        n1.metric(tr("met_noise_a"), f"{sn['spl_a_db']:.1f} dB(A)",
                  help=tr("help_noise_a"))
        n2.metric(tr("met_noise_z"), f"{sn['spl_z_db']:.1f} dB",
                  help=tr("help_noise_z"))
        n3.metric(tr("met_snr"),
                  f"{94.0 - sn['spl_a_db']:.1f} dB(A)",
                  help=tr("help_snr"))
        n4.metric(tr("met_noise_dom"), _dom[1],
                  delta=f"{_dom[0]:.1f} dB(A)", delta_color="off",
                  help=tr("help_noise_dom",
                          f=sn["spl_a_front_db"], m=sn["spl_a_mem_db"],
                          r=sn["spl_a_rear_db"]))

    col_bode, col_polar = st.columns([11, 9], gap="medium")
    with col_bode:
        st.plotly_chart(bode_figure(fr, params["normalize_1khz"]),
                        width="stretch",
                        config={"displayModeBar": False})
    with col_polar:
        st.plotly_chart(polar_figure(di), width="stretch",
                        config={"displayModeBar": False})

    # Auslegungs-Diagnose: Richtwirkung über die Frequenz + Phasenschieber D_r
    col_rear, col_dr = st.columns(2, gap="medium")
    with col_rear:
        st.plotly_chart(rear_bode_figure(fr, aux), width="stretch",
                        config={"displayModeBar": False})
    with col_dr:
        if aux["D_r"] is not None:
            st.plotly_chart(dr_figure(fr, aux), width="stretch",
                            config={"displayModeBar": False})
        else:
            st.info(tr("info_no_dr"))

    # Eigenrauschen: äquivalente Eingangs-Rauschdichte über die Frequenz
    if noise is not None:
        st.plotly_chart(noise_figure(noise), width="stretch",
                        config={"displayModeBar": False})
    elif capsule.squeeze_model == "3d":
        st.info(tr("noise_no_3d"))

    with st.expander(tr("exp_diag")):
        # gecachter Text: summary() enthält eine Netzwerkauswertung und
        # liefe sonst bei jedem Rerun mit
        st.code(summary_text, language=None)

    # ---------------------------------------------------------------------------
    # Datenansicht & CSV-Export
    # ---------------------------------------------------------------------------
    st.subheader(tr("hd_data"))

    df_fr = pd.DataFrame({
        tr("col_freq"): fr["frequency_hz"],
        tr("col_sens"): np.abs(fr["sensitivity_v_pa"]) * 1e3,
        tr("col_amp"): fr["amplitude_db"],
        tr("col_ampn"): fr["amplitude_db_norm"],
        tr("col_phase"): fr["phase_deg"],
        tr("col_l90"): aux["level_90_db"],
        tr("col_l180"): aux["level_180_db"],
    })
    if "pressure_db" in fr:
        df_fr[tr("col_pressure")] = fr["pressure_db"]
    if aux["D_r"] is not None:
        df_fr[tr("col_drmag")] = np.abs(aux["D_r"])
        df_fr[tr("col_drph")] = np.rad2deg(np.unwrap(np.angle(aux["D_r"])))
    df_di = pd.DataFrame({tr("col_angle"): di["angles_deg"]})
    for f, pat in sorted(di["patterns"].items()):
        tag = f"{f:.0f}hz"
        df_di[tr("col_lvl_prefix") + tag] = pat["db"]
        df_di[tr("col_lin_prefix") + tag] = pat["linear"]

    tab_fr, tab_di = st.tabs([tr("tab_fr"), tr("tab_di")])
    with tab_fr:
        st.dataframe(df_fr, height=240, width="stretch", hide_index=True)
    with tab_di:
        st.dataframe(df_di, height=240, width="stretch", hide_index=True)

    exp1, exp2, exp3 = st.columns([2, 2, 3])
    # Kanonische Werte ("intl"/"excel_de") im Session-State, damit ein
    # Sprachwechsel die Auswahl nicht ungültig macht (die Anzeige übersetzt
    # format_func — Sprache zur Renderzeit eingefroren, s. _label_formatter).
    _csv_disp = {"intl": tr("csv_intl"), "excel_de": tr("csv_excel_de")}
    sep_choice = exp3.selectbox(
        tr("lbl_csv"), ["intl", "excel_de"],
        format_func=lambda c: _csv_disp.get(c, c),
        key="csv_format",
    )
    _sep, _dec = (";", ",") if sep_choice == "excel_de" else (",", ".")


    def _to_csv(df):
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep=_sep, decimal=_dec)
        return buf.getvalue().encode("utf-8-sig")  # BOM: Umlaute in Excel


    exp1.download_button(tr("btn_csv_fr"), _to_csv(df_fr),
                         file_name=tr("csv_fr_name"), mime="text/csv",
                         width="stretch")
    exp2.download_button(tr("btn_csv_di"), _to_csv(df_di),
                         file_name=tr("csv_di_name"), mime="text/csv",
                         width="stretch")

    st.caption(tr("footer"))

    # Homogenisierungsgrenze des 1D/2D-Modells (Lochbild bzw. Lochkreise):
    # zugeklappt, ganz unten
    if _hinweis_hom is not None:
        with st.expander(tr("exp_hint"), expanded=False):
            st.markdown(_hinweis_hom)


if get_script_run_ctx() is not None:
    _hauptbereich()
