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
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from microphone_capsule import MicrophoneCapsule

# ---------------------------------------------------------------------------
# Seiten-Setup & Designsprache
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Capsim — Kapselsimulation",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

DIRECTIVITY_OPTIONS = [50, 100, 125, 250, 500, 1000, 2000, 4000,
                       5000, 8000, 10000, 12500, 16000, 20000]

# (key, Widget-Art, Default) — Gruppen s. Sidebar-Aufbau weiter unten
# Voreinstellungen beim Start: die Debenham/Robinson/Stebbings-Kapsel
# ("A Stereo Condenser Microphone", Hi-Fi News) — Braunmühl-Weber-Bauform
# mit einteiliger durchbohrter Mittelelektrode; identisch zu
# examples/debenham_stereo_condenser.json.
DEFAULTS = {
    # Membran
    "material": "PET (Mylar)",
    "use_f_res": True,
    "f_res_hz": 2100.0,
    "mem_diameter_mm": 25.4,
    "mem_thickness_um": 6.0,
    "mem_tension_npm": 45.0,
    # Backplate
    "air_gap_um": 38.1,
    "bp_diameter_mm": 23.9,
    "bp_thickness_mm": 3.125,
    "bias_v": 50.0,
    "architecture": "Doppelmembran (K67-Bauform)",
    "center_gap_um": 0.0,
    # Lochmuster als Lochkreis-Listen [Anzahl, Lochkreis-Ø in mm];
    # Lochkreis-Ø 0 = gleichmäßig verteilt. Debenham-Zeichnung: Lochkreise
    # 0.860/0.688/0.516/0.344/0.172 Zoll = 21.84/17.48/13.11/8.74/4.37 mm.
    "d_through_mm": 0.71,
    "th_rings": [[6, 21.84], [3, 17.48], [3, 8.74]],
    "th_stepped": False,
    "d_blind_mm": 1.2,
    "blind_depth_mm": 3.0,
    "bh_rings": [[12, 21.84], [12, 17.48], [12, 13.11], [6, 8.74],
                 [4, 4.37]],
    # Klemmringe vor den Membranen (nur K67-Bauform); 0 = keine.
    # Debenham: 2 mm dick, Aussen-Ø 32 mm -> 3 mm breit.
    "clamp_ring_mm": 2.0,
    "clamp_width_mm": 3.0,
    # Clearance-Ring: Stirnflaechen-Freistich am Elektrodenrand der
    # Debenham-Elektrode (aus der Konstruktionszeichnung: Abtrag 0.038 mm
    # ueber die aeusseren 1.27 mm; mittlerer Ring-Oe 23.9 - 1.27 mm).
    "clr_dia_mm": 22.63,
    "clr_width_mm": 1.27,
    "clr_depth_mm": 0.038,
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
    # Gehäuse & Beugung
    "diffraction_on": True,
    "body_diameter_mm": 32.0,
    # Axialer Körper für den Front-Rück-Transfer der Doppelmembran:
    # Kugel (d_ext) = montierte Kapsel (Standard); Sphäroid = frei
    # stehende Scheibe (Referenzfall, Gegenprobe 20); BEM = montagetreue
    # Kontur Kopf + Mikrofonkörper (Gegenprobe 21).
    "axial_body": "Kugel (d_ext, montiert)",
    "bem_body_dia_mm": 56.0,
    "bem_body_gap_mm": 15.0,
    "bem_body_len_mm": 80.0,
    # Spaltfilm-Modell (Debenham braucht 2D für die tiefe Niere;
    # 3D = diskrete Löcher; center_gap > 0 -> K67-Modus: Zwischenspalt
    # als dritter Film, Stufenbohrungen als Zweitor-Kette, Elektroden-
    # hälften gegeneinander verdreht — Gegenprobe 22)
    "squeeze_2d": True,
    "squeeze_3d": False,
    # Verdrehung der Elektrodenhälften (nur 3D-K67-Modus): automatisch =
    # halbe Teilung des Durchgangs-Lochbilds (180°/n_th, reale K67)
    "half_rot_auto": True,
    "half_rot_deg": 3.0,
    # Simulation
    "n_points": 400,
    "normalize_1khz": True,
    "dir_freqs": [100, 1000, 10000],
}

_FLOAT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, float)}
_INT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, int)
             and not isinstance(v, bool)}
_BOOL_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, bool)}

# Lochkreis-Listen: Projekt-/DEFAULTS-Schlüssel -> Session-Key-Präfix.
# Die Ringzeilen leben als einzelne Widget-Keys p_<prefix>_ring_n_<i> /
# p_<prefix>_ring_pcd_<i> plus Zeilenzähler <prefix>_ring_count, weil
# Streamlit-Widgets nur skalare Zustände tragen.
_RING_PREFIX = {"th_rings": "th", "bh_rings": "bh"}
MAX_RINGS = 8


def _set_ring_state(prefix, rings):
    """Ringliste [[Anzahl, Lochkreis-Ø mm], ...] in die Widget-Session-Keys
    schreiben; überzählige Zeilen eines früheren Zustands werden entfernt."""
    old = st.session_state.get(f"{prefix}_ring_count", 0)
    for i in range(len(rings), old):
        st.session_state.pop(f"p_{prefix}_ring_n_{i}", None)
        st.session_state.pop(f"p_{prefix}_ring_pcd_{i}", None)
    for i, (cnt, pcd) in enumerate(rings):
        st.session_state[f"p_{prefix}_ring_n_{i}"] = int(cnt)
        st.session_state[f"p_{prefix}_ring_pcd_{i}"] = float(pcd)
    st.session_state[f"{prefix}_ring_count"] = len(rings)


def _rings_from_state(prefix):
    n = st.session_state.get(f"{prefix}_ring_count", 1)
    return [[int(st.session_state[f"p_{prefix}_ring_n_{i}"]),
             float(st.session_state[f"p_{prefix}_ring_pcd_{i}"])]
            for i in range(n)]


def _init_state():
    for key, val in DEFAULTS.items():
        if key in _RING_PREFIX:
            if f"{_RING_PREFIX[key]}_ring_count" not in st.session_state:
                _set_ring_state(_RING_PREFIX[key], val)
        else:
            st.session_state.setdefault("p_" + key, val)


def _current_params():
    return {k: (_rings_from_state(_RING_PREFIX[k]) if k in _RING_PREFIX
                else st.session_state["p_" + k])
            for k in DEFAULTS}


def _coerce(key, val):
    """Robuste Typkonvertierung beim Projekt-Laden."""
    if key in _RING_PREFIX:
        rings = [[max(0, int(r[0])), max(0.0, float(r[1]))]
                 for r in list(val)[:MAX_RINGS]]
        return rings or [[0, 0.0]]
    if key in _BOOL_KEYS:
        return bool(val)
    if key in _FLOAT_KEYS:
        return float(val)
    if key in _INT_KEYS:
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
    raise ValueError(f"ungültiger Wert für '{key}': {val!r}")


def _load_project():
    """on_change-Callback des Uploaders: läuft vor dem Widget-Aufbau,
    darf daher st.session_state der Parameter-Widgets setzen."""
    up = st.session_state.get("project_upload")
    if up is None:
        return
    try:
        data = json.load(up)
        if data.get("format") != "capsim-project":
            raise ValueError("kein Capsim-Projektformat")
        params_in = data.get("params", {})
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
        # enthaltene Parameter fallen auf die Voreinstellung zurück
        # (ältere Projekte kennen z. B. Spacer/Rückplatte noch nicht).
        for key, default in DEFAULTS.items():
            val = staged.get(key, default)
            if key in _RING_PREFIX:
                _set_ring_state(_RING_PREFIX[key], val)
            else:
                st.session_state["p_" + key] = val
        st.session_state["_load_msg"] = (
            "success",
            f"Projekt geladen — {len(staged)} Parameter übernommen.")
    except Exception as exc:  # defekte Datei darf die App nicht stoppen
        st.session_state["_load_msg"] = (
            "error", f"Projekt konnte nicht geladen werden: {exc}")


def _add_ring(prefix):
    """+‑Button: hängt einen weiteren (leeren) Lochkreis an."""
    n = st.session_state[f"{prefix}_ring_count"]
    if n >= MAX_RINGS:
        return
    st.session_state[f"p_{prefix}_ring_n_{n}"] = 0
    st.session_state[f"p_{prefix}_ring_pcd_{n}"] = 0.0
    st.session_state[f"{prefix}_ring_count"] = n + 1


def _remove_ring(prefix):
    """−‑Button: entfernt den letzten Lochkreis (mindestens einer bleibt)."""
    n = st.session_state[f"{prefix}_ring_count"]
    if n <= 1:
        return
    st.session_state.pop(f"p_{prefix}_ring_n_{n - 1}", None)
    st.session_state.pop(f"p_{prefix}_ring_pcd_{n - 1}", None)
    st.session_state[f"{prefix}_ring_count"] = n - 1


def _ring_rows(prefix, bp_diameter_mm):
    """Dynamische Lochkreis-Zeilen eines Lochtyps (Anzahl + Lochkreis-Ø)
    mit ➕/➖-Buttons; gibt die Gesamt-Lochzahl zurück."""
    n_rings = st.session_state[f"{prefix}_ring_count"]
    h1, h2 = st.columns(2)
    h1.caption("Anzahl")
    h2.caption("Lochkreis Ø [mm]")
    total = 0
    for i in range(n_rings):
        pcd_key = f"p_{prefix}_ring_pcd_{i}"
        # Schrumpft die Backplate, wird der gespeicherte Lochkreis still
        # auf den neuen Maximalwert geklammert (wie Blindlochtiefe).
        st.session_state[pcd_key] = min(st.session_state[pcd_key],
                                        bp_diameter_mm)
        c1, c2 = st.columns(2)
        # Beschriftung nur für Screenreader (kompakte Tabellenoptik;
        # die sichtbare Kopfzeile liefern die Captions darüber).
        c1.number_input(f"Anzahl · Kreis {i + 1}", 0, 2000, step=1,
                        key=f"p_{prefix}_ring_n_{i}",
                        label_visibility="collapsed")
        c2.number_input(f"Lochkreis Ø [mm] · Kreis {i + 1}", 0.0,
                        bp_diameter_mm, step=0.5, key=pcd_key,
                        label_visibility="collapsed")
        total += st.session_state[f"p_{prefix}_ring_n_{i}"]
    b1, b2 = st.columns(2)
    b1.button("➕ Lochkreis", key=f"btn_add_{prefix}",
              on_click=_add_ring, args=(prefix,),
              disabled=n_rings >= MAX_RINGS, width="stretch",
              help="Fügt einen weiteren Lochkreis hinzu (je Druck ein "
                   "Kreis), um reale Lochmuster nachzubilden.")
    b2.button("➖ letzter Kreis", key=f"btn_del_{prefix}",
              on_click=_remove_ring, args=(prefix,),
              disabled=n_rings <= 1, width="stretch",
              help="Entfernt den letzten Lochkreis.")
    if n_rings > 1:
        st.caption(f"gesamt: {total} Löcher auf {n_rings} Lochkreisen")
    return total


def build_capsule(p):
    """Anzeigewerte -> SI -> MicrophoneCapsule."""
    return MicrophoneCapsule(
        membrane_material=MATERIAL_LABELS[p["material"]],
        membrane_resonance_hz=p["f_res_hz"] if p["use_f_res"] else None,
        membrane_diameter=p["mem_diameter_mm"] * 1e-3,
        membrane_thickness=p["mem_thickness_um"] * 1e-6,
        membrane_tension=p["mem_tension_npm"],
        air_gap=p["air_gap_um"] * 1e-6,
        backplate_diameter=p["bp_diameter_mm"] * 1e-3,
        backplate_thickness=p["bp_thickness_mm"] * 1e-3,
        bias_voltage=p["bias_v"],
        architecture=ARCH_LABELS[p["architecture"]],
        center_gap=p["center_gap_um"] * 1e-6,
        # Lochkreis-Ø 0 in der GUI = dieser Anteil gleichmäßig verteilt
        through_hole_rings=[(int(n), d * 1e-3 if d > 0 else None)
                            for n, d in p["th_rings"]],
        through_hole_diameter=p["d_through_mm"] * 1e-3,
        blind_hole_rings=[(int(n), d * 1e-3 if d > 0 else None)
                          for n, d in p["bh_rings"]],
        blind_hole_diameter=p["d_blind_mm"] * 1e-3,
        blind_hole_depth=p["blind_depth_mm"] * 1e-3,
        through_holes_stepped=p["th_stepped"],
        clamp_ring_thickness=p["clamp_ring_mm"] * 1e-3,
        clamp_ring_width=p["clamp_width_mm"] * 1e-3,
        clearance_ring_diameter=p["clr_dia_mm"] * 1e-3,
        clearance_ring_width=p["clr_width_mm"] * 1e-3,
        clearance_ring_depth=p["clr_depth_mm"] * 1e-3,
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
        body_diameter=p["body_diameter_mm"] * 1e-3,
        include_diffraction=p["diffraction_on"],
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
    )


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
            progress(0.0, "Modell aufbauen")
        capsule = build_capsule(params)
        store[key] = capsule
        while len(store) > _CAPSULE_CACHE_MAX:
            store.pop(next(iter(store)))
    return capsule


def compute_results(cache_key, capsule, progress=None):
    """Frequenzgang, Richtdiagramme, Diagnose und Summary — gecacht.

    ``cache_key`` ist das JSON der physikrelevanten Parameter. Der
    Frequenzgang wird blockweise gerechnet, damit ``progress`` (Callback
    frac, label) einen echten Fortschritt melden kann; die Blöcke sind
    numerisch identisch zum Gesamtaufruf, weil alle Modelle je Frequenz
    unabhängig rechnen (frequency_response-Logik gespiegelt). Auch der
    summary()-Text gehört hierher: er enthält eine Netzwerkauswertung
    (bei 3D eine volle LU-Lösung) und würde sonst bei JEDEM Rerun im
    Diagnose-Expander mitgerechnet — Streamlit führt auch eingeklappte
    Expander-Inhalte aus.
    """
    store = _session_store("_results_cache")
    hit = store.get(cache_key)
    if hit is not None:
        return hit
    p = json.loads(cache_key)
    n_pts = int(p["n_points"])
    dir_freqs = sorted(p["dir_freqs"]) or [1000]
    n_work = n_pts + len(dir_freqs)
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
    _tick(0, "Frequenzgang")
    f = np.logspace(np.log10(10.0), np.log10(25000.0), n_pts)
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
        _tick(j - i, "Frequenzgang")
    amp_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-30))
    ref_db = np.interp(np.log10(1000.0), np.log10(f), amp_db)
    fr = {
        "frequency_hz": f,
        "sensitivity_v_pa": H,
        "amplitude_db": amp_db,
        "amplitude_db_norm": amp_db - ref_db,
        "phase_deg": np.rad2deg(np.unwrap(np.angle(H))),
    }
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
        _tick(1, f"Richtdiagramm {fd:.0f} Hz")
    sens_1k = float(abs(capsule.transfer_function(np.array([1000.0]))[0]))
    delay = capsule.delay_diagnostics()
    summary_text = capsule.summary()
    result = {"fr": fr, "di": di, "aux": aux, "sens_1k": sens_1k,
              "delay": delay, "summary": summary_text}
    store[cache_key] = result
    while len(store) > _RESULTS_CACHE_MAX:
        store.pop(next(iter(store)))
    return result


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
    amp_title = "Amplitude [dB rel. 1 kHz]" if normalized \
        else "Amplitude [dB re 1 V/Pa]"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.58, 0.42])
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=amp, mode="lines",
        line=dict(color=SERIES[0], width=2), name="Amplitude",
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=fr["phase_deg"], mode="lines",
        line=dict(color=SERIES[4], width=2), name="Phase",
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
    fig.update_xaxes(title_text="Frequenz [Hz]", title_font=dict(color=INK_2),
                     row=2, col=1)
    fig.update_yaxes(title_text=amp_title, row=1, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2),
                     zeroline=False)
    fig.update_yaxes(title_text="Phase [°]", row=2, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2),
                     zeroline=False)
    fig.update_layout(showlegend=False, hovermode="x unified",
                      title=dict(text="Frequenzgang (0° Einfall)",
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 560)


def rear_bode_figure(fr, aux):
    """Richtwirkung über die Frequenz: Pegel bei 90°/180° relativ zu 0°."""
    f = fr["frequency_hz"]
    fig = go.Figure()
    fig.add_hline(y=-6.0, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="−6 dB (ideale Niere @90°)",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_trace(go.Scatter(
        x=f, y=aux["level_90_db"], mode="lines", name="90° rel. 0°",
        line=dict(color=SERIES[1], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra>90°</extra>"))
    fig.add_trace(go.Scatter(
        x=f, y=aux["level_180_db"], mode="lines", name="180° rel. 0°",
        line=dict(color=SERIES[5], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra>180°</extra>"))
    fig.update_xaxes(type="log", gridcolor=GRID, griddash="dot",
                     linecolor=AXIS, tickcolor=AXIS,
                     tickfont=dict(color=MUTED), zeroline=False,
                     tickvals=[10, 100, 1000, 10000],
                     ticktext=["10", "100", "1k", "10k"],
                     title_text="Frequenz [Hz]",
                     title_font=dict(color=INK_2))
    y_min = float(min(np.min(aux["level_180_db"]), -20.0))
    fig.update_yaxes(title_text="Pegel rel. 0° [dB]",
                     range=[max(y_min - 3.0, -45.0), 3.0],
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.0, xanchor="right", x=1.0,
                                  font=dict(color=INK_2)),
                      title=dict(text="Richtwirkung über die Frequenz "
                                      "(seitlich/rückwärtig)",
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 420)


def dr_figure(fr, aux):
    """Phasenschieber-Diagnose: |D_r| und Phase vs. externes Ziel G(180°)."""
    f = fr["frequency_hz"]
    D_r, G180 = aux["D_r"], aux["G180"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=f, y=np.abs(D_r), mode="lines", name="|D_r| intern",
        line=dict(color=SERIES[0], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.2f}<extra>|D_r|</extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=np.abs(G180), mode="lines", name="|G(180°)| extern (Ziel)",
        line=dict(color=SERIES[2], width=2, dash="dash"),
        hovertemplate="%{x:.0f} Hz · %{y:.2f}<extra>|G|</extra>"),
        row=1, col=1)
    ph_d = np.rad2deg(np.unwrap(np.angle(D_r)))
    ph_g = np.rad2deg(np.unwrap(np.angle(G180)))
    fig.add_trace(go.Scatter(
        x=f, y=ph_d, mode="lines", name="arg D_r intern",
        line=dict(color=SERIES[4], width=2),
        hovertemplate="%{x:.0f} Hz · %{y:.1f}°<extra>arg D_r</extra>"),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=ph_g, mode="lines", name="arg G(180°) extern (Ziel)",
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
    fig.update_xaxes(title_text="Frequenz [Hz]",
                     title_font=dict(color=INK_2), row=2, col=1)
    # Betragsachse deckeln: oberhalb der Resonanz explodiert |D_r|
    dmax = float(np.max(np.abs(D_r)))
    fig.update_yaxes(title_text="Betrag [–]", row=1, col=1,
                     range=[0.0, min(max(1.6, 1.1 * dmax), 4.0)],
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_yaxes(title_text="Phase [°]", row=2, col=1,
                     gridcolor=GRID, griddash="dot", linecolor=AXIS,
                     tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2), zeroline=False)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1.0,
                                  font=dict(color=INK_2)),
                      title=dict(text="Phasenschieber D_r — trifft er das "
                                      "externe Ziel?",
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 420)


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
        title=dict(text="Richtdiagramm (normiert auf 0°)",
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
    return _base_layout(fig, 560)


# ---------------------------------------------------------------------------
# Sidebar: Projekt + Parameter
# ---------------------------------------------------------------------------
_init_state()

with st.sidebar:
    st.title("🎙️ Capsim")
    st.caption("Lumped-Element-Simulation einer Kondensatormikrofonkapsel")

    # ---------------- Projekt speichern / laden ------------------------
    with st.expander("📁 Projekt", expanded=False):
        st.file_uploader(
            "Projekt laden (.json)", type=["json"],
            key="project_upload", on_change=_load_project,
        )
        project_json = json.dumps(
            {
                "format": "capsim-project",
                # Version 2: Lochmuster als Lochkreis-Listen th_rings/
                # bh_rings; Version-1-Projekte (n_through/th_pcd_mm, ...)
                # werden beim Laden automatisch konvertiert.
                "version": 2,
                "saved": _dt.date.today().isoformat(),
                "params": _current_params(),
            },
            indent=2, ensure_ascii=False,
        )
        st.download_button(
            "💾 Projekt speichern (.json)", data=project_json,
            file_name="capsim_projekt.json", mime="application/json",
            width="stretch",
        )
    if "_load_msg" in st.session_state:
        kind, msg = st.session_state.pop("_load_msg")
        (st.success if kind == "success" else st.error)(msg)

    # ---------------- Membran ------------------------------------------
    with st.expander("Membran", expanded=True):
        st.selectbox("Material", list(MATERIAL_LABELS), key="p_material")
        st.checkbox("Resonanzfrequenz vorgeben", key="p_use_f_res",
                    help="Deaktiviert: Resonanz wird aus Vorspannung, "
                         "Biegesteifigkeit und Flächendichte berechnet.")
        st.number_input("Resonanzfrequenz [Hz]", 100.0, 50000.0, step=100.0,
                        format="%.0f", key="p_f_res_hz",
                        disabled=not st.session_state["p_use_f_res"])
        st.number_input("Durchmesser [mm]", 3.0, 60.0, step=0.5,
                        key="p_mem_diameter_mm")
        st.number_input("Dicke [µm]", 0.5, 100.0, step=0.5,
                        key="p_mem_thickness_um")
        st.number_input("Vorspannung [N/m]", 1.0, 5000.0, step=10.0,
                        key="p_mem_tension_npm")

    # ---------------- Backplate ----------------------------------------
    with st.expander("Backplate", expanded=True):
        st.number_input("Luftspalt [µm]", 5.0, 500.0, step=1.0,
                        key="p_air_gap_um")
        st.number_input("Durchmesser [mm]", 2.0, 60.0, step=0.5,
                        key="p_bp_diameter_mm")
        st.number_input("Dicke [mm]", 0.2, 20.0, step=0.1,
                        key="p_bp_thickness_mm")
        st.number_input("Polarisationsspannung [V]", 0.5, 400.0, step=1.0,
                        key="p_bias_v")
        st.radio("Architektur", list(ARCH_LABELS), key="p_architecture",
                 help="K67-Bauform: zwei Membranen außen, zwei innen-"
                      "liegende Backplates, getrennt nur durch den "
                      "Backplate-Spalt. Die passive Rückmembran bildet "
                      "das Phasenschiebernetzwerk (Niere) — Laufzeitglied "
                      "und Hohlraum entfallen.")
        st.number_input("Backplate-Spalt (K67) [µm]", 0.0, 500.0, step=5.0,
                        key="p_center_gap_um",
                        disabled=st.session_state["p_architecture"]
                        != K67_LABEL,
                        help="Spacer zwischen den beiden Backplate-Hälften "
                             "der Doppelmembran-Bauform. 0 = einteilige, "
                             "komplett durchbohrte Mittelelektrode "
                             "(Braunmühl-Weber); die Backplate-Dicke ist "
                             "dann die HALBE Plattendicke je Seite.")

        st.markdown("**Lochmuster**")
        st.caption("Jeder Lochtyp lässt sich mit ➕ auf mehrere Lochkreise "
                   "verteilen (je Kreis: Anzahl + Sitz-Ø; Lochkreis-Ø 0 = "
                   "gleichmäßig verteilt). Die radiale Anordnung wirkt im "
                   "2D-Feldmodell und auf die Elektrodenporosität; das "
                   "1D-Spaltmodell nutzt nur die Gesamtzahlen.")
        _bp_d = st.session_state["p_bp_diameter_mm"]

        st.markdown("Durchgangslöcher",
                    help="Die Durchgangslöcher sind der einzige Weg durch "
                         "die Backplate. 0 Löcher insgesamt = Backplate "
                         "geschlossen → Kapsel hermetisch dicht "
                         "(Druckempfänger), unabhängig von der Rückseite. "
                         "Bei Dual-Architektur ist mindestens 1 Loch nötig.")
        _ring_rows("th", _bp_d)
        st.number_input("Durchgangslöcher — Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_through_mm",
                        help="Bohrungsdurchmesser (gilt für alle "
                             "Lochkreise dieses Typs).")
        st.toggle("Stufenbohrung (konzentrisch im Sackloch)",
                  key="p_th_stepped",
                  help="K67/K87-Bauweise: jedes Durchgangsloch sitzt am "
                       "GRUND eines Sacklochs mit Sackloch-Ø und Sackloch-"
                       "Tiefe — nur die Restdicke der Platte ist eng "
                       "durchbohrt. Zählweise NUR bei aktivem Schalter: "
                       "Blindlöcher = GESAMTZAHL aller Sacklöcher, "
                       "Durchgangslöcher = wie viele davon zusätzlich "
                       "durchgebohrt sind (K67: 120 Sacklöcher, davon 60 "
                       "durchgebohrt). Bei ausgeschaltetem Schalter bleiben "
                       "beide Lochtypen unabhängig wie bisher. Erfordert "
                       "Sackloch-Ø > Durchgangsloch-Ø und Durchgangs- ≤ "
                       "Blindlochzahl.")

        st.markdown("Blindlöcher",
                    help="Sacklöcher auf der Membranseite: Dämpfungs- und "
                         "Volumen-Bohrungen, kein Weg durch die Platte.")
        _ring_rows("bh", _bp_d)
        st.number_input("Blindlöcher — Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_blind_mm",
                        help="Bohrungsdurchmesser (gilt für alle "
                             "Lochkreise dieses Typs).")
        _bd_max = max(0.1, st.session_state["p_bp_thickness_mm"] - 0.1)
        st.session_state["p_blind_depth_mm"] = min(
            st.session_state["p_blind_depth_mm"], _bd_max)
        st.number_input("Blindlöcher — Tiefe [mm]", 0.05, _bd_max, step=0.05,
                        key="p_blind_depth_mm")

        st.markdown("**Clearance-Ring (Freistich der Stirnflächen)**",
                    help="Ringförmiger Freistich in den Elektroden-Stirn"
                         "flächen (je Seite): Position über den Ring-Ø, "
                         "radiale Breite, axiale Tiefe. Breite Ringe "
                         "(≥ 1 Gitterzelle) vertiefen den Spalt lokal und "
                         "ENTLASTEN die Mündungs-Engstellen dort sitzender "
                         "Bohrungen — entscheidend für die Nierentiefe bei "
                         "wenigen engen Durchgangslöchern (z. B. Debenham). "
                         "Sehr schmale Ringe wirken als Schlitz-Stub. Nur "
                         "im 2D-Feldmodell wirksam. 0 = kein Ring.")
        st.number_input("Clearance-Ring — Ø [mm]", 0.0, 60.0, step=0.5,
                        key="p_clr_dia_mm",
                        help="Mittlerer Durchmesser des Rings (z. B. der "
                             "Lochkreis der Durchgangslöcher).")
        st.number_input("Clearance-Ring — Breite [mm]", 0.0, 10.0, step=0.1,
                        key="p_clr_width_mm",
                        help="Radiale Breite des Freistichs.")
        st.number_input("Clearance-Ring — Tiefe [mm]", 0.0, 5.0, step=0.01,
                        format="%.3f", key="p_clr_depth_mm",
                        help="Axialer Abtrag (zusätzliche Spalthöhe im "
                             "Ringbereich).")

        # Klemmringe vor den Membranen — nur bei K67-Bauform relevant
        if st.session_state["p_architecture"] == K67_LABEL:
            st.markdown("**Klemmringe (vor den Membranen)**",
                        help="Ringe vor beiden Membranen (K67/K87). Sie "
                             "versenken die Membranen um ihre Dicke → die "
                             "geometrische Front-Rück-Distanz d_ext wächst "
                             "um 2×Dicke. Die Nierennull entsteht, wenn die "
                             "interne Laufzeit des Phasenschieber-Netzwerks "
                             "(Bohrungen, Spaltfilme, Spacer) diese externe "
                             "Laufzeit trifft. Die Breite geht nur in den "
                             "Außenradius ein. 0 = keine Ringe.")
            st.number_input("Klemmring — Dicke je Seite [mm]", 0.0, 10.0,
                            step=0.5, key="p_clamp_ring_mm",
                            help="Axiale Auftragung vor jeder Membran.")
            st.number_input("Klemmring — Breite [mm]", 0.0, 10.0, step=0.5,
                            key="p_clamp_width_mm",
                            help="Radiale Ausdehnung des Rings.")

    # ---------------- Rückseite / Laufzeitglied -------------------------
    _is_k67 = st.session_state["p_architecture"] == K67_LABEL
    with st.expander("Rückseite & Laufzeitglied", expanded=not _is_k67):
        if _is_k67:
            st.caption("Bei der K67-Bauform übernimmt die passive "
                       "Rückmembran diese Funktion — die folgenden "
                       "Parameter sind inaktiv.")
        st.toggle("Rückseite aktiv", key="p_rear_enabled",
                  disabled=_is_k67,
                  help="Deaktiviert: die rückwärtige Baugruppe (Laufzeit-"
                       "glied, Hohlraum, Einlasslöcher) entfällt — die "
                       "Durchgangslöcher der Backplate münden dann durch "
                       "das rückwärtige Gewebe DIREKT ins Schallfeld "
                       "(einfacher Gradientenempfänger, Wegdifferenz = "
                       "Spalt + Backplate-Dicke). Hermetisch geschlossen "
                       "ist die Kapsel nur mit 0 Durchgangslöchern.")
        _rear_on = st.session_state["p_rear_enabled"] and not _is_k67
        st.markdown("**Spacer & Rückplatte (K103-Bauform)**",
                    help="Direkt hinter der Backplate: dünner Distanzring "
                         "(Spacer) und massive, gelochte Rückplatte — wie "
                         "beim Neumann K103 (TLM 103), dessen K87-artige "
                         "Front statt einer Rückmembran durch eine Platte "
                         "abgeschlossen ist. Der enge Spacer liefert den "
                         "Reibungswiderstand des Nieren-Phasenschiebers. "
                         "Sind Laufzeitglied, Hohlraum und Einlasslöcher 0, "
                         "münden die Plattenlöcher direkt ins rückwärtige "
                         "Schallfeld. Eine Rückplatte ohne Löcher "
                         "verschließt die Kapsel (Druckempfänger).")
        st.number_input("Spacer — Höhe [µm]", 0.0, 1000.0, step=5.0,
                        key="p_spacer_um", disabled=not _rear_on,
                        help="Luftschicht zwischen Backplate und Rück-"
                             "platte. 0 = kein Spacer. Enger Spalt = mehr "
                             "Reibung (R ~ 1/h³) — das Abstimmelement der "
                             "Richtcharakteristik.")
        st.number_input("Rückplatte — Dicke [mm]", 0.0, 20.0, step=0.1,
                        key="p_rearplate_mm", disabled=not _rear_on,
                        help="Massive Platte hinter dem Spacer. "
                             "0 = keine Rückplatte.")
        _rp_on = _rear_on and st.session_state["p_rearplate_mm"] > 0.0
        st.number_input("Rückplatte — Löcher Anzahl", 0, 2000, step=1,
                        key="p_n_rearplate", disabled=not _rp_on,
                        help="0 = Rückplatte ohne Löcher → Rückseite "
                             "verschlossen (Druckempfänger).")
        st.number_input("Rückplatte — Löcher Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_rearplate_mm", disabled=not _rp_on)

        st.markdown("**Laufzeitglied & Hohlraum**")
        st.number_input("Laufzeitglied — Länge [mm]", 0.0, 100.0, step=0.5,
                        key="p_delay_mm", disabled=not _rear_on,
                        help="Akustische Leitung hinter der Membran; "
                             "Laufzeit τ = L/c.")
        st.number_input("Hohlraum — Länge [mm]", 0.0, 100.0, step=0.5,
                        key="p_cavity_length_mm", disabled=not _rear_on)
        st.number_input("Hohlraum — Wandstärke [mm]", 0.0, 10.0, step=0.1,
                        key="p_cavity_wall_mm", disabled=not _rear_on)
        st.radio("Hohlraumlöcher — Position", list(POS_LABELS),
                 key="p_hole_position", disabled=not _rear_on)
        st.number_input("Hohlraumlöcher — Anzahl", 0, 5000, step=1,
                        key="p_n_cavity", disabled=not _rear_on,
                        help="0 = Rückseite geschlossen → Druckempfänger "
                             "(Kugelcharakteristik).")
        st.number_input("Hohlraumlöcher — Ø [mm]", 0.0, 5.0, step=0.05,
                        key="p_d_cavity_mm", disabled=not _rear_on,
                        help="0 = Rückseite geschlossen.")
        _ax_max = st.session_state["p_cavity_length_mm"]
        st.session_state["p_cavity_axial_mm"] = min(
            st.session_state["p_cavity_axial_mm"], _ax_max)
        st.number_input("Hohlraumlöcher — axiale Position [mm]", 0.0,
                        max(_ax_max, 0.5), step=0.5, key="p_cavity_axial_mm",
                        disabled=(not _rear_on)
                        or st.session_state["p_hole_position"] != "Umfang"
                        or _ax_max <= 0.0,
                        help="Abstand vom Hohlraumeingang "
                             "(nur bei Position 'Umfang').")

    # ---------------- Akustisches Gewebe --------------------------------
    with st.expander("Akustisches Gewebe", expanded=True):
        st.number_input("Vor der Membran [Rayl]", 0.0, 100000.0, step=5.0,
                        key="p_fabric_front_rayl")
        st.number_input("Hinter der Backplate [Rayl]", 0.0, 100000.0,
                        step=5.0, key="p_fabric_rear_rayl")

    # ---------------- Gehäuse & Beugung ----------------------------------
    with st.expander("Gehäuse & Beugung", expanded=False):
        st.toggle("Beugung am Gehäuse (Druckstau)", key="p_diffraction_on",
                  help="Streuung der ebenen Welle am starren Kugel-Ersatz-"
                       "gehäuse (Morse-Reihe): frontaler Druckstau bis "
                       "+6 dB, rückwärtige Abschattung und Apertureffekt "
                       "der Membran. Dadurch richtet auch ein reiner "
                       "Druckempfänger zu hohen Frequenzen hin — wie in "
                       "der Realität. Deaktivieren nur zum Vergleich mit "
                       "dem idealisierten Punktmodell.")
        _bd_min = max(st.session_state["p_mem_diameter_mm"],
                      st.session_state["p_bp_diameter_mm"])
        st.session_state["p_body_diameter_mm"] = max(
            st.session_state["p_body_diameter_mm"], _bd_min)
        st.number_input("Gehäusedurchmesser [mm]", _bd_min, 100.0, step=0.5,
                        key="p_body_diameter_mm",
                        disabled=not st.session_state["p_diffraction_on"],
                        help="Durchmesser des kugelförmigen Ersatz-"
                             "gehäuses; bestimmt, ab welcher Frequenz "
                             "Druckstau und Abschattung einsetzen: "
                             "ka = 1 bei f ≈ 109/d Hz (d in m) — für "
                             "Ø 26 mm also ab ≈ 4 kHz.")
        if st.session_state["p_architecture"] == K67_LABEL:
            st.selectbox("Axialer Körper (Front-Rück-Transfer)",
                         list(AX_LABELS), key="p_axial_body",
                         disabled=not st.session_state["p_diffraction_on"],
                         help="Referenzkörper für den axialen Front-Rück-"
                              "Transfer G(180°) der Doppelmembran-Bauform. "
                              "Kugel (d_ext): Standard — beschreibt die am "
                              "Mikrofonkörper MONTIERTE Kapsel (der Körper "
                              "unterbindet den Scheibenrand-Umweg). "
                              "Sphäroid: exakte Streuung an der FREI "
                              "stehenden Scheibe (radial R_body, axial "
                              "d_ext/2, eigene Spezialfunktionen mit "
                              "Wronski-Selbstprüfung) — deutlich längere "
                              "effektive Distanz (K67: ~32 statt 18 mm, "
                              "dünne Scheibe: 4R/π am Pol), Minimum "
                              "wandert weit vor 180°. Dokumentierter "
                              "Referenzfall, s. Gegenprobe 20. BEM: "
                              "montagetreues Randelementverfahren auf der "
                              "Kontur Kopf + Mikrofonkörper (m=0, gegen "
                              "Kugel- UND Sphäroid-Reihe validiert) — "
                              "liegt zwischen beiden Referenzkörpern. "
                              "DEUTLICH langsamer (~1-2 s je Frequenz-"
                              "punkt), Ergebnisse werden gecacht.")
            if AX_LABELS.get(st.session_state["p_axial_body"]) == "bem":
                st.number_input("BEM: Körper-Ø [mm]", 0.0, 200.0, step=1.0,
                                key="p_bem_body_dia_mm",
                                help="Mikrofonkörper-Zylinder unter dem "
                                     "Kapselkopf; 0 = frei stehender Kopf "
                                     "(Scheiben-Referenz).")
                st.number_input("BEM: Luftspalt Kopf→Körper [mm]", 3.0,
                                100.0, step=1.0, key="p_bem_body_gap_mm",
                                help="Axialer Abstand zwischen Kopf-Rück"
                                     "seite und Körper-Oberseite.")
                st.number_input("BEM: Körperlänge [mm]", 10.0, 300.0,
                                step=5.0, key="p_bem_body_len_mm",
                                help="Länge des Körperzylinders (endlich, "
                                     "verrundet gekappt).")
                st.caption("⏳ BEM rechnet je Frequenzpunkt ein Rand"
                           "elementsystem (~200 Elemente). Empfehlung: "
                           "Frequenzpunkte ≤ 150.")

    # ---------------- Spaltfilm-Modell -----------------------------------
    with st.expander("Spaltfilm-Modell", expanded=False):
        st.toggle("2D-Feldmodell (modifizierte Reynolds-Gleichung)",
                  key="p_squeeze_2d",
                  help="Aus: der Luftspalt ist ein Lumped-Element (Škvor-"
                       "Widerstand + Nachgiebigkeit + Lochimpedanz). Schnell "
                       "und für dichte gleichmäßige Lochmuster ausreichend.\n\n"
                       "An: das Druckfeld im Spalt wird als modifizierte "
                       "Reynolds-Gleichung (Homentcovschi & Miles) axial-"
                       "symmetrisch gelöst. Trennt den Nachgiebigkeits-"
                       "Rückweg (Spaltvolumen + Blindlöcher) vom Rück-"
                       "kopplungsweg (nur Durchgangslöcher) und erfasst den "
                       "radialen Druckaufbau. Beseitigt die überhöhte "
                       "Spaltresonanz bei wenigen engen Löchern und nutzt "
                       "die Lochkreis-Radien (PCD). Etwas langsamer.")
        if st.session_state["p_squeeze_2d"]:
            st.caption("Die Lochkreise im Abschnitt Backplate steuern "
                       "jetzt die radiale Lochverteilung im Spaltfeld.")
        st.toggle("3D-Feldmodell (diskrete Löcher, r-φ-Sandwich)",
                  key="p_squeeze_3d",
                  help="Volles (r, φ)-Feldmodell: alle Spaltfilme UND "
                       "beide Membranen als Felder, Durchgangs- und "
                       "Sacklöcher sitzen DISKRET an ihren Positionen "
                       "(azimutale Zuströmung und teilentkoppelte Sack-"
                       "löcher werden aufgelöst; bedämpft die interne "
                       "Helmholtz-Resonanz realistisch). Nur für die "
                       "Doppelmembran-Bauform mit Durchgangslöchern. "
                       "Bei Mittelabstand > 0 rechnet der Löser die "
                       "ZWEITEILIGE Elektrode (K67-Typ): Zwischenspalt "
                       "als dritter Film, Stufenbohrungen als Zweitor-"
                       "Kette je Loch, Elektrodenhälften gegeneinander "
                       "verdreht. Hat Vorrang vor dem 2D-Schalter. "
                       "DEUTLICH langsamer (einteilig ~1–2 s, K67-Typ "
                       "je nach Lochzahl bis ~10 s je Frequenzpunkt) — "
                       "Frequenzpunkte reduzieren!")
        if st.session_state["p_squeeze_3d"]:
            st.toggle("Verdrehung der Hälften automatisch "
                      "(halbe Lochteilung)",
                      key="p_half_rot_auto",
                      help="Nur K67-Typ (Mittelabstand > 0): die realen "
                           "Elektrodenhälften sind so verdreht, dass die "
                           "Durchgangslöcher nicht zueinander zeigen — "
                           "automatisch 180°/n_Durchgangslöcher (bei 60 "
                           "Löchern also 3°). Ausgerichtete Löcher (0°) "
                           "kurzschließen den Nieren-Phasenschieber "
                           "durch den Zwischenspalt: flachere 180°-Aus"
                           "löschung, höhere Empfindlichkeit.")
            if not st.session_state["p_half_rot_auto"]:
                st.number_input("Verdrehung der Elektrodenhälften [°]",
                                0.0, 180.0, step=0.5,
                                key="p_half_rot_deg",
                                help="0° = Durchgangslöcher beider Hälften "
                                     "zeigen aufeinander (Kurzschluss des "
                                     "Phasenschiebers); halbe Teilung = "
                                     "maximaler Versatz wie an der realen "
                                     "K67.")
            st.caption("⏳ 3D rechnet je Frequenzpunkt eine LU-Faktori"
                       "sierung (einteilig ~24 000 Unbekannte; K67-Typ "
                       "mit drittem Film und feinerer Azimut-Auflösung "
                       "entsprechend mehr). Empfehlung: Frequenzpunkte "
                       "≤ 150 (K67-Typ ≤ 100). Ergebnisse werden je "
                       "Parametersatz gecacht — Reruns ohne Änderung "
                       "sind sofort da.")

    # ---------------- Simulation ---------------------------------------
    with st.expander("Simulation", expanded=False):
        st.slider("Frequenzpunkte", 100, 1500, step=50, key="p_n_points")
        st.toggle("Amplitude auf 1 kHz normieren", key="p_normalize_1khz")
        st.multiselect("Richtdiagramm-Frequenzen [Hz]", DIRECTIVITY_OPTIONS,
                       key="p_dir_freqs", max_selections=10)

# ---------------------------------------------------------------------------
# Berechnung
# ---------------------------------------------------------------------------
params = _current_params()

# Platzhalter VOR dem Modellaufbau: schon der Konstruktor (Elektrostatik,
# bei 3D der Gitteraufbau) soll als Fortschritt sichtbar sein.
_prog_slot = st.empty()


def _show_progress(frac, label):
    _prog_slot.progress(frac, text=f"⚙️ Berechnung — {label} … "
                                   f"{100 * frac:.0f} %")


try:
    capsule = get_capsule(params, _show_progress)
except ValueError as exc:
    _prog_slot.empty()
    st.error(f"⚠️ Ungültige Parameterkombination: {exc}")
    st.stop()

# Cache-Schlüssel: alle Parameter, die Physik oder berechnete Daten ändern.
# Reine Darstellungsoptionen (Normierungs-Toggle) bleiben draußen, damit
# ihr Umschalten keine Neuberechnung auslöst.
_key_params = {**params, "dir_freqs": sorted(params["dir_freqs"])}
_key_params.pop("normalize_1khz", None)
_cache_key = json.dumps(_key_params, sort_keys=True)

_res = compute_results(_cache_key, capsule, _show_progress)
_prog_slot.empty()
fr, di, aux = _res["fr"], _res["di"], _res["aux"]
sens_1k, delay, summary_text = _res["sens_1k"], _res["delay"], _res["summary"]

# ---------------------------------------------------------------------------
# Hauptbereich
# ---------------------------------------------------------------------------
st.title("Kondensatormikrofonkapsel — Simulation")
st.caption("Elektroakustisches Ersatzschaltbild (Lumped-Element-Modell) · "
           "10 Hz – 25 kHz")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Empfindlichkeit @ 1 kHz", f"{sens_1k * 1e3:.1f} mV/Pa",
          help=f"{20 * np.log10(max(sens_1k, 1e-12)):.1f} dB re 1 V/Pa "
               "(Leerlauf, ohne Streukapazität)")
m2.metric("Membranresonanz", f"{capsule.f_res:.0f} Hz",
          help="Konsistenz-Check aus Vorspannung/Biegesteifigkeit: "
               f"{capsule.f_res_from_tension:.0f} Hz")
m3.metric("Ruhekapazität C₀", f"{capsule.C_elec_0 * 1e12:.1f} pF",
          help=f"je Backplate · Architektur: {capsule.n_bp} Backplate(s)")
_upi = (f"{capsule.U_pullin:.0f} V" if np.isfinite(capsule.U_pullin)
        else "> 20 kV")
m4.metric("Feder-Erweichung (Bias)",
          f"{capsule.softening_ratio * 100:.1f} %",
          help="Anteil der Membransteifigkeit, den die elektrostatische "
               "Anziehung am Arbeitspunkt aufzehrt. Pull-in-Spannung "
               f"dieser Konfiguration: ≈ {_upi}; statische Durchbiegung "
               f"{capsule.w0_static * 1e6:.1f} µm (Restspalt Mitte "
               f"{capsule.h_min_static * 1e6:.1f} µm).")

# Laufzeit-Anpassung des Nieren-Phasenschiebers: die 180°-Null entsteht,
# wenn die interne Rück-Übertragung D_r des Netzwerks die externe
# Front-Rück-Übertragung (Körperbeugung) trifft. Beide als äquivalente
# Laufzeit bei 1 kHz ausgewertet (s. delay_diagnostics; die interne
# Laufzeit ist frequenzabhängig — RC-Glied, kein reines Laufzeitglied).
if delay is not None:
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Externe Laufzeit τ_ext",
              f"{delay['tau_ext_s'] * 1e6:.1f} µs",
              help="Front-Rück-Übertragung G(180°) des Schallfelds um "
                   "den Kapselkörper (axiale Körperbeugung), als Phase "
                   f"bei {delay['f_probe_hz']:.0f} Hz ausgewertet. "
                   "Äquivalente Wegstrecke: "
                   f"{delay['dist_ext_m'] * 1e3:.1f} mm.")
    l2.metric("Interne Laufzeit τ_int",
              f"{delay['tau_int_s'] * 1e6:.1f} µs",
              help="Rück-Übertragung D_r des internen Phasenschieber-"
                   "Netzwerks (Bohrungen, Spaltfilme, Spacer, Rückseite), "
                   f"als Phase bei {delay['f_probe_hz']:.0f} Hz "
                   "ausgewertet — frequenzabhängig, da RC-Phasenschieber "
                   "mit Filmträgheit. Äquivalente Wegstrecke: "
                   f"{delay['dist_int_m'] * 1e3:.1f} mm.")
    _ratio = delay["ratio"]
    l3.metric("Verhältnis intern / extern", f"{_ratio:.2f}",
              delta=f"{(_ratio - 1.0) * 100:+.0f} % vs. Anpassung",
              delta_color="off",
              help="≈ 1: Laufzeiten angepasst → tiefste Auslöschung bei "
                   "180°. < 1: interne Laufzeit zu kurz — das Pattern-"
                   "Minimum wandert vor 180° (Richtung Hyperniere). "
                   "> 1: interne Laufzeit zu lang — das Minimum bleibt "
                   "bei 180° gepinnt, die Auslöschung wird aber flacher. "
                   "Nur nahe der Sondenfrequenz aussagekräftig, wenn die "
                   "interne Helmholtz-Resonanz im Band liegt!")
    _fh = aux["f_helmholtz_hz"]
    l4.metric("Interne Helmholtz-Resonanz",
              f"{_fh / 1000:.2f} kHz" if _fh is not None else "> 25 kHz",
              delta=None if _fh is None or _fh > 8000.0
              else "im Übertragungsband!",
              delta_color="off",
              help="Resonanz der Durchgangsloch-Trägheit gegen die innere "
                   "Nachgiebigkeit (Spalt + Blindlöcher), bestimmt als "
                   "90°-Phasendurchgang von D_r. Liegt sie IM Band, "
                   "bricht |D_r| darunter ein und die Pattern-Form "
                   "wandert über die Frequenz (Superniere → breite Niere "
                   "→ Kugel) — Abhilfe: Stufenbohrung, dünnere Platte, "
                   "größere/mehr Durchgangslöcher. Gesund: deutlich "
                   "oberhalb des Übertragungsbands.")

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
        st.info("Rückseite geschlossen (Druckempfänger) — es gibt keinen "
                "internen Phasenschieber-Pfad und damit kein D_r.")

with st.expander("Abgeleitete Modellparameter (Diagnose)"):
    # gecachter Text: summary() enthält eine Netzwerkauswertung (bei 3D
    # eine volle LU-Lösung) und liefe sonst bei jedem Rerun mit
    st.code(summary_text, language=None)

# ---------------------------------------------------------------------------
# Datenansicht & CSV-Export
# ---------------------------------------------------------------------------
st.subheader("Daten & Export")

df_fr = pd.DataFrame({
    "frequenz_hz": fr["frequency_hz"],
    "empfindlichkeit_mv_pa": np.abs(fr["sensitivity_v_pa"]) * 1e3,
    "amplitude_db_re_1v_pa": fr["amplitude_db"],
    "amplitude_db_norm_1khz": fr["amplitude_db_norm"],
    "phase_deg": fr["phase_deg"],
    "pegel_90_rel0_db": aux["level_90_db"],
    "pegel_180_rel0_db": aux["level_180_db"],
})
if aux["D_r"] is not None:
    df_fr["dr_betrag"] = np.abs(aux["D_r"])
    df_fr["dr_phase_deg"] = np.rad2deg(np.unwrap(np.angle(aux["D_r"])))
df_di = pd.DataFrame({"winkel_deg": di["angles_deg"]})
for f, pat in sorted(di["patterns"].items()):
    tag = f"{f:.0f}hz"
    df_di[f"pegel_db_{tag}"] = pat["db"]
    df_di[f"linear_{tag}"] = pat["linear"]

tab_fr, tab_di = st.tabs(["Frequenzgang", "Richtdiagramm"])
with tab_fr:
    st.dataframe(df_fr, height=240, width="stretch", hide_index=True)
with tab_di:
    st.dataframe(df_di, height=240, width="stretch", hide_index=True)

exp1, exp2, exp3 = st.columns([2, 2, 3])
sep_choice = exp3.selectbox(
    "CSV-Format",
    ["Komma / Punkt (international)", "Semikolon / Komma (Excel DE)"],
    key="csv_format",
)
_sep, _dec = (";", ",") if "Semikolon" in sep_choice else (",", ".")


def _to_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=_sep, decimal=_dec)
    return buf.getvalue().encode("utf-8-sig")  # BOM: Umlaute in Excel


exp1.download_button("⬇️ Frequenzgang als CSV", _to_csv(df_fr),
                     file_name="capsim_frequenzgang.csv", mime="text/csv",
                     width="stretch")
exp2.download_button("⬇️ Richtdiagramm als CSV", _to_csv(df_di),
                     file_name="capsim_richtdiagramm.csv", mime="text/csv",
                     width="stretch")

st.caption("Capsim · Physikmodell: ABCD-Kettenmatrizen, Zwikker–Kosten-"
           "Lochimpedanzen, Škvor-Squeeze-Film, elektrostatische Wandlung "
           "mit Feder-Erweichung. Empfindlichkeit = Leerlaufspannung ohne "
           "Streukapazität/Verstärkerlast.")
