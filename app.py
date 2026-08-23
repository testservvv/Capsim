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
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from microphone_capsule import MicrophoneCapsule
from translations import TR, LABEL_TR

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")

# ---------------------------------------------------------------------------
# Sprache (GUI-Einstellung, KEIN Kapselparameter — bleibt außerhalb der
# Projektdateien). Standard: Englisch; umschaltbar in der Seitenleiste.
# ---------------------------------------------------------------------------
def _lang():
    try:
        return st.session_state.get("ui_lang", "en")
    except Exception:      # ohne Streamlit-Runtime (bare Tests)
        return "en"


def tr(key, **kw):
    """Übersetzten GUI-Text holen; Platzhalter per str.format füllen."""
    txt = TR[key][_lang()]
    return txt.format(**kw) if kw else txt


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
    initial_sidebar_state="expanded",
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

# (key, Widget-Art, Default) — Gruppen s. Sidebar-Aufbau weiter unten
# Voreinstellungen beim Start: die K67/K870-Kapsel (U87Ai, Nierenmodus)
# in der echten Doppelmembran-Bauform mit zweiteiliger Elektrode und
# Stufenbohrungen, gerechnet im 2D-Feldmodell — identisch zu
# examples/u87_k67_projekt.json (dort ausführlich dokumentiert).
# Die Debenham-Kapsel liegt weiter unter examples/.
DEFAULTS = {
    # Membran
    "material": "PET (Mylar)",
    "use_f_res": True,
    "f_res_hz": 1150.0,
    "mem_diameter_mm": 26.0,
    "mem_thickness_um": 6.0,
    "mem_tension_npm": 13.7,
    "mem_modes": 1,
    "modal_source": False,
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
    # Simulation
    "n_points": 400,
    "normalize_1khz": True,
    "dir_freqs": [125, 1000, 4000, 8000, 16000],
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
    if key == "fabric_rear_pos" and val in FAB_POS_LABELS:
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
            "success", tr("load_ok", n=len(staged)))
    except Exception as exc:  # defekte Datei darf die App nicht stoppen
        st.session_state["_load_msg"] = (
            "error", tr("load_fail", exc=exc))


# Null-Zustand für den Reset-Button: 0 überall, wo die Widgets 0 zulassen,
# sonst der KLEINSTE zulässige Widget-Wert (die Kapsel bleibt so baubar
# und die App rechenfähig). Auswahl- (Material, Architektur, Position,
# axialer Körper), Modell- (2D/3D, Beugung, Verdrehung) und Simulations-
# einstellungen (Frequenzpunkte, Normierung, Sprache) bleiben unberührt.
_ZERO_STATE = {
    # air_gap 50 statt Widget-Minimum 5 µm: mit der weichsten Membran
    # (f_res 100 Hz) läge die Pull-in-Spannung sonst unter dem kleinsten
    # Bias (0.5 V) — der Null-Zustand muss BAUBAR bleiben (U_PI ≈ 1.2 V).
    "f_res_hz": 100.0, "mem_diameter_mm": 3.0, "mem_thickness_um": 0.5,
    "mem_tension_npm": 1.0, "mem_modes": 1, "modal_source": False,
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
    _set_ring_state("th", [[0, 0.0]])
    _set_ring_state("bh", [[0, 0.0]])
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
    h1.caption(tr("cap_ring_n"))
    h2.caption(tr("cap_ring_pcd"))
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
        c1.number_input(tr("lbl_ring_n", i=i + 1), 0, 2000, step=1,
                        key=f"p_{prefix}_ring_n_{i}",
                        label_visibility="collapsed")
        c2.number_input(tr("lbl_ring_pcd", i=i + 1), 0.0,
                        bp_diameter_mm, step=0.5, key=pcd_key,
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


def build_capsule(p):
    """Anzeigewerte -> SI -> MicrophoneCapsule."""
    return MicrophoneCapsule(
        membrane_material=MATERIAL_LABELS[p["material"]],
        membrane_resonance_hz=p["f_res_hz"] if p["use_f_res"] else None,
        membrane_diameter=p["mem_diameter_mm"] * 1e-3,
        membrane_thickness=p["mem_thickness_um"] * 1e-6,
        membrane_tension=p["mem_tension_npm"],
        membrane_modes=int(p.get("mem_modes", 1)),
        modal_source=1 if p.get("modal_source", False) else 0,
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
            progress(0.0, tr("prog_build"))
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
    _tick(0, tr("prog_fr"))
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
    # Eigenrauschen (thermisch-akustisch, FDT/Nyquist) — nur 1D/2D; der
    # 3D-Feldlöser hat keinen konzentrierten Membranzweig.
    noise = None
    if capsule.squeeze_model != "3d":
        try:
            f_n = np.logspace(np.log10(10.0), np.log10(25000.0), 400)
            spn = capsule.noise_spectrum(f_n)
            noise = {"self": capsule.self_noise(),
                     "f": f_n, "asd": spn["asd_pa_shz"]}
        except Exception:            # Rauschen darf nie die App stoppen
            noise = None
    result = {"fr": fr, "di": di, "aux": aux, "sens_1k": sens_1k,
              "delay": delay, "noise": noise}
    store[cache_key] = result
    while len(store) > _RESULTS_CACHE_MAX:
        store.pop(next(iter(store)))
    return result


def get_summary(cache_key, capsule, lang):
    """Diagnose-Summary getrennt gecacht (Schlüssel inkl. Sprache).

    So bleibt der teure Ergebnis-Cache sprachunabhängig gültig, und ein
    Sprachwechsel formatiert nur den Text neu (bei 3D eine 1-kHz-Lösung
    je Sprache/Parametersatz — danach sofort aus dem Cache).
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
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=amp, mode="lines",
        line=dict(color=SERIES[0], width=2), name=tr("fig_amp_abs"),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fr["frequency_hz"], y=fr["phase_deg"], mode="lines",
        line=dict(color=SERIES[4], width=2), name=tr("fig_phase"),
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
    fig.update_layout(showlegend=False, hovermode="x unified",
                      title=dict(text=tr("fig_bode_title"),
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 560)


def rear_bode_figure(fr, aux):
    """Richtwirkung über die Frequenz: Pegel bei 90°/180° relativ zu 0°."""
    f = fr["frequency_hz"]
    fig = go.Figure()
    fig.add_hline(y=-6.0, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=tr("fig_rear_ann"),
                  annotation_font=dict(color=MUTED, size=11))
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
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.0, xanchor="right", x=1.0,
                                  font=dict(color=INK_2)),
                      title=dict(text=tr("fig_rear_title"),
                                 font=dict(color=INK, size=16)))
    return _base_layout(fig, 420)


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
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1.0,
                                  font=dict(color=INK_2)),
                      title=dict(text=tr("fig_dr_title"),
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
    return _base_layout(fig, 560)


# ---------------------------------------------------------------------------
# Sidebar: Projekt + Parameter
# ---------------------------------------------------------------------------
_init_state()
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
            tr("save_btn"), data=project_json,
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
        (st.success if kind == "success" else st.error)(msg)

    # ---------------- Membran ------------------------------------------
    with st.expander(tr("exp_membrane"), expanded=True):
        st.selectbox(tr("lbl_material"), list(MATERIAL_LABELS),
                     format_func=_label_formatter(), key="p_material")
        st.checkbox(tr("lbl_use_fres"), key="p_use_f_res",
                    help=tr("help_use_fres"))
        st.number_input(tr("lbl_fres"), 100.0, 50000.0, step=100.0,
                        format="%.0f", key="p_f_res_hz",
                        disabled=not st.session_state["p_use_f_res"])
        st.number_input(tr("lbl_mem_dia"), 3.0, 60.0, step=0.5,
                        key="p_mem_diameter_mm")
        st.number_input(tr("lbl_mem_thick"), 0.5, 100.0, step=0.5,
                        key="p_mem_thickness_um")
        st.number_input(tr("lbl_mem_tension"), 1.0, 5000.0, step=10.0,
                        key="p_mem_tension_npm")
        st.number_input(tr("lbl_mem_modes"), 1, 5, step=1,
                        key="p_mem_modes", help=tr("help_mem_modes"))
        st.checkbox(tr("lbl_modal_source"), key="p_modal_source",
                    help=tr("help_modal_source"))

    # ---------------- Backplate ----------------------------------------
    with st.expander(tr("exp_backplate"), expanded=True):
        st.number_input(tr("lbl_air_gap"), 5.0, 500.0, step=1.0,
                        key="p_air_gap_um")
        st.number_input(tr("lbl_bp_dia"), 2.0, 60.0, step=0.5,
                        key="p_bp_diameter_mm", help=tr("help_bp_dia"))
        st.number_input(tr("lbl_bp_thick"), 0.2, 20.0, step=0.1,
                        key="p_bp_thickness_mm")
        st.number_input(tr("lbl_bias"), 0.5, 400.0, step=1.0,
                        key="p_bias_v", help=tr("help_bias"))
        st.radio(tr("lbl_arch"), list(ARCH_LABELS),
                 format_func=_label_formatter(), key="p_architecture",
                 help=tr("help_arch"))
        st.number_input(tr("lbl_center_gap"), 0.0, 500.0, step=5.0,
                        key="p_center_gap_um",
                        disabled=st.session_state["p_architecture"]
                        != K67_LABEL,
                        help=tr("help_center_gap"))

        st.markdown(tr("hd_holes"))
        st.caption(tr("cap_holes"))
        _bp_d = st.session_state["p_bp_diameter_mm"]

        st.markdown(tr("md_through"), help=tr("help_through"))
        _ring_rows("th", _bp_d)
        st.number_input(tr("lbl_th_dia"), 0.05, 5.0, step=0.05,
                        key="p_d_through_mm", help=tr("help_hole_dia"))
        st.toggle(tr("lbl_stepped"), key="p_th_stepped",
                  help=tr("help_stepped"))

        st.markdown(tr("md_blind"), help=tr("help_blind"))
        _ring_rows("bh", _bp_d)
        st.number_input(tr("lbl_bh_dia"), 0.05, 5.0, step=0.05,
                        key="p_d_blind_mm", help=tr("help_hole_dia"))
        _bd_max = max(0.1, st.session_state["p_bp_thickness_mm"] - 0.1)
        st.session_state["p_blind_depth_mm"] = min(
            st.session_state["p_blind_depth_mm"], _bd_max)
        st.number_input(tr("lbl_bh_depth"), 0.05, _bd_max, step=0.05,
                        key="p_blind_depth_mm")

        st.markdown(tr("hd_clr"), help=tr("help_clr"))
        st.number_input(tr("lbl_clr_dia"), 0.0, 60.0, step=0.5,
                        key="p_clr_dia_mm", help=tr("help_clr_dia"))
        st.number_input(tr("lbl_clr_w"), 0.0, 10.0, step=0.1,
                        key="p_clr_width_mm", help=tr("help_clr_w"))
        st.number_input(tr("lbl_clr_d"), 0.0, 5.0, step=0.01,
                        format="%.3f", key="p_clr_depth_mm",
                        help=tr("help_clr_d"))

        # Durchgehender Randspalt (B&K) — nicht bei der K67-Bauform
        if st.session_state["p_architecture"] != K67_LABEL:
            st.number_input(tr("lbl_ring_vent"), 0.0, 500.0, step=5.0,
                            key="p_ring_vent_um",
                            help=tr("help_ring_vent"))
            st.number_input(tr("lbl_ring_vent_len"), 0.0, 20.0, step=0.1,
                            format="%.2f", key="p_ring_vent_len_mm",
                            help=tr("help_ring_vent_len"))

        # Klemmringe vor den Membranen — nur bei K67-Bauform relevant
        if st.session_state["p_architecture"] == K67_LABEL:
            st.markdown(tr("hd_clamp"), help=tr("help_clamp"))
            st.number_input(tr("lbl_clamp_t"), 0.0, 10.0,
                            step=0.5, key="p_clamp_ring_mm",
                            help=tr("help_clamp_t"))
            st.number_input(tr("lbl_clamp_w"), 0.0, 10.0, step=0.5,
                            key="p_clamp_width_mm",
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
        st.number_input(tr("lbl_spacer"), 0.0, 1000.0, step=5.0,
                        key="p_spacer_um", disabled=not _rear_on,
                        help=tr("help_spacer"))
        st.number_input(tr("lbl_rp_t"), 0.0, 20.0, step=0.1,
                        key="p_rearplate_mm", disabled=not _rear_on,
                        help=tr("help_rp_t"))
        _rp_on = _rear_on and st.session_state["p_rearplate_mm"] > 0.0
        st.number_input(tr("lbl_rp_n"), 0, 2000, step=1,
                        key="p_n_rearplate", disabled=not _rp_on,
                        help=tr("help_rp_n"))
        st.number_input(tr("lbl_rp_d"), 0.05, 5.0, step=0.05,
                        key="p_d_rearplate_mm", disabled=not _rp_on)

        st.markdown(tr("hd_delay"))
        st.number_input(tr("lbl_delay"), 0.0, 100.0, step=0.5,
                        key="p_delay_mm", disabled=not _rear_on,
                        help=tr("help_delay"))
        st.number_input(tr("lbl_cav_len"), 0.0, 100.0, step=0.5,
                        key="p_cavity_length_mm", disabled=not _rear_on)
        st.number_input(tr("lbl_cav_wall"), 0.0, 10.0, step=0.1,
                        key="p_cavity_wall_mm", disabled=not _rear_on)
        st.radio(tr("lbl_hole_pos"), list(POS_LABELS),
                 format_func=_label_formatter(),
                 key="p_hole_position", disabled=not _rear_on)
        st.number_input(tr("lbl_cav_n"), 0, 5000, step=1,
                        key="p_n_cavity", disabled=not _rear_on,
                        help=tr("help_cav_n"))
        st.number_input(tr("lbl_cav_d"), 0.0, 5.0, step=0.05,
                        key="p_d_cavity_mm", disabled=not _rear_on,
                        help=tr("help_cav_d"))
        _ax_max = st.session_state["p_cavity_length_mm"]
        st.session_state["p_cavity_axial_mm"] = min(
            st.session_state["p_cavity_axial_mm"], _ax_max)
        st.number_input(tr("lbl_cav_ax"), 0.0,
                        max(_ax_max, 0.5), step=0.5, key="p_cavity_axial_mm",
                        disabled=(not _rear_on)
                        or st.session_state["p_hole_position"] != "Umfang"
                        or _ax_max <= 0.0,
                        help=tr("help_cav_ax"))

    # ---------------- Akustisches Gewebe --------------------------------
    with st.expander(tr("exp_fabric"), expanded=True):
        st.number_input(tr("lbl_fab_front"), 0.0, 100000.0, step=5.0,
                        key="p_fabric_front_rayl")
        st.number_input(tr("lbl_fab_rear"), 0.0, 100000.0,
                        step=5.0, key="p_fabric_rear_rayl")
        st.radio(tr("lbl_fab_pos"), list(FAB_POS_LABELS),
                 format_func=_label_formatter(), key="p_fabric_rear_pos",
                 disabled=_is_k67, help=tr("help_fab_pos"))
        if _is_k67:
            st.caption(tr("cap_fab_pos_k67"))

    # ---------------- Gehäuse & Beugung ----------------------------------
    with st.expander(tr("exp_body"), expanded=False):
        st.toggle(tr("lbl_diffr"), key="p_diffraction_on",
                  help=tr("help_diffr"))
        _bd_min = max(st.session_state["p_mem_diameter_mm"],
                      st.session_state["p_bp_diameter_mm"])
        st.session_state["p_body_diameter_mm"] = max(
            st.session_state["p_body_diameter_mm"], _bd_min)
        st.number_input(tr("lbl_body_dia"), _bd_min, 100.0, step=0.5,
                        key="p_body_diameter_mm",
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
                st.number_input(tr("lbl_body_len"), 4.5, 200.0, step=0.5,
                                key="p_body_length_mm",
                                help=tr("help_body_len"))
            st.number_input(tr("lbl_bem_dia"), 0.0, 200.0, step=1.0,
                            key="p_bem_body_dia_mm",
                            help=tr("help_bem_dia"))
            if st.session_state["p_bem_body_dia_mm"] > 0.0:
                st.number_input(tr("lbl_bem_gap"), 3.0,
                                100.0, step=1.0, key="p_bem_body_gap_mm",
                                help=tr("help_bem_gap"))
                st.number_input(tr("lbl_bem_len"), 10.0, 300.0,
                                step=5.0, key="p_bem_body_len_mm",
                                help=tr("help_bem_len"))
            st.caption(tr("cap_bem") if _k67 else tr("cap_bem_front"))

    # ---------------- Spaltfilm-Modell -----------------------------------
    with st.expander(tr("exp_squeeze"), expanded=False):
        st.toggle(tr("lbl_2d"), key="p_squeeze_2d", help=tr("help_2d"))
        if st.session_state["p_squeeze_2d"]:
            st.caption(tr("cap_2d"))
        st.toggle(tr("lbl_3d"), key="p_squeeze_3d", help=tr("help_3d"))
        if st.session_state["p_squeeze_3d"]:
            st.toggle(tr("lbl_rot_auto"), key="p_half_rot_auto",
                      help=tr("help_rot_auto"))
            if not st.session_state["p_half_rot_auto"]:
                st.number_input(tr("lbl_rot_deg"), 0.0, 180.0, step=0.5,
                                key="p_half_rot_deg", help=tr("help_rot_deg"))
            st.caption(tr("cap_3d"))

    # ---------------- Simulation ---------------------------------------
    with st.expander(tr("exp_sim"), expanded=False):
        st.slider(tr("lbl_npts"), 100, 1500, step=50, key="p_n_points")
        st.toggle(tr("lbl_norm"), key="p_normalize_1khz")
        st.multiselect(tr("lbl_dirf"), DIRECTIVITY_OPTIONS,
                       key="p_dir_freqs", max_selections=10)

# ---------------------------------------------------------------------------
# Berechnung
# ---------------------------------------------------------------------------
params = _current_params()

# Platzhalter VOR dem Modellaufbau: schon der Konstruktor (Elektrostatik,
# bei 3D der Gitteraufbau) soll als Fortschritt sichtbar sein.
_prog_slot = st.empty()


def _show_progress(frac, label):
    _prog_slot.progress(frac, text=tr("prog_fmt", label=label,
                                      pct=100 * frac))


try:
    capsule = get_capsule(params, _show_progress)
except ValueError as exc:
    _prog_slot.empty()
    st.error(tr("err_params", exc=exc))
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
sens_1k, delay = _res["sens_1k"], _res["delay"]
noise = _res.get("noise")
summary_text = get_summary(_cache_key, capsule, _lang())

# ---------------------------------------------------------------------------
# Hauptbereich
# ---------------------------------------------------------------------------
st.title(tr("app_title"))
st.caption(tr("app_caption"))

# BEM mit offenem Rückeinlass: gerechnet, aber mit Modellgrenze — der
# Bohrungskranz sitzt als idealer Ring bei seiner Einbautiefe auf einer
# glatten Zylinderkontur (s. Gegenprobe 44).
if capsule.axial_body_model == "bem" and capsule.rear_open \
        and capsule.architecture != "dual_diaphragm":
    st.warning(tr("warn_bem_gradient", d=capsule.d_ext * 1e3,
                  wo=tr("warn_bem_end"
                        if capsule.cavity_hole_position == "end"
                        else "warn_bem_circ")))

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
    # gecachter Text: summary() enthält eine Netzwerkauswertung (bei 3D
    # eine volle LU-Lösung) und liefe sonst bei jedem Rerun mit
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
