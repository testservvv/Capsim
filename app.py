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
      [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
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

DIRECTIVITY_OPTIONS = [50, 100, 125, 250, 500, 1000, 2000, 4000,
                       5000, 8000, 10000, 12500, 16000, 20000]

# (key, Widget-Art, Default) — Gruppen s. Sidebar-Aufbau weiter unten
DEFAULTS = {
    # Membran
    "material": "PET (Mylar)",
    "use_f_res": True,
    "f_res_hz": 8000.0,
    "mem_diameter_mm": 22.0,
    "mem_thickness_um": 6.0,
    "mem_tension_npm": 400.0,
    # Backplate
    "air_gap_um": 40.0,
    "bp_diameter_mm": 20.0,
    "bp_thickness_mm": 3.0,
    "bias_v": 60.0,
    "architecture": "Single Backplate",
    "center_gap_um": 50.0,
    "n_through": 60,
    "d_through_mm": 1.0,
    "th_pcd_mm": 0.0,
    "n_blind": 30,
    "d_blind_mm": 1.2,
    "blind_depth_mm": 1.5,
    "bh_pcd_mm": 0.0,
    # Rückseite / akustische Netzwerke
    "rear_enabled": True,
    "delay_mm": 3.0,
    "cavity_length_mm": 12.0,
    "cavity_wall_mm": 1.5,
    "hole_position": "Umfang",
    "n_cavity": 200,
    "d_cavity_mm": 0.2,
    "cavity_axial_mm": 6.0,
    "fabric_front_rayl": 10.0,
    "fabric_rear_rayl": 25.0,
    # Gehäuse & Beugung
    "diffraction_on": True,
    "body_diameter_mm": 26.4,
    # Spaltfilm-Modell
    "squeeze_2d": False,
    # Simulation
    "n_points": 400,
    "normalize_1khz": False,
    "dir_freqs": [100, 1000, 5000, 10000],
}

_FLOAT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, float)}
_INT_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, int)
             and not isinstance(v, bool)}
_BOOL_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, bool)}


def _init_state():
    for key, val in DEFAULTS.items():
        st.session_state.setdefault("p_" + key, val)


def _current_params():
    return {k: st.session_state["p_" + k] for k in DEFAULTS}


def _coerce(key, val):
    """Robuste Typkonvertierung beim Projekt-Laden."""
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
        loaded = 0
        for key, val in data.get("params", {}).items():
            if key in DEFAULTS:
                st.session_state["p_" + key] = _coerce(key, val)
                loaded += 1
        st.session_state["_load_msg"] = (
            "success", f"Projekt geladen — {loaded} Parameter übernommen.")
    except Exception as exc:  # defekte Datei darf die App nicht stoppen
        st.session_state["_load_msg"] = (
            "error", f"Projekt konnte nicht geladen werden: {exc}")


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
        n_through_holes=p["n_through"],
        through_hole_diameter=p["d_through_mm"] * 1e-3,
        through_hole_pcd=(p["th_pcd_mm"] * 1e-3 if p["th_pcd_mm"] > 0 else None),
        n_blind_holes=p["n_blind"],
        blind_hole_diameter=p["d_blind_mm"] * 1e-3,
        blind_hole_depth=p["blind_depth_mm"] * 1e-3,
        blind_hole_pcd=(p["bh_pcd_mm"] * 1e-3 if p["bh_pcd_mm"] > 0 else None),
        # Rückseite deaktiviert -> keine rückwärtige Baugruppe: die
        # Durchgangslöcher der Backplate münden (durch das rückwärtige
        # Gewebe) direkt ins Schallfeld. Hermetisch dicht ist die Kapsel
        # nur bei 0 Durchgangslöchern — das entscheidet das Physikmodell.
        rear_network_enabled=p["rear_enabled"],
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
        squeeze_model=("2d" if p["squeeze_2d"] else "1d"),
    )


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


def _freq_label(f):
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


def polar_figure(di):
    angles = di["angles_deg"]
    fig = go.Figure()
    r_max = 0.0
    for i, (f, pat) in enumerate(sorted(di["patterns"].items())):
        r_max = max(r_max, float(np.max(pat["db"])))
        fig.add_trace(go.Scatterpolar(
            theta=angles, r=pat["db"], mode="lines",
            name=_freq_label(f),
            line=dict(color=SERIES[i % len(SERIES)], width=2),
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
                "version": 1,
                "saved": _dt.date.today().isoformat(),
                "params": _current_params(),
            },
            indent=2, ensure_ascii=False,
        )
        st.download_button(
            "💾 Projekt speichern (.json)", data=project_json,
            file_name="capsim_projekt.json", mime="application/json",
            use_container_width=True,
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
        st.number_input("Durchgangslöcher — Anzahl", 0, 2000, step=1,
                        key="p_n_through",
                        help="Die Durchgangslöcher sind der einzige Weg "
                             "durch die Backplate. 0 = Backplate "
                             "geschlossen → Kapsel hermetisch dicht "
                             "(Druckempfänger), unabhängig von der "
                             "Rückseite. Bei Dual-Architektur ist "
                             "mindestens 1 Loch nötig.")
        st.number_input("Durchgangslöcher — Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_through_mm")
        _bp_d = st.session_state["p_bp_diameter_mm"]
        st.number_input("Durchgangslöcher — Lochkreis Ø [mm]", 0.0, _bp_d,
                        step=0.5, key="p_th_pcd_mm",
                        help="Mittlerer Sitzradius der Durchgangslöcher "
                             "(nur im 2D-Modell wirksam). 0 = gleichmäßig "
                             "verteilt. Der radiale Versatz zu den "
                             "Blindlöchern bildet die Laufzeitstrecke der "
                             "Niere ab.")
        st.number_input("Blindlöcher — Anzahl", 0, 2000, step=1,
                        key="p_n_blind")
        st.number_input("Blindlöcher — Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_blind_mm")
        _bd_max = max(0.1, st.session_state["p_bp_thickness_mm"] - 0.1)
        st.session_state["p_blind_depth_mm"] = min(
            st.session_state["p_blind_depth_mm"], _bd_max)
        st.number_input("Blindlöcher — Tiefe [mm]", 0.05, _bd_max, step=0.05,
                        key="p_blind_depth_mm")
        st.number_input("Blindlöcher — Lochkreis Ø [mm]", 0.0, _bp_d,
                        step=0.5, key="p_bh_pcd_mm",
                        help="Mittlerer Sitzradius der Blindlöcher (nur im "
                             "2D-Modell wirksam). 0 = gleichmäßig verteilt.")

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
        st.number_input("Laufzeitglied — Länge [mm]", 0.0, 100.0, step=0.5,
                        key="p_delay_mm", disabled=not _rear_on,
                        help="Akustische Leitung hinter der Membran; "
                             "Laufzeit τ = L/c.")
        st.number_input("Hohlraum — Länge [mm]", 1.0, 100.0, step=0.5,
                        key="p_cavity_length_mm", disabled=not _rear_on)
        st.number_input("Hohlraum — Wandstärke [mm]", 0.2, 10.0, step=0.1,
                        key="p_cavity_wall_mm", disabled=not _rear_on)
        st.radio("Hohlraumlöcher — Position", list(POS_LABELS),
                 key="p_hole_position", disabled=not _rear_on)
        st.number_input("Hohlraumlöcher — Anzahl", 0, 5000, step=1,
                        key="p_n_cavity", disabled=not _rear_on,
                        help="0 = Rückseite geschlossen → Druckempfänger "
                             "(Kugelcharakteristik).")
        st.number_input("Hohlraumlöcher — Ø [mm]", 0.05, 5.0, step=0.05,
                        key="p_d_cavity_mm", disabled=not _rear_on)
        _ax_max = st.session_state["p_cavity_length_mm"]
        st.session_state["p_cavity_axial_mm"] = min(
            st.session_state["p_cavity_axial_mm"], _ax_max)
        st.number_input("Hohlraumlöcher — axiale Position [mm]", 0.1, _ax_max,
                        step=0.5, key="p_cavity_axial_mm",
                        disabled=(not _rear_on)
                        or st.session_state["p_hole_position"] != "Umfang",
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
            st.caption("Die Lochkreis-Durchmesser (PCD) im Abschnitt "
                       "Backplate steuern jetzt die radiale Lochverteilung.")

    # ---------------- Simulation ---------------------------------------
    with st.expander("Simulation", expanded=False):
        st.slider("Frequenzpunkte", 100, 1500, step=50, key="p_n_points")
        st.toggle("Amplitude auf 1 kHz normieren", key="p_normalize_1khz")
        st.multiselect("Richtdiagramm-Frequenzen [Hz]", DIRECTIVITY_OPTIONS,
                       key="p_dir_freqs", max_selections=6)

# ---------------------------------------------------------------------------
# Berechnung
# ---------------------------------------------------------------------------
params = _current_params()
try:
    capsule = build_capsule(params)
except ValueError as exc:
    st.error(f"⚠️ Ungültige Parameterkombination: {exc}")
    st.stop()

fr = capsule.frequency_response(10.0, 25000.0, n_points=params["n_points"])
dir_freqs = sorted(params["dir_freqs"]) or [1000]
di = capsule.directivity(frequencies_hz=[float(f) for f in dir_freqs])

# ---------------------------------------------------------------------------
# Hauptbereich
# ---------------------------------------------------------------------------
st.title("Kondensatormikrofonkapsel — Simulation")
st.caption("Elektroakustisches Ersatzschaltbild (Lumped-Element-Modell) · "
           "10 Hz – 25 kHz")

sens_1k = abs(capsule.transfer_function(1000.0)[0])
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

col_bode, col_polar = st.columns([11, 9], gap="medium")
with col_bode:
    st.plotly_chart(bode_figure(fr, params["normalize_1khz"]),
                    use_container_width=True,
                    config={"displayModeBar": False})
with col_polar:
    st.plotly_chart(polar_figure(di), use_container_width=True,
                    config={"displayModeBar": False})

with st.expander("Abgeleitete Modellparameter (Diagnose)"):
    st.code(capsule.summary(), language=None)

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
})
df_di = pd.DataFrame({"winkel_deg": di["angles_deg"]})
for f, pat in sorted(di["patterns"].items()):
    tag = f"{f:.0f}hz"
    df_di[f"pegel_db_{tag}"] = pat["db"]
    df_di[f"linear_{tag}"] = pat["linear"]

tab_fr, tab_di = st.tabs(["Frequenzgang", "Richtdiagramm"])
with tab_fr:
    st.dataframe(df_fr, height=240, use_container_width=True, hide_index=True)
with tab_di:
    st.dataframe(df_di, height=240, use_container_width=True, hide_index=True)

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
                     use_container_width=True)
exp2.download_button("⬇️ Richtdiagramm als CSV", _to_csv(df_di),
                     file_name="capsim_richtdiagramm.csv", mime="text/csv",
                     use_container_width=True)

st.caption("Capsim · Physikmodell: ABCD-Kettenmatrizen, Zwikker–Kosten-"
           "Lochimpedanzen, Škvor-Squeeze-Film, elektrostatische Wandlung "
           "mit Feder-Erweichung. Empfindlichkeit = Leerlaufspannung ohne "
           "Streukapazität/Verstärkerlast.")
