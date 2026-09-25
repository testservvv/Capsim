"""Gemeinsame Grundlage der Gegenproben.

Stellt alle Namen des Moduls ``microphone_capsule`` bereit, auch die
privaten, die die Gegenproben prüfen (``from basis import *``). Dazu
kommen die Referenzkapseln und Messdaten, die mehrere Gegenproben teilen,
an EINER Stelle. Die Fixtures dazu stehen in ``conftest.py``.
"""
import numpy as np

import microphone_capsule as _mc

_MODUL = {k: v for k, v in vars(_mc).items() if not k.startswith("__")}
globals().update(_MODUL)
MicrophoneCapsule = _mc.MicrophoneCapsule


# ----------------------------------------------------------------------
# Referenzkapseln
# ----------------------------------------------------------------------
def demo_capsule():
    """1"-Großmembrankapsel, Nieren-artig, einzelne Backplate."""
    return MicrophoneCapsule(
        # Membran
        membrane_material="PET",          # Mylar-Folie
        membrane_resonance_hz=8000.0,     # [Hz]
        membrane_diameter=22e-3,          # [m]
        membrane_thickness=6e-6,          # [m]
        membrane_tension=400.0,           # [N/m]
        # Backplate
        air_gap=40e-6,                    # [m]
        backplate_diameter=20e-3,         # [m]
        backplate_thickness=3e-3,         # [m]
        bias_voltage=60.0,                # [V]
        architecture="single",
        n_through_holes=60, through_hole_diameter=1.0e-3,
        n_blind_holes=30, blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        # Rückseite
        delay_length=3e-3,                # Laufzeitglied [m]
        cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
        # Feiner Lochkranz am Umfang: viele Mikrolöcher wirken als
        # WIDERSTANDSDOMINIERTER Einlass (aperiodisch, Q < 1) — das
        # akustische Pendant zum Gewebe über Schlitzen realer Kapseln.
        # Wenige große Löcher wären massedominiert und bildeten mit dem
        # Hohlraum einen unbedämpften Helmholtz-Resonator, der die
        # Richtwirkung im Mittenband zerstört.
        cavity_hole_position="circumference",
        n_cavity_holes=200, cavity_hole_diameter=0.2e-3,
        cavity_hole_axial_position=6e-3,
        fabric_front_rayl=10.0, fabric_rear_rayl=25.0,
    )


# K67 (Doppelmembran): verifizierte Geometrie (120 Senkungen 1.3x3.7 mm,
# 60 durchgebohrt mit 0.6-mm-Kern, kein Gewebe, Klemmringe 2x2 mm vor
# beiden Membranen). Die interne Laufzeit fällt aus den physikalischen
# Parametern (Bohrungen, Spaltfilme, Spacer), die externe aus der
# axialen Körperbeugung (d_ext = axial + 2·Klemmringdicke) — beides ohne
# Fit-Koeffizient (Gegenproben 6, 17).
K67_KWARGS = dict(
    membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
    membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
    backplate_diameter=25e-3, backplate_thickness=4e-3,
    bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
    n_through_holes=60, through_hole_diameter=0.6e-3,
    n_blind_holes=120, blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
    through_holes_stepped=True,
    clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
    fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3,
    squeeze_model="2d",   # K67 braucht die radiale Druckauflösung
)


def k67_capsule(**extra):
    return MicrophoneCapsule(**{**K67_KWARGS, **extra})


def k67_null_angle(cap, f=1000.0):
    """Winkel des Pattern-Minimums in 0…180° (Gegenproben 6, 17)."""
    di = cap.directivity(frequencies_hz=(f,))
    ang = di["angles_deg"]
    lin = di["patterns"][f]["linear"]
    return ang[ang <= 180][int(np.argmin(lin[ang <= 180]))]


# Debenham/Robinson/Stebbings: einteilige durchbohrte Mittelelektrode,
# 1"-Membranen, Lochkreise der Konstruktionszeichnung (Gegenproben
# 15–18, 22).
DEB_KWARGS = dict(
    architecture="dual_diaphragm", membrane_resonance_hz=2100.0,
    membrane_diameter=25.4e-3, membrane_thickness=6e-6,
    membrane_tension=45.0, air_gap=38.1e-6,
    backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
    bias_voltage=50.0, center_gap=0.0,
    through_hole_diameter=0.71e-3,
    through_hole_rings=[(6, 21.84e-3), (3, 17.48e-3),
                        (3, 8.74e-3)],
    blind_hole_diameter=1.2e-3, blind_hole_depth=3.0e-3,
    blind_hole_rings=[(12, 21.84e-3), (12, 17.48e-3),
                      (12, 13.11e-3), (6, 8.74e-3), (4, 4.37e-3)],
    clamp_ring_thickness=2e-3, clamp_ring_width=3e-3,
    body_diameter=32e-3, squeeze_model="2d")


def debenham(**extra):
    return MicrophoneCapsule(**{**DEB_KWARGS, **extra})


# Clearance-Ring der Konstruktionszeichnung (Rand-Freistich) und der
# breite Mündungs-Freistich des 3D-Vergleichs (Gegenprobe 16)
DEB_CLEARANCE = dict(clearance_ring_diameter=22.63e-3,
                     clearance_ring_width=1.27e-3,
                     clearance_ring_depth=38e-6)
DEB_CLEARANCE_WIDE = dict(clearance_ring_diameter=13.0e-3,
                          clearance_ring_width=18.0e-3,
                          clearance_ring_depth=38e-6)


def hermetic_capsule():
    """K67 ohne Durchgangslöcher: Druckempfänger (Gegenproben 17, 18)."""
    return MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=0, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3)


# Prüffrequenzen der Lochkreis-Äquivalenzen (Gegenproben 9–11)
_fchk = np.array([100.0, 1000.0, 10000.0])


# ----------------------------------------------------------------------
# Messanker Grinnip 2006, Fig. 5/6/7 (Gegenproben 41, 43)
# ----------------------------------------------------------------------
# R. S. Grinnip III, "Advanced Simulation of a Condenser Microphone
# Capsule", J. Audio Eng. Soc. 54(3), 157–167 (2006), Fig. 5 (0°),
# Fig. 6 (90°), Fig. 7 (180°). Die Kurven sind aus dem PDF
# DIGITALISIERT, nicht geschätzt; die Kalibrierung prüft sich selbst:
#
#   * dB-Achse aus den waagerechten Gitterlinien +20/+10/0/−10/−20;
#     Ausgleichsgerade auf 1.1 px genau, die elf Achsenbeschriftungen
#     liegen darauf innerhalb von 0.27 dB.
#   * Frequenzachse aus den senkrechten Gitterlinien, die auf
#     n·10^k fallen müssen; Ausgleichsgerade auf 0.5 px (0.1 %).
#   * Nullprobe: der flache Ast 150…1200 Hz liegt danach auf 0.0 dB.
#
# Getrennt werden die Kurven über ihre Graustufe: VC (Grinnips
# gekoppelte FE/BE-Rechnung) ist dick und grau, EXP (die Messung) und
# LE (Ersatzschaltbild) sind dünn. VC und EXP decken sich bis 7 kHz;
# darüber nennt Grinnip selbst bis ~5 dB Abweichung. Auf Achse ist
# EXP nur zwischen 8 und 15 kHz von VC zu trennen und liegt dort
# 2…4 dB darunter; bei 180° ist EXP laut Grinnip oberhalb 10 kHz
# durch die Messvorrichtung verfälscht und wird nicht benutzt.
# Alle Werte sind dS gegen den flachen Ast, den das Diagramm auf 0 dB
# legt; das Modell wird dafür auf 100 Hz normiert.
f_grin = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                   7000.0, 8000.0, 9000.0, 10000.0, 12000.0, 14000.0,
                   16000.0, 18000.0])
vc_grin = {
    0.0: np.array([0.3, 1.8, 4.8, 7.3, 10.2, 12.2, 13.3, 12.9, 12.6,
                   13.1, 12.7, 11.6, 9.4, 5.2]),
    90.0: np.array([0.2, 0.4, 0.6, 1.5, 3.4, 5.0, 6.1, 4.2, 2.7, 1.2,
                    -2.4, -5.7, -11.1, -18.6]),
    180.0: np.array([0.3, 1.1, 2.9, 3.7, 6.2, 6.3, 6.5, 4.0, 1.9, 0.9,
                     -1.6, -4.2, -8.4, -14.3])}
# Messung: bei 90° über das ganze Band, auf Achse nur im Fenster, in
# dem EXP von VC zu trennen ist.
f_exp90 = np.array([1000.0, 3000.0, 5000.0, 7000.0, 8000.0, 9000.0,
                    10000.0, 12000.0, 14000.0, 16000.0])
exp90_grin = np.array([0.0, 0.0, 3.9, 5.7, 4.1, 2.4, 0.1, -5.2, -9.1,
                       -11.3])
f_exp0 = np.array([9000.0, 10000.0, 12000.0, 14000.0, 15000.0])
exp0_grin = np.array([10.5, 10.2, 8.8, 8.5, 7.5])


def _grin_rms(cap, ang, f_ref=None, a_ref=None):
    """Frequenzgang gegen die digitalisierte Kurve, LF-normiert."""
    f_ref = f_grin if f_ref is None else f_ref
    a_ref = vc_grin[ang] if a_ref is None else a_ref
    aa = 20 * np.log10(np.abs(cap.transfer_function(
        np.concatenate([[100.0], f_ref]), angle_deg=ang)))
    d = (aa[1:] - aa[0]) - a_ref
    return float(np.sqrt(np.mean(d ** 2))), d


__all__ = sorted(set(_MODUL) | {
    "MicrophoneCapsule", "demo_capsule", "K67_KWARGS", "k67_capsule",
    "k67_null_angle", "DEB_KWARGS", "debenham", "DEB_CLEARANCE",
    "DEB_CLEARANCE_WIDE", "hermetic_capsule", "_fchk", "f_grin",
    "vc_grin", "f_exp90", "exp90_grin", "f_exp0", "exp0_grin",
    "_grin_rms"})
