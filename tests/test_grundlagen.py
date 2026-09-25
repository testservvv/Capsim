"""Gegenproben: Netzwerk und Grundfunktionen."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


def test_gp00_kennwerte_frequenzgang_richtdiagramm(capsule):
    """Demo: Kennwerte, Frequenzgang, Richtdiagramm."""
    print(capsule.summary())
    print()

    # ------------------------- Frequenzgang --------------------------------
    fr = capsule.frequency_response(10.0, 25000.0, n_points=400)
    f = fr["frequency_hz"]
    assert f.shape == (400,)
    assert fr["amplitude_db"].shape == f.shape
    assert fr["phase_deg"].shape == f.shape
    assert np.all(np.isfinite(fr["amplitude_db"]))
    assert np.all(np.isfinite(fr["phase_deg"]))

    print("Frequenzgang (Auszug, 0° Einfall):")
    print(f"{'f [Hz]':>10} {'Ampl. [dB re 1V/Pa]':>22} {'norm. [dB]':>12} "
          f"{'Phase [°]':>11}")
    for f_show in (20, 100, 1000, 5000, 10000, 20000):
        i = int(np.argmin(np.abs(f - f_show)))
        print(f"{f[i]:10.0f} {fr['amplitude_db'][i]:22.2f} "
              f"{fr['amplitude_db_norm'][i]:12.2f} {fr['phase_deg'][i]:11.1f}")
    print()

    # ------------------------- Richtdiagramm -------------------------------
    di = capsule.directivity(frequencies_hz=(100.0, 1000.0, 5000.0, 10000.0),
                             n_angles=361)
    angles = di["angles_deg"]
    assert angles.shape == (361,)
    print("Richtdiagramm |e(θ)|/|e(0°)| (linear):")
    print(f"{'f [Hz]':>10} {'0°':>7} {'90°':>7} {'135°':>7} {'180°':>7}")
    for f_pat, pat in di["patterns"].items():
        lin = pat["linear"]
        assert lin.shape == angles.shape
        assert np.all(np.isfinite(lin))
        print(f"{f_pat:10.0f} {lin[0]:7.3f} {lin[90]:7.3f} "
              f"{lin[135]:7.3f} {lin[180]:7.3f}")
    print()


def test_gp01_geschlossene_ruckseite_kugel():
    """Gegenprobe 1: geschlossene Rückseite -> Kugel."""
    # (ohne Beugung: exakte Kugel als Netzwerk-Konsistenzprüfung)
    omni = MicrophoneCapsule(n_cavity_holes=0, include_diffraction=False)
    di_o = omni.directivity(frequencies_hz=(1000.0,))
    lin_o = di_o["patterns"][1000.0]["linear"]
    assert np.allclose(lin_o, 1.0, atol=1e-9), "Druckempfänger muss Kugel sein"
    print("Gegenprobe geschlossene Rückseite (ohne Beugung): Kugel  OK")


def test_gp02_dual_backplate_gegentakt(capsule):
    """Gegenprobe 2: Dual-Backplate (Gegentakt)."""
    # Im steifigkeitskontrollierten Bereich (deutlich unterhalb der
    # Resonanz) muss der Gegentakt der beiden Backplates ~+6 dB liefern.
    # Dazu kommt hier ~+2 dB, weil die vordere Backplate den vorderen
    # Einlass nach vorn verschiebt (größeres d_ext -> mehr Gradienten-
    # antrieb). Nahe der Resonanz wird ein Teil des Gewinns durch die
    # zusätzliche Squeeze-Film-Dämpfung des VORDEREN Spalts aufgezehrt.
    dual = MicrophoneCapsule(architecture="dual")
    fr_d = dual.frequency_response(n_points=100)
    assert np.all(np.isfinite(fr_d["amplitude_db"]))
    e_s = 20 * np.log10(abs(capsule.transfer_function(100.0)[0]))
    e_d = 20 * np.log10(abs(dual.transfer_function(100.0)[0]))
    assert 4.0 < e_d - e_s < 9.5, "Gegentakt-Gewinn außerhalb Erwartung"
    print(f"Dual-Backplate @100 Hz: {e_d:.2f} dB re 1 V/Pa "
          f"(single: {e_s:.2f} dB) — Gegentakt-Gewinn "
          f"{e_d - e_s:+.2f} dB  OK")


def test_gp03_durchgangslocher_ruckseiten_baugruppe():
    """Gegenprobe 3: Durchgangslöcher & Rückseiten-Baugruppe."""
    # a) Ohne rückwärtige Baugruppe münden die Durchgangslöcher direkt ins
    #    Schallfeld -> Gradientenempfänger mit kurzer Wegdifferenz.
    vented = MicrophoneCapsule(rear_network_enabled=False)
    di_v = vented.directivity(frequencies_hz=(500.0,))
    lin_v = di_v["patterns"][500.0]["linear"]
    assert lin_v[90] < 0.9 or lin_v[180] < 0.9, \
        "direkt belüftete Backplate muss Richtwirkung zeigen"
    # b) Ohne Durchgangslöcher ist die Kapsel hermetisch dicht — auch mit
    #    montierter Rückseite (die dann akustisch unerreichbar ist).
    sealed = MicrophoneCapsule(n_through_holes=0, include_diffraction=False)
    di_s = sealed.directivity(frequencies_hz=(1000.0,))
    assert np.allclose(di_s["patterns"][1000.0]["linear"], 1.0, atol=1e-9), \
        "geschlossene Backplate muss Kugel sein"
    fr_s = sealed.frequency_response(n_points=50)
    assert np.all(np.isfinite(fr_s["amplitude_db"]))
    print(f"Belüftete Backplate (ohne Baugruppe): 90°={lin_v[90]:.2f}, "
          f"180°={lin_v[180]:.2f} — Richtwirkung  OK")
    print("Geschlossene Backplate (0 Durchgangslöcher): Kugel, "
          "hermetisch dicht  OK")


def test_gp12_vollstandige_verlustmechanismen(capsule):
    """Gegenprobe 12: vollständige Verlustmechanismen."""
    # a) Viskose Mündung (Sampson/Weissberg): der DC-Grenzfall zweier
    #    Mündungen muss exakt 3·mu/r³ betragen.
    om_lo = np.array([2.0 * np.pi * 2.0])
    r12 = 0.4e-3
    Z_no = MicrophoneCapsule._hole_impedance(om_lo, r12, 1e-3, 1)
    Z_ve = MicrophoneCapsule._hole_impedance(om_lo, r12, 1e-3, 1,
                                             visc_ends=2)
    R_samp = 3.0 * MU_AIR / r12**3
    assert abs((Z_ve - Z_no).real[0] - R_samp) / R_samp < 0.02, \
        "viskose Mündung muss im DC-Grenzfall Sampson (3µ/r³) treffen"
    if _HAS_SCIPY:
        # b) Geschlossener thermoviskoser Stub gegen fein diskretisierte
        #    Leiter (40 Segmente, gleiche Beläge): coth-Lösung korrekt.
        om12 = 2.0 * np.pi * np.array([100.0, 1000.0, 10000.0])
        r_s, L_s = 0.65e-3, 3.7e-3
        Z_stub = capsule._closed_hole_stub(om12, r_s, L_s)
        g_s, Zc_s = MicrophoneCapsule._narrow_duct_propagation(om12, r_s)
        Zp, Yp = g_s * Zc_s, g_s / Zc_s
        n_seg = 40
        dl = L_s / n_seg
        Z_lad = Zp * dl + 1.0 / (Yp * dl)     # letztes Segment (Ende zu)
        for _ in range(n_seg - 1):
            Z_lad = Zp * dl + 1.0 / (Yp * dl + 1.0 / Z_lad)
        assert np.max(np.abs(Z_stub / Z_lad - 1.0)) < 0.02, \
            "Stub-coth muss der diskretisierten Leiter entsprechen"
        # c) LF-Nachgiebigkeit des Stubs ISOTHERM (V/P_atm), nicht
        #    adiabatisch — das alte Lumped-Modell lag hier um gamma daneben.
        om3 = np.array([2.0 * np.pi * 3.0])
        V_s = np.pi * r_s**2 * L_s
        C_lf = -1.0 / (om3[0] * capsule._closed_hole_stub(om3, r_s,
                                                          L_s).imag[0])
        assert 0.9 < C_lf / (V_s / P_ATM) < 1.1, \
            "Stub-Nachgiebigkeit muss bei tiefen Frequenzen isotherm sein"
    # d) Filmkorrektur: Φ(ω→0) = 1; bei 25 kHz/60 µm dominiert die
    #    laterale Trägheit (Faktor ~4, Phase > 60°).
    phi_lo = MicrophoneCapsule._film_R_dynamic(np.array([2.0 * np.pi]),
                                               60e-6)[0]
    assert abs(phi_lo - 1.0) < 1e-3
    phi_hi = MicrophoneCapsule._film_R_dynamic(
        np.array([2.0 * np.pi * 25000.0]), 60e-6)[0]
    assert 3.0 < abs(phi_hi) < 5.0 and np.degrees(np.angle(phi_hi)) > 60.0
    print(f"Verlustmechanismen: Sampson-Mündung = 3µ/r³ (DC), Stub ≡ "
          f"diskrete Leiter (<2 %), LF-Nachgiebigkeit isotherm, "
          f"Filmkorrektur Φ(0)=1 / |Φ(25 kHz)|={abs(phi_hi):.1f}  OK")


def test_gp13_ruckwarts_durchlauf_port_tausch():
    """Gegenprobe 13: Rückwärts-Durchlauf (Port-Tausch)."""
    # a) Algebra: der Port-Tausch eines Zweitor-Produkts muss der
    #    umgekehrten Elementreihenfolge entsprechen (so macht es der
    #    1D-Pfad) — die Matrix-Inverse täte das NICHT (negative Elemente).
    om13 = np.array([2.0 * np.pi * 700.0])
    E1 = MicrophoneCapsule._abcd_series(2.5e6 + 1j * 4e5, om13)
    E2 = MicrophoneCapsule._abcd_shunt(1j * 3e-10, om13)
    E3 = MicrophoneCapsule._abcd_series(8.0e5, om13)
    chain = reduce(MicrophoneCapsule._mmul, [E1, E2, E3])
    chain_rev = reduce(MicrophoneCapsule._mmul, [E3, E2, E1])
    assert np.allclose(
        np.array(MicrophoneCapsule._abcd_reverse(chain), dtype=complex),
        np.array(chain_rev, dtype=complex)), \
        "Port-Tausch muss der umgekehrten Elementreihenfolge entsprechen"
    # b) Physik: mit korrektem Rückwärts-Durchlauf entsteht die interne
    #    Phasenschieber-Laufzeit aus den akustischen Parametern — die
    #    K67-Niere (Null bei 180°) darf deshalb NICHT mehr an der
    #    Membranresonanz hängen (vorher kollabierte sie: die Laufzeit kam
    #    aus der f_res-abhängigen statischen Spaltasymmetrie).
    def _k67_null(fres):
        c = MicrophoneCapsule(
            membrane_resonance_hz=fres, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, body_diameter=34e-3, squeeze_model="2d")
        di = c.directivity(frequencies_hz=(1000.0,))
        pat = di["patterns"][1000.0]
        na = di["angles_deg"][:181][int(np.argmin(pat["linear"][:181]))]
        return na, pat["db"][180]
    na_lo, p_lo = _k67_null(1150.0)
    na_hi, p_hi = _k67_null(4000.0)
    assert na_lo > 150.0 and na_hi > 150.0, \
        f"Nullwinkel muss f_res-robust nahe 180° liegen ({na_lo:.0f}°/{na_hi:.0f}°)"
    assert p_lo < -15.0 and p_hi < -15.0 and abs(p_lo - p_hi) < 12.0, \
        f"Nierentiefe muss f_res-robust sein ({p_lo:.1f} / {p_hi:.1f} dB)"
    print(f"Rückwärts-Durchlauf: Port-Tausch ≡ umgekehrte Elementreihen"
          f"folge; K67-Niere f_res-robust (180° @1 kHz: {p_lo:.1f} dB "
          f"@1150 Hz / {p_hi:.1f} dB @4000 Hz, Null bei "
          f"{na_lo:.0f}°/{na_hi:.0f}°)  OK")


def test_gp14_ubrige_architekturen_unter_dem_port():
    """Gegenprobe 14: übrige Architekturen unter dem Port-Tausch."""
    # Der zweite Nutzer des Rückwärts-Durchlaufs ist die VORDERE Backplate
    # der Dual-Architektur (2D). Das normierte Richtdiagramm darf davon
    # nicht abhängen (Serienelemente des Frontzweigs kürzen sich) — 1D-
    # und 2D-Pfad müssen übereinstimmen; der Frequenzgang bleibt endlich
    # und Gewebe wirkt als passive Seriendämpfung. K103-Rückplatte dicht
    # -> exakte Kugel auch im 2D-Modell (Vorwärtspfade unberührt).
    if _HAS_SCIPY:
        du1 = MicrophoneCapsule(architecture="dual", squeeze_model="1d")
        du2 = MicrophoneCapsule(architecture="dual", squeeze_model="2d")
        pd1 = du1.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"]
        pd2 = du2.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"]
        assert np.max(np.abs(pd1 - pd2)) < 1.0, \
            "Dual-Backplate: 1D- und 2D-Richtdiagramm müssen übereinstimmen"
        fr_du = du2.frequency_response(n_points=80)
        assert np.all(np.isfinite(fr_du["amplitude_db"]))
        s_du = abs(du2.transfer_function(1000.0)[0])
        s_fab = abs(MicrophoneCapsule(
            architecture="dual", squeeze_model="2d", fabric_front_rayl=150.0,
            fabric_rear_rayl=150.0).transfer_function(1000.0)[0])
        assert 0.5 < s_fab / s_du < 0.9999, \
            "Gewebe muss passiv dämpfen (Empfindlichkeit leicht senken)"
        k103_dicht = MicrophoneCapsule(
            squeeze_model="2d", rear_spacer_height=100e-6,
            rear_plate_thickness=2e-3, n_rear_plate_holes=0,
            include_diffraction=False)
        lin_k1 = k103_dicht.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["linear"]
        assert np.max(np.abs(lin_k1 - 1.0)) < 1e-3, \
            "K103 mit dichter Rückplatte muss auch in 2D exakte Kugel sein"
        print(f"Architektur-Konsistenz: Dual-Backplate 1D≡2D (max Abw. "
              f"{np.max(np.abs(pd1 - pd2)):.2f} dB), Gewebe dämpft passiv "
              f"({s_fab / s_du:.3f}×), K103 dicht = Kugel (2D)  OK")


def test_gp18_angle_responses_helmholtz_frequenz(deb1, deb3, deb_kwargs, hermetic, k67):
    """Gegenprobe 18: angle_responses / Helmholtz-Frequenz."""
    # a) Superposition: H(θ) aus EINEM Durchlauf == transfer_function(θ)
    #    für 0/90/180° (1D/2D-Netzwerk UND 3D-Feldlöser).
    # b) f_H (90°-Kreuzung von arg D_r) liegt für die bekannten Kapseln im
    #    erwarteten Bereich; eine dickere Platte (längere enge Bohrungen
    #    -> mehr Trägheit) muss die Resonanz absenken.
    # c) Geschlossene Rückseite: D_r und f_H -> None, H bleibt berechenbar.
    f_grid = np.logspace(np.log10(50.0), np.log10(20000.0), 120)
    ar_k = k67.angle_responses(f_grid)
    for angd in (0.0, 90.0, 180.0):
        ref = k67.transfer_function(f_grid[::20], angle_deg=angd)
        assert np.allclose(ar_k["H"][angd][::20], ref, rtol=1e-9), \
            f"angle_responses muss transfer_function reproduzieren ({angd}°)"
    fH_k67 = MicrophoneCapsule.helmholtz_resonance_hz(f_grid, ar_k["D_r"])
    assert fH_k67 is not None and 2800.0 < fH_k67 < 3800.0, \
        f"K67: interne Resonanz ~3.3 kHz erwartet ({fH_k67})"
    if _HAS_SCIPY:
        ar_d1 = deb1.angle_responses(f_grid)
        fH_deb = MicrophoneCapsule.helmholtz_resonance_hz(f_grid,
                                                          ar_d1["D_r"])
        assert fH_deb is not None and 2000.0 < fH_deb < 2800.0, \
            f"Debenham: interne Resonanz ~2.4 kHz erwartet ({fH_deb})"
        deb_dick = MicrophoneCapsule(clearance_ring_diameter=22.63e-3,
                                     clearance_ring_width=1.27e-3,
                                     clearance_ring_depth=38e-6,
                                     **{**deb_kwargs,
                                        "backplate_thickness": 6.0e-3})
        fH_dick = MicrophoneCapsule.helmholtz_resonance_hz(
            f_grid, deb_dick.angle_responses(f_grid)["D_r"])
        assert fH_dick is not None and fH_dick < fH_deb - 300.0, \
            (f"6-mm-Platte: längere Bohrungen müssen die Resonanz senken "
             f"({fH_dick:.0f} vs. {fH_deb:.0f} Hz)")
        f1 = np.array([1000.0])
        ar3 = deb3.angle_responses(f1)
        ref3 = deb3.transfer_function(f1, angle_deg=90.0)
        assert np.allclose(ar3["H"][90.0], ref3, rtol=1e-9), \
            "angle_responses (3D) muss transfer_function reproduzieren"
        assert ar3["D_r"] is not None
    ar_h = hermetic.angle_responses(np.array([100.0, 1000.0]))
    assert ar_h["D_r"] is None
    assert MicrophoneCapsule.helmholtz_resonance_hz(
        np.array([100.0, 1000.0]), ar_h["D_r"]) is None
    assert all(np.all(np.isfinite(np.abs(v))) for v in ar_h["H"].values())
    print("angle_responses: Superposition == transfer_function (0/90/180°, "
          f"auch 3D); f_H: K67 {fH_k67:.0f} Hz"
          + (f", Debenham {fH_deb:.0f} Hz -> 6-mm-Platte {fH_dick:.0f} Hz"
             if _HAS_SCIPY else "")
          + "; geschlossene Rückseite -> None  OK")
