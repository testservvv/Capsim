"""Gegenproben: Beugung, Strahlung und BEM."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


def test_gp04_druckstau_beugung_am_kapselkorper():
    """Gegenprobe 4: Druckstau/Beugung am Kapselkörper."""
    # Ein Druckempfänger MIT Beugung muss bei tiefen Frequenzen praktisch
    # kugelförmig sein, zu hohen Frequenzen hin aber zunehmend richten;
    # frontal steigt die Freifeldempfindlichkeit durch den Druckstau.
    omni_d = MicrophoneCapsule(n_through_holes=0)   # Beugung standardmäßig an
    di_hf = omni_d.directivity(frequencies_hz=(100.0, 4000.0, 16000.0))
    p100 = di_hf["patterns"][100.0]["linear"]
    p4k = di_hf["patterns"][4000.0]["linear"]
    p16k = di_hf["patterns"][16000.0]["linear"]
    assert abs(p100[135] - 1.0) < 0.03, "100 Hz muss nahezu Kugel sein"
    assert p16k[135] < p4k[135] < 1.0, \
        "Richtwirkung muss mit der Frequenz zunehmen"
    e_lo = abs(omni_d.transfer_function(200.0)[0])
    e_hi = abs(omni_d.transfer_function(16000.0, angle_deg=0.0)[0])
    e_hi_nod = abs(MicrophoneCapsule(
        n_through_holes=0, include_diffraction=False
    ).transfer_function(16000.0)[0])
    boost_db = 20 * np.log10(e_hi / e_hi_nod)
    assert 1.0 < boost_db < 7.0, "Druckstau frontal: erwarte ~+2..6 dB"
    print(f"Druckstau/Beugung: Kugel @100 Hz, 135°-Pegel "
          f"{p4k[135]:.2f} (4 kHz) -> {p16k[135]:.2f} (16 kHz), "
          f"frontaler Druckstau @16 kHz: +{boost_db:.1f} dB  OK")


def test_gp20_spharoid_korpermodell_axialer_transfer(k67):
    """Gegenprobe 20: Sphäroid-Körpermodell (axialer Transfer)."""
    # Eigene oblate Spezialfunktionen (scipy obl_rad2 ist für ξ0 < 1
    # unbrauchbar). Verifikation:
    # a) Kugel-Grenzfall b -> a reproduziert die Morse-Reihe (der Fehler
    #    skaliert linear mit 1 - b/a: reine Geometrie).
    # b) Dünne-Scheiben-Grenzfall: effektive LF-Distanz am Pol -> 4a/π
    #    (klassisches Scheibenresultat).
    # c) Wronski-Selbstprüfung W{R1,R3} = i/(c(ξ²+1)) über das Band.
    # d) BEFUND als Anker: die FREI STEHENDE K67-Scheibe hätte
    #    d_eff ~ 0.94·2·R_body >> intern (~17 mm) -> Superniere weit vor
    #    180°. Die reale (montierte) Kapsel folgt der d_ext-Kugel —
    #    deshalb bleibt 'sphere' Standard; 'spheroid' ist die
    #    dokumentierte Referenz der freien Scheibe.
    if _HAS_SCIPY:
        th20 = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0]))
        R20 = 9e-3
        worst = 0.0
        for f20 in (100.0, 1000.0, 5000.0, 15000.0):
            om20 = np.array([2.0 * np.pi * f20])
            d_save = k67.d_ext
            k67.d_ext = 2.0 * R20
            G_ref = k67._axial_body_transfer(om20, th20)[0]
            k67.d_ext = d_save
            G_sph = k67._spheroid_pole_transfer(om20, th20, R20,
                                                0.9995 * R20)[0]
            worst = max(worst, float(np.max(np.abs(G_sph - G_ref)
                                            / np.abs(G_ref))))
        assert worst < 3e-3, \
            f"Sphäroid muss im Kugel-Grenzfall die Morse-Reihe treffen " \
            f"({worst:.2e})"
        om50 = np.array([2.0 * np.pi * 50.0])
        G_disc = k67._spheroid_pole_transfer(om50, np.array([np.pi]),
                                             17e-3, 0.02 * 17e-3)[0, 0]
        d_eff_disc = float(np.angle(G_disc)) / (om50[0] / C_AIR)
        assert abs(d_eff_disc / 17e-3 - 4.0 / np.pi) < 0.04, \
            f"Scheiben-Grenzfall 4a/π verfehlt ({d_eff_disc/17e-3:.3f})"
        assert k67._spheroid_wronski_max < 1e-4, \
            "Wronski-Selbstprüfung des Sphäroids"
        G_k67 = k67._axial_spheroid_transfer(om50, np.array([np.pi]))[0, 0]
        d_eff_k67 = float(np.angle(G_k67)) / (om50[0] / C_AIR)
        assert 0.85 < d_eff_k67 / (2.0 * k67.R_body) < 1.0, \
            f"freie K67-Scheibe: d_eff ~ 0.94·2R erwartet ({d_eff_k67})"
        k67_sph = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="2d", axial_body_model="spheroid")
        di_sph = k67_sph.directivity(frequencies_hz=(1000.0,))
        lin_s = di_sph["patterns"][1000.0]["linear"]
        a_s = di_sph["angles_deg"]
        na_s = a_s[a_s <= 180][int(np.argmin(lin_s[a_s <= 180]))]
        assert na_s < 150.0, \
            (f"freie Scheibe: Minimum deutlich vor 180° erwartet "
             f"({na_s:.0f}°) — Beleg, warum 'sphere' Standard bleibt")
        try:
            MicrophoneCapsule(architecture="single",
                              axial_body_model="spheroid",
                              membrane_resonance_hz=8000.0)
            raise AssertionError("spheroid ohne dual_diaphragm muss scheitern")
        except ValueError:
            pass
        print(f"Sphäroid: Kugel-Grenzfall {worst:.1e}; Scheibe "
              f"d_eff/a = {d_eff_disc/17e-3:.3f} (4/π = {4/np.pi:.3f}); "
              f"Wronski {k67._spheroid_wronski_max:.1e}; freie K67-Scheibe "
              f"d_eff = {d_eff_k67*1e3:.1f} mm -> Minimum {na_s:.0f}° "
              "(montiert: Kugel bleibt Standard)  OK")


@pytest.mark.bem
def test_gp21_axisymmetrisches_bem_kopf_korper(k67, k67_bem):
    """Gegenprobe 21: axisymmetrisches BEM (Kopf + Körper)."""
    # a) Kugelkontur reproduziert die Morse-Reihe (Pol-zu-Pol) < 2e-3.
    # b) Sphäroidkontur reproduziert die Sphäroid-Reihe < 2e-3 — zwei
    #    unabhängige exakte Referenzen für denselben Löser.
    # c) MONTIERT: die effektive Distanz der Kopf+Körper-Kontur liegt
    #    ZWISCHEN d_ext-Kugel und frei stehender Scheibe (der Körper
    #    unterbindet einen Teil des Scheibenrand-Umwegs); |G| ~ 1 im
    #    Tiefband; Raumwinkel-Residuum (Gitterqualität) klein.
    # d) Gatter: nur Doppelmembran-Bauform.
    if _HAS_SCIPY:
        th21 = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0]))

        def _inject(pts, i_rear=-1):
            el = MicrophoneCapsule._bem_elems(pts)
            n_el = el["L"].size
            fr_m = np.zeros(n_el, bool)
            fr_m[0] = True
            re_w = np.zeros(n_el)
            re_w[i_rear] = 1.0
            k67_bem._bem_geo = dict(
                elems=el, chief=[(0.0, 0.0)],
                w_area=2.0 * np.pi * el["mid_r"] * el["L"],
                front=fr_m, w_rear=re_w)

        psi21 = np.linspace(0.0, np.pi, 121)
        R21 = 9e-3
        _inject(np.stack([R21 * np.sin(psi21), R21 * np.cos(psi21)], 1))
        worst_k = 0.0
        for f21 in (100.0, 1000.0, 5000.0):
            om21 = np.array([2.0 * np.pi * f21])
            G_b = k67_bem._bem_axial_transfer(om21, th21)[0]
            d_save = k67.d_ext
            k67.d_ext = 2.0 * R21
            G_ref = k67._axial_body_transfer(om21, th21)[0]
            k67.d_ext = d_save
            worst_k = max(worst_k, float(np.max(np.abs(G_b - G_ref)
                                                / np.abs(G_ref))))
        assert worst_k < 2e-3, \
            f"BEM-Kugelkontur muss die Morse-Reihe treffen ({worst_k:.1e})"
        _inject(np.stack([17e-3 * np.sin(psi21),
                          6.1e-3 * np.cos(psi21)], 1))
        worst_o = 0.0
        for f21 in (1000.0, 5000.0):
            om21 = np.array([2.0 * np.pi * f21])
            G_b = k67_bem._bem_axial_transfer(om21, th21)[0]
            G_ref = k67._spheroid_pole_transfer(om21, th21, 17e-3,
                                                6.1e-3)[0]
            worst_o = max(worst_o, float(np.max(np.abs(G_b - G_ref)
                                                / np.abs(G_ref))))
        assert worst_o < 2e-3, \
            f"BEM-Sphäroidkontur muss die Sphäroid-Reihe treffen " \
            f"({worst_o:.1e})"
        # c) montierte Kapsel vs. freie Scheibe vs. d_ext-Kugel
        k67_bem._bem_geo = None
        om100 = np.array([2.0 * np.pi * 100.0])
        th180 = np.array([np.pi])
        k100 = om100[0] / C_AIR
        G_mnt = k67_bem._bem_axial_transfer(om100, th180)[0, 0]
        d_mnt = float(np.angle(G_mnt)) / k100
        assert k67_bem._bem_solid_angle_residual < 5e-3, \
            "BEM-Gitterqualität (Raumwinkel-Residuum)"
        k67_frei = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            axial_body_model="bem", bem_body_diameter=0.0)
        G_frei = k67_frei._bem_axial_transfer(om100, th180)[0, 0]
        d_frei = float(np.angle(G_frei)) / k100
        d_kugel = 1.5 * k67_bem.d_ext
        assert d_kugel < d_mnt < d_frei, \
            (f"montiert muss zwischen Kugel und freier Scheibe liegen "
             f"({d_kugel * 1e3:.1f} < {d_mnt * 1e3:.1f} < "
             f"{d_frei * 1e3:.1f} mm)")
        assert abs(abs(G_mnt) - 1.0) < 0.02, "BEM: |G| ~ 1 im Tiefband"
        # d) Gatter. 'spheroid' ist ein reines Front-Rück-Transfermodell und
        #    bleibt der Doppelmembran vorbehalten; 'bem' gilt auch für
        #    Ein-Membran-Kapseln und seit Gegenprobe 44 auch bei offener
        #    Rückseite (dort nur noch eine Warnung), braucht aber immer
        #    eine angegebene, hinreichend lange Körperlänge.
        _g21 = dict(membrane_resonance_hz=8000.0, architecture="single")
        _dicht = dict(_g21, n_cavity_holes=0, rear_network_enabled=True)
        for kw, was in (
                (dict(_g21, axial_body_model="spheroid"),
                 "spheroid ohne dual_diaphragm"),
                (dict(_dicht, axial_body_model="bem"),
                 "BEM ohne body_length"),
                (dict(_dicht, axial_body_model="bem", body_length=1e-3),
                 "BEM mit zu kurzem Körper"),
                (dict(membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
                      architecture="dual_diaphragm", center_gap=50e-6,
                      axial_body_model="bem", body_length=20e-3),
                 "body_length bei dual_diaphragm")):
            try:
                MicrophoneCapsule(**kw)
                raise AssertionError(f"{was} muss scheitern")
            except ValueError:
                pass
        print(f"BEM: Kugelkontur {worst_k:.1e}, Sphäroidkontur "
              f"{worst_o:.1e}; d_eff Kugel {d_kugel*1e3:.1f} < montiert "
              f"{d_mnt*1e3:.1f} < freie Scheibe {d_frei*1e3:.1f} mm "
              "(Körper unterbindet Teil des Rand-Umwegs)  OK")


@pytest.mark.bem
def test_gp26_bem_frontfaktor_exakter_druckstau_der():
    """Gegenprobe 26: BEM-Frontfaktor (exakter Druckstau der."""
    # FLACHEN Stirnfläche ersetzt die Kugelkalotten-Näherung im BEM-Modus).
    # Der BEM-Modus treibt die Frontmembran jetzt mit dem ABSOLUTEN
    # Flächenmittel ⟨p⟩ der realen flachen Stirnfläche (aus demselben
    # Lösungsgang wie der Front-Rück-Transfer) statt mit der Morse-
    # Kugelkalotte (bei der K67 um ±50° gekrümmt). Kein freier Parameter:
    # a) KUGELKONTUR-GRENZFALL: BEM-Frontmittel über die ±50°-Kalotte
    #    einer Kugelkontur == Kalottenmittel der Morse-Reihe (identische
    #    Geometrie, zwei unabhängige exakte Methoden).
    # b) FLACHKÖRPER-ABSOLUTREFERENZ: am oblaten Sphäroid (17 x 6.1 mm)
    #    muss das BEM-Frontmittel den ANALYTISCHEN Absolutdruck der
    #    Flammer-Reihe treffen:
    #        p(eta) = 2i/(c(xi0²+1)) Σ (−i)^n S_n(cosθ) S_n(eta)/(N_n R3'_n)
    #    — Vorfaktor aus der ebenen-Wellen-Expansion (Identität numerisch
    #    auf Maschinengenauigkeit geprüft) plus Wronski-Identität; eine
    #    unabhängige exakte Referenz für einen FLACHEN Körper.
    # c) PHYSIK DER FLACHEN STIRNFLÄCHE: ka→0 ⇒ F→1; im Band 5–9 kHz
    #    (ka ≈ 2..3) staut die SENKRECHT zur Welle stehende flache
    #    Stirnfläche +6.5..+8 dB (nahe Druckverdopplung samt Randbeugungs-
    #    Überschwingen), die ±50°-Kalotte nur +3.3..+4.1 dB — diese
    #    3–4-dB-Unterschätzung war die künstliche Vertiefung der
    #    7-kHz-Senke des Kalottenmodells (Kalotte bleibt für die
    #    Kugel-/Sphäroidmodi in Kraft, dort ist sie die konsistente
    #    Geometrie).
    # d) KONSISTENZ: H_neu = H_alt · F_bem/F_cap exakt — der Frontfaktor
    #    geht multiplikativ ein (H ∝ F·(a + b·G)), das Netzwerk bleibt
    #    unberührt.
    if _HAS_SCIPY:
        par26 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="2d", axial_body_model="bem",
            bem_body_diameter=0.0)
        k67f26 = MicrophoneCapsule(**par26)
        # a) Kugelkontur: Kalottenrand fällt exakt auf eine Gitterlinie
        R26 = k67f26.R_body
        psi_c = float(np.arcsin(k67f26.a_mem / R26))
        psi26 = np.concatenate([np.linspace(0.0, psi_c, 41),
                                np.linspace(psi_c, np.pi, 121)[1:]])
        el26 = MicrophoneCapsule._bem_elems(
            np.stack([R26 * np.sin(psi26), R26 * np.cos(psi26)], 1))
        ps_m = np.arctan2(el26["mid_r"], el26["mid_z"])
        k67f26._bem_geo = dict(
            elems=el26, chief=[(0.0, 0.0)],
            w_area=2.0 * np.pi * el26["mid_r"] * el26["L"],
            front=ps_m < psi_c,
            w_rear=((2.0 * np.pi * el26["mid_r"] * el26["L"])
                    * (ps_m > np.pi - psi_c)
                    * k67f26._membrane_mode_weight(el26["mid_r"])))
        th26 = np.deg2rad(np.array([0.0, 60.0, 120.0]))
        worst_f26 = 0.0
        for f26 in (1000.0, 7000.0):
            om26 = np.array([2.0 * np.pi * f26])
            F_b26 = k67f26._bem_axial_fields(om26, th26)[0]
            F_ref26, _ = k67f26._diffraction_factors(om26, th26)
            worst_f26 = max(worst_f26, float(np.max(
                np.abs(F_b26 - F_ref26) / np.abs(F_ref26))))
        assert worst_f26 < 2e-3, \
            f"BEM-Frontmittel muss die Morse-Kalotte treffen ({worst_f26:.1e})"
        # b) Sphäroid: BEM-Frontmittel vs. Flammer-Absolutreihe
        a26, b26 = 17e-3, 6.1e-3
        foc26 = np.sqrt(a26**2 - b26**2)
        xi26 = b26 / foc26
        psi_s = np.linspace(0.0, np.pi, 181)
        el_s = MicrophoneCapsule._bem_elems(
            np.stack([a26 * np.sin(psi_s), b26 * np.cos(psi_s)], 1))
        mrs, mzs = el_s["mid_r"], el_s["mid_z"]
        fr_s = (mzs > 0) & (mrs <= 13e-3)
        k67f26._bem_geo = dict(
            elems=el_s, chief=[(0.0, 0.0)],
            w_area=2.0 * np.pi * mrs * el_s["L"],
            front=fr_s,
            w_rear=((2.0 * np.pi * mrs * el_s["L"])
                    * ((mzs < 0) & (mrs <= 13e-3))
                    * k67f26._membrane_mode_weight(mrs)))
        # Referenzmittel MIT DEMSELBEN Operator wie das BEM-Frontmittel:
        # Fläche × Membran-Modengewicht (s. _cap_mode_quad). Sonst
        # verglichen man zwei verschiedene Mittelungen miteinander.
        w_s = ((2.0 * np.pi * mrs * el_s["L"])
               * k67f26._membrane_mode_weight(mrs))[fr_s]
        A_s = (mrs**2 + mzs**2) / foc26**2 - 1.0
        xi_s = np.sqrt(0.5 * (A_s + np.sqrt(A_s**2
                                            + 4.0 * mzs**2 / foc26**2)))
        eta_s = np.clip(mzs / (foc26 * xi_s), -1.0, 1.0)[fr_s]
        worst_s26 = 0.0
        for f26, thd26 in ((7000.0, 0.0), (7000.0, 60.0)):
            k26 = 2.0 * np.pi * f26 / C_AIR
            c26 = k26 * foc26
            th_s = float(np.deg2rad(thd26))
            modes26 = MicrophoneCapsule._oblate_modes(
                c26, int(np.ceil(c26)) + 12)
            p26 = np.zeros(int(np.sum(fr_s)), dtype=complex)
            for m26 in modes26:
                _, dR3_26 = MicrophoneCapsule._oblate_R3(m26, c26, xi26)
                S_th26 = MicrophoneCapsule._oblate_S(m26, np.cos(th_s))[0]
                p26 += ((-1j) ** m26["n"] * S_th26
                        * MicrophoneCapsule._oblate_S(m26, eta_s)
                        / (m26["N"] * dR3_26))
            p_ref26 = np.conj(2j / (c26 * (xi26**2 + 1.0))
                              * (w_s @ p26) / np.sum(w_s))
            F_b26 = k67f26._bem_axial_fields(
                np.array([2.0 * np.pi * f26]), np.array([th_s]))[0][0, 0]
            worst_s26 = max(worst_s26,
                            float(abs(F_b26 - p_ref26) / abs(p_ref26)))
        assert worst_s26 < 1e-3, \
            (f"BEM muss die Flammer-Absolutreihe am Sphäroid treffen "
             f"({worst_s26:.1e})")
        # c) reale K67-Kopfgeometrie (frei): Grenzfälle und Band 5-9 kHz
        k67f26._bem_geo = None
        om_c26 = 2.0 * np.pi * np.array([100.0, 5000.0, 7000.0, 9000.0])
        th0_26 = np.array([0.0])
        F_flat26 = k67f26._bem_axial_fields(om_c26, th0_26)[0][:, 0]
        F_cap26, _ = k67f26._diffraction_factors(om_c26, th0_26)
        assert abs(abs(F_flat26[0]) - 1.0) < 5e-3, \
            f"ka->0 muss F->1 liefern ({abs(F_flat26[0]):.4f})"
        d_band26 = 20.0 * np.log10(np.abs(F_flat26[1:] / F_cap26[1:, 0]))
        assert np.all((d_band26 > 2.5) & (d_band26 < 4.5)), \
            (f"flache Stirnfläche muss die Kalotte um 3-4 dB übertreffen "
             f"({np.round(d_band26, 2)})")
        f_band26 = 20.0 * np.log10(np.abs(F_flat26[1:]))
        # Der Druckstau der flachen Stirnfläche liegt ÜBER der starren
        # unendlichen Wand (+6.02 dB): die Mitte einer Scheibe ist ein
        # Fokus, weil der Rand von dort überall gleich weit entfernt ist
        # und die Randwellen kohärent addieren. Wie weit darüber, hängt
        # vom MITTELUNGSGEWICHT ab — deshalb hier keine kalibrierte
        # Bandbreite, sondern die weichungsfreie ORDNUNG
        #     Flächenmittel < Modenmittel < Scheibenmitte,
        # die für jedes mittenlastige Gewicht gelten MUSS. Nur die untere
        # Schranke ist absolut (starre Wand).
        assert np.all(f_band26 > 6.02), \
            (f"Druckstau muss die starre Wand (+6.02 dB) übertreffen "
             f"({np.round(f_band26, 2)})")

        class _FlaechenFrontBem(MicrophoneCapsule):
            """Frontmittel flächengleich — untere Ordnungsschranke."""

            def _membrane_mode_weight(self, r, mode=1):
                return np.ones_like(np.asarray(r, dtype=float))

        class _ZentrumFrontBem(MicrophoneCapsule):
            """Frontmittel nur über die Scheibenmitte — obere Schranke."""

            def _membrane_mode_weight(self, r, mode=1):
                return (np.asarray(r, dtype=float)
                        < 0.1 * self.a_mem).astype(float)

        om_o26 = 2.0 * np.pi * np.array([9000.0])       # ungünstigster Fall
        F_ar26 = abs(_FlaechenFrontBem(**par26)._bem_axial_fields(
            om_o26, th0_26)[0][0, 0])
        F_ze26 = abs(_ZentrumFrontBem(**par26)._bem_axial_fields(
            om_o26, th0_26)[0][0, 0])
        F_mo26 = abs(F_flat26[3])
        assert F_ar26 < F_mo26 < F_ze26, \
            (f"Modenmittel muss zwischen Flächenmittel und Scheibenmitte "
             f"liegen ({20 * np.log10(F_ar26):.2f} / "
             f"{20 * np.log10(F_mo26):.2f} / {20 * np.log10(F_ze26):.2f} dB)")
        # d) Multiplikativität über den vollen Signalpfad (Netzwerk unberührt)

        class _KalottenFrontBem(MicrophoneCapsule):
            """Alter BEM-Modus (Kalotten-F, BEM-G) — nur Konsistenzprobe."""

            def _bem_axial_fields(self, omega, theta):
                _, G_b = MicrophoneCapsule._bem_axial_fields(
                    self, omega, theta)
                return self._diffraction_factors(omega, theta)[0], G_b

        f26h = [1000.0, 7000.0]
        om26h = 2.0 * np.pi * np.asarray(f26h)
        H_neu26 = k67f26.transfer_function(f26h)
        H_alt26 = _KalottenFrontBem(**par26).transfer_function(f26h)
        F_n26 = k67f26._bem_axial_fields(om26h, th0_26)[0][:, 0]
        F_c26 = k67f26._diffraction_factors(om26h, th0_26)[0][:, 0]
        ratio26 = float(np.max(np.abs(H_neu26 / H_alt26 - F_n26 / F_c26)
                               / np.abs(F_n26 / F_c26)))
        assert ratio26 < 1e-9, \
            f"Frontfaktor muss exakt multiplikativ eingehen ({ratio26:.1e})"
        print(f"BEM-Frontfaktor: Kugelkontur == Morse-Kalotte "
              f"{worst_f26:.1e}; Sphäroid-Absolutreihe {worst_s26:.1e}; "
              f"ka->0: |F| = {abs(F_flat26[0]):.4f}; flache Stirnfläche "
              f"+{f_band26[1]:.1f} dB bei 7 kHz (Kalotte unterschätzt um "
              f"{d_band26[1]:.1f} dB); multiplikativ {ratio26:.1e}  OK")


@pytest.mark.feld3d
def test_gp27_strahlungsimpedanz_3d_aussenknoten():
    """Gegenprobe 27: Strahlungsimpedanz + 3D-Außenknoten."""
    # a) EXAKTE KOLBENSTRAHLUNG statt Kleinargument-Asymptote. Bis
    #    Gegenprobe 26 stand im Frontzweig R ~ (ka)²/2, X ~ 8ka/(3pi) —
    #    für ka -> 0 exakt, oberhalb ka ~ 1 aber grob falsch: die
    #    Reaktanz X1 hat ein MAXIMUM und fällt danach ab, die Asymptote
    #    wächst linear weiter. Folge war eine KONSTANTE Zusatzmasse von
    #    25 kg/m^4 (die Membran selbst hat nur 20.9), die den Hochton um
    #    ~7 dB bei 16 kHz niederhielt. Verankert an:
    #      - Lehrbuch-Stützwert R1(2) = 1 - J1(2) = 0.4233,
    #      - Kleinargument-Grenzfall == alte Asymptote (Stetigkeit),
    #      - Hochton: mitschwingende Masse MUSS zusammenbrechen,
    #      - omega = 0 bleibt endlich (keine 0/0-Auslöschung).
    # b) 3D-AUSSENKNOTEN der Doppelmembran-Bauform. Der 3D-Löser trieb die
    #    Membranaußenseiten direkt aus der Quelle — Strahlungsimpedanz und
    #    Gewebe fehlten ERSATZLOS (fabric_* blieb im 3D-Modus exakt
    #    wirkungslos, ein still falsches Ergebnis). Jetzt zwei Sammel-
    #    knoten wie bei single/dual. Verankert an:
    #      - GRENZFALL Z_außen -> 0 reproduziert den Direktantrieb, und
    #        zwar mit ERSTER ORDNUNG (Z zehnfach kleiner -> Abstand zum
    #        Grenzwert zehnfach kleiner) — beweist Vorzeichen und Struktur,
    #      - Passivität: Gewebe dämpft monoton (nie Verstärkung),
    #      - Reziprozität X_r = -B_f bleibt erhalten.
    #    Die Dämpfung im 1D/2D-Pfad wird nur zum Vergleich ausgegeben (s. u.).
    if _HAS_SCIPY:
        from scipy.special import j1 as _j1_27, struve as _struve_27
        cap27 = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=40e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, squeeze_model="2d")
        Z0_27 = RHO0 * C_AIR / cap27.S_mem
        a27 = cap27.a_mem
        # Lehrbuch-Stützwert bei 2ka = 2
        f_x2 = C_AIR / (2.0 * np.pi * a27)            # -> ka = 1, x = 2
        Z_x2 = cap27._radiation_impedance_membrane(
            np.array([2.0 * np.pi * f_x2]))[0]
        assert abs(Z_x2.real / Z0_27 - (1.0 - float(_j1_27(2.0)))) < 1e-9, \
            f"R1(2) muss 1-J1(2) = 0.4233 sein ({Z_x2.real / Z0_27:.4f})"
        assert abs(Z_x2.imag / Z0_27 - float(_struve_27(1, 2.0))) < 1e-9, \
            "X1(2) muss 2*H1(2)/2 treffen"
        # Kleinargument: exakte Form == alte Asymptote (stetiger Anschluss)
        om_lf27 = np.array([2.0 * np.pi * 20.0])
        Z_lf = cap27._radiation_impedance_membrane(om_lf27)[0]
        ka_lf = om_lf27[0] * a27 / C_AIR
        assert abs(Z_lf.imag / Z0_27 / (8.0 * ka_lf / (3.0 * np.pi)) - 1.0) \
            < 1e-5, "ka -> 0 muss die alte Asymptote reproduzieren"
        M_class = 8.0 * RHO0 / (3.0 * np.pi ** 2 * a27)
        M_lf = Z_lf.imag / om_lf27[0]
        assert abs(M_lf / M_class - 1.0) < 1e-4, \
            (f"LF-Luftmasse muss 8*rho0/(3 pi^2 a) = {M_class:.1f} sein "
             f"({M_lf:.1f} kg/m^4)")
        # Hochton: die mitschwingende Masse MUSS zusammenbrechen
        om_hf27 = 2.0 * np.pi * np.array([16000.0])
        M_hf = (cap27._radiation_impedance_membrane(om_hf27)[0].imag
                / om_hf27[0])
        assert M_hf < 0.1 * M_lf, \
            (f"Strahlungsmasse muss im Hochton verschwinden "
             f"({M_hf:.2f} vs. {M_lf:.1f} kg/m^4)")
        assert np.all(np.isfinite(cap27._radiation_impedance_membrane(
            np.array([0.0])))), "omega = 0 muss endlich bleiben"

        # ---- b) 3D-Außenknoten ----
        par27 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=40e-6, n_through_holes=12,
            through_hole_diameter=0.6e-3, n_blind_holes=24,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True)
        om27 = 2.0 * np.pi * np.array([1000.0])
        _rad_orig = MicrophoneCapsule._radiation_impedance_membrane

        def _rad_eps(self, omega, _e=1.0):
            om_ = np.asarray(omega, dtype=float)
            return np.full(om_.shape, _e * RHO0 * C_AIR / self.S_mem,
                           dtype=complex)

        try:
            X_lim = {}
            for e27 in (1e-3, 1e-4, 1e-5):
                MicrophoneCapsule._radiation_impedance_membrane = \
                    (lambda s, o, _e=e27: _rad_eps(s, o, _e))
                X_lim[e27] = MicrophoneCapsule(
                    squeeze_model="3d", fabric_front_rayl=0.0,
                    fabric_rear_rayl=0.0, **par27)._solve_3d(om27)[0][0]
        finally:
            MicrophoneCapsule._radiation_impedance_membrane = _rad_orig
        d1 = abs(X_lim[1e-3] - X_lim[1e-5])
        d2 = abs(X_lim[1e-4] - X_lim[1e-5])
        assert d2 < 0.2 * d1, \
            (f"Z -> 0 muss mit erster Ordnung konvergieren "
             f"(zehnfach kleineres Z: {d2:.3e} vs. {d1:.3e})")
        # Passivität + Topologie-Konsistenz gegen den 1D/2D-Pfad
        att = {}
        for mdl27 in ("2d", "3d"):
            H0_27 = MicrophoneCapsule(
                squeeze_model=mdl27, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, **par27).transfer_function([1000.0])
            lv = []
            for r27 in (1.0e3, 1.0e4, 1.0e5):
                H_27 = MicrophoneCapsule(
                    squeeze_model=mdl27, fabric_front_rayl=r27,
                    fabric_rear_rayl=0.0, **par27).transfer_function([1000.0])
                lv.append(float(20.0 * np.log10(np.abs(H_27[0])
                                                / np.abs(H0_27[0]))))
            att[mdl27] = lv
            assert lv[0] < -1e-3, \
                f"{mdl27}: Gewebe muss im 3D-Modus überhaupt wirken ({lv[0]})"
            assert lv[0] > lv[1] > lv[2], \
                f"{mdl27}: Gewebe muss monoton dämpfen (Passivität) {lv}"
        # Dämpfung gegen die 1D/2D-Kette bei 100 Hz — nur AUSGEGEBEN, kein
        # Gleichheitstest mehr. Bis Gegenprobe 51 trafen sich beide auf
        # 0,04 dB; das war Zufall. Beim Druckgradientenempfänger ist die
        # Gewebedämpfung im Tiefton die Differenz aus Gradientenantrieb
        # (∝ jkd) und dem Gleichtakt-Druckabfall am Gewebe (∝ jω·Z·Y_gl);
        # beide sind hier vergleichbar groß, die Differenz verstärkt jeden
        # Unterschied der inneren Gleichtaktantwort. Die bestimmt bei
        # diesem Prüfling (12 gestufte Löcher, Zwischenspalt) der innere
        # Widerstandspfad, den 2D (Škvor-Zellregel) und 3D verschieden
        # rechnen: Gleichtaktantwort im 3D bei 10 Hz um 34 % größer
        # (dichte einteilige Platte, 96 Löcher: 4 %). Bis Gegenprobe 52
        # glich das der unbelastete Membranring zufällig aus; danach
        # −4,9 gegen −6,0 dB, statisch (10 Hz: −4,7 gegen −5,9, bei 1 V
        # Vorspannung ebenso), mit der Θ-konsistenten 3D-Wandlung
        # (Gegenprobe 54) −5,3 dB. Der Außenknoten selbst ist durch Grenzfall,
        # Passivität und Reziprozität belegt.
        att100 = {}
        for mdl27 in ("2d", "3d"):
            h27 = [abs(MicrophoneCapsule(
                squeeze_model=mdl27, fabric_front_rayl=r27,
                fabric_rear_rayl=0.0, **par27).transfer_function([100.0])[0])
                for r27 in (0.0, 1.0e5)]
            att100[mdl27] = float(20.0 * np.log10(h27[1] / h27[0]))
        assert att100["3d"] < -3.0 and att100["2d"] < -3.0, \
            (f"Gewebe (10⁵ Rayl) muss in beiden Modellen deutlich dämpfen "
             f"({att100['3d']:.2f} vs. {att100['2d']:.2f} dB bei 100 Hz)")
        # Reziprozität der Membranports bleibt erhalten
        c27r = MicrophoneCapsule(squeeze_model="3d", fabric_front_rayl=0.0,
                                 fabric_rear_rayl=0.0, **par27)
        Xf27, Xr27, Bf27, Br27 = c27r._solve_3d(om27, want_rear=True,
                                                 weight="volume")
        assert abs(Xr27[0] / Bf27[0] + 1.0) < 5e-3, \
            (f"Reziprozität X_r = -B_f muss erhalten bleiben "
             f"({Xr27[0] / Bf27[0]:.4f})")
        print(f"Strahlung/3D-Knoten: R1(2) = {Z_x2.real / Z0_27:.4f} "
              f"(= 1-J1(2)); Luftmasse {M_lf:.1f} kg/m^4 (LF, klassisch) "
              f"-> {M_hf:.2f} bei 16 kHz; 3D-Grenzfall Z->0 konvergiert "
              f"1. Ordnung ({d2 / d1:.3f}); Gewebe wirkt jetzt im 3D "
              f"({att100['3d']:.2f} dB vs. 2D {att100['2d']:.2f} dB bei "
              f"100 Hz, 1 kHz {att['3d'][2]:.1f} vs. {att['2d'][2]:.1f} dB "
              f"— innerer Gleichtaktpfad, nur ausgegeben); reziprok  OK")


@pytest.mark.slow
@pytest.mark.bem
def test_gp41_bem_frontfaktor_der_flachen_stirnflache():
    """Gegenprobe 41: BEM-Frontfaktor der flachen Stirnfläche."""
    # Eine Ein-Membran-Kapsel ist kein Ball. Die Kugelkalotte
    # (_diffraction_factors) legt die Membran auf eine um ±40..50°
    # gekrümmte Fläche; die reale Kapsel hat eine FLACHE Stirnfläche, auf
    # der die Membran senkrecht zur Einfallsrichtung steht. Der Druckstau
    # ist dort deutlich größer, und weil eine endliche Scheibe zusätzlich
    # Randwellen auf die Achse fokussiert, übersteigt er die
    # Verdopplung (+6 dB) der unendlichen Wand.
    #
    # ANKER: A. J. Zuckerwars Gegenstück auf der Messseite ist hier
    # R. S. Grinnip III, "Advanced Simulation of a Condenser Microphone
    # Capsule", J. Audio Eng. Soc. 54(3), 157–167 (2006). Tabelle 1 gibt
    # den Prototyp vollständig (Membran ⌀21.89 mm / 2.4 µm / 1630 kg/m³,
    # f_vak 3500 Hz, Spalt 50.8 µm, Backplate ⌀22 mm / 0.762 mm mit 84
    # Bohrungen ⌀1.524 mm auf fünf Lochkreisen, Rückkammer, Körper
    # ⌀33 × 11.5 mm, 73.5 V), Fig. 5/6/7 gemessene Kurven bei 0/90/180°.
    # Die Kapsel ist absichtlich extrem: gemessen wird bis 18 kHz, also
    # das 5.1-fache der Vakuumresonanz.
    #
    # ZWEI LESARTEN in Tabelle 1: "h_c = 5.955e-7/b²" ist als Zahl
    # mehrdeutig. Als Kammerhöhe gelesen (4.921 mm, V = 1.87 cm³) liegt
    # unser Modell 10.4 dB RMS daneben, als Volumen (V = 5.955e-7 m³,
    # h_c = 1.567 mm) 1.6 dB — die Physik entscheidet eindeutig für die
    # zweite. Die Kurvenpunkte stehen oben (digitalisiert, nicht
    # geschätzt).
    #
    # WAS DIE PROBE ZEIGT — und was NICHT. Auf Achse braucht die Messung
    # einen Frontfaktor von +6.9…+9.6 dB im Band 5…16 kHz. Eine starre
    # 33-mm-KUGEL kann das prinzipiell nicht: ihre Kalottenmittelung
    # sättigt bei ~+5 dB. Die flache Stirnfläche erreicht +8.9 dB und
    # trifft Fig. 5 damit auf 1.0 dB RMS statt 3.1 dB mit der Kugel. Das
    # ist der eigentliche Befund und wird unten beidseitig geprüft.
    # OFF-AXIS bleibt ein Rest, aber ein viel kleinerer, als hier früher
    # stand: mit EINER Mode fehlen bei 90° 4.9 dB bei 14 kHz und 9.7 dB
    # bei 18 kHz (3.7 dB RMS über das Band), bei 180° 2.9 dB RMS. Der
    # frühere Eintrag "~17 dB bei 90°/14 kHz" war ein Ablesefehler in
    # Fig. 6 und ist mit der digitalisierten Kurve gegenstandslos.
    # Einen Teil des Restes trägt der modenweise projizierte Antrieb,
    # den Grinnip mitrechnet (Gegenprobe 43: 3.7 -> 2.4 dB RMS bei 90°,
    # 3.0 -> 1.7 dB bei 180°); der Rest ist offen. Die Schranken unten
    # halten den Stand fest, damit er nur kleiner werden kann.
    if _HAS_SCIPY:
        g41 = dict(
            membrane_material={"rho": 1630.0, "E": 4.9e9, "nu": 0.37},
            membrane_resonance_hz=None, membrane_diameter=2 * 1.0945e-2,
            membrane_thickness=2.4e-6,
            # T aus c_s = 2π f_vak a / j01 (Tab. 1, Gl. 56/57)
            membrane_tension=(1630.0 * 2.4e-6
                              * (2 * np.pi * 3500.0 * 1.0945e-2
                                 / 2.404825557695773) ** 2),
            air_gap=5.08e-5, backplate_diameter=2 * 1.1e-2,
            backplate_thickness=7.62e-4, bias_voltage=73.5,
            architecture="single",
            through_hole_rings=[(6, 4e-3), (12, 8e-3), (18, 12e-3),
                                (24, 16e-3), (24, 20e-3)],
            through_hole_diameter=2 * 7.62e-4, n_blind_holes=0,
            rear_network_enabled=True, delay_length=0.0,
            cavity_length=5.955e-7 / (np.pi * 1.1e-2 ** 2),
            n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=33e-3, include_diffraction=True,
            squeeze_model="2d")
        c41s = MicrophoneCapsule(**g41)
        c41b = MicrophoneCapsule(**dict(g41, axial_body_model="bem",
                                        body_length=11.5e-3,
                                        bem_body_diameter=0.0))
        th41 = np.deg2rad(np.array([0.0, 90.0, 180.0]))
        # a) ka -> 0: der BETRAG geht gegen 1 (der Körper staut nichts
        #    mehr). Die PHASE tut das nicht und soll es auch nicht: sie
        #    trägt die Dipolstreuung des Körpers, also den akustischen
        #    Mittelpunkt. Für eine Kugel ist der bekannte Grenzwert 1.5·R;
        #    für die flache Scheibe muss er in derselben Größenordnung
        #    liegen (die Scheibe ist radial so groß wie die Kugel, axial
        #    dünner). Geprüft wird beides.
        om41a = np.array([2 * np.pi * 20.0])
        F0 = c41b._source_pressures(om41a, th41)[0]
        assert np.max(np.abs(np.abs(F0) - 1.0)) < 1e-3, \
            f"|F| muss für ka -> 0 gegen 1 gehen ({np.abs(F0)})"
        d_ac = float(np.angle(F0[0, 0])) / (om41a[0] / C_AIR)
        assert 0.5 * c41b.R_body < d_ac < 2.0 * c41b.R_body, \
            (f"akustischer Mittelpunkt bei ka -> 0: {d_ac * 1e3:.1f} mm "
             f"muss in der Größenordnung von R_body = "
             f"{c41b.R_body * 1e3:.1f} mm liegen")
        # b) flache Stirnfläche gegen Kugelkalotte im Band ka = 1.5..4
        f41 = np.array([5000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0])
        om41, ax41 = 2 * np.pi * f41, th41[:1]
        Fb = 20 * np.log10(np.abs(
            c41b._source_pressures(om41, ax41)[0][:, 0]))
        Fs = 20 * np.log10(np.abs(
            c41s._source_pressures(om41, ax41)[0][:, 0]))
        assert np.all(Fb > Fs + 1.0), \
            (f"die flache Stirnfläche muss mehr stauen als die Kalotte "
             f"(BEM {np.round(Fb, 2)} gegen Kugel {np.round(Fs, 2)} dB)")
        assert Fs.max() < 5.5, \
            (f"die Kugelkalotte kann den gemessenen Druckstau prinzipiell "
             f"nicht liefern — sie sättigt bei {Fs.max():.2f} dB")
        assert Fb.max() > 6.0, \
            (f"die flache Stirnfläche muss über die Verdopplung hinaus "
             f"fokussieren ({Fb.max():.2f} dB)")
        assert c41b._bem_solid_angle_residual < 5e-3, \
            "BEM-Gitterqualität (Raumwinkel-Residuum)"

        # b2) GITTERKONVERGENZ: dasselbe mit dreifach feinerem Meridian.
        #     Ein Randelementergebnis ohne diesen Nachweis ist wertlos.
        class _Fein41(MicrophoneCapsule):
            _BEM_H_MAX = MicrophoneCapsule._BEM_H_MAX / 3.0

        c41f = _Fein41(**dict(g41, axial_body_model="bem",
                              body_length=11.5e-3, bem_body_diameter=0.0))
        f41c = np.array([2000.0, 6000.0, 10000.0, 14000.0, 18000.0])
        om41c = 2 * np.pi * f41c
        d41c = np.max(np.abs(
            20 * np.log10(np.abs(c41f._source_pressures(om41c, th41)[0]))
            - 20 * np.log10(np.abs(c41b._source_pressures(om41c, th41)[0]))))
        n_grob = c41b._bem_geometry()["elems"]["L"].size
        n_fein = c41f._bem_geometry()["elems"]["L"].size
        assert n_fein > 2.5 * n_grob, "feineres Gitter muss feiner sein"
        assert d41c < 0.2, \
            (f"BEM-Frontfaktor muss gitterkonvergent sein ({n_grob} -> "
             f"{n_fein} Elemente ändern {d41c:.3f} dB)")
        # c) Frequenzgang auf Achse gegen Fig. 5 (VC, digitalisiert)
        r41b, _ = _grin_rms(c41b, 0.0)
        r41s, _ = _grin_rms(c41s, 0.0)
        assert r41b < 1.5, \
            f"BEM auf Achse muss Fig. 5 auf < 1.5 dB treffen ({r41b:.2f})"
        assert r41b < r41s - 1.0, \
            (f"der BEM-Frontfaktor muss die Kugel deutlich schlagen "
             f"({r41b:.2f} gegen {r41s:.2f} dB)")
        # c2) und gegen die MESSUNG dort, wo EXP von VC zu trennen ist.
        #     Beide Rechnungen liegen im Fenster 9…15 kHz über der
        #     Messung — Grinnip nennt für seine eigene bis ~5 dB.
        r41e, d41e = _grin_rms(c41b, 0.0, f_exp0, exp0_grin)
        r41ve = float(np.sqrt(np.mean(
            (np.interp(f_exp0, f_grin, vc_grin[0.0]) - exp0_grin) ** 2)))
        assert r41e < 5.0, \
            (f"auf Achse gegen die Messung im Fenster 9…15 kHz: "
             f"{r41e:.2f} dB RMS (Grinnips eigene Rechnung {r41ve:.2f})")
        assert np.all(d41e > 0.0), \
            (f"beide Rechnungen liegen dort ÜBER der Messung "
             f"({np.round(d41e, 1)})")
        # d) dokumentierter Rest off-axis bei EINER Mode
        r41_90, d41_90 = _grin_rms(c41b, 90.0)
        d41 = float(d41_90[f_grin == 14000.0][0])
        assert r41_90 < 4.5, \
            (f"90° mit einer Mode: {r41_90:.2f} dB RMS gegen Fig. 6 — "
             f"Schranke, damit der Rest nur kleiner wird")
        assert 2.0 < d41 < 6.5, \
            (f"dokumentierter Off-Axis-Rest bei 90°/14 kHz "
             f"({d41:+.1f} dB) — Schranke, damit er nur kleiner wird")
        print(f"BEM-Frontfaktor (Grinnip 2006, Shure-Prototyp): "
              f"|F|(ka->0) = 1 ({np.max(np.abs(np.abs(F0) - 1.0)):.0e}), "
              f"akust. Mittelpunkt {d_ac * 1e3:.1f} mm bei "
              f"R_body {c41b.R_body * 1e3:.1f} mm; flache "
              f"Stirnfläche staut bis {Fb.max():+.1f} dB, die Kugelkalotte "
              f"sättigt bei {Fs.max():+.1f} dB (gemessen nötig +6.9…+9.6); "
              f"Fig. 5 auf Achse {r41b:.2f} dB RMS gegen {r41s:.2f} dB mit "
              f"Kugel, gegen die Messung {r41e:.2f} dB (Grinnip selbst "
              f"{r41ve:.2f}); gitterkonvergent ({n_grob}->{n_fein} "
              f"Elemente: {d41c:.3f} dB); off-axis bleiben mit einer Mode "
              f"{r41_90:.2f} dB RMS bei 90° ({d41:+.1f} dB bei 14 kHz), "
              f"dokumentiert  OK")


@pytest.mark.slow
@pytest.mark.bem
def test_gp43_modenfaktoren_aus_demselben_korper():
    """Gegenprobe 43: Modenfaktoren aus DEMSELBEN Körper."""
    # Mit ``modal_source`` wird jede Membranmode von ihrer eigenen
    # Galerkin-Projektion getrieben, und die Kette zieht das über die
    # Modenadmittanzen zu einer Ersatzquelle zusammen (Gegenprobe 34).
    # Zwei Dinge waren daran inkonsistent:
    #
    # 1) HERKUNFT. Seit Gegenprobe 41 kommt der Frontfaktor der Grundmode
    #    aus dem BEM (reale flache Stirnfläche), die Verhältnisse p_m/p_1
    #    kamen aber weiter aus der Kugelkalotte — zwei Körpermodelle in
    #    einer Größe. Bei 14 kHz und 90° unterscheiden sich die beiden um
    #    0.72 im komplexen Faktor, das ist keine Feinheit. Jetzt liefert
    #    EIN BEM-Lösungsgang alle Moden: die Randintegralgleichung hängt
    #    nicht von der Modenform ab, nur die Projektion danach.
    #
    # 2) DÄMPFUNG. Die Gewichtung benutzte für die höheren Zweige nur den
    #    Filmwiderstand R, während _modal_parallel sie zusätzlich mit
    #    ihrer inneren Umverteilung belastet (_modal_internal_Z). Damit
    #    schoss Y_m an der zweiten Modenfrequenz hoch und zog p_eff dort
    #    auf p_2 — im Frequenzgang eine scharfe Senke von 4.4 dB bei
    #    8.2 kHz, die in Grinnips Messung nicht existiert. Jetzt stehen an
    #    beiden Stellen dieselben Zweige.
    #
    # 3) WOFÜR DAS GANZE. Der modenweise projizierte Antrieb ist Grinnips
    #    eigentliche Lehre und muss sich messbar auszahlen: bei
    #    Streifeinfall ist der Druck über die Membran stark
    #    ungleichförmig, die höheren Moden bekommen Amplitude, und ihr
    #    Flächenmittel 2·J1(z_m)/z_m ist klein. Genau dieser Hochtonabfall
    #    fehlt der Einmoden-Rechnung. Teil f/g prüft beides gegen die
    #    digitalisierten Kurven.
    #
    # Verankert wird das an exakten Grenzwerten und an der Messung.
    if _HAS_SCIPY:
        g43 = dict(
            membrane_material={"rho": 1630.0, "E": 4.9e9, "nu": 0.37},
            membrane_resonance_hz=None, membrane_diameter=2 * 1.0945e-2,
            membrane_thickness=2.4e-6,
            membrane_tension=(1630.0 * 2.4e-6
                              * (2 * np.pi * 3500.0 * 1.0945e-2
                                 / 2.404825557695773) ** 2),
            air_gap=5.08e-5, backplate_diameter=2 * 1.1e-2,
            backplate_thickness=7.62e-4, bias_voltage=73.5,
            architecture="single",
            through_hole_rings=[(6, 4e-3), (12, 8e-3), (18, 12e-3),
                                (24, 16e-3), (24, 20e-3)],
            through_hole_diameter=2 * 7.62e-4, n_blind_holes=0,
            rear_network_enabled=True, delay_length=0.0,
            cavity_length=5.955e-7 / (np.pi * 1.1e-2 ** 2),
            n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=33e-3, include_diffraction=True,
            squeeze_model="2d", axial_body_model="bem",
            body_length=11.5e-3, bem_body_diameter=0.0)
        c43 = MicrophoneCapsule(**dict(g43, membrane_modes=5, modal_source=1))
        th43 = np.deg2rad(np.array([90.0]))

        # a) ka -> 0: der Druck ist über die Membran gleichförmig, also
        #    ist JEDE Modenprojektion gleich der der Grundmode und die
        #    Ersatzquelle exakt 1. Das prüft die BEM-Projektion selbst.
        om43a = np.array([2.0 * np.pi * 20.0])
        F43a, _ = c43._bem_front_modes(om43a, th43)
        r43a = np.max(np.abs(F43a[:, 0, 0] / F43a[0, 0, 0] - 1.0))
        assert r43a < 1e-3, \
            f"ka -> 0: alle Modenfaktoren müssen 1 sein ({r43a:.1e})"
        s43a = c43._modal_source_scale(om43a, th43, F43a[0])
        assert abs(s43a[0, 0] - 1.0) < 1e-3, \
            f"ka -> 0: die Ersatzquelle muss 1 sein ({s43a[0, 0]})"

        # b) EIN Lösungsgang für alle Moden, und der Aufsatz ist derselbe
        F43b, G43b = c43._bem_front_modes(om43a, th43)
        assert F43b is F43a, "der Lösungsgang darf sich nicht wiederholen"
        assert F43b.shape[0] == 5, \
            f"alle Moden aus einem Lauf ({F43b.shape})"
        assert np.array_equal(c43._bem_axial_fields(om43a, th43)[0],
                              F43b[0]), \
            "_bem_axial_fields muss die Grundmode dieses Laufs sein"

        # c) die Modenfaktoren dürfen NICHT die der Kugelkalotte sein
        cs43 = MicrophoneCapsule(**dict(g43, membrane_modes=5,
                                        modal_source=1,
                                        axial_body_model="sphere",
                                        body_length=None,
                                        bem_body_diameter=56e-3))
        om43c = np.array([2.0 * np.pi * 14000.0])
        Fb43, _ = c43._bem_front_modes(om43c, th43)
        F1s43, _ = cs43._diffraction_factors(om43c, th43)
        F2s43, _ = cs43._diffraction_factors(om43c, th43, mode=2)
        d43 = abs(Fb43[1, 0, 0] / Fb43[0, 0, 0] - (F2s43 / F1s43)[0, 0])
        assert d43 > 0.3, \
            (f"flache Stirnfläche und Kugelkalotte müssen sich in den "
             f"Modenfaktoren unterscheiden ({d43:.3f})")

        # d) keine künstliche Senke an der zweiten Modenfrequenz. Der
        #    Gegenbeweis läuft mit: OHNE die innere Umverteilung in der
        #    Gewichtung muss die Senke wieder auftauchen.
        class _OhneUmverteilung43(MicrophoneCapsule):
            def _modal_internal_Z(self, omega, h_film):
                n = max(self.membrane_modes - 1, 0)
                return [np.zeros_like(np.atleast_1d(omega), dtype=complex)
                        for _ in range(n)]

        f2_43 = c43.f_res * MicrophoneCapsule._J0_ZEROS[1] \
            / MicrophoneCapsule._J0_ZEROS[0]
        ff43 = np.array([0.8 * f2_43, f2_43, 1.25 * f2_43])
        kerbe = {}
        for lbl43, cc43 in (("ok", c43),
                            ("roh", _OhneUmverteilung43(
                                **dict(g43, membrane_modes=5,
                                       modal_source=1)))):
            a43 = 20 * np.log10(np.abs(
                cc43.transfer_function(ff43, angle_deg=90.0)))
            kerbe[lbl43] = float(a43[1] - 0.5 * (a43[0] + a43[2]))
        assert abs(kerbe["ok"]) < 0.5, \
            (f"keine künstliche Senke bei der zweiten Modenfrequenz "
             f"({kerbe['ok']:+.2f} dB)")
        assert kerbe["roh"] < -2.0, \
            (f"ohne die innere Umverteilung MUSS die Senke auftreten — "
             f"sonst prüft dieser Test nichts ({kerbe['roh']:+.2f} dB)")

        # e) membrane_modes = 1: die Ersatzquelle ist exakt 1, also darf
        #    modal_source nichts ändern
        f43e = np.array([1000.0, 12000.0])
        h43a = MicrophoneCapsule(**dict(g43, membrane_modes=1,
                                        modal_source=1)
                                 ).transfer_function(f43e, angle_deg=90.0)
        h43b = MicrophoneCapsule(**dict(g43, membrane_modes=1,
                                        modal_source=0)
                                 ).transfer_function(f43e, angle_deg=90.0)
        assert np.max(np.abs(h43a - h43b)) == 0.0, \
            "bei einer Mode darf modal_source nichts ändern"

        # f) gegen Grinnip Fig. 5/6/7 (digitalisiert, s. Messanker oben).
        #    Schranken als dokumentierter Stand.
        rms43 = {}
        for ang43, lim43 in ((0.0, 1.5), (90.0, 3.0), (180.0, 2.3)):
            rms43[ang43], _ = _grin_rms(c43, ang43)
            assert rms43[ang43] < lim43, \
                (f"{ang43:.0f}°: {rms43[ang43]:.2f} dB RMS gegen Fig. 5/6/7 "
                 f"(Schranke {lim43})")
        r43e, _ = _grin_rms(c43, 90.0, f_exp90, exp90_grin)
        assert r43e < 3.5, \
            f"90° gegen die MESSUNG: {r43e:.2f} dB RMS (Schranke 3.5)"

        # g) DIE EIGENTLICHE LEHRE aus Grinnip: der modenweise projizierte
        #    Antrieb muss die Streifeinfall-Rechnung messbar verbessern,
        #    ohne die Achse zu verschlechtern. Bei 90° und 180° ist der
        #    Antrieb über die Membran stark ungleichförmig, die höheren
        #    Moden bekommen dort Amplitude, und ihr Flächenmittel
        #    2·J1(z_m)/z_m ist klein — genau das fehlt der Einmoden-
        #    Rechnung als Hochtonabfall. Ohne diese Probe wäre
        #    ``modal_source`` eine Behauptung.
        c43e = MicrophoneCapsule(**dict(g43, membrane_modes=1,
                                        modal_source=0))
        gew43 = {}
        for ang43 in (0.0, 90.0, 180.0):
            e43, _ = _grin_rms(c43e, ang43)
            gew43[ang43] = e43 - rms43[ang43]
        assert gew43[90.0] > 1.0 and gew43[180.0] > 1.0, \
            (f"der modenweise Antrieb muss off-axis mindestens 1 dB RMS "
             f"bringen (90°: {gew43[90.0]:+.2f}, 180°: {gew43[180.0]:+.2f})")
        assert gew43[0.0] > -0.2, \
            (f"und darf die Achse nicht verschlechtern "
             f"({gew43[0.0]:+.2f} dB RMS)")
        print(f"Modenfaktoren aus dem BEM: ka->0 alle 1 ({r43a:.0e}), ein "
              f"Lösungsgang für {F43b.shape[0]} Moden; gegen die Kalotte "
              f"unterscheiden sie sich bei 14 kHz um {d43:.2f}; "
              f"Modensenke bei {f2_43 / 1e3:.1f} kHz beseitigt "
              f"({kerbe['ok']:+.2f} statt {kerbe['roh']:+.2f} dB); Grinnip "
              f"VC 0/90/180° = {rms43[0.0]:.2f}/{rms43[90.0]:.2f}/"
              f"{rms43[180.0]:.2f} dB RMS (Messung bei 90°: {r43e:.2f}), "
              f"der modenweise Antrieb bringt "
              f"{gew43[0.0]:+.2f}/{gew43[90.0]:+.2f}/{gew43[180.0]:+.2f} dB"
              f"  OK")


@pytest.mark.bem
def test_gp44_ruckpatch_des_gradientenempfangers(k67_bem):
    """Gegenprobe 44: Rückpatch des Gradientenempfängers."""
    # Bis hierher lieferte der BEM bei einer Ein-Membran-Kapsel NUR den
    # Frontfaktor: der rückwärtige Einlass hatte keinen Patch auf der
    # Kontur, deshalb war der Gradientenfall gesperrt statt still falsch.
    # Jetzt sitzt der Bohrungskranz dort, wo er wirklich sitzt — bei
    # seiner axialen Einbautiefe d_rear_ax, radial im Mantel oder (bei
    # cavity_hole_position='end') in der hinteren Stirnfläche.
    #
    # WARUM EIN RING und kein Flächenmittel: die m=0-Formulierung löst
    # bereits den azimutal gemittelten Oberflächendruck. Genau das greift
    # ein gleichmäßig verteilter Lochkranz ab — und es ist das direkte
    # Gegenstück zum Ring der Kugelrechnung (_ring_cos), nur eben auf der
    # realen Kontur statt auf einer Ersatzkugel.
    #
    # a) ABSOLUTPROBE gegen Morse. Gegenprobe 21 prüft auf der Kugel-
    #    kontur das VERHÄLTNIS Pol/Pol. Hier wird der Ringdruck selbst
    #    geprüft, absolut und an fünf Ringwinkeln von 30° bis 180° —
    #    denn der Rückpatch ist genau der Ring, und nur so ist er
    #    verankert und nicht bloß plausibel.
    # b) UNABHÄNGIG VON DER ELEMENTTEILUNG: am Mantel wird linear in z
    #    interpoliert; ein dreifach feineres Gitter darf G kaum ändern.
    # c) GEOMETRIE: der Ring sitzt bei d_rear_ax; 'end' legt ihn in die
    #    hintere Stirnfläche; ein zu kurzer Kopf klemmt ihn dorthin und
    #    sagt es in der Warnung.
    # d) PHYSIK: |G| -> 1 für ka -> 0, und der effektive Außenweg ist
    #    länger als der geometrische (der Schall muss um die Frontkante)
    #    UND länger als bei der Ersatzkugel, deren Ring am Äquator sitzt.
    #    Der flache Kopf mit scharfer Kante zwingt den längeren Weg.
    # e) Das Gatter ist nur noch eine Warnung, der Fall rechnet, und die
    #    Kapsel bleibt eine Niere — mit MEHR Rückdämpfung als mit der
    #    Kugel, weil die externe Laufzeit länger ist.
    # f) DRUCKEMPFÄNGER UNVERÄNDERT: bei dichter Rückseite bleibt
    #    p_rear == p_front, der Rückpatch darf dort nichts tun.
    # g) ZWEI SYMMETRISCHE BACKPLATES. Dort ist der vordere Einlass die
    #    Außenseite der VORDEREN Backplate, nicht die Membran — der
    #    Abstand zum Rückeinlass ist d_ext, nicht d_rear_ax. Geprüft wird
    #    das für alle drei Wege gleichzeitig: ohne Beugung MUSS der Weg
    #    exakt d_ext sein, der BEM-Ring sitzt d_ext hinter der
    #    Stirnfläche, und der Kugelring misst dieselbe Tiefe ab der
    #    Kalotte. Der letzte Punkt war eine echte Inkonsistenz:
    #    ``_ring_cos`` rechnete mit d_rear_ax und legte den Ring bei zwei
    #    Backplates um Spalt + Plattendicke zu weit vorn. Bei EINER
    #    Backplate sind beide Größen identisch — dort ändert sich nichts,
    #    und auch das wird geprüft.
    if _HAS_SCIPY:
        # a) Kugelkontur: Ringdruck absolut gegen die Morse-Reihe
        def _morse_ring44(cap, R, psi, om, th):
            """Morse-Ringmittel bei Polarwinkel psi auf einer R-Kugel."""
            R0, rc0 = cap.R_body, cap._ring_cos
            cap.R_body, cap._ring_cos = R, float(np.cos(psi))
            try:
                return cap._diffraction_factors(om, th)[1]
            finally:
                cap.R_body, cap._ring_cos = R0, rc0

        R44 = 9e-3
        psi44 = np.linspace(0.0, np.pi, 121)
        el44 = MicrophoneCapsule._bem_elems(
            np.stack([R44 * np.sin(psi44), R44 * np.cos(psi44)], 1))
        psm44 = np.arctan2(el44["mid_r"], el44["mid_z"])
        th44 = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0]))
        fr44 = np.zeros(el44["L"].size, bool)
        fr44[0] = True
        worst44 = 0.0
        for i44 in (20, 40, 60, 90, 119):
            w44 = np.zeros(el44["L"].size)
            w44[i44] = 1.0
            k67_bem._bem_geo = dict(
                elems=el44, chief=[(0.0, 0.0)],
                w_area=2.0 * np.pi * el44["mid_r"] * el44["L"],
                front=fr44, w_rear=w44)
            k67_bem._bem_cache = None
            for f44 in (100.0, 1000.0, 5000.0):
                om44 = np.array([2.0 * np.pi * f44])
                F44, G44 = k67_bem._bem_front_modes(om44, th44)
                for got, psi in ((F44[0], psm44[0]),
                                 (F44[0] * G44, psm44[i44])):
                    ref = _morse_ring44(k67_bem, R44, psi, om44, th44)
                    worst44 = max(worst44, float(np.max(
                        np.abs(got - ref) / np.abs(ref))))
        assert worst44 < 2e-4, \
            (f"BEM-Ringdruck muss die Morse-Reihe absolut treffen "
             f"({worst44:.1e})")
        k67_bem._bem_geo = None
        k67_bem._bem_cache = None

        # Ein Gradientenempfänger als Prüfling (die Beispielkapsel aus
        # dem Kopf dieses Testlaufs, Lochkranz am Umfang bei 6 mm).
        g44 = dict(
            membrane_material="PET", membrane_resonance_hz=8000.0,
            membrane_diameter=22e-3, membrane_thickness=6e-6,
            membrane_tension=400.0, air_gap=40e-6,
            backplate_diameter=20e-3, backplate_thickness=3e-3,
            bias_voltage=60.0, architecture="single", n_through_holes=60,
            through_hole_diameter=1.0e-3, n_blind_holes=30,
            blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
            delay_length=3e-3, cavity_length=12e-3,
            cavity_wall_thickness=1.5e-3, n_cavity_holes=200,
            cavity_hole_diameter=0.2e-3, cavity_hole_axial_position=6e-3,
            fabric_front_rayl=10.0, fabric_rear_rayl=25.0,
            body_diameter=24e-3)
        b44 = dict(axial_body_model="bem", bem_body_diameter=0.0,
                   body_length=20e-3)

        def _bau44(**kw):
            """Baut den Prüfling und liefert (Kapsel, Warnungstexte)."""
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                cc = MicrophoneCapsule(**{**g44, **kw})
            return cc, [str(r.message) for r in rec
                        if issubclass(r.category, UserWarning)]

        c44, warn44 = _bau44(cavity_hole_position="circumference", **b44)
        assert len(warn44) == 1 and "offener Rückseite" in warn44[0], \
            f"das Gatter muss WARNEN statt zu sperren ({warn44})"
        assert "ACHTUNG" not in warn44[0], \
            f"stimmige Geometrie darf keine Zusatzwarnung geben ({warn44[0]})"

        # b) Elementteilung: dreifach feiner darf G kaum ändern
        class _Fein44(MicrophoneCapsule):
            _BEM_H_MAX = MicrophoneCapsule._BEM_H_MAX / 3.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f44c = _Fein44(**{**g44, "cavity_hole_position": "circumference",
                              **b44})
        om44b = 2.0 * np.pi * np.array([50.0, 1000.0, 5000.0, 10000.0])
        G44g = c44._bem_front_modes(om44b, th44)[1]
        G44f = f44c._bem_front_modes(om44b, th44)[1]
        d44g = float(np.max(np.abs(G44g - G44f) / np.abs(G44f)))
        n44g = c44._bem_geometry()["elems"]["L"].size
        n44f = f44c._bem_geometry()["elems"]["L"].size
        assert n44f > 2.0 * n44g, "feineres Gitter muss feiner sein"
        assert d44g < 0.02, \
            (f"der Ring darf nicht an der Elementteilung hängen "
             f"({n44g} -> {n44f} Elemente ändern G um {d44g:.1e})")

        # c) Geometrie des Patches
        def _patch44(cc):
            gg = cc._bem_geometry()
            ww, ee = gg["w_rear"], gg["elems"]
            nz = ww != 0
            return (float(ww[nz] @ ee["mid_z"][nz] / np.sum(ww[nz])),
                    ee["mid_r"][nz], ee["mid_z"][nz],
                    0.5 * cc._bem_head_len)
        z44, r44m, _, zf44 = _patch44(c44)
        assert abs(z44 - (zf44 - c44.d_rear_ax)) < 1e-9, \
            (f"der Ring muss bei d_rear_ax sitzen ({z44 * 1e3:.3f} statt "
             f"{(zf44 - c44.d_rear_ax) * 1e3:.3f} mm)")
        assert np.all(np.abs(r44m - c44.R_body) < 1e-9), \
            "bei 'circumference' muss der Ring im Mantel liegen"
        c44e, _ = _bau44(cavity_hole_position="end",
                         **dict(b44, body_length=19.5e-3))
        z44e, r44e, mz44e, zf44e = _patch44(c44e)
        assert np.all(np.abs(mz44e + zf44e) < 1e-9), \
            "bei 'end' muss der Ring in der hinteren Stirnfläche liegen"
        assert r44e.max() <= c44e.a_bp + 1e-9, \
            "die Endlöcher münden innerhalb des Hohlraumradius"
        c44k, warn44k = _bau44(cavity_hole_position="circumference",
                               **dict(b44, body_length=8e-3))
        _, r44k, mz44k, zf44k = _patch44(c44k)
        assert np.all(np.abs(mz44k + zf44k) < 1e-9), \
            "ein zu kurzer Kopf muss den Ring auf die Stirnfläche klemmen"
        assert "ACHTUNG" in warn44k[0], \
            f"und das muss in der Warnung stehen ({warn44k})"

        # d) ka -> 0: |G| = 1, und der Außenweg ist länger als geometrisch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k44 = MicrophoneCapsule(**dict(
                g44, cavity_hole_position="circumference"))   # Ersatzkugel
        om44a = np.array([2.0 * np.pi * 50.0])
        th44a = np.array([np.pi])
        kk44 = om44a[0] / C_AIR
        G44b = c44._bem_front_modes(om44a, th44a)[1][0, 0]
        Ff44, Fr44 = k44._diffraction_factors(om44a, th44a)
        G44k = (Fr44 / Ff44)[0, 0]
        assert abs(abs(G44b) - 1.0) < 0.02, \
            f"|G| muss für ka -> 0 gegen 1 gehen ({abs(G44b):.4f})"
        d44b = float(np.angle(G44b)) / kk44
        d44k = float(np.angle(G44k)) / kk44
        assert c44.d_rear_ax < d44k < d44b < 2.5 * c44.d_rear_ax, \
            (f"Außenweg: geometrisch {c44.d_rear_ax * 1e3:.1f} < Kugel "
             f"{d44k * 1e3:.1f} < BEM {d44b * 1e3:.1f} mm erwartet")

        # e) der Fall rechnet, und die Niere wird durch den längeren
        #    Außenweg hinten dichter als mit der Ersatzkugel
        f44p = np.array([125.0, 1000.0])
        fb44 = {}
        for lbl44, cc44 in (("bem", c44), ("kugel", k44)):
            rr44 = cc44.angle_responses(f44p, angles_deg=(0.0, 180.0))
            fb44[lbl44] = (20 * np.log10(np.abs(rr44["H"][0.0]))
                           - 20 * np.log10(np.abs(rr44["H"][180.0])))
        assert np.all(fb44["bem"] > 3.0), \
            f"die Kapsel muss gerichtet bleiben ({np.round(fb44['bem'], 1)})"
        assert np.all(fb44["bem"] > fb44["kugel"] + 1.0), \
            (f"der längere Außenweg muss hinten mehr dämpfen "
             f"({np.round(fb44['bem'], 1)} gegen "
             f"{np.round(fb44['kugel'], 1)} dB)")

        # f) Druckempfänger unverändert: dichte Rückseite -> p_r == p_f
        c44d = MicrophoneCapsule(**dict(
            g44, n_cavity_holes=0, cavity_length=0.0, delay_length=0.0,
            **b44))
        assert not c44d.rear_open
        pf44, pr44 = c44d._source_pressures(om44b, th44)
        assert np.array_equal(pf44, pr44), \
            "bei dichter Rückseite darf der Rückpatch nichts ändern"

        # g) ZWEI SYMMETRISCHE BACKPLATES. Dort liegt der vordere
        #    Schalleinlass nicht auf der Membran, sondern auf der
        #    Außenseite der VORDEREN Backplate — die Kette führt sie als
        #    eigenes Zweitor vor der Membran (_assemble_parts). Der
        #    Abstand zum Rückeinlass ist damit d_ext und nicht d_rear_ax,
        #    und zwar in JEDEM Körpermodell: die beugungsfreie Rechnung
        #    setzt exp(-j·k·d_ext·cosθ) an, der BEM-Ring sitzt d_ext
        #    hinter der Stirnfläche, und die Kugel misst ihren
        #    Ringpolarwinkel ab der Kalotte am vorderen Pol. Genau das
        #    war inkonsistent: _ring_cos rechnete mit d_rear_ax, also bei
        #    zwei Backplates um Spalt + Plattendicke zu weit vorn. Bei
        #    EINER Backplate sind beide Größen identisch, dort ändert
        #    sich dadurch nichts.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d44 = {a44: (
                MicrophoneCapsule(**dict(g44, architecture=a44,
                                         cavity_hole_position="circumference",
                                         **dict(b44, body_length=24e-3))),
                MicrophoneCapsule(**dict(g44, architecture=a44)),
                MicrophoneCapsule(**dict(g44, architecture=a44,
                                         include_diffraction=False)))
                for a44 in ("single", "dual")}
        s44, du44 = d44["single"][0], d44["dual"][0]
        assert s44.d_ext == s44.d_rear_ax, \
            ("bei EINER Backplate ist die Membran der vordere Einlass — "
             "d_ext und d_rear_ax müssen gleich bleiben")
        assert abs((du44.d_ext - du44.d_rear_ax)
                   - (du44.h_gap + du44.t_bp)) < 1e-12, \
            (f"bei ZWEI Backplates muss d_ext um Spalt + Plattendicke "
             f"größer sein ({(du44.d_ext - du44.d_rear_ax) * 1e3:.3f} statt "
             f"{(du44.h_gap + du44.t_bp) * 1e3:.3f} mm)")
        weg44 = {}
        for a44, (cb44, ck44, cn44) in d44.items():
            # der Rückpatch des BEM sitzt bei d_ext
            gb44 = cb44._bem_geometry()
            nz44 = gb44["w_rear"] != 0
            z_p44 = float(gb44["w_rear"][nz44]
                          @ gb44["elems"]["mid_z"][nz44]
                          / np.sum(gb44["w_rear"][nz44]))
            assert abs(z_p44 - (0.5 * cb44._bem_head_len - cb44.d_ext)) < 1e-9, \
                f"{a44}: der BEM-Ring muss d_ext hinter der Stirnfläche sitzen"
            # die Kugel misst dieselbe Tiefe
            assert abs(ck44._ring_cos
                       - (ck44.R_body - ck44.d_ext) / ck44.R_body) < 1e-12, \
                f"{a44}: der Kugelring muss d_ext hinter dem Pol sitzen"
            # und ohne Beugung ist der Weg exakt d_ext
            pf, pr = cn44._source_pressures(om44a, th44a)
            w44 = {"ohne": float(np.angle((pr / pf)[0, 0])) / kk44,
                   "kugel": float(np.angle(
                       (lambda F: F[1] / F[0])(
                           ck44._diffraction_factors(om44a, th44a))[0, 0]))
                   / kk44,
                   "bem": float(np.angle(
                       cb44._bem_front_modes(om44a, th44a)[1][0, 0])) / kk44}
            assert abs(w44["ohne"] - cn44.d_ext) < 1e-9, \
                (f"{a44}: ohne Beugung MUSS der Weg exakt d_ext sein "
                 f"({w44['ohne'] * 1e3:.3f} statt {cn44.d_ext * 1e3:.3f} mm)")
            assert cb44.d_ext < w44["kugel"] < w44["bem"], \
                (f"{a44}: geometrisch < Kugel < BEM erwartet "
                 f"({cb44.d_ext * 1e3:.1f}/{w44['kugel'] * 1e3:.1f}/"
                 f"{w44['bem'] * 1e3:.1f} mm)")
            weg44[a44] = w44
        for mod44 in ("ohne", "kugel", "bem"):
            assert (weg44["dual"][mod44]
                    > weg44["single"][mod44] + 0.5 * (du44.h_gap + du44.t_bp)), \
                (f"die vordere Backplate MUSS den Außenweg verlängern "
                 f"({mod44}: {weg44['single'][mod44] * 1e3:.1f} -> "
                 f"{weg44['dual'][mod44] * 1e3:.1f} mm)")
        fb44d = {}
        for lbl44, cc44 in (("bem", d44["dual"][0]), ("kugel", d44["dual"][1])):
            rr44 = cc44.angle_responses(f44p, angles_deg=(0.0, 180.0))
            fb44d[lbl44] = (20 * np.log10(np.abs(rr44["H"][0.0]))
                            - 20 * np.log10(np.abs(rr44["H"][180.0])))
        assert np.all(fb44d["bem"] > fb44d["kugel"]), \
            (f"auch bei zwei Backplates muss der längere Außenweg des BEM "
             f"hinten mehr dämpfen ({np.round(fb44d['bem'], 1)} gegen "
             f"{np.round(fb44d['kugel'], 1)} dB)")
        print(f"Rückpatch (Gradientenempfänger): Ringdruck trifft die "
              f"Morse-Reihe absolut an fünf Ringwinkeln ({worst44:.0e}); "
              f"gitterunabhängig ({n44g}->{n44f} Elemente: {d44g:.1e}); "
              f"Ring sitzt d_ext hinter dem vorderen Einlass, 'end' in der "
              f"Stirnfläche, zu kurzer Kopf geklemmt (mit Warnung); "
              f"Außenweg 1 Backplate {c44.d_ext * 1e3:.1f} < Kugel "
              f"{d44k * 1e3:.1f} < BEM {d44b * 1e3:.1f} mm (F/B "
              f"{fb44['bem'][0]:.1f} statt {fb44['kugel'][0]:.1f} dB bei "
              f"125 Hz), 2 Backplates {du44.d_ext * 1e3:.1f} < "
              f"{weg44['dual']['kugel'] * 1e3:.1f} < "
              f"{weg44['dual']['bem'] * 1e3:.1f} mm (F/B "
              f"{fb44d['bem'][0]:.1f} statt {fb44d['kugel'][0]:.1f} dB); "
              f"Druckempfänger unverändert  OK")
