"""Gegenproben: Membran und Elektrostatik."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


def test_gp05_elektrostatik_pull_in():
    """Gegenprobe 5: Elektrostatik / Pull-in."""
    # Eine sehr weiche Membran (f_res = 800 Hz -> T ~ 4 N/m bei 22 mm/6 µm)
    # kann 60 V bei 40 µm Spalt statisch nicht tragen (das Luftpolster
    # entweicht durch die Löcher) -> Pull-in. Bei reduzierter Spannung
    # existiert ein stabiler Arbeitspunkt mit statischer Durchbiegung.
    try:
        MicrophoneCapsule(membrane_resonance_hz=800.0)
        raise AssertionError("800 Hz / 40 µm / 60 V müsste kollabieren")
    except ValueError as exc:
        assert "Pull-in" in str(exc)
    soft = MicrophoneCapsule(membrane_resonance_hz=800.0, bias_voltage=20.0)
    # Konsistenz: kollabiert bei 60 V, stabil bei 20 V
    assert 20.0 < soft.U_pullin < 60.0, "U_PI muss zwischen 20 und 60 V liegen"
    assert 0.0 < soft.w0_static < soft.h_gap
    fr_soft = soft.frequency_response(n_points=50)
    assert np.all(np.isfinite(fr_soft["amplitude_db"]))
    print(f"Pull-in-Gegenprobe: 800 Hz kollabiert bei 60 V (U_PI = "
          f"{soft.U_pullin:.1f} V), stabil bei 20 V mit statischer "
          f"Durchbiegung {soft.w0_static * 1e6:.1f} µm  OK")


def test_gp07_elektrostatik_der_doppelmembran(k67, stand):
    """Gegenprobe 7: Elektrostatik der Doppelmembran."""
    # Im Nierenmodus ist NUR die Frontmembran polarisiert: die passive
    # Rückmembran benutzt die unpolarisierte Nachgiebigkeit (keine Feder-
    # Erweichung), und Wandlung/Ruhekapazität rechnen einseitig (n_bp = 1,
    # anders als die echte Gegentakt-Dual-Backplate mit n_bp = 2).
    assert k67.n_bp == 1, "K67-Bauform: nur die Frontseite ist polarisiert"
    Z_front = k67._membrane_impedance(np.array([2 * np.pi * 1000.0]))[0]
    Z_pass = k67._membrane_impedance_passive(np.array([2 * np.pi * 1000.0]))[0]
    assert abs(Z_front - Z_pass) > 0, "Rückmembran muss unpolarisiert sein"
    # Die Polarisation wirkt NUR in realistischem Maß auf das Richtdiagramm:
    # über den statisch verkleinerten Frontspalt (h³-Filmwiderstand), nicht
    # über die Feder-Erweichung (die kürzt sich als Serienelement aus dem
    # normierten Muster). Erwartung: wenige dB Verschiebung bei 180°, kein
    # Umkippen des Patterns.
    def _p180(bias):
        c = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=bias, architecture="dual_diaphragm", center_gap=50e-6,
            n_through_holes=60, through_hole_diameter=0.6e-3,
            n_blind_holes=120, blind_hole_diameter=1.3e-3,
            blind_hole_depth=3.7e-3, through_holes_stepped=True,
            clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=34e-3, squeeze_model="2d")
        return c.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
    d20, d60 = _p180(20.0), _p180(60.0)
    stand.wert("niere_20v", d20, "dB", "K67 180° bei 1 kHz, 20 V")
    stand.wert("niere_60v", d60, "dB", "K67 180° bei 1 kHz, 60 V")
    print(f"Elektrostatik Doppelmembran: nur Front polarisiert (n_bp=1); "
          f"Bias-Wirkung aufs Pattern realistisch begrenzt "
          f"(180° @1 kHz: {d20:.1f} dB @20 V -> {d60:.1f} dB @60 V)  OK")


def test_gp28_spaltmundung_mehrmoden_membran():
    """Gegenprobe 28: Spaltmündung + Mehrmoden-Membran."""
    # a) KEINE DOPPELZÄHLUNG DER LATERALEN SPALTMASSE. Der Škvor-Term
    #    trägt die Massenwirkung der radial zu den Löchern gequetschten
    #    Spaltluft bereits vollständig — nachgewiesen als IDENTITÄT mit
    #    der Baird/Zuckerwar-Spaltmasse:
    #        M_gap = rho0*B(q)/(n*pi*h)  ==  R_Škvor * rho0*h²/(12 mu),
    #    und die Frequenzkorrektur Φ(ω) realisiert diese Masse auch
    #    wirklich: im Tiefton ist Im(R·Φ)/ω = (6/5)·M_gap — der Faktor
    #    6/5 ist der kinetische Profilfaktor der Poiseuille-Verteilung,
    #    den die reine Lumped-Masse nicht kennt. Eine ZUSÄTZLICHE
    #    Freifeld-Flanschmasse 0.85·r auf der SPALTSEITE wäre daher
    #    Doppelzählung; sie entfällt jetzt auch im 1D-Pfad (2D- und
    #    3D-Pfad führen sie ohnehin nicht).
    # b) MEHRMODEN-MEMBRAN (Galerkin, membrane_modes > 1). Die höheren
    #    axialsymmetrischen (0,m)-Moden liegen PARALLEL zur Grundmode.
    #    Verankert an: membrane_modes = 1 bit-für-bit wie bisher;
    #    Modenfrequenzen und -massen exakt im Bessel-Verhältnis;
    #    massegesteuerter Hochton-Grenzwert der Parallelschaltung;
    #    Passivität; Konvergenz; Gatter.
    cap28 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm",
        center_gap=40e-6, n_through_holes=60,
        through_hole_diameter=0.6e-3, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, squeeze_model="2d")
    # a) Identität Škvor <-> Baird/Zuckerwar
    h28 = cap28.h_gap
    n_dr28 = cap28.n_th + cap28.n_bh
    q28 = cap28._q_drain
    B_q28 = (0.5 * np.log(1.0 / np.sqrt(q28)) - 3.0 / 8.0
             + q28 / 2.0 - q28**2 / 8.0)
    M_baird28 = RHO0 * B_q28 / (n_dr28 * np.pi * h28)
    M_skvor28 = cap28._skvor_R(h28) * RHO0 * h28**2 / (12.0 * MU_AIR)
    assert abs(M_skvor28 / M_baird28 - 1.0) < 1e-12, \
        (f"Škvor muss die Baird/Zuckerwar-Spaltmasse exakt enthalten "
         f"({M_skvor28:.4f} vs. {M_baird28:.4f} kg/m^4)")
    om28 = np.array([2.0 * np.pi * 100.0])
    Z_film28 = cap28._skvor_R(h28) * cap28._film_R_dynamic(om28, h28)[0]
    assert abs((Z_film28.imag / om28[0]) / M_baird28 - 6.0 / 5.0) < 2e-3, \
        (f"Φ(ω) muss im Tiefton (6/5)·M_gap realisieren "
         f"({(Z_film28.imag / om28[0]) / M_baird28:.4f})")
    # Spaltseitige Flanschmasse ist raus: die ungestufte Durchgangsloch-
    # Impedanz trägt NUR noch den portseitigen Fok-Term.
    cap28p = MicrophoneCapsule(
        architecture="single", membrane_resonance_hz=2100.0,
        membrane_diameter=25.4e-3, membrane_thickness=6e-6,
        membrane_tension=45.0, air_gap=38.1e-6,
        backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
        bias_voltage=50.0, n_through_holes=48,
        through_hole_diameter=1.0e-3, n_blind_holes=24,
        blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, squeeze_model="1d")
    om28b = 2.0 * np.pi * np.array([1000.0, 9000.0])
    S28 = np.pi * cap28p.r_th**2
    Z_ref28 = (cap28p._hole_impedance(om28b, cap28p.r_th, cap28p.t_bp,
                                      cap28p.n_th, end_correction=False,
                                      visc_ends=1)
               + 1j * om28b * RHO0 * (0.85 * cap28p.r_th * cap28p._fok_th)
               / (S28 * cap28p.n_th))
    Z_got28 = cap28p._through_hole_impedance(om28b, cap28p.n_th)
    assert np.max(np.abs(Z_got28 - Z_ref28) / np.abs(Z_ref28)) < 1e-12, \
        "Durchgangsloch darf nur noch die PORTSEITIGE Mündungsmasse tragen"
    # b) Mehrmoden-Membran
    mk28 = dict(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm",
        center_gap=40e-6, n_through_holes=60,
        through_hole_diameter=0.6e-3, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, squeeze_model="2d")
    c28_1 = MicrophoneCapsule(membrane_modes=1, **mk28)
    assert c28_1._higher_mode_branches() == [], \
        "membrane_modes = 1 darf keine Zusatzzweige erzeugen"
    om28c = 2.0 * np.pi * np.logspace(2, 4.3, 40)
    Z1_28 = c28_1._membrane_impedance(om28c)
    R28 = c28_1._membrane_film_damping(om28c, c28_1.h_gap_front,
                                       c28_1.R_A_gap_front)
    Z1_ref = (R28 + 1j * om28c * c28_1.M_A_mem
              + 1.0 / (1j * om28c * c28_1.C_A_eff))
    assert np.max(np.abs(Z1_28 - Z1_ref)) == 0.0, \
        "membrane_modes = 1 muss bit-für-bit die Einmoden-Impedanz sein"
    c28_3 = MicrophoneCapsule(membrane_modes=3, **mk28)
    br28 = c28_3._higher_mode_branches()
    x28 = MicrophoneCapsule._J0_ZEROS
    f1_28 = 1.0 / (2.0 * np.pi * np.sqrt(c28_3.M_A_mem * c28_3.C_A_mem))
    for i28, (M_m, C_m) in enumerate(br28):
        rat28 = x28[i28 + 1] / x28[0]
        assert abs(M_m / c28_3.M_A_mem - rat28**2) < 1e-12, \
            f"Modenmasse muss (x_m/x_1)² folgen (Mode {i28 + 2})"
        f_m = 1.0 / (2.0 * np.pi * np.sqrt(M_m * C_m))
        assert abs(f_m / f1_28 - rat28) < 1e-9, \
            (f"Modenfrequenz muss x_m/x_1 folgen: {f_m:.1f} statt "
             f"{f1_28 * rat28:.1f} Hz")
    # Massegesteuerter Grenzwert der Parallelschaltung: die wirksame Masse
    # sinkt auf s_N/Σ(1/M_m) — der Faktor s_N ist die nachgiebigkeits-
    # erhaltende Normierung der Modenaufteilung (s. _modal_split_factor
    # und Gegenprobe 42). Ohne Dämpfung (R = 0) isoliert das die
    # Modenalgebra von der Filmdämpfung.
    # PHYSIKALISCH schärfer als die reine Algebra: die exakten Modenmassen
    # erfüllen Σ 1/M_m = S/σ, eine vielmodige Membran verhält sich weit
    # oberhalb aller Resonanzen also wie ein freier KOLBEN. Mit der
    # Normierung nähert sich unsere wirksame Masse diesem Grenzwert
    # MONOTON VON OBEN (1 Mode 8/j01² = 1.383, 3 Moden 1.138, 5 Moden
    # 1.083 mal σ/S) — ohne sie lief die Reihe darüber hinaus und wurde
    # mit mehr Termen schlechter. Mit dem Massenfaktor 8/j01² ist der
    # Grenzwert für N -> ∞ EXAKT σ/S (Gegenprobe 55); mit dem alten 4/3
    # lief er auf 0.964·σ/S, also unter die Kolbenmasse.
    s_N28 = c28_3._modal_split_factor()
    M_eff28 = s_N28 / sum(1.0 / M for M in
                          [c28_3.M_A_mem] + [b[0] for b in br28])
    om_hf28 = np.array([2.0 * np.pi * 2.0e5])
    Z1_hf28 = (1j * om_hf28 * c28_3.M_A_mem
               + 1.0 / (1j * om_hf28 * c28_3.C_A_eff))
    Z_hf28 = c28_3._modal_parallel(om_hf28, Z1_hf28, 0.0)[0]
    assert abs(Z_hf28.imag / om_hf28[0] / M_eff28 - 1.0) < 1e-3, \
        (f"Massegrenzwert muss s_N/Σ(1/M_m) treffen "
         f"({Z_hf28.imag / om_hf28[0]:.3f} vs. {M_eff28:.3f})")
    # Kolben der WIRKSAMEN Membranfläche S·u_w, u_w = ((a − √(D/T))/a)²:
    # bewegt wird die Membran innerhalb der Randschicht (Gegenprobe 62)
    M_pist28 = c28_3.mat_rho * c28_3.t_mem / (c28_3.S_mem * c28_3._u_w)
    rel28 = []
    for n28 in (1, 3, 5):
        cc28 = MicrophoneCapsule(membrane_modes=n28, **mk28)
        rel28.append(
            (cc28._modal_split_factor()
             / sum(1.0 / M for M in [cc28.M_A_mem]
                   + [b[0] for b in cc28._higher_mode_branches()]))
            / M_pist28)
    assert abs(rel28[0] - 8.0 / x28[0] ** 2) < 1e-9, \
        f"eine Mode muss die Masse 8/j01²·σ/S haben ({rel28[0]:.4f})"
    assert 1.0 < rel28[2] < rel28[1] < rel28[0], \
        (f"die wirksame Masse muss monoton von oben gegen die Kolbenmasse "
         f"σ/S laufen ({np.round(rel28, 4)})")
    assert np.all(np.real(c28_3._membrane_impedance(om28c)) > 0.0), \
        "Passivität: Re{Z_mem} > 0 über das Band"
    # Konvergenz + Wirkung im Hochton (Grundmode-Anker unberührt)
    F28 = [1000.0, 7000.0, 16000.0]
    lev28 = {}
    for m28 in (1, 3, 5):
        H28 = MicrophoneCapsule(membrane_modes=m28,
                                **mk28).transfer_function(F28)
        lev28[m28] = 20.0 * np.log10(np.abs(H28) / np.abs(H28[0]))
        if m28 > 1:
            assert abs(abs(H28[0]) / abs(
                MicrophoneCapsule(membrane_modes=1,
                                  **mk28).transfer_function(F28)[0])
                - 1.0) < 1e-3, \
                "Empfindlichkeit bei 1 kHz muss unberührt bleiben"
    d13 = abs(lev28[3][2] - lev28[1][2])
    d35 = abs(lev28[5][2] - lev28[3][2])
    assert d35 < d13, \
        f"Modenreihe muss konvergieren ({d35:.2f} vs. {d13:.2f} dB)"
    # Die höheren Moden machen die Membran im massegesteuerten Bereich
    # leichter (Grenzwert oben) und würden den Hochton entsprechend
    # anheben — aber im Spaltfilm tragen sie ihre eigene innere
    # Umverteilung (_modal_internal_Z), und die frisst die Anhebung fast
    # vollständig auf. Verankert wird das STRUKTURELL an der Differenz:
    # UNBELASTETE Zweige heben klar an, mit Filmlast bleibt ein Bruchteil.
    # (Seit der nachgiebigkeitserhaltenden Normierung, Gegenprobe 42, ist
    # der belastete Rest bei dieser Kapsel sogar leicht NEGATIV: bei
    # 16 kHz ist sie noch nicht massegesteuert, dort überwiegt der um
    # 1/s_N steifere Zweig. Ein Vorzeichen ist hier deshalb nichts, worauf
    # man sich festlegen sollte — der Betrag ist die Aussage.)

    class _UngedaempfteModen(MicrophoneCapsule):
        """Modenzweige OHNE innere Umverteilung — nur zum Vergleich."""

        def _modal_internal_Z(self, omega, h_film):
            return [0.0] * max(self.membrane_modes - 1, 0)

    H28u = _UngedaempfteModen(membrane_modes=3,
                              **mk28).transfer_function(F28)
    lev28u = 20.0 * np.log10(np.abs(H28u) / np.abs(H28u[0]))
    lift28 = lev28[3][2] - lev28[1][2]
    lift28u = lev28u[2] - lev28[1][2]
    assert lift28u > 1.0, \
        (f"unbelastete Modenzweige müssen den Hochton klar anheben "
         f"({lift28u:.2f} dB)")
    assert abs(lift28) < 0.25 * lift28u, \
        (f"die Filmlast der Moden muss die Anhebung deutlich dämpfen "
         f"({lift28u:+.2f} dB unbelastet gegen {lift28:+.2f} dB mit Last)")
    # Der 7-kHz-Sattel ist NICHT modal: er bleibt praktisch unverändert
    assert abs(lev28[3][1] - lev28[1][1]) < 0.5, \
        (f"7-kHz-Sattel ist kein Modeneffekt "
         f"({lev28[1][1]:.1f} -> {lev28[3][1]:.1f} dB)")
    try:
        MicrophoneCapsule(membrane_modes=0, **mk28)
        raise AssertionError("membrane_modes = 0 muss scheitern")
    except ValueError:
        pass
    print(f"Spaltmündung/Moden: Škvor == Baird-Spaltmasse "
          f"({M_skvor28:.2f} kg/m^4, Φ realisiert 6/5 davon) -> keine "
          f"Doppelzählung, Freifeld-Flansch spaltseitig raus; Moden "
          f"{f1_28:.0f}/{f1_28 * x28[1] / x28[0]:.0f}/"
          f"{f1_28 * x28[2] / x28[0]:.0f} Hz, modes=1 bit-identisch, "
          f"16 kHz {lev28[1][2]:+.1f} -> {lev28[3][2]:+.1f} dB "
          f"(konvergent; ohne Filmlast wären es {lev28u[2]:+.1f} dB), "
          f"7 kHz unverändert  OK")


def test_gp33_modengewicht_der_frontmittelung():
    """Gegenprobe 33: Modengewicht der Frontmittelung."""
    # Der Antrieb einer Membranmode ist die Galerkin-Projektion
    # <p_f · psi> / <psi>, nicht der flächengleiche Mittelwert (Šimonová/
    # Honzík, JASA 159, 4512 (2026), Gl. 5 + A2; Lavergne et al.). Bis
    # hierher mittelte die Kalotte flächengleich — das erzeugt bei
    # u = k·a·sin(theta) = 3.83 eine Auslöschung, die die Grundmode gar
    # nicht hat (ihre erste Nullstelle liegt bei u = 5.52).
    #
    # Verankert an drei Aussagen:
    # a) FREIFELD-GRENZFALL, exakt und ohne freien Parameter: für eine
    #    kleine Kalotte auf großem Körper (quasi flache Membran) muss die
    #    Quadratur die geschlossene Form
    #        D(u) = z01²·J0(u)/(z01² − u²)
    #    liefern — hergeleitet aus <J0(k r sin θ) · J0(z01 r/a)> mit dem
    #    Bessel-Produktintegral (Gl. A2 der Arbeit).
    # b) GRENZFALL GEWICHT ≡ 1: mit konstantem Gewicht muss dieselbe
    #    Quadratur das ALTE Flächenmittel A(u) = 2·J1(u)/u reproduzieren
    #    — der Umbau ist also eine echte Verallgemeinerung, kein Bruch.
    # c) NORMIERUNG: Σw = 1, und für ka → 0 bleibt |F| = 1 (kein Gewinn
    #    aus dem Nichts), geprüft an einer realen Kapsel.
    if _HAS_SCIPY:
        from scipy.special import j0 as _j0_33, j1 as _j1_33
        z01_33 = MicrophoneCapsule._J0_ZEROS[0]
        c33 = MicrophoneCapsule(membrane_diameter=25.4e-3,
                                body_diameter=25.4e-3 * 40.0)
        u_33, w_33 = c33._cap_mode_quad()
        assert abs(float(np.sum(w_33)) - 1.0) < 1e-12, \
            f"Modengewichte müssen auf 1 normiert sein ({np.sum(w_33)})"
        r_33 = c33.R_body * np.sqrt(np.clip(1.0 - u_33**2, 0.0, None))
        worst_D33 = 0.0
        # u = k·a mit dem Radius der WIRKSAMEN Membran: das Modengewicht
        # ist J0(z01·r/(a − √(D/T))) (Randschicht, Gegenprobe 62)
        for u33 in (0.5, 1.0, 2.0, 3.0, 3.8317, 4.5, 5.0):
            got = float(np.dot(w_33, _j0_33((u33 / c33._a_w) * r_33)))
            ref = (0.5 * z01_33 * _j1_33(z01_33) if abs(u33 - z01_33) < 1e-9
                   else z01_33**2 * _j0_33(u33) / (z01_33**2 - u33**2))
            worst_D33 = max(worst_D33, abs(got - ref))
        assert worst_D33 < 1e-4, \
            (f"Modenprojektion muss D(u) = z01²J0(u)/(z01²−u²) treffen "
             f"({worst_D33:.1e})")
        # b) Gewicht ≡ 1 -> altes Flächenmittel A(u) = 2 J1(u)/u
        u0_33 = c33._cap_cos
        x33, wg33 = np.polynomial.legendre.leggauss(48)
        uu33 = 0.5 * (1.0 + u0_33) + 0.5 * (1.0 - u0_33) * x33
        wf33 = 0.5 * (1.0 - u0_33) * wg33
        wf33 = wf33 / np.sum(wf33)
        rr33 = c33.R_body * np.sqrt(np.clip(1.0 - uu33**2, 0.0, None))
        worst_A33 = 0.0
        for u33 in (1.0, 2.0, 3.0, 3.8317):
            got = float(np.dot(wf33, _j0_33((u33 / c33.a_mem) * rr33)))
            worst_A33 = max(worst_A33, abs(got - 2.0 * _j1_33(u33) / u33))
        assert worst_A33 < 1e-4, \
            (f"mit konstantem Gewicht muss das alte Flächenmittel "
             f"2·J1(u)/u herauskommen ({worst_A33:.1e})")
        # c) reale Kapsel: ka -> 0 gibt keinen Gewinn
        c33b = MicrophoneCapsule(architecture="single",
                                 membrane_diameter=25.4e-3,
                                 body_diameter=28e-3)
        F33 = c33b._diffraction_factors(np.array([2.0 * np.pi * 5.0]),
                                        np.array([0.0]))[0]
        assert abs(abs(complex(F33[0, 0])) - 1.0) < 5e-3, \
            f"ka->0 muss |F| = 1 liefern ({abs(complex(F33[0, 0])):.4f})"
        # Größe des Effekts, zur Einordnung im Protokoll
        d33 = 20.0 * np.log10(
            abs(z01_33**2 * _j0_33(3.0) / (z01_33**2 - 9.0))
            / abs(2.0 * _j1_33(3.0) / 3.0))
        print(f"Modengewicht der Frontmittelung: Projektion trifft D(u) "
              f"({worst_D33:.1e}), Gewicht≡1 reproduziert 2·J1(u)/u "
              f"({worst_A33:.1e}), Σw = 1, ka→0 neutral; Unterschied zum "
              f"Flächenmittel bei u = 3 rund {d33:+.1f} dB "
              f"(dessen Nullstelle bei u = 3.83 entfällt)  OK")


def test_gp34_modenabhangiger_quelldruck(stand):
    """Gegenprobe 34: modenabhängiger Quelldruck."""
    # Schalter ``modal_source`` (Voreinstellung 0 = aus). Bei 1 wird jede
    # Membranmode von ihrer EIGENEN Galerkin-Projektion getrieben statt von
    # einem gemeinsamen Skalar; die Moden liegen im Kettenmodell parallel
    # am selben Spaltknoten, deshalb ist die Zusammenfassung zu einer
    # Ersatzquelle p_eff = Σ Y_m p_m / Σ Y_m exakt (s. _modal_source_scale).
    #
    # MOTIVATION, gemessen: gegen die COMSOL-Referenz (Gegenprobe 32) bei
    # STREIFENDEM Einfall — so wird sie angeregt, mit dem Radialprofil
    # J0(k0 r) über die Membranfläche — fehlten dem Modell oberhalb 5 kHz
    # 14 dB. Der Nachweis, dass das die ANREGUNG ist und nicht die
    # Innenakustik: treibt man das (FEM-validierte) Modell der Autoren
    # ebenfalls uniform statt mit ebener Welle, weicht es genauso ab
    # (bei 10 kHz +17.5 dB dort gegen +17.3 dB hier), und die beiden
    # uniform getriebenen Modelle treffen sich auf 0.2 dB.
    #
    # Verankert:
    # a) AUS ist bit-für-bit der Bestand, und der Schalter hat ein Gatter.
    # b) FREIFELD-EXAKTHEIT: ohne Beugung muss der Faktor bei
    #    membrane_modes = 1 exakt D_1(u) = z01²J0(u)/(z01²−u²) sein.
    # c) MIT BEUGUNG und einer Mode ist der Faktor exakt 1 — dort trägt
    #    die Kette die Projektion bereits (Gegenprobe 33), doppelt wäre
    #    falsch.
    # d) WIRKUNG: gegen die COMSOL-Referenz sinkt die RMS-Abweichung ab
    #    5 kHz deutlich unter die 14.1 dB der uniformen Anregung.
    # e) KONVERGENZ über die Modenzahl. Hier stand lange, sie sei NICHT
    #    monoton (fünf Moden schlechter als drei). Das war eine Folge der
    #    inkonsistenten Gewichtung: die Zweige trugen in der Gewichtung
    #    nur die Materialdämpfung, waren also praktisch ungedämpft, und
    #    Zweig 4 bei 5.1 kHz mit Q ~ 1e4 bekam dort zu viel Gewicht.
    #    Seit Gegenprobe 43 stehen an beiden Stellen dieselben Zweige
    #    inklusive innerer Umverteilung; die Reihe fällt seither monoton
    #    (1..5 Moden). Das wird jetzt GEPRÜFT statt behauptet — und die
    #    Zweigresonanz bleibt als Strukturaussage daneben stehen.
    # Prüfling und Referenz sind die der Gegenprobe 32, eine reine Membran
    # (Spannung und Resonanz nach der Membranformel) — ohne Randschicht
    # der Folie (Gegenprobe 62).
    MicrophoneCapsule._RANDSCHICHT = False
    if _HAS_SCIPY:
        from scipy.special import j0 as _j0_34
        z34 = MicrophoneCapsule._J0_ZEROS
        par34 = dict(
            membrane_material={"rho": 1944.0, "E": 4.0e9, "nu": 0.35},
            membrane_resonance_hz=1040.0, membrane_diameter=36.0e-3,
            membrane_thickness=25e-6, membrane_tension=116.27,
            air_gap=230e-6, backplate_diameter=36.0e-3,
            backplate_thickness=1.6e-3, bias_voltage=1.0,
            architecture="single", n_through_holes=4,
            through_hole_diameter=1.0e-3, through_hole_pcd=2 * 8.4853e-3,
            n_blind_holes=0, rear_network_enabled=True,
            cavity_length=7.6e-3, n_cavity_holes=0, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, include_diffraction=False,
            squeeze_model="2d")
        f34 = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 7000.0,
                        10000.0])
        # a) aus == Bestand, und das Gatter greift
        h_a = MicrophoneCapsule(**par34).transfer_function(f34)
        h_b = MicrophoneCapsule(modal_source=0, **par34).transfer_function(f34)
        assert np.array_equal(h_a, h_b), \
            "modal_source = 0 muss bit-für-bit der Bestand sein"
        try:
            MicrophoneCapsule(modal_source=2, **par34)
            raise AssertionError("modal_source braucht ein Gatter")
        except ValueError:
            pass
        # b) Freifeld: eine Mode -> exakt D_1(u)
        c34 = MicrophoneCapsule(modal_source=1, membrane_modes=1, **par34)
        th34 = np.array([np.pi / 2])                  # streifend: u maximal
        om34 = 2.0 * np.pi * f34
        s34 = c34._modal_source_scale(
            om34, th34, c34._source_pressures_fundamental(om34, th34)[0])
        # (u mit dem Radius der wirksamen Membran, Gegenprobe 62)
        u34 = om34 / C_AIR * c34._a_w
        D34 = z34[0]**2 * _j0_34(u34) / (z34[0]**2 - u34**2)
        assert np.max(np.abs(s34[:, 0] - D34)) < 1e-12, \
            (f"Freifeld-Faktor muss exakt D_1(u) sein "
             f"({np.max(np.abs(s34[:, 0] - D34)):.1e})")
        # c) mit Beugung und einer Mode: Faktor exakt 1 (keine Doppelung)
        c34d = MicrophoneCapsule(
            **{**par34, "include_diffraction": True, "body_diameter": 40e-3,
               "modal_source": 1, "membrane_modes": 1})
        s34d = c34d._modal_source_scale(
            om34, th34, c34d._source_pressures_fundamental(om34, th34)[0])
        assert np.max(np.abs(s34d - 1.0)) < 1e-12, \
            "mit Beugung trägt die Kette die Projektion bereits (Faktor 1)"
        # d) Wirkung gegen die COMSOL-Referenz bei streifendem Einfall
        ref34 = np.array([0.00, -17.84, -28.49, -33.59, -30.94, -29.78,
                          -40.16])          # COMSOL, auf 1 kHz normiert
        def _graz34(**kw):
            cc = MicrophoneCapsule(**{**par34, **kw})
            H = np.asarray(cc.angle_responses(f34, angles_deg=(90.0,))
                           ["H"][90.0])
            a = 20.0 * np.log10(np.abs(H))
            return a - a[0]
        hi34 = f34 >= 5000.0
        rms_off = float(np.sqrt(np.mean(
            (_graz34(modal_source=0)[hi34] - ref34[hi34])**2)))
        reihe34 = [float(np.sqrt(np.mean(
            (_graz34(modal_source=1, membrane_modes=n)[hi34]
             - ref34[hi34])**2))) for n in range(1, 6)]
        rms_on = reihe34[2]                            # drei Moden
        assert rms_off > 12.0, \
            f"ohne Projektion muss die bekannte Lücke bleiben ({rms_off:.1f})"
        # Die Schranke war einmal 6 dB. Sie wurde mit UNGEDÄMPFTEN
        # Modenadmittanzen in der Gewichtung erreicht — die höheren
        # Zweige zählten dort schwerer, als das Netzwerk sie danach
        # rechnete. Gegenprobe 43 hat diese Inkonsistenz beseitigt
        # (dieselben Zweige inklusive _modal_internal_Z an beiden
        # Stellen); die Lücke schließt seither weniger weit, dafür
        # richtig. Festgehalten wird jetzt das VERHÄLTNIS; die Restlücke
        # selbst ist eine Sperrklinke (sie darf nur kleiner werden).
        assert rms_on < 0.6 * rms_off, \
            (f"modenabhängige Quelle muss die Lücke deutlich schließen "
             f"({rms_off:.1f} -> {rms_on:.1f} dB)")
        stand.sperrklinke("restluecke_3_moden", rms_on, "dB",
                          "RMS gegen COMSOL ab 5 kHz, streifend, 3 Moden",
                          besser="kleiner", toleranz=0.15)
        # KONVERGENZ: jede weitere Mode muss die Abweichung verkleinern.
        assert all(b < a for a, b in zip(reihe34, reihe34[1:])), \
            (f"die Reihe über die Modenzahl muss monoton fallen "
             f"({np.round(reihe34, 2)})")
        # e) Zweigdämpfung: die höheren Moden tragen ihre innere
        #    Umverteilung im Spaltfilm (_modal_internal_Z). Ohne sie wäre
        #    Zweig 4 mit Q ~ 1e4 praktisch ungedämpft; mit ihr liegt die
        #    Güte in der Größenordnung 10. Genau daran hing die Monotonie
        #    oben — die Zweigresonanz bleibt hier als Strukturaussage
        #    stehen, damit sichtbar ist, WORAN sie hängt.
        c34m = MicrophoneCapsule(modal_source=1, membrane_modes=5, **par34)
        M4, C4 = c34m._higher_mode_branches()[2]
        f4_0 = 1.0 / (2.0 * np.pi * np.sqrt(M4 * C4))
        om4 = np.array([2.0 * np.pi * f4_0])
        R4 = float(np.atleast_1d(c34m._membrane_film_damping(
            om4, c34m.h_gap_front, c34m.R_A_gap_front))[0])
        Zi4 = complex(np.atleast_1d(
            c34m._modal_internal_Z(om4, c34m.h_gap_front)[2])[0])
        Q4_0 = float(np.sqrt(M4 / C4) / R4)
        M4n = M4 + Zi4.imag / (2.0 * np.pi * f4_0)
        f4 = 1.0 / (2.0 * np.pi * np.sqrt(M4n * C4))
        Q4 = float(np.sqrt(M4n / C4) / (R4 + Zi4.real))
        assert Q4_0 > 1000.0, \
            "ohne innere Umverteilung wären die Zweige praktisch ungedämpft"
        assert 3.0 < Q4 < 200.0, \
            (f"Zweig 4 muss durch die innere Umverteilung realistisch "
             f"bedämpft sein (Q = {Q4:.1f} statt {Q4_0:.0f})")
        assert Zi4.real > 0.0 and Zi4.imag > 0.0, \
            "innere Umverteilung muss dissipativ UND träge sein"
        print(f"Modenabhängiger Quelldruck: aus ≡ Bestand, Gatter greift; "
              f"Freifeld exakt D_1(u) ({np.max(np.abs(s34[:, 0] - D34)):.0e}), "
              f"mit Beugung Faktor 1; streifend gegen COMSOL ab 5 kHz "
              f"{rms_off:.1f} -> {rms_on:.1f} dB (3 Moden); Zweig 4 durch "
              f"die innere Umverteilung von Q = {Q4_0:.0f} auf {Q4:.1f} "
              f"bedämpft ({f4_0:.0f} -> {f4:.0f} Hz); Konvergenz über die "
              f"Modenzahl monoton "
              f"({'->'.join(f'{x:.1f}' for x in reihe34)} dB)  OK")


def test_gp39_pull_in_der_gegentakt_bauform():
    """Gegenprobe 39: Pull-in der Gegentakt-Bauform."""
    # Die Erwartung „zwei Backplates -> doppelte Pull-in-Spannung" trifft
    # NICHT zu, und der Grund ist lehrreich genug, ihn festzuschreiben.
    # Bei ``dual`` liegt u_bias an BEIDEN Spalten voll an (±U gegen die
    # Membran). Damit
    #   * heben sich die statischen Kräfte auf: w0 = 0,
    #   * verdoppelt sich der Wandlerkoeffizient (das ist der Sinn der
    #     Bauform),
    #   * ADDIERT sich aber auch die Feder-Erweichung beider Spalte.
    # Pull-in ist deshalb das Kleinsignal-Kriterium k_gen = 2·eps0·U²·I_k(0)
    # am RUHESPALT, während die Einzel-Backplate erst kollabiert, nachdem
    # die Membran auf w0 = x*·h gekrochen ist — dort ist der Spalt kleiner
    # und I_k entsprechend größer. Genau diese Verschiebung ist der ganze
    # Gewinn.
    #
    # Mit dem Modenprofil phi = 1 - r²/a² und t = phi in Modenkoordinate:
    #     A2(x) = Int_0^1 t /(1 - x t)² dt = [1/(1-x) - 1 + ln(1-x)] / x²
    #     A3(x) = Int_0^1 t²/(1 - x t)³ dt
    #           = [1/(2(1-x)²) - 2/(1-x) - ln(1-x) + 3/2] / x³
    # Gleichgewicht UND tangentiale Instabilität zugleich liefern für die
    # Einzel-Backplate x* = A2(x*)/(2·A3(x*)) = 0.44042, und daraus
    #     U_PI(dual)/U_PI(single) = sqrt(A3(x*)/(2·A3(0))) = sqrt(1.5·A3(x*))
    #                             = 1.34640.
    # (Für einen starren Kolben wäre x* = 1/3 und das Verhältnis
    # sqrt(27/16) = 1.29904 — die Parabelmode kriecht weiter, deshalb der
    # etwas größere Gewinn.) Der Faktor 2 gälte nur, wenn man EINE
    # Versorgung symmetrisch auf beide Spalte teilt (U/2 je Spalt); dann
    # fällt die Empfindlichkeit aber auf den Einzel-Backplate-Wert zurück.
    def _A2_39(x):
        return (1.0 / (1.0 - x) - 1.0 + np.log(1.0 - x)) / x**2

    def _A3_39(x):
        return (0.5 / (1.0 - x)**2 - 2.0 / (1.0 - x)
                - np.log(1.0 - x) + 1.5) / x**3

    # a) die geschlossenen Formen gegen numerische Integration
    t39 = np.linspace(0.0, 1.0, 400001)
    worst39 = 0.0
    for x39 in (0.1, 0.3, 0.44042, 0.7):
        worst39 = max(
            worst39,
            abs(_A2_39(x39) / np.trapezoid(t39 / (1 - x39 * t39)**2, t39) - 1),
            abs(_A3_39(x39)
                / np.trapezoid(t39**2 / (1 - x39 * t39)**3, t39) - 1))
    assert worst39 < 1e-7, \
        f"geschlossene Form von A2/A3 muss stimmen ({worst39:.1e})"
    # b) Pull-in-Punkt der Einzel-Backplate und daraus das Verhältnis
    lo39, hi39 = 1e-6, 0.95
    for _ in range(80):
        mid39 = 0.5 * (lo39 + hi39)
        if mid39 - _A2_39(mid39) / (2.0 * _A3_39(mid39)) < 0.0:
            lo39 = mid39
        else:
            hi39 = mid39
    x_pi39 = 0.5 * (lo39 + hi39)
    rat39 = float(np.sqrt(1.5 * _A3_39(x_pi39)))
    # c) das Modell rechnet seit Gegenprobe 49 den EXAKTEN Arbeitspunkt;
    #    das Ein-Moden-Verhältnis oben ist damit die Näherung, nicht mehr
    #    der Sollwert. Exakte Anker, beide ohne freien Parameter:
    #      * dual: −∇²w = k²w mit k² = 2ε0U²/(T h³), eingespannt bei a
    #        -> k·a = j01, also Ā = j01²/4 = 1.44580 (geschlossen);
    #      * single: Warrens Ā = 0.789 (Gegenprobe 45).
    #    Elektrode nahezu voll, damit die Lochprofile die Aussage nicht
    #    verwischen; kleiner Bias, damit w0 -> 0.
    g39 = dict(membrane_diameter=25.4e-3, backplate_diameter=25.4e-3,
               membrane_resonance_hz=8000.0, air_gap=40e-6,
               backplate_thickness=3e-3, n_through_holes=4,
               through_hole_diameter=0.05e-3, n_blind_holes=0,
               bias_voltage=1.0, include_diffraction=False)
    c39s = MicrophoneCapsule(architecture="single", **g39)
    c39d = MicrophoneCapsule(architecture="dual", **g39)
    r39 = c39d.U_pullin / c39s.U_pullin

    def _abar39(cc):
        # Spannung, die die statische Nachgiebigkeit trägt (s. _static_
        # setup), Radius der WIRKSAMEN Membran: eingespannt ist sie um die
        # Randschicht √(D/T) vor dem Rand (Gegenprobe 62)
        return (cc.U_pullin**2 * cc._a_w**2 * EPS0
                / (2.0 * cc._st["tension"] * cc.h_gap**3))

    ad39, as39 = _abar39(c39d), _abar39(c39s)
    rex39 = float(np.sqrt((2.404825557695773**2 / 4.0) / 0.789))
    assert abs(ad39 / (2.404825557695773**2 / 4.0) - 1.0) < 2e-3, \
        (f"Gegentakt-Pull-in muss j01²/4 = 1.44580 sein ({ad39:.5f})")
    assert abs(as39 / 0.789 - 1.0) < 2e-3, \
        f"Einzel-Backplate muss Warrens 0.789 treffen ({as39:.5f})"
    assert abs(r39 / rex39 - 1.0) < 3e-3, \
        (f"Pull-in-Verhältnis dual/single muss sqrt(1.4458/0.789) = "
         f"{rex39:.5f} sein ({r39:.5f})")
    assert r39 > rat39, \
        ("das Ein-Moden-Bild unterschätzt den Gewinn (es überschätzt den "
         "Pull-in der Einzel-Backplate stärker als den der Gegentakt-"
         "Bauform)")
    assert r39 > 1.0, "zwei symmetrische Backplates müssen den Pull-in ANHEBEN"
    assert abs(r39 - 2.0) > 0.5, \
        "der Gewinn ist NICHT Faktor 2 — die Erweichung beider Spalte addiert sich"
    # d) die Gegentakt-Verschaltung selbst: w0 = 0, theta und Erweichung x2
    assert c39d.w0_static == 0.0, \
        "bei symmetrischen Backplates heben sich die statischen Kräfte auf"
    assert abs(c39d._theta / c39s._theta - 2.0) < 1e-3, \
        (f"Gegentakt muss den Wandlerkoeffizienten verdoppeln "
         f"({c39d._theta / c39s._theta:.5f})")
    assert abs(c39d.softening_ratio / c39s.softening_ratio - 2.0) < 1e-3, \
        (f"die Feder-Erweichung beider Spalte muss sich addieren "
         f"({c39d.softening_ratio / c39s.softening_ratio:.5f})")
    print(f"Gegentakt-Pull-in: U_PI(dual)/U_PI(single) = {r39:.5f} gegen "
          f"exakt sqrt(1.4458/0.789) = {rex39:.5f} (Ā dual {ad39:.4f} = "
          f"j01²/4, single {as39:.4f} = Warren; Ein-Moden-Bild "
          f"{rat39:.5f} mit x* = {x_pi39:.5f}, starrer Kolben "
          f"{np.sqrt(27 / 16):.5f}); w0 = 0, Wandlerkoeffizient und "
          f"Feder-Erweichung beide ×2 — der Faktor 2 im Pull-in gälte nur "
          f"bei geteilter Versorgung  OK")


def test_gp42_flachenmittel_der_modenreihe():
    """Gegenprobe 42: Flächenmittel der Modenreihe."""
    # Grinnip (JAES 54(3), 2006, Gl. 54/55) gibt als Signal das
    # FLÄCHENMITTEL der Membranauslenkung aus: jede Mode wird einzeln per
    # Galerkin-Projektion angetrieben und trägt mit ihrem eigenen
    # Flächenmittel ⟨ψ_m⟩ = 2·J1(z_m)/z_m zum Ausgang bei. Genau diese
    # Reihe steht hinter unseren PARALLELEN Modenzweigen — und sie hat
    # zwei exakt bekannte Grenzwerte, an denen sich alles aufhängen lässt:
    #
    #   Σ 1/z_m² = 1/4      ->  Σ 1/M_m = S/σ  (Hochton: freier KOLBEN)
    #   Σ 1/z_m⁴ = 1/32     ->  Σ C_m  = πa⁴/(8T)  (exakte Statik)
    #
    # Die zweite Summe ist der Grund für :meth:`_modal_split_factor`: die
    # Grundmode allein trägt 32/z_1⁴ = 95.68 % der statischen
    # Nachgiebigkeit, unser Ein-Freiheitsgrad-Modell aber 100 %. Ohne
    # Normierung addieren die höheren Zweige noch einmal 4.4 % dazu.
    #
    # Geprüft wird DREIERLEI, jeweils gegen eine geschlossene Lösung:
    # a) die beiden Rayleigh-Summen selbst,
    # b) die Modenreihe des Flächenmittels gegen die exakte dynamische
    #    Vakuumlösung η(r) = (p/σω²)[J0(kr)/J0(ka) − 1] — dieselbe
    #    Funktion, die Zuckerwar als Ansatz benutzt — über die Grundmode
    #    hinweg, plus die statische Bessel-Anregung gegen Quadratur,
    # c) das MODELL: die Tieftonempfindlichkeit darf nicht davon abhängen,
    #    mit wie vielen Moden gerechnet wird, und die Modenresonanzen
    #    müssen dabei stehen bleiben.
    if _HAS_SCIPY:
        from scipy.integrate import quad as _quad42
        from scipy.special import j0 as _j0_42, j1 as _j1_42
        from scipy.special import jn_zeros as _jnz_42
        # a) Rayleigh-Summen (die Identitäten, auf denen die Normierung ruht)
        z42 = _jnz_42(0, 4000)
        s4_42 = float(np.sum(1.0 / z42**4))
        s2_42 = float(np.sum(1.0 / z42**2))
        assert abs(s4_42 * 32.0 - 1.0) < 1e-9, \
            f"Rayleigh-Summe Σ1/z⁴ muss 1/32 sein ({s4_42:.12f})"
        assert abs(s2_42 * 4.0 - 1.0) < 1e-3, \
            f"Rayleigh-Summe Σ1/z² muss 1/4 sein ({s2_42:.9f})"

        # b) Flächenmittel der Modenreihe gegen geschlossene Lösungen
        a42, T42, sg42 = 1.0945e-2, 39.19, 1630.0 * 2.4e-6

        def _amean42(p_func, om, n_mod):
            """⟨η⟩ nach Grinnip Gl. (54)/(55), Vakuum."""
            tot = 0.0
            for m in range(n_mod):
                zm = z42[m]
                proj = _quad42(
                    lambda x, _z=zm: p_func(x * a42) * _j0_42(_z * x) * 2.0 * x,
                    0.0, 1.0, limit=200)[0]
                tot += (proj / ((T42 * (zm / a42)**2 - om**2 * sg42)
                                * _j1_42(zm)**2)) * 2.0 * _j1_42(zm) / zm
            return tot

        # b1) statisch, gleichförmig: Konvergenz gegen a²/(8T)
        st42 = a42**2 / (8.0 * T42)
        konv42 = [abs(_amean42(lambda r: 1.0, 0.0, n) / st42 - 1.0)
                  for n in (1, 5, 60)]
        assert konv42[0] > 0.04 and abs(konv42[0] - (1.0 - 32.0 / z42[0]**4)) \
            < 1e-3, \
            (f"die Grundmode allein muss 32/z_1⁴ = 95.7 % tragen "
             f"({100 * (1 - konv42[0]):.2f} %)")
        assert konv42[2] < 1e-4, \
            f"die Reihe muss gegen die exakte Statik konvergieren ({konv42})"
        # b2) statisch, Bessel-Antrieb, gegen direkte Quadratur der ODE
        worst42 = 0.0
        for u42 in (1.0, 2.8, 5.0):
            pf42 = (lambda r, _u=u42: _j0_42(_u * r / a42))

            def _eta42(r, _p=pf42):
                inner = lambda s: _quad42(lambda t: _p(t) * t, 0.0, s,
                                          limit=200)[0]
                return _quad42(lambda s: inner(s) / s, r, a42,
                               limit=200)[0] / T42

            ex42 = _quad42(lambda r: _eta42(r) * 2.0 * r / a42**2, 0.0, a42,
                           limit=200)[0]
            worst42 = max(worst42,
                          abs(_amean42(pf42, 0.0, 60) / ex42 - 1.0))
        assert worst42 < 1e-4, \
            f"Reihe muss die statische Quadratur treffen ({worst42:.1e})"
        # b3) dynamisch im Vakuum über die Grundmode hinweg
        worst42d = 0.0
        for f42 in (100.0, 2500.0, 5000.0, 15000.0):
            om42 = 2 * np.pi * f42
            k42 = om42 * np.sqrt(sg42 / T42) * a42
            ex42d = ((2.0 * _j1_42(k42) / (k42 * _j0_42(k42)) - 1.0)
                     / (sg42 * om42**2))
            worst42d = max(worst42d,
                           abs(_amean42(lambda r: 1.0, om42, 60) / ex42d - 1.0))
        assert worst42d < 1e-4, \
            (f"Reihe muss die geschlossene Vakuumlösung treffen "
             f"({worst42d:.1e})")

        # c) DAS MODELL: Tiefton modenunabhängig, Resonanzen unverändert
        g42 = dict(membrane_diameter=25.4e-3, membrane_resonance_hz=8000.0,
                   air_gap=40e-6, backplate_diameter=25e-3,
                   backplate_thickness=3e-3, n_through_holes=60,
                   through_hole_diameter=1.0e-3, n_blind_holes=30,
                   include_diffraction=False, squeeze_model="2d")
        f_lo42 = np.array([20.0])
        s_ref42 = abs(MicrophoneCapsule(membrane_modes=1,
                                        **g42).transfer_function(f_lo42)[0])
        dlf42, fm42 = [], None
        for nm42 in (2, 3, 4, 5):
            c42 = MicrophoneCapsule(membrane_modes=nm42, **g42)
            dlf42.append(20 * np.log10(
                abs(c42.transfer_function(f_lo42)[0]) / s_ref42))
            got = [1.0 / (2 * np.pi * np.sqrt(M * C))
                   for M, C in c42._higher_mode_branches()]
            soll = [c42.f_res * z42[m] / z42[0] for m in range(1, nm42)]
            assert np.allclose(got, soll, rtol=1e-9), \
                (f"die Normierung darf die Modenresonanzen nicht "
                 f"verschieben ({np.round(got)} gegen {np.round(soll)})")
            fm42 = got
        assert max(abs(d) for d in dlf42) < 0.02, \
            (f"die Tieftonempfindlichkeit darf nicht von der Modenzahl "
             f"abhängen ({np.round(dlf42, 4)} dB — ohne Normierung waren "
             f"es +0.30…+0.37 dB)")
        # d) dokumentierter Preis: Hochtongrenzwert gegen die Kolbenmasse
        hf42 = [0.75 * float(np.sum((z42[0] / z42[:n])**2))
                / float(np.sum((z42[0] / z42[:n])**4)) for n in (1, 3, 5)]
        assert all(hf42[i] < hf42[i + 1] < 1.0 for i in range(2)), \
            (f"der Hochtongrenzwert muss MONOTON von unten gegen die "
             f"Kolbenmasse S/σ laufen ({np.round(hf42, 4)})")
        print(f"Flächenmittel der Modenreihe (Grinnip 2006, Gl. 54/55): "
              f"Rayleigh Σ1/z⁴·32 = {s4_42 * 32:.9f}; Reihe trifft die "
              f"statische Quadratur ({worst42:.0e}) und die geschlossene "
              f"Vakuumlösung ({worst42d:.0e}); Grundmode allein trägt "
              f"{100 * 32 / z42[0]**4:.2f} % der Statik — deshalb normiert: "
              f"Tiefton jetzt modenunabhängig "
              f"({max(abs(d) for d in dlf42):.4f} dB statt +0.37), "
              f"Modenresonanzen unverändert ({fm42[0] / 1e3:.1f} kHz…), "
              f"Hochtongrenzwert monoton {np.round(hf42, 3)} gegen "
              f"Kolbenmasse  OK")


def test_gp45_mittenterminierung_ringmembran():
    """Gegenprobe 45: Mittenterminierung (Ringmembran)."""
    # Eine in der Mitte festgelegte Membran — Kontaktstift, Mittenbolzen —
    # ist im Fachsinn eine RINGMEMBRAN. Das ist kein kleiner Korrekturterm,
    # sondern eine Änderung der Randwertaufgabe, und der Unterschied ist
    # LOGARITHMISCH: die statische Lösung von T∇²w = −p mit zwei Rändern
    # ist w ∝ (a²−r²) + (a²−r_i²)·ln(r/a)/ln(a/r_i), und ein Logarithmus
    # verschwindet nicht wie r_i². Schon r_i/a = 1 % nimmt 22 % der
    # Nachgiebigkeit weg. Deshalb ist der Grenzwert ρ → 0 zwar 1, aber
    # unbrauchbar als Prüfung — jede Verifikation muss BEI ENDLICHEM ρ
    # stattfinden.
    #
    # a) r_i = 0 ist EXAKT der Bestand (die Momente werden dort analytisch
    #    gesetzt, nicht quadriert).
    # b) STATISCHE FORM gegen die Differentialgleichung selbst: eine
    #    unabhängige Finite-Volumen-Lösung von (1/r)(r w')' = −4 mit
    #    w(r_i) = w(a) = 0 muss das geschlossene Profil treffen, und der
    #    geschlossene Nachgiebigkeitsfaktor g(ρ) seine Quadratur.
    # c) MODENINTEGRALE: die geschlossenen Formen für ∫ψ dA und ∫ψ² dA
    #    (Wronski-Identität) gegen numerische Quadratur der Modenform.
    # d) RINGVARIANTE DER RAYLEIGH-SUMME: Σ C_m muss die statische
    #    Nachgiebigkeit der RINGmembran treffen — die Verallgemeinerung
    #    von Σ1/z⁴ = 1/32 (Gegenprobe 42), und zugleich der Beweis, dass
    #    Eigenwerte, Modenintegrale und geschlossene Nachgiebigkeit
    #    zueinander passen.
    # e) MASSENSUMMENREGEL: Σ 1/M_m = S_Ring/σ — weit oberhalb aller
    #    Resonanzen bewegt sich die Ringmembran wie ein freier Kolben der
    #    RINGFLÄCHE.
    # f) PULL-IN gegen J. E. Warren, JASA 58(3), 733–740 (1975): der
    #    kritische Antriebsparameter Ā = V²a²ε₀/(2Th³) einer flachen,
    #    LOCHFREIEN Elektrode ist 0.789 für die Kreis- und 1.548 für die
    #    Ringmembran mit ρ = 0.1. Der ursprüngliche Ein-Moden-Galerkin
    #    lag systematisch darüber (Ring +2.6 %, Vollkreis +5.0 %); seit
    #    Gegenprobe 49 rechnet das Modell den exakten Arbeitspunkt und
    #    trifft beide Werte.
    # g) GATTER: 3D-Löser, zu großer Pfosten, biegesteife Platte.
    #
    # Alle Referenzen hier — Rayleigh, die geschlossenen Formen, Warren —
    # rechnen die REINE Membran; die Probe läuft deshalb ohne Randschicht
    # der Folie. Dass das Modell mit Randschicht dieselben Formeln auf die
    # wirksame Membran (Radius a − √(D/T)) anwendet, prüft Gegenprobe 62.
    MicrophoneCapsule._RANDSCHICHT = False
    if _HAS_SCIPY:
        # a) Grenzfall
        c45o = MicrophoneCapsule()
        assert c45o._mass_factor_rayleigh == 4.0 / 3.0, \
            "ohne Pfosten ist der Rayleigh-Wert exakt 4/3"
        assert c45o._piston_factor == 8.0 / 2.404825557695773 ** 2, \
            "ohne Pfosten ist der Massenfaktor exakt 8/j01² (Gegenprobe 55)"
        assert c45o.S_eff_mem == 0.5 * c45o.S_mem, "ohne Pfosten S_eff = S/2"
        assert _ring_compliance_factor(0.0) == 1.0
        assert c45o._k_gen == c45o.S_mem**2 / (4.0 * c45o.C_A_mem), \
            "ohne Pfosten muss k_gen bitgleich S²/(4C_A) sein"

        # b) statische Form gegen die DGL und gegen die Quadratur
        w45, g45 = 0.0, 0.0
        for rho45 in (0.02, 0.1, 0.3):
            ui45 = rho45**2
            uq45 = np.linspace(ui45, 1.0, 400001)
            g_num = 2.0 * float(np.trapezoid(
                _ring_static_shape(uq45, ui45), uq45))
            g45 = max(g45, abs(g_num / _ring_compliance_factor(rho45) - 1.0))
            n45 = 4001
            r45 = np.linspace(rho45, 1.0, n45)
            dr45 = r45[1] - r45[0]
            rf45 = 0.5 * (r45[:-1] + r45[1:])
            ab45 = np.zeros((3, n45))
            ab45[1, 0] = ab45[1, -1] = 1.0
            ab45[1, 1:-1] = -(rf45[:-1] + rf45[1:]) / (r45[1:-1] * dr45**2)
            ab45[0, 2:] = rf45[1:] / (r45[1:-1] * dr45**2)
            ab45[2, :-2] = rf45[:-1] / (r45[1:-1] * dr45**2)
            b45 = np.full(n45, -4.0)
            b45[0] = b45[-1] = 0.0
            w_num = _solve_banded((1, 1), ab45, b45)
            w_cf = _ring_static_shape(r45**2, ui45)
            w45 = max(w45, float(np.max(np.abs(w_num - w_cf))
                                 / np.max(np.abs(w_cf))))
        assert g45 < 1e-8, f"g(ρ) muss die Quadratur treffen ({g45:.1e})"
        assert w45 < 1e-5, \
            f"die geschlossene Form muss die DGL lösen ({w45:.1e})"

        # c) Modenintegrale geschlossen gegen Quadratur
        i45 = 0.0
        for rho45 in (0.05, 0.2):
            md45 = MicrophoneCapsule._ring_eigen(rho45, 6)
            rq45 = np.linspace(rho45, 1.0, 400001)
            for m45 in range(6):
                z45 = md45["z"][m45]
                ps45 = (_besselj(0, z45 * rq45) * _bessely(0, z45 * rho45)
                        - _bessely(0, z45 * rq45) * _besselj(0, z45 * rho45))
                for got, ref in (
                        (md45["I1"][m45],
                         2.0 * np.trapezoid(ps45 * rq45, rq45)),
                        (md45["I2"][m45],
                         2.0 * np.trapezoid(ps45**2 * rq45, rq45))):
                    i45 = max(i45, abs(got / ref - 1.0))
        assert i45 < 1e-6, \
            f"Modenintegrale: geschlossen gegen Quadratur ({i45:.1e})"

        # d/e) Ring-Rayleigh-Summe und Massensummenregel
        s45, p45 = 0.0, 0.0
        for rho45 in (0.02, 0.1, 0.3):
            md45 = MicrophoneCapsule._ring_eigen(rho45, 400)
            Cm45 = md45["I1"]**2 / (md45["z"]**2 * md45["I2"])
            s45 = max(s45, abs(8.0 * float(np.sum(Cm45))
                               / _ring_compliance_factor(rho45) - 1.0))
            # Die Massensummenregel konvergiert nur wie 1/N (die Terme
            # gehen wie 1/z²), die Nachgiebigkeitssumme wie 1/N³. Deshalb
            # steht hier eine Teilsummen-Aussage: von UNTEN und auf 5e-3.
            q45 = (float(np.sum(md45["I1"]**2 / md45["I2"]))
                   / (1.0 - rho45**2))
            assert q45 < 1.0, \
                f"Teilsumme Σ1/M_m muss von unten kommen ({q45:.6f})"
            p45 = max(p45, abs(q45 - 1.0))
        assert s45 < 1e-6, \
            (f"Σ C_m muss die statische Ring-Nachgiebigkeit treffen "
             f"({s45:.1e}) — Ringvariante von Σ1/z⁴ = 1/32")
        assert p45 < 5e-3, \
            f"Σ 1/M_m muss die RINGkolbenmasse treffen ({p45:.1e}, 400 Moden)"

        # f) Pull-in gegen Warren (flache, LOCHFREIE Elektrode über der
        #    ganzen Membran — genau Warrens Konfiguration)
        g45w = dict(
            membrane_resonance_hz=None, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=26e-3, backplate_thickness=4e-3,
            bias_voltage=1.0, architecture="single", n_through_holes=0,
            n_blind_holes=0, rear_network_enabled=False,
            squeeze_model="1d", include_diffraction=False)
        warren45 = {0.0: 0.789, 0.1: 1.548}          # Warren 1975
        ab45w = {}
        for rho45, ref45 in warren45.items():
            cw45 = MicrophoneCapsule(
                **dict(g45w, center_post_diameter=2.0 * rho45 * 13e-3))
            # T = die Spannung, die die statische Nachgiebigkeit trägt
            # (reine Membran wie bei Warren; mit Randschicht gilt Ā für
            # den wirksamen Radius, s. Gegenprobe 62)
            A45 = (cw45.U_pullin**2 * cw45.a_mem**2 * EPS0
                   / (2.0 * cw45._st["tension"] * cw45.h_gap**3))
            ab45w[rho45] = (A45, A45 / ref45 - 1.0)
        # seit Gegenprobe 49 der EXAKTE Arbeitspunkt: Warren wird
        # getroffen (vorher Ein-Moden-Galerkin +5.0 % bzw. +2.6 %)
        assert abs(ab45w[0.0][1]) < 2e-3, \
            (f"Vollkreis muss Warrens 0.789 treffen "
             f"({ab45w[0.0][0]:.4f})")
        assert abs(ab45w[0.1][1]) < 2e-3, \
            (f"Ringmembran ρ = 0.1 muss Warrens 1.548 treffen "
             f"({ab45w[0.1][0]:.4f})")
        # und die Wirkung selbst: ρ = 0.1 hebt Ā um Faktor ~1.96
        assert 1.9 < ab45w[0.1][0] / ab45w[0.0][0] < 2.0, \
            (f"Ringmembran muss fast doppelt so stabil sein "
             f"({ab45w[0.1][0] / ab45w[0.0][0]:.3f})")

        # g) Gatter
        for kw45, was45 in (
                (dict(center_post_diameter=0.7 * 22e-3),
                 "Pfosten über 0.6·a"),
                (dict(center_post_diameter=1e-3, membrane_thickness=200e-6,
                      membrane_resonance_hz=None),
                 "biegesteife Platte mit Ringfaktor"),
                (dict(center_post_diameter=1e-3, backplate_diameter=0.5e-3),
                 "Pfosten deckt die Backplate ab")):
            try:
                MicrophoneCapsule(**kw45)
                raise AssertionError(f"{was45} muss scheitern")
            except ValueError:
                pass
        print(f"Mittenterminierung (Ringmembran): r_i = 0 bitgleich; "
              f"geschlossene Form löst die DGL ({w45:.0e}) und trifft ihre "
              f"Quadratur ({g45:.0e}); Modenintegrale {i45:.0e}; Ring-"
              f"Rayleigh-Summe Σ C_m = C_A ({s45:.0e}) und Σ1/M_m = "
              f"S_Ring/σ ({p45:.0e}); Pull-in gegen Warren 1975: "
              f"{ab45w[0.0][0]:.4f} gegen 0.789 ({100 * ab45w[0.0][1]:+.2f} %) "
              f"und {ab45w[0.1][0]:.4f} gegen 1.548 "
              f"({100 * ab45w[0.1][1]:+.2f} %), Ring also fast doppelt so "
              f"stabil ({ab45w[0.1][0] / ab45w[0.0][0]:.2f}×); Gatter "
              f"greifen  OK")


def test_gp49_exakter_statischer_arbeitspunkt(stand):
    """Gegenprobe 49: exakter statischer Arbeitspunkt."""
    # Der Arbeitspunkt der Membran unter Polarisationsspannung ist eine
    # nichtlineare Randwertaufgabe (s. _static_setup). Bis Gegenprobe 48
    # stand ein Ein-Moden-Galerkin da, am Pull-in um +5.0 % (Kreis) bzw.
    # +2.6 % (Ring) in Warrens Ā zu steif. Geprüft wird die exakte Lösung
    # mit UNABHÄNGIGEN Methoden:
    # a) FORM: ein Schießverfahren (solve_ivp in r, Reihenstart an der
    #    Achse) muss dieselbe statische Auslenkung liefern wie das Finite-
    #    Volumen-System in u = r²/a².
    # b) LINEARISIERUNG: die Kleinsignal-Nachgiebigkeit aus L + λ∂p/∂w muss
    #    die Ableitung ∂V/∂p des NICHTLINEAREN Asts sein (zusätzlicher
    #    Gleichdruck, finite Differenz).
    # c) FALTE: zum Pull-in hin divergiert die Nachgiebigkeit.
    # d) Warren und j01²/4 prüfen Gegenproben 39 und 45; hier steht der
    #    Vergleich mit dem Ein-Moden-Bild in geschlossener Form:
    #    Ā_Galerkin = max 2x/A2(x) = 0.8275 bei x* = 0.4404.
    # e) WIRKUNG an der K67: Pull-in, Arbeitspunkt, C0.
    if _HAS_SCIPY:
        from scipy.integrate import solve_ivp as _ivp49
        from scipy.optimize import brentq as _brentq49
        g49 = dict(
            membrane_resonance_hz=None, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=26e-3, backplate_thickness=4e-3,
            architecture="single", n_through_holes=0, n_blind_holes=0,
            rear_network_enabled=False, squeeze_model="1d",
            include_diffraction=False)
        c49p = MicrophoneCapsule(bias_voltage=1.0, **g49)
        U49 = 0.8 * c49p.U_pullin
        c49 = MicrophoneCapsule(bias_voltage=U49, **g49)
        T49 = c49._st["tension"]
        # eingespannt am Rand der WIRKSAMEN Membran (Randschicht √(D/T),
        # Gegenprobe 62)
        a49, h49 = c49._a_w, c49.h_gap
        p49 = 0.5 * EPS0 * U49**2

        # a) Schießverfahren
        def _shoot49(wc):
            r_s = 1e-6 * a49
            w_s = wc - p49 / (h49 - wc)**2 * r_s**2 / (4.0 * T49)

            def rhs(r, y):
                return [y[1], -y[1] / r - p49 / ((h49 - y[0])**2 * T49)]
            sol = _ivp49(rhs, (r_s, a49),
                         [w_s, -p49 / (h49 - wc)**2 * r_s / (2.0 * T49)],
                         rtol=1e-11, atol=1e-16, dense_output=True)
            return sol

        wc49 = _brentq49(lambda wc: _shoot49(wc).y[0, -1], 0.0,
                         1.02 * c49.w0_static, xtol=1e-16)
        shot49 = _shoot49(wc49)
        r_chk49 = a49 * np.array([0.0, 0.3, 0.6, 0.9])
        w_fv49 = np.interp((r_chk49 / c49.a_mem)**2, c49._st["u"],
                           c49._w_static)
        w_sh49 = np.array([wc49] + list(shot49.sol(r_chk49[1:])[0]))
        dev_a49 = float(np.max(np.abs(w_fv49 - w_sh49)) / wc49)
        assert dev_a49 < 1e-4, \
            (f"statische Form: Finite-Volumen und Schießverfahren müssen "
             f"übereinstimmen ({dev_a49:.1e})")

        # b) Linearisierung gegen die Ableitung des nichtlinearen Asts
        st49 = c49._st
        free49 = np.ones(st49["u"].size)
        free49[-1] = 0.0
        lam49 = EPS0 * U49**2

        def _vol49(dp):
            w = c49._w_static.copy()
            for _ in range(60):
                F, J, _, _ = c49._st_system(w, lam49)
                F = F + dp * st49["vol"] * free49
                dw = _solve_banded((1, 1), J, -F)
                w = w + dw
                if np.max(np.abs(dw)) < 1e-15 * h49:
                    break
            return float(np.dot(st49["vol"], w)) * st49["S"]

        dp49 = 1e-3
        C_fd49 = (_vol49(dp49) - _vol49(-dp49)) / (2.0 * dp49)
        C_ref49 = c49._st_compliance(np.zeros(st49["u"].size))
        dev_b49 = abs((C_fd49 / C_ref49) / (c49.C_A_eff / c49.C_A_mem) - 1.0)
        assert dev_b49 < 1e-5, \
            (f"Kleinsignal-Nachgiebigkeit muss die Ableitung des statischen "
             f"Asts sein ({dev_b49:.1e})")

        # c) Falte: die Nachgiebigkeit divergiert zum Pull-in hin
        ce49 = [MicrophoneCapsule(bias_voltage=q * c49p.U_pullin,
                                  **g49).C_A_eff / c49p.C_A_mem
                for q in (0.5, 0.9, 0.99, 0.999)]
        assert all(b > a for a, b in zip(ce49, ce49[1:])) \
            and ce49[-1] > 5.0, \
            f"Nachgiebigkeit muss am Pull-in divergieren ({np.round(ce49, 2)})"
        try:
            MicrophoneCapsule(bias_voltage=1.001 * c49p.U_pullin, **g49)
            raise AssertionError("über dem Pull-in muss die Kapsel kollabieren")
        except ValueError:
            pass
        x_pi49 = c49p._st_branch()[0][-1][2].max() / h49

        # d) Ein-Moden-Bild in geschlossener Form
        def _A2_49(x):
            return (1.0 / (1.0 - x) - 1.0 + np.log(1.0 - x)) / x**2
        xs49 = np.linspace(0.3, 0.6, 30001)
        abar_g49 = float(np.max(2.0 * xs49 / _A2_49(xs49)))
        abar_x49 = (c49p.U_pullin**2 * a49**2 * EPS0
                    / (2.0 * T49 * h49**3))
        assert abs(abar_x49 / 0.789 - 1.0) < 2e-3, \
            f"exakter Pull-in muss Warren treffen ({abar_x49:.4f})"
        assert abs(abar_g49 / 0.8275 - 1.0) < 1e-3 \
            and abar_g49 > 1.04 * abar_x49, \
            (f"das Ein-Moden-Bild liegt geschlossen bei 0.8275 "
             f"({abar_g49:.4f}) — deutlich zu steif")

        # e) Wirkung an der K67 (nur Stand, keine Stellschraube)
        k49 = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3)
        # Dass das Ein-Moden-Bild den Pull-in überschätzt, prüft d)
        # exakt am Antriebsparameter; hier nur der Stand der K67.
        stand.wert("k67_pull_in", k49.U_pullin, "V",
                   "K67: exakter Pull-in")
        stand.wert("k67_w0", k49.w0_static * 1e6, "µm",
                   "K67: statische Durchbiegung bei 60 V")
        print(f"Exakter Arbeitspunkt: Form gegen Schießverfahren "
              f"{dev_a49:.0e}, Nachgiebigkeit == ∂V/∂p des Asts "
              f"({dev_b49:.0e}), divergiert an der Falte "
              f"({' -> '.join(f'{x:.2f}' for x in ce49)}× bei 0.5…0.999 U_PI), "
              f"Faltpunkt w_max/h = {x_pi49:.3f} (Ein-Moden-Bild 0.440); "
              f"Ā = {abar_x49:.4f} gegen Warren 0.789, Ein-Moden-Bild "
              f"{abar_g49:.4f} ({100 * (abar_g49 / abar_x49 - 1):+.1f} %); "
              f"K67: U_PI {k49.U_pullin:.1f} V, w0 "
              f"{k49.w0_static * 1e6:.1f} µm, C0 {k49.C_elec_0 * 1e12:.1f} pF, "
              f"Erweichung {100 * k49.softening_ratio:.1f} %  OK")


@pytest.mark.feld3d
def test_gp55_massenfaktor_8_z12_g_der():
    """Gegenprobe 55: Massenfaktor 8/(z1²·g) der Kette."""
    # Ein Freiheitsgrad trifft nur zwei Dinge exakt. Die Kette nimmt die
    # statische Nachgiebigkeit C_T = S·a²·g/(8T) und wählte die Masse bis
    # hierher mit dem Rayleigh-Wert der statischen Form (Parabel 4/3). Der
    # ist eine obere Schranke der Frequenz: die Resonanz lag 1.9 % zu
    # hoch, oder — bei vorgegebener Resonanz — die statische Nachgiebigkeit
    # 3.75 % zu hoch (+0.32 dB). Jetzt μ = 8/(z1²·g) (ohne Pfosten 8/j01²
    # = 1.383): Statik UND Grundresonanz exakt.
    # a) RESONANZ: bei vorgegebener Vorspannung trifft die Kette den
    #    Eigenwert der Membran (Vollkreis und Ringmembran) exakt — seit
    #    Gegenprobe 62 den der WIRKSAMEN Membran, eingespannt um die
    #    Randschicht √(D/T) vor den Rändern. Gegen die eingespannte Platte
    #    unter Zug bleibt nur die Plattendispersion im Innern, ≈ z1²λ²/2.
    # b) STATIK: bei vorgegebener Resonanz ist C_A_mem die statische
    #    Nachgiebigkeit der Membran, deren exakte Grundmode f_res ist —
    #    „f_res vorgeben" und „Spannung vorgeben" sind dieselbe Kapsel
    #    (seit Gegenprobe 62 ohne Rest: die Biegung steckt in der
    #    Randschicht, auf beiden Wegen dieselbe).
    # c) GEGEN DAS 3D-FELD (volle Membran, keine Moden): ½"-Prüfling der
    #    Gegenprobe 48 über f_res vorgegeben, 20 Hz. Mit 4/3 lag 2D
    #    +0.12 dB über 3D, jetzt gleich.
    # d) MODENREIHE: die nachgiebigkeitserhaltend normierte Mehrmoden-
    #    Kette (s. _modal_split_factor) läuft für N -> ∞ genau dann auf den
    #    freien Kolben der Membranfläche, wenn μ·g·z1²/8 = 1 ist — jeder
    #    Zweig trägt dann exakt Modenmasse und -nachgiebigkeit. Mit 8/j01²
    #    konvergiert sie von oben auf 1, mit 4/3 lief sie auf 0.964, also
    #    unter die Kolbenmasse. Die Ringmembran geht auf die Ringfläche
    #    S·(1 − ρ²). (Reine Membranmathematik, ohne Randschicht.)
    if _HAS_SCIPY:
        from scipy.special import jn_zeros as _jn_zeros55
        j01_55 = 2.404825557695773
        # a) Resonanz bei vorgegebener Vorspannung
        pT55 = dict(membrane_resonance_hz=None, membrane_tension=45.0,
                    membrane_diameter=25.4e-3, membrane_thickness=6e-6)
        fr55 = {}
        for dp55 in (0.0, 1.0e-3, 3.0e-3):
            c55 = MicrophoneCapsule(center_post_diameter=dp55, **pT55)
            # wirksame Membran aus den Stoffwerten, unabhängig vom Modell
            D55 = (c55.mat_E * c55.t_mem ** 3
                   / (12.0 * (1.0 - c55.mat_nu ** 2)))
            l55 = np.sqrt(D55 / 45.0)
            aw55 = c55.a_mem - l55
            z55w = (j01_55 if dp55 == 0.0 else MicrophoneCapsule._ring_eigen(
                (0.5 * dp55 + l55) / aw55, 1)["z"][0])
            f_w55 = z55w / (2.0 * np.pi * aw55) * np.sqrt(45.0 / c55.sigma_mem)
            lam55 = l55 / c55.a_mem
            fr55[dp55] = (c55.f_res / f_w55 - 1.0,
                          c55._piston_factor, c55._mass_factor_rayleigh,
                          c55.f_res / c55.f_res_modal_exact - 1.0,
                          -0.5 * (z55w * lam55) ** 2)
            assert abs(fr55[dp55][0]) < 1e-10, \
                (f"a) Pfosten {dp55 * 1e3:.0f} mm: die Kette muss den "
                 f"Eigenwert der wirksamen Membran treffen "
                 f"({fr55[dp55][0]:+.1e})")
            assert abs(fr55[dp55][3] / fr55[dp55][4] - 1.0) < 0.15, \
                (f"a) Pfosten {dp55 * 1e3:.0f} mm: gegen die Platte bleibt "
                 f"die Plattendispersion ({fr55[dp55][3]:+.2e} statt "
                 f"{fr55[dp55][4]:+.2e})")
        uw55 = ((25.4e-3 / 2 - l55) / (25.4e-3 / 2)) ** 2
        assert abs(fr55[0.0][1] / (8.0 / (j01_55 ** 2 * uw55)) - 1.0) < 1e-12, \
            "a) ohne Pfosten 8/j01² auf die wirksame Fläche S·u_w"
        assert all(v[1] > v[2] for v in fr55.values()), \
            "a) der Rayleigh-Wert ist eine obere Schranke der Frequenz"
        fbk55 = MicrophoneCapsule(
            membrane_material={"rho": 8900.0, "E": 200.0e9, "nu": 0.31},
            membrane_resonance_hz=None, membrane_diameter=2 * 4.445e-3,
            membrane_thickness=5.0e-6, membrane_tension=3162.3,
            air_gap=2.077e-5, backplate_diameter=2 * 3.607e-3,
            n_through_holes=6, through_hole_diameter=2 * 5.080e-4,
            through_hole_pcd=2 * 2.032e-3, n_blind_holes=0)
        ebk55 = fbk55.f_res / fbk55.f_res_modal_exact - 1.0
        lbk55 = fbk55._ell / fbk55.a_mem
        assert abs(ebk55 / (-0.5 * (j01_55 * lbk55) ** 2) - 1.0) < 0.15, \
            (f"a) B&K-Nickelfolie: nur die Plattendispersion bleibt "
             f"({ebk55:+.1e})")
        # b) f_res vorgegeben == dieselbe Membran über die Spannung
        pB55 = dict(
            architecture="single", membrane_resonance_hz=8000.0,
            membrane_diameter=12.0e-3, membrane_thickness=5e-6,
            membrane_tension=400.0, air_gap=25e-6,
            backplate_diameter=11.0e-3, backplate_thickness=1.5e-3,
            bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
            delay_length=0.0, cavity_length=4.0e-3,
            cavity_wall_thickness=1.0e-3, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=14e-3, n_through_holes=48,
            through_hole_diameter=0.33e-3, squeeze_model="2d")
        cB55 = MicrophoneCapsule(**pB55)
        TB55 = cB55._membrane_tension_3d()      # wirksame Grundmode 8 kHz
        cBt55 = MicrophoneCapsule(**dict(pB55, membrane_resonance_hz=None,
                                         membrane_tension=TB55))
        eC55 = cB55.C_A_mem / cBt55.C_A_mem - 1.0
        ef55 = cBt55.f_res / 8000.0 - 1.0
        assert abs(eC55) < 1e-10 and abs(ef55) < 1e-10, \
            (f"b) f_res vorgegeben und Spannung vorgegeben müssen dieselbe "
             f"Kapsel sein (C {eC55:+.1e}, f_res {ef55:+.1e})")
        f55 = np.array([20.0])
        eH55 = float(20 * np.log10(abs(cB55.transfer_function(f55)[0]
                                       / cBt55.transfer_function(f55)[0])))
        assert abs(eH55) < 1e-6, f"b) Ausgang gleich ({eH55:+.2e} dB)"
        # c) gegen das 3D-Feld, alter und neuer Massenfaktor
        dB55 = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for me55 in (False, True):
                MicrophoneCapsule._MASS_EXACT = me55
                try:
                    q55 = dict(pB55)
                    h2_55 = MicrophoneCapsule(**q55).transfer_function(f55)[0]
                    q55["squeeze_model"] = "3d"
                    h3_55 = MicrophoneCapsule(**q55).transfer_function(f55)[0]
                finally:
                    MicrophoneCapsule._MASS_EXACT = True
                dB55[me55] = float(20 * np.log10(abs(h2_55 / h3_55)))
        # Kette mit 4/3: C = 0.75·S/(σ·ω0²); exakte Membran gleicher
        # Resonanz: C = (j01²/8)·S/(σ·ω0²) — Verhältnis 6/j01²
        lump55 = 20 * np.log10(6.0 / j01_55 ** 2)
        assert 0.05 < dB55[False] < lump55 + 0.01, \
            (f"c) mit 4/3 lag 2D über 3D, höchstens {lump55:.2f} dB "
             f"({dB55[False]:+.3f} dB)")
        assert abs(dB55[True]) < 0.02, \
            f"c) mit 8/j01²: 2D == 3D ({dB55[True]:+.3f} dB)"
        # d) Hochtongrenzwert der normierten Modenreihe
        N55 = 400
        lim55 = {}
        for rho55 in (0.0, 0.1):
            if rho55 == 0.0:
                z55 = _jn_zeros55(0, N55)
                cm55, mm55 = 1.0 / z55 ** 4, z55 ** 2 / 4.0
                g55 = 1.0
            else:
                md55 = MicrophoneCapsule._ring_eigen(rho55, N55)
                z55 = md55["z"]
                cm55 = md55["I1"] ** 2 / (z55 ** 2 * md55["I2"])
                mm55 = md55["I2"] / md55["I1"] ** 2
                g55 = _ring_compliance_factor(rho55)
            MicrophoneCapsule._RANDSCHICHT = False   # reine Membran
            try:
                mu_r55 = MicrophoneCapsule(
                    center_post_diameter=2 * rho55 * 13e-3,
                    membrane_diameter=26e-3)._mass_factor_rayleigh
            finally:
                MicrophoneCapsule._RANDSCHICHT = True
            for lab55, mu55 in (("8/(z1²g)", 8.0 / (z55[0] ** 2 * g55)),
                                ("Rayleigh", mu_r55)):
                s55 = np.sum(cm55) / cm55[0]
                inv55 = np.sum(mm55[0] / (mu55 * mm55))
                lim55[(rho55, lab55)] = s55 / inv55 * (1.0 - rho55 ** 2)
            tail55 = 4.0 / (np.pi ** 2 * N55)       # Rest von Σ1/z² ab N
            new55 = lim55[(rho55, "8/(z1²g)")]
            assert 1.0 < new55 < 1.0 + 1.1 * tail55, \
                (f"d) ρ = {rho55}: mit 8/(z1²g) muss die Reihe von oben auf "
                 f"den Kolben laufen ({new55:.5f})")
            assert lim55[(rho55, "Rayleigh")] < 1.0, \
                (f"d) ρ = {rho55}: mit dem Rayleigh-Wert lief sie darunter "
                 f"({lim55[(rho55, 'Rayleigh')]:.4f})")
        print(f"Massenfaktor 8/(z1²g): Resonanz der wirksamen Membran "
              f"exakt (Kreis {fr55[0.0][0]:+.0e}, Pfosten 3 mm "
              f"{fr55[3e-3][0]:+.0e}; gegen die Platte nur die Dispersion, "
              f"B&K {ebk55:+.1e}; Rayleigh {fr55[0.0][2]:.4f} "
              f"-> {fr55[0.0][1]:.4f}); f_res == Spannung (C {eC55:+.1e}); "
              f"½\" 2D/3D {dB55[False]:+.3f} -> {dB55[True]:+.3f} dB; "
              f"Modenreihe N = {N55} -> Kolben {lim55[(0.0, '8/(z1²g)')]:.4f} "
              f"(Rayleigh {lim55[(0.0, 'Rayleigh')]:.4f}), Ring ρ = 0.1 "
              f"{lim55[(0.1, '8/(z1²g)')]:.4f} "
              f"({lim55[(0.1, 'Rayleigh')]:.4f})  OK")


@pytest.mark.feld3d
def test_gp61_folienverlust(stand):
    """Gegenprobe 61: Folienverlust — hysteretisch, durch die Vorspannung
    verdünnt, in 2D und 3D gleich."""
    # Bis Gegenprobe 60 stand hier eine Materialgüte Q = 100 als
    # „numerischer Boden", in der 2D-Kette als fester Widerstand
    # R = √(M/C)/Q (an der Resonanz definiert), im 3D-Feld als Masse
    # σ·(1 − j/Q) (mit ω wachsend) — zwei verschiedene Gesetze und keines
    # physikalisch begründet. Bei erzwungener Form (f_res = 300 kHz)
    # verfälschte das den Vergleich der Filmwiderstände um bis zu 8 %
    # (Gegenprobe 60).
    # PHYSIK: die Folie verliert über den komplexen E-Modul E·(1 + j·η)
    # ihres Werkstoffs. Bei Vorspannung ist die Arbeit gegen T in erster
    # Ordnung verlustfrei (die Dehnung aus der Auslenkung ist quadratisch,
    # die Vorspannung statisch); verlustbehaftet ist nur die Biegeenergie
    # (Dissipationsverdünnung, Fedorov et al., PRB 99, 054107, 2019). Der
    # wirksame Verlustfaktor der Mode ist η_m = η·∂ln ω_m²/∂ln D an der
    # exakten Eigenwertgleichung der eingespannten Platte unter Zug,
    # hysteretisch (frequenzunabhängig) auf der Steifigkeit.
    # a) Die Eigenwertgleichung gegen die Randschicht-Asymptotik
    #    q = λ + z²λ² (Kreis) bzw. λ·[C1(z)² + 4/(π²z²ρ)]/I2 + z²λ²
    #    (Ring mit Pfosten): Abweichung O(λ), mit λ fallend; dazu die
    #    komplexe Eigenwertrechnung mit D·(1 + jη) — ihr Im(ω²)/Re(ω²) ist
    #    η·q. Die Grundfrequenz steigt um den Faktor ≈ 1 + λ (Stand-Wert).
    # b) 2D und 3D tragen denselben Verlust: ein großer Verlustfaktor
    #    verschiebt die Phase im steifigkeitsbestimmten Tiefton in beiden
    #    Modellen gleich (auf 3 %). Darüber mischt sich der Filmterm ω·R·C
    #    ein, und die Phasenverschiebung hängt am Gesamtnetz der Modelle.
    # c) Wirkung an realen Kapseln: η_eff ist 0.1…0.7 % von η; selbst das
    #    Zehnfache ändert den Frequenzgang höchstens um 0.05 dB — die
    #    genaue Werkstoffzahl ist belanglos.
    if not _HAS_SCIPY:
        return
    from scipy.optimize import newton
    from scipy.special import ive, jn_zeros, jv, yv

    z = jn_zeros(0, 2)
    D61 = 4.9e9 * (6e-6) ** 3 / (12 * (1 - 0.37**2))        # PET 6 µm
    rs61, a61 = 1390.0 * 6e-6, 12.7e-3
    fehler = []
    for T in (4.0e3, 450.0, 45.0):              # λ ≈ 0.0004 … 0.004
        lam = np.sqrt(D61 / (T * a61 * a61))
        for m in (0, 1):
            q, f = _platten_verduennung(z[m], D61, T, rs61, a61)
            fehler.append((lam, abs(q / (lam + (z[m] * lam) ** 2) - 1.0)))
            assert abs(q / (lam + (z[m] * lam) ** 2) - 1.0) < 2.0 * lam, \
                (f"Kreis λ = {lam:.1e}, Mode {m + 1}: Eigenwertgleichung "
                 f"gegen Randschicht-Asymptotik ({q:.4e})")
            assert abs(f - (1.0 + lam)) < 3.0 * (z[m] * lam) ** 2 + lam ** 2, \
                f"Frequenzfaktor ω/ω_T = {f:.6f} gegen 1 + λ"
    # Ring (Pfosten 3 mm): Randschicht an BEIDEN Rändern
    md61 = MicrophoneCapsule._ring_eigen(3e-3 / a61, 1)
    zr, I2r, rho61 = float(md61["z"][0]), float(md61["I2"][0]), 3e-3 / a61
    c1 = jv(1, zr) * yv(0, zr * rho61) - yv(1, zr) * jv(0, zr * rho61)
    for T in (450.0, 45.0):
        lam = np.sqrt(D61 / (T * a61 * a61))
        q, _ = _platten_verduennung(zr, D61, T, rs61, a61, ri=3e-3)
        q_as = (lam * (c1**2 + 4.0 / (np.pi**2 * zr**2 * rho61)) / I2r
                + (zr * lam) ** 2)
        assert abs(q / q_as - 1.0) < 3.0 * lam, \
            f"Ring λ = {lam:.1e}: gegen Randschicht-Asymptotik ({q / q_as:.4f})"
    # komplexe Eigenwerte: Im(ω²)/Re(ω²) == η·q (Kreis, PET, 45 N/m)
    T, eta61 = 45.0, 0.02

    def _det(x):
        om2 = x * (z[0] / a61) ** 2 * T / rs61
        Dc = D61 * (1 + 1j * eta61)
        s = np.sqrt(T * T + 4 * Dc * rs61 * om2 + 0j)
        al, be = np.sqrt((s - T) / (2 * Dc)), np.sqrt((s + T) / (2 * Dc))
        return (jv(0, al * a61) * be * ive(1, be * a61) / ive(0, be * a61)
                + al * jv(1, al * a61))
    x61 = newton(_det, 1.01 + 1e-5j, tol=1e-13, maxiter=200)
    q61, f61 = _platten_verduennung(z[0], D61, T, rs61, a61)
    assert abs((x61.imag / x61.real) / (eta61 * q61) - 1.0) < 0.01, \
        (f"komplexe Eigenwerte: Im/Re = {x61.imag / x61.real:.4e} gegen "
         f"η·q = {eta61 * q61:.4e}")
    stand.wert("frequenzfaktor_pet_45", f61, "",
               "ω/ω_T der Grundmode, PET 6 µm, 45 N/m, a = 12.7 mm")

    # b) 2D und 3D: derselbe Verlust
    a1 = dict(architecture="single", membrane_resonance_hz=2100.0,
              membrane_diameter=25.4e-3, membrane_thickness=6e-6,
              air_gap=38.1e-6, backplate_diameter=23.9e-3,
              backplate_thickness=3.125e-3, bias_voltage=1.0,
              n_blind_holes=0, rear_network_enabled=True, delay_length=0.0,
              cavity_length=8.0e-3, cavity_wall_thickness=1.5e-3,
              n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
              body_diameter=28e-3, n_through_holes=24,
              through_hole_diameter=0.99e-3)

    def _mat(eta):
        return {"rho": 1390.0, "E": 4.9e9, "nu": 0.37, "eta": eta}
    # Prüfhebel: Werkstoff-η so groß, dass η_eff = 0.01 sichtbar wird
    eta_hebel = 0.01 / MicrophoneCapsule(
        squeeze_model="2d", membrane_material=_mat(1.0), **a1).eta_membrane
    fb = np.array([5.0, 10.0, 20.0])     # rein steifigkeitsbestimmt
    dphi = {}
    for sm in ("2d", "3d"):
        h0 = MicrophoneCapsule(squeeze_model=sm, membrane_material=_mat(0.0),
                               **a1).transfer_function(fb)
        h1 = MicrophoneCapsule(squeeze_model=sm,
                               membrane_material=_mat(eta_hebel),
                               **a1).transfer_function(fb)
        dphi[sm] = np.angle(h1 / h0)
    assert np.all(dphi["2d"] < 0.0), "Verlust muss die Phase nacheilen lassen"
    assert np.max(np.abs(dphi["3d"] / dphi["2d"] - 1.0)) < 0.03, \
        (f"2D und 3D müssen denselben Folienverlust tragen (Δφ 2D "
         f"{np.round(dphi['2d'] * 1e3, 3)}, 3D {np.round(dphi['3d'] * 1e3, 3)}"
         f" mrad)")

    # c) Wirkung an realen Kapseln (K67 mit PET; B&K-artige Nickelfolie)
    fw = np.geomspace(20.0, 20e3, 200)
    ni61 = dict(membrane_material={"rho": 8908.0, "E": 200e9, "nu": 0.31},
                membrane_resonance_hz=None, membrane_diameter=9.0e-3,
                membrane_thickness=5.0e-6, membrane_tension=3160.0,
                air_gap=18.6e-6, backplate_diameter=7.2e-3,
                backplate_thickness=1.029e-3, bias_voltage=200.0,
                architecture="single", n_through_holes=6,
                through_hole_diameter=1.0e-3, through_hole_pcd=3.4e-3,
                n_blind_holes=0, ring_vent_width=0.86e-3,
                ring_vent_length=0.30e-3, rear_network_enabled=True,
                cavity_length=131e-9 / (np.pi * 3.6e-3**2), n_cavity_holes=0,
                fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
                include_diffraction=False, squeeze_model="2d")
    wirk = {}
    for nm, kw, base in (
            ("K67", dict(K67_KWARGS, squeeze_model="2d"),
             MicrophoneCapsule.MATERIALS["pet"]),
            ("Nickel", ni61, MicrophoneCapsule.MATERIALS["nickel"])):
        hh = []
        for fac in (0.0, 1.0, 10.0):
            m = dict(base, eta=base["eta"] * fac)
            cc = MicrophoneCapsule(**{**kw, "membrane_material": m})
            hh.append(20 * np.log10(np.abs(cc.transfer_function(fw))))
        wirk[nm] = (float(np.max(np.abs(hh[1] - hh[0]))),
                    float(np.max(np.abs(hh[2] - hh[1]))))
        assert wirk[nm][1] < 0.05, \
            (f"{nm}: zehnfacher Folienverlust darf den Frequenzgang kaum "
             f"ändern ({wirk[nm][1]:.3f} dB)")
    k67 = MicrophoneCapsule(**dict(K67_KWARGS, squeeze_model="2d"))
    stand.wert("guete_folie_k67", 1.0 / k67.eta_membrane, "",
               "wirksame Güte der Folie, K67 (PET 6 µm)")
    stand.wert("folie_k67_db", wirk["K67"][0], "dB",
               "größte Wirkung des Folienverlusts auf den K67-Frequenzgang")
    print(f"Folienverlust: Eigenwertgleichung gegen Randschicht-Asymptotik "
          f"(Kreis, Ring) und komplexe Eigenwerte (Im/Re = η·q auf 1 %); "
          f"ω/ω_T = {f61:.5f} (1 + λ); 2D/3D gleich (Δφ 5 Hz "
          f"{dphi['2d'][0] * 1e3:.3f} / {dphi['3d'][0] * 1e3:.3f} mrad); "
          f"K67: Güte {1.0 / k67.eta_membrane:.0f}, Wirkung "
          f"{wirk['K67'][0]:.1e} dB (×10: {wirk['K67'][1]:.1e} dB)  OK")


def _platte62(a, ri, D, T, N=20000):
    """Unabhängige statische Lösung der eingespannten Platte unter Zug,
    D·∇⁴w − T·∇²w = p (p = 1 Pa), w = w' = 0 am Rand und am Pfosten.

    Einmal integriert (Schubkraft) für θ = w':
        D·(θ'' + θ'/r − θ/r²) − T·θ = r/2 + C/r,
    θ = 0 an beiden Rändern; C = 0 ohne Pfosten, mit Pfosten so, dass
    ∫θ dr = 0 (w verschwindet an BEIDEN Rändern). Finite Differenzen
    zweiter Ordnung auf N Intervallen. Rückgabe: Volumen je Pa, r, w(r)."""
    from scipy.linalg import solve_banded
    r = np.linspace(ri, a, N + 1)
    h = r[1] - r[0]
    rr = r[1:-1]
    ab = np.zeros((3, rr.size))
    ab[0, 1:] = (D * (1.0 / h**2 + 0.5 / (h * rr)))[:-1]
    ab[1] = D * (-2.0 / h**2 - 1.0 / rr**2) - T
    ab[2, :-1] = (D * (1.0 / h**2 - 0.5 / (h * rr)))[1:]

    def _loes(rhs):
        th = np.zeros(r.size)
        th[1:-1] = solve_banded((1, 1), ab, rhs)
        return th
    th = _loes(0.5 * rr)
    if ri > 0.0:
        th1 = _loes(1.0 / rr)
        th = th - np.trapezoid(th, r) / np.trapezoid(th1, r) * th1
    seg = 0.5 * (th[1:] + th[:-1]) * h
    w = np.concatenate([-np.cumsum(seg[::-1])[::-1], [0.0]])
    return -np.pi * np.trapezoid(r**2 * th, r), r, w


@pytest.mark.feld3d
def test_gp62_randschicht_der_folie(stand):
    """Gegenprobe 62: Randschicht der eingespannten Folie."""
    # Die Folie ist eine Platte unter Zug, D·∇⁴w − T·∇²w = p, eingespannt
    # mit w = w' = 0. Für λ = √(D/(T a²)) ≪ 1 ist sie im Innern Membran;
    # in einer Randschicht der Breite ℓ = √(D/T) biegt sie sich in die
    # Einspannung. Außerhalb der Schicht erfüllt die Membranlösung
    # w = ℓ·∂w/∂r und verschwindet um ℓ vor dem Rand — in erster Ordnung
    # eine Membran mit dem WIRKSAMEN Radius a − ℓ (am Pfosten r_i + ℓ).
    # Das senkt die statische Nachgiebigkeit um ≈ 4λ und hebt die Grund-
    # frequenz um ≈ λ (B&K-Nickel: −2.5 %, +0.65 %). Bis Gegenprobe 61
    # stand die Biegung als parallele Plattenfeder πa⁶/(192D) in der
    # Kette, eine Größe der Ordnung λ²; die Randschicht fehlte.
    # a) STATIK gegen eine unabhängige Lösung der Plattengleichung
    #    (_platte62) und, ohne Pfosten, gegen die geschlossene Form
    #    C = πa⁴/(8T)·[1 − 4λ(I0(1/λ)/I1(1/λ) − 2λ)]: die wirksame Membran
    #    trifft beide bis auf O(λ²·λ) (Kreis 10⁻⁶, Ring 10⁻⁵).
    # b) FREQUENZ gegen den exakten Eigenwert der Platte unter Zug
    #    (_platten_verduennung): es bleibt die Plattendispersion im
    #    Innern, ω²/ω_w² = 1 + k1²ℓ², also ≈ −z1²λ²/2 — nicht gerechnet.
    # c) FORM über der Elektrode gegen die exakte statische Plattenform.
    # d) 2D UND 3D GLEICH: der 3D-Löser spannt seine Membranfelder an den
    #    wirksamen Rändern ein (Gitterende bzw. Schnittzelle mit
    #    Elementgewicht und logarithmischem Leitwert). Statische
    #    Nachgiebigkeit des Membranfelds gegen die Kette, und Ende zu Ende
    #    verschiebt die Randschicht 2D und 3D gleich.
    # e) PULL-IN: Warrens Ā gilt für die wirksame Membran.
    # f) GATTER: über 1 % Abweichung gegen die Platte warnt das Modell,
    #    bei einer Randschicht über ein Viertel der Membranbreite bricht es
    #    ab.
    # g) WIRKUNG (Stand-Werte): B&K-Nickelfolie bei vorgegebener Spannung;
    #    die K67 (PET, vorgegebene Resonanz).
    if not _HAS_SCIPY:
        return
    from scipy.special import ive as _ive62
    j01 = 2.404825557695773
    bk62 = dict(
        membrane_material={"rho": 8900.0, "E": 200.0e9, "nu": 0.31},
        membrane_resonance_hz=None, membrane_diameter=2 * 4.445e-3,
        membrane_thickness=5.0e-6, membrane_tension=3162.3,
        air_gap=2.077e-5, backplate_diameter=2 * 3.607e-3,
        backplate_thickness=0.843e-3, bias_voltage=1.0,
        architecture="single", n_through_holes=6,
        through_hole_diameter=2 * 5.080e-4, through_hole_pcd=2 * 2.032e-3,
        n_blind_holes=0, ring_vent_width=0.838e-3,
        ring_vent_length=3.048e-4, rear_network_enabled=True,
        delay_length=0.0, cavity_length=1.264e-7 / (np.pi * 3.607e-3**2),
        n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
        include_diffraction=False, squeeze_model="2d")
    pet62 = dict(membrane_resonance_hz=None, membrane_tension=45.0,
                 membrane_diameter=25.4e-3, membrane_thickness=6e-6,
                 architecture="single", bias_voltage=1.0,
                 backplate_diameter=23.9e-3, squeeze_model="2d")

    # a) Statik
    stat62 = {}
    for nm, kw in (("B&K", bk62), ("PET", pet62),
                   ("PET, Pfosten 3 mm", dict(pet62,
                                              center_post_diameter=3e-3)),
                   ("Nickel, Pfosten 1 mm", dict(bk62,
                                                 center_post_diameter=1e-3))):
        c = MicrophoneCapsule(**kw)
        T = kw["membrane_tension"]
        V, r, w = _platte62(c.a_mem, c.r_post, c._D_plate, T)
        C_mem = (np.pi * c.a_mem**4 / (8.0 * T)
                 * _ring_compliance_factor(c.rho_post))
        stat62[nm] = (c.C_A_mem / V - 1.0, V / C_mem - 1.0, c, r, w)
        lam = c._ell / c.a_mem
        assert abs(stat62[nm][0]) < 3.0 * lam**2, \
            (f"a) {nm}: die wirksame Membran muss die statische Platte "
             f"treffen ({stat62[nm][0]:+.1e}, Randschicht "
             f"{stat62[nm][1]:+.2%})")
        if c.r_post == 0.0:
            x = 1.0 / lam
            k_ex = 1.0 - 4.0 * lam * (_ive62(0, x) / _ive62(1, x) - 2.0 * lam)
            assert abs(V / (C_mem * k_ex) - 1.0) < 1e-6, \
                (f"a) {nm}: unabhängige Lösung gegen die geschlossene Form "
                 f"({V / (C_mem * k_ex) - 1.0:+.1e})")
            assert abs(c.C_A_mem / (C_mem * k_ex) - 1.0) < 5.0 * lam**3, \
                (f"a) {nm}: Kette gegen die geschlossene Form "
                 f"({c.C_A_mem / (C_mem * k_ex) - 1.0:+.1e})")
    # ohne Randschicht fehlte genau dieser Anteil
    MicrophoneCapsule._RANDSCHICHT = False
    try:
        c_ohne = MicrophoneCapsule(**bk62)
    finally:
        MicrophoneCapsule._RANDSCHICHT = True
    c_bk = stat62["B&K"][2]
    lam_bk = c_bk._ell / c_bk.a_mem
    assert abs(c_ohne.C_A_mem / c_bk.C_A_mem - 1.0 - 4.0 * lam_bk) \
        < 12.0 * lam_bk**2, \
        (f"a) ohne Randschicht ist die B&K-Folie um ≈ 4λ zu nachgiebig "
         f"({c_ohne.C_A_mem / c_bk.C_A_mem - 1.0:+.3%})")

    # b) Frequenz gegen den exakten Eigenwert der Platte
    fq62 = {}
    for nm in stat62:
        c = stat62[nm][2]
        lam_w = c._ell / c._a_w
        z1_w = c._ring_modes()["z_w"][0]
        disp = -0.5 * (z1_w * lam_w) ** 2
        fq62[nm] = (c.f_res / c.f_res_modal_exact - 1.0, disp)
        assert abs(fq62[nm][0] / disp - 1.0) < 0.15, \
            (f"b) {nm}: gegen die Platte bleibt nur die Dispersion "
             f"({fq62[nm][0]:+.2e} statt {disp:+.2e})")
    f_fak_bk = c_bk.f_res / c_ohne.f_res
    assert abs(f_fak_bk - 1.0 - lam_bk) < 3.0 * lam_bk**2, \
        f"b) die Randschicht hebt die Resonanz um ≈ λ ({f_fak_bk:.5f})"

    # c) Form über der Elektrode (Maximum 1) gegen die exakte Plattenform
    #    Punktweise außerhalb der Randschichten (fünf Breiten), wo das
    #    Membranbild gilt; in der Schicht selbst weicht es um O(λ) ab,
    #    das Elektrodenmittel aber nur um O(λ²). Geprüft wird die
    #    Ordnung: der Rest muss von zweiter Ordnung in λ sein (Koeffizient
    #    am Pfosten rund 4, ohne Pfosten unter 1), ohne Randschicht ist er
    #    von erster.
    form62 = {}
    for nm in ("B&K", "PET, Pfosten 3 mm"):
        _, _, c, r, w = stat62[nm]
        u_ex = (r / c.a_mem) ** 2
        w_ex = np.interp(c._es_u, u_ex, w) / np.max(w)
        r_es = c.a_mem * np.sqrt(c._es_u)
        frei = ((r_es > c.r_post + 5.0 * c._ell)
                & (r_es < c.a_mem - 5.0 * c._ell))
        mitte = (np.trapezoid(c._es_phi, c._es_u)
                 / np.trapezoid(w_ex, c._es_u) - 1.0)
        form62[nm] = (float(np.max(np.abs(c._es_phi - w_ex)[frei])), mitte)
        lam = c._ell / c.a_mem
        assert form62[nm][0] < 5.0 * lam**2 and abs(mitte) < 2.0 * lam**2, \
            (f"c) {nm}: Form über der Elektrode gegen die Platte "
             f"({form62[nm][0]:.1e} punktweise, Mittel {mitte:+.1e})")
    # ohne Randschicht lag die Form um O(λ) daneben
    MicrophoneCapsule._RANDSCHICHT = False
    try:
        c_of = MicrophoneCapsule(**bk62)
    finally:
        MicrophoneCapsule._RANDSCHICHT = True
    _, _, _, r, w = stat62["B&K"]
    form_ohne = float(np.max(np.abs(
        c_of._es_phi - np.interp(c_of._es_u, (r / c_of.a_mem) ** 2, w)
        / np.max(w))))
    assert form_ohne > 20.0 * form62["B&K"][0], \
        (f"c) ohne Randschicht weicht die Form deutlich ab "
         f"({form_ohne:.1e} gegen {form62['B&K'][0]:.1e})")

    # d) 2D und 3D gleich
    from scipy.sparse import coo_matrix as _coo62
    from scipy.sparse.linalg import spsolve as _sps62

    def _feld62(kw):
        """Statische Nachgiebigkeit des 3D-Membranfelds allein."""
        cc = MicrophoneCapsule(**dict(kw, squeeze_model="3d"))
        cc._n_phi_3d = 8
        cc._build_3d_geometry()
        g = cc._g3d
        ow, NM = g["n_films"] * g["NF"], g["NM"]
        rr, co, vv = g["static"]
        m = (rr >= ow) & (rr < ow + NM) & (co >= ow) & (co < ow + NM)
        L = _coo62((vv[m].real, (rr[m] - ow, co[m] - ow)),
                   shape=(NM, NM)).tocsc()
        q = np.repeat(g["A_mw"], g["Np"])
        return float(np.dot(q, _sps62(L, q))) / cc.C_A_mem - 1.0
    feld62 = {}
    for nm, kw in (("B&K (Gitterende)", bk62),
                   ("PET, a_bp = a_mem (Schnittzelle)",
                    dict(pet62, backplate_diameter=25.4e-3)),
                   ("PET, Pfosten 3 mm (Schnittzelle)",
                    dict(pet62, center_post_diameter=3e-3))):
        feld62[nm] = _feld62(dict(kw, bias_voltage=1e-6))
        assert abs(feld62[nm]) < 2e-3, \
            (f"d) {nm}: 3D-Membranfeld gegen die Kette "
             f"({feld62[nm]:+.1e})")
    f20 = np.array([20.0])
    ende62 = {}
    for nm, kw in (("B&K", bk62),
                   ("PET, a_bp = a_mem", dict(pet62,
                                             backplate_diameter=25.4e-3))):
        dd = []
        for rand in (False, True):
            MicrophoneCapsule._RANDSCHICHT = rand
            try:
                h2 = MicrophoneCapsule(**kw).transfer_function(f20)[0]
                h3 = MicrophoneCapsule(**dict(kw, squeeze_model="3d")) \
                    .transfer_function(f20)[0]
            finally:
                MicrophoneCapsule._RANDSCHICHT = True
            dd.append((float(20 * np.log10(abs(h3 / h2))),
                       float(20 * np.log10(abs(h2)))))
        ende62[nm] = (dd[1][0] - dd[0][0], dd[1][1] - dd[0][1])
        assert abs(ende62[nm][0]) < 0.005, \
            (f"d) {nm}: die Randschicht muss 2D und 3D gleich verschieben "
             f"(3D−2D ändert sich um {ende62[nm][0]:+.4f} dB, die "
             f"Randschicht selbst {ende62[nm][1]:+.3f} dB)")

    # e) Warren mit dem wirksamen Radius (lochfreie volle Elektrode)
    cw = MicrophoneCapsule(
        membrane_resonance_hz=None, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=26e-3, backplate_thickness=4e-3, bias_voltage=1.0,
        architecture="single", n_through_holes=0, n_blind_holes=0,
        rear_network_enabled=False, squeeze_model="1d",
        include_diffraction=False)
    A_w = (cw.U_pullin**2 * cw._a_w**2 * EPS0
           / (2.0 * cw._st["tension"] * cw.h_gap**3))
    assert abs(A_w / 0.789 - 1.0) < 2e-3, \
        f"e) Warren mit dem wirksamen Radius ({A_w:.4f})"
    A_a = A_w * (cw.a_mem / cw._a_w) ** 2

    # f) Gatter
    steif = dict(membrane_resonance_hz=None, membrane_tension=5.0,
                 membrane_diameter=10e-3, membrane_thickness=25e-6,
                 membrane_material="nickel")
    lam_s = np.sqrt(200e9 * 25e-6**3 / (12 * (1 - 0.31**2)) / 5.0) / 5e-3
    with pytest.raises(ValueError):
        MicrophoneCapsule(**steif)
    with pytest.warns(UserWarning, match="Biegesteife Folie"):
        MicrophoneCapsule(**dict(steif, membrane_tension=200.0,
                                 membrane_thickness=12e-6))

    # g) Wirkung
    stand.wert("bk_resonanzfaktor", f_fak_bk, "",
               "B&K-Nickelfolie, Resonanz mit/ohne Randschicht (Spannung vorgegeben)")
    stand.wert("bk_tiefton_db", ende62["B&K"][1], "dB",
               "B&K-Nickelfolie, Tiefton mit/ohne Randschicht (Spannung vorgegeben)")
    k67r = MicrophoneCapsule(**dict(K67_KWARGS, squeeze_model="2d"))
    MicrophoneCapsule._RANDSCHICHT = False
    try:
        k67o = MicrophoneCapsule(**dict(K67_KWARGS, squeeze_model="2d"))
    finally:
        MicrophoneCapsule._RANDSCHICHT = True
    k67_db = float(20 * np.log10(abs(k67r.transfer_function(f20)[0]
                                     / k67o.transfer_function(f20)[0])))
    stand.wert("k67_tiefton_db", k67_db, "dB",
               "K67 (PET, Resonanz vorgegeben), Tiefton mit/ohne Randschicht")
    print(f"Randschicht der Folie: Statik gegen die Platte "
          + ", ".join(f"{k} {v[0]:+.0e}" for k, v in stat62.items())
          + f" (Randschicht {stat62['B&K'][1]:+.2%} bei B&K); Frequenz "
          f"gegen die Platte nur die Dispersion (B&K "
          f"{fq62['B&K'][0]:+.1e}, Pfosten 3 mm "
          f"{fq62['PET, Pfosten 3 mm'][0]:+.1e}), Resonanz ×{f_fak_bk:.5f} "
          f"(1 + λ, λ = {lam_bk:.4f}); Form über der Elektrode "
          f"{form62['B&K'][0]:.0e} (ohne {form_ohne:.0e}); 3D-Feld gegen die "
          f"Kette " + ", ".join(f"{v:+.0e}" for v in feld62.values())
          + f"; 2D/3D-Verschiebung gleich auf "
          f"{max(abs(v[0]) for v in ende62.values()):.4f} dB (B&K Tiefton "
          f"{ende62['B&K'][1]:+.3f} dB); Warren Ā = {A_w:.4f} mit a − ℓ "
          f"({A_a:.4f} mit a); K67 {k67_db:+.3f} dB; Gatter (λ = "
          f"{lam_s:.2f} bricht ab)  OK")
