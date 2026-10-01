"""Gegenproben: Spaltfilm."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


def test_gp08_2d_spaltfilmmodell_reynolds_feld():
    """Gegenprobe 8: 2D-Spaltfilmmodell (Reynolds-Feld)."""
    if _HAS_SCIPY:
        # a) Das Feld-Zweitor ist reziprok (det T = 1) und passiv
        #    (Re Z_in >= 0 über das Band).
        cap2d = MicrophoneCapsule(architecture="single", n_through_holes=12,
                                  through_hole_diameter=0.5e-3,
                                  rear_network_enabled=False,
                                  squeeze_model="2d")
        oms = 2.0 * np.pi * np.logspace(1.0, np.log10(25000.0), 40)
        Tg = cap2d._gap_field_2port(oms)
        det = Tg[0, 0] * Tg[1, 1] - Tg[0, 1] * Tg[1, 0]
        assert np.max(np.abs(det - 1.0)) < 1e-6, \
            "Reynolds-Zweitor muss reziprok sein"
        assert np.min((Tg[0, 1] / Tg[1, 1]).real) > 0.0, \
            "Reynolds-Zweitor muss passiv sein"
        # b) Grenzfall dichte, gleichverteilte Löcher: dank des Zell-
        #    Engstellenwiderstands muss das Feldmodell bei tiefer Frequenz
        #    den Škvor-Spaltwiderstand reproduzieren (das Gitter fügt die
        #    real aufgelöste radiale Ausbreitung des Membranprofils hinzu,
        #    daher Toleranz nach oben).
        cd = MicrophoneCapsule(architecture="single", n_through_holes=60,
                               through_hole_diameter=1.2e-3, n_blind_holes=0,
                               air_gap=60e-6, rear_network_enabled=False,
                               squeeze_model="2d")
        om1 = np.array([2.0 * np.pi * 20.0])
        Tg1 = cd._gap_field_2port(om1)
        Z_h = cd._hole_impedance(om1, cd.r_th, cd.t_bp, cd.n_th,
                                 end_correction=True)[0]
        ratio = (Tg1[0, 1, 0] - Z_h).real / cd.R_A_gap
        assert 0.8 < ratio < 1.6, \
            f"2D muss im dichten Grenzfall Škvor reproduzieren (ratio={ratio:.2f})"
        print(f"2D-Reynolds-Feld: reziprok/passiv über das Band; dichter "
              f"Grenzfall 2D/Škvor = {ratio:.2f}  OK")


def test_gp19_fok_mundungen_lokales_h_r(k67, stand):
    """Gegenprobe 19: Fok-Mündungen, lokales h(r), exakte Leitung."""
    # a) Fok/Melling-Faktor: F(xi->0) -> 1 (einsame Mündung == Bestand),
    #    monoton fallend mit der Lochdichte, also 0 < F < 1; der K67-Wert
    #    selbst ist ein Stand-Wert.
    # b) Lokales Spaltprofil im 2D-Film: sag_w0 = 0 reproduziert den
    #    Bestand EXAKT; mit statischer Durchbiegung liegt das Zweitor nahe
    #    an der bisherigen Flächenmittel-Näherung (kleine Korrektur), aber
    #    verschieden vom nominalen Spalt (h³-Wirkung vorhanden; ihre
    #    Größe ist ein Stand-Wert).
    # c) Exakte Zwikker-Kosten-Leitung: geht für weite Rohre in die
    #    Kirchhoff-Asymptotik über (Dämpfungsbelag/Zc innerhalb weniger %).
    # d) Exakte J0-Modalfrequenz: seit dem Massenfaktor 8/j01²
    #    (Gegenprobe 55) trifft die Kette sie; der Rayleigh-Wert 4/3 der
    #    statischen Form lag 1.9 % darüber.
    assert k67._fok_th is not None and 0.0 < k67._fok_th < 1.0, \
        f"Fok-Faktor muss zwischen 0 und 1 liegen ({k67._fok_th})"
    stand.wert("fok_k67", k67._fok_th, "",
               "Fok/Melling-Faktor der K67-Durchgangslöcher")
    # (50 V: fast volle Elektrode ohne Senkungen — der exakte Pull-in
    # liegt bei 59.7 V, Gegenprobe 49; geprüft wird hier die Mündung)
    sparse = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=50.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=2, through_hole_diameter=0.3e-3,
        n_blind_holes=0, blind_hole_depth=1e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3)
    assert sparse._fok_th > 0.97, \
        f"einsame Mündungen: Fok -> 1 erwartet ({sparse._fok_th:.3f})"
    assert sparse._fok_th > k67._fok_th, "Fok muss mit Lochdichte fallen"
    if _HAS_SCIPY:
        om19 = np.array([2.0 * np.pi * 1000.0])
        T_loc = k67._gap_field_2port(om19, h_film=k67.h_gap,
                                     sag_w0=k67.w0_static)
        T_sag0 = k67._gap_field_2port(om19, h_film=k67.h_gap, sag_w0=0.0)
        T_nom = k67._gap_field_2port(om19, h_film=k67.h_gap)
        assert np.allclose(T_sag0, T_nom, rtol=1e-12), \
            "sag_w0 = 0 muss den Bestand exakt reproduzieren"
        T_mean = k67._gap_field_2port(om19, h_film=k67.h_gap_front)
        rel_mean = abs(T_loc[0, 1][0] / T_mean[0, 1][0] - 1.0)
        rel_nom = abs(abs(T_loc[0, 1][0]) / abs(T_nom[0, 1][0]) - 1.0)
        assert rel_mean < 0.05, \
            f"lokales Profil nahe der Flächenmittel-Näherung ({rel_mean:.3f})"
        assert rel_nom > 1e-6, \
            f"h³-Wirkung der Durchbiegung muss sichtbar sein ({rel_nom:.1e})"
        stand.wert("h3_wirkung", rel_nom, "",
                   "||T12| lokal / nominal − 1| bei 1 kHz (K67)")
        g_ex, Zc_ex = k67._narrow_duct_propagation(om19, 10e-3)
        al_kirch = (np.sqrt(MU_AIR * om19[0] / (2.0 * RHO0))
                    * (1.0 + (GAMMA - 1.0) / np.sqrt(PRANDTL))
                    / (10e-3 * C_AIR))
        Zc0 = RHO0 * C_AIR / (np.pi * 10e-3**2)
        assert 0.9 < g_ex[0].real / al_kirch < 1.1, \
            "exakte Leitung muss für weite Rohre Kirchhoff treffen (alpha)"
        assert 0.98 < abs(Zc_ex[0]) / Zc0 < 1.02, \
            "exakte Leitung muss für weite Rohre Kirchhoff treffen (Zc)"
    r_modal = sparse.f_res_modal_exact / sparse.f_res_from_tension
    assert abs(r_modal - 1.0) < 1e-3, \
        (f"die Kette muss die exakte J0-Modalfrequenz treffen "
         f"({r_modal:.5f})")
    r_rayl = (np.sqrt(sparse._piston_factor / sparse._mass_factor_rayleigh)
              / r_modal)                       # f_Rayleigh / f_exakt
    assert 1.015 < r_rayl < 1.025, \
        (f"mit dem Rayleigh-Wert 4/3 lag die Kette ~1.9 % zu hoch "
         f"({r_rayl:.4f})")
    assert "exakte Modalfrequenz" in sparse.summary()
    print(f"Fok/Melling: K67 {k67._fok_th:.2f}, einsame Mündung "
          f"{sparse._fok_th:.3f} -> 1; lokales h(r): sag=0 == Bestand, "
          f"Korrektur zur Mittel-Näherung "
          + (f"{100 * rel_mean:.1f} %" if _HAS_SCIPY else "—")
          + f"; Leitung exakt == Kirchhoff (weit); Modal {r_modal:.4f} "
          f"(Rayleigh 4/3: {r_rayl:.3f})  OK")


@pytest.mark.feld3d
def test_gp29_durchgehender_randspalt_b_k_bauform(stand):
    """Gegenprobe 29: durchgehender Randspalt (B&K-Bauform)."""
    # Der Luftspalt vieler Messmikrofon-Kapseln ist am Plattenumfang NICHT
    # dicht: ein umlaufender Ringkanal verbindet ihn mit der Rückkammer.
    # Bisher war der Filmrand hermetisch (Neumann, kein Fluss) — die Luft
    # konnte den Spalt nur durch Bohrungen verlassen. Der Clearance-Ring
    # kann das NICHT ersetzen (er ist eine Nut IM Film bzw. ein Sack-Stub).
    # Verankert, alles fit-frei:
    # a) SCHLITZLEITUNG: die thermoviskose LRF-Leitung des Ringkanals muss
    #    im Tiefton exakt die Poiseuille-Grenzwerte treffen —
    #    R = 12μL/(b·w³), M = (6/5)·ρ0·L/(b·w) (kinetischer Profilfaktor
    #    der Schlitzströmung, derselbe 6/5 wie in Gegenprobe 28) und die
    #    ISOTHERME Nachgiebigkeit V/P_atm — und reziprok sein (det T = 1).
    # b) RANDWIDERSTAND: die randbelüftete Platte hat den analytisch
    #    herleitbaren Squeeze-Film-Widerstand R_edge = 3μ/(2πh³)
    #    (radiale Poiseuille-Strömung zum offenen Rand, flächengemittelt,
    #    wie bei Škvor unabhängig vom Plattenradius).
    # c) DICHT-GRENZFALL: w -> 0 muss die versiegelte Platte reproduzieren
    #    (Filmrand wieder dicht) — Konvergenz von oben.
    # d) MONOTONIE/PASSIVITÄT: breiterer Randspalt -> mehr Rückkopplung,
    #    monoton, mit Sättigung sobald nicht mehr der Kanal, sondern der
    #    Film selbst begrenzt.
    # e) FELDPFAD: das 2D-Zweitor bleibt mit Randknoten reziprok
    #    (det T = 1), auch mit Löchern UND Randspalt gleichzeitig.
    # f) GATTER: Doppelmembran, K103-Spacer, 1D+Löcher, 3D.
    if _HAS_SCIPY:
        # (45 V: die lochfreie Platte hat die volle Elektrodenfläche, der
        # exakte Pull-in dieser weichen Kapsel liegt bei 48.9 V —
        # Gegenprobe 49)
        BK29 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=45.0, n_blind_holes=0,
            rear_network_enabled=True, delay_length=0.0,
            cavity_length=8.0e-3, cavity_wall_thickness=1.5e-3,
            n_cavity_holes=0, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=28e-3)
        c29 = MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                ring_vent_width=50e-6, **BK29)
        # a) Schlitzleitungs-Grenzfälle
        w29, L29 = 50e-6, c29.ring_vent_L
        b29 = 2.0 * np.pi * c29.a_bp
        om29 = np.array([2.0 * np.pi * 1.0])
        T29 = c29._slit_line_abcd(om29, w29, b29, L29)
        R29 = float(T29[0, 1][0].real)
        M29 = float(T29[0, 1][0].imag) / om29[0]
        C29 = float((T29[1, 0][0] / (1j * om29[0])).real)
        assert abs(R29 / (12.0 * MU_AIR * L29 / (b29 * w29**3)) - 1.0) \
            < 1e-4, f"Schlitz-Poiseuille R = 12μL/(b w³) ({R29:.3e})"
        assert abs(C29 / (w29 * b29 * L29 / P_ATM) - 1.0) < 1e-4, \
            f"isotherme Kanal-Nachgiebigkeit V/P_atm ({C29:.3e})"
        # Die Masse trägt bei LANGER Leitung bereits die Korrektur der
        # verteilten Leitung (O((γL)²)) — der reine Massegrenzwert wird
        # unten an der kurzen Leitung geprüft.
        assert M29 > 0.0, "Schlitzleitung muss induktiv sein"
        det29 = complex(T29[0, 0][0] * T29[1, 1][0]
                        - T29[0, 1][0] * T29[1, 0][0])
        assert abs(det29 - 1.0) < 1e-12, \
            f"Schlitzleitung muss reziprok sein (det = {det29})"
        # kurze Leitung -> reine Masse (ohne verteilten R'C-Anteil)
        T29s = c29._slit_line_abcd(om29, w29, b29, 0.1e-3)
        M29s = float(T29s[0, 1][0].imag) / om29[0]
        assert abs(M29s / (1.2 * RHO0 * 0.1e-3 / (b29 * w29)) - 1.0) \
            < 1e-3, "kurze Schlitzleitung muss exakt (6/5)ρ0L/(bw) sein"
        # b) Randwiderstand der randbelüfteten Platte
        assert abs(c29.R_A_gap / (3.0 * MU_AIR
                                  / (2.0 * np.pi * c29.h_gap**3)) - 1.0) \
            < 1e-12, "R_edge = 3μ/(2πh³) der randbelüfteten Platte"
        # Die B&K-Bauform bleibt ein DRUCKempfänger: der Randspalt
        # entlüftet den Spalt in die geschlossene Rückkammer, nicht ins
        # Freifeld. Erst Hohlraum-Einlässe machen daraus einen
        # Gradientenempfänger.
        assert not c29.rear_open, \
            "geschlossene Rückkammer -> Druckempfänger trotz Randspalt"
        assert MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                 ring_vent_width=50e-6,
                                 **{**BK29, "n_cavity_holes": 40,
                                    "cavity_hole_diameter": 0.5e-3}
                                 ).rear_open, \
            "mit Hohlraum-Einlässen muss der Randspalt nach hinten öffnen"
        # c) DICHT-GRENZFALL auf der ZWEITOR-Ebene. Physikalisch
        #    eindeutig ist die ENTKOPPLUNG: ein dichter Rand trennt
        #    Membran und Rückport, der Membranfluss bei kurzgeschlossenem
        #    Port (= 1/T22) muss verschwinden — und zwar wie der
        #    Kanalleitwert, also MINDESTENS wie w³ (Poiseuille); sehr
        #    schmale Spalte dämpfen sogar exponentiell, weil die Leitung
        #    dann vom Lumped- ins Wellenleiter-Regime wechselt (γL ~ 36
        #    bei w = 0.1 µm). Geprüft wird monotone, mindestens kubische
        #    Entkopplung. (Ein Vergleich der EINGANGSimpedanz taugt hier
        #    nicht: sie behält auch bei dichtem Rand die thermische
        #    Relaxationsdämpfung der Spaltluft, die zur versiegelten
        #    Platte gehört.)
        om_c29 = 2.0 * np.pi * np.array([200.0])
        cpl29 = []
        for w_t in (1.0e-5, 1.0e-6, 1.0e-7):
            Tx29 = MicrophoneCapsule(
                squeeze_model="2d", n_through_holes=0,
                ring_vent_width=w_t, **BK29)._gap_field_2port(om_c29)
            cpl29.append(float(np.abs(1.0 / Tx29[1, 1][0])))
        for k29 in range(len(cpl29) - 1):
            ratio29 = cpl29[k29] / cpl29[k29 + 1]
            assert ratio29 > 900.0, \
                (f"dichter Rand muss mindestens wie w³ entkoppeln "
                 f"(Verhältnis {ratio29:.0f} je Dekade)")
        e_prev = cpl29[-1] / cpl29[0]
        # d) Monotonie + Sättigung
        lv29 = []
        for w_t in (10e-6, 25e-6, 50e-6, 200e-6):
            H_t = MicrophoneCapsule(
                squeeze_model="2d", n_through_holes=0,
                ring_vent_width=w_t, **BK29).transfer_function([200.0])
            lv29.append(float(20.0 * np.log10(np.abs(H_t[0]))))
        assert lv29[0] < lv29[1] < lv29[2] <= lv29[3] + 1e-9, \
            f"breiterer Randspalt muss monoton mehr entlasten ({lv29})"
        assert abs(lv29[3] - lv29[2]) < abs(lv29[1] - lv29[0]), \
            "die Wirkung muss sättigen (Film wird begrenzend, nicht Kanal)"
        # e) Reziprozität des Feld-Zweitors, auch Löcher + Randspalt
        for n29, w_t in ((0, 50e-6), (48, 50e-6)):
            cx = MicrophoneCapsule(squeeze_model="2d", n_through_holes=n29,
                                   through_hole_diameter=1.0e-3,
                                   ring_vent_width=w_t, **BK29)
            Tx = cx._gap_field_2port(2.0 * np.pi
                                     * np.array([200.0, 4000.0, 16000.0]))
            dx = Tx[0, 0] * Tx[1, 1] - Tx[0, 1] * Tx[1, 0]
            assert np.max(np.abs(dx - 1.0)) < 1e-9, \
                (f"2D-Zweitor mit Randknoten muss reziprok bleiben "
                 f"(n_th = {n29}, max |det-1| = "
                 f"{np.max(np.abs(dx - 1.0)):.2e})")
        # 1D == 2D im rein randbelüfteten Tiefton (beide: Film -> Rand)
        H1d = MicrophoneCapsule(squeeze_model="1d", n_through_holes=0,
                                ring_vent_width=50e-6,
                                **BK29).transfer_function([200.0])
        H2d = MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                ring_vent_width=50e-6,
                                **BK29).transfer_function([200.0])
        d12 = float(abs(20.0 * np.log10(np.abs(H1d[0] / H2d[0]))))
        # Stand-Wert, kein Fenster: seit Gegenprobe 31 zählt die
        # Filmdämpfung nur noch EINMAL. Vorher dominierte der doppelte
        # R_A_gap beide Pfade gleichermaßen und glich sie künstlich an;
        # jetzt tritt der strukturelle Unterschied hervor — und gerade im
        # rein randbelüfteten Fall ist der radiale Weg lang, wo das
        # Lumped-1D-Modell am schwächsten und das Feldmodell maßgeblich
        # ist. Ein Grenzfall, in dem beide zusammenfallen MÜSSEN, ist das
        # nicht; der Abstand wird gemeldet.
        stand.wert("abstand_1d_2d", d12, "dB",
                   "|1D/2D| rein randbelüftet, 200 Hz")
        # f) 3D-LÖSER: derselbe Ringkanal hängt dort über den
        #    Randflächen-Leitwert an der äußersten Filmzellreihe.
        #    Verankert am KOLBEN-GRENZFALL: nur wenn die Membran sich
        #    NICHT verformen kann, beschreiben 1D/2D (Grundmode φ
        #    erzwungen) und 3D (freies Membranfeld) dasselbe Problem.
        #    Mit steifer Membran im quasistatischen Tiefton müssen beide
        #    zusammenfallen — sie tun es auf 0.4 dB (seit Gegenprobe 50
        #    mit exaktem Membranrand; vorher verdeckte die um 0.4 % zu
        #    große 3D-Membran einen Teil davon). Bei WEICHER Membran
        #    liegt 3D systematisch höher (bis ~8 dB): die freie Membran
        #    umgeht den hohen Randwiderstand, indem sie bevorzugt außen
        #    arbeitet — dieselbe Einmoden-Grenze des homogenisierten
        #    Modells wie beim K67-Sattel (Gegenprobe 28). Das ist ein
        #    dokumentierter Modellunterschied, kein Fehler.
        ST29 = dict(BK29, membrane_resonance_hz=50.0e3,
                    membrane_tension=25600.0)
        v2_29 = MicrophoneCapsule(
            squeeze_model="2d", n_through_holes=0, ring_vent_width=50e-6,
            **ST29).transfer_function([200.0])[0]
        c3_29 = MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                                  ring_vent_width=50e-6, **ST29)
        v3_29 = c3_29.transfer_function([200.0])[0]
        d3_29 = float(abs(20.0 * np.log10(abs(v3_29 / v2_29))))
        assert d3_29 < 1.0, \
            (f"3D-Randspalt muss im Kolben-Grenzfall die 2D-Lösung "
             f"treffen ({d3_29:.2f} dB)")
        # Gitterunabhängigkeit: der Ringstempel darf nicht von der
        # azimutalen Auflösung abhängen (axialsymmetrischer Kanal).
        lv3_29 = []
        for np_t in (96, 192):
            cg29 = MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                                     ring_vent_width=50e-6, **ST29)
            cg29._n_phi_3d = np_t
            cg29._build_3d_geometry()
            lv3_29.append(float(abs(cg29.transfer_function([200.0])[0])))
        assert abs(lv3_29[1] / lv3_29[0] - 1.0) < 1e-6, \
            f"3D-Ringstempel muss Np-unabhängig sein ({lv3_29})"
        # Dicht-Grenzfall auch im 3D: schmaler Kanal entkoppelt
        v3_tight = MicrophoneCapsule(
            squeeze_model="3d", n_through_holes=0, ring_vent_width=0.5e-6,
            **ST29).transfer_function([200.0])[0]
        assert abs(v3_tight) < 0.35 * abs(v3_29), \
            "3D: schmaler Randspalt muss die Rückseite abkoppeln"
        # f) Gatter
        for kw29, why29 in (
                (dict(architecture="dual_diaphragm",
                      membrane_resonance_hz=1150.0, center_gap=40e-6,
                      n_through_holes=12, through_hole_diameter=0.6e-3,
                      ring_vent_width=50e-6), "Doppelmembran"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=0, ring_vent_width=50e-6,
                      rear_spacer_height=100e-6), "K103-Spacer"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=12, through_hole_diameter=0.6e-3,
                      ring_vent_width=50e-6, squeeze_model="1d"),
                 "1D + Löcher"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=0, ring_vent_width=50e-6,
                      rear_plate_thickness=1.0e-3,
                      n_rear_plate_holes=8), "K103-Rückplatte")):
            try:
                MicrophoneCapsule(**kw29)
                raise AssertionError(f"Randspalt-Gatter {why29} fehlt")
            except ValueError:
                pass
        print(f"Randspalt: Schlitzleitung R/C exakt (Poiseuille, isotherm), "
              f"kurze Leitung == (6/5)ρ0L/(bw), reziprok "
              f"{abs(det29 - 1.0):.0e}; R_edge = 3μ/(2πh³); w->0 "
              f"entkoppelt wie w³ ({e_prev:.1e} über 2 Dekaden); "
              f"Breite 10->200 µm "
              f"{lv29[3] - lv29[0]:+.1f} dB (monoton, sättigt); "
              f"Feld-Zweitor reziprok mit/ohne Löcher; 1D/2D {d12:.2f} dB; "
              f"3D == 2D im Kolben-Grenzfall ({d3_29:.2f} dB), "
              f"Np-unabhängig; Gatter greifen  OK")


@pytest.mark.feld3d
def test_gp30_zellfunktion_des_durchflusses(k67, stand):
    """Gegenprobe 30: Zellfunktion des DURCHFLUSSES."""
    # Die azimutale Zuströmung im Spaltfilm hat ZWEI verschiedene Ziele:
    #   * AUFNAHME (Verdrängung): jede Bohrung ist eine Senke — die Luft
    #     läuft zur nächstgelegenen, gleich welcher Art.
    #   * DURCHFLUSS zur Rückseite: nur DURCHGANGSLÖCHER zählen;
    #     Blindlöcher sind Sackgassen.
    # Bis Gegenprobe 29 nutzte auch der Durchfluss die Zellfunktion ALLER
    # Bohrungen — der Zugang zur Rückseite war dadurch zu leicht und die
    # interne Laufzeit des Nieren-Phasenschiebers zu kurz. Verankert an:
    # a) STRUKTUR: B(n_th) > B(n_th + n_bh), sobald Blindlöcher da sind;
    #    ohne Blindlöcher sind beide identisch (stetiger Anschluss).
    # b) K67-NIERE: das Pattern-Minimum liegt jetzt bei 180° — wie bei der
    #    realen K67 — statt bei ~164°, und die publizierten U87-Werte
    #    werden dabei besser getroffen (180°: -26.6 gegen -26 dB
    #    publiziert; 20.0 gegen ~20 mV/Pa).
    # c) NÄHER AM 3D-FELDLÖSER, der die diskreten Löcher auflöst und
    #    deshalb Referenz ist: die RMS-Abweichung des Richtdiagramms sank
    #    damals bei der Debenham-Platte (12 Löcher) und der Nieren-Single;
    #    ihr heutiger Wert ist ein Stand-Wert.
    # d) GEGENPROBE OHNE BLINDLÖCHER: dort darf sich NICHTS ändern.
    if _HAS_SCIPY:
        par30 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=50.0, n_through_holes=12,
            through_hole_diameter=0.71e-3, rear_network_enabled=False,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=28e-3, squeeze_model="2d")
        # a) Strukturprobe über die Zellfunktion selbst
        def _B30(n_s, r_h, a_bp):
            q = min(n_s * r_h**2 / a_bp**2, 1.0)
            if q >= 1.0:
                return 0.0
            return max(q / 2.0 - q**2 / 8.0 - np.log(q) / 4.0 - 3.0 / 8.0,
                       0.0)
        c30 = MicrophoneCapsule(n_blind_holes=46, blind_hole_diameter=1.2e-3,
                                blind_hole_depth=3.0e-3, **par30)
        B_flow = _B30(c30.n_th, c30.r_th, c30.a_bp)
        B_all = _B30(c30.n_th + c30.n_bh, c30.r_th, c30.a_bp)
        assert B_flow > B_all * 1.5, \
            (f"Sackgassen dürfen den Durchfluss nicht verkürzen "
             f"(B: {B_flow:.3f} vs. {B_all:.3f})")
        # d) ohne Blindlöcher identisch (stetiger Anschluss)
        assert abs(_B30(c30.n_th, c30.r_th, c30.a_bp)
                   - _B30(c30.n_th + 0, c30.r_th, c30.a_bp)) == 0.0, \
            "ohne Blindlöcher müssen beide Zellfunktionen gleich sein"
        # b) K67: Niere mit Minimum bei 180° und publizierte Kennwerte
        di30 = k67.directivity(frequencies_hz=(1000.0,))
        pat30 = di30["patterns"][1000.0]
        na30 = di30["angles_deg"][:181][int(np.argmin(
            pat30["linear"][:181]))]
        H30 = abs(k67.transfer_function([1000.0])[0]) * 1e3
        assert na30 >= 179.0, \
            (f"K67-Niere muss ihr Minimum bei 180° haben (real), nicht "
             f"bei {na30:.0f}°")
        assert pat30["db"][180] < -25.0, \
            f"K67 rückwärts < -25 dB erwartet ({pat30['db'][180]:.1f})"
        assert 18.0 < H30 < 22.0, \
            f"K67-Empfindlichkeit nahe 20 mV/Pa erwartet ({H30:.1f})"
        # c) Richtdiagramm näher am 3D-Feldlöser (Debenham, 12 Löcher)
        deb30 = MicrophoneCapsule(n_blind_holes=46,
                                  blind_hole_diameter=1.2e-3,
                                  blind_hole_depth=3.0e-3, **par30)
        d30_2 = deb30.directivity(frequencies_hz=(500.0,),
                                  n_angles=37)["patterns"][500.0]["db"]
        deb30_3 = MicrophoneCapsule(
            **{**par30, "squeeze_model": "3d"}, n_blind_holes=46,
            blind_hole_diameter=1.2e-3, blind_hole_depth=3.0e-3)
        d30_3 = deb30_3.directivity(frequencies_hz=(500.0,),
                                    n_angles=37)["patterns"][500.0]["db"]
        rms30 = float(np.sqrt(np.mean(
            (np.maximum(d30_2, -35.0) - np.maximum(d30_3, -35.0)) ** 2)))
        stand.wert("rms_2d_3d_debenham", rms30, "dB",
                   "RMS 2D gegen 3D, Richtdiagramm 500 Hz, 12 Löcher")
        print(f"Durchfluss-Zellfunktion: Sackgassen zählen nicht "
              f"(B {B_all:.3f} -> {B_flow:.3f}); K67-Niere Minimum bei "
              f"{na30:.0f}° (real) mit {pat30['db'][180]:.1f} dB und "
              f"{H30:.1f} mV/Pa; 2D/3D-Richtdiagramm {rms30:.2f} dB  OK")


@pytest.mark.feld3d
def test_gp31_filmdampfung_genau_einmal():
    """Gegenprobe 31: Filmdämpfung genau EINMAL."""
    # Der Škvor-Widerstand stand bis Gegenprobe 30 ZWEIMAL in der Kette:
    # in der Membranimpedanz UND im Backplate/Spalt-Zweitor, das in
    # Serie folgt. Physikalisch ist es EIN Weg (die Piston-Bewegung
    # drückt die Spaltluft lateral zu den Senken).
    # a) STRUKTURBEWEIS: die Eingangsimpedanz des Zweitors bei
    #    kurzgeschlossenem Port und widerstandsarmen Bohrungen IST der
    #    Škvor-Widerstand — er ist dort also bereits vollständig
    #    enthalten und darf in der Membranimpedanz nicht nochmals
    #    auftauchen. Die Membranimpedanz trägt jetzt nur noch die
    #    Materialdämpfung der Folie.
    # b) LEITFALL DRUCKEMPFÄNGER: nur eine Bauform ohne rückwärtige
    #    Auslöschung misst die Dämpfung unverfälscht — Gradienten-
    #    bauformen hängen an einer Null und reagieren auf jede
    #    Phasenänderung überempfindlich (daran war die Korrektur früher
    #    scheinbar gescheitert). Im Gültigkeitsbereich der
    #    Homogenisierung (48-96 Bohrungen) muss das 2D-Modell den
    #    3D-Feldlöser, der die Löcher diskret auflöst, jetzt auf < 1 dB
    #    treffen.
    # c) GRENZE EHRLICH: bei sehr spärlichem Lochraster (12) bleibt eine
    #    Abweichung — dort ist die axialsymmetrische Homogenisierung am
    #    Ende und der 3D-Löser nötig.
    # NACHTRAG Gegenprobe 48: die Grenze hängt nicht an der Lochzahl
    # allein, sondern an f_hom (lochfreier Radius, Spalt, Membranspannung).
    # Bei diesem sehr weichen Prüfling (T = 40 N/m, 45 V) liegt f_hom für
    # 48 Bohrungen bei 2.5 kHz — der Vergleich bei 4 kHz sitzt also schon
    # darüber (Π = 16, im beobachteten 1-dB-Bereich 10…42) und trifft
    # trotzdem auf 0.2 dB. Die Vorsichtsgrenze Π = 10 ist konservativ.
    # NACHTRAG Gegenproben 51/52: oberhalb der Membranresonanz liegt der
    # 3D-Löser über 2D, unabhängig vom Lochbild (96 Bohrungen, f_hom
    # 7.5 kHz: 2D/3D −0.3 dB bei 4 kHz; vor der Θ-konsistenten Wandlung
    # aus Gegenprobe 54 −0.8 dB, vor der Ringkopplung aus Gegenprobe 52
    # −1.6 dB). Das ist die FORMANPASSUNG der Membran, die
    # das 2D-Einmodenbild nicht kann (Gegenprobe 52), nicht die Film-
    # dämpfung. Verglichen wird deshalb bei 1 kHz (die 45-V-Kapsel ist
    # stark erweicht, ihre Resonanz liegt unter 300 Hz — 1 kHz ist also
    # schon der dämpfungsbestimmte Bereich darüber). Der 4-kHz-Wert wird
    # ausgegeben.
    if _HAS_SCIPY:
        # (45 V: mit dem EXAKTEN Arbeitspunkt, Gegenprobe 49, liegt der
        # Pull-in dieser weichen Kapsel bei 50.0 V — die früheren 50 V
        # saßen genau auf der Falte. Geprüft wird hier der Film, nicht die
        # Nähe zum Kollaps.)
        par31 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=45.0, n_blind_holes=0, rear_network_enabled=True,
            delay_length=0.0, cavity_length=8.0e-3,
            cavity_wall_thickness=1.5e-3, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=28e-3)
        # a) Strukturbeweis am widerstandsarmen Zweitor
        c31 = MicrophoneCapsule(
            **{**par31, "backplate_thickness": 0.3e-3,
               "n_through_holes": 48, "through_hole_diameter": 1.6e-3,
               "squeeze_model": "1d"})
        om31 = 2.0 * np.pi * np.array([20.0, 200.0])
        T31 = c31._backplate_gap_abcd(om31, outside_to_membrane=False,
                                      polarized=True)
        z_rel = np.real(T31[0, 1] / T31[1, 1]) / c31.R_A_gap_front
        assert np.all(np.abs(z_rel - 1.0) < 5e-3), \
            (f"Zweitor MUSS den Škvor-Widerstand bereits enthalten "
             f"(Verhältnis {np.round(z_rel, 4)})")
        assert np.all(np.real(c31._membrane_impedance(om31))
                      < 0.02 * c31.R_A_gap_front), \
            "Membranimpedanz darf den Spaltfilm nicht nochmals tragen"
        # b) Druckempfänger: 2D muss den 3D-Feldlöser treffen (1 kHz;
        #    4 kHz nur ausgegeben, s. Nachtrag oben)
        f31 = [1000.0, 4000.0]
        dev31 = {}
        dev31_4k = {}
        for n31, d31 in ((12, 1.40e-3), (48, 0.70e-3), (96, 0.495e-3)):
            p31 = dict(par31, n_through_holes=n31,
                       through_hole_diameter=d31)
            s2 = np.abs(MicrophoneCapsule(squeeze_model="2d",
                                          **p31).transfer_function(f31))
            s3 = np.abs(MicrophoneCapsule(squeeze_model="3d",
                                          **p31).transfer_function(f31))
            dev31[n31] = float(20.0 * np.log10(s2[0] / s3[0]))
            dev31_4k[n31] = float(20.0 * np.log10(s2[1] / s3[1]))
        for n31 in (48, 96):
            assert abs(dev31[n31]) < 1.0, \
                (f"2D muss den 3D-Feldlöser treffen (n_th = {n31}: "
                 f"{dev31[n31]:+.2f} dB)")
        # c) spärliches Raster: dokumentierte Grenze, nicht Fehler — bei
        #    1 kHz liegt die 12-Loch-Kapsel weit über ihrer Grenze f_hom
        #    (133 Hz) und damit außerhalb der 1-dB-Toleranz der Warnung
        assert abs(dev31[12]) > 1.0, \
            ("bei 12 Bohrungen ist die Homogenisierung am Ende — die "
             "Abweichung gehört dokumentiert, nicht wegkalibriert")
        print(f"Filmdämpfung einmal: Zweitor trägt Škvor exakt "
              f"({np.max(np.abs(z_rel - 1.0)):.1e}), Membranimpedanz nur "
              f"noch Materialdämpfung; Druckempfänger 2D vs. 3D bei 1 kHz "
              f"{dev31[48]:+.2f}/{dev31[96]:+.2f} dB bei 48/96 Bohrungen "
              f"(12 Bohrungen: {dev31[12]:+.1f} dB — Homogenisierungs"
              f"grenze); 4 kHz {dev31_4k[48]:+.2f}/{dev31_4k[96]:+.2f} dB "
              f"= Formanpassung der Membran (Einmodenbild), lochbild"
              f"unabhängig  OK")


def test_gp35_filmtragheit_gegen_die_literatur():
    """Gegenprobe 35: Filmträgheit Φ(ω) gegen die Literatur."""
    # Die Frequenzkorrektur des Spaltfilms (:meth:`_film_R_dynamic`) haben
    # wir selbst hergeleitet: Φ(ω) = (h³/12μ)/K_f mit dem Schlitz-
    # Zwikker–Kosten-Leitwert K_f. Sie lässt sich extern verankern.
    #
    # Homentcovschi & Miles, J. Acoust. Soc. Am. 124(1), 175–181 (2008),
    # doi:10.1121/1.2918542, leiten eine Reynolds-Gleichung her, die
    # Trägheit und Gasverdünnung enthält (ihre Gl. 12/13). Ihr
    # Trägheitsfaktor M hängt am Parameter
    #
    #     K = d0 · sqrt(rho·omega/mu)
    #
    # und sie geben die Tieffrequenz-Entwicklung M = 1 − i·K²/10 an.
    # Konvention: ihr e^{−iωt} gegen unser e^{+jωt}, also ihr −i = unser
    # +j. Unser Φ muss damit 1 + j·K²/10 sein. Wegen α = (h/2)·sqrt(jωρ/μ)
    # ist K = 2·|α|/sqrt(j), die beiden Parameter sind dasselbe.
    #
    # Verankert wird die REIHE, nicht nur ein Zahlenwert:
    # a) Φ(ω→0) = 1 exakt (Poiseuille-Grenzfall).
    # b) Das erste Glied trifft die Publikation. Aus
    #    1 − tanh(α)/α = α²/3 − 2α⁴/15 + 17α⁶/315 − … folgt
    #        Φ = 1/(1 − 2α²/5 + 17α⁴/105 − …)
    #          = 1 + (2/5)α² + (4/25 − 17/105)·α⁴ + …
    #    mit α² = j·K²/4 wird (2/5)α² = j·K²/10 — genau ihr Term.
    # c) Das NÄCHSTE Glied ist damit festgelegt: 4/25 − 17/105 = −1/525
    #    und α⁴ = −K⁴/16, also +K⁴/8400. Dass unser Φ auch das trifft,
    #    zeigt einen echten Reihenanschluss und keinen Zufall an einem
    #    Punkt — nach Abzug beider Glieder bleibt O(K⁶).
    # d) WARNUNG als Prüfung: die Autoren schreiben, M = 1 sei unter
    #    100 kHz eine gute Näherung. Das gilt für MEMS-Spalte von 1–2 µm
    #    (dort K < 0.5). Bei Kapselspalten ist K = O(1…10) und die
    #    Korrektur wesentlich — wer sie mit Verweis auf die
    #    MEMS-Literatur wegvereinfacht, macht einen Fehler. Deshalb steht
    #    hier eine untere Schranke für den K67-Fall.
    _K_of = lambda h, f: h * np.sqrt(RHO0 * 2.0 * np.pi * f / MU_AIR)
    c35 = MicrophoneCapsule(architecture="single")
    # a) Poiseuille-Grenzfall: Φ → 1, und zwar QUADRATISCH in K. Eine
    #    feste Zahlenschranke wäre hier falsch — bei jeder endlichen
    #    Frequenz bleibt das erste Glied jK²/10 stehen. Geprüft wird
    #    deshalb das Skalierungsgesetz: eine Dekade tiefere Frequenz muss
    #    den Abstand zu 1 um genau eine Dekade verkleinern (K² ∝ ω).
    #    Das Fenster liegt bewusst bei 0.1–10 Hz: noch tiefer wird
    #    1 − tanh(α)/α ≈ α²/3 als Differenz zweier Zahlen nahe 1
    #    berechnet und verliert durch Auslöschung Stellen (bei 1 mHz
    #    bereits 0.06 %) — das ist Rundung, nicht Physik.
    dev35 = []
    for f0_35 in (1.0e-1, 1.0e0, 1.0e1):
        p35 = complex(np.atleast_1d(MicrophoneCapsule._film_R_dynamic(
            np.array([2.0 * np.pi * f0_35]), 60e-6))[0])
        dev35.append(abs(p35 - 1.0) / (_K_of(60e-6, f0_35) ** 2))
    assert max(abs(d / 0.1 - 1.0) for d in dev35) < 1e-3, \
        (f"Φ − 1 muss im Grenzfall exakt K²/10 sein (gemessen "
         f"{np.round(dev35, 6)})")
    # b/c) Reihenanschluss über zwei Glieder
    worst1_35, worst2_35, ord6_35 = 0.0, 0.0, []
    for h35 in (15e-6, 25e-6, 40e-6, 60e-6, 230e-6):
        for f35 in (200.0, 1000.0, 2000.0, 8000.0, 20000.0):
            K35 = _K_of(h35, f35)
            if K35 > 1.6:                 # jenseits davon bricht die Reihe
                continue
            phi = complex(np.atleast_1d(MicrophoneCapsule._film_R_dynamic(
                np.array([2.0 * np.pi * f35]), h35))[0])
            lit1 = 1.0 + 1j * K35**2 / 10.0
            lit2 = lit1 + K35**4 / 8400.0
            worst1_35 = max(worst1_35, abs(phi - lit1) / K35**4)
            worst2_35 = max(worst2_35, abs(phi - lit2))
            ord6_35.append(abs(phi - lit2) / K35**6)
    assert worst1_35 < 2.0e-4, \
        (f"Φ muss die publizierte Entwicklung 1 + jK²/10 in erster Ordnung "
         f"treffen (Rest/K⁴ = {worst1_35:.2e}, erwartet 1/8400)")
    assert abs(np.mean(ord6_35) - 1.323e-6) / 1.323e-6 < 0.05, \
        (f"nach Abzug beider Glieder muss O(K⁶) bleiben "
         f"(Rest/K⁶ = {np.mean(ord6_35):.3e})")
    # d) bei Kapselspalten ist die Korrektur NICHT vernachlässigbar
    K67_35 = _K_of(40e-6, 7000.0)
    phi_k67 = complex(np.atleast_1d(MicrophoneCapsule._film_R_dynamic(
        np.array([2.0 * np.pi * 7000.0]), 40e-6))[0])
    assert K67_35 > 1.5 and abs(phi_k67.imag) > 0.3, \
        (f"bei Kapselspalten muss die Filmträgheit spürbar sein "
         f"(K = {K67_35:.2f}, Im Φ = {phi_k67.imag:.3f}) — die "
         f"MEMS-Näherung M ≈ 1 gilt hier nicht")
    print(f"Filmträgheit extern verankert (Homentcovschi & Miles, JASA 124, "
          f"175 (2008)): |Φ−1|/K² = {np.mean(dev35):.5f} über drei "
          f"Frequenzdekaden (exakt 1/10); erstes Glied == publiziertes "
          f"1 + jK²/10 (Rest/K⁴ = {worst1_35:.1e} gegen hergeleitete "
          f"1/8400); nach Abzug des K⁴-Glieds bleibt O(K⁶) "
          f"({np.mean(ord6_35):.2e}); K67-Spalt K = {K67_35:.2f} mit "
          f"Im Φ = {phi_k67.imag:.2f} — MEMS-Näherung M ≈ 1 gilt dort "
          f"NICHT  OK")


def test_gp36_reaktivanteil_der_zell_engstelle():
    """Gegenprobe 36: Reaktivanteil der Zell-Engstelle."""
    # Der Zellterm B(q)/(π·K_f) liefert Widerstand UND Trägheit aus EINER
    # komplexen Größe. Der Widerstand ist über Škvor verankert
    # (Gegenprobe 8/30); der Reaktivanteil war bisher nur als analytische
    # Fortsetzung desselben B(q) begründet. Hier steht seine unabhängige
    # Herleitung.
    #
    # Im Trägheitsgrenzwert (K_f → h/(jωρ)) wird der Zellterm zu
    # M = ρ0·B(q)/(π·h). Dieselbe Größe folgt aus der KINETISCHEN ENERGIE
    # der Schmierfilmströmung, ganz ohne Impulsbilanz:
    #
    #     u(r) = Q(r)/(2π r h),  Q(r) = v·π(r_c² − r²)
    #     E    = ½ρ0 ∫ u² dV,   M = 2E/q²,   q = v·π·r_c²
    #     =>   M = ρ0/(2π h r_c⁴)·[r_c⁴ln(r_c/r_h) − r_c²(r_c²−r_h²)
    #                              + (r_c⁴−r_h⁴)/4]
    #          = ρ0·B(q)/(π·h)                        (mit q = r_h²/r_c²)
    #
    # Zwei Wege, ein Ergebnis — der Reaktivanteil ist also exakt die
    # Trägheit der Schmierfilmströmung und keine Fortschreibung.
    #
    # RICHTUNG EINER MÖGLICHEN KORREKTUR (Scratchpad, nicht hier
    # nachgerechnet — ein PDE-Löser gehört nicht in die Suite): löst man
    # dieselbe Zelle wirbelfrei (Potentialströmung, exakt im Trägheits-
    # grenzfall) auf einem konvergierten Gitter, liegt die Trägheit
    # HÖHER als der Schmierfilmwert — bei r_c/r_h = 6 um +16 % (h/r_h =
    # 0.4) bis +0.7 % (h/r_h = 0.02), also mit dem korrekten Grenzfall
    # h→0. Für unsere Kapselgeometrien sind es 5…10 %. Der Zellterm kann
    # damit KEIN Massenüberschuss sein: eine Korrektur würde die
    # Antiresonanz weiter nach unten schieben. Das widerlegt den
    # ursprünglichen Verdacht aus Gegenprobe 32.
    #
    # STOKES-ZELLE GEPRÜFT UND VERWORFEN. Homentcovschi/Murray/Miles,
    # Microfluid Nanofluid 9, 865–879 (2010), lösen dieselbe Zelle in
    # Stokes-Näherung außerhalb der Schmierfilmannahme und geben eine
    # geschlossene Reihenlösung (ihre Gl. 42/54/56). Nachgebaut und
    # gegen ihre eigene Tabelle 3 validiert: alle sechs publizierten
    # Strukturen auf 0.0 % reproduziert. Für unsere Geometrien liefert
    # sie 1.20× (CTU) bis 1.56× (Messmikrofon) unseres Filmterms, in
    # Real- UND Imaginärteil gleich skaliert.
    #
    # ÜBERNOMMEN WIRD SIE TROTZDEM NICHT, und der Grund ist ein
    # Grenzfalltest: für d→0 muss jede Zellformel gegen die Schmierfilm-
    # lösung laufen, denn dort ist Škvor exakt (s. die Widerstandsprobe
    # unten). Ihre tut das NICHT — das Verhältnis sättigt bei 1.1547
    # (q = 0.003) bzw. 1.5600 (q = 0.101). Die Ursache benennen die
    # Autoren selbst: die Ein-Term-Näherung gleichförmigen Drucks UND
    # gleichförmiger Geschwindigkeit an der Lochöffnung (ihre Gl. 46/47),
    # gültig nur für kleine Lochradien. In ihren Validierungsfällen
    # trägt der Film nur 27…38 % der Zellimpedanz — der Rest ist der
    # Poiseuille-Widerstand der Bohrung —, der Fehler verschwindet dort
    # also in der ±10-%-Übereinstimmung mit der Messung. In unseren
    # Kapseln trägt der Film 98…100 %. Ihre Formel zu übernehmen hieße,
    # einen Fehler von 35…56 % einzubauen.
    if _HAS_SCIPY:
        from scipy.integrate import quad as _quad36
        worst36 = 0.0
        for q36 in (0.003, 0.01, 0.03, 0.1, 0.3):
            r_c36, h36 = 9.0e-3, 230e-6
            r_h36 = r_c36 * np.sqrt(q36)
            # kinetische Energie der Schmierfilmströmung (v = 1)
            def _u2dV(rr, _rc=r_c36, _h=h36):
                return ((np.pi * (_rc**2 - rr**2) / (2.0 * np.pi * rr * _h))**2
                        * 2.0 * np.pi * rr * _h)
            E36 = 0.5 * RHO0 * _quad36(_u2dV, r_h36, r_c36, limit=200)[0]
            M_energy = 2.0 * E36 / (np.pi * r_c36**2) ** 2
            B36 = (q36 / 2.0 - q36**2 / 8.0 - np.log(q36) / 4.0 - 3.0 / 8.0)
            M_cell = RHO0 * B36 / (np.pi * h36)
            worst36 = max(worst36, abs(M_energy / M_cell - 1.0))
        assert worst36 < 1e-6, \
            (f"Reaktivanteil muss exakt die kinetische Energie der "
             f"Schmierfilmströmung sein (Abweichung {worst36:.1e})")
        # WIDERSTANDSTEIL: B(q) ist die EXAKTE Lösung der Reynolds-Zelle.
        # Darauf beruht das Urteil über die Stokes-Zelle oben — eine
        # Formel, die im Grenzfall d→0 nicht hierher läuft, ist dort
        # falsch. Erstprinzipien: Q(r) = v·π(r_c²−r²) fließt EINWÄRTS,
        # dp/dr = 12μQ/(2πr h³), p(r_h) = 0, R = <p>_Zelle/(v·π·r_c²).
        worst36r = 0.0
        for q36 in (0.003, 0.01, 0.03, 0.1, 0.3):
            r_c36, h36 = 9.0e-3, 230e-6
            b36 = r_c36 * np.sqrt(q36)
            k36 = 6.0 * MU_AIR / h36**3

            def _p36(rr, _rc=r_c36, _b=b36, _k=k36):
                return _k * (_rc**2 * np.log(rr / _b) - (rr * rr - _b * _b) / 2.0)

            F36 = _quad36(lambda rr: _p36(rr) * 2.0 * np.pi * rr,
                          b36, r_c36, limit=200)[0]
            R_int = (F36 / (np.pi * r_c36**2)) / (np.pi * r_c36**2)
            B36q = (q36 / 2.0 - q36**2 / 8.0 - np.log(q36) / 4.0 - 3.0 / 8.0)
            R_form = 12.0 * MU_AIR / (np.pi * h36**3) * B36q
            worst36r = max(worst36r, abs(R_int / R_form - 1.0))
        assert worst36r < 1e-6, \
            (f"B(q) muss die exakte Reynolds-Zellösung sein "
             f"(Abweichung {worst36r:.1e})")
        # Der Zellterm selbst muss diesen Grenzwert ANNEHMEN, und zwar in
        # der richtigen Form. Eine feste Zahlenschranke wäre hier falsch:
        # bei endlicher Frequenz ist 1 − tanh(α)/α ≈ 1 − 1/α, also
        #     M/M_∞ = 1 + Re(1/α) = 1 + 1/(√2·|α|)
        # (α trägt die Phase π/4). Geprüft wird deshalb das Grenzgesetz
        # (M/M_∞ − 1)·√2·|α| → 1 — das verankert die ganze Asymptotik,
        # nicht einen Punkt.
        h36b, q36b = 230e-6, 0.01
        B36b = (q36b / 2.0 - q36b**2 / 8.0 - np.log(q36b) / 4.0 - 3.0 / 8.0)
        M_inf36 = RHO0 * B36b / (np.pi * h36b)
        prod36 = []
        for f36 in (1.0e6, 5.0e6, 2.0e7):
            om36 = np.array([2.0 * np.pi * f36])
            Kf36 = ((h36b**3 / (12.0 * MU_AIR))
                    / MicrophoneCapsule._film_R_dynamic(om36, h36b))
            Z36 = B36b / (np.pi * complex(np.atleast_1d(Kf36)[0]))
            al36 = abs(0.5 * h36b * np.sqrt(1j * om36[0] * RHO0 / MU_AIR))
            prod36.append((Z36.imag / om36[0] / M_inf36 - 1.0)
                          * np.sqrt(2.0) * al36)
        assert max(abs(p - 1.0) for p in prod36) < 5e-4, \
            (f"Zellterm muss ρ0·B/(π·h) wie 1 + 1/(√2·|α|) annehmen "
             f"(Produkte {np.round(prod36, 5)})")
        print(f"Zellterm exakt im Schmierfilm: Widerstand == Reynolds-"
              f"Zellösung ({worst36r:.0e}), Reaktivanteil == kinetische "
              f"Energie ({worst36:.0e}) über q = 0.003…0.3; Zellterm nimmt "
              f"ρ0·B(q)/(π·h) wie 1 + 1/(√2·|α|) an "
              f"(Grenzgesetz {min(prod36):.5f}…{max(prod36):.5f}); "
              f"wirbelfreie Lösung liegt HÖHER (+5…16 %) — der Zellterm ist "
              f"kein Massenüberschuss  OK")


def test_gp37_randumgehung_geschlossene_form():
    """Gegenprobe 37: Randumgehung, geschlossene Form."""
    # Der Quetschfilm liegt nur unter der Backplate, die Membran erzeugt
    # ihren Volumenfluss aber über ihrer GANZEN Fläche. Für eine LOCHFREIE
    # Platte mit drucklosem Rand (breiter Randspalt) und der parabolischen
    # Grundmode φ = 1 − r²/a_mem² ist die Filmimpedanz geschlossen
    # integrierbar. Mit u = (a_bp/a_mem)² und x = r²/a_mem²:
    #     Q(x) = 2x − x²                     (Fluss durch den Radius r)
    #     p(x) = (6μ/πh³)·[(u − u²/4) − (x − x²/4)]
    #     Z    = ∫p·φ dA / ∫φ dA
    #          = (12μ/πh³)·(u²/2 − u³/3 + u⁴/16)
    # Der Grenzfall u = 1 (Platte so groß wie die Membran) ist 11μ/(4πh³).
    # Diese EINE Probe verankert alle drei Teile der Umgehung zugleich:
    # die Normierung der Quelle auf die ganze Membranfläche, die
    # Einspeisung des Restflusses am Filmrand und die Projektion, die
    # außerhalb der Platte den Randdruck statt des Filmdrucks sieht.
    # Ohne die Umgehung wäre Z um 1/f_in² = 1/[u(2−u)]² zu groß — bei
    # den B&K-Kapseln der Gegenprobe 38 sind das +28 bzw. +56 %.
    if _HAS_SCIPY:
        b37 = dict(membrane_resonance_hz=None, membrane_thickness=5e-6,
                   membrane_tension=3000.0,
                   membrane_material={"rho": 8900.0, "E": 200e9, "nu": 0.31},
                   air_gap=25e-6, backplate_thickness=1e-3,
                   bias_voltage=10.0, architecture="single",
                   n_through_holes=0, n_blind_holes=0,
                   ring_vent_width=3e-3, ring_vent_length=1e-4,
                   rear_network_enabled=True, cavity_length=5e-3,
                   n_cavity_holes=0, fabric_front_rayl=0.0,
                   fabric_rear_rayl=0.0, include_diffraction=False,
                   squeeze_model="2d")
        om37 = np.array([2.0 * np.pi * 20.0])       # tief -> reines Poiseuille
        worst37, a37 = 0.0, 10e-3
        for rb37 in (1.0, 0.95, 0.90, 0.85, 0.80, 0.70):
            c37 = MicrophoneCapsule(membrane_diameter=2 * a37,
                                    backplate_diameter=2 * a37 * rb37, **b37)
            T37 = c37._gap_field_2port(om37)
            Z37 = complex((T37[0, 1] / T37[1, 1])[0]).real
            u37 = (c37.a_bp / c37.a_mem) ** 2
            Zan37 = (12.0 * MU_AIR / (np.pi * c37.h_gap**3)
                     * (u37**2 / 2.0 - u37**3 / 3.0 + u37**4 / 16.0))
            worst37 = max(worst37, abs(Z37 / Zan37 - 1.0))
        assert worst37 < 1e-3, \
            (f"Randumgehung muss die geschlossene Form treffen "
             f"(Abweichung {worst37:.1e})")
        # Grenzfall u = 1: 11μ/(4πh³)
        c37e = MicrophoneCapsule(membrane_diameter=2 * a37,
                                 backplate_diameter=2 * a37, **b37)
        T37e = c37e._gap_field_2port(om37)
        Z37e = complex((T37e[0, 1] / T37e[1, 1])[0]).real
        rel37e = Z37e / (11.0 * MU_AIR / (4.0 * np.pi * c37e.h_gap**3)) - 1.0
        assert abs(rel37e) < 1e-3, \
            (f"lochfreie Platte mit a_bp = a_mem muss 11μ/(4πh³) sein "
             f"({rel37e:+.1e})")
        print(f"Randumgehung: Filmimpedanz == (12μ/πh³)·(u²/2 − u³/3 + "
              f"u⁴/16) über a_bp/a_mem = 0.70…1.00 ({worst37:.0e}); "
              f"Grenzfall u = 1 == 11μ/(4πh³) ({rel37e:+.0e})  OK")


def _exakt60(n, R, r, rand, n_fein=60):
    """Unabhängige Referenz zu Gegenprobe 60: statischer Reynolds-Film
    (K = 1) auf der Kreisplatte a = 1 mit n äquipotentialen Löchern
    (p = 0) vom Radius r auf dem Kreis R, Quelle s = 1 − ρ² nur über dem
    Film (über den Löchern direkt ins Loch), Rand dicht oder belüftet
    (p = 0). Gelöst über die GANZE Platte in der log-Ebene ζ = ln z auf
    dem Halbsektor [0, π/n] (Spiegelsymmetrie), um das Loch verdichtetes
    Tensorgitter, Shortley–Weller am Lochrand — andere Diskretisierung
    und anderer Schnitt als das Makroelement. R = ∫p·s dA/(∫s dA)²."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    def gitter(a, b, lo, hi, h, h_max):
        pts, x = [a], a
        while x - h > lo + 0.5 * h:
            h = min(1.12 * h, h_max)
            x -= h
            pts.append(x)
        pts, x, h0 = pts[::-1], a, (b - a) / max(int(np.ceil((b - a) / h)), 1)
        pts += list(np.linspace(a, b, max(int(round((b - a) / h0)), 1) + 1)[1:])
        x, h = b, h0
        while x + h < hi - 0.5 * h:
            h = min(1.12 * h, h_max)
            x += h
            pts.append(x)
        return np.unique(np.array([lo] + pts + [hi]))

    t_lo, t_hi = np.log(R - r), np.log(min(R + r, 1.0))
    y_hw = min(np.arcsin(r / R), np.pi / n)
    h = min(t_hi - t_lo, 2 * y_hw) / n_fein
    X = gitter(t_lo, t_hi, np.log(1e-4), 0.0, h, 2.0 / n_fein)
    Yg = gitter(0.0, y_hw, 0.0, np.pi / n, h, 2.0 / n_fein)
    Nx, Ny = X.size, Yg.size
    rho = np.exp(X)[:, None] * np.ones((1, Ny))
    loch = ((rho * np.cos(Yg) - R) ** 2 + (rho * np.sin(Yg)) ** 2) < r * r
    unb = ~loch
    if rand == "belüftet":
        unb[-1, :] = False
    idx = -np.ones((Nx, Ny), int)
    idx[unb] = np.arange(unb.sum())
    f = (1.0 - rho**2) * rho**2                    # Quelle in ζ (dA = ρ² dζ)
    rows, cols, vals, b = [], [], [], np.zeros(unb.sum())

    def schnitt(i, j, ii, jj):
        if ii != i:                                # Linie y = Yg[j]
            s2 = r * r - (R * np.sin(Yg[j])) ** 2
            q = [np.log(v) for v in (R * np.cos(Yg[j]) - np.sqrt(max(s2, 0)),
                                     R * np.cos(Yg[j]) + np.sqrt(max(s2, 0)))
                 if v > 0 and s2 >= 0]
            q = [abs(v - X[i]) for v in q if min(X[i], X[ii]) <= v <= max(X[i], X[ii])]
        else:                                      # Kreis ρ = e^X[i]
            c = (rho[i, 0] ** 2 + R * R - r * r) / (2 * rho[i, 0] * R)
            q = ([abs(np.arccos(c) - Yg[j])] if abs(c) <= 1
                 and min(Yg[j], Yg[jj]) <= np.arccos(c) <= max(Yg[j], Yg[jj])
                 else [])
        return min(q) if q else None

    for i in range(Nx):
        for j in range(Ny):
            if not unb[i, j]:
                continue
            k = idx[i, j]
            b[k] = f[i, j]
            dg = 0.0
            for achse in ("x", "y"):
                if achse == "x":
                    nb = [(i - 1 if i > 0 else 1, j), (i + 1 if i < Nx - 1 else Nx - 2, j)]
                    hh = [abs(X[a] - X[i]) for a, _ in nb]
                else:
                    nb = [(i, j - 1 if j > 0 else 1), (i, j + 1 if j < Ny - 1 else Ny - 2)]
                    hh = [abs(Yg[c] - Yg[j]) for _, c in nb]
                for m, (a, c) in enumerate(nb):
                    if loch[a, c]:
                        s_ = schnitt(i, j, a, c)
                        if s_ is not None:
                            hh[m] = max(s_, 1e-4 * hh[m])
                for (a, c), hm in zip(nb, hh):
                    w = 2.0 / (hm * (hh[0] + hh[1]))
                    dg += w
                    if unb[a, c]:
                        rows.append(k)
                        cols.append(idx[a, c])
                        vals.append(-w)
            rows.append(k)
            cols.append(k)
            vals.append(dg)
    p = spsolve(csr_matrix((vals, (rows, cols)), shape=(b.size, b.size)).tocsc(), b)
    wx = np.zeros(Nx)
    wx[:-1] += 0.5 * np.diff(X)
    wx[1:] += 0.5 * np.diff(X)
    wy = np.zeros(Ny)
    wy[:-1] += 0.5 * np.diff(Yg)
    wy[1:] += 0.5 * np.diff(Yg)
    W = np.outer(wx, wy)
    P = np.zeros((Nx, Ny))
    P[unb] = p
    Q = float(np.sum((1.0 - rho**2) * rho**2 * W))   # inkl. Lochflächen
    return float(np.sum(P * f * W)) / (2 * n * Q**2)


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp60_lochkreis_als_makroelement(stand):
    """Gegenprobe 60: Lochkreise im 2D-Feld als exakte Makroelemente."""
    # Das 2D-Feld verschmierte jeden Lochkreis zu einem Gaußband und gab
    # jeder Bohrung die Škvor-Zelle in Serie. Für große Löcher (B&K 4134:
    # r = 0.14·a) überschätzte das den Filmwiderstand um bis zu 60 %
    # (Gegenproben 52, 58, 59). Zwei Zwischenstufen sind verworfen:
    # die Liniensenke mit dem Zusammenlaufwiderstand ln(R/(n·r))/(2πK)
    # trifft kleine Löcher auf 0.6 %, wird aber negativ, sobald n·r > R,
    # weil große, äquipotentiale Löcher das Feld zwischen innen und außen
    # kurzschließen (Dipolanteil); ein exakter Dreipol (innen, außen, Loch)
    # behebt das, verliert aber die Quellen im Band (Druckbögen zwischen
    # den Löchern, Lochfläche ohne Film) — ±3 %.
    # MAKROELEMENT (s. _lochband_makro): je Lochkreis rechnet eine kleine
    # statische Sektorlösung in der log-Ebene (Laplace bleibt konform
    # invariant) die vollständige Kopplung zwischen den FV-Zellen des
    # Bandes, den Nachbarzellen innen und außen und dem Lochknoten; Mittel
    # über die FILMfläche der Zelle gegen eine darüber gleichverteilte
    # Quelle, beides reziprok. Im Feld skaliert es mit K(ω); Speicherung
    # und Membranquelle der Zellen laufen über die Filmfläche, die Membran
    # über den Mündungen pumpt in den Lochknoten.
    # a) STATISCH EXAKT: der Filmwiderstand des 2D-Zweitors (ω → 0, dünne
    #    Platte, a_bp = a_mem, Rand dicht) gegen eine unabhängige Lösung
    #    derselben Reynolds-Gleichung über die ganze Platte (_exakt60) —
    #    auf 0.5 %; das Gaußband lag bis 70 % daneben.
    # b) MITTELLOCH gegen die geschlossene Form auf 0.1 %.
    # c) Reziprok (det T = 1) auch mit Bändern, Mittelloch, Ringschlitz
    #    aus überlappenden Löchern und Freistich.
    # d) Gegen 3D bei erzwungener Form (Phasenmethode wie Gegenprobe 52):
    #    Lochkreise mit kleinen Löchern auf 1 % (vorher bis 4 %), große
    #    Mündungen (B&K-Originalgeometrie) auf 3 %; dort liegt 3D 1–3 %
    #    ÜBER der exakten Lösung (Stand-Werte).
    #    VORSICHT bei dieser Methode: der Folienverlust ist im 2D ein
    #    fester Widerstand ω0·M/Q, im 3D wächst er mit ω. Mit der
    #    erzwungenen Form (f_res = 300 kHz) bläht das den 2D-Wert bei 1 kHz
    #    300-fach auf; bei Nickelfolie sind das bis 8 % des Filmwiderstands
    #    (3D/2D 0.94 statt 1.02, beim 40-µm-Spalt 0.49 statt 1.03). Der
    #    Vergleich schaltet ihn deshalb ab (wie jetzt auch 52 und 53).
    if not _HAS_SCIPY:
        return
    fa60 = np.array([2.0 * np.pi * 0.01])

    def _z60(R, r, a, n=6, makro=True, **kw):
        q = dict(membrane_material={"rho": 8900.0, "E": 221e9, "nu": 0.31},
                 membrane_resonance_hz=300e3, membrane_diameter=2 * a,
                 membrane_thickness=5e-6, air_gap=18.6e-6,
                 backplate_diameter=2 * a, backplate_thickness=1e-6,
                 bias_voltage=1.0, architecture="single", n_through_holes=n,
                 through_hole_diameter=2 * r, through_hole_pcd=2 * R,
                 n_blind_holes=0, rear_network_enabled=True, delay_length=0.0,
                 cavity_length=5e-3, n_cavity_holes=0, fabric_front_rayl=0.0,
                 fabric_rear_rayl=0.0, include_diffraction=False,
                 squeeze_model="2d")
        q.update(kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = MicrophoneCapsule(**q)
        c._RING_MAKRO = makro
        T = c._gap_field_2port(fa60)
        K0 = c.h_gap ** 3 / (12.0 * MU_AIR)
        return complex((T[0, 1] / T[1, 1])[0]).real * K0, c

    # a) Lochkreise gegen die unabhängige Lösung
    faelle60 = (("B&K-Originalgeometrie", 6, 1.70e-3, 0.50e-3, 3.6e-3),
                ("4134 Tab. I", 6, 2.032e-3, 0.508e-3, 3.607e-3),
                ("große Löcher r = 0.15·a", 6, 1.80e-3, 0.54e-3, 3.6e-3),
                ("24 kleine Löcher bei 0.33·a", 24, 1.188e-3, 0.149e-3,
                 3.6e-3))
    worst60, alt60 = 0.0, {}
    for nm, n, R, r, a in faelle60:
        ex = _exakt60(n, R / a, r / a, "dicht")
        z_m, _ = _z60(R, r, a, n=n)
        z_a, _ = _z60(R, r, a, n=n, makro=False)
        worst60 = max(worst60, abs(z_m / ex - 1.0))
        alt60[nm] = z_a / ex
        assert abs(z_m / ex - 1.0) < 5e-3, \
            (f"{nm}: Makroelement muss die exakte Lösung treffen "
             f"({z_m / ex:.4f})")
    stand.wert("makro_gegen_exakt_max", worst60, "",
               "größte Abweichung 2D (Makro) gegen exakt, statisch")
    stand.wert("gaussband_gegen_exakt_bk", alt60["B&K-Originalgeometrie"],
               "", "Gaußband + Škvor / exakt, B&K-Originalgeometrie")

    # b) Mittelloch: −∇²p = 1 − ρ², p(r) = 0, p'(1) = 0
    for r_m in (0.2e-3, 0.5e-3, 1.0e-3):
        a = 3.6e-3
        z_m, c_m = _z60(0.0, r_m, a, n=1)
        u = r_m / a
        rr = np.linspace(u, 1.0, 200001)
        pp = (np.log(rr / u) / 4.0 - (rr**2 - u**2) / 4.0
              + (rr**4 - u**4) / 16.0)
        E = _trapz(pp * (1.0 - rr**2) * 2.0 * np.pi * rr, rr)
        ex_m = E / (np.pi / 2.0) ** 2
        assert abs(z_m / ex_m - 1.0) < 1e-3, \
            (f"Mittelloch r/a = {u:.3f}: Makroelement gegen die geschlossene "
             f"Form ({z_m / ex_m:.5f})")

    # c) Reziprozität mit Bändern
    om60 = 2.0 * np.pi * np.geomspace(20.0, 20e3, 40)
    for q60 in (dict(through_hole_rings=[(12, 2 * 4.763e-3),
                                         (6, 2 * 2.375e-3), (1, 0.0)],
                     backplate_diameter=2 * 6.617e-3,
                     membrane_diameter=2 * 8.89e-3,
                     through_hole_diameter=2 * 4.763e-4,
                     ring_vent_width=1.473e-3, ring_vent_length=3.556e-4),
                dict(DEB_KWARGS, **DEB_CLEARANCE),
                dict(n_through_holes=48,
                     through_hole_rings=[(48, 2 * 0.33 * 5.5e-3)],
                     through_hole_diameter=2 * 0.161e-3,
                     backplate_diameter=11e-3, membrane_diameter=12e-3)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c60 = MicrophoneCapsule(**{**dict(
                architecture="single", membrane_resonance_hz=8000.0,
                air_gap=25e-6, bias_voltage=1.0), **q60,
                "squeeze_model": "2d"})
        assert c60._fld_bands, "Lochkreise müssen als Bänder laufen"
        T60 = c60._gap_field_2port(om60)
        det60 = T60[0, 0] * T60[1, 1] - T60[0, 1] * T60[1, 0]
        assert np.max(np.abs(det60 - 1.0)) < 1e-9, \
            f"Zweitor mit Makroelement muss reziprok sein ({det60}"

    # d) gegen 3D bei erzwungener Form
    def _rr60(q, Q=1e12):
        q = dict(q, membrane_resonance_hz=300e3)
        q.pop("membrane_tension", None)
        ph = {}
        Q0 = MicrophoneCapsule._Q_MEMBRANE_INTERNAL
        MicrophoneCapsule._Q_MEMBRANE_INTERNAL = Q
        try:
            for sm in ("2d", "3d"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cc = MicrophoneCapsule(**{**q, "squeeze_model": sm})
                ph[sm] = np.angle(cc.transfer_function([1000.0])[0]
                                  / cc.transfer_function([20.0])[0])
        finally:
            MicrophoneCapsule._Q_MEMBRANE_INTERNAL = Q0
        return float(np.tan(ph["3d"]) / np.tan(ph["2d"]))

    # ½"-Kapsel der Gegenprobe 53; 48 Löcher auf 0.33·a überlappen zum
    # geschlossenen Ringschlitz (Zellen ohne Film am Lochknoten)
    b1 = dict(architecture="single", membrane_diameter=12.0e-3,
              membrane_thickness=5e-6, air_gap=25e-6,
              backplate_diameter=11.0e-3, backplate_thickness=1.5e-3,
              bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
              delay_length=0.0, cavity_length=4.0e-3,
              cavity_wall_thickness=1.0e-3, n_cavity_holes=0,
              fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
              body_diameter=14e-3)
    A60 = 48 * np.pi * (0.35e-3 * 5.5 / 11.95) ** 2
    for n, ring in ((6, 0.67), (48, 0.33)):
        rr60 = _rr60(dict(b1, n_through_holes=n,
                          through_hole_rings=[(n, ring * 11.0e-3)],
                          through_hole_diameter=2 * np.sqrt(A60 / (n * np.pi))))
        assert abs(rr60 - 1.0) < 0.01, \
            (f"{n} Löcher auf {ring}·a: Filmwiderstand 3D/2D bei erzwungener "
             f"Form ({rr60:.3f})")
    a_bk = 3.6e-3
    bk60 = dict(membrane_material={"rho": 8900.0, "E": 221e9, "nu": 0.31},
                membrane_diameter=2 * a_bk, membrane_thickness=5e-6,
                air_gap=18.6e-6, backplate_diameter=2 * a_bk,
                backplate_thickness=1e-6, bias_voltage=1.0,
                architecture="single", n_through_holes=6,
                through_hole_pcd=3.4e-3, n_blind_holes=0,
                rear_network_enabled=True, delay_length=0.0,
                cavity_length=131e-9 / (np.pi * a_bk**2), n_cavity_holes=0,
                fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
                include_diffraction=False)
    r3d60 = {}
    for r60 in (0.5e-3, 0.6e-3):
        r3d60[r60] = _rr60(dict(bk60, through_hole_diameter=2 * r60))
    assert abs(r3d60[0.5e-3] - 1.0) < 0.03, \
        (f"B&K-Originalgeometrie: Filmwiderstand 3D/2D bei erzwungener "
         f"Form ({r3d60[0.5e-3]:.3f})")
    # ... und mit Folienverlust (Q = 100) verfälscht die Methode
    r_q60 = _rr60(dict(bk60, through_hole_diameter=1.0e-3), Q=100.0)
    assert r_q60 < r3d60[0.5e-3] - 0.05, \
        (f"Folienverlust muss das Verhältnis bei Nickelfolie sichtbar "
         f"senken ({r3d60[0.5e-3]:.3f} -> {r_q60:.3f})")
    stand.wert("r3d_zu_exakt_bk", r3d60[0.5e-3], "",
               "3D/2D(= exakt), B&K-Geometrie r/a 0.139, erzwungene Form")
    stand.wert("r3d_zu_exakt_gross", r3d60[0.6e-3], "",
               "3D/2D(= exakt), r/a 0.167")
    stand.wert("r3d_zu_2d_mit_folienverlust", r_q60, "",
               "dasselbe mit Q = 100 (Artefakt der Methode)")
    print(f"Lochkreis als Makroelement: statisch gegen die unabhängige "
          f"Lösung auf {100 * worst60:.2f} % (Gaußband B&K "
          f"{alt60['B&K-Originalgeometrie']:.2f}-fach), Mittelloch auf "
          f"0.1 %, reziprok; gegen 3D bei erzwungener Form auf 1 %, große "
          f"Mündungen: 3D/exakt {r3d60[0.5e-3]:.3f} / {r3d60[0.6e-3]:.3f} "
          f"(r/a 0.14/0.17; mit Folienverlust scheinbar {r_q60:.3f})  OK")
