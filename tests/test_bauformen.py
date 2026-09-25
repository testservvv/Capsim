"""Gegenproben: Bauformen: K67, Lochkreise, Spacer, Stufen, Clearance-Ring, Gewebe, Rauschen."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


def test_gp06_k67_bauform_doppelmembran(k67):
    """Gegenprobe 6: K67-Bauform (Doppelmembran)."""
    # Zwei Membranen außen, Backplates innen (center_gap): die passive
    # Rückmembran bildet das Phasenschiebernetzwerk -> Nierencharakteristik
    # ohne Laufzeitglied/Hohlraum; die Gegentakt-Mode beider Membranen
    # gegen das innere Luftpolster erzeugt die K67-typische Präsenz-
    # anhebung im 10-kHz-Bereich.
    # verifizierte K67-Geometrie (120 Senkungen 1.3x3.7 mm, 60 durchgebohrt
    # mit 0.6-mm-Kern, kein Gewebe, Klemmringe 2x2 mm vor beiden Membranen):
    # die interne Laufzeit fällt seit der Port-Tausch-Korrektur aus den
    # physikalischen Parametern (Bohrungen, Spaltfilme, Spacer), die
    # externe Laufzeit aus der axialen Körperbeugung (_axial_body_transfer,
    # d_ext = axial + 2·Klemmringdicke, LF-Grenzfall 1.5·d_ext) — beides
    # ohne Fit-Koeffizient. Mit den nominellen Maßen: ~-26 dB bei 180°/
    # 1 kHz, tiefstes Minimum knapp vor 180° (extern 18.3 mm vs. intern
    # ~17 mm; Spacer 45 µm pinnt die Null exakt auf 180°, s.
    # examples/cardiodtest.json). Präsenz im 8-12-kHz-Band.
    di_k = k67.directivity(frequencies_hz=(1000.0,))
    ang_k = di_k["angles_deg"]
    lin_k = di_k["patterns"][1000.0]["linear"]
    pk67 = di_k["patterns"][1000.0]["db"]
    na_k = ang_k[ang_k <= 180][int(np.argmin(lin_k[ang_k <= 180]))]
    assert na_k > 150.0, \
        f"K67 muss echte Niere sein (Null bei {na_k:.0f}° statt nahe 180°)"
    assert pk67[180] < -18.0, \
        f"K67 muss rückwärts tief auslöschen ({pk67[180]:.1f} dB)"
    fr_k = k67.frequency_response(n_points=150)
    assert np.all(np.isfinite(fr_k["amplitude_db"]))
    fk, ak = fr_k["frequency_hz"], fr_k["amplitude_db_norm"]
    ihf = (fk > 5000) & (fk < 16000)
    assert 1.0 < np.max(ak[ihf]) < 12.0, \
        "Präsenzanhebung erwartet (roh; Korb/Elektronik glätten auf +2..3)"
    print(f"K67-Bauform: echte Niere (Null @{na_k:.0f}°, 180° = "
          f"{pk67[180]:.1f} dB @1 kHz), Präsenzanhebung "
          f"+{np.max(ak[ihf]):.1f} dB bei "
          f"{fk[ihf][np.argmax(ak[ihf])]/1000:.1f} kHz  OK")


def test_gp09_mehrfach_lochkreise():
    """Gegenprobe 9: Mehrfach-Lochkreise."""
    # Die Ringlisten sind eine reine VERALLGEMEINERUNG der Skalar-Parameter:
    # a) ein einzelner Ring muss exakt dem Skalar-PCD entsprechen,
    # b) mehrere gleichmäßige Anteile exakt der Gleichverteilung.
    _base = dict(architecture="single", backplate_diameter=20e-3,
                 n_blind_holes=30, blind_hole_diameter=1.2e-3)
    ref_pcd = MicrophoneCapsule(n_through_holes=12, through_hole_pcd=16e-3,
                                **_base)
    one_ring = MicrophoneCapsule(through_hole_rings=[(12, 16e-3)], **_base)
    assert np.allclose(ref_pcd.transfer_function(_fchk)[0],
                       one_ring.transfer_function(_fchk)[0], rtol=1e-12), \
        "1 Lochkreis muss dem Skalar-PCD exakt entsprechen"
    ref_uni = MicrophoneCapsule(n_through_holes=12, **_base)
    two_uni = MicrophoneCapsule(through_hole_rings=[(8, None), (4, None)],
                                **_base)
    assert np.allclose(ref_uni.transfer_function(_fchk)[0],
                       two_uni.transfer_function(_fchk)[0], rtol=1e-12), \
        "gleichmäßige Ringanteile müssen der Gleichverteilung entsprechen"
    # c) Mehrere echte Ringe: Gesamtzahlen = Summen, Dichten normiert
    #    (Σ dens·A = 1, Lochleitwerte bleiben erhalten), Ergebnis endlich —
    #    auch im 2D-Feldmodell.
    multi = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        through_hole_rings=[(6, 17e-3), (3, 12e-3), (3, 6e-3)],
        through_hole_diameter=0.7e-3,
        blind_hole_rings=[(12, 17e-3), (12, 12e-3), (6, 6e-3)],
        blind_hole_diameter=1.0e-3,
        squeeze_model="2d" if _HAS_SCIPY else "1d",
    )
    assert multi.n_th == 12 and multi.n_bh == 30
    for _dens in (multi._fld_dens_th, multi._fld_dens_bh):
        assert abs(np.sum(_dens * multi._fld_area) - 1.0) < 1e-9, \
            "Ring-Dichteprofil muss auf Σ dens·A = 1 normiert sein"
    fr_multi = multi.frequency_response(20.0, 20000.0, n_points=60)
    assert np.all(np.isfinite(fr_multi["amplitude_db"]))
    print(f"Mehrfach-Lochkreise: 1 Ring ≡ Skalar-PCD, gleichmäßige Anteile "
          f"≡ Gleichverteilung; {multi.n_th}+{multi.n_bh} Löcher auf 3+3 "
          f"Kreisen ({multi.squeeze_model}-Modell) lauffähig  OK")


def test_gp10_spacer_ruckplatte_k103_bauform():
    """Gegenprobe 10: Spacer + Rückplatte (K103-Bauform)."""
    # a) Spacer/Rückplatte = 0 muss exakt dem bisherigen Verhalten
    #    entsprechen (reine Erweiterung).
    ref10 = MicrophoneCapsule()
    zero10 = MicrophoneCapsule(rear_spacer_height=0.0,
                               rear_plate_thickness=0.0,
                               n_rear_plate_holes=0)
    assert np.allclose(ref10.transfer_function(_fchk)[0],
                       zero10.transfer_function(_fchk)[0], rtol=1e-12), \
        "Spacer/Rückplatte = 0 darf nichts ändern"
    # b) Rückplatte OHNE Löcher verschließt die Rückseite: Druckempfänger
    #    (ohne Beugung exakt Kugelcharakteristik), egal was dahinter kommt.
    sealed_rp = MicrophoneCapsule(rear_spacer_height=80e-6,
                                  rear_plate_thickness=2e-3,
                                  n_rear_plate_holes=0,
                                  include_diffraction=False)
    assert not sealed_rp.rear_open
    H0 = sealed_rp.transfer_function(1000.0, angle_deg=0.0)[0]
    H180 = sealed_rp.transfer_function(1000.0, angle_deg=180.0)[0]
    assert abs(abs(H180) / abs(H0) - 1.0) < 1e-9, \
        "dichte Rückplatte muss Druckempfänger (Kugel) ergeben"
    # c) K103-artige Konfiguration: Membran -> Backplate -> Spacer ->
    #    gelochte massive Rückplatte -> Schallfeld (Laufzeitglied/Hohlraum
    #    leer). Die Rückseite ist offen, die Plattenlöcher münden direkt,
    #    der Außenweg endet an der Rückplatte, das Pattern ist gerichtet.
    k103 = MicrophoneCapsule(
        membrane_resonance_hz=1500.0, membrane_diameter=25e-3,
        air_gap=50e-6, backplate_diameter=23e-3, backplate_thickness=3e-3,
        bias_voltage=45.0, architecture="single",
        n_through_holes=24, through_hole_diameter=0.8e-3,
        n_blind_holes=30, blind_hole_diameter=1.2e-3,
        rear_network_enabled=True,
        rear_spacer_height=50e-6, rear_plate_thickness=2e-3,
        n_rear_plate_holes=30, rear_plate_hole_diameter=0.5e-3,
        delay_length=0.0, cavity_length=0.0, n_cavity_holes=0,
        include_diffraction=False,
    )
    assert k103.rear_open and k103._plate_vents
    assert abs(k103.d_ext - (50e-6 + 3e-3 + 50e-6 + 2e-3)) < 1e-12, \
        "d_ext muss an der direkt mündenden Rückplatte enden"
    H0 = k103.transfer_function(1000.0, angle_deg=0.0)[0]
    H180 = k103.transfer_function(1000.0, angle_deg=180.0)[0]
    ratio_k103 = abs(H180) / abs(H0)
    assert ratio_k103 < 0.5, \
        f"K103-Konfiguration muss richten (180°/0° = {ratio_k103:.2f})"
    fr_k103 = k103.frequency_response(20.0, 20000.0, n_points=60)
    assert np.all(np.isfinite(fr_k103["amplitude_db"]))
    print(f"Spacer/Rückplatte (K103): 0-Werte ≡ Bestand, dichte Platte -> "
          f"Kugel, K103-Konfiguration richtet (180°/0° @1 kHz = "
          f"{20 * np.log10(ratio_k103):.1f} dB)  OK")


def test_gp11_stufenbohrung_k67_k87():
    """Gegenprobe 11: Stufenbohrung (K67/K87)."""
    # a) Grenzfall verschwindende Senkung (Tiefe -> 0, Senkungs-Ø knapp
    #    über Kern-Ø): muss die normale Durchgangsbohrung reproduzieren.
    #    n_blind = n_through (alle 24 Senkungen durchgebohrt, 0 rein blind).
    plain11 = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        n_through_holes=24, through_hole_diameter=1.0e-3,
        n_blind_holes=0, rear_network_enabled=False)
    tiny11 = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        n_through_holes=24, through_hole_diameter=1.0e-3,
        n_blind_holes=24, blind_hole_diameter=1.02e-3,
        blind_hole_depth=0.02e-3, through_holes_stepped=True,
        rear_network_enabled=False)
    assert tiny11.n_bh == 0    # 24 Senkungen, alle 24 durchgebohrt
    H_p = plain11.transfer_function(_fchk)[0]
    H_t = tiny11.transfer_function(_fchk)[0]
    assert np.max(np.abs(H_t / H_p - 1.0)) < 0.05, \
        "verschwindende Senkung muss die normale Bohrung reproduzieren"
    # b) K67-Geometrie (verifiziert: je Seite 120 Senkungen 1.3 mm x
    #    3.7 mm in der 4-mm-Halbplatte, jede zweite mit 0.6-mm-Durchbruch
    #    am Grund -> n_blind=120 gesamt, davon 60 durchgebohrt, 60 rein
    #    blind): das enge Rohr ist nur noch t_bp − Tiefe = 0.3 mm lang ->
    #    deutlich kleinere Durchgangsimpedanz; die Niere der Doppel-
    #    membran-Bauform muss ihre NULLSTELLE bei ~180° behalten.
    k67_kwargs = dict(
        membrane_material="pet", membrane_resonance_hz=1150.0,
        membrane_diameter=26e-3, membrane_thickness=6e-6,
        membrane_tension=13.7, air_gap=65e-6, backplate_diameter=25e-3,
        backplate_thickness=4e-3, bias_voltage=60.0,
        architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=120, blind_hole_diameter=1.3e-3,
        blind_hole_depth=3.7e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
        body_diameter=34e-3,
    )
    # K67 braucht das 2D-Feldmodell (radiale Druckausbreitung -> volle
    # interne Laufzeit, passend zur Klemmring-verlängerten d_ext)
    k67_st = MicrophoneCapsule(through_holes_stepped=True,
                               squeeze_model="2d", **k67_kwargs)
    # Zählweise: 120 Senkungen gesamt, 60 durchgebohrt, 60 rein blind
    assert k67_st.n_th == 60 and k67_st.n_bh == 60
    k67_full = MicrophoneCapsule(   # gleiche Kerne, volle Plattendicke
        **{**k67_kwargs, "n_blind_holes": 60})
    om1k = np.array([2.0 * np.pi * 1000.0])
    Zst = abs(k67_st._through_hole_impedance(om1k, 60)[0])
    Zpl = abs(k67_full._hole_impedance(om1k, k67_st.r_th, k67_st.t_bp, 60,
                                       end_correction=True, visc_ends=1)[0])
    assert Zst < 0.85 * Zpl, \
        "Stufenbohrung muss die Durchgangsimpedanz senken (kürzeres Rohr)"
    assert k67_st.C_A_cb > 0.0
    # Ruhekapazität trifft den nachgemessenen K67-Wert (~50 pF)
    assert 47.0 < k67_st.C_elec_0 * 1e12 < 54.0, \
        f"K67-C0 = {k67_st.C_elec_0 * 1e12:.1f} pF (erwartet ~50 pF)"
    # NIERE: tiefstes Minimum nahe 180° (mit dem axialen Beugungsumweg
    # liegt es bei den nominellen Maßen knapp davor, ~158-165°; deutlich
    # kleinere Winkel wären eine echte Superniere) über das Mittenband;
    # H180/H0 klein.
    di_st = k67_st.directivity(frequencies_hz=[500.0, 1000.0])
    ang = di_st["angles_deg"]
    for f in (500.0, 1000.0):
        lin = di_st["patterns"][f]["linear"]
        na = ang[ang <= 180][int(np.argmin(lin[ang <= 180]))]
        assert na > 150.0, \
            f"K67 muss Niere sein (Null bei {na:.0f}° statt nahe 180° -> Super)"
    st_180 = 20.0 * np.log10(
        abs(k67_st.transfer_function(1000.0, angle_deg=180.0)[0])
        / abs(k67_st.transfer_function(1000.0, angle_deg=0.0)[0]))
    # c) 2D-Feldmodell mit Stufenbohrung: reziprok und endlich
    if _HAS_SCIPY:
        st2d = MicrophoneCapsule(through_holes_stepped=True,
                                 squeeze_model="2d", **k67_kwargs)
        Tg2 = st2d._gap_field_2port(2.0 * np.pi
                                    * np.logspace(1.5, 4.3, 12))
        det2 = Tg2[0, 0] * Tg2[1, 1] - Tg2[0, 1] * Tg2[1, 0]
        assert np.max(np.abs(det2 - 1.0)) < 1e-6
        fr_st = st2d.frequency_response(20.0, 20000.0, n_points=40)
        assert np.all(np.isfinite(fr_st["amplitude_db"]))
    print(f"Stufenbohrung: Grenzfall ≡ normale Bohrung, K67 120 Senkungen/"
          f"60 durchgebohrt, Null bei ~180° (Niere), "
          f"|Z_th| um {100 * (1 - Zst / Zpl):.0f} % kleiner, Niere bleibt "
          f"(180° @1 kHz = {st_180:.1f} dB), 2D reziprok  OK")


def test_gp15_clearance_ring_stirnflachen_freistich(deb0, deb1, deb_kwargs):
    """Gegenprobe 15: Clearance-Ring (Stirnflächen-Freistich)."""
    # a) Ring aus (0) ≡ exakt das Bestandsverhalten.
    # b) Debenham-artige Platte (12 enge Durchgangslöcher auf Lochkreisen):
    #    der Freistich am Elektrodenrand entlastet die Mündungs-Engstellen
    #    des Nieren-Phasenschiebers -> deutlich tiefere 180°-Auslöschung,
    #    Null bleibt bei 180°.
    if _HAS_SCIPY:
        deb0_ref = MicrophoneCapsule(clearance_ring_diameter=0.0,
                                     clearance_ring_width=0.0,
                                     clearance_ring_depth=0.0, **deb_kwargs)
        f_chk = np.array([250.0, 1000.0, 5000.0])
        assert np.allclose(deb0.transfer_function(f_chk),
                           deb0_ref.transfer_function(f_chk),
                           rtol=1e-12, atol=0.0), \
            "Clearance-Ring = 0 muss exakt dem Bestand entsprechen"
        def _n180(c, f):
            di = c.directivity(frequencies_hz=(f,))
            pat = di["patterns"][f]
            na = di["angles_deg"][:181][int(np.argmin(pat["linear"][:181]))]
            return pat["db"][180], na
        p0, _ = _n180(deb0, 500.0)
        p1, na1 = _n180(deb1, 500.0)
        assert p1 < p0 - 8.0, \
            f"Freistich muss die 180°-Auslöschung vertiefen ({p0:.1f} -> {p1:.1f})"
        assert na1 > 172.0, \
            f"Null muss bei 180° bleiben ({na1:.0f}°)"
        print(f"Clearance-Ring: 0 ≡ Bestand; Freistich am Elektrodenrand "
              f"vertieft die Debenham-Null @500 Hz von {p0:.1f} auf "
              f"{p1:.1f} dB (Null {na1:.0f}°)  OK")


def test_gp17_laufzeit_diagnose_delay_diagnostics(deb0, deb1, deb3, deb3b, hermetic, k67, na_k):
    """Gegenprobe 17: Laufzeit-Diagnose (delay_diagnostics)."""
    # Deutung des Verhältnisses intern/extern (Sonde 1 kHz):
    # a) K67 nominal (Spacer 50 µm): Verhältnis knapp ÜBER 1 -> das
    #    Pattern-Minimum ist bei 180° GEPINNT (na_k aus Gegenprobe 6).
    #    ACHTUNG, hier stand bis Gegenprobe 29 ein Bereich knapp UNTER 1
    #    mit Minimum bei ~165°. Das war kein Messwert, sondern der aus
    #    der damaligen Simulation abgelesene Selbstbestätigungs-Bereich —
    #    und er widersprach der realen K67, deren Niere ihr Minimum bei
    #    180° hat. Mit der korrigierten Durchfluss-Zellfunktion
    #    (_cell_B_flow, Gegenprobe 30) liegt es dort; die publizierten
    #    U87-Werte werden dabei besser getroffen (180°: -26.6 statt
    #    -26.2 dB gegen -26 dB publiziert; 20 statt 21 mV/Pa gegen
    #    ~20 mV/Pa). Die DEUTUNG des Verhältnisses (unten) ist unberührt.
    # b) Spacer 45 µm (cardiodtest): kleinerer Spalt -> größerer Film-R ->
    #    LÄNGERE interne Laufzeit (Verhältnis > 1) -> Minimum bei 180°
    #    gepinnt (kein Außenwinkel bietet mehr Phase), dafür flacher.
    # c) Debenham (2D), Sonde 250 Hz (= Region der tiefsten Null ~290 Hz):
    #    Verhältnis ~ 1; externe Laufzeit ~ Kugel-Grenzfall 1.5·d_ext/c.
    # d) 3D: der Rand-Freistich allein lässt die Mündungen verengt ->
    #    hochohmiges RC, stark ÜBER-verzögert (Null bei 180°, aber flach,
    #    s. Gegenprobe 16); der breite Freistich senkt R -> Verhältnis
    #    rückt Richtung 1 (und die Null wird tiefer).
    # e) Geschlossene Rückseite (0 Durchgangslöcher) -> None.
    dd_k67 = k67.delay_diagnostics()
    assert dd_k67 is not None and 1.0 < dd_k67["ratio"] < 1.20, \
        f"K67 nominal: Verhältnis knapp >1 erwartet ({dd_k67['ratio']:.3f})"
    assert na_k >= 179.0, \
        "Konsistenz: Verhältnis >1 muss zum Minimum bei 180° gehören"
    k67_45 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=45e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=120, blind_hole_diameter=1.3e-3,
        blind_hole_depth=3.7e-3, through_holes_stepped=True,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3,
        squeeze_model="2d")
    dd_45 = k67_45.delay_diagnostics()
    assert 1.05 < dd_45["ratio"] < 1.45, \
        f"45-µm-Spacer: Über-Verzögerung erwartet ({dd_45['ratio']:.3f})"
    di45 = k67_45.directivity(frequencies_hz=(1000.0,))
    lin45 = di45["patterns"][1000.0]["linear"]
    a45 = di45["angles_deg"]
    na45 = a45[a45 <= 180][int(np.argmin(lin45[a45 <= 180]))]
    assert na45 > 179.0, \
        f"Über-Verzögerung muss das Minimum bei 180° pinnen ({na45:.0f}°)"
    if _HAS_SCIPY:
        # MIT Rand-Freistich (deb1, wie die Beispiel-JSON): angepasst in
        # der Region der tiefsten Null (~290 Hz); OHNE Ring (deb0) sind
        # die Mündungen verengt -> hochohmig -> stark über-verzögert
        # (konsistent zu Gegenprobe 15: Null 180°, aber nur -3.8 dB).
        dd_deb = deb1.delay_diagnostics(f_probe_hz=250.0)
        tau_kugel = 1.5 * deb1.d_ext / C_AIR
        # Verhältnis > 1: die 12 engen Durchgangslöcher der Braunmühl-
        # Weber-Platte verzögern ÜBER die externe Laufzeit hinaus — die
        # Null bleibt bei 180° gepinnt, wird aber flacher (gemessen
        # -13.6 dB bei 317 Hz). Das ist die Deutung (b) unten und deckt
        # sich mit dem 3D-Feldlöser, der die azimutale Zuströmung zu den
        # wenigen Löchern diskret auflöst und noch flacher ausfällt.
        # (Bis Gegenprobe 29 stand hier ~1: damals zählte die
        # Durchfluss-Zellfunktion alle 58 Bohrungen als Senken statt der
        # 12 Durchgangslöcher — s. _cell_B_flow, Gegenprobe 30.)
        assert 1.2 < dd_deb["ratio"] < 1.9, \
            f"Debenham @250 Hz: Verhältnis >1 erwartet ({dd_deb['ratio']:.3f})"
        assert 0.85 < dd_deb["tau_ext_s"] / tau_kugel < 1.15, \
            "externe Laufzeit muss dem Kugel-Grenzfall 1.5·d_ext/c folgen"
        dd_deb0 = deb0.delay_diagnostics(f_probe_hz=250.0)
        assert dd_deb0["ratio"] > 1.5, \
            (f"ohne Freistich: verengte Mündungen -> Über-Verzögerung "
             f"erwartet ({dd_deb0['ratio']:.3f})")
        dd_3d = deb3.delay_diagnostics()
        assert dd_3d is not None and np.isfinite(dd_3d["ratio"]) \
            and dd_3d["ratio"] > 1.5, \
            (f"3D/Rand-Freistich: verengte Mündungen -> starke Über-"
             f"Verzögerung erwartet ({dd_3d['ratio']:.3f})")
        dd_3db = deb3b.delay_diagnostics()
        assert 1.0 < dd_3db["ratio"] < dd_3d["ratio"], \
            (f"breiter Freistich muss die Über-Verzögerung abbauen "
             f"({dd_3db['ratio']:.3f} vs. {dd_3d['ratio']:.3f})")
    assert hermetic.delay_diagnostics() is None, \
        "Druckempfänger: keine Laufzeit-Anpassung -> None"
    print(f"Laufzeit-Diagnose @1 kHz: K67 nominal intern/extern = "
          f"{dd_k67['ratio']:.2f} ({dd_k67['dist_int_m']*1e3:.1f}/"
          f"{dd_k67['dist_ext_m']*1e3:.1f} mm, Minimum {na_k:.0f}°), "
          f"Spacer 45 µm -> {dd_45['ratio']:.2f} (Null gepinnt {na45:.0f}°); "
          f"geschlossene Rückseite -> None  OK")


@pytest.mark.feld3d
def test_gp24_position_des_ruckwartigen_gewebes():
    """Gegenprobe 24: Position des rückwärtigen Gewebes."""
    # fabric_rear_position: "backplate" (im Zylinder hinter der Platte,
    # über die volle Bohrung gespannt — Bestand) vs. "inlet" (außen ÜBER
    # den Einlassöffnungen). Physik ohne Fit-Koeffizient:
    # a) Ohne Gewebe (0 Rayl) ist die Position EXAKT wirkungslos.
    # b) Dasselbe Tuch wirkt am Einlass um S_bp/S_Löcher stärker (nur
    #    die Lochfläche wird durchströmt) und liegt HINTER den Shunt-
    #    Volumina sowie in Serie mit der Einlassloch-Masse: interne
    #    Laufzeit steigt deutlich (0.64 -> 3.0), die rückwärtige
    #    Auslöschung bricht ein (Empfindlichkeit +38 %), die
    #    250-Hz-Auslöschung vertieft sich.
    # c) Der 3D-Löser teilt die Kette (_rear_chain_mats) und zeigt
    #    dieselbe Richtung.
    # d) K103 (großflächige Plattenlöcher): kleiner, gleichgerichteter
    #    Effekt. e) Gatter: Doppelmembran hat keinen Einlass.
    if _HAS_SCIPY:
        g24 = dict(architecture="single",
                   membrane_resonance_hz=2100.0, membrane_diameter=25.4e-3,
                   membrane_thickness=6e-6, membrane_tension=45.0,
                   air_gap=38.1e-6, backplate_diameter=23.9e-3,
                   backplate_thickness=3.125e-3, bias_voltage=50.0,
                   n_through_holes=48, through_hole_diameter=1.0e-3,
                   n_blind_holes=24, blind_hole_diameter=1.2e-3,
                   blind_hole_depth=1.5e-3, delay_length=3e-3,
                   cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
                   n_cavity_holes=60, cavity_hole_diameter=0.6e-3,
                   cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
                   fabric_rear_rayl=25.0, body_diameter=28e-3)
        # a) 0 Rayl: Position exakt wirkungslos
        z24 = {**g24, "fabric_rear_rayl": 0.0}
        f24 = np.array([100.0, 1000.0, 8000.0])
        Ha = MicrophoneCapsule(**z24, squeeze_model="2d",
                               fabric_rear_position="backplate"
                               ).transfer_function(f24)
        Hb = MicrophoneCapsule(**z24, squeeze_model="2d",
                               fabric_rear_position="inlet"
                               ).transfer_function(f24)
        assert np.allclose(Ha, Hb, rtol=1e-12, atol=0.0), \
            "0 Rayl: Gewebe-Position muss exakt wirkungslos sein"

        def _fab24(sm, pos):
            c = MicrophoneCapsule(**g24, squeeze_model=sm,
                                  fabric_rear_position=pos)
            H1 = abs(c.transfer_function(np.array([1000.0]))[0]) * 1e3
            rat = c.delay_diagnostics()["ratio"]
            p250 = c.directivity(
                frequencies_hz=(250.0,))["patterns"][250.0]["db"][180]
            return H1, rat, p250

        Hb2, rb2, pb2 = _fab24("2d", "backplate")
        Hi2, ri2, pi2 = _fab24("2d", "inlet")
        assert rb2 < 0.8 and ri2 > 2.0, \
            (f"Einlass-Gewebe muss die interne Laufzeit stark verlängern "
             f"({rb2:.2f} -> {ri2:.2f})")
        assert Hi2 > 1.2 * Hb2, \
            (f"Einlass-Gewebe muss die rückwärtige Auslöschung schwächen "
             f"({Hb2:.1f} -> {Hi2:.1f} mV/Pa)")
        # Schwelle 0.7 dB (vorher 1.0): mit der korrigierten Durchfluss-
        # Zellfunktion (_cell_B_flow, Gegenprobe 30) ist die interne
        # Laufzeit ohnehin länger, der Zusatzeffekt des Einlass-Gewebes
        # damit etwas kleiner. Die RICHTUNG — Vertiefung — ist unberührt.
        assert pi2 < pb2 - 0.7, \
            (f"250-Hz-Auslöschung muss sich vertiefen "
             f"({pb2:.1f} -> {pi2:.1f} dB)")
        # c) 3D teilt die Kette: gleiche Richtung
        Hb3, rb3, _ = _fab24("3d", "backplate")
        Hi3, ri3, _ = _fab24("3d", "inlet")
        assert ri3 > 2.0 and ri3 > rb3 + 1.5 and Hi3 > Hb3, \
            (f"3D muss die Einlass-Gewebe-Richtung teilen "
             f"(ratio {rb3:.2f} -> {ri3:.2f}, H {Hb3:.1f} -> {Hi3:.1f})")
        # d) K103: Plattenlöcher großflächig -> kleiner Effekt, gleiche
        # Richtung
        k24 = dict(g24, delay_length=0.0, cavity_length=0.0,
                   cavity_wall_thickness=0.0, n_cavity_holes=0,
                   cavity_hole_diameter=0.0, cavity_hole_axial_position=0.0,
                   rear_spacer_height=60e-6, rear_plate_thickness=2.5e-3,
                   n_rear_plate_holes=60, rear_plate_hole_diameter=1.0e-3)
        rk_b = MicrophoneCapsule(**k24, squeeze_model="2d",
                                 fabric_rear_position="backplate"
                                 ).delay_diagnostics()["ratio"]
        rk_i = MicrophoneCapsule(**k24, squeeze_model="2d",
                                 fabric_rear_position="inlet"
                                 ).delay_diagnostics()["ratio"]
        assert rk_i > rk_b + 0.01, \
            f"K103: gleiche Wirkrichtung erwartet ({rk_b:.2f} -> {rk_i:.2f})"
        # e) Gatter
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              membrane_resonance_hz=1150.0,
                              fabric_rear_position="inlet")
            raise AssertionError("dual_diaphragm+inlet müsste scheitern")
        except ValueError:
            pass
        print(f"Gewebe-Position: 0 Rayl exakt wirkungslos; Einlass statt "
              f"Backplate (25 Rayl): intern/extern {rb2:.2f} -> {ri2:.2f}, "
              f"Empf. {Hb2:.1f} -> {Hi2:.1f} mV/Pa, 250 Hz/180° {pb2:.1f} "
              f"-> {pi2:.1f} dB (3D gleichgerichtet: {rb3:.2f} -> "
              f"{ri3:.2f}); K103 {rk_b:.2f} -> {rk_i:.2f}; Gatter "
              f"Doppelmembran greift  OK")


@pytest.mark.feld3d
def test_gp25_eigenrauschen_fdt_nyquist_fit_frei():
    """Gegenprobe 25: Eigenrauschen (FDT/Nyquist, fit-frei)."""
    # a) METHODEN-VALIDIERUNG an einem selbstständigen Mini-ABCD-Netzwerk
    #    MIT Shunt-Zweigen (die Strom umleiten): das verallgemeinerte
    #    Nyquist-Ergebnis S_qq = 4kT·Re{Z_tot}/|Z_tot|² muss der
    #    BRUTE-FORCE-Superposition über JEDEN einzelnen Widerstand
    #    (Norton-Rauschquelle 4kT/R, Übertragung auf den Membranstrom)
    #    exakt entsprechen — beweist, dass die Port-Impedanz-Formel und
    #    die Theorem-Anwendung alle Widerstände korrekt wichten.
    # b) Die _membrane_port_impedance-Zerlegung Z_front+Z_mem+Z_rear muss
    #    die unabhängig gerechnete Treibpunkt-Impedanz Z_mem+Z_a+Z_b
    #    treffen (validiert die ABCD-Konventionen/Vorzeichen).
    # c) Kapsel-Plausibilität: Pegel endlich und positiv, Pfad-Anteile
    #    summieren zu 1, Re{Z_tot} >= 0 (Passivität); T -> 2T ergibt
    #    +3 dB (S_p ∝ T); Gatter: 3D hat keinen Membranzweig.
    om25 = np.array([2.0 * np.pi * 1000.0, 2.0 * np.pi * 5000.0])
    Rs, C1 = 3.0e6, 1.5e-12
    Rmem, Xmem, C2, Rr = 2.0e6, 4.0e6, 2.5e-12, 1.2e6
    kT4 = 4.0 * K_BOLTZ * T_KELVIN
    for om in om25:
        Zmem = Rmem + 1j * Xmem
        Za = 1.0 / (1.0 / Rs + 1j * om * C1)         # Front, Quelle kurz
        Zb = 1.0 / (1.0 / Rr + 1j * om * C2)         # Rück, Port kurz
        Z_tot = Zmem + Za + Zb
        S_nyq = kT4 * np.real(Z_tot) / abs(Z_tot) ** 2
        # Brute force: Knoten A, M (in Zmem zwischen R und jX), B
        Gs, Gm, Gr = 1.0 / Rs, 1.0 / Rmem, 1.0 / Rr
        Yx = 1.0 / (1j * Xmem)
        yc1, yc2 = 1j * om * C1, 1j * om * C2
        Y = np.array([
            [Gs + yc1 + Gm, -Gm,       0.0],
            [-Gm,           Gm + Yx,  -Yx],
            [0.0,          -Yx,        Yx + yc2 + Gr]], dtype=complex)
        Yinv = np.linalg.inv(Y)

        def _qmem(J):
            # Zweigstrom am QUELLENFREIEN Element jX_mem messen (M<->B):
            # die R_mem-Norton-Quelle sitzt parallel zu R_mem (A<->M) und
            # speist zusätzlich in den Zweig, daher wäre Gm·(V_A−V_M)
            # falsch — jX_mem trägt immer den vollen Membranstrom.
            V = Yinv @ J
            return Yx * (V[1] - V[2])

        S_bf = 0.0
        # Rs: Norton A<->gnd
        S_bf += abs(_qmem(np.array([1.0, 0, 0], complex))) ** 2 * kT4 * Gs
        # Rr: Norton B<->gnd
        S_bf += abs(_qmem(np.array([0, 0, 1.0], complex))) ** 2 * kT4 * Gr
        # R_mem: Norton A<->M
        S_bf += abs(_qmem(np.array([1.0, -1.0, 0], complex))) ** 2 * kT4 * Gm
        assert abs(S_bf - S_nyq) < 1e-9 * abs(S_nyq), \
            (f"Nyquist muss Brute-Force treffen ({S_bf:.3e} vs. "
             f"{S_nyq:.3e} bei {om / (2 * np.pi):.0f} Hz)")

    # b)+c) an einer echten Kapsel (Nieren-Single, 2D)
    ncap = MicrophoneCapsule(
        architecture="single", membrane_resonance_hz=2100.0,
        membrane_diameter=25.4e-3, membrane_thickness=6e-6,
        membrane_tension=45.0, air_gap=38.1e-6, backplate_diameter=23.9e-3,
        backplate_thickness=3.125e-3, bias_voltage=50.0, n_through_holes=48,
        through_hole_diameter=1.0e-3, n_blind_holes=24,
        blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        delay_length=3e-3, cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
        n_cavity_holes=60, cavity_hole_diameter=0.6e-3,
        cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, body_diameter=28e-3, squeeze_model="2d")
    om_c = np.array([2.0 * np.pi * 1000.0])
    Zt, Zf, Zm, Zr = ncap._membrane_port_impedance(om_c)
    assert np.real(Zt)[0] > 0.0, "Passivität: Re{Z_tot} muss >= 0 sein"
    sp = ncap.noise_spectrum(np.array([1000.0]))
    fr_sum = (sp["frac_front"] + sp["frac_mem"] + sp["frac_rear"])[0]
    assert abs(fr_sum - 1.0) < 1e-9, \
        f"Pfad-Anteile müssen zu 1 summieren ({fr_sum:.6f})"
    sn = ncap.self_noise()
    assert np.isfinite(sn["spl_a_db"]) and np.isfinite(sn["spl_z_db"])
    assert 0.0 < sn["spl_a_db"] < 40.0, \
        f"Ersatzgeräuschpegel unplausibel ({sn['spl_a_db']:.1f} dB-A)"
    assert sn["spl_z_db"] > sn["spl_a_db"], \
        "linear (Z) muss über A-bewertet liegen"
    # Pfad-Zerlegung: energetische Summe = Gesamtpegel
    p_sum = 10.0 * np.log10(
        10 ** (sn["spl_a_front_db"] / 10) + 10 ** (sn["spl_a_mem_db"] / 10)
        + 10 ** (sn["spl_a_rear_db"] / 10))
    assert abs(p_sum - sn["spl_a_db"]) < 0.05, \
        (f"Pfad-Zerlegung muss sich energetisch zum Gesamtpegel summieren "
         f"({p_sum:.2f} vs. {sn['spl_a_db']:.2f} dB-A)")
    # T -> 2T: +3 dB (S_p ∝ T). Modulkonstante temporär anheben.
    import microphone_capsule as _mc25
    _T0 = _mc25.T_KELVIN
    try:
        _mc25.T_KELVIN = 2.0 * _T0
        sn2 = ncap.self_noise()
    finally:
        _mc25.T_KELVIN = _T0
    assert abs((sn2["spl_a_db"] - sn["spl_a_db"]) - 10.0 * np.log10(2.0)) \
        < 0.02, "T -> 2T muss den Rauschpegel um +3 dB anheben"
    # d) Gatter 3D
    try:
        MicrophoneCapsule(
            architecture="dual_diaphragm", membrane_resonance_hz=1150.0,
            center_gap=50e-6, n_through_holes=12, through_hole_diameter=0.6e-3,
            squeeze_model="3d").self_noise()
        raise AssertionError("self_noise im 3D-Modus müsste scheitern")
    except ValueError:
        pass
    print(f"Eigenrauschen: Nyquist == Brute-Force (Mini-Netz, "
          f"{len(om25)} Frequenzen, rel < 1e-9); Nieren-Single "
          f"{sn['spl_a_db']:.1f} dB-A ({sn['spl_z_db']:.1f} dB lin) — "
          f"Front {sn['spl_a_front_db']:.1f} / Membranfilm "
          f"{sn['spl_a_mem_db']:.1f} / Rückpfad {sn['spl_a_rear_db']:.1f} "
          f"dB-A; T->2T +3.01 dB; Anteile summieren zu 1; 3D-Gatter "
          f"greift  OK")


@pytest.mark.bem
def test_gp46_laufzeit_und_rauschintegral():
    """Gegenprobe 46: Laufzeit und Rauschintegral."""
    # Meldung aus der Praxis: mit BEM-Kopf UND -Körper lief die K67 rund
    # 24 Minuten, davon 19 STILL hinter einem Fortschrittsbalken, der
    # schon 100 % zeigte. Dazu Warnungen "divide by zero encountered in
    # matmul" mitten in der Projektion. Beides hatte je eine eigene
    # Ursache, und beide sind hier festgenagelt.
    #
    # a) DOPPELTE LÖSUNGSGÄNGE. Teuer am BEM ist der Matrixaufbau, und der
    #    hängt NUR von ω ab — Einfallsrichtungen sind bloß weitere rechte
    #    Seiten. Der Frequenzgang rechnet mit θ = 0/90/180°, das
    #    Eigenrauschen danach mit θ = 0° über DASSELBE Raster: ohne
    #    Ergebnisspeicher wurde alles ein zweites Mal gelöst. Geprüft wird
    #    beides — dass der Speicher greift (Zähler) und dass er das
    #    Ergebnis nicht verändert (bitgleich, stückweise gerechnet).
    # b) DER STATISCHE KERN ist frequenzunabhängig und wurde je Frequenz
    #    neu gebaut. Jetzt einmal je Geometrie.
    # c) WARNUNGEN. LAPACK setzt auf manchen Plattformen Gleitkomma-Flags,
    #    die numpy erst beim nächsten ufunc meldet. Der Lösungsgang ist
    #    deshalb gekapselt und prüft sein ERGEBNIS; ein wirklich
    #    unlösbares System wirft jetzt einen benannten Fehler.
    # d) DAS RAUSCHINTEGRAL war unabhängig davon zu grob. S_p = S_v/|H|²
    #    hat Spitzen dort, wo die Kapsel TAUB ist — an einer Antiresonanz
    #    von |H|. Die Prüfkapsel hat eine mit Q ≈ 375; ein logarithmisches
    #    Raster über drei Dekaden trifft sie nicht, und die bisherigen
    #    1200 Punkte lagen 0.53 dB zu tief. Eine lokale Nachverfeinerung
    #    bringt das auf 0.0002 dB.
    if _HAS_SCIPY:
        g46 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3, squeeze_model="1d",
            axial_body_model="bem")
        c46 = MicrophoneCapsule(**g46)          # Kopf UND Körper
        n46 = c46._bem_geometry()["elems"]["L"].size
        assert n46 > 100, f"Kopf+Körper muss viele Elemente haben ({n46})"
        f46 = np.array([200.0, 1000.0, 5000.0, 12000.0])
        om46 = 2.0 * np.pi * f46
        th46 = np.array([0.0, 0.5 * np.pi, np.pi])
        c46._bem_solves = 0
        F46, G46 = c46._bem_front_modes(om46, th46)
        s_voll = c46._bem_solves
        assert s_voll == f46.size, \
            f"vier Frequenzen = vier Lösungsgänge ({s_voll})"
        # a1) derselbe Satz noch einmal: KEIN Lösungsgang mehr
        c46._bem_cache = None                    # Identitäts-Abkürzung aus
        F46b, G46b = c46._bem_front_modes(om46, th46)
        assert c46._bem_solves == s_voll, \
            f"Wiederholung darf nicht neu lösen ({c46._bem_solves})"
        assert np.array_equal(F46, F46b) and np.array_equal(G46, G46b), \
            "der Speicher muss bitgleich dasselbe liefern"
        # a2) Teilmenge der Winkel (so ruft noise_spectrum auf): auch frei
        c46._bem_cache = None
        F46c, _ = c46._bem_front_modes(om46, th46[:1])
        assert c46._bem_solves == s_voll, \
            "eine Teilmenge der Winkel darf nichts kosten"
        assert np.array_equal(F46c[:, :, 0], F46[:, :, 0]), \
            "Teilmenge muss dieselben Zahlen liefern"
        # a3) stückweise gerechnet == in einem Zug
        c47 = MicrophoneCapsule(**g46)
        F47a, G47a = c47._bem_front_modes(om46[:2], th46)
        c47._bem_cache = None
        F47b, G47b = c47._bem_front_modes(om46[2:], th46)
        c47._bem_cache = None
        F47, G47 = c47._bem_front_modes(om46, th46)
        assert (np.array_equal(F47[:, :2], F47a)
                and np.array_equal(F47[:, 2:], F47b)
                and np.array_equal(G47[:2], G47a)
                and np.array_equal(G47[2:], G47b)), \
            "stückweise gerechnet muss bitgleich sein"
        # b) der statische Kern liegt an der Geometrie, nicht an der Frequenz
        assert "_K0" in c46._bem_geometry(), \
            "der statische Kern muss je Geometrie zwischengespeichert sein"
        assert c46._bem_solid_angle_residual < 5e-3, "Gitterqualität"

        # d) Rauschintegral: die Spitze ist real, das grobe Raster trifft
        #    sie nicht, die Verfeinerung schon.
        c48 = MicrophoneCapsule()               # Nieren-Single, 1D, schnell
        f48 = np.logspace(np.log10(20.0), np.log10(20000.0), 60000)
        S48 = c48.noise_spectrum(f48)["psd_pa2_hz"] * c48._a_weighting(f48)**2
        i48 = int(np.argmax(S48))
        halb = f48[S48 > 0.5 * S48[i48]]
        breite = float(halb.max() - halb.min()) / float(f48[i48])
        H48 = np.abs(c48.transfer_function(f48))
        assert abs(float(f48[int(np.argmin(H48))]) / float(f48[i48]) - 1.0) \
            < 1e-3, "die Spitze muss auf der Antiresonanz von |H| sitzen"
        assert breite < 0.01, \
            (f"die Spitze ist schmal — das ist der ganze Punkt "
             f"({100 * breite:.2f} % der Frequenz)")
        ref48 = c48.self_noise(n_points=16000, refine=False)["spl_a_db"]
        roh48 = c48.self_noise(refine=False)["spl_a_db"]
        fein48 = c48.self_noise()["spl_a_db"]
        assert ref48 - roh48 > 0.4, \
            (f"ohne Verfeinerung MUSS das Integral zu tief liegen — sonst "
             f"prüft das hier nichts ({roh48:.3f} gegen {ref48:.3f} dB-A)")
        assert abs(fein48 - ref48) < 0.02, \
            (f"mit Verfeinerung muss der A-Pegel die Referenz treffen "
             f"({fein48:.4f} gegen {ref48:.4f} dB-A)")
        # und auf dem gröberen Raster der Anzeige ebenfalls brauchbar
        sp48 = c48.noise_spectrum(
            np.logspace(np.log10(10.0), np.log10(25000.0), 400))
        grob48 = c48.self_noise(spectrum=sp48)["spl_a_db"]
        assert abs(grob48 - ref48) < 0.1, \
            (f"auch das Anzeigeraster muss mit Verfeinerung treffen "
             f"({grob48:.4f} gegen {ref48:.4f} dB-A)")
        print(f"Laufzeit und Rauschintegral: BEM Kopf+Körper {n46} Elemente, "
              f"{s_voll} Frequenzen = {s_voll} Lösungsgänge; Wiederholung "
              f"und Winkel-Teilmenge kosten nichts mehr und sind bitgleich, "
              f"stückweise == am Stück; statischer Kern je Geometrie. "
              f"Rauschspitze {100 * breite:.2f} % breit auf der "
              f"|H|-Antiresonanz: ohne Verfeinerung {roh48:.2f}, mit "
              f"{fein48:.4f} gegen Referenz {ref48:.4f} dB-A (Anzeigeraster "
              f"{grob48:.3f})  OK")
