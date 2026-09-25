"""Gegenproben: 3D-Feldlöser."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


@pytest.mark.feld3d
def test_gp16_3d_r_phi_loser_diskrete(deb3, deb3b):
    """Gegenprobe 16: 3D-(r,phi)-Löser (diskrete Löcher)."""
    # a) Gültigkeits-Gatter: '3d' erfordert Durchgangslöcher; Stufen-
    #    bohrungen der Doppelmembran-Bauform erfordern center_gap > 0
    #    (zwei Elektrodenhälften, K67-Typ, s. Gegenprobe 22). single/dual
    #    rechnet der Löser seit Gegenprobe 23 ebenfalls.
    # b) Reziprozität des Feldsystems: Frontantwort auf Rückdruck ==
    #    Rückantwort auf Frontdruck (±3 %).
    # c) Physik: Null bei 180°; ein Freistich, der ALLE Loch-Mündungen
    #    abdeckt, vertieft die Auslöschung gegenüber dem reinen Rand-Ring
    #    deutlich (Mündungs-Engstellen-Mechanismus, s. Gegenprobe 15).
    if _HAS_SCIPY:
        try:
            MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                              n_blind_holes=30)
            raise AssertionError("'3d' ohne Durchgangslöcher müsste "
                                 "scheitern")
        except ValueError:
            pass
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              center_gap=0.0, through_holes_stepped=True,
                              n_blind_holes=120, blind_hole_depth=2.0e-3,
                              squeeze_model="3d",
                              membrane_resonance_hz=2100.0)
            raise AssertionError("'3d' gestuft ohne center_gap müsste "
                                 "scheitern")
        except ValueError:
            pass
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              center_gap=50e-6, n_through_holes=0,
                              squeeze_model="3d",
                              membrane_resonance_hz=2100.0)
            raise AssertionError("'3d' ohne Durchgangslöcher müsste "
                                 "scheitern")
        except ValueError:
            pass
        Xf3, Xr3, Bf3, Br3 = deb3._solve_3d(
            np.array([2.0 * np.pi * 1000.0]), want_rear=True,
            weight="volume")
        assert 0.97 < abs(Xr3[0]) / abs(Bf3[0]) < 1.03, \
            "3D-Feldsystem muss reziprok sein"
        di3 = deb3.directivity(frequencies_hz=(1000.0,))
        p3 = di3["patterns"][1000.0]
        na3 = di3["angles_deg"][:181][int(np.argmin(p3["linear"][:181]))]
        assert na3 > 172.0, f"3D-Niere muss bei 180° nullen ({na3:.0f}°)"
        p3b = deb3b.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
        assert p3b < p3["db"][180] - 6.0, \
            (f"Mündungs-Freistich muss die 3D-Null deutlich vertiefen "
             f"({p3['db'][180]:.1f} -> {p3b:.1f})")
        fr3 = deb3.frequency_response(f_min=100.0, f_max=8000.0,
                                      n_points=8)
        assert np.all(np.isfinite(fr3["amplitude_db"]))
        print(f"3D-Löser: Gatter greifen, reziprok "
              f"({abs(Xr3[0]) / abs(Bf3[0]):.3f}), Null @{na3:.0f}°; "
              f"Mündungs-Freistich vertieft 180°/1 kHz von "
              f"{p3['db'][180]:.1f} auf {p3b:.1f} dB  OK")


# --------- Gegenprobe 22: 3D-K67-Modus (Zwischenspalt, Stufen, Drehung) ----
# Der 3D-Löser rechnet auch die ZWEITEILIGE Elektrode: Zwischenspalt als
# dritter Reynolds-Film, Stufenbohrungen als Zweitor-Kette je Loch,
# Elektrodenhälften gegeneinander verdreht. Grenzfälle und Physik ohne
# Fit-Koeffizient, aufgeteilt in unabhängige Teilprüfungen (parallel
# lauffähig; zusammen waren es über drei Minuten in einem Test).


def _deb22():
    """Debenham ohne Sacklöcher, 3D. (45 V: ohne Sacklöcher ist die
    Elektrode fast voll, der exakte Pull-in liegt dann bei 49.2 V —
    Gegenprobe 49; a/b prüfen Struktur, nicht die Nähe zum Kollaps.)"""
    deb22 = dict(DEB_KWARGS)
    deb22.update(squeeze_model="3d", bias_voltage=45.0,
                 blind_hole_rings=[(0, None)],
                 blind_hole_diameter=1.2e-3, blind_hole_depth=1e-3)
    return deb22


_OM22 = np.array([2.0 * np.pi * 500.0, 2.0 * np.pi * 2000.0])
# komplette K67 im 3D-Feld
_K67_3D = dict(K67_KWARGS, squeeze_model="3d")


def _pattern22(cap):
    """180°-Pegel und Minimum-Winkel des 1-kHz-Patterns."""
    di = cap.directivity(frequencies_hz=(1000.0,))
    db = di["patterns"][1000.0]["db"]
    na = di["angles_deg"][:181][
        int(np.argmin(di["patterns"][1000.0]["linear"][:181]))]
    return db[180], na


@pytest.fixture(scope="module")
def k67_3d_verdreht():
    """K67 mit AUTOMATISCHER Verdrehung (kreisweise, s. _hole_positions)
    und ihr 1-kHz-Pattern. Bis Gegenprobe 48 stand hier pauschal 3°
    (= 180°/60), was auf dem Mehrkreis-Raster die Kerne im Zwischenspalt
    fast übereinanderlegt (Gegenprobe 48 c) — also praktisch
    ausgerichtet."""
    if not _HAS_SCIPY:
        pytest.skip("SciPy fehlt (3D-Feldlöser)")
    cap = MicrophoneCapsule(**_K67_3D, half_rotation_deg=None)
    return cap, _pattern22(cap)


@pytest.mark.feld3d
def test_gp22a_einteiliger_grenzfall():
    """Gegenprobe 22 a: zweiteilig ausgerichtet (rot = 0) mit winzigem
    Zwischenspalt (5 µm) reproduziert den einteiligen Löser — zwei völlig
    verschiedene Codepfade: Lochpaar-Leitwert gegen die Kette
    Loch–Film–Loch. Die Restabweichung ist die Impedanz des 5-µm-Films
    selbst."""
    if not _HAS_SCIPY:
        return
    deb22 = _deb22()
    ein = MicrophoneCapsule(**{**deb22, "center_gap": 0.0})
    zwei = MicrophoneCapsule(**{**deb22, "center_gap": 5e-6,
                                "half_rotation_deg": 0.0})
    Xf_e, Xr_e = ein._solve_3d(_OM22)
    Xf_z, Xr_z = zwei._solve_3d(_OM22)
    dev_a = max(float(np.max(np.abs(Xf_z / Xf_e - 1.0))),
                float(np.max(np.abs(Xr_z / Xr_e - 1.0))))
    assert dev_a < 0.04, \
        (f"3D-K67-Modus muss im 5-µm-Grenzfall den einteiligen Löser "
         f"reproduzieren (Abweichung {dev_a:.3f})")
    print(f"3D-K67-Modus a) einteiliger Grenzfall {dev_a * 100:.1f} %  OK")


@pytest.mark.feld3d
def test_gp22b_stufen_grenzfall():
    """Gegenprobe 22 b: Stufenbohrung mit winziger Senkung reproduziert
    die ungestufte Bohrung. Die Senkung ist nur 1 µm weiter als der Kern:
    seit Gegenprobe 48 zählen die Fußabdrücke nach exaktem Abstand, und
    die 0.71-mm-Löcher sind hier kleiner als eine Gitterzelle — schon
    0.75 mm holten die Nachbarzelle dazu. Geprüft wird die KETTE, nicht
    die Auflösung."""
    if not _HAS_SCIPY:
        return
    deb22 = _deb22()
    plain = MicrophoneCapsule(**{**deb22, "center_gap": 50e-6,
                                 "half_rotation_deg": 0.0})
    step = MicrophoneCapsule(**{**deb22, "center_gap": 50e-6,
                                "half_rotation_deg": 0.0,
                                "through_holes_stepped": True,
                                "blind_hole_rings": [(12, None)],
                                "blind_hole_diameter": 0.711e-3,
                                "blind_hole_depth": 0.15e-3})
    Xf_p, Xr_p = plain._solve_3d(_OM22)
    Xf_s, Xr_s = step._solve_3d(_OM22)
    dev_b = max(float(np.max(np.abs(Xf_s / Xf_p - 1.0))),
                float(np.max(np.abs(Xr_s / Xr_p - 1.0))))
    assert dev_b < 0.03, \
        (f"Stufenbohrung mit winziger Senkung muss die ungestufte "
         f"Bohrung reproduzieren (Abweichung {dev_b:.3f})")
    print(f"3D-K67-Modus b) Stufen-Grenzfall {dev_b * 100:.1f} %  OK")


@pytest.mark.feld3d
def test_gp22c_verdrehung(k67_3d_verdreht):
    """Gegenprobe 22 c: VERDREHUNG. Die reale K67 verdreht die Hälften so,
    dass die Durchgangslöcher nicht zueinander zeigen. Ausgerichtete
    Löcher (rot = 0) schließen den Phasenschieber kurz -> flache
    Auslöschung; versetzte zwingen den Pfad durch den Zwischenspalt-Film
    -> tiefe 180°-Null, das Minimum wandert nach hinten."""
    cap0 = MicrophoneCapsule(**_K67_3D, half_rotation_deg=0.0)
    p0, na0 = _pattern22(cap0)
    _, (pv, nav) = k67_3d_verdreht
    assert p0 > -12.0, \
        (f"ausgerichtete Löcher müssen den Phasenschieber kurz-"
         f"schließen (180° = {p0:.1f} dB)")
    assert nav > na0 + 15.0, \
        (f"Verdrehung muss das Minimum Richtung 180° schieben "
         f"({na0:.0f}° -> {nav:.0f}°)")
    assert nav >= 170.0, \
        f"versetzte Hälften: Nierenminimum hinten ({nav:.0f}°)"
    print(f"3D-K67-Modus c) Verdrehung 0°->automatisch: 180° {p0:.1f} -> "
          f"{pv:.1f} dB, Minimum {na0:.0f}° -> {nav:.0f}°  OK")


@pytest.mark.feld3d
def test_gp22d_reziprok(k67_3d_verdreht):
    """Gegenprobe 22 d: das Feldsystem der kompletten K67 (gestuft,
    verdreht) ist reziprok."""
    cap, _ = k67_3d_verdreht
    Xf_r, Xr_r, Bf_r, Br_r = cap._solve_3d(
        np.array([2.0 * np.pi * 1000.0]), want_rear=True, weight="volume")
    rez22 = abs(Xr_r[0]) / abs(Bf_r[0])
    assert 0.97 < rez22 < 1.03, \
        f"3D-K67-Feldsystem muss reziprok sein ({rez22:.3f})"
    print(f"3D-K67-Modus d) reziprok ({rez22:.4f})  OK")


@pytest.mark.feld3d
def test_gp22e_kernlage_gegen_2d(k67_3d_verdreht):
    """Gegenprobe 22 e: gegen das homogenisierte 2D-Modell.

    Die EMPFINDLICHKEIT stimmt; die TIEFE der Auslöschung hängt dagegen
    an einem Maß, das niemand dokumentiert hat: wie die Kerne beider
    Hälften im 50-µm-Zwischenspalt zueinander liegen. Mit konturtreuen
    Mündungen (Gegenprobe 51, grob und fein auf ~0.5 dB gleich) und
    Θ-konsistenter Wandlung (Gegenprobe 54): vollständig versetzt
    (automatisch: jeder Kern über einer Sacksenkung der Gegenseite, ~2 mm
    Querweg) −17 dB, global 6°/9°/12°/15°/18° −10/−19/−37/−31/−14 dB.
    Zwischen 9° und 12° liegt eine Kernlage, die das 2D-Modell (−28 dB)
    trifft, dessen Škvor-Zelle im Zwischenspalt einen mittleren Querweg
    von etwa einem Zellradius annimmt. (Vor der Θ-konsistenten Wandlung
    traf 12° mit −29 dB; die Auslöschung reagiert auf jede Gewichtung von
    Front- gegen Rückantrieb, und bei 60 V gewichtet 1/g² die Mitte. Vor
    der Konturkorrektur: versetzt −11 dB, 9° −29 dB.) Das ist KEIN
    Modellfehler, sondern eine offene Geometriefrage an der realen
    Kapsel — festgehalten, damit sie nicht wieder als gelöst gilt
    (Gegenprobe 48).
    """
    cap, (pv, _) = k67_3d_verdreht
    ev = abs(cap.transfer_function(np.array([1000.0]))[0]) * 1e3
    k2d = MicrophoneCapsule(**{**_K67_3D, "squeeze_model": "2d"})
    e2d = abs(k2d.transfer_function(np.array([1000.0]))[0]) * 1e3
    p2d = k2d.directivity(
        frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
    assert abs(20.0 * np.log10(ev / e2d)) < 3.0, \
        (f"3D verdreht muss nahe der 2D-Empfindlichkeit liegen "
         f"({ev:.1f} vs. {e2d:.1f} mV/Pa)")
    p09, p12 = (MicrophoneCapsule(**_K67_3D, half_rotation_deg=rot_).
                directivity(frequencies_hz=(1000.0,))
                ["patterns"][1000.0]["db"][180] for rot_ in (9.0, 12.0))
    assert p12 < p2d < p09, \
        (f"zwischen 9° und 12° muss eine Kernlage die 2D-Auslöschung "
         f"treffen ({p09:.1f} > {p2d:.1f} > {p12:.1f} dB)")
    assert p12 < pv - 8.0, \
        (f"die Auslöschung MUSS von der Lage der Kerne abhängen "
         f"(12°: {p12:.1f} dB, versetzt: {pv:.1f} dB)")
    print(f"3D-K67-Modus e) Empf. verdreht {ev:.1f} mV/Pa (2D {e2d:.1f}); "
          f"Auslöschung hängt an der Kernlage: 9°/12° {p09:.1f}/"
          f"{p12:.1f} dB, 2D {p2d:.1f} dB dazwischen — offene "
          f"Geometriefrage  OK")


# --------- Gegenprobe 23: 3D-Löser für single/dual-Architekturen ----------
# Der 3D-Löser rechnet auch die Einzel-Backplate- und die Dual-Backplate-
# Bauform: EIN Membranfeld, ein Film je Backplate, Durchgangslöcher als
# Zweitor in SAMMELKNOTEN, deren Abschluss die baugleiche Lumped-Kette des
# 1D/2D-Pfads bildet (_rear_chain_mats; vorn: Strahlung + Gewebe).
# Verankert ohne Fit-Koeffizient, in unabhängigen Teilprüfungen a)–f).
_F23 = np.array([100.0, 1000.0])
_SG23 = dict(architecture="single",
             membrane_material="PET", membrane_resonance_hz=8000.0,
             membrane_diameter=22e-3, membrane_thickness=6e-6,
             membrane_tension=400.0, air_gap=40e-6,
             backplate_diameter=20e-3, backplate_thickness=3e-3,
             bias_voltage=60.0, n_through_holes=60,
             through_hole_diameter=1.0e-3, n_blind_holes=30,
             blind_hole_diameter=1.2e-3, delay_length=3e-3,
             cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
             n_cavity_holes=200, cavity_hole_diameter=0.2e-3,
             cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
             fabric_rear_rayl=0.0, body_diameter=24e-3,
             include_diffraction=False)
# K103: Spacer + Rückplatte
_K23 = dict(architecture="single",
            membrane_resonance_hz=1800.0, membrane_diameter=25.4e-3,
            membrane_thickness=6e-6, membrane_tension=100.0,
            air_gap=50e-6, backplate_diameter=24e-3,
            backplate_thickness=3e-3, bias_voltage=60.0,
            n_through_holes=36, through_hole_diameter=1.0e-3,
            n_blind_holes=0, blind_hole_diameter=1.2e-3,
            blind_hole_depth=1e-3, rear_spacer_height=40e-6,
            rear_plate_thickness=2e-3,
            rear_plate_hole_diameter=1.0e-3, delay_length=0.0,
            cavity_length=0.0, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=27e-3, include_diffraction=False)


@pytest.mark.feld3d
def test_gp23a_reziprok_single():
    """Gegenprobe 23 a: REZIPROZITÄT der akustischen Ports,
    q_rück(p_front = 1) == q_front(p_rück = 1) — exakt
    (Maschinengenauigkeit), validiert alle Kopplungsvorzeichen
    (Film-Membran, Knoten, Ketten). Offene Niere, fordert alle Pfade;
    bei geschlossener Rückseite ist der Kreuzfluss trivial 0."""
    if not _HAS_SCIPY:
        return
    card3 = MicrophoneCapsule(**{**_SG23, "n_cavity_holes": 60,
                                 "cavity_hole_diameter": 0.6e-3},
                              squeeze_model="3d")
    card3.transfer_function(np.array([1000.0]))
    rez_s = abs(card3._recip_3d[0]) / abs(card3._recip_3d[1])
    assert abs(rez_s - 1.0) < 1e-6, \
        f"3D single muss reziprok sein ({rez_s:.8f})"
    print(f"3D single/dual a) reziprok single {rez_s:.6f}  OK")


@pytest.mark.feld3d
def test_gp23b_geschlossen_kugel():
    """Gegenprobe 23 b: GESCHLOSSENE RÜCKSEITE — exakte Kugel (ohne
    Beugung) und Übereinstimmung mit 1D nach Betrag UND PHASE: die
    3D-Ausgänge folgen der Ketten-Flussrichtung (Vorzeichenkonvention)."""
    if not _HAS_SCIPY:
        return
    cl3 = MicrophoneCapsule(**{**_SG23, "n_cavity_holes": 0},
                            squeeze_model="3d")
    cl1 = MicrophoneCapsule(**{**_SG23, "n_cavity_holes": 0},
                            squeeze_model="1d")
    lin_c = cl3.directivity(
        frequencies_hz=(1000.0,))["patterns"][1000.0]["linear"]
    assert np.max(np.abs(lin_c - 1.0)) < 1e-6, \
        "3D single geschlossen muss exakte Kugel liefern"
    r_cl = (cl3.transfer_function(_F23[:1])
            / cl1.transfer_function(_F23[:1]))[0]
    assert 0.90 < abs(r_cl) < 1.02, \
        f"3D/1D geschlossen @100 Hz ({abs(r_cl):.3f})"
    assert abs(np.angle(r_cl)) < np.deg2rad(12.0), \
        (f"3D muss der Ketten-Phasenkonvention folgen "
         f"({np.rad2deg(np.angle(r_cl)):.1f}°)")
    print(f"3D single/dual b) geschlossen = Kugel, 3D/1D @100 Hz "
          f"{abs(r_cl):.3f} ∠{np.rad2deg(np.angle(r_cl)):+.1f}°  OK")


@pytest.mark.feld3d
def test_gp23c_k103_dicht():
    """Gegenprobe 23 c: K103-GRENZFALL DICHT (Spacer + Rückplatte ohne
    Durchlass): 3D == 1D auf wenige Prozent über das Band.

    Die Struktur wird an der VOLUMENverschiebung geprüft; die
    Θ-konsistente Wandlung prüft Gegenprobe 54. Die Spannung trifft die
    Kette im Tiefton; bei 1 kHz liegt sie im 3D tiefer: bei 60 V ist der
    Spalt in der Mitte nur 0.74·h, der Film dort 2.5-fach steifer, die
    Mitte bleibt zurück — und die Spannung gewichtet die Mitte (1/g²).
    Formanpassung, die das Einmodenbild nicht kann (ausgegeben)."""
    if not _HAS_SCIPY:
        return
    kd3 = MicrophoneCapsule(**_K23, n_rear_plate_holes=0,
                            squeeze_model="3d")
    kd1 = MicrophoneCapsule(**_K23, n_rear_plate_holes=0,
                            squeeze_model="1d")
    h1_kd = kd1.transfer_function(_F23)
    pf_kd, pr_kd = kd3._source_pressures(2 * np.pi * _F23,
                                         np.array([0.0]))
    Xf_kd, Xr_kd = kd3._solve_3d(2 * np.pi * _F23, weight="volume")
    r_kdv = np.abs((Xf_kd * pf_kd[:, 0] + Xr_kd * pr_kd[:, 0])
                   / (h1_kd / kd1._theta))
    assert np.all((r_kdv > 0.93) & (r_kdv < 1.07)), \
        f"K103 dicht: 3D-Volumenfluss muss 1D treffen ({r_kdv})"
    r_kd = np.abs(kd3.transfer_function(_F23) / h1_kd)
    assert abs(r_kd[0] - 1.0) < 0.03, \
        f"K103 dicht: Spannung im Tiefton == 1D ({r_kd[0]:.3f})"
    print(f"3D single/dual c) K103 dicht Volumen {r_kdv.round(3)}, "
          f"Spannung {r_kd.round(3)} (1 kHz: Formanpassung bei 60 V)  OK")


@pytest.mark.feld3d
def test_gp23d_k103_offen():
    """Gegenprobe 23 d: K103 OFFEN — die interne Rück-Übertragung D_r
    (das Verhältnis beider Pfade) stimmt mit dem 2D-Feldmodell auf ~1 %
    überein. Die absoluten Empfindlichkeiten tragen die dokumentierte
    Membranfeld-Klasse (±2–3 dB), ihr VERHÄLTNIS ist robust."""
    if not _HAS_SCIPY:
        return
    dr23 = {}
    for sm in ("2d", "3d"):
        ko = MicrophoneCapsule(**_K23, n_rear_plate_holes=60,
                               squeeze_model=sm)
        dr23[sm] = ko.angle_responses(_F23)["D_r"]
    d_dr = np.max(np.abs(dr23["3d"] - dr23["2d"]))
    assert d_dr < 0.05, \
        (f"K103 offen: interne Rück-Übertragung D_r muss das "
         f"2D-Feldmodell treffen (|ΔD_r| = {d_dr:.3f})")
    print(f"3D single/dual d) K103 offen |ΔD_r| = {d_dr:.3f}  OK")


@pytest.mark.feld3d
def test_gp23e_niere():
    """Gegenprobe 23 e: NIERE (Laufzeitglied + Hohlraum) — Richtdiagramm
    3D nahe 2D (90°/180°/Minimum-Winkel), Empfindlichkeit in der
    Klasse."""
    if not _HAS_SCIPY:
        return
    n23 = dict(architecture="single",
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
               fabric_rear_rayl=0.0, body_diameter=28e-3)
    pat23 = {}
    for sm in ("2d", "3d"):
        cn = MicrophoneCapsule(**n23, squeeze_model=sm)
        di = cn.directivity(frequencies_hz=(1000.0,))
        db = di["patterns"][1000.0]["db"]
        lin = di["patterns"][1000.0]["linear"]
        na = di["angles_deg"][:181][int(np.argmin(lin[:181]))]
        H1 = abs(cn.transfer_function(np.array([1000.0]))[0])
        pat23[sm] = (db[90], db[180], na, H1)
    assert abs(pat23["3d"][0] - pat23["2d"][0]) < 1.5, \
        (f"Niere 90°: 3D nahe 2D ({pat23['2d'][0]:.1f} vs. "
         f"{pat23['3d'][0]:.1f} dB)")
    assert abs(pat23["3d"][1] - pat23["2d"][1]) < 2.5, \
        (f"Niere 180°: 3D nahe 2D ({pat23['2d'][1]:.1f} vs. "
         f"{pat23['3d'][1]:.1f} dB)")
    assert abs(pat23["3d"][2] - pat23["2d"][2]) <= 15.0, \
        (f"Minimum-Winkel: 3D nahe 2D ({pat23['2d'][2]:.0f}° vs. "
         f"{pat23['3d'][2]:.0f}°)")
    r_e = pat23["3d"][3] / pat23["2d"][3]
    assert 0.6 < r_e < 1.05, \
        f"Niere Empfindlichkeit 3D/2D @1 kHz ({r_e:.2f})"
    print(f"3D single/dual e) Niere 90/180/Min: 2D {pat23['2d'][0]:.1f}/"
          f"{pat23['2d'][1]:.1f}/{pat23['2d'][2]:.0f}° vs. 3D "
          f"{pat23['3d'][0]:.1f}/{pat23['3d'][1]:.1f}/"
          f"{pat23['3d'][2]:.0f}°, Empf. {r_e:.2f}  OK")


@pytest.mark.feld3d
def test_gp23f_dual():
    """Gegenprobe 23 f: DUAL — Reziprozität + LF-Empfindlichkeit nahe
    2D."""
    if not _HAS_SCIPY:
        return
    du23 = dict(_SG23, architecture="dual", n_cavity_holes=200,
                cavity_hole_diameter=0.2e-3)
    du3 = MicrophoneCapsule(**du23, squeeze_model="3d")
    du2 = MicrophoneCapsule(**du23, squeeze_model="2d")
    r_du = (du3.transfer_function(_F23[:1])
            / du2.transfer_function(_F23[:1]))[0]
    rez_d = abs(du3._recip_3d[0]) / abs(du3._recip_3d[1])
    assert abs(rez_d - 1.0) < 1e-6, \
        f"3D dual muss reziprok sein ({rez_d:.8f})"
    assert 0.75 < abs(r_du) < 1.05, \
        f"dual: 3D/2D @100 Hz ({abs(r_du):.3f})"
    print(f"3D single/dual f) reziprok dual {rez_d:.6f}, 3D/2D @100 Hz "
          f"{abs(r_du):.3f}  OK")


@pytest.mark.feld3d
def test_gp47_ringmembran_im_3d_feldloser():
    """Gegenprobe 47: Ringmembran im 3D-Feldlöser."""
    # Gegenprobe 45 hat die Mittenterminierung in 1D/2D gebracht und den
    # 3D-Löser gesperrt: dort sind die Membranen FD-FELDER auf einem
    # (r,phi)-Gitter, und ein innerer Rand war nicht gebaut. Jetzt ist er
    # gebaut — und zwar OHNE Fallunterscheidung.
    #
    # DER TRICK. Der radiale Flächenleitwert einer Fläche bei r ist
    # T·r·dphi/dr. Beginnt das Gitter am Pfostenrand r0, so hat die
    # innerste Fläche den Radius r0; sie liegt eine halbe Zelle vor der
    # ersten Zellmitte, trägt also den doppelten Leitwert auf der
    # Diagonalen — das ist die EINGESPANNTE Wand. Für r0 = 0 wird
    # derselbe Term null, und das ist exakt die Achsenbedingung. Ein
    # Ausdruck, zwei Randbedingungen. Ebenso beim Film: dessen innerste
    # Fläche trägt schon immer nichts, und eine Wand tut dasselbe.
    #
    # a) r_i = 0 ist bitgleich der bisherige Stand (Gitter, Flächendichte,
    #    Spannung, Eigenwert).
    # b) DER OPERATOR GEGEN DIE EXAKTE LÖSUNG: eine statische
    #    Gleichlast auf das reine Membranfeld muss die geschlossene
    #    Ring-Nachgiebigkeit π·r_a⁴·g(ρ)/(8T) treffen, und zwar
    #    GITTERKONVERGENT (zweiter Ordnung). Das prüft Innenrand,
    #    Flächenradien und Gitterstart in einem.
    # c) Auch die FORM muss stimmen, nicht nur ihr Integral.
    # d) STRUKTUR: mit Pfosten muss die innerste Zelle fast still stehen
    #    (halbe Zelle vor der Wand), ohne Pfosten ist sie das Maximum.
    # e) ENDE ZU ENDE: 3D und 2D müssen mit Pfosten so gut
    #    zusammenpassen wie ohne.
    if _HAS_SCIPY:
        from scipy.sparse import coo_matrix as _coo47
        from scipy.sparse.linalg import spsolve as _spsolve47
        g47 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            architecture="dual_diaphragm", center_gap=50e-6,
            n_through_holes=12, through_hole_diameter=0.6e-3,
            n_blind_holes=24, blind_hole_diameter=1.3e-3,
            blind_hole_depth=3.7e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, squeeze_model="3d")

        # a) Grenzfall: die alten geschlossenen Formeln, exakt
        c47a = MicrophoneCapsule(bias_voltage=60.0, **g47)
        ga = c47a._g3d
        assert ga["q0"] == 0.0 and ga["dr"] == c47a.a_bp / ga["Nr"], \
            "ohne Pfosten muss das Gitter bei 0 beginnen"
        assert np.array_equal(
            ga["r_f"], (np.arange(ga["Nr"]) + 0.5) * ga["dr"]), \
            "ohne Pfosten bitgleich das alte Filmgitter"
        assert ga["sigma"] == c47a.mat_rho * c47a.t_mem, \
            "die Flächendichte des Felds ist ρ·t"
        assert ga["T_mem"] == ga["sigma"] * (
            2.0 * np.pi * c47a.f_res * c47a.a_mem
            / 2.404825557695773) ** 2, \
            "ohne Pfosten muss die Spannung aus j01 = 2.4048 kommen"

        # b/c/d) reines Membranfeld gegen die geschlossene Ringlösung
        def _mem47(cap, n_r):
            """Statische Gleichlast auf das Membranfeld allein."""
            cap._n_r_3d = n_r
            cap._n_phi_3d = 8                 # azimutal irrelevant, spart Zeit
            cap._build_3d_geometry()
            gg = cap._g3d
            Np7, NM7 = gg["Np"], gg["NM"]
            ow = gg["n_films"] * gg["NF"]
            rr, cc, vv = gg["static"]
            msk = ((rr >= ow) & (rr < ow + NM7)
                   & (cc >= ow) & (cc < ow + NM7))
            Lm = _coo47((vv[msk].real, (rr[msk] - ow, cc[msk] - ow)),
                        shape=(NM7, NM7)).tocsc()
            rhs = np.repeat(gg["A_m"], Np7)
            w = _spsolve47(Lm, rhs)
            r_a = gg["r_m"][-1] + 0.5 * gg["drm"][-1]
            r_i = gg["r_m"][0] - 0.5 * gg["dr"]
            C_ex = (np.pi * r_a**4
                    * _ring_compliance_factor(r_i / r_a) / (8.0 * gg["T_mem"]))
            wr = w.reshape(gg["Nr_m"], Np7)[:, 0]
            pr = _ring_static_shape((gg["r_m"] / r_a) ** 2, (r_i / r_a) ** 2)
            pr = pr / pr.max() * wr.max()
            return (abs(float(np.sum(w * rhs)) / C_ex - 1.0),
                    float(np.max(np.abs(wr - pr)) / wr.max()),
                    float(wr[0] / wr.max()))

        konv47 = {}
        for d47 in (0.0, 0.5e-3, 1.0e-3):
            cap47 = MicrophoneCapsule(bias_voltage=1e-6,
                                      center_post_diameter=d47, **g47)
            konv47[d47] = [_mem47(cap47, n) for n in (60, 120, 240)]
        for d47, rr47 in konv47.items():
            assert rr47[0][0] < 6e-3, \
                (f"Pfosten {d47 * 1e3:.1f} mm: das Membranfeld muss die "
                 f"geschlossene Ring-Nachgiebigkeit treffen "
                 f"({rr47[0][0]:.1e})")
            assert rr47[2][0] < 1e-3, \
                f"auf feinem Gitter erst recht ({rr47[2][0]:.1e})"
            # zweite Ordnung: jede Halbierung von dr muss den Fehler
            # mindestens dritteln (theoretisch vierteln)
            for k47 in (0, 1):
                assert rr47[k47][0] > 3.0 * rr47[k47 + 1][0], \
                    (f"Pfosten {d47 * 1e3:.1f} mm: der Fehler muss mit dem "
                     f"Gitter fallen ({[f'{x[0]:.1e}' for x in rr47]})")
            assert rr47[2][1] < 3e-3, \
                (f"auch die FORM muss stimmen "
                 f"({100 * rr47[2][1]:.3f} % bei feinem Gitter)")
        # d) Struktur des inneren Randes
        assert konv47[0.0][0][2] > 0.99, \
            "ohne Pfosten ist die innerste Zelle das Maximum"
        assert konv47[1.0e-3][2][2] < 0.05, \
            (f"mit Pfosten muss die innerste Zelle fast stillstehen "
             f"({konv47[1.0e-3][2][2]:.4f} des Maximums)")

        # e) Ende zu Ende gegen das 2D-Feld — die Mechanik der Ringmembran
        #    an der VOLUMENverschiebung (die Wandlung prüft Gegenprobe 54).
        #    Die Spannung selbst liegt im 3D bei 60 V (32 % Durchbiegung,
        #    weit über f_hom = 38 Hz dieses 12-Loch-Prüflings) rund 10 %
        #    tiefer: die Mitte bleibt hinter dem dort engsten Film zurück,
        #    und die Wandlung gewichtet die Mitte (1/g²) — ausgegeben.
        f47 = np.array([100.0, 500.0, 2000.0])

        def _hvol47(cc, ff):
            om_ = 2.0 * np.pi * ff
            pf_, pr_ = cc._source_pressures(om_, np.array([0.0]))
            Xf_, Xr_ = cc._solve_3d(om_, weight="volume")
            return cc._theta * (Xf_ * pf_[:, 0] + Xr_ * pr_[:, 0])

        e47, o47 = {}, {}
        for d47 in (0.0, 1.0e-3):
            c2 = MicrophoneCapsule(bias_voltage=60.0, squeeze_model="2d",
                                   center_post_diameter=d47,
                                   **{k: v for k, v in g47.items()
                                      if k != "squeeze_model"})
            c3 = MicrophoneCapsule(bias_voltage=60.0,
                                   center_post_diameter=d47, **g47)
            h2_47 = c2.transfer_function(f47)
            e47[d47] = np.abs(_hvol47(c3, f47) / h2_47)
            o47[d47] = np.abs(c3.transfer_function(f47) / h2_47)
        for d47, q47 in e47.items():
            assert np.all(np.abs(q47 - 1.0) < 0.05), \
                (f"3D und 2D müssen bei {d47 * 1e3:.1f} mm Pfosten "
                 f"zusammenpassen ({np.round(q47, 4)})")
        assert np.max(np.abs(e47[1.0e-3] - e47[0.0])) < 0.02, \
            (f"der Pfosten muss BEIDE Modelle gleich bewegen "
             f"({np.round(e47[1.0e-3] - e47[0.0], 4)})")
        # und die Spannung sinkt, wie es die Ringmembran verlangt
        c47r = MicrophoneCapsule(bias_voltage=60.0,
                                 center_post_diameter=1.0e-3, **g47)
        assert c47r._g3d["T_mem"] < 0.75 * ga["T_mem"], \
            (f"eine Ringmembran braucht für dieselbe Resonanz deutlich "
             f"weniger Zug ({c47r._g3d['T_mem']:.2f} gegen "
             f"{ga['T_mem']:.2f} N/m)")
        print(f"Ringmembran im 3D-Feld: r_i = 0 bitgleich; Membranoperator "
              f"trifft die geschlossene Ring-Nachgiebigkeit gitterkonvergent "
              f"("
              + "; ".join(
                  f"{1e3 * d:.1f} mm: "
                  + "->".join(f"{x[0]:.0e}" for x in v)
                  for d, v in konv47.items())
              + f"), Form auf {100 * max(v[2][1] for v in konv47.values()):.2f} %; "
              f"innerste Zelle mit Pfosten "
              f"{konv47[1.0e-3][2][2]:.3f} statt 1.0; 3D/2D (Volumen) mit "
              f"Pfosten {np.round(e47[1.0e-3], 3)} gegen "
              f"{np.round(e47[0.0], 3)} ohne, Spannung "
              f"{np.round(o47[1.0e-3], 3)} gegen {np.round(o47[0.0], 3)} "
              f"(60 V, Formanpassung); "
              f"Zug {ga['T_mem']:.1f} -> {c47r._g3d['T_mem']:.1f} N/m  OK")


@pytest.mark.feld3d
def test_gp50_3d_gitter_grob_fein():
    """Gegenprobe 50: 3D-Gitter grob/fein."""
    # grid_3d='coarse' (Standard) ist das bisherige Gitter; 'fine' löst
    # die kleinste Mündung mit mindestens 2 Zellen je Radius auf (radial
    # und azimutal am äußersten Lochmittenkreis) und ist in beiden
    # Richtungen mindestens 1.5-mal feiner. Der Gitterfehler kommt von der
    # Treppenkontur der Mündungen; welche Richtung ihn bestimmt, hängt von
    # der Bauform ab (K67: radial, 96er-Umfangsraster: azimutal).
    # a) Regel: grob = 60 × 96…320, fein erfüllt die Mündungsregel
    # b) KONVERGENZ gegen ein nochmals 1.5-mal feineres Referenzgitter
    #    (12 × Ø1.4 mm auf 1"): mit konturtreuen Mündungen (Gegenprobe
    #    51) liegt grob bei ~0.1 dB, fein bei ~0.03 dB. Ohne sie lag grob
    #    1 dB daneben, und die Treppenkontur konvergierte so langsam und
    #    unregelmäßig, dass selbst 180 × 324 noch 0.45 dB vom Grenzwert
    #    entfernt war.
    # c) MEMBRANRAND: die Einspannung liegt auf jedem Gitter exakt bei
    #    a_mem. Bis hier rastete sie auf das nächste Vielfache von dr ein
    #    (mit a_bp = a_mem eine ganze Zelle zu weit) — die Nachgiebigkeit
    #    sprang mit der Auflösung, der lochfreie Kolben-Grenzfall streute
    #    zwischen den Gittern um 0.3 dB statt zu konvergieren.
    # d) Clearance-Ring auf dem 3D-Gitter (fein: als Relief aufgelöst,
    #    wo das 2D-Feld einen Stub braucht; Stub-Zelle ab Pfostenrand)
    # e) Obergrenze der Zellen: winzige Löcher -> Warnung statt Absturz
    if _HAS_SCIPY:
        p50 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
            delay_length=0.0, cavity_length=8.0e-3,
            cavity_wall_thickness=1.5e-3, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=28e-3, n_through_holes=12,
            through_hole_diameter=1.4e-3)
        c50 = MicrophoneCapsule(squeeze_model="3d", **p50)
        f50 = MicrophoneCapsule(squeeze_model="3d", grid_3d="fine", **p50)
        g50c, g50f = c50._g3d, f50._g3d

        # a) Regel
        assert c50.grid_3d == "coarse" and (g50c["Nr"], g50c["Np"]) == (
            c50._fld_N, 96), "Standard muss das bisherige grobe Gitter sein"
        r_m50, R50 = f50._grid_3d_mouths()
        assert abs(r_m50 - 0.7e-3) < 1e-12, "kleinste Mündung = Loch"
        assert (g50f["Nr"] >= 1.5 * g50c["Nr"]
                and g50f["Np"] >= 1.5 * g50c["Np"]), \
            "fein muss in beiden Richtungen mindestens 1.5-mal feiner sein"
        assert g50f["cells_rm"] >= 2.0 > g50c["cells_rm"], \
            (f"fein: ≥ 2 Zellen je Mündungsradius ({g50f['cells_rm']:.2f}), "
             f"grob darunter ({g50c['cells_rm']:.2f})")

        # b) Konvergenz gegen ein 1.5-mal feineres Referenzgitter
        ref50 = MicrophoneCapsule(squeeze_model="3d", **p50)
        ref50._n_r_3d = int(np.ceil(1.5 * g50f["Nr"]))
        ref50._n_phi_3d = 2 * int(np.ceil(0.75 * g50f["Np"]))
        ref50._build_3d_geometry()
        fq50 = [1000.0]
        l_ref50 = 20.0 * np.log10(abs(ref50.transfer_function(fq50)[0]))
        e_c50 = 20.0 * np.log10(abs(c50.transfer_function(fq50)[0])) - l_ref50
        e_f50 = 20.0 * np.log10(abs(f50.transfer_function(fq50)[0])) - l_ref50
        assert abs(e_f50) < 0.06 and abs(e_c50) < 0.25 \
            and abs(e_f50) < 0.5 * abs(e_c50), \
            (f"fein muss am Referenzgitter liegen ({e_f50:+.3f} dB), grob "
             f"nahe dabei ({e_c50:+.3f} dB)")
        # ... und ohne konturtreue Mündungen läge grob 1 dB daneben
        MicrophoneCapsule._SHORTLEY_WELLER = False
        e_c50o = 20.0 * np.log10(abs(MicrophoneCapsule(
            squeeze_model="3d", **p50).transfer_function(fq50)[0])) - l_ref50
        MicrophoneCapsule._SHORTLEY_WELLER = True
        assert abs(e_c50o) > 0.5 and abs(e_c50o) > 4.0 * abs(e_c50), \
            (f"ohne Konturkorrektur muss grob sichtbar daneben liegen "
             f"({e_c50o:+.2f} dB)")

        # c) Membranrand exakt bei a_mem, Fläche exakt — grob, fein und
        #    a_bp = a_mem (kein Überstand)
        for cc50 in (c50, f50, MicrophoneCapsule(
                squeeze_model="3d", **dict(p50, backplate_diameter=25.4e-3))):
            gg50 = cc50._g3d
            edge50 = gg50["r_m"][-1] + 0.5 * gg50["drm"][-1]
            S50 = float(np.sum(gg50["A_m"])) * gg50["Np"]
            assert abs(edge50 / cc50.a_mem - 1.0) < 1e-12, \
                f"Einspannung muss bei a_mem liegen ({edge50 * 1e3:.4f} mm)"
            assert abs(S50 / (np.pi * cc50.a_mem**2) - 1.0) < 1e-12, \
                "Membranfläche des Gitters muss π·a_mem² sein"
        # ... und der lochfreie Kolben-Grenzfall (steife Membran, nur
        # Randspalt; mit und ohne Mittenpfosten) konvergiert jetzt glatt
        st50 = dict(p50, membrane_resonance_hz=50.0e3,
                    membrane_tension=25600.0, bias_voltage=45.0,
                    n_through_holes=0, ring_vent_width=50e-6)
        spread50 = []
        for d_post50 in (0.0, 8e-3):
            kw50 = dict(st50, center_post_diameter=d_post50)
            v2_50 = MicrophoneCapsule(squeeze_model="2d",
                                      **kw50).transfer_function([200.0])[0]
            lv50 = []
            for nr50 in (30, 60, 90, 135):
                cp50 = MicrophoneCapsule(squeeze_model="3d", **kw50)
                cp50._n_r_3d, cp50._n_phi_3d = nr50, 8
                cp50._build_3d_geometry()
                lv50.append(20.0 * np.log10(abs(
                    cp50.transfer_function([200.0])[0] / v2_50)))
            lv50 = np.array(lv50)
            assert np.all(np.diff(lv50) <= 1e-4) \
                    and abs(lv50[-1] - lv50[-2]) < 5e-3, \
                (f"Kolben-Grenzfall muss monoton konvergieren (3D−2D "
                 f"{np.round(lv50, 3)} dB)")
            assert abs(lv50[-1]) < 1.0, \
                f"3D und 2D müssen im Kolben-Grenzfall zusammenliegen"
            spread50.append(float(np.ptp(lv50)))

        # d) Clearance-Ring schmaler als eine 2D-Zelle, breiter als eine
        #    feine 3D-Zelle; mit Mittenpfosten
        clr50 = dict(p50, clearance_ring_diameter=16e-3,
                     clearance_ring_width=0.17e-3,
                     clearance_ring_depth=38e-6,
                     center_post_diameter=2e-3)
        cl2_50 = MicrophoneCapsule(squeeze_model="2d", **clr50)
        cl3_50 = MicrophoneCapsule(squeeze_model="3d", **clr50)
        cl3f_50 = MicrophoneCapsule(squeeze_model="3d", grid_3d="fine",
                                    **clr50)
        dr2_50 = (cl2_50.a_bp - cl2_50.r_post) / cl2_50._fld_N
        assert cl2_50._clr_stub_cell is not None \
            and cl3_50._g3d["stub_cell"] == cl2_50._clr_stub_cell, \
            "grob: Stub wie im 2D-Feld"
        i50 = cl2_50._clr_stub_cell
        assert (cl2_50.r_post + i50 * dr2_50 <= 8e-3
                < cl2_50.r_post + (i50 + 1) * dr2_50), \
            "Stub-Zelle muss den Ringradius enthalten (gezählt ab Pfosten)"
        assert cl3f_50._g3d["stub_cell"] is None \
            and cl3f_50._g3d["relief"].max() == 38e-6, \
            "fein: Ring als Relief aufgelöst"

        # e) Obergrenze: 24 × Ø0.1 mm verlangte ~250 000 Zellen je Feld
        with warnings.catch_warnings(record=True) as rec50:
            warnings.simplefilter("always")
            cap50 = MicrophoneCapsule(
                squeeze_model="3d", grid_3d="fine",
                **dict(p50, n_through_holes=24, through_hole_diameter=0.1e-3))
        g50x = cap50._g3d
        assert g50x["Nr"] * g50x["Np"] <= MicrophoneCapsule._GRID_FINE_MAX \
            and any("Obergrenze" in str(r.message) for r in rec50) \
            and g50x["fine_capped"] and not g50f["fine_capped"], \
            "Obergrenze muss greifen und warnen (und nur dann)"
        try:
            MicrophoneCapsule(grid_3d="medium")
            raise AssertionError("unbekannte Gitterstufe muss scheitern")
        except ValueError:
            pass
        print(f"3D-Gitter grob/fein: grob {g50c['Nr']}×{g50c['Np']} "
              f"({g50c['cells_rm']:.2f} Zellen je Mündungsradius) "
              f"{e_c50:+.3f} dB (ohne Konturkorrektur {e_c50o:+.2f} dB), "
              f"fein {g50f['Nr']}×{g50f['Np']} ({g50f['cells_rm']:.2f}) "
              f"{e_f50:+.3f} dB gegen {ref50._g3d['Nr']}×"
              f"{ref50._g3d['Np']}; Membranrand exakt, Kolben-Grenzfall "
              f"über Nr 30…135 innerhalb {max(spread50):.3f} dB; "
              f"Clearance/Obergrenze  OK")


@pytest.mark.feld3d
def test_gp51_konturtreue_mundungen_shortley_weller():
    """Gegenprobe 51: konturtreue Mündungen (Shortley–Weller)."""
    # Der 3D-Löser setzt jede Mündung aus den Zellen zusammen, deren Mitte
    # in ihr liegt; auf den Filmflächen am Mündungsrand zählt seit hier
    # nur das Stück des Mittenabstands, das außerhalb der Kreiskontur
    # liegt (Faktor Δ/ℓ). Belege:
    # a) EXAKTE Referenz: Äquipotentialscheibe exzentrisch in einer
    #    Kreisscheibe mit festem Randdruck — der Leitwert ist in bipolaren
    #    Koordinaten geschlossen, G = 2πK / arcosh((a² + r² − e²)/(2ar)).
    #    Gerechnet wird mit den Fußabdrücken UND Faktoren des Modells
    #    selbst (Film 0, stationär). Ohne Korrektur liegt das grobe Gitter
    #    bei −3…−6 %, konvergiert linear und unregelmäßig; mit Korrektur
    #    bei < 0.5 % und konvergiert quadratisch (Gitter halbiert ->
    #    Fehler geviertelt).
    # b) Faktoren: genau 1 abseits der Mündungsränder, sonst 1 < Δ/ℓ ≤
    #    _SW_CAP, nur auf Flächen zwischen Mündungs- und Filmzellen.
    # c) Akustisch (Gegenprobe 50 b): grobes Gitter 1 dB -> 0.1 dB neben
    #    dem 1.5-mal feineren Referenzgitter; der Kurzschluss G_s bleibt
    #    ein rein numerischer Parameter (Gegenprobe 48 b, mit Korrektur).
    # GRENZE: eine Mündung kleiner als eine Zelle (Mittelpunkt der
    # einzigen Zelle außerhalb der Kontur) bleibt unkorrigiert — dort hilft
    # nur das feine Gitter (≥ 2 Zellen je Radius).
    if _HAS_SCIPY:
        from scipy.sparse import coo_matrix as _coo51
        from scipy.sparse.linalg import spsolve as _sps51
        a51 = 11.95e-3

        def _ecc51(e, rm, nr, nphi, sw=True):
            """Leitwert (K = 1) Mündung -> Außenrand auf dem Modellgitter."""
            MicrophoneCapsule._SHORTLEY_WELLER = sw
            cc = MicrophoneCapsule(
                squeeze_model="3d", architecture="single",
                membrane_diameter=25.4e-3, backplate_diameter=2 * a51,
                n_blind_holes=0, bias_voltage=1.0,
                through_hole_rings=[(1, 2 * e)], through_hole_diameter=2 * rm)
            cc._n_r_3d, cc._n_phi_3d = nr, nphi
            cc._build_3d_geometry()
            MicrophoneCapsule._SHORTLEY_WELLER = True
            g = cc._g3d
            Nr_, Np5, q0_ = g["Nr"], g["Np"], g["q0"]
            f_r, f_a = g["sw"][0]
            NF_ = Nr_ * Np5
            idx = np.arange(NF_)
            k1, k2 = idx[:(Nr_ - 1) * Np5], idx[Np5:]
            Gv = np.repeat((q0_ + np.arange(1, Nr_)) * g["dphi"], Np5) \
                * f_r.ravel()
            k3 = (idx // Np5) * Np5 + (idx % Np5 + 1) % Np5
            Ga = np.repeat(g["dr"] / (g["r_f"] * g["dphi"]), Np5) * f_a.ravel()
            edge = (Nr_ - 1) * Np5 + np.arange(Np5)
            Ge = 2.0 * (q0_ + Nr_) * g["dphi"]
            L = _coo51((np.concatenate([-Gv, -Gv, Gv, Gv, -Ga, -Ga, Ga, Ga,
                                        np.full(Np5, Ge)]),
                        (np.concatenate([k1, k2, k1, k2, idx, k3, idx, k3,
                                         edge]),
                         np.concatenate([k2, k1, k1, k2, k3, idx, idx, k3,
                                         edge]))),
                       shape=(NF_, NF_)).tocsr()
            fp = np.zeros(NF_, bool)
            for cells in g["th_f"]:
                fp[cells] = True
            p = fp.astype(float)
            p[~fp] = _sps51(L[~fp][:, ~fp].tocsc(), -L[~fp][:, fp] @ p[fp])
            G_ex = 2 * np.pi / np.arccosh((a51**2 + rm**2 - e**2)
                                          / (2 * a51 * rm))
            return float(np.sum(Ge * p[edge])) / G_ex - 1.0, g

        # a) exakte Referenz, zwei Lagen/Größen, grob und doppelt so fein
        rows51 = []
        for e51, rm51 in ((6.0e-3, 0.7e-3), (3.0e-3, 1.0e-3)):
            e1, g51 = _ecc51(e51, rm51, 60, 96)
            e2, _ = _ecc51(e51, rm51, 120, 192)
            eo, _ = _ecc51(e51, rm51, 60, 96, sw=False)
            assert abs(e1) < 5e-3 and abs(e2) < 1e-3, \
                (f"konturtreue Mündung muss die exakte Lösung treffen "
                 f"(e = {e51 * 1e3:.0f} mm: {100 * e1:+.3f} / "
                 f"{100 * e2:+.3f} %)")
            assert abs(e1 / e2) > 3.0, \
                f"Konvergenz zweiter Ordnung erwartet ({e1 / e2:.1f}×)"
            assert abs(eo) > 2e-2 and abs(eo) > 10.0 * abs(e1), \
                f"ohne Korrektur: Treppenfehler ({100 * eo:+.2f} %)"
            rows51.append((e51, rm51, eo, e1, e2))

            # b) Faktoren (am groben Gitter dieses Falls)
            f_r51, f_a51 = g51["sw"][0]
            cap51 = MicrophoneCapsule._SW_CAP
            assert (f_r51.min() >= 1.0 and f_a51.min() >= 1.0
                    and f_r51.max() <= cap51 + 1e-12
                    and f_a51.max() <= cap51 + 1e-12), \
                "Konturfaktoren müssen in [1, _SW_CAP] liegen"
            Np51 = g51["Np"]
            fp51 = np.zeros(g51["NF"], bool)
            fp51[g51["th_f"][0]] = True
            fp51 = fp51.reshape(g51["Nr"], Np51)
            rim_r = fp51[:-1] ^ fp51[1:]            # radiale Randflächen
            rim_a = fp51 ^ np.roll(fp51, -1, axis=1)
            assert np.all(f_r51[~rim_r] == 1.0) \
                and np.all(f_a51[~rim_a] == 1.0) \
                and np.all(f_r51[rim_r] > 1.0) and np.all(f_a51[rim_a] > 1.0), \
                "Faktor ≠ 1 genau auf den Flächen Mündung | Film"
        print("Konturtreue Mündungen (Shortley–Weller): exzentrische Äqui"
              "potentialscheibe gegen bipolare Lösung " + "; ".join(
                  f"e {e * 1e3:.0f} mm/⌀{2 * r * 1e3:.1f} mm: ohne "
                  f"{100 * o:+.2f} %, mit {100 * a:+.3f} % -> "
                  f"{100 * b:+.3f} % (Gitter halbiert)"
                  for e, r, o, a, b in rows51)
              + "; Faktoren nur am Mündungsrand, in [1, "
              f"{MicrophoneCapsule._SW_CAP:.0f}]  OK")


@pytest.mark.feld3d
def test_gp52_hochtonuberschuss_des_3d_losers_aufgeklart():
    """Gegenprobe 52: Hochtonüberschuss des 3D-Lösers aufgeklärt."""
    # Oberhalb der Membranresonanz lag 3D 1…2 dB über 2D, auch bei dichten
    # Lochbildern (an der B&K 4134 3.7 dB über der Messung). Zerlegt:
    # a) EIN FEHLER des 3D-Lösers: der Membranring außerhalb der Platte
    #    (a_bp < r < a_mem) war hinten UNBELASTET. Er liegt über dem tiefen
    #    Ringraum, dessen Druck der Randdruck ist; jetzt spürt er ihn und
    #    speist seinen Volumenfluss dort ein (mit Randspalt: eigener
    #    Ringknoten zwischen Filmrand und Schlitzleitung) — wie die
    #    Randumgehung des 2D-Felds.
    #    Referenz: ein UNABHÄNGIGER axialsymmetrischer Löser (knoten-
    #    zentrierte Differenzen statt Zell-FV, eigene Assemblierung) für
    #    Membranfeld + Reynolds-Film + Ringknoten + Schlitz + Rückkette +
    #    Frontknoten, dazu die geschlossene Tieftonform
    #    V = f_in·C_m/(1 + C_m/C_b). Prüfling: B&K-4134-Geometrie nur mit
    #    Randschlitz (axialsymmetrisch).
    # b) KEIN FEHLER: die FORMANPASSUNG der Membran. Wo die Filmkraft
    #    gegen die Spannung zählt, weicht die Membran dem Filmdruck aus —
    #    beim Randschlitz (Druck in der Mitte am größten) bis zum
    #    Ringbuckel. 3D und unabhängiger Löser zeigen das gleich; das
    #    2D-Einmodenbild (feste Form) kann es nicht und liegt bei 20 kHz
    #    1.8 dB tiefer. Auch mehrere 2D-Moden helfen nicht (sie teilen
    #    sich einen Spaltknoten).
    # c) KEIN FEHLER: bei erzwungener Grundmodenform (sehr steife Membran)
    #    ist der Filmwiderstand von 3D und 2D für gleichverteilte Löcher
    #    IDENTISCH. Für Löcher auf EINEM Lochkreis überschätzt das 2D-Feld
    #    ihn (B&K 4134: 1.7-fach) — es löst die radiale Zuströmung zum
    #    Lochring selbst auf und addiert die volle Škvor-Zelle.
    # OFFEN: die B&K-Messung (Zuckerwar 1978, Aktuator) folgt dem 2D-
    #    Modell. Der 3D-Löser rechnet die Modellgleichungen nachweislich
    #    richtig — also dämpft die reale Kapsel stärker als der Reynolds-
    #    Film. Die zweite Möglichkeit, dass die Aktuatormessung im Hochton
    #    vom Druckfrequenzgang abweicht, ist in d) eingegrenzt: höchstens
    #    +0.6 dB, falsches Vorzeichen.
    if _HAS_SCIPY:
        from functools import reduce as _red52
        from scipy.sparse import lil_matrix as _lil52
        from scipy.sparse.linalg import spsolve as _sps52
        _NI52 = {"rho": 8900.0, "E": 200.0e9, "nu": 0.31}
        bk52 = dict(
            membrane_material=_NI52, membrane_resonance_hz=None,
            membrane_diameter=2 * 4.445e-3, membrane_thickness=5.0e-6,
            membrane_tension=3162.3, air_gap=2.077e-5,
            backplate_diameter=2 * 3.607e-3, backplate_thickness=0.843e-3,
            bias_voltage=1.0, architecture="single", n_through_holes=6,
            through_hole_diameter=2 * 5.080e-4,
            through_hole_pcd=2 * 2.032e-3, n_blind_holes=0,
            ring_vent_width=0.838e-3, ring_vent_length=3.048e-4,
            rear_network_enabled=True, delay_length=0.0,
            cavity_length=1.264e-7 / (np.pi * 3.607e-3**2),
            n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            include_diffraction=False)

        def _radial52(c, f, M=300):
            """Unabhängiger axialsymmetrischer Löser (nur Randschlitz):
            Volumenverschiebung über der Elektrode je Pa Quelldruck."""
            g = c._g3d
            T, sig = g["T_mem"], g["sigma"]
            a, b = c.a_mem, c.a_bp
            Mb = int(round(M * b / a))
            r = np.concatenate([np.linspace(0.0, b, Mb + 1),
                                np.linspace(b, a, M - Mb + 1)[1:]])
            n = r.size
            rm = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [a]])
            A = np.pi * (rm[1:] ** 2 - rm[:-1] ** 2)
            nf = Mb + 1
            Ab = A.copy()
            Ab[nf - 1] = np.pi * (b ** 2 - rm[nf - 1] ** 2)
            out = []
            for fq in f:
                om = 2 * np.pi * fq
                om_a = np.array([om])
                av = 0.5 * c.h_gap * np.sqrt(1j * om * RHO0 / MU_AIR)
                K = c.h_gap / (1j * om * RHO0) * (1.0 - np.tanh(av) / av)
                at = av * np.sqrt(PRANDTL)
                cg = c.h_gap / (GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(at)
                                         / at) * P_ATM)
                iw, ip = 0, n
                ir, ib, ifr = n + nf, n + nf + 1, n + nf + 2
                S = _lil52((n + nf + 3,) * 2, dtype=complex)
                rhs = np.zeros(n + nf + 3, dtype=complex)
                mq = -om ** 2 * sig * (1 - 1j / c._Q_MEMBRANE_INTERNAL)
                # Membran (w zur Platte hin): K·w − ω²M·w = A(p_f − p_r)
                for i in range(n - 1):
                    S[iw + i, iw + i] += mq * A[i]
                    for j in (i - 1, i + 1):
                        if 0 <= j < n:
                            G = T * np.pi * (r[i] + r[j]) / abs(r[j] - r[i])
                            S[iw + i, iw + i] += G
                            if j < n - 1:
                                S[iw + i, iw + j] -= G
                    S[iw + i, ifr] -= A[i]
                    S[iw + i, ip + i if i < nf else ir] += A[i]
                S[iw + n - 1, iw + n - 1] = 1.0
                # Film: Σ K·2πr/dr (p_i − p_j) + jωc·A·p = jω·A·w
                for i in range(nf - 1):
                    S[ip + i, ip + i] += 1j * om * cg * Ab[i]
                    S[ip + i, iw + i] -= 1j * om * Ab[i]
                    for j in (i - 1, i + 1):
                        if 0 <= j < nf:
                            G = K * np.pi * (r[i] + r[j]) / abs(r[j] - r[i])
                            S[ip + i, ip + i] += G
                            S[ip + i, ip + j] -= G
                S[ip + nf - 1, ip + nf - 1] = 1.0          # Rand = Ringraum
                S[ip + nf - 1, ir] = -1.0
                # Ringknoten: letztes Filmsegment, Randknoten, Membranring,
                # Schlitzleitung zum Rückknoten
                Tl = c._slit_line_abcd(om_a, c.ring_vent_w, 2 * np.pi * b,
                                       c.ring_vent_L)
                Al, Bl, Dl = (complex(Tl[0, 0][0]), complex(Tl[0, 1][0]),
                              complex(Tl[1, 1][0]))
                G = K * np.pi * (r[nf - 1] + r[nf - 2]) / (r[nf - 1]
                                                          - r[nf - 2])
                S[ir, ir] += G + 1j * om * cg * Ab[nf - 1] + Dl / Bl
                S[ir, ip + nf - 2] -= G
                for k in range(nf - 1, n - 1):
                    S[ir, iw + k] -= 1j * om * A[k]
                S[ir, ib] -= 1.0 / Bl
                Tb = _red52(c._mmul, c._rear_chain_mats(om_a))
                S[ib, ib] += Al / Bl + complex(Tb[1, 0][0]) / complex(Tb[0, 0][0])
                S[ib, ir] -= 1.0 / Bl
                Zf = (c._radiation_impedance_membrane(om_a)[0]
                      + c.rayl_front / c.S_mem)
                S[ifr, ifr] += 1.0 / Zf
                for k in range(n - 1):
                    S[ifr, iw + k] += 1j * om * A[k]
                rhs[ifr] = 1.0 / Zf
                x = _sps52(S.tocsr(), rhs)
                out.append(np.sum(Ab[:nf] * x[iw:iw + nf]))
            return np.array(out)

        def _cap3d52(p, np_=8):
            cc = MicrophoneCapsule(squeeze_model="3d", **p)
            cc._n_phi_3d = np_                   # axialsymmetrisch
            cc._build_3d_geometry()
            return cc

        # a) unabhängige Referenz (nur Randschlitz), mit und ohne Ring-
        #    kopplung; dazu die geschlossene Tieftonform
        slit52 = dict(bk52, n_through_holes=0)
        f52 = np.array([20.0, 1000.0, 5000.0, 10000.0, 20000.0])
        c52 = _cap3d52(slit52)
        X52 = c52._solve_3d(2 * np.pi * f52, weight="volume")[0]
        V52 = _radial52(c52, f52)
        dev52 = np.abs(20 * np.log10(np.abs(X52 / V52)))
        assert np.all(dev52 < 0.02), \
            (f"3D muss den unabhängigen radialen Löser treffen "
             f"({np.round(dev52, 3)} dB)")
        MicrophoneCapsule._ANNULUS_COUPLED = False
        X52o = _cap3d52(slit52)._solve_3d(2 * np.pi * f52,
                                          weight="volume")[0]
        MicrophoneCapsule._ANNULUS_COUPLED = True
        dev52o = np.abs(20 * np.log10(np.abs(X52o / V52)))
        assert dev52o[0] > 5.0 * max(dev52[0], 1e-4) and dev52o[0] > 0.03, \
            (f"ohne Ringkopplung muss der Tiefton sichtbar abweichen "
             f"({dev52o[0]:.3f} dB)")
        # geschlossene Tieftonform mit weitem Schlitz (keine Druckdifferenz
        # Ring/Rückraum): Membran gegen Rückvolumen + Film
        wide52 = dict(slit52, air_gap=300e-6, ring_vent_width=3e-3)
        cw52 = _cap3d52(wide52)
        om20 = np.array([2 * np.pi * 20.0])
        Tb52 = _red52(cw52._mmul, cw52._rear_chain_mats(om20))
        Cb52 = (complex(Tb52[1, 0][0]) / complex(Tb52[0, 0][0])).imag / om20[0]
        av20 = 0.5 * cw52.h_gap * np.sqrt(1j * om20[0] * RHO0 / MU_AIR)
        at20 = av20 * np.sqrt(PRANDTL)
        Cb52 += (cw52.h_gap / (GAMMA / (1 + (GAMMA - 1) * np.tanh(at20)
                                        / at20) * P_ATM)).real \
            * np.pi * cw52.a_bp ** 2
        Cm52 = np.pi * cw52.a_mem ** 4 / (8 * cw52._g3d["T_mem"])
        u52 = (cw52.a_bp / cw52.a_mem) ** 2
        V_ex52 = u52 * (2 - u52) * Cm52 / (1 + Cm52 / Cb52)
        e_ex52 = abs(abs(cw52._solve_3d(om20, weight="volume")[0][0])
                     / V_ex52 - 1)
        assert e_ex52 < 0.01, \
            f"Tiefton gegen die geschlossene Form ({100 * e_ex52:.2f} %)"

        # b) Formanpassung: 2D (feste Form) liegt bei 20 kHz deutlich
        #    tiefer als 3D und die unabhängige Referenz (a)
        h2_52 = MicrophoneCapsule(squeeze_model="2d",
                                  **slit52).transfer_function(f52)
        n52 = lambda x: 20 * np.log10(np.abs(x / x[0]))
        gap52 = float((n52(X52) - n52(h2_52))[-1])
        assert gap52 > 1.5, \
            (f"Einmodenbild muss oberhalb der Filmgrenze zurückbleiben "
             f"({gap52:+.2f} dB bei 20 kHz)")

        # c) Filmwiderstand bei erzwungener Form (sehr steife Membran):
        #    tan(Phase) ≈ −ω·R·C im Steifigkeitsbereich
        def _Rratio52(p):
            q = dict(p, membrane_resonance_hz=300e3)
            q.pop("membrane_tension", None)
            ff = np.array([1000.0])
            ph = {}
            for sm in ("2d", "3d"):
                cc = MicrophoneCapsule(squeeze_model=sm, **q)
                ph[sm] = np.angle(cc.transfer_function(ff)[0]
                                  / cc.transfer_function([20.0])[0])
            return float(np.tan(ph["3d"]) / np.tan(ph["2d"]))

        a24_52 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            air_gap=38.1e-6, backplate_diameter=23.9e-3,
            backplate_thickness=3.125e-3, bias_voltage=1.0,
            n_blind_holes=0, rear_network_enabled=True, delay_length=0.0,
            cavity_length=8.0e-3, cavity_wall_thickness=1.5e-3,
            n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=28e-3, n_through_holes=24,
            through_hole_diameter=0.99e-3)
        rr_uni52 = _Rratio52(a24_52)
        rr_bk52 = _Rratio52(bk52)
        assert abs(rr_uni52 - 1.0) < 0.01, \
            (f"gleichverteilte Löcher: Filmwiderstand 3D == 2D "
             f"({rr_uni52:.3f})")
        assert rr_bk52 < 0.7, \
            (f"Lochkreis (B&K 4134): 2D überschätzt den Filmwiderstand "
             f"(R_3D/R_2D = {rr_bk52:.2f})")
        # d) OFFENER PUNKT eingegrenzt: misst der Aktuator etwas anderes als
        #    den Druckfrequenzgang? Er treibt mit gleichmäßigem elektro-
        #    statischem Druck; die bewegte Membran erzeugt unter der Platte
        #    einen Zusatzdruck (Frederiksen 2013, Int. J. Metrol. Qual. Eng.
        #    4, Abschn. 11), im Modell eine Luftmasse jω·ρ·L/S vor der
        #    Membran statt der Abstrahlung (Plattengeometrie unbekannt:
        #    L = 10 mm ist eine großzügige obere Schranke). Die Druckantwort
        #    ist der Grenzfall L -> 0 (erste Ordnung, Gegenprobe 27). Die
        #    Last hebt die 4134 bei 13…20 kHz um höchstens 0.6 dB an —
        #    zu klein und mit falschem Vorzeichen für die 2.2…3.5 dB, um
        #    die 3D über der Messung liegt.
        f52d = np.array([13000.0, 16000.0, 20000.0])
        _rad52 = MicrophoneCapsule._radiation_impedance_membrane
        h52d = {}
        try:
            for L52 in (1e-8, 10e-3):
                MicrophoneCapsule._radiation_impedance_membrane = (
                    lambda s, o, _L=L52: 1j * np.asarray(o, dtype=float)
                    * RHO0 * _L / s.S_mem + 0j)
                h52d[L52] = MicrophoneCapsule(
                    squeeze_model="3d", **bk52).transfer_function(f52d)
        finally:
            MicrophoneCapsule._radiation_impedance_membrane = _rad52
        act52 = 20.0 * np.log10(np.abs(h52d[10e-3] / h52d[1e-8]))
        assert np.all(act52 > -0.05) and np.all(act52 < 0.7), \
            (f"Aktuatorlast (L = 10 mm) muss klein und positiv bleiben "
             f"({np.round(act52, 2)} dB bei 13/16/20 kHz)")
        print(f"3D-Hochtonüberschuss aufgeklärt: gegen unabhängigen "
              f"radialen Löser max {np.max(dev52):.3f} dB (Ring frei: "
              f"{dev52o[0]:.3f} dB im Tiefton), Tiefton gegen geschlossene "
              f"Form {100 * e_ex52:.2f} %; Formanpassung: 2D-Einmodenbild "
              f"{gap52:+.2f} dB unter 3D bei 20 kHz; Filmwiderstand bei "
              f"erzwungener Form 3D/2D = {rr_uni52:.3f} (gleichverteilt), "
              f"{rr_bk52:.2f} (B&K-Lochkreis); Aktuatorlast höchstens "
              f"{np.max(act52):+.2f} dB (erklärt die Messabweichung "
              f"nicht)  OK")


@pytest.mark.feld3d
def test_gp54_statischer_versatz_2d_3d_aufgeklart():
    """Gegenprobe 54: statischer Versatz 2D/3D aufgeklärt."""
    # An der B&K 4134 lagen 2D und 3D schon im Tiefton 1.38 dB auseinander,
    # frequenzunabhängig. Zwei Fehler im 3D-Pfad, keine Physik:
    # a) WANDLUNG: die Kette rechnet e = Θ·V mit der Volumenverschiebung V
    #    ihrer Grundmode über der GANZEN Membran; das Elektrodenintegral
    #    steckt in Θ. Der 3D-Löser setzte die Volumenverschiebung über der
    #    ELEKTRODE ein — bei a_bp < a_mem um u·(2 − u), u = (a_bp/a_mem)²,
    #    zu leise (B&K: −1.08 dB). Jetzt wandelt er sein Feld mit
    #    demselben Elektrodenintegral (s. _output_weight_3d); für die
    #    Grundmode ist das exakt die Kette.
    # b) SPANNUNG: bei vorgegebener Vorspannung rechnete der 3D-Löser die
    #    Spannung aus der Resonanz der Kette zurück, die mit dem Kolben-
    #    faktor 4/3 der statischen Form 1.9 % zu hoch liegt — 3.75 % zu viel
    #    Spannung, −0.3 dB. Jetzt die physikalische (s. _membrane_tension_3d).
    # c) Der Rest bei VORGEGEBENER RESONANZ war die Ein-Moden-Kalibrierung
    #    der Kette (Kolbenfaktor 4/3, statisch 3.75 % zu nachgiebig);
    #    behoben mit dem Massenfaktor 8/j01², s. Gegenprobe 55.
    # Die Zerlegung b) stellt deshalb den Stand VOR Gegenprobe 55 nach
    # (_MASS_EXACT = False): mit dem neuen Massenfaktor trifft schon die
    # Resonanz der Kette die physikalische Spannung, Schritt b) wäre leer.
    # d) Aufgedeckt durch a): Θ der KETTE rechnete auch bei Mitten-
    #    terminierung mit dw = 2V/S (Parabel). Die Ringmembran hat die
    #    mittlere Auslenkung m1 > 1/2 (3 mm Pfosten auf 25 mm: 0.632), die
    #    2D-Spannung lag um 2·m1 zu hoch (+2 dB). Jetzt dw = V/S_eff. Der
    #    alte 3D-Pfad trug denselben Fehler und verdeckte ihn.
    if _HAS_SCIPY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # a) Gewicht: die Grundmode ergibt exakt V = dw·S_eff wie Θ
            #    (Parabel: S/2; Ringmembran: S·m1)
            for arch54, kw54 in (("single", dict(backplate_diameter=15e-3)),
                                 ("dual", dict(bias_voltage=60.0)),
                                 ("single", dict(center_post_diameter=2e-3))):
                c54 = MicrophoneCapsule(squeeze_model="2d",
                                        architecture=arch54, **kw54)
                r54 = np.linspace(c54.r_post, c54.a_mem, 20001)
                u54 = (r54 / c54.a_mem) ** 2
                psi54 = np.maximum(_ring_static_shape(
                    np.minimum(u54, 1.0), c54.u_post), 0.0) / c54._phi_max
                V54 = np.trapezoid(c54._output_weight_3d(r54) * psi54
                                   * 2 * np.pi * r54, r54)
                assert abs(V54 / c54.S_eff_mem - 1.0) < 2e-3, \
                    (f"{arch54} {kw54}: Ausgangsgewicht muss für die Grundmode "
                     f"die Volumenverschiebung der ganzen Membran ergeben "
                     f"({V54 / c54.S_eff_mem:.5f})")
            # a2) Θ der Ringmembran: dw = V/S_eff, nicht 2V/S. Gegen das
            #     3D-Feld (das die Spannung direkt aus dem Feld bildet)
            #     ändert der Pfosten 3D/2D nicht — bei 1 V (kein Profil)
            #     und 20 Hz, weit unter f_hom (dichtes Raster). Mit dem
            #     alten 2/S läge 3D/2D um 1/(2·m1) ≈ 0.80 tiefer.
            p54 = dict(
                architecture="single", membrane_resonance_hz=2100.0,
                membrane_diameter=25.4e-3, membrane_thickness=6e-6,
                membrane_tension=45.0, air_gap=38.1e-6,
                backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
                bias_voltage=1.0, n_through_holes=96,
                through_hole_diameter=0.495e-3, n_blind_holes=0,
                rear_network_enabled=True, delay_length=0.0,
                cavity_length=8.0e-3, cavity_wall_thickness=1.5e-3,
                n_cavity_holes=0, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, body_diameter=28e-3)
            rp54, m1_54 = {}, {}
            for dp54 in (0.0, 3e-3):
                q54p = dict(p54, center_post_diameter=dp54)
                c2p = MicrophoneCapsule(squeeze_model="2d", **q54p)
                m1_54[dp54] = c2p._phi_m1
                rp54[dp54] = float(abs(
                    MicrophoneCapsule(squeeze_model="3d", **q54p)
                    .transfer_function([20.0])[0]
                    / c2p.transfer_function([20.0])[0]))
            assert abs(rp54[3e-3] - rp54[0.0]) < 0.01, \
                (f"a2) Pfosten: 3D/2D unverändert ({rp54[0.0]:.4f} -> "
                 f"{rp54[3e-3]:.4f}; mit 2/S wären es "
                 f"{rp54[0.0] / (2 * m1_54[3e-3]):.3f})")
            # b) B&K 4134 (Vorspannung vorgegeben), 20 Hz
            bk54 = dict(
                membrane_material={"rho": 8900.0, "E": 200.0e9, "nu": 0.31},
                membrane_resonance_hz=None, membrane_diameter=2 * 4.445e-3,
                membrane_thickness=5.0e-6, membrane_tension=3162.3,
                air_gap=2.077e-5, backplate_diameter=2 * 3.607e-3,
                backplate_thickness=0.843e-3, bias_voltage=1.0,
                architecture="single", n_through_holes=6,
                through_hole_diameter=2 * 5.080e-4,
                through_hole_pcd=2 * 2.032e-3, n_blind_holes=0,
                ring_vent_width=0.838e-3, ring_vent_length=3.048e-4,
                rear_network_enabled=True, delay_length=0.0,
                cavity_length=1.264e-7 / (np.pi * 3.607e-3 ** 2),
                n_cavity_holes=0, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, include_diffraction=False)
            f54 = np.array([20.0])
            h2_54 = MicrophoneCapsule(squeeze_model="2d",
                                      **bk54).transfer_function(f54)[0]
            d54 = {}
            for oe54, te54, me54 in ((False, False, False),
                                     (True, False, False),
                                     (True, True, False), (True, True, True)):
                MicrophoneCapsule._OUTPUT_EXACT_3D = oe54
                MicrophoneCapsule._TENSION_EXACT_3D = te54
                MicrophoneCapsule._MASS_EXACT = me54
                try:
                    h3_54 = MicrophoneCapsule(
                        squeeze_model="3d", **bk54).transfer_function(f54)[0]
                finally:
                    MicrophoneCapsule._OUTPUT_EXACT_3D = True
                    MicrophoneCapsule._TENSION_EXACT_3D = True
                    MicrophoneCapsule._MASS_EXACT = True
                d54[(oe54, te54, me54)] = float(
                    20 * np.log10(abs(h2_54 / h3_54)))
            cbk54 = MicrophoneCapsule(squeeze_model="2d", **bk54)
            ubk54 = (cbk54.a_bp / cbk54.a_mem) ** 2
            fin54 = -20 * np.log10(ubk54 * (2 - ubk54))
            old54, out54, both54, now54 = (
                d54[(False, False, False)], d54[(True, False, False)],
                d54[(True, True, False)], d54[(True, True, True)])
            assert old54 > 1.2, f"alter Versatz sichtbar ({old54:+.2f} dB)"
            assert abs((old54 - out54) - fin54) < 0.05, \
                (f"a) die Wandlung erklärt u·(2−u) = {fin54:.2f} dB "
                 f"({old54 - out54:+.2f} dB)")
            assert abs(both54) < 0.02, \
                (f"b) mit physikalischer Spannung verschwindet der Rest "
                 f"({out54:+.2f} -> {both54:+.3f} dB)")
            assert abs(now54) < 0.02, \
                f"heutiger Stand: 2D == 3D ({now54:+.3f} dB)"
        print(f"Statischer Versatz 2D/3D (B&K 4134, 20 Hz, Stand vor "
              f"Gegenprobe 55): {old54:+.2f} dB = Wandlung "
              f"{old54 - out54:+.2f} (u·(2−u): {fin54:+.2f}) + Spannung "
              f"{out54 - both54:+.2f} -> {both54:+.3f} dB, heute "
              f"{now54:+.3f} dB; Ringmembran (m1 = {m1_54[3e-3]:.3f}): Θ "
              f"mit S_eff, 3D/2D {rp54[0.0]:.4f} -> {rp54[3e-3]:.4f} mit "
              f"Pfosten  OK")


@pytest.mark.feld3d
def test_gp56_wiederverwendung_der_3d_loesung():
    """Gegenprobe 56: Wiederverwendung der 3D-Feldlösung je Frequenz.

    directivity, transfer_function, delay_diagnostics und die Diagnosen
    lösen bei gleicher Frequenz dasselbe System; seit dieser Gegenprobe
    wird es nur einmal faktorisiert (Lösungsspeicher an der Geometrie,
    s. _solve_3d_cache). Geprüft wird:
    a) BITGLEICH gegen den Löser ohne Speicher, über dieselbe Folge von
       Aufrufen (Pattern, Übertragung, Rückmembran-Antworten mit dem
       Gewicht 'volume', Laufzeit-Diagnose);
    b) WIRKUNG: jede Frequenz genau einmal faktorisiert, eine neue
       Frequenz in einem gemischten Aufruf genau einmal mehr;
    c) VERFALL bei geändertem Zustand derselben Kapsel: Klassenmethode
       (Strahlungsimpedanz) oder Instanzmethode getauscht -> neu gelöst,
       bitgleich mit einer frisch gebauten Kapsel; zurückgetauscht ->
       wieder bitgleich mit dem Original;
    d) VERFALL bei neuem Gitter (_build_3d_geometry) -> neu gelöst,
       bitgleich mit einer frisch gebauten Kapsel dieses Gitters;
    e) Einzel-Backplate: die Reziprozitäts-Diagnose _recip_3d kommt auch
       bei einem Treffer aus dem Speicher.
    """
    if not _HAS_SCIPY:
        return
    kw56 = dict(DEB_KWARGS, squeeze_model="3d", **DEB_CLEARANCE)
    f56 = np.array([500.0, 1000.0])
    om56 = 2.0 * np.pi * f56

    def _lauf(cap):
        return (cap.directivity(
                    frequencies_hz=(1000.0,))["patterns"][1000.0]["db"],
                cap.transfer_function(f56),
                cap._solve_3d(om56, want_rear=True, weight="volume"),
                cap.delay_diagnostics()["ratio"])

    def _gleich(a, b):
        return (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                and all(np.array_equal(x, y) for x, y in zip(a[2], b[2]))
                and a[3] == b[3])

    # a) bitgleich gegen den Löser ohne Speicher
    MicrophoneCapsule._REUSE_3D = False
    try:
        ref56 = MicrophoneCapsule(**kw56)
        r_ref56 = _lauf(ref56)
    finally:
        MicrophoneCapsule._REUSE_3D = True
    cap56 = MicrophoneCapsule(**kw56)
    r56 = _lauf(cap56)
    assert _gleich(r_ref56, r56), \
        "a) mit Lösungsspeicher muss alles bitgleich sein"
    # b) Wirkung
    n_ohne56, n_mit56 = ref56._lu_3d_count, cap56._lu_3d_count
    assert n_ohne56 == 6 and n_mit56 == 2, \
        (f"b) zwei Frequenzen, je einmal faktorisiert ({n_ohne56} ohne, "
         f"{n_mit56} mit Speicher)")
    cap56.transfer_function([1000.0, 2000.0])
    assert cap56._lu_3d_count == 3, \
        (f"b) bekannte + neue Frequenz: genau eine Faktorisierung mehr "
         f"({cap56._lu_3d_count})")

    # c) Zustand derselben Kapsel geändert: Klassen- und Instanzmethode
    _rad56 = MicrophoneCapsule._radiation_impedance_membrane
    h1_56 = cap56.transfer_function([1000.0])

    def _rad2(self, omega):
        return 50.0 * _rad56(self, omega)

    try:
        MicrophoneCapsule._radiation_impedance_membrane = _rad2
        n0 = cap56._lu_3d_count
        h_tausch56 = cap56.transfer_function([1000.0])
        assert cap56._lu_3d_count == n0 + 1, \
            "c) getauschte Klassenmethode: neu lösen"
        h_frisch56 = MicrophoneCapsule(**kw56).transfer_function([1000.0])
    finally:
        MicrophoneCapsule._radiation_impedance_membrane = _rad56
    d_tausch56 = float(abs(20 * np.log10(abs(h_tausch56[0] / h1_56[0]))))
    assert np.array_equal(h_tausch56, h_frisch56) and d_tausch56 > 1e-3, \
        (f"c) getauscht: wie eine frisch gebaute Kapsel und sichtbar anders "
         f"({d_tausch56:.4f} dB)")
    assert np.array_equal(cap56.transfer_function([1000.0]), h1_56), \
        "c) zurückgetauscht: wieder das Original"
    cap56._radiation_impedance_membrane = types.MethodType(_rad2, cap56)
    n0 = cap56._lu_3d_count
    h_inst56 = cap56.transfer_function([1000.0])
    del cap56._radiation_impedance_membrane
    assert cap56._lu_3d_count == n0 + 1 and \
        np.array_equal(h_inst56, h_tausch56), \
        "c) auf der Instanz getauschte Methode: neu lösen, gleiches Ergebnis"

    # d) neues Gitter
    cap56._n_phi_3d = cap56._g3d["Np"] // 2
    cap56._build_3d_geometry()
    n0 = cap56._lu_3d_count
    h_gitter56 = cap56.transfer_function([1000.0])
    frisch56 = MicrophoneCapsule(**kw56)
    frisch56._n_phi_3d = cap56._n_phi_3d
    frisch56._build_3d_geometry()
    assert cap56._lu_3d_count == n0 + 1 and \
        np.array_equal(h_gitter56, frisch56.transfer_function([1000.0])) \
        and not np.array_equal(h_gitter56, h1_56), \
        "d) neues Gitter: neu lösen, wie eine frisch gebaute Kapsel"

    # e) Einzel-Backplate: _recip_3d aus dem Speicher
    cs56 = MicrophoneCapsule(
        architecture="single", membrane_resonance_hz=8000.0,
        membrane_diameter=12.0e-3, membrane_thickness=5e-6,
        membrane_tension=400.0, air_gap=25e-6,
        backplate_diameter=11.0e-3, backplate_thickness=1.5e-3,
        bias_voltage=1.0, n_blind_holes=0, n_through_holes=48,
        through_hole_diameter=0.33e-3, cavity_length=4.0e-3,
        n_cavity_holes=0, squeeze_model="3d")
    om1_56 = np.array([2.0 * np.pi * 1000.0])
    x1_56 = cs56._solve_3d(om1_56)
    rec56 = cs56._recip_3d
    cs56._recip_3d = None
    x2_56 = cs56._solve_3d(om1_56, weight="volume")
    x3_56 = cs56._solve_3d(om1_56)
    assert cs56._lu_3d_count == 1 and cs56._recip_3d == rec56 and \
        all(np.array_equal(a, b) for a, b in zip(x1_56, x3_56)) and \
        not np.array_equal(x1_56[0], x2_56[0]), \
        ("e) Treffer: eine Faktorisierung, _recip_3d wie gelöst, "
         "Gewichte getrennt")
    print(f"Wiederverwendung 3D-Lösung: bitgleich (Pattern, H, Rückmembran, "
          f"Laufzeit); Faktorisierungen {n_ohne56} -> {n_mit56}, neue "
          f"Frequenz +1; Strahlungsimpedanz getauscht -> neu gelöst "
          f"({d_tausch56:.2f} dB anders, wie frisch gebaut), zurück "
          f"bitgleich; Instanzmethode und neues Gitter -> neu gelöst; "
          f"Einzel-Backplate: Reziprozitäts-Diagnose aus dem Speicher  OK")


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp57_symmetrische_zerlegung():
    """Gegenprobe 57: LU-Zerlegung des 3D-Systems mit symmetrischer
    Ordnung und Diagonal-Pivots (s. _lu_solve_3d).

    Die pivotisierende Zerlegung (COLAMD, Zeilentausch nach Betrag)
    zerstörte mit ihren Tauschen die füllungsarme Ordnung des strukturell
    symmetrischen Systems: an der Doppel-Backplate 101 statt 8,6 Mio.
    Einträge, 137 statt 1,4 s je Frequenz. Geprüft wird:
    a) RÜCKWÄRTSSTABIL an allen Bauformen — Einzel- und Doppel-
       Backplate, einteilige Doppelmembran (Debenham), K67 mit
       Zwischenspalt, B&K 4134 mit Randspalt — bei 20 Hz, 1 kHz und
       20 kHz: komponentenweiser Rückwärtsfehler höchstens 1e-12 (über
       den ganzen Selbsttest gemessen ≤ 7e-14), ohne Rückfall;
    b) DASSELBE ERGEBNIS wie die pivotisierende Zerlegung (Einzel-
       Backplate, Debenham, B&K, Doppel-Backplate auf 40 × 160 Zellen),
       deren Rückwärtsfehler dabei nicht kleiner ist;
    c) AUFFÜLLUNG der Doppel-Backplate höchstens ein Drittel;
    d) RÜCKFALL: erzwungen (_LU_BERR_MAX = 0) bitgleich mit der
       pivotisierenden Zerlegung; an einer Matrix, deren Diagonale
       überall winzig ist ([[ε, 1], [1, ε]] — keine symmetrische
       Umordnung hilft), erkennt die Prüfung die instabile Zerlegung und
       rechnet pivotisierend.
    """
    if not _HAS_SCIPY:
        return
    import scipy.sparse as _sp57
    sg57 = dict(membrane_material="PET", membrane_resonance_hz=8000.0,
                membrane_diameter=22e-3, membrane_thickness=6e-6,
                membrane_tension=400.0, air_gap=40e-6,
                backplate_diameter=20e-3, backplate_thickness=3e-3,
                bias_voltage=60.0, n_through_holes=60,
                through_hole_diameter=1.0e-3, n_blind_holes=30,
                blind_hole_diameter=1.2e-3, delay_length=3e-3,
                cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
                n_cavity_holes=200, cavity_hole_diameter=0.2e-3,
                cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, body_diameter=24e-3,
                include_diffraction=False)
    bk57 = dict(
        membrane_material={"rho": 8900.0, "E": 200.0e9, "nu": 0.31},
        membrane_resonance_hz=None, membrane_diameter=2 * 4.445e-3,
        membrane_thickness=5.0e-6, membrane_tension=3162.3,
        air_gap=2.077e-5, backplate_diameter=2 * 3.607e-3,
        backplate_thickness=0.843e-3, bias_voltage=28.0,
        architecture="single", n_through_holes=6,
        through_hole_diameter=2 * 5.080e-4,
        through_hole_pcd=2 * 2.032e-3, n_blind_holes=0,
        ring_vent_width=0.838e-3, ring_vent_length=3.048e-4,
        rear_network_enabled=True, delay_length=0.0,
        cavity_length=1.264e-7 / (np.pi * 3.607e-3 ** 2),
        n_cavity_holes=0, fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
        include_diffraction=False)
    faelle57 = {
        "single": dict(sg57, architecture="single"),
        "dual": dict(sg57, architecture="dual"),
        "Debenham": dict(DEB_KWARGS, **DEB_CLEARANCE),
        "K67": dict(K67_KWARGS),
        "B&K 4134": bk57}

    def _bau57(kw, gitter=None):
        cap = MicrophoneCapsule(**{**kw, "squeeze_model": "3d"})
        if gitter is not None:
            cap._n_r_3d, cap._n_phi_3d = gitter
            cap._build_3d_geometry()
        return cap

    # a) rückwärtsstabil ohne Rückfall
    berr57 = {}
    for name57, kw57 in faelle57.items():
        cap57 = _bau57(kw57)
        b57 = []
        for f57 in (20.0, 1000.0, 20000.0):
            cap57._solve_3d(np.array([2.0 * np.pi * f57]))
            art57, be57, _ = cap57._lu_3d_info
            assert art57 == "symmetrisch", \
                f"a) {name57} {f57:.0f} Hz: ohne Rückfall ({art57})"
            b57.append(be57)
        berr57[name57] = max(b57)
        assert berr57[name57] <= 1e-12, \
            (f"a) {name57}: Rückwärtsfehler {berr57[name57]:.1e} (höchstens "
             f"1e-12)")

    # b) + c) gegen die pivotisierende Zerlegung
    om57 = np.array([2.0 * np.pi * 1000.0])
    dev57, fill57 = {}, None
    for name57, kw57, gitter57 in (
            ("single", faelle57["single"], None),
            ("Debenham", faelle57["Debenham"], None),
            ("B&K 4134", bk57, None),
            ("dual 40×160", faelle57["dual"], (40, 160))):
        erg57 = {}
        for sym57 in (True, False):
            MicrophoneCapsule._LU_SYMMETRIC = sym57
            try:
                cap57 = _bau57(kw57, gitter57)
                erg57[sym57] = (np.array(cap57._solve_3d(om57,
                                                         want_rear=True)),
                                cap57._lu_3d_info)
            finally:
                MicrophoneCapsule._LU_SYMMETRIC = True
        (xs57, info_s57), (xp57, info_p57) = erg57[True], erg57[False]
        nz57 = np.abs(xp57) > 0.0         # geschlossene Rückseite: X_r = 0
        assert np.array_equal(xs57[~nz57], xp57[~nz57]), \
            f"b) {name57}: exakte Nullen bleiben Nullen"
        dev57[name57] = float(np.max(np.abs(xs57[nz57] / xp57[nz57] - 1.0)))
        assert dev57[name57] < 1e-6, \
            (f"b) {name57}: symmetrisch wie pivotisiert "
             f"({dev57[name57]:.1e})")
        assert info_p57[1] >= info_s57[1], \
            (f"b) {name57}: die pivotisierende Zerlegung ist nicht genauer "
             f"({info_p57[1]:.1e} gegen {info_s57[1]:.1e})")
        if gitter57 is not None:
            fill57 = info_s57[2] / info_p57[2]
    assert fill57 < 1.0 / 3.0, \
        f"c) Auffüllung der Doppel-Backplate höchstens 1/3 ({fill57:.2f})"

    # d) Rückfall
    MicrophoneCapsule._LU_BERR_MAX = 0.0
    try:
        capr57 = _bau57(faelle57["Debenham"])
        xr57 = np.array(capr57._solve_3d(om57, want_rear=True))
    finally:
        MicrophoneCapsule._LU_BERR_MAX = 1e-11
    MicrophoneCapsule._LU_SYMMETRIC = False
    try:
        xq57 = np.array(_bau57(faelle57["Debenham"])._solve_3d(
            om57, want_rear=True))
    finally:
        MicrophoneCapsule._LU_SYMMETRIC = True
    assert capr57._lu_3d_fallback == 1 and \
        capr57._lu_3d_info[0] == "pivotisiert" and \
        np.array_equal(xr57, xq57), \
        "d) erzwungener Rückfall: pivotisierend, bitgleich"
    S57 = _sp57.csc_matrix(np.array([[1e-20, 1.0], [1.0, 1e-20]],
                                    dtype=complex))
    rhs57 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    capm57 = MicrophoneCapsule()
    xm57 = capm57._lu_solve_3d(S57, rhs57)
    assert capm57._lu_3d_fallback == 1 and \
        np.allclose(S57 @ xm57, rhs57, rtol=1e-12, atol=0.0), \
        "d) winzige Diagonale: Prüfung muss zurückfallen, Lösung korrekt"
    print(f"Symmetrische LU-Zerlegung: Rückwärtsfehler "
          + ", ".join(f"{k} {v:.0e}" for k, v in berr57.items())
          + f" (ohne Rückfall); wie pivotisiert bis "
          f"{max(dev57.values()):.0e}; Auffüllung dual {fill57:.2f}; "
          f"Rückfall erzwungen bitgleich, winzige Diagonale erkannt  OK")
