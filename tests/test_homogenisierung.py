"""Gegenproben: Homogenisierungsgrenze und Warnungen."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


# --------- Gegenprobe 48: Homogenisierungsgrenze + 3D-Referenz ------------
# Frage: ab wann darf man einem 1D/2D-Ergebnis mit spärlichem Lochbild
# nicht mehr trauen? Die Antwort kann nur der 3D-Löser geben — und der
# musste dafür erst selbst belastbar werden. Beim Aufstellen der Grenze
# fielen VIER Fehler im 3D-Löser auf (a–d), jeder physikalisch begründet
# behoben und hier festgehalten. Dann die GRENZE selbst (s.
# homogenization_limit): der Film über dem größten lochfreien Bereich
# (Radius ρ) staut sich, die gespannte Membran beult sich dort aus — das
# kann nur der 3D-Löser. Kennzahl im Tiefton Π = ω·12μρ⁴/(h³·T·j01²),
# 1-dB-Einsatz oberhalb Π = 10 (allgemein mit Filmträgheit und
# Beulresonanz, dazu die Grenze der Lochkreis-Darstellung: Gegenprobe 53;
# s. _PI_HOM). GRENZE DES 3D-LÖSERS SELBST: das grobe Standardgitter löst
# kleine Mündungen nur mit rund einer Zelle je Radius auf; e) rechnet
# deshalb mit grid_3d='fine' (Gegenprobe 50/51).
_PA48 = dict(
    architecture="single", membrane_resonance_hz=2100.0,
    membrane_diameter=25.4e-3, membrane_thickness=6e-6,
    membrane_tension=45.0, air_gap=38.1e-6,
    backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
    bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
    delay_length=0.0, cavity_length=8.0e-3,
    cavity_wall_thickness=1.5e-3, n_cavity_holes=0,
    fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
    body_diameter=28e-3)
_PB48 = dict(
    architecture="single", membrane_resonance_hz=8000.0,
    membrane_diameter=12.0e-3, membrane_thickness=5e-6,
    membrane_tension=400.0, air_gap=25e-6,
    backplate_diameter=11.0e-3, backplate_thickness=1.5e-3,
    bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
    delay_length=0.0, cavity_length=4.0e-3,
    cavity_wall_thickness=1.0e-3, n_cavity_holes=0,
    fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
    body_diameter=14e-3)
_A12_48 = dict(_PA48, n_through_holes=12, through_hole_diameter=1.4e-3)
_F48 = np.array([300.0, 4000.0])


def _cap48(sm, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cc = MicrophoneCapsule(squeeze_model=sm, **kw)
    return cc, [str(r.message) for r in rec
                if issubclass(r.category, UserWarning)]


def _grid48(cc, nr, nphi):
    cc._n_r_3d = nr
    cc._n_phi_3d = nphi
    cc._build_3d_geometry()
    return cc


def _db48(x):
    return 20.0 * np.log10(np.abs(x))


def _fein48(kw48, ff48):
    """2D gegen das feine 3D-Gitter [dB] und f_hom des 2D-Modells."""
    c2_48, _ = _cap48("2d", **kw48)
    c3_48 = MicrophoneCapsule(squeeze_model="3d", grid_3d="fine", **kw48)
    ff48 = np.array(ff48)
    return (c2_48.homogenization_limit()["f_hom"],
            _db48(c2_48.transfer_function(ff48)
                  / c3_48.transfer_function(ff48)))


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp48a_fussabdruck_konvergenz():
    """Gegenprobe 48 a: FUSSABDRUCK. Das Suchfenster war fest ±4 Zellen;
    größere Mündungen wurden abgeschnitten (12 × Ø1.4 mm auf 3 mm Radius:
    wirksam Ø0.8 mm), das Ergebnis WANDERTE mit der Gitterfeinheit. Jetzt
    exakter Abstand, Fenster nach Lochgröße -> konvergent. Zellfläche je
    Mündung ≈ πr², auch auf dem inneren Hilfskreis (dort sind die Zellen
    azimutal schmal — genau da schnitt das alte Fenster ab)."""
    if not _HAS_SCIPY:
        return
    c48a, _ = _cap48("3d", **_A12_48)
    g48a = c48a._g3d
    Acell48 = np.repeat(g48a["A_f"], g48a["Np"])
    fill48 = np.array([Acell48[c].sum() for c in g48a["th_f"]]) \
        / (np.pi * c48a.r_th ** 2)
    assert np.all(np.abs(fill48 - 1.0) < 0.25), \
        (f"jeder Fußabdruck muss die ganze Mündung decken "
         f"(Flächenverhältnis {np.round(fill48, 2)})")
    # ... und das Ergebnis konvergiert mit dem Radialgitter
    lv48 = [_db48(_grid48(MicrophoneCapsule(squeeze_model="3d", **_A12_48),
                          nr, 192).transfer_function(_F48))
            for nr in (30, 60, 120)]
    d1_48 = np.abs(lv48[1] - lv48[0])
    d2_48 = np.abs(lv48[2] - lv48[1])
    assert np.all(d2_48 < 0.5 * d1_48) and np.all(d2_48 < 0.4), \
        (f"3D muss mit dem Gitter konvergieren (Schritte "
         f"{np.round(d1_48, 2)} -> {np.round(d2_48, 2)} dB)")
    print(f"Homogenisierungsgrenze a) 3D-Fußabdruck deckt die Mündung "
          f"({np.min(fill48):.2f}…{np.max(fill48):.2f}), konvergent "
          f"({np.max(d1_48):.2f} -> {np.max(d2_48):.2f} dB)  OK")


@pytest.mark.feld3d
def test_gp48b_aequipotential():
    """Gegenprobe 48 b: ÄQUIPOTENTIAL. Über dem Lochquerschnitt gibt es
    keinen Film; die Mündungszellen sind kurzgeschlossen. Der
    Kurzschlussleitwert ist ein numerischer Parameter, kein
    physikalischer — das Ergebnis darf nicht davon abhängen."""
    if not _HAS_SCIPY:
        return
    lvs48 = []
    for fac48 in (1.0e3, 1.0e5):
        MicrophoneCapsule._EQUI_SHORT = fac48
        lvs48.append(_db48(MicrophoneCapsule(
            squeeze_model="3d", **_A12_48).transfer_function(_F48)))
    MicrophoneCapsule._EQUI_SHORT = 1.0e4
    assert np.max(np.abs(lvs48[1] - lvs48[0])) < 1e-3, \
        (f"Mündungs-Kurzschluss muss konvergiert sein "
         f"({np.max(np.abs(lvs48[1] - lvs48[0])):.1e} dB)")
    print(f"Homogenisierungsgrenze b) Kurzschluss "
          f"{np.max(np.abs(lvs48[1] - lvs48[0])):.0e} dB  OK")


def test_gp48c_lage_k67():
    """Gegenprobe 48 c: LAGE an der K67. Gleichverteilte Durchgangs- und
    Sacklöcher lagen als getrennte Raster auf denselben Hilfskreisen,
    15° versetzt — an der K67 überlappten die Senkungen. Jetzt EIN
    isotropes Raster mit abwechselnder Belegung, die Gegenelektrode
    kreisweise um die halbe Durchgangsteilung verdreht: keine Mündung
    überlappt eine andere, die Kerne beider Hälften liegen im
    Zwischenspalt auseinander. Pauschal 3° (= 180°/60, alte
    Voreinstellung) legt sie auf dem Mehrkreis-Raster übereinander."""
    if not _HAS_SCIPY:
        return
    k48 = dict(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm",
        center_gap=50e-6, n_through_holes=60,
        through_hole_diameter=0.6e-3, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0)
    ck48, _ = _cap48("2d", **k48)
    hp48 = ck48._hole_positions()

    def _xy48(plist, turn=0.0, auto=False):
        # auto: kreisweiser Rückversatz (automatische Verdrehung)
        r_ = np.array([p[0] for p in plist])
        d_ = np.array([p[1] + (p[2] if auto else turn) for p in plist])
        return r_ * np.cos(np.deg2rad(d_)), r_ * np.sin(np.deg2rad(d_))

    def _mind48(ax, ay, bx, by, same=False):
        dd = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
        if same:
            dd[np.diag_indices_from(dd)] = np.inf
        return float(np.min(dd))

    tx, ty = _xy48(hp48["th"])
    bx, by = _xy48(hp48["bh"])
    ax_ = np.concatenate([tx, bx])
    ay_ = np.concatenate([ty, by])
    gap_mouth48 = _mind48(ax_, ay_, ax_, ay_, same=True) \
        / (2.0 * ck48.r_bh)
    assert len(hp48["th"]) == 60 and len(hp48["bh"]) == 60, \
        "K67: 60 durchgebohrte + 60 blinde Senkungen"
    assert gap_mouth48 > 1.0, \
        (f"keine Senkung darf eine andere überlappen (kleinster "
         f"Mittenabstand {gap_mouth48:.2f} × Senkungs-Ø)")
    rx, ry = _xy48(hp48["th"], auto=True)      # automatische Drehung
    core_auto48 = _mind48(tx, ty, rx, ry) / (2.0 * ck48.r_th)
    rx3, ry3 = _xy48(hp48["th"], 3.0)          # alte Voreinstellung
    core_3deg48 = _mind48(tx, ty, rx3, ry3) / (2.0 * ck48.r_th)
    assert core_auto48 > 1.5, \
        (f"automatische Verdrehung muss die Kerne beider Hälften "
         f"trennen ({core_auto48:.2f} × Kern-Ø)")
    assert core_3deg48 < 1.0, \
        (f"pauschal 3° legt die Kerne übereinander — genau das war "
         f"der Fehler ({core_3deg48:.2f} × Kern-Ø)")
    print(f"Homogenisierungsgrenze c) K67-Lage Senkungen "
          f"{gap_mouth48:.2f}×Ø, Kerne {core_auto48:.2f}×Ø (pauschal 3°: "
          f"{core_3deg48:.2f}×Ø)  OK")


def test_gp48d_spaltprofil():
    """Gegenprobe 48 d: SPALTPROFIL. Der 3D-Film sieht dasselbe örtliche
    Profil h − w0·φ(r) wie das 2D-Feld (an dessen Zellmitten
    verglichen), nicht das Flächenmittel h_gap_front."""
    if not _HAS_SCIPY:
        return
    cz48, _ = _cap48("2d", **dict(_A12_48, bias_voltage=40.0))
    r2_48 = cz48.r_post + (np.arange(cz48._fld_N) + 0.5) * (
        (cz48.a_bp - cz48.r_post) / cz48._fld_N)
    prof48 = cz48._polarized_gap_profile(r2_48)
    ref48 = cz48.h_gap - cz48.w0_static * cz48._fld_sag_shape
    assert cz48.w0_static > 0.05 * cz48.h_gap, \
        "Prüfling muss merklich durchgebogen sein"
    assert np.max(np.abs(prof48 - ref48)) < 1e-12 * cz48.h_gap, \
        "3D-Spaltprofil muss das des 2D-Felds sein (exakte Form, GP 49)"
    assert prof48[0] < cz48.h_gap_front < prof48[-1], \
        "Mitte enger, Rand weiter als das Flächenmittel"
    print("Homogenisierungsgrenze d) 3D-Spaltprofil == 2D-Feld, Mitte "
          "enger, Rand weiter als das Flächenmittel  OK")


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp48e1_grenze_trennt_a48():
    """Gegenprobe 48 e1: die Grenze TRENNT. Unterhalb von f_hom trifft 2D
    das konvergierte 3D-Feld auf 1 dB, darüber nicht (48 Löcher,
    f_hom im Band; feines 3D-Gitter, s. GP 50)."""
    if not _HAS_SCIPY:
        return
    fhA48, dA48 = _fein48(dict(_PA48, n_through_holes=48,
                               through_hole_diameter=0.70e-3),
                          (1000.0, 12000.0))
    assert 1000.0 < fhA48 < 12000.0, \
        f"Prüfling A48 muss die Grenze im Band haben ({fhA48:.0f} Hz)"
    assert abs(dA48[0]) < 1.0, \
        (f"unterhalb f_hom muss 2D das 3D-Feld treffen "
         f"({dA48[0]:+.2f} dB bei 1 kHz, f_hom {fhA48:.0f} Hz)")
    assert abs(dA48[1]) > 1.5, \
        (f"oberhalb f_hom muss die Abweichung sichtbar sein "
         f"({dA48[1]:+.2f} dB bei 12 kHz)")
    print(f"Homogenisierungsgrenze e1) 2D/3D A48 {dA48[0]:+.2f} dB unter / "
          f"{dA48[1]:+.2f} dB über f_hom {fhA48 / 1e3:.1f} kHz  OK")


@pytest.mark.feld3d
def test_gp48e2_sehr_spaerlich_a12():
    """Gegenprobe 48 e2: sehr spärliches Raster (12 Löcher): f_hom unter
    1 kHz, bei 4 kHz weit daneben (feines 3D-Gitter)."""
    if not _HAS_SCIPY:
        return
    fhA12, dA12 = _fein48(_A12_48, (4000.0,))
    assert fhA12 < 1000.0 and abs(dA12[0]) > 4.0, \
        (f"sehr spärliches Raster: f_hom {fhA12:.0f} Hz, Abweichung "
         f"{dA12[0]:+.2f} dB bei 4 kHz")
    print(f"Homogenisierungsgrenze e2) A12 {dA12[0]:+.1f} dB  OK")


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp48e3_rand_des_pruefbands_b24():
    """Gegenprobe 48 e3: Kapsel mit f_hom am oberen Rand des Prüfbands
    (B24; rein viskos 13,5 kHz, mit Filmträgheit und Beulresonanz
    11,4 kHz, Gegenprobe 53).

    Oberhalb der Membranresonanz liegt 3D auch bei DICHTEN Lochbildern
    über 2D (gleich bei 24, 48 und 96 Löchern gleicher Lochfläche) — die
    Formanpassung der Membran, die das 2D-Einmodenbild nicht kann
    (Gegenprobe 52), keine Homogenisierung. Deren Anteil ist, was beim
    AUSDÜNNEN dazukommt: gegen das doppelt so dichte Raster gleicher
    Lochfläche bleibt er bis 12 kHz unter 1 dB — auch knapp über f_hom,
    die Grenze liegt auf der sicheren Seite; unterhalb des Hochtons
    trifft 2D auch absolut."""
    if not _HAS_SCIPY:
        return
    ffB24 = (1000.0, 4000.0, 12000.0)
    fhB24, dB24 = _fein48(dict(_PB48, n_through_holes=24,
                               through_hole_diameter=0.4554e-3), ffB24)
    _, dB48 = _fein48(dict(_PB48, n_through_holes=48,
                           through_hole_diameter=0.4554e-3 / np.sqrt(2.0)),
                      ffB24)
    assert 4000.0 < fhB24 < 20000.0 \
            and np.all(np.abs(dB24 - dB48) < 1.0) \
            and np.all(np.abs(dB24[:2]) < 1.0), \
        (f"Kapsel mit f_hom am Rand des Prüfbands: Homogenisierungsanteil "
         f"unter 1 dB (f_hom {fhB24:.0f} Hz, 2D/3D {np.round(dB24, 2)} "
         f"dB, doppelt so dicht {np.round(dB48, 2)} dB)")
    print(f"Homogenisierungsgrenze e3) B24 Homogenisierungsanteil max "
          f"{np.max(np.abs(dB24 - dB48)):.2f} dB (f_hom {fhB24 / 1e3:.1f} "
          f"kHz; Formanpassung {dB48[-1]:+.2f} dB bei 12 kHz)  OK")


def test_gp48f_warnung():
    """Gegenprobe 48 f: die Warnung kommt genau dann, wenn f_hom im
    Hörband liegt; summary() nennt die Grenze und die Empfehlung."""
    if not _HAS_SCIPY:
        return
    _, w2_48 = _cap48("2d", **_A12_48)
    _, w3_48 = _cap48("3d", **_A12_48)
    _, wd_48 = _cap48("2d")                    # Standardkapsel
    hom48 = [w for w in w2_48 if "zu spärlich" in w]
    assert len(hom48) == 1 and "0.2 kHz" in hom48[0], \
        f"2D mit 12 Löchern muss warnen, mit f_hom ({w2_48})"
    assert not any("zu spärlich" in w for w in w3_48 + wd_48), \
        "3D und dichtes Standardraster dürfen nicht warnen"
    s48 = _cap48("2d", **_A12_48)[0].summary()
    assert "Loch-Homogenisierung bis:" in s48 and "3D nehmen" in s48, \
        "summary() muss die Grenze und die Empfehlung nennen"
    assert "3D nehmen" not in _cap48("2d")[0].summary(), \
        "Standardkapsel: keine Empfehlung"
    print("Homogenisierungsgrenze f) Warnung genau bei f_hom im Hörband, "
          "summary() nennt Grenze und Empfehlung  OK")


def test_gp48g1_zellkompressibilitaet_moden():
    """Gegenprobe 48 g1/g2: VERWORFENE Ursachen, jeweils mit Beleg.
    g1) kompressible Škvor-Zelle: exakte Form (modifizierte
        Besselfunktionen, Knotenmodell des 2D-Felds) gegen direkte
        FD-Lösung der Zelle, dann ihre Wirkung auf B.
    g2) Modenabbruch: höhere Moden ändern die 2D-Rechnung praktisch
        nicht (sie liegen parallel am selben Spaltknoten)."""
    if not _HAS_SCIPY:
        return
    from scipy.special import ive as _ive48, kve as _kve48

    def _Bdyn48(q, kb2):
        kb = np.sqrt(kb2 + 0j)
        rq = np.sqrt(q)
        E = np.exp((kb.real + kb) * (rq - 1.0))
        N1 = (_kve48(1, kb * rq) * _ive48(1, kb)
              - _ive48(1, kb * rq) * _kve48(1, kb) * E)
        D = (_ive48(0, kb * rq) * _kve48(1, kb) * E
             + _kve48(0, kb * rq) * _ive48(1, kb))
        f1 = 2.0 * rq / (kb * (1.0 - q)) * N1 / D
        return (1.0 - f1) * (1.0 - q) / (kb2 * (q + (1.0 - q) * f1))

    def _Bfd48(q, kb2, M=4000):
        r_ = np.linspace(np.sqrt(q), 1.0, M + 1)
        h_ = r_[1] - r_[0]
        n_ = M + 1
        main = np.zeros(n_, complex)
        lo = np.zeros(n_, complex)
        up = np.zeros(n_, complex)
        rhs = np.zeros(n_, complex)
        for i in range(1, n_):
            last = i == n_ - 1
            vol = (r_[i] - h_ / 4) * h_ / 2 if last else r_[i] * h_
            cp = 0.0 if last else (r_[i] + h_ / 2) / h_
            cm = (r_[i] - h_ / 2) / h_
            main[i] = cp + cm + kb2 * vol
            lo[i] = -cm
            if not last:
                up[i] = -cp
            rhs[i] = vol
        main[0] = 1.0
        from scipy.sparse import diags as _diags48
        from scipy.sparse.linalg import spsolve as _sps48
        p_ = _sps48(_diags48([lo[1:], main, up[:-1]], [-1, 0, 1],
                             format="csc"), rhs)
        w_ = r_ * h_
        w_[0] *= 0.5
        w_[-1] *= 0.5
        Ip = 2.0 * np.pi * np.sum(w_ * p_)
        return (Ip / np.pi) / (np.pi - kb2 * Ip) * np.pi

    err48 = max(abs(_Bdyn48(q, 1j * s) / _Bfd48(q, 1j * s) - 1.0)
                for q in (0.01, 0.04, 0.2) for s in (0.1, 1.0, 10.0))
    B0_48 = 0.04 / 2 - 0.04**2 / 8 - np.log(0.04) / 4 - 3 / 8
    chg48 = abs(abs(_Bdyn48(0.04, 1j * 1.0)) / B0_48 - 1.0)
    assert err48 < 1e-5, \
        f"geschlossene Zellform muss die FD-Zelle treffen ({err48:.1e})"
    assert abs(_Bdyn48(0.04, 1e-9j) / B0_48 - 1.0) < 1e-6, \
        "statischer Grenzfall muss Škvor sein"
    assert chg48 < 0.01, \
        (f"Zellkompressibilität bis σ_c = 1 unter 1 % — also NICHT "
         f"die Ursache ({100 * chg48:.2f} %)")
    kA24 = dict(_PA48, n_through_holes=24, through_hole_diameter=0.99e-3)
    fm48 = np.array([4000.0, 8000.0])
    dmode48 = np.max(np.abs(
        _db48(_cap48("2d", membrane_modes=3, **kA24)[0]
              .transfer_function(fm48))
        - _db48(_cap48("2d", **kA24)[0].transfer_function(fm48))))
    assert dmode48 < 0.05, \
        f"Modenabbruch ist nicht die Ursache ({dmode48:.3f} dB)"
    print(f"Homogenisierungsgrenze g1/g2) verworfen: Zellkompressibilität "
          f"{100 * chg48:.2f} %, Moden {dmode48:.3f} dB  OK")


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp48g3_sackloecher():
    """Gegenprobe 48 g3: VERWORFENE Ursache — Sacklöcher retten die
    Homogenisierung nicht (36 Sacklöcher zwischen 12 Durchgangslöchern,
    3D auf 90 × 288 Zellen)."""
    if not _HAS_SCIPY:
        return
    kSB48 = dict(_A12_48, n_blind_holes=36, blind_hole_diameter=1.0e-3,
                 blind_hole_depth=1.5e-3)
    fsb48 = np.array([8000.0])
    dsb48 = float(_db48(
        _cap48("2d", **kSB48)[0].transfer_function(fsb48)
        / _grid48(MicrophoneCapsule(squeeze_model="3d", **kSB48),
                  90, 288).transfer_function(fsb48))[0])
    assert abs(dsb48) > 3.0, \
        (f"36 Sacklöcher zwischen 12 Durchgangslöchern dürfen die "
         f"Abweichung nicht beseitigen ({dsb48:+.2f} dB)")
    print(f"Homogenisierungsgrenze g3) verworfen: Sacklöcher "
          f"{dsb48:+.1f} dB  OK")


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp53_warnlucke_weiter_spalt_lochkreise():
    """Gegenprobe 53: Warnlücke weiter Spalt / Lochkreise."""
    # Die Homogenisierungsgrenze (Gegenprobe 48) war beim weiten Spalt und
    # bei Lochkreisen zu optimistisch; zwei Fälle mit > 1 dB blieben ganz
    # ohne Warnung. Neu vermessen gegen den 3D-Löser (114 Fälle: zwei
    # Kapseln, Spalte 20/25/38/65 µm, gleichverteilt, ein und zwei
    # Lochkreise) mit einem robusteren Maß und zwei Ursachen:
    # a) MESSMASS: Mehrabweichung E = |2D/3D| − |2D/3D dicht| gegen das
    #    dichte Raster gleicher Lochfläche. Die vorzeichenrichtige
    #    Differenz schob beim weiten Spalt die eigene Resonanzabweichung des
    #    dichten Rasters (scharfe Resonanz, ±1,5 dB) in den Befund: bei der
    #    1"-Kapsel mit 65 µm und 16 Löchern weicht das spärliche Lochbild
    #    bei 1,2 kHz selbst kaum ab.
    # b) WEITER SPALT: die Filmkraft über der Beule ist nicht rein viskos.
    #    Mit der vollen Filmleitfähigkeit K(ω) (Trägheit der Spaltluft)
    #    und der dynamischen Steifigkeit der Beule (Beulresonanz
    #    f_ρ = f_res·a_mem/ρ) liegt die Grenze der steifen Kapsel mit 65 µm
    #    und 16 Löchern bei 17,6 statt 107 kHz; die eigene Abweichung
    #    überschreitet 1 dB erst bei etwa 19 kHz.
    #    Die Schwelle Π = 10 bleibt; im Tiefton ändert sich nichts.
    # c) LOCHKREISE: das Radialfeld verschmiert jeden Lochkreis zu einem
    #    Band; bei vielen Löchern (Liniensenke) hängt das Ergebnis von
    #    dieser Darstellungswahl um mehrere dB ab, der Filmwiderstand bei
    #    erzwungener Form stimmt dagegen auf 3 % mit 3D überein. Gewarnt
    #    wird, wo Band und Liniensenke um mehr als 0,5 dB auseinander-
    #    liegen (s. _RING_REPR_DB). Die alte Grenze der steifen Kapsel mit
    #    48 Löchern auf einem Kreis (390 Hz) lag über dem Einsatz.
    # d) Vollständigkeit über alle 114 Fälle steht im README (knappster
    #    Fall 0,95 dB an der Warnfrequenz); hier die tragenden Stichproben.
    if _HAS_SCIPY:
        pA53 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
            delay_length=0.0, cavity_length=8.0e-3,
            cavity_wall_thickness=1.5e-3, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=28e-3)
        pB53 = dict(
            architecture="single", membrane_resonance_hz=8000.0,
            membrane_diameter=12.0e-3, membrane_thickness=5e-6,
            membrane_tension=400.0, air_gap=25e-6,
            backplate_diameter=11.0e-3, backplate_thickness=1.5e-3,
            bias_voltage=1.0, n_blind_holes=0, rear_network_enabled=True,
            delay_length=0.0, cavity_length=4.0e-3,
            cavity_wall_thickness=1.0e-3, n_cavity_holes=0,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=14e-3)

        def _pat53(base, n, ring=None, **extra):
            # gleiche Lochfläche wie 48 Löcher Ø0.35 mm·a/11.95 mm
            a_ = 0.5 * base["backplate_diameter"]
            A_ = 48 * np.pi * (0.35e-3 * a_ / 11.95e-3) ** 2
            q = dict(base, n_through_holes=n,
                     through_hole_diameter=2.0 * np.sqrt(A_ / (n * np.pi)),
                     **extra)
            if ring is not None:
                q["through_hole_rings"] = [(n, ring * 2.0 * a_)]
            return q

        def _dev53(q, f):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c2 = MicrophoneCapsule(squeeze_model="2d", **q)
            c3 = MicrophoneCapsule(squeeze_model="3d", **q)
            return (20.0 * np.log10(np.abs(c2.transfer_function(f)
                                           / c3.transfer_function(f))),
                    c2.homogenization_limit())

        # a) Messmaß: 1"-Kapsel, 65 µm — bei 1.2 kHz trägt das DICHTE
        #    Raster die Resonanzabweichung, nicht das spärliche
        fa53 = np.array([1185.0])
        rawA16, _ = _dev53(_pat53(pA53, 16, air_gap=65e-6), fa53)
        rawA96, _ = _dev53(_pat53(pA53, 96, air_gap=65e-6), fa53)
        assert abs(rawA16[0]) < 0.5 and abs(rawA96[0]) > 1.0, \
            (f"Messmaß: das spärliche Lochbild weicht selbst kaum ab "
             f"({rawA16[0]:+.2f} dB), das dichte trägt die Resonanz"
             f"abweichung ({rawA96[0]:+.2f} dB)")

        # b) weiter Spalt: steife Kapsel, 65 µm, 16 Löcher
        qW53 = _pat53(pB53, 16, air_gap=65e-6)
        with warnings.catch_warnings(record=True) as recW53:
            warnings.simplefilter("always")
            cW53 = MicrophoneCapsule(squeeze_model="2d", **qW53)
        limW53 = cW53.homogenization_limit()
        f_visc53 = MicrophoneCapsule._PI_HOM / (
            2.0 * np.pi * limW53["pi_per_omega"])
        assert f_visc53 > 50e3 and limW53["f_hom"] < 18.5e3, \
            (f"weiter Spalt: rein viskos {f_visc53 / 1e3:.0f} kHz (keine "
             f"Warnung), mit Trägheit + Beulresonanz "
             f"{limW53['f_hom'] / 1e3:.1f} kHz")
        assert limW53["f_hom"] < limW53["f_rho"], \
            "die Grenze liegt unter der Beulresonanz"
        assert any("zu spärlich" in str(r.message) for r in recW53), \
            "der Fall muss jetzt warnen"
        fW53 = np.array([limW53["f_hom"], 20000.0])
        dW16, _ = _dev53(qW53, fW53)
        dW96, _ = _dev53(_pat53(pB53, 96, air_gap=65e-6), fW53)
        eW53 = np.abs(dW16) - np.abs(dW96)
        # eigene 2D/3D-Abweichung (das dichte Raster kreuzt hier mit
        # −0.2 dB die Null, die Mehrabweichung liegt bei 20 kHz knapp
        # unter 1 dB und wird ausgegeben)
        assert abs(dW16[0]) < 1.0 < abs(dW16[1]), \
            (f"weiter Spalt: Abweichung an der Grenze unter, bei 20 kHz "
             f"über 1 dB ({np.round(dW16, 2)})")
        # Tiefton unverändert: dort ist die Filmkraft rein viskos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c12_53 = MicrophoneCapsule(squeeze_model="2d", **dict(
                pA53, n_through_holes=12, through_hole_diameter=1.4e-3))
        l12_53 = c12_53.homogenization_limit()
        f12v53 = MicrophoneCapsule._PI_HOM / (2.0 * np.pi
                                              * l12_53["pi_per_omega"])
        assert abs(l12_53["f_hom"] / f12v53 - 1.0) < 0.01, \
            (f"Tiefton: Grenze == rein viskose Form "
             f"({l12_53['f_hom']:.0f} vs. {f12v53:.0f} Hz)")

        # c) Lochkreis: steife Kapsel, 48 Löcher auf einem Kreis (0.67·a)
        qR53 = _pat53(pB53, 48, ring=0.67)
        with warnings.catch_warnings(record=True) as recR53:
            warnings.simplefilter("always")
            cR53 = MicrophoneCapsule(squeeze_model="2d", **qR53)
        limR53 = cR53.homogenization_limit()
        assert limR53["cause"] == "ring" and limR53["f_limit"] < 200.0 \
            and limR53["f_hom"] > 350.0, \
            (f"Lochkreis: Darstellungsgrenze {limR53['f_ring']:.0f} Hz vor "
             f"der lokalen Grenze {limR53['f_hom']:.0f} Hz")
        assert any("Lochkreis-Darstellung" in str(r.message)
                   for r in recR53), "der Lochkreis-Fall muss warnen"
        assert "Lochkreis-Darstellung" in cR53.summary(), \
            "summary() muss die Ursache nennen"
        fR53 = np.array([limR53["f_limit"], limR53["f_hom"]])
        dR48, _ = _dev53(qR53, fR53)
        dR96, _ = _dev53(_pat53(pB53, 96), fR53)
        eR53 = np.abs(dR48) - np.abs(dR96)
        assert eR53[0] < 1.0 < eR53[1], \
            (f"Lochkreis: Mehrabweichung an der neuen Grenze unter, an "
             f"der alten lokalen Grenze über 1 dB ({np.round(eR53, 2)})")
        # ... die Darstellung ist die Ursache, nicht der Filmwiderstand:
        #     bei erzwungener Form (steife Membran, Phasenmethode wie
        #     Gegenprobe 52) stimmt er auf 3 %
        def _rr53(q):
            q = dict(q, membrane_resonance_hz=300e3)
            q.pop("membrane_tension", None)
            ph = {}
            for sm in ("2d", "3d"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cc = MicrophoneCapsule(squeeze_model=sm, **q)
                ph[sm] = np.angle(cc.transfer_function([1000.0])[0]
                                  / cc.transfer_function([20.0])[0])
            return float(np.tan(ph["3d"]) / np.tan(ph["2d"]))
        rrR53 = _rr53(qR53)
        assert abs(rrR53 - 1.0) < 0.03, \
            (f"Lochkreis: Filmwiderstand bei erzwungener Form 3D/2D = "
             f"{rrR53:.3f}")
        # ... und die Spanne der Darstellungen ist groß (Kreis nahe der
        #     Mitte, 48 Löcher)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cL53 = MicrophoneCapsule(squeeze_model="2d",
                                     **_pat53(pB53, 48, ring=0.33))
        ffL53 = np.logspace(np.log10(20.0), np.log10(20e3), 64)
        pfL, prL = cL53._source_pressures(2 * np.pi * ffL53, np.array([0.0]))
        qL53 = []
        keepL = cL53._fld_dens_th
        for dens in (keepL, cL53._fld_dens_th_line):
            cL53._fld_dens_th = dens
            TtL, TrL = cL53._assemble_network(2 * np.pi * ffL53)
            qL53.append(cL53._membrane_volume_velocity(
                2 * np.pi * ffL53, TtL, TrL, pfL[:, 0], prL[:, 0]))
        cL53._fld_dens_th = keepL
        spanL53 = float(np.max(np.abs(20 * np.log10(np.abs(qL53[0]
                                                           / qL53[1])))))
        assert spanL53 > 3.0, \
            f"Band gegen Liniensenke muss mehrere dB ausmachen ({spanL53:.1f})"

        # e) Gleichverteilte Löcher und 3D prüfen die Darstellung nicht
        assert not np.isfinite(cW53.homogenization_limit()["f_ring"]), \
            "gleichverteilte Löcher: keine Lochkreis-Grenze"
        c3R53 = MicrophoneCapsule(squeeze_model="3d", **qR53)
        assert not np.isfinite(c3R53._ring_repr_limit()), \
            "3D löst die Löcher auf: keine Lochkreis-Grenze"
        print(f"Warnlücke geschlossen: Messmaß Mehrabweichung (1\" 65 µm, "
              f"16 Löcher selbst {rawA16[0]:+.2f} dB, dicht "
              f"{rawA96[0]:+.2f} dB); weiter Spalt 16 Löcher: Grenze "
              f"{limW53['f_hom'] / 1e3:.1f} kHz statt "
              f"{f_visc53 / 1e3:.0f} kHz (f_ρ {limW53['f_rho'] / 1e3:.1f} "
              f"kHz), Abweichung {abs(dW16[0]):.2f} -> {abs(dW16[1]):.2f} "
              f"dB (E {eW53[0]:.2f} -> {eW53[1]:.2f}); Lochkreis 48: "
              f"Grenze {limR53['f_limit']:.0f} Hz statt "
              f"{limR53['f_hom']:.0f} Hz, E {eR53[0]:.2f} -> {eR53[1]:.2f} "
              f"dB, Filmwiderstand 3D/2D {rrR53:.3f}, Spanne Band/Linie "
              f"{spanL53:.1f} dB  OK")
