"""Gegenprobe 63: Eingabebereiche der GUI und Beispielprojekte (app.py)."""
import glob
import json
import logging
import os
import warnings

import numpy as np
import pytest

from basis import *  # noqa: F401,F403

_WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BEISPIELE = sorted(glob.glob(os.path.join(_WURZEL, "examples", "*.json")))
_NPTS = 100     # Frequenzpunkte im App-Lauf (Simulationseinstellung)


@pytest.fixture(scope="module")
def app():
    """app.py als Modul. Es ist ein Streamlit-Skript: der Import führt es
    ohne Laufzeit einmal aus (bare mode); dessen Hinweise sind hier
    belanglos."""
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import app as A
    finally:
        logging.disable(logging.NOTSET)
    return A


def _projekt(A, pfad):
    with open(pfad, encoding="utf-8") as fh:
        data = json.load(fh)
    return data, A._projekt_params(data)[0]


def _app_lauf(A, params):
    """Ein Lauf der echten App mit ``params`` im Session-State, so wie
    _load_project() sie schreibt (der Uploader selbst ist im AppTest
    nicht bedienbar). Warnungen des Modells laufen in der App nur ins
    Konsolenprotokoll, deshalb hier ebenso keine Fehler."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=600)
    for key, val in params.items():
        if key in A._RING_PREFIX:
            pre = A._RING_PREFIX[key]
            for i, (cnt, pcd) in enumerate(val):
                at.session_state[f"p_{pre}_ring_n_{i}"] = cnt
                at.session_state[f"p_{pre}_ring_pcd_{i}"] = pcd
            at.session_state[f"{pre}_ring_count"] = len(val)
        else:
            at.session_state["p_" + key] = val
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    return at


def _zustand(A, at):
    """Parametersatz, mit dem die App nach dem Lauf rechnet."""
    p = {}
    for key in A.DEFAULTS:
        if key in A._RING_PREFIX:
            pre = A._RING_PREFIX[key]
            p[key] = [[at.session_state[f"p_{pre}_ring_n_{i}"],
                       at.session_state[f"p_{pre}_ring_pcd_{i}"]]
                      for i in range(at.session_state[f"{pre}_ring_count"])]
        else:
            p[key] = at.session_state["p_" + key]
    return p


def _klemm_warnungen(at):
    return [w.value for w in at.warning
            if "clamped" in w.value or "geklemmt" in w.value]


def _frequenzgang(at):
    """Die Amplitudenkurve des Frequenzgang-Diagramms (Plotly-Spezifikation
    mit binär kodierten Feldern)."""
    import base64
    spur = json.loads(at.get("plotly_chart")[0].proto.spec)["data"][0]

    def feld(v):
        if isinstance(v, dict):
            return np.frombuffer(base64.b64decode(v["bdata"]),
                                 dtype=np.dtype(v["dtype"]))
        return np.asarray(v, dtype=float)
    return feld(spur["x"]), feld(spur["y"])


def test_gp63_eingabebereiche_der_gui(app):
    """Gegenprobe 63: Projektwerte liegen in den Feldbereichen der GUI."""
    # Streamlit (1.64) setzt einen Session-Wert außerhalb [min, max] beim
    # Aufbau des Zahlenfelds still auf das Minimum. Das B&K-4134-Beispiel
    # (Randspalt 860 µm) lief bei einem Feldmaximum von 500 µm so mit
    # geschlossenem Randspalt. Diese Probe prüft jede Projektdatei, die
    # Voreinstellung und den Null-Zustand gegen die EINE Bereichstabelle,
    # aus der auch die Widgets ihre Grenzen nehmen (app._bereich). Jeder
    # Zahlenschlüssel muss einen Bereich haben.
    A = app
    assert len(_BEISPIELE) >= 6, _BEISPIELE
    for pfad in _BEISPIELE:
        name = os.path.basename(pfad)
        data, p = _projekt(A, pfad)
        unbekannt = sorted(set(data["params"]) - set(A.DEFAULTS))
        assert not unbekannt, \
            f"{name}: Schlüssel, die der Lader still verwirft: {unbekannt}"
        aus = A._bereichsverletzungen(p)
        assert not aus, f"{name}: außerhalb des Feldbereichs {aus}"
    assert not A._bereichsverletzungen(dict(A.DEFAULTS)), "DEFAULTS"
    null = dict(A.DEFAULTS, **A._ZERO_STATE,
                th_rings=[[0, 0.0]], bh_rings=[[0, 0.0]])
    assert not A._bereichsverletzungen(null), "Null-Zustand"
    # Gegenprobe der Prüfung: der frühere Fehler wäre gefunden worden
    _, p = _projekt(A, os.path.join(_WURZEL, "examples",
                                    "bk4134_comsol_geometrie.json"))
    alt = dict(A._BEREICH)
    try:
        A._BEREICH["ring_vent_um"] = (0.0, 500.0)
        assert A._bereichsverletzungen(p) == [
            ("ring_vent_um", 860.0, 0.0, 500.0)]
    finally:
        A._BEREICH.clear()
        A._BEREICH.update(alt)
    print(f"Eingabebereiche: {len(_BEISPIELE)} Beispielprojekte, "
          f"Voreinstellung und Null-Zustand in den Feldbereichen, keine "
          f"verworfenen Schlüssel; altes Randspalt-Maximum 500 µm wäre "
          f"gemeldet worden  OK")


def test_gp63b_beispiele_in_der_app(app):
    """Gegenprobe 63 b: die App rechnet jedes Beispiel so, wie es in der
    Datei steht."""
    # Durch die echte App (Streamlit-AppTest): kein Fehler, keine
    # Klemmung, jeder Wert unverändert im Session-State, und die
    # gezeichnete Amplitudenkurve ist der Druckfrequenzgang des Modells
    # aus denselben Parametern (bis auf die 1-kHz-Normierung).
    A = app
    for pfad in _BEISPIELE:
        name = os.path.basename(pfad)
        _, p = _projekt(A, pfad)
        p["n_points"] = _NPTS
        at = _app_lauf(A, p)
        assert not at.exception, \
            f"{name}: {[e.value for e in at.exception]}"
        assert not _klemm_warnungen(at), f"{name}: {_klemm_warnungen(at)}"
        jetzt = _zustand(A, at)
        geaendert = {k: (p[k], jetzt[k]) for k in p if jetzt[k] != p[k]}
        assert not geaendert, f"{name}: App rechnet mit {geaendert}"
        f, y = _frequenzgang(at)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = A.build_capsule(p).transfer_function(f)
        d = y - 20.0 * np.log10(np.abs(H))
        assert np.ptp(d) < 1e-9, f"{name}: Kurve weicht ab ({np.ptp(d)})"
        if not p["normalize_1khz"]:
            assert abs(d[0]) < 1e-9, f"{name}: Pegel {d[0]}"
        print(f"  {name}: {len(f)} Punkte, Abweichung "
              f"{np.ptp(d):.1e} dB")
    print(f"Beispiele in der App: {len(_BEISPIELE)} Projekte ohne Fehler "
          f"und ohne Klemmung, Kurve == Modell  OK")


def test_gp63c_klemmen_statt_nullsetzen(app):
    """Gegenprobe 63 c: Werte außerhalb werden auf die nächste Grenze
    geklemmt und gemeldet, nicht still auf das Minimum gesetzt."""
    A = app
    _, p = _projekt(A, os.path.join(_WURZEL, "examples",
                                    "bk4134_comsol_geometrie.json"))
    p.update(n_points=_NPTS, ring_vent_um=2500.0, th_rings=[[6, 9.0]],
             blind_depth_mm=5.0)
    at = _app_lauf(A, p)
    assert not at.exception, [e.value for e in at.exception]
    jetzt = _zustand(A, at)
    assert jetzt["ring_vent_um"] == 2000.0, jetzt["ring_vent_um"]
    assert jetzt["th_rings"] == [[6, 7.2]], jetzt["th_rings"]
    bd_max = max(0.1, p["bp_thickness_mm"] - 0.1)
    assert jetzt["blind_depth_mm"] == pytest.approx(bd_max), \
        jetzt["blind_depth_mm"]
    meld = _klemm_warnungen(at)
    assert len(meld) == 1, meld
    for teil in ("2500 → 2000", "9 → 7.2", "5 → "):
        assert teil in meld[0], (teil, meld[0])
    # Ein weiterer Lauf: der geklemmte Wert liegt jetzt im Bereich, die
    # Meldung entfällt, der Wert bleibt.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    assert not _klemm_warnungen(at)
    assert at.session_state["p_ring_vent_um"] == 2000.0
    # Vorher-Befund: genau der Randspalt 0 (Streamlits Minimum) erklärt
    # den gemeldeten Abfall ab ~700 Hz — 10 kHz liegt dann über 5 dB
    # unter dem Frequenzgang mit dem Randspalt aus der Datei.
    _, p = _projekt(A, os.path.join(_WURZEL, "examples",
                                    "bk4134_comsol_geometrie.json"))
    f = np.array([100.0, 700.0, 10000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = {v: 20.0 * np.log10(np.abs(A.build_capsule(
            dict(p, ring_vent_um=v)).transfer_function(f)))
            for v in (860.0, 0.0)}
    rel = {v: L[v] - L[v][0] for v in L}
    assert abs(rel[860.0][1]) < 0.05 and rel[860.0][2] > 0.5, rel
    assert rel[0.0][2] < rel[860.0][2] - 5.0, rel
    print(f"Klemmen: Randspalt 2500 → 2000 µm, Lochkreis 9 → 7.2 mm, "
          f"Sacklochtiefe 5 → {bd_max:g} mm, eine Meldung, im nächsten "
          f"Lauf keine. Vorher (Randspalt 0): 10 kHz "
          f"{rel[0.0][2]:+.1f} dB statt {rel[860.0][2]:+.1f} dB re 100 Hz "
          f"OK")
