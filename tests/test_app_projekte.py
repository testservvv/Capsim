"""Gegenproben 63, 66: Eingabebereiche, Beispielprojekte und Sprachen der
GUI (app.py)."""
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


def _texte(at):
    """Alle sichtbaren Texte eines App-Laufs: Überschriften, Meldungen,
    Beschriftungen und Hilfen der Felder, angezeigte Optionen (bei Radio-
    und Auswahlfeldern NICHT der Wert — der ist kanonisch, angezeigt wird
    die Übersetzung) und die Titel und Spurnamen der Diagramme."""
    out = set()
    for typ in ("markdown", "caption", "title", "subheader", "header",
                "warning", "info", "success", "error", "metric", "expander",
                "number_input", "toggle", "checkbox", "radio", "selectbox",
                "multiselect", "slider", "button", "tab", "code", "text"):
        try:
            els = at.get(typ)
        except Exception:
            continue
        auswahl = typ in ("radio", "selectbox", "multiselect")
        for e in els:
            for attr in (("label", "help") if auswahl
                         else ("value", "label", "help")):
                v = getattr(e, attr, None)
                if isinstance(v, str) and v:
                    out.add(v.strip())
            if auswahl:
                out.update(str(o).strip() for o in e.options)

    def sammle(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("text", "name") and isinstance(v, str):
                    out.add(v.strip())
                else:
                    sammle(v)
        elif isinstance(x, list):
            for v in x:
                sammle(v)
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        sammle(spec.get("layout", {}))
        sammle([{"name": t.get("name")} for t in spec.get("data", [])])
    return out


def test_gp66_uebersetzungen(app):
    """Gegenprobe 66: Übersetzungstabellen vollständig und benutzt."""
    # a) Jeder Eintrag in TR und LABEL_TR hat genau Englisch und Deutsch,
    #    beide nicht leer, mit denselben Platzhaltern (sonst scheitert
    #    tr(...).format in nur EINER Sprache), und beide lassen sich mit
    #    denselben Werten füllen.
    # b) Jeder tr()-Aufruf in app.py nennt einen vorhandenen Schlüssel
    #    (aus dem Quelltext, auch beide Zweige von "a if x else b"), es
    #    gibt keine dynamisch gebildeten Schlüssel, und kein Schlüssel ist
    #    unbenutzt. Gegenprobe: der unbenutzte Schlüssel prog_di war die
    #    Spur zum fest deutschen Fortschrittstext „Richtdiagramm … Hz“,
    #    den auch die englische Oberfläche zeigte.
    import ast
    import string
    import translations as T
    fmt = string.Formatter()

    def felder(txt):
        return {f for _, f, _, _ in fmt.parse(txt) if f is not None}

    fehler = []
    for name, tab in (("TR", T.TR), ("LABEL_TR", T.LABEL_TR)):
        for k, v in tab.items():
            if not isinstance(v, dict) or set(v) != {"en", "de"}:
                fehler.append(f"{name}[{k}]: Sprachen {v!r:.60}")
                continue
            if not all(isinstance(x, str) and x.strip() for x in v.values()):
                fehler.append(f"{name}[{k}]: leer")
                continue
            if felder(v["en"]) != felder(v["de"]):
                fehler.append(f"{name}[{k}]: Platzhalter "
                              f"{felder(v['en'])} / {felder(v['de'])}")
                continue
            werte = {f: 1.5 for f in felder(v["en"])}
            for lang in ("en", "de"):
                try:
                    v[lang].format(**werte)
                except Exception as exc:
                    fehler.append(f"{name}[{k}].{lang}: {exc}")
    assert not fehler, fehler
    src = open(os.path.join(_WURZEL, "app.py"), encoding="utf-8").read()
    baum = ast.parse(src)

    def konst(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.IfExp):
            a_, b_ = konst(node.body), konst(node.orelse)
            return a_ + b_ if a_ is not None and b_ is not None else None
        return None

    benutzt, dynamisch = set(), []
    for node in ast.walk(baum):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tr" and node.args):
            k = konst(node.args[0])
            if k is None:
                dynamisch.append(ast.get_source_segment(src, node))
            else:
                benutzt.update(k)
    assert not dynamisch, f"dynamische tr-Schlüssel: {dynamisch}"
    fehlt = sorted(benutzt - set(T.TR))
    assert not fehlt, f"tr() mit unbekanntem Schlüssel: {fehlt}"
    tot = sorted(set(T.TR) - benutzt)
    assert not tot, f"unbenutzte Übersetzungen (vergessene Stelle?): {tot}"
    print(f"Übersetzungen: {len(T.TR)} Texte + {len(T.LABEL_TR)} Auswahl-"
          f"werte, alle zweisprachig mit gleichen Platzhaltern; "
          f"{len(benutzt)} Schlüssel in app.py, keiner fehlt, keiner "
          f"unbenutzt  OK")


@pytest.mark.slow
def test_gp66b_oberflaeche_in_beiden_sprachen(app):
    """Gegenprobe 66 b: die Oberfläche spricht durchgehend EINE Sprache."""
    # Jedes Beispielprojekt und die Voreinstellung laufen auf Englisch und
    # auf Deutsch durch die App: kein Fehler, und kein sichtbarer Text ist
    # ein fester Text der jeweils anderen Sprache (verglichen werden alle
    # Einträge ohne Platzhalter, deren Sprachen sich unterscheiden, als
    # ganzer Text).
    import translations as T
    fremd = {"de": set(), "en": set()}
    for tab in (T.TR, T.LABEL_TR):
        for v in tab.values():
            if "{" in v["en"] or v["en"].strip() == v["de"].strip():
                continue
            fremd["de"].add(v["en"].strip())
            fremd["en"].add(v["de"].strip())
    A = app
    laeufe = 0
    for lang in ("de", "en"):
        for pfad in [None] + _BEISPIELE:
            if pfad is None:
                p = dict(A.DEFAULTS)
            else:
                _, p = _projekt(A, pfad)
            p["n_points"] = _NPTS
            from streamlit.testing.v1 import AppTest
            at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                                   default_timeout=600)
            at.session_state["ui_lang"] = lang
            for key, val in p.items():
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
            name = os.path.basename(pfad) if pfad else "Voreinstellung"
            assert not at.exception, \
                f"{lang}, {name}: {[e.value for e in at.exception]}"
            leck = sorted(t for t in _texte(at) if t in fremd[lang])
            assert not leck, f"{lang}, {name}: fremdsprachig {leck}"
            titel = T.TR["app_title"][lang]
            assert titel in _texte(at), f"{lang}, {name}: Titel fehlt"
            laeufe += 1
    print(f"Oberfläche in beiden Sprachen: {laeufe} App-Läufe "
          f"(Voreinstellung und {len(_BEISPIELE)} Beispiele, EN und DE) "
          f"ohne Fehler und ohne Text der anderen Sprache  OK")
