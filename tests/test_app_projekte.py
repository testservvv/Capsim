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
            _ringe_setzen(A, at, key, val)
        else:
            at.session_state["p_" + key] = val
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    return at


def _ringe_setzen(A, at, key, val):
    """Lochkreis-Zeilen [Anzahl, Ø(, Winkel)] in den Session-State, wie
    _set_ring_state() sie schreibt (Winkel None = Lagekonvention)."""
    pre = A._RING_PREFIX[key]
    for i, zeile in enumerate(val):
        at.session_state[f"p_{pre}_ring_n_{i}"] = zeile[0]
        at.session_state[f"p_{pre}_ring_pcd_{i}"] = zeile[1]
        at.session_state[f"p_{pre}_ring_phi_{i}"] = (
            zeile[2] if len(zeile) > 2 else None)
    at.session_state[f"{pre}_ring_count"] = len(val)


def _ss(at, key, sonst=None):
    try:
        return at.session_state[key]
    except KeyError:
        return sonst


def _zustand(A, at):
    """Parametersatz, mit dem die App nach dem Lauf rechnet."""
    p = {}
    for key in A.DEFAULTS:
        if key in A._RING_PREFIX:
            pre = A._RING_PREFIX[key]
            p[key] = [A._ringzeile(at.session_state[f"p_{pre}_ring_n_{i}"],
                                   at.session_state[f"p_{pre}_ring_pcd_{i}"],
                                   _ss(at, f"p_{pre}_ring_phi_{i}"))
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


def test_gp63d_lochkreise_nicht_still_gekuerzt(app):
    """Gegenprobe 63 d: Lochkreis-Listen werden beim Laden nicht still
    gekürzt."""
    # Bis Gegenprobe 71 übernahm der Projekt-Lader höchstens 8 Lochkreise
    # je Lochtyp und verwarf den Rest ohne Meldung. Das quadratische
    # 2-mm-Raster der K67 liegt auf 13 Radien: geladen wurden 32 statt 54
    # Durchgangslöcher, und die App rechnete eine andere Kapsel als die im
    # Projekt (das Raster allein: 2D-Auslöschung bei 1 kHz −11.6 statt
    # −13.4 dB).
    # a) Das K67-Projekt kommt vollständig an — seit Gegenprobe 73 als
    #    exaktes Raster: 27 Lochpaare je Lochtyp mit Winkel (zuvor 13
    #    Kreise nach der Lagekonvention), die Winkel unverändert.
    # b) Eine Liste über der Höchstzahl wird gekürzt UND als gekürzt
    #    gemeldet (der Lader zeigt das als Warnung).
    A = app
    data, p = _projekt(A, os.path.join(_WURZEL, "examples",
                                       "k67_experimentell_bohrbild.json"))
    assert len(p["th_rings"]) == len(data["params"]["th_rings"]) == 27
    assert p["th_rings"] == data["params"]["th_rings"], "Winkel verloren"
    assert _projekt_gekuerzt(A, data) == []
    n_th = sum(z[0] for z in p["th_rings"])
    assert n_th == 54, f"54 Durchgangslöcher erwartet ({n_th})"
    zu_viel = json.loads(json.dumps(data))
    zu_viel["params"]["bh_rings"] = [[2, 10.0]] * (A.MAX_RINGS + 4)
    assert _projekt_gekuerzt(A, zu_viel) == [("bh_rings", A.MAX_RINGS + 4)]
    print(f"Lochkreise beim Laden: 27 Lochpaare mit Winkel vollständig "
          f"({n_th} Durchgangslöcher), {A.MAX_RINGS + 4} Kreise gekürzt und "
          f"gemeldet  OK")


def _projekt_gekuerzt(A, data):
    return A._projekt_params(data)[2]


def test_gp63e_ausgeblendete_felder_behalten_wert(app):
    """Gegenprobe 63 e: ausgeblendete Felder behalten ihren Wert."""
    # Streamlit räumt den Zustand eines Widgets ab, das in einem Lauf nicht
    # gezeichnet wird, und die App setzte dann die Voreinstellung ein. Im
    # Projekt k67_experimentell_zwischenspalt45 (BEM, freier Kopf 0 mm)
    # brachte BEM → Kugel → BEM den Körper Ø 56 mm zurück: drei Felder,
    # Außenweg 24.5 statt 35.0 mm und 2D-Rechenzeit ~390 statt ~50 s.
    # Derselbe Weg trifft jedes bedingte Feld; geprüft wird er hier am
    # Klemmring, der nur bei der K67-Bauform zu sehen ist (die BEM-Läufe
    # mit Körper kosten Minuten). Bedient wird über die Widgets selbst —
    # direkt in den Session-State geschriebene Werte sind Sitzungswerte
    # und verdecken den Fehler.
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=600)
    # Schnell rechnen: 1D und Kugel statt der 3D-Voreinstellung mit BEM
    at.session_state["p_n_points"] = _NPTS
    at.session_state["p_squeeze_2d"] = False
    at.session_state["p_squeeze_3d"] = False
    at.session_state["p_axial_body"] = "Kugel (d_ext, montiert)"

    def lauf():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at.run()
        assert not at.exception, at.exception

    def feld(key):
        return [w for w in list(at.number_input) + list(at.radio)
                if w.key == key][0]
    lauf()
    assert at.session_state["p_architecture"] == app.K67_LABEL
    assert app.DEFAULTS["clamp_width_mm"] != 2.5
    feld("p_clamp_width_mm").set_value(2.5)
    lauf()
    feld("p_architecture").set_value("Single Backplate")
    lauf()
    assert not [w for w in at.number_input if w.key == "p_clamp_width_mm"]
    feld("p_architecture").set_value(app.K67_LABEL)
    lauf()
    wert = at.session_state["p_clamp_width_mm"]
    assert wert == 2.5, f"Klemmringbreite nach Aus-/Einblenden {wert}"
    assert app._halte_ausgeblendete.__module__ == app.__name__
    assert {"bem_body_dia_mm", "bem_body_gap_mm",
            "bem_body_len_mm"} <= set(app.DEFAULTS)
    print("Ausgeblendete Felder: Klemmringbreite 2.5 mm übersteht K67 → "
          "Single → K67 (dieselbe Haltung gilt für die BEM-Körpermaße)  OK")


def _voreinstellung_gueltig(A):
    """Gespeicherte Ergebnisse der Voreinstellung laden — oder sofort mit
    der Anleitung abbrechen (sonst rechnete die App über eine Stunde)."""
    import voreinstellung as V
    key = A._ergebnis_schluessel(A.DEFAULTS)
    erg, zus, meta = V.laden(key, V.fingerabdruck(
        key, A._VOREINSTELLUNG_FUNKTIONEN))
    if erg is None:
        pytest.fail(f"Gespeicherte Voreinstellung: {meta} — neu erzeugen "
                    f"mit `python voreinstellung.py` (~1,5 h)")
    return key, erg, zus, meta


@pytest.mark.slow
@pytest.mark.feld3d
@pytest.mark.bem
def test_gp63f_gespeicherte_voreinstellung(app):
    """Gegenprobe 63 f: die gespeicherte Voreinstellung ist das Modell."""
    # Die Voreinstellung (K67, 45 µm, feines 3D) kostet ~1,5 h; die App
    # setzt deshalb gespeicherte Ergebnisse ein (voreinstellung.py).
    # a) Ihr Fingerabdruck passt zum heutigen Modell, Parametersatz und
    #    den Rechenfunktionen der App.
    # b) Stichprobe: ein Frequenzpunkt nahe 1 kHz, das 1-kHz-Richtdiagramm
    #    und die Empfindlichkeit, frisch gerechnet, gleichen den
    #    gespeicherten Werten (bis auf die Rundung der Faktorisierung).
    # c) Der Fingerabdruck sieht Code, aber keine Kommentare/Docstrings,
    #    und jeder andere Parametersatz bekommt nichts Gespeichertes.
    # d) Die App zeigt die Voreinstellung ohne Rechnung und ohne die Kapsel
    #    zu bauen (Kennwerte aus der Datei), mit Hinweis.
    # e) Passt die Datei nicht, startet die Stundenrechnung NICHT von
    #    selbst (in der Streamlit Cloud drosselte das die App): Grund und
    #    Knopf, keine Diagramme. Der Schlüssel ist typnormiert.
    import time
    import voreinstellung as V
    A = app
    key, erg, zus, meta = _voreinstellung_gueltig(A)
    assert set(zus) == {"de", "en"}
    # b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kapsel = A.build_capsule(dict(A.DEFAULTS))
        f = erg["fr"]["frequency_hz"]
        i = int(np.argmin(np.abs(np.log(f / 1000.0))))
        r = kapsel.angle_responses(f[i:i + 1])
        di = kapsel.directivity(frequencies_hz=(1000.0,))
        s1k = float(abs(kapsel.transfer_function(np.array([1000.0]))[0]))
    # komplex, wo gespeichert; der 180°-Pegel und S(1 kHz) als Betrag.
    # Toleranz 1e-6 (mit Zustimmung von 1e-9 gelockert): das 3D-System ist
    # mit ~1e8 konditioniert, und die Rundung der BLAS hängt von der
    # Thread-Zahl ab — gemessen 0 (4 Threads, wie erzeugt), 1,3e-8 (1) und
    # 2,2e-8 (2). Im parallelen Gesamtlauf (ein Thread je Worker) scheiterte
    # die Probe an 1e-9; 1e-6 sind 1e-5 dB.
    paare = [("H 0°", r["H"][0.0][0], erg["fr"]["sensitivity_v_pa"][i]),
             ("D_r", r["D_r"][0], erg["aux"]["D_r"][i]),
             ("|H 180°|", abs(r["H"][180.0][0]),
              abs(erg["fr"]["sensitivity_v_pa"][i])
              * 10 ** (erg["aux"]["level_180_db"][i] / 20.0)),
             ("S 1 kHz", s1k, erg["sens_1k"])]
    for name, neu, alt in paare:
        assert abs(neu - alt) <= 1e-6 * abs(alt), (name, neu, alt)
    assert np.allclose(di["patterns"][1000.0]["db"],
                       erg["di"]["patterns"][1000.0]["db"], atol=1e-5)
    assert A._kennwerte(kapsel) == meta["kennwerte"], "Kennwerte"
    # Der Stellvertreter der gespeicherten Kapsel trägt rand_abweichung
    # nicht; die App nimmt dann 0 an (keine Warnung, Gegenprobe 63 i). Das
    # gilt nur, solange die Voreinstellung wirklich unter der Grenze liegt.
    assert abs(kapsel.rand_abweichung) < 0.1 * kapsel._RAND_GRENZE
    # c)
    code = "def f(x):\n    return x + 1\n"
    assert V._baum_hash(code) == V._baum_hash(
        'def f(x):\n    """Doku."""\n    # Kommentar\n    return x + 1\n')
    assert V._baum_hash(code) != V._baum_hash(
        "def f(x):\n    return x + 2\n")
    # ... und ist von der Python-Version unabhängig: ast.dump gab unter
    # 3.11, 3.12 und 3.13 drei verschiedene Werte, Streamlit Cloud hielt
    # die Datei deshalb für veraltet. Verglichen wird mit jedem hier
    # installierten Interpreter (der Fingerabdruck braucht nur die
    # Standardbibliothek).
    import shutil
    import subprocess
    import sys
    with open(V._MODELL, encoding="utf-8") as fh:
        h_hier = V._baum_hash(fh.read()).hex()
    versionen = []
    for v in ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14"):
        exe = shutil.which(f"python{v}")
        if exe is None:
            continue
        prog = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import voreinstellung as V; "
                "print(V._baum_hash(open(V._MODELL, encoding='utf-8')"
                ".read()).hex())")
        aus = subprocess.run([exe, "-I", "-c", prog, _WURZEL],
                             capture_output=True, text=True, timeout=120)
        assert aus.returncode == 0, (v, aus.stderr[-500:])
        assert aus.stdout.strip() == h_hier, f"Python {v}: anderer Hash"
        versionen.append(v)
    anders = A._ergebnis_schluessel(dict(A.DEFAULTS, bias_v=59.0))
    assert A._voreinstellung_einsetzen(anders) is None
    assert V.laden(anders, V.fingerabdruck(
        anders, A._VOREINSTELLUNG_FUNKTIONEN))[2] == "parameter"
    # d)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=120)
    at.session_state["ui_lang"] = "de"
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    dauer = time.time() - t0
    assert not at.exception, [e.value for e in at.exception]
    hinweis = [c.value for c in at.caption
               if c.value.startswith("Voreinstellung: gespeicherte")]
    assert hinweis, "Hinweis auf die gespeicherten Ergebnisse fehlt"
    assert not at.info or all("Voreinstellung" not in e.value
                              for e in at.info), [e.value for e in at.info]
    assert dauer < 60.0, f"App-Start {dauer:.0f} s — rechnet sie doch?"
    assert ("_capsule_cache" not in at.session_state
            or not at.session_state["_capsule_cache"]), "Kapsel doch gebaut"
    # e)
    assert A._ergebnis_schluessel(dict(A.DEFAULTS, n_points=400.0,
                                       bias_v=60)) == key
    echt = V.laden
    try:
        V.laden = lambda *a, **k: (None, None, "veraltet")
        at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                               default_timeout=120)
        at.session_state["ui_lang"] = "de"
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at.run()
        dauer_e = time.time() - t0
    finally:
        V.laden = echt
    assert not at.exception, [e.value for e in at.exception]
    assert any("passen nicht mehr zum Modellcode" in e.value
               for e in at.info), [e.value for e in at.info]
    assert [b for b in at.button if "jetzt rechnen" in b.label]
    assert not at.get("plotly_chart") and dauer_e < 60.0, dauer_e
    print(f"Gespeicherte Voreinstellung (erzeugt {meta['erzeugt']}, "
          f"{meta['rechenzeit_s'] / 60:.0f} min): Fingerabdruck gültig "
          f"und gleich unter Python {'/'.join(versionen) or sys.version[:4]}, "
          f"Stichprobe {f[i]:.0f} Hz, 1-kHz-Richtdiagramm und Kennwerte "
          f"gleich, App-Start {dauer:.1f} s ohne Kapselbau; veraltet: "
          f"Knopf statt Rechnung ({dauer_e:.1f} s)  OK")


def _at(app, **zustand):
    """AppTest mit Sitzungswerten; ohne weitere Angaben schnell: 1D, Kugel."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=600)
    werte = {"ui_lang": "de", "p_n_points": _NPTS, "p_squeeze_2d": False,
             "p_squeeze_3d": False,
             "p_axial_body": "Kugel (d_ext, montiert)"}
    werte.update(zustand)
    for k, v in werte.items():
        at.session_state[k] = v
    return at


def _lauf(at):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _widget(at, key):
    for art in ("number_input", "radio", "selectbox", "button", "toggle",
                "text_area"):
        for w in at.get(art):
            if w.key == key:
                return w
    raise KeyError(key)


def test_gp63g_ausgeblendetes_wirkt_nicht_und_null_reset(app):
    """Gegenprobe 63 g: was die Oberfläche ausblendet, wirkt nicht, und der
    Null-Reset bleibt rechenfähig."""
    # a) Sackgasse: seit ausgeblendete Felder ihren Wert behalten
    #    (Gegenprobe 63 e), blieb ein bei 'Single' gesetzter Randspalt nach
    #    dem Wechsel zur K67 stehen. Das Modell lehnte ihn ab, das Feld war
    #    nicht mehr zu sehen. Jetzt rechnet die K67 ohne ihn, der Wert
    #    bleibt für den Rückweg erhalten und steht nicht im Projekt.
    # b) Null-Reset: der Null-Zustand (100 Hz Resonanz auf 3 mm) scheiterte
    #    seit der Randschicht-Prüfung (Gegenprobe 62) für JEDE Bauform, mit
    #    der 3D-Voreinstellung zusätzlich an „3D erfordert Löcher“. Geprüft
    #    aus der echten Voreinstellung und für alle drei Bauformen.
    A = app
    at = _lauf(_at(A))
    _widget(at, "p_architecture").set_value("Single Backplate")
    _lauf(at)
    _widget(at, "p_ring_vent_um").set_value(100.0)
    _lauf(at)
    _widget(at, "p_architecture").set_value(A.K67_LABEL)
    _lauf(at)
    assert not at.error, [e.value for e in at.error]
    assert len(at.get("plotly_chart")) >= 4
    assert not [w for w in at.number_input if w.key == "p_ring_vent_um"]
    assert at.session_state["p_ring_vent_um"] == 100.0
    p_k67 = A._wirksam({k: at.session_state["p_" + k] for k in A.DEFAULTS
                        if k not in A._RING_PREFIX})
    assert p_k67["ring_vent_um"] == 0.0 and p_k67["ring_vent_len_mm"] == 0.0
    _widget(at, "p_architecture").set_value("Single Backplate")
    _lauf(at)
    assert _widget(at, "p_ring_vent_um").value == 100.0
    assert A._wirksam(dict(A.DEFAULTS, architecture="Single Backplate",
                           ring_vent_um=100.0))["ring_vent_um"] == 100.0
    # b) aus der Voreinstellung (feines 3D, BEM; gespeichert, also schnell)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=600)
    at.session_state["ui_lang"] = "de"
    _lauf(at)
    _widget(at, "btn_reset_ask").click()
    _lauf(at)
    _widget(at, "btn_reset_yes").click()
    _lauf(at)
    assert not at.error, [e.value for e in at.error]
    assert len(at.get("plotly_chart")) >= 4, "Null-Zustand rechnet nicht"
    assert at.session_state["p_squeeze_3d"] is False
    assert at.session_state["p_axial_body"] == "Kugel (d_ext, montiert)"
    # ... und für jede Bauform, mit und ohne vorgegebene Resonanz
    gerechnet = 0
    for arch in A.ARCH_LABELS:
        for use in (True, False):
            at = _lauf(_at(A, p_architecture=arch, p_use_f_res=use,
                           p_squeeze_2d=True))
            _widget(at, "btn_reset_ask").click()
            _lauf(at)
            _widget(at, "btn_reset_yes").click()
            _lauf(at)
            assert not at.error, (arch, use, [e.value for e in at.error])
            assert len(at.get("plotly_chart")) >= 4, (arch, use)
            gerechnet += 1
    print(f"Ausgeblendetes wirkt nicht: K67 rechnet mit verstecktem "
          f"Randspalt 100 µm (bleibt für den Rückweg erhalten); Null-Reset "
          f"rechnet aus der Voreinstellung und für {gerechnet} "
          f"Bauform-Fälle  OK")


def test_gp63h_projektlader_notizen_beispiele(app):
    """Gegenprobe 63 h: der Projekt-Lader übergeht nichts still; Notizen,
    Beispielauswahl und der Rückweg zur Voreinstellung."""
    # a) Ein vertippter Schlüssel (air_gap statt air_gap_um) lud still als
    #    Fehlwert 65 µm: jetzt als unbekannter Eintrag gemeldet. Eine Datei
    #    ohne Parameter überschrieb alles und meldete Erfolg: abgelehnt.
    #    "false" als Text wurde True, NaN ging durch: abgelehnt. Eine
    #    neuere Formatversion wird gemeldet. Die Meldung für ungültige
    #    Werte scheiterte selbst an einem TypeError (tr(key=…)).
    # b) Notizen (_kommentar) überstehen Laden und Speichern.
    # c) In der App: Beispiel wählen und laden, zurück zur Voreinstellung.
    import copy
    import streamlit as st
    A = app
    pfad = os.path.join(_WURZEL, "examples", "u87_k67_projekt.json")
    with open(pfad, encoding="utf-8") as fh:
        basis = json.load(fh)
    for name in A._beispiele():
        with open(os.path.join(_WURZEL, "examples", name),
                  encoding="utf-8") as fh:
            assert A._projekt_params(json.load(fh))[3] == [], name
    d = copy.deepcopy(basis)
    d["params"]["air_gap"] = d["params"].pop("air_gap_um")
    p, n, _, unbekannt = A._projekt_params(d)
    assert unbekannt == ["air_gap"] and p["air_gap_um"] == 65.0
    art, msg = A._projekt_anwenden(d)
    assert art == "warning" and "air_gap" in msg, (art, msg)
    d = copy.deepcopy(basis)
    d["version"] = 7
    art, msg = A._projekt_anwenden(d)
    assert art == "warning" and "7" in msg, (art, msg)
    art, msg = A._projekt_anwenden(copy.deepcopy(basis))
    assert art == "success", (art, msg)
    abgelehnt = 0
    for aendern in (lambda q: q.__setitem__("params", {}),
                    lambda q: q.pop("params"),
                    lambda q: q["params"].__setitem__("squeeze_2d", "false"),
                    lambda q: q["params"].__setitem__("air_gap_um",
                                                      float("nan")),
                    lambda q: q["params"].__setitem__("material", "Holz"),
                    lambda q: q.__setitem__("format", "anderes")):
        d = copy.deepcopy(basis)
        aendern(d)
        with pytest.raises(ValueError):
            A._projekt_params(d)
        abgelehnt += 1
    with pytest.raises(ValueError, match="material"):
        A._coerce("material", "Holz")
    # b)
    notiz = A._notizen_aus(basis)
    assert notiz.startswith(basis["_kommentar"][0]) and "\n" in notiz
    zurueck = json.loads(A._projekt_json(A._projekt_params(basis)[0], notiz))
    assert zurueck["_kommentar"] == basis["_kommentar"]
    assert A._projekt_params(zurueck)[0] == A._projekt_params(basis)[0]
    assert "_kommentar" not in json.loads(A._projekt_json(
        A._projekt_params(basis)[0], "  "))
    st.session_state.clear()
    # c) in der App (Start mit der gespeicherten Voreinstellung)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                           default_timeout=600)
    at.session_state["ui_lang"] = "de"
    _lauf(at)
    assert _widget(at, "beispiel_wahl").options == A._beispiele()
    _widget(at, "beispiel_wahl").set_value("u87_k67_projekt.json")
    _lauf(at)
    _widget(at, "btn_example").click()
    _lauf(at)
    soll = A._projekt_params(basis)[0]
    ist = _zustand(A, at)
    assert ist == soll, {k: (soll[k], ist[k]) for k in soll
                         if ist[k] != soll[k]}
    assert at.session_state["projekt_notizen"] == notiz
    assert any("Projekt geladen" in e.value for e in at.success)
    _widget(at, "btn_default").click()
    _lauf(at)
    ist = _zustand(A, at)
    assert ist == dict(A.DEFAULTS), {k: (A.DEFAULTS[k], ist[k])
                                     for k in ist if ist[k] != A.DEFAULTS[k]}
    assert at.session_state["projekt_notizen"] == ""
    assert [c for c in at.caption
            if c.value.startswith("Voreinstellung: gespeicherte")]
    print(f"Projekt-Lader: Tippfehler und neuere Version gemeldet, "
          f"{abgelehnt} defekte Dateien abgelehnt, Notizen überstehen die "
          f"Rundreise; {len(A._beispiele())} Beispiele wählbar, Beispiel "
          f"geladen und zurück zur gespeicherten Voreinstellung  OK")


def test_gp63i_warnung_darstellung_threads(app):
    """Gegenprobe 63 i: die Warnung des Modells erreicht die Oberfläche;
    Diagramme und Thread-Zahl."""
    # a) „Biegesteife Folie“ (Gegenprobe 62) stand nur im Konsolen-
    #    protokoll: 20 µm Titan auf 27,2 mm verfehlt die Platte um 1,5 %.
    # b) Die Legende des Phasenschieber-Diagramms überdeckte den Titel,
    #    die Winkelbeschriftungen des Polardiagramms waren abgeschnitten.
    # c) Rechen-Threads nach der CPU-Quote: darf der Prozess weniger
    #    rechnen, als er Kerne sieht (Cloud), bremsen sich die BLAS-Threads
    #    aus — eine Frequenz im feinen 3D bei 1 von 4 Kernen 47,8 s mit vier
    #    Threads, 15,8 s mit einem. Ohne Quote bleibt der Standard (dort
    #    schadet die Drossel gleichzeitigen Sitzungen: 25 gegen 11 s).
    A = app
    at = _lauf(_at(A, p_material="Titan", p_mem_thickness_um=20.0,
                   p_bias_v=20.0))
    warn = [w.value for w in at.warning if "Biegesteife Folie" in w.value]
    assert warn and "-1.5 %" in warn[0], [w.value for w in at.warning]
    at = _lauf(_at(A))
    assert not [w for w in at.warning if "Biegesteife" in w.value]
    # b)
    _, erg, _, _ = _voreinstellung_gueltig(A)
    dr = A.dr_figure(erg["fr"], erg["aux"])
    assert dr.layout.legend.y < 0 and dr.layout.legend.yanchor == "top"
    pol = A.polar_figure(erg["di"])
    assert min(pol.layout.margin.l, pol.layout.margin.r) >= 40
    # c)
    import tempfile

    def kerne(dateien, cgroup="0::/\n"):
        with tempfile.TemporaryDirectory() as tmp:
            for rel, inhalt in dateien.items():
                os.makedirs(os.path.dirname(os.path.join(tmp, rel)),
                            exist_ok=True)
                with open(os.path.join(tmp, rel), "w") as fh:
                    fh.write(inhalt)
            with open(os.path.join(tmp, "proc"), "w") as fh:
                fh.write(cgroup)
            return A._erlaubte_kerne(tmp, os.path.join(tmp, "proc"))
    assert kerne({}) is None
    assert kerne({"cpu.max": "max 100000\n"}) is None
    assert kerne({"cpu.max": "200000 100000\n"}) == 2.0
    assert kerne({"cpu.max": "unsinn"}) is None
    v1 = "cpu/cpu.cfs_quota_us", "cpu/cpu.cfs_period_us"
    assert kerne({v1[0]: "-1", v1[1]: "100000"}) is None
    assert kerne({v1[0]: "50000", v1[1]: "100000"}) == 0.5
    # Quote in der eigenen Untergruppe und engste Quote der Kette
    assert kerne({"cpu/kind/cpu.cfs_quota_us": "100000",
                  "cpu/kind/cpu.cfs_period_us": "100000",
                  v1[0]: "300000", v1[1]: "100000"},
                 "4:cpu,cpuacct:/kind\n") == 1.0
    assert kerne({"a/b/cpu.max": "max 100000", "a/cpu.max": "150000 100000"},
                 "0::/a/b\n") == 1.5
    assert A._thread_vorgabe(None, 8) is None        # keine Quote
    assert A._thread_vorgabe(8.0, 8) is None         # Quote = alle Kerne
    assert A._thread_vorgabe(2.0, 8) == 2
    assert A._thread_vorgabe(0.078, 8) == 1          # Cloud-Minimum
    gemessen = "ohne Quote Bibliotheksstandard"
    if os.environ.get("OMP_NUM_THREADS"):
        soll = int(os.environ["OMP_NUM_THREADS"])
        try:
            from threadpoolctl import threadpool_info
            pools = [i["num_threads"] for i in threadpool_info()]
            assert pools and all(n == soll for n in pools), pools
            gemessen = f"{len(pools)} BLAS-Bibliotheken auf {soll}"
        except ImportError:
            gemessen = f"Umgebung auf {soll} (threadpoolctl fehlt)"
    print(f"Warnung „Biegesteife Folie“ in der Oberfläche ({warn[0][:40]}…),"
          f" Legende unter dem Phasenschieber-Diagramm, Polardiagramm mit "
          f"Rand; Threads: {gemessen}  OK")


@pytest.mark.slow
def test_gp75b_3d_parallel_und_gespeicherte_loesungen(app):
    """Gegenprobe 75 b: 3D-Frequenzen parallel in Arbeiterprozessen, und
    die gespeicherte Voreinstellung bringt ihre Lösungen je Frequenz mit."""
    # a) PARALLEL: _vorrechnen_3d verteilt die fehlenden Frequenzen des
    #    Frequenzplans (Raster, Richtfrequenzen, 1 kHz) auf Prozesse;
    #    danach rechnet compute_results im Hauptprozess keine einzige
    #    Zerlegung mehr, und das Ergebnis trifft die serielle Rechnung (die
    #    Arbeiter rechnen mit einem BLAS-Faden: ~1e-8, s. Gegenprobe 63 f).
    # b) WIE VIELE: so viele Prozesse, wie Kerne UND freier Speicher
    #    erlauben (je Prozess 2,5 GB fein, 1,2 GB grob, 0,5 GB Reserve); in
    #    der Streamlit Cloud (2 Kerne, 2,7 GB) einer — dort nichts parallel.
    # c) Der freie Speicher kommt aus MemAvailable und den cgroup-Grenzen
    #    (v1 und v2, eigene Gruppe und alle darüber; es gilt die engste).
    # d) GESPEICHERT: die Voreinstellung trägt ihre 3D-Lösungen; ist die
    #    Kapsel physikalisch die Voreinstellung (nur Punktzahl oder
    #    Richtfrequenzen anders), setzt die App sie ein: eine weitere
    #    Richtfrequenz kostet genau eine Zerlegung statt ~85 min, eine
    #    vorhandene keine — und trifft das gespeicherte Richtdiagramm.
    import tempfile
    import voreinstellung as V
    A = app
    _, p = _projekt(A, os.path.join(_WURZEL, "examples",
                                    "debenham_stereo_condenser.json"))
    p.update(squeeze_3d=True, grid_3d_fine=False, n_points=12,
             dir_freqs=[1000, 4000])
    key = A._ergebnis_schluessel(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_s = A.build_capsule(p)
        r_s = A.compute_results(key, c_s)
        A._session_store("_results_cache").clear()
        c_p = A.build_capsule(p)
        n_par, w = A._vorrechnen_3d(p, c_p, prozesse=2)
        plan = A._frequenzplan(p)
        assert (n_par, w) == (plan.size, 2), (n_par, w, plan.size)
        assert getattr(c_p, "_lu_3d_count", 0) == 0
        r_p = A.compute_results(key, c_p)
    assert getattr(c_p, "_lu_3d_count", 0) == 0, \
        "nach der Vorrechnung darf der Hauptprozess nicht mehr zerlegen"
    abw = float(np.max(np.abs(r_p["fr"]["sensitivity_v_pa"]
                              / r_s["fr"]["sensitivity_v_pa"] - 1.0)))
    for fd in (1000.0, 4000.0):
        abw = max(abw, float(np.max(np.abs(
            r_p["di"]["patterns"][fd]["linear"]
            - r_s["di"]["patterns"][fd]["linear"]))))
    assert abw < 1e-6, f"parallel gegen seriell {abw:.1e}"
    # a2) über Sitzungen hinweg: was eine Kapsel gelöst hat, findet eine
    #     neu gebaute (Neuladen der Seite) im Serverprozess wieder
    A._prozess_loesungen(key, c_s, merken=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_n = A.build_capsule(p)
    assert A._prozess_loesungen(key, c_n) == plan.size
    assert c_n.loesungen_3d() == c_s.loesungen_3d()
    # b)
    fein = type("K", (), {"grid_3d": "fine"})()
    grob = type("K", (), {"grid_3d": "coarse"})()
    assert A._prozesse_3d(fein, kerne=2.0, frei=2.7e9 - 0.6e9) == 1
    assert A._prozesse_3d(fein, kerne=4.0, frei=15e9) == 4
    assert A._prozesse_3d(fein, kerne=8.0, frei=6e9) == 2
    assert A._prozesse_3d(grob, kerne=8.0, frei=6e9) == 4
    assert A._prozesse_3d(fein, kerne=4.0, frei=0.0) == 1
    # c)
    def frei(dateien, cgroup="0::/\n", mem=None):
        with tempfile.TemporaryDirectory() as tmp:
            for rel, inhalt in dateien.items():
                os.makedirs(os.path.dirname(os.path.join(tmp, rel)),
                            exist_ok=True)
                with open(os.path.join(tmp, rel), "w") as fh:
                    fh.write(inhalt)
            with open(os.path.join(tmp, "proc"), "w") as fh:
                fh.write(cgroup)
            with open(os.path.join(tmp, "meminfo"), "w") as fh:
                fh.write(mem or "MemTotal: 16000000 kB\n"
                         "MemAvailable: 8000000 kB\n")
            return A._freier_speicher(tmp, os.path.join(tmp, "proc"),
                                      os.path.join(tmp, "meminfo"))
    assert frei({}) == 8000000 * 1024
    assert frei({"memory.max": "max\n", "memory.current": "1"}) \
        == 8000000 * 1024
    assert frei({"memory.max": "3000000000\n",
                 "memory.current": "1000000000\n"}) == 2_000_000_000
    assert frei({"a/memory.max": "2000000000", "a/memory.current": "500",
                 "memory.max": "max", "memory.current": "0"},
                "0::/a\n") == 2_000_000_000 - 500
    v1 = "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"
    assert frei({v1[0]: str(1 << 62), v1[1]: "5"}) == 8000000 * 1024
    assert frei({v1[0]: "2700000000", v1[1]: "700000000"},
                "5:memory:/\n") == 2_000_000_000
    # d) die gespeicherten Lösungen der Voreinstellung
    physik = A._physik_schluessel(A._ergebnis_schluessel(A.DEFAULTS))
    loes = V.laden_loesungen(physik,
                             V.fingerabdruck(physik, A._BAU_FUNKTIONEN))
    if loes is None:
        pytest.fail("Gespeicherte Voreinstellung ohne passende 3D-Lösungen "
                    "— neu erzeugen mit `python voreinstellung.py`")
    plan = A._frequenzplan(A.DEFAULTS)
    assert set(loes) == {float(om) for om in 2.0 * np.pi * plan}, \
        "die Datei muss den ganzen Frequenzplan tragen"
    q = dict(A.DEFAULTS, dir_freqs=sorted(A.DEFAULTS["dir_freqs"]) + [3150])
    q_key = A._ergebnis_schluessel(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_v = A.build_capsule(q)
    assert A._voreinstellung_loesungen(q_key, c_v) == plan.size
    assert A._voreinstellung_loesungen(q_key, c_v) == 0, "nur einmal"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_alt = c_v.directivity(frequencies_hz=(2000.0,))
        assert getattr(c_v, "_lu_3d_count", 0) == 0
        c_v.directivity(frequencies_hz=(3150.0,))
    assert c_v._lu_3d_count == 1, "die neue Richtfrequenz: eine Zerlegung"
    erg, _, _ = V.laden(A._ergebnis_schluessel(A.DEFAULTS),
                        V.fingerabdruck(A._ergebnis_schluessel(A.DEFAULTS),
                                        A._VOREINSTELLUNG_FUNKTIONEN))
    assert erg is not None
    gesp = erg["di"]["patterns"][2000.0]["linear"]
    d2k = float(np.max(np.abs(d_alt["patterns"][2000.0]["linear"] - gesp)))
    assert d2k < 1e-9, f"Richtdiagramm 2 kHz gegen gespeichert {d2k:.1e}"
    print(f"3D parallel: {n_par} Frequenzen in {w} Prozessen, Hauptprozess "
          f"ohne Zerlegung, gegen seriell {abw:.0e}; Prozesszahl nach Kernen "
          f"und Speicher (Cloud: 1); freier Speicher aus cgroup v1/v2; "
          f"Voreinstellung: {len(loes)} 3D-Lösungen eingesetzt, weitere "
          f"Richtfrequenz = 1 Zerlegung, 2 kHz trifft gespeichert ({d2k:.0e})"
          f"  OK")


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


def test_gp66c_modellmeldungen():
    """Gegenprobe 66 c: Meldungen des Modells in beiden Sprachen."""
    # Bis hier waren die Fehlermeldungen des Modells nur deutsch und
    # erschienen so auch in der englischen Oberfläche. Jetzt wirft jede
    # Parameterprüfung einen ParameterFehler (ein ValueError mit der
    # deutschen Meldung wie bisher), dessen Text die Oberfläche aus
    # _MELDUNGEN in ihrer Sprache bildet.
    # a) Jeder Eintrag hat Deutsch und Englisch, nicht leer, mit denselben
    #    Platzhaltern, und beide lassen sich mit denselben Werten füllen.
    # b) Jeder ParameterFehler(...) im Modell nennt einen vorhandenen
    #    Schlüssel als Konstante und genau dessen Platzhalter als Werte;
    #    kein Eintrag ist unbenutzt.
    # c) In den Methoden, die Parameter prüfen, steht kein einfacher
    #    ValueError mehr (der käme nur deutsch an). Einfache ValueError
    #    bleiben für falsche Aufrufe aus eigenem Code (interne Löser,
    #    Rauschspektrum, Druckgang bei offener Rückseite).
    import ast
    import string
    import microphone_capsule as M
    fmt = string.Formatter()

    def felder(txt):
        return {f for _, f, _, _ in fmt.parse(txt) if f is not None}

    fehler = []
    for k, v in M._MELDUNGEN.items():
        if set(v) != {"de", "en"} or not all(
                isinstance(x, str) and x.strip() for x in v.values()):
            fehler.append(f"{k}: Sprachen/leer")
            continue
        if felder(v["de"]) != felder(v["en"]):
            fehler.append(f"{k}: Platzhalter {felder(v['de'])} / "
                          f"{felder(v['en'])}")
            continue
        for lang in ("de", "en"):
            try:
                v[lang].format(**{f: 1.5 for f in felder(v[lang])})
            except Exception as exc:
                fehler.append(f"{k}.{lang}: {exc}")
    assert not fehler, fehler
    src = open(os.path.join(_WURZEL, "microphone_capsule.py"),
               encoding="utf-8").read()
    pruefend = {"__init__", "_normalize_rings", "_derive_parameters",
                "_wirk_membran", "_ring_eigen", "_bem_front_modes",
                "_build_3d_geometry"}
    benutzt, einfach = set(), []

    def walk(node, stack):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.ClassDef)):
                walk(ch, stack + [ch.name])
                continue
            if isinstance(ch, ast.Call) and getattr(
                    ch.func, "id", "") == "ParameterFehler":
                a0 = ch.args[0] if ch.args else None
                if not (isinstance(a0, ast.Constant)
                        and a0.value in M._MELDUNGEN):
                    fehler.append(f"Zeile {ch.lineno}: Schlüssel "
                                  f"{ast.get_source_segment(src, ch)}")
                else:
                    benutzt.add(a0.value)
                    soll = felder(M._MELDUNGEN[a0.value]["de"])
                    ist = {kw.arg for kw in ch.keywords}
                    if soll != ist:
                        fehler.append(f"Zeile {ch.lineno} {a0.value}: "
                                      f"Werte {ist}, Platzhalter {soll}")
            if (isinstance(ch, ast.Raise) and isinstance(ch.exc, ast.Call)
                    and getattr(ch.exc.func, "id", "") == "ValueError"
                    and stack and stack[-1] in pruefend):
                einfach.append(f"{stack[-1]} Zeile {ch.lineno}")
            walk(ch, stack)
    walk(ast.parse(src), [])
    assert not fehler, fehler
    tot = sorted(set(M._MELDUNGEN) - benutzt)
    assert not tot, f"unbenutzte Meldungen: {tot}"
    assert not einfach, f"einsprachige Parameterfehler: {einfach}"
    # die Meldung selbst: deutsch als str(exc), englisch über text()
    with pytest.raises(ValueError) as ei:
        MicrophoneCapsule(architecture="quer")
    assert isinstance(ei.value, M.ParameterFehler)
    assert str(ei.value) == M._MELDUNGEN["architektur"]["de"]
    assert ei.value.text("en") == M._MELDUNGEN["architektur"]["en"]
    print(f"Modellmeldungen: {len(M._MELDUNGEN)} zweisprachig mit "
          f"gleichen Platzhaltern, alle benutzt; jede Parameterprüfung "
          f"wirft ParameterFehler  OK")


def test_gp66d_modellmeldung_in_der_oberflaeche(app):
    """Gegenprobe 66 d: die App zeigt die Modellmeldung in ihrer
    Sprache — die Voreinstellung (K67) mit 400 V kollabiert (Pull-in)."""
    from streamlit.testing.v1 import AppTest
    import translations as T
    import microphone_capsule as M
    gesehen = {}
    for lang in ("en", "de"):
        at = AppTest.from_file(os.path.join(_WURZEL, "app.py"),
                               default_timeout=120)
        at.session_state["ui_lang"] = lang
        for key, val in app.DEFAULTS.items():
            if key not in app._RING_PREFIX:
                at.session_state["p_" + key] = val
        at.session_state["p_bias_v"] = 400.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at.run()
        assert not at.exception, [e.value for e in at.exception]
        fehl = [e.value for e in at.error]
        # (das führende ⚠️ macht Streamlit zum Symbol des Kastens)
        kopf = T.TR["err_params"][lang].split("{")[0].replace(
            "⚠️", "").strip()
        anfang = M._MELDUNGEN["pullin"][lang].split("{")[0]
        assert any(kopf in t and anfang in t for t in fehl), \
            f"{lang}: Pull-in-Meldung in der Oberflächensprache erwartet " \
            f"({fehl})"
        gesehen[lang] = fehl[0][:60]
    print(f"Modellmeldung in der Oberfläche: en „{gesehen['en']}…“, "
          f"de „{gesehen['de']}…“  OK")


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
    # Die Voreinstellung läuft UNVERÄNDERT (feines 3D, aus den gespeicherten
    # Ergebnissen, Gegenprobe 63 f); passen die nicht mehr zum Modell,
    # gleich abbrechen statt zweimal über eine Stunde zu rechnen.
    _voreinstellung_gueltig(A)
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
                    _ringe_setzen(A, at, key, val)
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
