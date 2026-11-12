# -*- coding: utf-8 -*-
"""Gespeicherte Ergebnisse der App-Voreinstellung (Gegenprobe 63 f).

Die Voreinstellung der App ist die experimentelle K67 mit 45 µm
Zwischenspalt im FEINEN 3D-Feldmodell: ~13 s je Frequenzpunkt, beim Start
also weit über eine Stunde. Ihre Ergebnisse (Frequenzgang, Richt-
diagramme, Diagnose, Summary in beiden Sprachen und die Kennwerte der
Kapsel für die Anzeige) liegen deshalb fertig in
``assets/voreinstellung.json`` — NUR für genau diesen Parametersatz. Jede
andere Einstellung rechnet die App wie bisher.

Gültig ist die Datei nur, solange ihr Fingerabdruck stimmt:

* der Ergebnisschlüssel der App (alle physikrelevanten Parameter),
* die Modellquelle ``microphone_capsule.py``,
* die Bau- und Rechenfunktionen der App (``build_capsule``,
  ``compute_results``, ``_kennwerte``),

jeweils als Syntaxbaum ohne Kommentare und Docstrings: eine reine
Kommentaränderung lässt die Datei gültig, jede Codeänderung macht sie
ungültig. Dann zeigt die App den Grund und rechnet die Voreinstellung
erst auf Knopfdruck — veraltete Ergebnisse erscheinen nie still, und eine
Stundenrechnung startet nie von selbst (in der Streamlit Cloud drosselte
das die App). Beim Start baut die App die Kapsel nicht: ihre Kennwerte
für die Anzeige stehen mit in der Datei.

Seit Format 3 (Gegenprobe 75) trägt die Datei auch die 3D-Lösungen je
Frequenz, mit eigenem Fingerabdruck aus dem Physik-Schlüssel (ohne
Punktzahl und Richtfrequenzen), der Modellquelle und den Baufunktionen.
Unterscheidet sich eine Einstellung nur in Punktzahl oder
Richtfrequenzen, setzt die App sie ein und rechnet nur die neuen
Frequenzen. Hier liegt auch der Prozess-Speicher, mit dem 3D-Lösungen
eine neue Sitzung (Neuladen der Seite) überstehen.

Neu erzeugen (nach jeder Modelländerung; seriell ~1,5 h, die
3D-Frequenzen rechnen parallel in Arbeiterprozessen, soweit Kerne und
Speicher reichen):

    python voreinstellung.py
"""
import ast
import datetime as _dt
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time

_WURZEL = os.path.dirname(os.path.abspath(__file__))
DATEI = os.path.join(_WURZEL, "assets", "voreinstellung.json")
_MODELL = os.path.join(_WURZEL, "microphone_capsule.py")
FORMAT = 3          # 2: mit den Kennwerten der Kapsel (Start ohne Bau)
                    # 3: mit den 3D-Lösungen je Frequenz (Gegenprobe 75)


def _ohne_docstrings(baum):
    """Docstrings aus Modul, Klassen und Funktionen entfernen (Kommentare
    kennt der Syntaxbaum ohnehin nicht)."""
    for knoten in ast.walk(baum):
        if isinstance(knoten, (ast.Module, ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            rumpf = knoten.body
            if (rumpf and isinstance(rumpf[0], ast.Expr)
                    and isinstance(rumpf[0].value, ast.Constant)
                    and isinstance(rumpf[0].value.value, str)):
                knoten.body = rumpf[1:] or [ast.Pass()]
    return baum


def _kanonisch(knoten):
    """Syntaxbaum als verschachtelte Liste, UNABHÄNGIG von der Python-
    Version: ast.dump schreibt ab 3.12 neue leere Felder (type_params) und
    lässt ab 3.13 leere/None-Felder weg — dieselbe Quelle hatte unter
    3.11, 3.12 und 3.13 drei verschiedene Fingerabdrücke (Streamlit Cloud
    rechnete deshalb neu). Hier zählen nur belegte Felder."""
    if isinstance(knoten, ast.AST):
        teile = [type(knoten).__name__]
        for feld in knoten._fields:
            wert = getattr(knoten, feld, None)
            if wert is None or (isinstance(wert, list) and not wert):
                continue
            teile.append([feld, _kanonisch(wert)])
        return teile
    if isinstance(knoten, list):
        return [_kanonisch(w) for w in knoten]
    return repr(knoten)


def _baum_hash(quelltext):
    baum = _ohne_docstrings(ast.parse(textwrap.dedent(quelltext)))
    return hashlib.sha256(repr(_kanonisch(baum)).encode()).digest()


_MODELL_HASH = {}


def _modell_hash():
    """Baum-Hash der Modellquelle, EINMAL je Prozess und Dateistand: das
    Zerlegen der ~9000 Zeilen kostet ~1 s (in der gedrosselten Cloud ein
    Vielfaches) und fiele sonst bei jeder Sitzung an."""
    st_ = os.stat(_MODELL)
    marke = (st_.st_mtime_ns, st_.st_size)
    if marke not in _MODELL_HASH:
        with open(_MODELL, encoding="utf-8") as fh:
            _MODELL_HASH.clear()
            _MODELL_HASH[marke] = _baum_hash(fh.read())
    return _MODELL_HASH[marke]


def fingerabdruck(schluessel, funktionen):
    """Fingerabdruck aus Ergebnisschlüssel, Modellquelle und den
    übergebenen App-Funktionen (s. Moduldoku)."""
    h = hashlib.sha256()
    h.update(schluessel.encode())
    h.update(_modell_hash())
    for fn in funktionen:
        h.update(_baum_hash(inspect.getsource(fn)))
    return h.hexdigest()


def _kodieren(x):
    """Ergebnisbaum -> JSON-taugliche Form, verlustfrei (Gleitkommazahlen
    über repr, komplexe Felder als Real-/Imaginärteil, Wörterbücher als
    Paarlisten, weil die Richtdiagramme Gleitkomma-Schlüssel haben)."""
    import numpy as np
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {"__re__": x.real.tolist(), "__im__": x.imag.tolist()}
        return {"__arr__": x.tolist()}
    if isinstance(x, dict):
        return {"__dict__": [[k, _kodieren(v)] for k, v in x.items()]}
    if isinstance(x, (list, tuple)):
        return [_kodieren(v) for v in x]
    if isinstance(x, (complex, np.complexfloating)):
        return {"__c__": [float(x.real), float(x.imag)]}
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _dekodieren(x):
    import numpy as np
    if isinstance(x, dict):
        if "__arr__" in x:
            return np.asarray(x["__arr__"], dtype=float)
        if "__re__" in x:
            return (np.asarray(x["__re__"], dtype=float)
                    + 1j * np.asarray(x["__im__"], dtype=float))
        if "__c__" in x:
            return complex(*x["__c__"])
        if "__dict__" in x:
            return {k: _dekodieren(v) for k, v in x["__dict__"]}
    if isinstance(x, list):
        return [_dekodieren(v) for v in x]
    return x


def speichern(schluessel, abdruck, ergebnis, zusammenfassungen,
              kennwerte, rechenzeit_s, datei=DATEI, erzeugt=None,
              physik=None, abdruck_3d=None, loesungen=None):
    daten = {
        "format": FORMAT,
        "erzeugt": erzeugt or _dt.datetime.now().isoformat(
            timespec="seconds"),
        "rechenzeit_s": round(float(rechenzeit_s), 1),
        "fingerabdruck": abdruck,
        "schluessel": schluessel,
        "kennwerte": _kodieren(kennwerte),
        "zusammenfassung": zusammenfassungen,
        "ergebnis": _kodieren(ergebnis),
        # 3D-Lösungen je Frequenz (Gegenprobe 75): gültig für denselben
        # Physik-Schlüssel (ohne Punktzahl und Richtfrequenzen) und
        # dieselbe Modellquelle samt Baufunktionen — unabhängig davon, wie
        # die App die Ergebnisse auswertet
        "physik": physik,
        "abdruck_3d": abdruck_3d,
        "loesungen_3d": _kodieren(loesungen or {}),
    }
    tmp = datei + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(daten, fh, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, datei)


def laden(schluessel, abdruck, datei=DATEI):
    """(ergebnis, zusammenfassungen, meta) bei passendem Fingerabdruck —
    meta mit Erzeugungsdatum, Rechenzeit und den Kennwerten der Kapsel —,
    sonst (None, None, grund) mit grund 'fehlt', 'parameter' oder
    'veraltet'."""
    if not os.path.exists(datei):
        return None, None, "fehlt"
    with open(datei, encoding="utf-8") as fh:
        daten = json.load(fh)
    if daten.get("format") != FORMAT or daten.get("schluessel") != schluessel:
        return None, None, "parameter"
    if daten.get("fingerabdruck") != abdruck:
        return None, None, "veraltet"
    meta = {"erzeugt": daten["erzeugt"], "rechenzeit_s": daten["rechenzeit_s"],
            "kennwerte": _dekodieren(daten["kennwerte"])}
    return _dekodieren(daten["ergebnis"]), daten["zusammenfassung"], meta


_LOESUNGEN = {}

# 3D-Lösungen je Kapsel über Sitzungen hinweg, solange der Serverprozess
# lebt (Gegenprobe 75): Schlüssel ist der Fingerabdruck aus Physik-Schlüssel,
# Modellquelle und Baufunktionen. Hier und nicht in app.py, weil Streamlit
# das App-Skript bei jedem Lauf neu ausführt (seine Globals überleben
# nicht); höchstens _PROZESS_MAX Kapseln, die zuletzt benutzte am Ende.
_PROZESS_3D = {}
_PROZESS_MAX = 40


def prozess_holen(abdruck_3d):
    """Die im Prozess gemerkten 3D-Lösungen dieser Kapsel ({} ohne)."""
    loes = _PROZESS_3D.pop(abdruck_3d, None)
    if loes is None:
        return {}
    _PROZESS_3D[abdruck_3d] = loes
    return dict(loes)


def prozess_merken(abdruck_3d, loesungen):
    """3D-Lösungen dieser Kapsel im Prozess merken (zu den vorhandenen)."""
    if not loesungen:
        return
    alt = _PROZESS_3D.pop(abdruck_3d, {})
    alt.update(loesungen)
    _PROZESS_3D[abdruck_3d] = alt
    while len(_PROZESS_3D) > _PROZESS_MAX:
        _PROZESS_3D.pop(next(iter(_PROZESS_3D)))


def laden_loesungen(physik, abdruck_3d, datei=DATEI):
    """Die gespeicherten 3D-Lösungen {ω: Eintrag} bei passendem Physik-
    Schlüssel und Fingerabdruck, sonst None (Gegenprobe 75). Einmal je
    Prozess und Dateistand gelesen."""
    try:
        st_ = os.stat(datei)
    except OSError:
        return None
    marke = (datei, st_.st_mtime_ns, st_.st_size)
    if marke not in _LOESUNGEN:
        with open(datei, encoding="utf-8") as fh:
            daten = json.load(fh)
        _LOESUNGEN.clear()
        _LOESUNGEN[marke] = (
            daten.get("format"), daten.get("physik"),
            daten.get("abdruck_3d"),
            _dekodieren(daten.get("loesungen_3d") or {"__dict__": []}))
    fmt, phys, abdr, loes = _LOESUNGEN[marke]
    if fmt != FORMAT or phys != physik or abdr != abdruck_3d or not loes:
        return None
    return loes


def _gleich(a, b):
    """Bitgleichheit zweier Ergebnisbäume (NaN gleich NaN)."""
    import numpy as np
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_gleich(a[k], b[k]) for k in a))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=True))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def erzeugen(datei=DATEI):
    """Voreinstellung durch die Bau- und Rechenfunktionen der App rechnen
    und speichern (der App-Import rechnet ohne Streamlit-Laufzeit nichts)."""
    import warnings
    warnings.simplefilter("ignore")
    import app as A
    p = dict(A.DEFAULTS)
    schluessel = A._ergebnis_schluessel(p)
    abdruck = fingerabdruck(schluessel, A._VOREINSTELLUNG_FUNKTIONEN)
    t0 = time.time()
    letzte = [0.0]

    def fortschritt(anteil, text):
        if time.time() - letzte[0] > 30 or anteil >= 1.0:
            letzte[0] = time.time()
            print(f"  {100 * anteil:5.1f} %  {text}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    kapsel = A.build_capsule(p)
    # die 3D-Frequenzen parallel in Arbeiterprozessen, soweit Kerne und
    # Speicher reichen (Gegenprobe 75); compute_results findet sie dann
    n_par, n_proz = A._vorrechnen_3d(p, kapsel, fortschritt)
    if n_par:
        print(f"  {n_par} Frequenzen in {n_proz} Prozessen vorgerechnet "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    ergebnis = A.compute_results(schluessel, kapsel, fortschritt)
    zusammenfassungen = {sp: kapsel.summary(lang=sp) for sp in ("de", "en")}
    kennwerte = A._kennwerte(kapsel)
    physik = A._physik_schluessel(schluessel)
    abdruck_3d = fingerabdruck(physik, A._BAU_FUNKTIONEN)
    loesungen = kapsel.loesungen_3d()
    dauer = time.time() - t0
    speichern(schluessel, abdruck, ergebnis, zusammenfassungen, kennwerte,
              dauer, datei, physik=physik, abdruck_3d=abdruck_3d,
              loesungen=loesungen)
    zurueck, zus, meta = laden(schluessel, abdruck, datei)
    loes_zurueck = laden_loesungen(physik, abdruck_3d, datei) or {}
    if zurueck is None or not _gleich(ergebnis, zurueck) \
            or zus != zusammenfassungen \
            or not _gleich(kennwerte, meta["kennwerte"]) \
            or not _gleich({k: {w: list(v) if v is not None else None
                                for w, v in e.items()}
                            for k, e in loesungen.items()}, loes_zurueck):
        raise RuntimeError("Gespeicherte Voreinstellung liest sich nicht "
                           "bitgleich zurück")
    print(f"Voreinstellung gespeichert: {datei} "
          f"({os.path.getsize(datei) / 1e3:.0f} kB, {dauer / 60:.1f} min "
          f"gerechnet, {len(loesungen)} 3D-Lösungen, bitgleich "
          f"zurückgelesen)")


if __name__ == "__main__":
    erzeugen(sys.argv[1] if len(sys.argv) > 1 else DATEI)
