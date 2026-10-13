# -*- coding: utf-8 -*-
"""Gespeicherte Ergebnisse der App-Voreinstellung (Gegenprobe 63 f).

Die Voreinstellung der App ist die experimentelle K67 mit 45 µm
Zwischenspalt im FEINEN 3D-Feldmodell: ~13 s je Frequenzpunkt, beim Start
also weit über eine Stunde. Ihre Ergebnisse (Frequenzgang, Richt-
diagramme, Diagnose, Summary in beiden Sprachen) liegen deshalb fertig in
``assets/voreinstellung.json`` — NUR für genau diesen Parametersatz. Jede
andere Einstellung rechnet die App wie bisher.

Gültig ist die Datei nur, solange ihr Fingerabdruck stimmt:

* der Ergebnisschlüssel der App (alle physikrelevanten Parameter),
* die Modellquelle ``microphone_capsule.py``,
* die Bau- und Rechenfunktionen der App (``build_capsule``,
  ``compute_results``),

jeweils als Syntaxbaum ohne Kommentare und Docstrings: eine reine
Kommentaränderung lässt die Datei gültig, jede Codeänderung macht sie
ungültig. Dann rechnet die App neu und sagt das — veraltete Ergebnisse
erscheinen nie still.

Neu erzeugen (nach jeder Modelländerung; ~1,5 h):

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

import numpy as np

_WURZEL = os.path.dirname(os.path.abspath(__file__))
DATEI = os.path.join(_WURZEL, "assets", "voreinstellung.json")
_MODELL = os.path.join(_WURZEL, "microphone_capsule.py")
FORMAT = 1


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


def _baum_hash(quelltext):
    baum = _ohne_docstrings(ast.parse(textwrap.dedent(quelltext)))
    return hashlib.sha256(
        ast.dump(baum, include_attributes=False).encode()).digest()


def fingerabdruck(schluessel, funktionen):
    """Fingerabdruck aus Ergebnisschlüssel, Modellquelle und den
    übergebenen App-Funktionen (s. Moduldoku)."""
    h = hashlib.sha256()
    h.update(schluessel.encode())
    with open(_MODELL, encoding="utf-8") as fh:
        h.update(_baum_hash(fh.read()))
    for fn in funktionen:
        h.update(_baum_hash(inspect.getsource(fn)))
    return h.hexdigest()


def _kodieren(x):
    """Ergebnisbaum -> JSON-taugliche Form, verlustfrei (Gleitkommazahlen
    über repr, komplexe Felder als Real-/Imaginärteil, Wörterbücher als
    Paarlisten, weil die Richtdiagramme Gleitkomma-Schlüssel haben)."""
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
              rechenzeit_s, datei=DATEI):
    daten = {
        "format": FORMAT,
        "erzeugt": _dt.datetime.now().isoformat(timespec="seconds"),
        "rechenzeit_s": round(float(rechenzeit_s), 1),
        "fingerabdruck": abdruck,
        "schluessel": schluessel,
        "zusammenfassung": zusammenfassungen,
        "ergebnis": _kodieren(ergebnis),
    }
    tmp = datei + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(daten, fh, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, datei)


def laden(schluessel, abdruck, datei=DATEI):
    """(ergebnis, zusammenfassungen, meta) bei passendem Fingerabdruck,
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
    meta = {"erzeugt": daten["erzeugt"], "rechenzeit_s": daten["rechenzeit_s"]}
    return _dekodieren(daten["ergebnis"]), daten["zusammenfassung"], meta


def _gleich(a, b):
    """Bitgleichheit zweier Ergebnisbäume (NaN gleich NaN)."""
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
    ergebnis = A.compute_results(schluessel, kapsel, fortschritt)
    zusammenfassungen = {sp: kapsel.summary(lang=sp) for sp in ("de", "en")}
    dauer = time.time() - t0
    speichern(schluessel, abdruck, ergebnis, zusammenfassungen, dauer, datei)
    zurueck, zus, meta = laden(schluessel, abdruck, datei)
    if zurueck is None or not _gleich(ergebnis, zurueck) \
            or zus != zusammenfassungen:
        raise RuntimeError("Gespeicherte Voreinstellung liest sich nicht "
                           "bitgleich zurück")
    print(f"Voreinstellung gespeichert: {datei} "
          f"({os.path.getsize(datei) / 1e3:.0f} kB, {dauer / 60:.1f} min "
          f"gerechnet, bitgleich zurückgelesen)")


if __name__ == "__main__":
    erzeugen(sys.argv[1] if len(sys.argv) > 1 else DATEI)
