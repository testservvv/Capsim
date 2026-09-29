"""Stand-Werte der Gegenproben: Basisdatei, Sperrklinken, Bericht.

Die Gegenproben prüfen drei Arten von Aussagen (README, Selbsttest):

* INVARIANTE — exakte Identität, Grenzfall, Reziprozität, Konvergenz,
  Literaturformel, Richtungsaussage, Programmverhalten: hartes ``assert``.
* FREMDREFERENZ — Messung, FEM, digitalisierte Kurve, Datenblatt:
  ``assert`` mit Toleranz aus der Unsicherheit der Referenz.
* STAND-WERT — eigener Rechenwert ohne äußeren Grund (ein Zahlenfenster
  um das, was das Modell heute liefert). Kein ``assert``: der Test meldet
  ihn mit ``stand.wert``, der Lauf vergleicht ihn mit
  ``tests/basis_werte.json``, und der Bericht am Ende nennt jede
  Verschiebung. Eine Modelländerung bricht so keine Proben mehr, sie
  zeigt ihre Wirkung als Liste.

Sonderfall SPERRKLINKE (``stand.sperrklinke``): ein bekannter Restfehler,
der nur kleiner werden darf. Der Test scheitert, wenn der Wert schlechter
wird als in der Basis (plus Toleranz); eine Verbesserung meldet der
Bericht.

Die Basis ändert sich nur bewusst: ``--basis-uebernehmen`` schreibt die
Werte aller bestandenen Tests hinein. Eine gescheiterte Sperrklinke kommt
so nicht hinein (ihr Test ist nicht bestanden); sie zu lockern heißt,
ihren Eintrag von Hand zu entfernen.
"""
import json
import math
import re
from pathlib import Path

DATEI = Path(__file__).with_name("basis_werte.json")
# Rundungsrauschen (Thread-Zahl, Zerlegung) gilt nicht als Änderung
RTOL, ATOL = 1e-5, 1e-12


def lade():
    try:
        return json.loads(DATEI.read_text())
    except FileNotFoundError:
        return {}


def schreibe(daten):
    DATEI.write_text(json.dumps(dict(sorted(daten.items())),
                                ensure_ascii=False, indent=1) + "\n")


def _kurz(nodeid):
    """'tests/x.py::test_gp22e_kernlage…' -> 'gp22e'"""
    m = re.search(r"test_(gp\d+[a-z]?\d?)", nodeid)
    return m.group(1) if m else nodeid


class Stand:
    """Meldestelle eines Tests für Stand-Werte und Sperrklinken."""

    def __init__(self, nodeid, basis, melden):
        self.nodeid = nodeid
        self.basis = basis
        self._melden = melden
        self._namen = set()

    def _eintrag(self, name, wert, einheit, text, **extra):
        schluessel = f"{_kurz(self.nodeid)}.{name}"
        if schluessel in self._namen:
            raise ValueError(f"Stand-Wert {schluessel} doppelt gemeldet")
        self._namen.add(schluessel)
        wert = float(wert)
        if not math.isfinite(wert):
            raise ValueError(f"Stand-Wert {schluessel} ist nicht endlich")
        e = dict(wert=wert, einheit=einheit, text=text, test=self.nodeid,
                 **extra)
        self._melden(schluessel, e)
        return schluessel, e

    def wert(self, name, wert, einheit="", text=""):
        """Stand-Wert melden (kein assert)."""
        self._eintrag(name, wert, einheit, text, art="stand")

    def sperrklinke(self, name, wert, einheit="", text="", besser="kleiner",
                    toleranz=0.0):
        """Bekannten Restfehler melden; schlechter als die Basis (über die
        Toleranz hinaus) lässt den Test scheitern."""
        if besser not in ("kleiner", "größer"):
            raise ValueError("besser muss 'kleiner' oder 'größer' sein")
        schluessel, _ = self._eintrag(name, wert, einheit, text,
                                      art="sperrklinke", besser=besser,
                                      toleranz=float(toleranz))
        alt = self.basis.get(schluessel)
        if alt is None:
            return
        if besser == "kleiner":
            schlechter = float(wert) > alt["wert"] + toleranz
        else:
            schlechter = float(wert) < alt["wert"] - toleranz
        if schlechter:
            raise AssertionError(
                f"Sperrklinke {schluessel} ({text}): schlechter geworden — "
                f"{float(wert):.6g} gegen Basis {alt['wert']:.6g} {einheit} "
                f"(Toleranz {toleranz:g}). Ein bekannter Restfehler darf "
                f"nur kleiner werden. Bewusst lockern: den Eintrag aus "
                f"{DATEI.name} entfernen und mit --basis-uebernehmen neu "
                f"schreiben.")


def _gleich(a, b):
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def vergleiche(basis, gemeldet, bestanden):
    """Gemeldete Werte (nur bestandene Tests) gegen die Basis.

    Rückgabe: dict mit 'geaendert', 'neu', 'verbessert', 'toleriert'
    (Sperrklinke schlechter, aber in der Toleranz) und 'fehlt' (Test
    bestanden, Wert nicht mehr gemeldet)."""
    erg = dict(geaendert=[], neu=[], verbessert=[], toleriert=[], fehlt=[],
               gleich=0)
    for k, e in sorted(gemeldet.items()):
        alt = basis.get(k)
        if alt is None:
            erg["neu"].append((k, None, e))
        elif _gleich(e["wert"], alt["wert"]):
            erg["gleich"] += 1
        elif e.get("art") == "sperrklinke":
            besser = (e["wert"] < alt["wert"] if e.get("besser") == "kleiner"
                      else e["wert"] > alt["wert"])
            erg["verbessert" if besser else "toleriert"].append((k, alt, e))
        else:
            erg["geaendert"].append((k, alt, e))
    erg["fehlt"] = sorted(k for k, alt in basis.items()
                          if alt.get("test") in bestanden
                          and k not in gemeldet)
    return erg


def bericht(erg, uebernommen=None):
    """Zeilen für den Abschnitt am Ende des Laufs."""
    def _zeile(marke, k, alt, e):
        w = e["wert"]
        if alt is None:
            d = ""
        else:
            a = alt["wert"]
            d = f"  (vorher {a:.6g}" + (
                f", {100.0 * (w / a - 1.0):+.2f} %)" if a != 0 else ")")
        return f"  {marke:10s} {k:34s} {w:.6g} {e['einheit']}{d}  {e['text']}"

    n = (erg["gleich"] + len(erg["geaendert"]) + len(erg["neu"])
         + len(erg["verbessert"]) + len(erg["toleriert"]))
    zeilen = [f"{n} Stand-Werte gemeldet: {erg['gleich']} wie in der Basis, "
              f"{len(erg['geaendert'])} geändert, {len(erg['neu'])} neu, "
              f"{len(erg['verbessert'])} Sperrklinken verbessert, "
              f"{len(erg['toleriert'])} in der Toleranz schlechter, "
              f"{len(erg['fehlt'])} nicht mehr gemeldet."]
    for marke, liste in (("geändert", erg["geaendert"]),
                         ("verbessert", erg["verbessert"]),
                         ("toleriert", erg["toleriert"]),
                         ("neu", erg["neu"])):
        zeilen += [_zeile(marke, k, alt, e) for k, alt, e in liste]
    zeilen += [f"  fehlt      {k}" for k in erg["fehlt"]]
    offen = (erg["geaendert"] or erg["neu"] or erg["verbessert"]
             or erg["toleriert"] or erg["fehlt"])
    if uebernommen is not None:
        zeilen.append(f"Basis übernommen: {uebernommen} Werte in "
                      f"{DATEI.name} geschrieben.")
    elif offen:
        zeilen.append("Bewusst übernehmen: python microphone_capsule.py "
                      "--basis-uebernehmen")
    return zeilen


def uebernehme(basis, gemeldet, bestanden, gibt_es):
    """Neue Basis: gemeldete Werte übernehmen, nicht mehr gemeldete
    Werte bestandener Tests und Werte verschwundener Tests entfernen."""
    neu = {k: v for k, v in basis.items()
           if v.get("test") not in bestanden and gibt_es(v.get("test", ""))}
    neu.update(gemeldet)
    return neu
