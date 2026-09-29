"""Gegenprobe des Prüfrahmens: Stand-Werte, Sperrklinken, Basis (stand.py).

Keine Physik — aber der Bericht ersetzt für die Stand-Werte das assert,
also muss er selbst verlässlich sein: gleich bleibt gleich, jede
Verschiebung erscheint, eine schlechtere Sperrklinke lässt den Test
scheitern, und die Basis wird nur fortgeschrieben, was bestanden hat.
"""
import pytest

import stand as sw

_ID = "tests/test_x.py::test_gp99_beispiel"


def _stand(basis):
    gemeldet = {}
    return sw.Stand(_ID, basis, gemeldet.__setitem__), gemeldet


def test_stand_meldung():
    st, gem = _stand({})
    st.wert("a", 1.5, "dB", "Text")
    assert gem == {"gp99.a": dict(wert=1.5, einheit="dB", text="Text",
                                  test=_ID, art="stand")}
    with pytest.raises(ValueError, match="doppelt"):
        st.wert("a", 2.0)
    with pytest.raises(ValueError, match="nicht endlich"):
        st.wert("b", float("nan"))


def test_stand_sperrklinke():
    basis = {"gp99.k": dict(wert=1.0), "gp99.g": dict(wert=1.0)}
    for w in (0.5, 1.0, 1.1):                  # besser, gleich, in Toleranz
        st, _ = _stand(basis)
        st.sperrklinke("k", w, besser="kleiner", toleranz=0.1)
    st, _ = _stand(basis)
    with pytest.raises(AssertionError, match="schlechter"):
        st.sperrklinke("k", 1.11, besser="kleiner", toleranz=0.1)
    st, _ = _stand(basis)
    st.sperrklinke("g", 0.95, besser="größer", toleranz=0.05)
    st, _ = _stand(basis)
    with pytest.raises(AssertionError, match="schlechter"):
        st.sperrklinke("g", 0.9, besser="größer", toleranz=0.05)
    st, _ = _stand({})                         # ohne Basis: nur melden
    st.sperrklinke("k", 1e9)
    with pytest.raises(ValueError):
        st.sperrklinke("x", 1.0, besser="anders")


def test_stand_vergleich_und_bericht():
    def e(w, art="stand", **kw):
        return dict(wert=w, einheit="", text="", test=_ID, art=art, **kw)
    basis = {"gp99.gleich": e(1.0), "gp99.rausch": e(1.0),
             "gp99.anders": e(1.0), "gp99.besser": e(1.0, "sperrklinke",
                                                      besser="kleiner"),
             "gp99.toleriert": e(1.0, "sperrklinke", besser="kleiner"),
             "gp99.weg": e(1.0),
             "gp98.fremd": dict(e(1.0), test="tests/t.py::test_gp98_f")}
    gemeldet = {"gp99.gleich": e(1.0), "gp99.rausch": e(1.0 + 1e-7),
                "gp99.anders": e(1.01),
                "gp99.besser": e(0.9, "sperrklinke", besser="kleiner"),
                "gp99.toleriert": e(1.05, "sperrklinke", besser="kleiner"),
                "gp99.neu": e(3.0)}
    erg = sw.vergleiche(basis, gemeldet, {_ID})
    assert erg["gleich"] == 2                  # Rundungsrauschen zählt nicht
    assert [k for k, *_ in erg["geaendert"]] == ["gp99.anders"]
    assert [k for k, *_ in erg["verbessert"]] == ["gp99.besser"]
    assert [k for k, *_ in erg["toleriert"]] == ["gp99.toleriert"]
    assert [k for k, *_ in erg["neu"]] == ["gp99.neu"]
    assert erg["fehlt"] == ["gp99.weg"]        # gp98 lief nicht: kein Befund
    text = "\n".join(sw.bericht(erg))
    for k in ("gp99.anders", "gp99.besser", "gp99.toleriert", "gp99.neu",
              "gp99.weg", "--basis-uebernehmen"):
        assert k in text
    assert "+1.00 %" in text
    ruhig = sw.bericht(sw.vergleiche({"gp99.gleich": e(1.0)},
                                     {"gp99.gleich": e(1.0)}, {_ID}))
    assert len(ruhig) == 1 and "1 wie in der Basis" in ruhig[0]


def test_stand_uebernahme(tmp_path, monkeypatch):
    basis = {"gp99.alt": dict(wert=1.0, test=_ID),
             "gp98.nicht_gelaufen": dict(wert=2.0, test="da"),
             "gp97.verschwunden": dict(wert=3.0, test="weg")}
    gemeldet = {"gp99.neu": dict(wert=4.0, test=_ID)}
    neu = sw.uebernehme(basis, gemeldet, {_ID}, lambda t: t != "weg")
    # bestandener Test: nur noch, was er meldet; nicht gelaufener Test:
    # unverändert; verschwundener Test: entfernt
    assert neu == {"gp99.neu": gemeldet["gp99.neu"],
                   "gp98.nicht_gelaufen": basis["gp98.nicht_gelaufen"]}
    monkeypatch.setattr(sw, "DATEI", tmp_path / "basis.json")
    assert sw.lade() == {}
    sw.schreibe(neu)
    assert sw.lade() == neu
