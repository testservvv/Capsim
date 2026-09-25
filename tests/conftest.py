"""pytest-Rahmen der Gegenproben.

* Fixtures für die Referenzkapseln, die mehrere Gegenproben teilen
  (Definitionen in ``basis.py``). Die Namen entsprechen den Variablen,
  unter denen der alte Selbsttest sie weiterreichte.
* Klassenschalter (``MicrophoneCapsule._MASS_EXACT`` usw.) werden nach
  jedem Test zurückgesetzt, auch wenn er mittendrin scheitert.
* Am Ende steht das Protokoll: die OK-Zeilen aller bestandenen
  Gegenproben in Nummernfolge (auch mit ``-n``, s. pytest_terminal_summary).
"""
import os
import re

# Parallel (pytest-xdist) rechnet jeder Worker einen Test; mehrere
# BLAS-Threads je Worker würden die Kerne nur überbuchen. Muss vor dem
# ersten numpy-Import stehen.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import basis  # noqa: E402
from basis import MicrophoneCapsule, _HAS_SCIPY  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def _need_scipy():
    if not _HAS_SCIPY:
        pytest.skip("SciPy fehlt (2D-/3D-Feldmodell)")


# ----------------------------------------------------------------------
# Referenzkapseln
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def capsule():
    return basis.demo_capsule()


@pytest.fixture(scope="session")
def k67():
    return basis.k67_capsule()


@pytest.fixture(scope="session")
def na_k(k67):
    """Minimum des K67-Patterns bei 1 kHz (Gegenprobe 6)."""
    return basis.k67_null_angle(k67)


@pytest.fixture(scope="session")
def k67_bem():
    _need_scipy()
    return basis.k67_capsule(axial_body_model="bem")


@pytest.fixture
def deb_kwargs():
    _need_scipy()
    return dict(basis.DEB_KWARGS)


@pytest.fixture(scope="session")
def deb0():
    _need_scipy()
    return basis.debenham()


@pytest.fixture(scope="session")
def deb1():
    _need_scipy()
    return basis.debenham(**basis.DEB_CLEARANCE)


@pytest.fixture(scope="session")
def deb3():
    _need_scipy()
    return basis.debenham(squeeze_model="3d", **basis.DEB_CLEARANCE)


@pytest.fixture(scope="session")
def deb3b():
    _need_scipy()
    return basis.debenham(squeeze_model="3d", **basis.DEB_CLEARANCE_WIDE)


@pytest.fixture(scope="session")
def hermetic():
    return basis.hermetic_capsule()


# ----------------------------------------------------------------------
# Klassenschalter nach jedem Test zurücksetzen
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _klassenzustand():
    vorher = dict(vars(MicrophoneCapsule))
    yield
    jetzt = dict(vars(MicrophoneCapsule))
    for k in jetzt.keys() - vorher.keys():
        delattr(MicrophoneCapsule, k)
    for k, v in vorher.items():
        if jetzt.get(k, v) is not v or k not in jetzt:
            setattr(MicrophoneCapsule, k, v)


# ----------------------------------------------------------------------
# Protokoll
# ----------------------------------------------------------------------
_PROTOKOLL = {}


def _gp_key(nodeid):
    m = re.search(r"test_gp(\d+)", nodeid)
    return (int(m.group(1)) if m else 10**6, nodeid)


def pytest_runtest_logreport(report):
    _DAUERN[report.nodeid] = _DAUERN.get(report.nodeid, 0.0) + report.duration
    if report.when == "call" and report.passed:
        text = "".join(c for name, c in report.sections
                       if name.startswith("Captured stdout"))
        if text.strip():
            _PROTOKOLL[report.nodeid] = text.rstrip("\n")


def pytest_sessionfinish(session, exitstatus):
    """Laufzeiten für die Reihenfolge des nächsten Laufs merken (nur der
    steuernde Prozess, bei xdist nicht die Worker)."""
    config = session.config
    cache = getattr(config, "cache", None)
    if cache is None or hasattr(config, "workerinput") or not _DAUERN:
        return
    alt = cache.get(_DAUER_KEY, {})
    alt.update({k: round(v, 2) for k, v in _DAUERN.items()})

    def _gibt_es(nodeid):
        # umbenannte oder aufgeteilte Tests nicht ewig mitschleppen
        datei, _, name = nodeid.partition("::")
        pfad = config.rootpath / datei
        return pfad.is_file() and f"def {name}(" in pfad.read_text()

    cache.set(_DAUER_KEY, {k: v for k, v in alt.items() if _gibt_es(k)})


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _PROTOKOLL or config.getoption("--kein-protokoll"):
        return
    terminalreporter.section("Protokoll der Gegenproben")
    for nodeid in sorted(_PROTOKOLL, key=_gp_key):
        terminalreporter.write_line(_PROTOKOLL[nodeid])


_DAUER_KEY = "capsim/dauern"
_DAUERN = {}


def pytest_collection_modifyitems(config, items):
    """Reihenfolge für eine gleichmäßige Auslastung der Worker.

    Grundlage sind die Laufzeiten des letzten Laufs (pytest-Cache, s.
    pytest_sessionfinish); ohne sie gelten slow-Proben als lang. Seriell:
    längste zuerst. Parallel mit ``--dist worksteal`` (Voreinstellung von
    ``python microphone_capsule.py``) bekommt jeder Worker anfangs einen
    zusammenhängenden Block gleicher Testanzahl, danach stehlen freie
    Worker die hintere Hälfte der längsten Warteschlange. Die Blöcke
    werden deshalb so gefüllt, dass jeder etwa gleich lange rechnet
    (längste zuerst, jeweils in den bisher kürzesten Block mit freiem
    Platz). Die Reihenfolge hängt nur vom Cache und der Worker-Zahl ab,
    ist also in allen Workern gleich, wie xdist es verlangt.
    """
    cache = getattr(config, "cache", None)
    alt = cache.get(_DAUER_KEY, {}) if cache is not None else {}

    def _dauer(it):
        if it.nodeid in alt:
            return alt[it.nodeid]
        return 60.0 if it.get_closest_marker("slow") is not None else 1.0

    items.sort(key=lambda it: -_dauer(it))
    n = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    if n < 2 or len(items) <= n:
        return
    groesse, rest = [], len(items)
    for k in range(n):                       # wie worksteal anfangs teilt
        groesse.append(rest // (n - k))
        rest -= groesse[-1]
    bloecke, last = [[] for _ in range(n)], [0.0] * n
    for it in items:
        k = min((k for k in range(n) if len(bloecke[k]) < groesse[k]),
                key=lambda k: last[k])
        bloecke[k].append(it)
        last[k] += _dauer(it)
    items[:] = [it for blk in bloecke for it in blk]


def pytest_addoption(parser):
    parser.addoption("--kein-protokoll", action="store_true",
                     help="OK-Zeilen der Gegenproben nicht ausgeben")
