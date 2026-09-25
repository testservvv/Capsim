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
    if report.when == "call" and report.passed:
        text = "".join(c for name, c in report.sections
                       if name.startswith("Captured stdout"))
        if text.strip():
            _PROTOKOLL[report.nodeid] = text.rstrip("\n")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _PROTOKOLL or config.getoption("--kein-protokoll"):
        return
    terminalreporter.section("Protokoll der Gegenproben")
    for nodeid in sorted(_PROTOKOLL, key=_gp_key):
        terminalreporter.write_line(_PROTOKOLL[nodeid])


def pytest_addoption(parser):
    parser.addoption("--kein-protokoll", action="store_true",
                     help="OK-Zeilen der Gegenproben nicht ausgeben")
