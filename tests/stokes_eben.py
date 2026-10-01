"""Ebener Gegenlöser zur Spalt-Loch-Mündung (Gegenprobe 67).

Für h ≪ a ist die Mündung von der Spaltseite her eine EBENE Umlenkung:
ein Kanal der Höhe h, dessen Boden an der Bohrungskante endet, während
die Decke (Membran) weiterläuft. Ihre Zusatzlänge in Spalthöhen ist eine
reine Zahl, der Grenzwert der achsensymmetrischen Zelle (stokes_zelle.py)
für h/a → 0. Gerechnet wird sie hier mit einem EIGENSTÄNDIGEN kartesischen
Löser — stationäre Stokes-Gleichung, μ = 1, MAC-Gitter mit beliebiger
Maske —, der zuvor an der exakten Lösung von Hasimoto (1958) geprüft wird:
Schlitz der Breite h in dünner Wand, Δp = 32·μ·q'/(π·h²).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def gitter(feinpunkte, x0, x1, d_min, wachs=1.06, d_max=2.0):
    """Flächen auf [x0, x1]: Weite d_min an den Feinpunkten, geometrisch
    wachsend bis d_max; die Feinpunkte liegen exakt auf Flächen."""
    xs = np.linspace(x0, x1, 200001)
    dist = np.min(np.abs(xs[:, None] - np.array(feinpunkte)[None, :]), axis=1)
    dd = np.minimum(d_min + (wachs - 1.0) * dist, d_max)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (1 / dd[1:] + 1 / dd[:-1])
                                         * np.diff(xs))])
    f = np.interp(np.linspace(0.0, s[-1], int(np.ceil(s[-1])) + 1), s, xs)
    for p in feinpunkte:
        f[np.argmin(np.abs(f - p))] = p
    return f


def loese(xf, zf, fluid, rand):
    """Stationäre ebene Stokes-Strömung (μ = 1) auf dem Gitter xf × zf.

    ``fluid`` (Nx, Nz) markiert die Fluidzellen; Flächen zwischen Fluid-
    und Festzellen sind haftende Wände. ``rand`` je 'links', 'rechts',
    'unten', 'oben': ('wand',), ('sym',) Symmetrie, ('u', f) vorgegebene
    Normalgeschwindigkeit f(Tangentialkoordinate) oder ('p', p0) Druck
    bei entwickelter Strömung. Rückgabe dict(p, u, w, Gitter)."""
    Nx, Nz = xf.size - 1, zf.size - 1
    xc, zc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    dx, dz = np.diff(xf), np.diff(zf)
    pid = -np.ones((Nx, Nz), int)
    pid[fluid] = np.arange(fluid.sum())
    k = int(fluid.sum())
    # u auf x-Flächen i = 0..Nx (links von Zelle i), w auf z-Flächen
    uid = -np.ones((Nx + 1, Nz), int)
    ufix = {}
    for j in range(Nz):
        for i in range(Nx + 1):
            L = bool(fluid[i - 1, j]) if i > 0 else None
            R = bool(fluid[i, j]) if i < Nx else None
            if L and R:
                uid[i, j] = k
                k += 1
            elif (i == 0 and R) or (i == Nx and L):
                t = rand["links" if i == 0 else "rechts"]
                if t[0] == "p":
                    uid[i, j] = k
                    k += 1
                else:
                    ufix[(i, j)] = t[1](zc[j]) if t[0] == "u" else 0.0
            elif L is not None and R is not None and L != R:
                ufix[(i, j)] = 0.0                     # Wandfläche
    wid = -np.ones((Nx, Nz + 1), int)
    wfix = {}
    for i in range(Nx):
        for j in range(Nz + 1):
            D = bool(fluid[i, j - 1]) if j > 0 else None
            U = bool(fluid[i, j]) if j < Nz else None
            if D and U:
                wid[i, j] = k
                k += 1
            elif (j == 0 and U) or (j == Nz and D):
                t = rand["unten" if j == 0 else "oben"]
                if t[0] == "p":
                    wid[i, j] = k
                    k += 1
                else:
                    wfix[(i, j)] = t[1](xc[i]) if t[0] == "u" else 0.0
            elif D is not None and U is not None and D != U:
                wfix[(i, j)] = 0.0
    N = k
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    def uval(i, j):
        if 0 <= j < Nz and 0 <= i <= Nx:
            if uid[i, j] >= 0:
                return ("var", uid[i, j])
            if (i, j) in ufix:
                return ("fix", ufix[(i, j)])
        return None

    def wval(i, j):
        if 0 <= i < Nx and 0 <= j <= Nz:
            if wid[i, j] >= 0:
                return ("var", wid[i, j])
            if (i, j) in wfix:
                return ("fix", wfix[(i, j)])
        return None

    for i in range(Nx):                                # Kontinuität
        for j in range(Nz):
            if not fluid[i, j]:
                continue
            r = pid[i, j]
            for v, sg, d in ((uval(i + 1, j), 1.0, dx[i]),
                             (uval(i, j), -1.0, dx[i]),
                             (wval(i, j + 1), 1.0, dz[j]),
                             (wval(i, j), -1.0, dz[j])):
                if v[0] == "var":
                    add(r, v[1], sg / d)
                else:
                    rhs[r] -= sg * v[1] / d

    def lap(row, nb):
        # −∂²u aus zwei Nachbarn [(Abstand, Wert)]; Wert ('var', id),
        # ('fix', v), ('spiegel',) Wand (u_Geist = −u), ('neumann',)
        (dp_, vp), (dm_, vm) = nb
        cp, cm = 2.0 / (dp_ * (dp_ + dm_)), 2.0 / (dm_ * (dp_ + dm_))
        add(row, row, cp + cm)
        for c, v in ((cp, vp), (cm, vm)):
            if v[0] == "var":
                add(row, v[1], -c)
            elif v[0] == "fix":
                rhs[row] += c * v[1]
            elif v[0] == "spiegel":
                add(row, row, c)
            else:
                add(row, row, -c)

    def quer(v, abstand, zw, rd):
        # Nachbar quer zur Komponente: Fläche (Fluid oder Wandfläche)
        # oder Spiegelung an Wand bzw. Rand
        if v is not None:
            return (abstand, v)
        art = "neumann" if rd is not None and rd[0] in ("sym", "p") \
            else "spiegel"
        return (2.0 * zw, (art,))

    for i in range(Nx + 1):                            # Impuls u
        for j in range(Nz):
            row = uid[i, j]
            if row < 0:
                continue
            if 0 < i < Nx:
                hx = xc[i] - xc[i - 1]
                add(row, pid[i, j], 1.0 / hx)
                add(row, pid[i - 1, j], -1.0 / hx)
                lap(row, [(xf[i + 1] - xf[i], uval(i + 1, j)),
                          (xf[i] - xf[i - 1], uval(i - 1, j))])
            else:                                      # Druckrand
                ci, sg = (0, -1.0) if i == 0 else (Nx - 1, 1.0)
                pr = rand["links" if i == 0 else "rechts"][1]
                add(row, pid[ci, j], -sg / (0.5 * dx[ci]))
                rhs[row] -= sg * pr / (0.5 * dx[ci])
                lap(row, [(dx[ci], uval(i + 1 if i == 0 else i - 1, j)),
                          (dx[ci], ("neumann",))])
            nb = []
            for jj in (j + 1, j - 1):
                v = uval(i, jj) if 0 <= jj < Nz else None
                rd = rand["unten"] if jj < 0 else (rand["oben"] if jj >= Nz
                                                   else None)
                zw = abs((zf[j + 1] if jj > j else zf[j]) - zc[j])
                nb.append(quer(v, abs(zc[jj] - zc[j]) if v is not None
                               else 0.0, zw, rd))
            lap(row, nb)
    for i in range(Nx):                                # Impuls w
        for j in range(Nz + 1):
            row = wid[i, j]
            if row < 0:
                continue
            if 0 < j < Nz:
                hz = zc[j] - zc[j - 1]
                add(row, pid[i, j], 1.0 / hz)
                add(row, pid[i, j - 1], -1.0 / hz)
                lap(row, [(zf[j + 1] - zf[j], wval(i, j + 1)),
                          (zf[j] - zf[j - 1], wval(i, j - 1))])
            else:
                cj, sg = (0, -1.0) if j == 0 else (Nz - 1, 1.0)
                pr = rand["unten" if j == 0 else "oben"][1]
                add(row, pid[i, cj], -sg / (0.5 * dz[cj]))
                rhs[row] -= sg * pr / (0.5 * dz[cj])
                lap(row, [(dz[cj], wval(i, j + 1 if j == 0 else j - 1)),
                          (dz[cj], ("neumann",))])
            nb = []
            for ii in (i + 1, i - 1):
                v = wval(ii, j) if 0 <= ii < Nx else None
                rd = rand["links"] if ii < 0 else (rand["rechts"] if ii >= Nx
                                                   else None)
                xw = abs((xf[i + 1] if ii > i else xf[i]) - xc[i])
                nb.append(quer(v, abs(xc[ii] - xc[i]) if v is not None
                               else 0.0, xw, rd))
            lap(row, nb)
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    x = spsolve(A, rhs)
    p = np.full((Nx, Nz), np.nan)
    p[fluid] = x[:pid.max() + 1]
    w = np.full((Nx, Nz + 1), np.nan)
    for (i, j), v in wfix.items():
        w[i, j] = v
    m = wid >= 0
    w[m] = x[wid[m]]
    return dict(p=p, w=w, xc=xc, zc=zc, xf=xf, zf=zf, dx=dx, dz=dz)


def hasimoto(t_h, d_min=0.01, wachs=1.08, X=40.0, D=40.0):
    """Schlitz der Breite h = 2 in einer Wand der Dicke t = t_h·h (halbes
    Gebiet, Symmetrie bei x = 0), Druckdifferenz 1. Rückgabe Δp·h²/(μq')
    — exakt 32/π für t → 0 (Hasimoto 1958)."""
    b, h = 1.0, 2.0
    t = t_h * h
    xf = gitter([0.0, b], 0.0, X, d_min * b, wachs)
    zf = gitter([0.0, t], -D, D + t, d_min * b, wachs)
    xc, zc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    fluid = np.ones((xc.size, zc.size), bool)
    fluid[np.ix_(xc > b, (zc > 0) & (zc < t))] = False
    s = loese(xf, zf, fluid, dict(links=("sym",), rechts=("sym",),
                                  unten=("p", 0.0), oben=("p", 1.0)))
    jz = int(np.argmin(np.abs(s["zf"])))
    q = -2.0 * np.nansum(s["w"][:, jz] * s["dx"])      # beide Hälften
    return h**2 / q


def ecke(d_min=0.005, wachs=1.06, W=40.0, D=80.0, Xc=10.0):
    """Ebene Umlenkung Spalt → Bohrung (h = 1): Spalt 0 < z < 1, x > 0,
    endet bei x = 0 an der Bohrungswand; die Deckwand z = 1 läuft über
    die Bohrung (−W < x < 0, z < 1) weiter. Parabel-Einlass (q' = 1) bei
    x = Xc, Auslass p = 0 bei z = −D, Symmetrie bei x = −W. Rückgabe die
    Zusatzlänge in Spalthöhen: Einlassdruck minus Kanal- und Bohrungs-
    Poiseuille, geteilt durch den Kanalgradienten 12μq'/h³."""
    h = 1.0
    xf = gitter([0.0], -W, Xc, d_min * h, wachs)
    zf = gitter([0.0, h], -D, h, d_min * h, wachs)
    xc, zc = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    fluid = np.zeros((xc.size, zc.size), bool)
    fluid[xc < 0, :] = True
    fluid[np.ix_(xc > 0, (zc > 0) & (zc < h))] = True

    def parabel(z):
        return np.where((z > 0) & (z < h), -6.0 * z * (h - z) / h**3, 0.0)

    s = loese(xf, zf, fluid, dict(links=("sym",), rechts=("u", parabel),
                                  unten=("p", 0.0), oben=("wand",)))
    i = xc.size - 1
    im_spalt = (zc > 0) & (zc < h)
    p_ein = np.sum(s["p"][i, im_spalt] * s["dz"][im_spalt]) / h
    # Bohrung: Breite W mit Wand und Symmetrie = halber Kanal 2W mit 2q'
    dp_kanal = 12.0 / h**3 * xc[i]
    dp_bohrung = 3.0 / W**3 * D
    return (p_ein - dp_kanal - dp_bohrung) / (12.0 / h**3) / h
