"""Referenzlöser der Spalt-Loch-Mündung (Gegenprobe 67).

Löst die linearisierte, inkompressible, instationäre Stokes-Gleichung

    jω·ρ0·u = −∇p + μ·(∇²u − e_r·u_r/r²),     ∇·u = 0

in einer axialsymmetrischen LOCHZELLE: Spaltfilm 0 < z < h über
0 < r < R_c, Bohrung 0 < r < a, −L < z < 0. Die Membran bei z = h bewegt
sich gleichförmig (u_z = −1, u_r = 0), Backplate (z = 0, r > a) und
Lochwand (r = a) haften, r = R_c ist Symmetrierand, am Bohrungsende
z = −L ist p = 0 bei voll entwickelter Strömung. Diskretisierung: MAC-
Gitter (versetzte Geschwindigkeiten), Finite Volumen, zur Ecke r = a,
z = 0 und zu allen Wänden verdichtet.

ABZUG AUF DEMSELBEN GITTER. Gesucht ist die MÜNDUNGSKORREKTUR

    ΔZ = Z_Zelle − [Z_Reynolds-Zelle + Z_Rohr],

also eine kleine Differenz großer Größen. Die beiden Abzugsterme werden
deshalb NICHT analytisch, sondern mit derselben Diskretisierung gerechnet
(1D-Schlitzprofil auf dem z-Gitter des Films, 1D-Rohrprofil auf dem
r-Gitter der Bohrung, radiale FV-Reynolds-Gleichung auf den Filmflächen).
Der Gitterfehler der ungestörten Bereiche hebt sich so heraus, übrig
bleibt der der Ecke (Gitterprobe fein 1 gegen 3: < 0.2 % von ΔZ, auch
bei h/a = 0.01). Die analytischen Formen (Zwikker–Kosten, B(q)/πK)
trifft der Abzug auf < 0.1 %. Mit μ = 0 rechnet derselbe Löser die
Potentialströmung (Wände ohne Haftung) — den Trägheitsgrenzfall.

DREITOR. Die Ecke koppelt drei Ströme: den Filmzufluss Q_f, den Fluss
der Membran über der Öffnung Q_m und den Rohrstrom Q = Q_f + Q_m. Für
JEDE Zelle mit Lochflächenanteil q = a²/R_c² gilt

    ΔZ_Zelle(q) = (1−q)²·z_ff + 2q(1−q)·z_fm + q²·z_mm,

und zwar bis auf Rundung: die Schmierfilmlösung mit gleichförmigem
Quetschen ist im Ring selbst eine exakte Stokes-Lösung (für u_r ∝
R_c²/r − r verschwindet der radiale Zähigkeitsterm, u_z hängt nicht von
r ab), Abweichungen entstehen nur an der Ecke. Drei Zellgrößen bestimmen
(z_ff, z_fm, z_mm); jede weitere prüft die Form.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from microphone_capsule import RHO0, MU_AIR, MicrophoneCapsule


def _gestuft(x0, x1, d0, wachs, d_max):
    """Flächen von x0 nach x1: erste Weite d0, geometrisch wachsend bis
    d_max; die letzte Zelle wird an x1 angeglichen."""
    s = np.sign(x1 - x0)
    xs, d = [x0], d0
    while abs(x1 - xs[-1]) > 1e-12 * max(abs(x0), abs(x1), d0):
        xs.append(xs[-1] + s * min(d, abs(x1 - xs[-1])))
        d = min(d * wachs, d_max)
    xs = np.array(xs)
    if xs.size > 2 and abs(xs[-1] - xs[-2]) < 0.3 * abs(xs[-2] - xs[-3]):
        xs = np.delete(xs, -2)
    return xs


def loese_zelle(a, h, R_c, L, omega, fein=1.5, rho=RHO0, mu=MU_AIR):
    """Zellimpedanz Z = <p>_Membran / Q und die Abzugsgröße auf demselben
    Gitter. Rückgabe dict(Z, Z_ref, N).

    Gitter: von der Ecke (r = a, z = 0) aus geometrisch gestuft, erste
    Weite 0.01·min(h, a)/fein — bezogen auf den SPALT, denn bei h ≪ a
    ist die Ecke so groß wie h (die ebene Probe konvergiert erst ab
    ~0.005·h). Im Film quer Kosinus-Verdichtung zu beiden Wänden."""
    nz_film = int(30 * fein)
    wachs = 1.0 + 0.08 / fein
    d0 = 0.01 * min(h, a) / fein
    zfilm = 0.5 * h * (1.0 - np.cos(np.pi * np.linspace(0, 1, nz_film + 1)))
    d0z = zfilm[1]
    r_in = _gestuft(a, 0.0, d0, wachs, a / (15 * fein))[::-1]
    r_out = _gestuft(a, R_c, d0, wachs, (R_c - a) / (15 * fein))
    rf = np.concatenate([r_in, r_out[1:]])
    z_loch = _gestuft(0.0, -L, d0z, wachs, L / (15 * fein))[::-1]
    zf = np.concatenate([z_loch, zfilm[1:]])
    Nr, Nz = rf.size - 1, zf.size - 1
    rc, zc = 0.5 * (rf[1:] + rf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    dr, dz = np.diff(rf), np.diff(zf)
    i_a = int(np.argmin(np.abs(rf - a)))
    j0 = int(np.argmin(np.abs(zf)))
    fluid = np.zeros((Nr, Nz), bool)
    fluid[:, j0:] = True                       # Spalt
    fluid[:i_a, :j0] = True                    # Bohrung
    pid = -np.ones((Nr, Nz), int)
    pid[fluid] = np.arange(fluid.sum())
    n_p = int(fluid.sum())
    k = n_p
    urid = -np.ones((Nr + 1, Nz), int)         # u_r auf inneren r-Flächen
    for j in range(Nz):
        for i in range(1, Nr):
            if fluid[i - 1, j] and fluid[i, j]:
                urid[i, j] = k
                k += 1
    uzid = -np.ones((Nr, Nz + 1), int)         # u_z; j = 0 ist der Auslass
    for i in range(Nr):
        for j in range(Nz):
            if fluid[i, j] and (j == 0 or fluid[i, j - 1]):
                uzid[i, j] = k
                k += 1
    N = k
    rows, cols, vals = [], [], []
    rhs = np.zeros(N, complex)
    w0 = -1.0                                  # Membran nach unten
    jw = 1j * omega * rho

    def add(r_, c_, v_):
        rows.append(r_)
        cols.append(c_)
        vals.append(v_)

    # Kontinuität je Fluidzelle (Volumen je 2π: r·dr·dz)
    for i in range(Nr):
        for j in range(Nz):
            if not fluid[i, j]:
                continue
            row, vol = pid[i, j], rc[i] * dr[i]
            for ii, sgn in ((i + 1, 1.0), (i, -1.0)):
                if urid[ii, j] >= 0:
                    add(row, urid[ii, j], sgn * rf[ii] / vol)
            for jj, sgn in ((j + 1, 1.0), (j, -1.0)):
                if jj == Nz:
                    rhs[row] -= sgn * w0 / dz[j]
                elif uzid[i, jj] >= 0:
                    add(row, uzid[i, jj], sgn / dz[j])
    # Impuls u_r
    for j in range(Nz):
        for i in range(1, Nr):
            row = urid[i, j]
            if row < 0:
                continue
            add(row, row, jw)
            hdr = rc[i] - rc[i - 1]
            add(row, pid[i, j], 1.0 / hdr)
            add(row, pid[i - 1, j], -1.0 / hdr)
            # radial: d/dr[(1/r)·d(r·u)/dr]
            for kz, sgn in ((i, 1.0), (i - 1, -1.0)):
                for ff, s2 in ((kz + 1, 1.0), (kz, -1.0)):
                    if 0 < ff < Nr and urid[ff, j] >= 0:
                        add(row, urid[ff, j],
                            -mu * sgn * s2 * rf[ff] / (rc[kz] * dr[kz] * hdr))
            # axial, Wände über Spiegelung (u_Geist = −u)
            nb = []
            for jj in (j + 1, j - 1):
                if 0 <= jj < Nz and urid[i, jj] >= 0:
                    nb.append((zc[jj], urid[i, jj]))
                else:
                    zw = zf[j + 1] if jj > j else zf[j]
                    if jj < j and 0 <= jj < Nz and i == i_a and fluid[i - 1, jj]:
                        nb.append((zc[jj], None))      # Lochwandfläche: u = 0
                    else:
                        nb.append((2 * zw - zc[j], "spiegel"))
            (zp, up), (zm, um) = nb
            Dp, Dm = zp - zc[j], zc[j] - zm
            cp, cm = 2.0 / (Dp * (Dp + Dm)), 2.0 / (Dm * (Dp + Dm))
            add(row, row, mu * (cp + cm))
            for cc, u in ((cp, up), (cm, um)):
                if u is None:
                    continue
                if u == "spiegel":
                    add(row, row, mu * cc)
                else:
                    add(row, u, -mu * cc)
    # Impuls u_z
    for i in range(Nr):
        for j in range(Nz):
            row = uzid[i, j]
            if row < 0:
                continue
            add(row, row, jw)
            if j == 0:                             # p = 0 genau bei z = −L
                add(row, pid[i, 0], 2.0 / dz[0])
            else:
                hdz = zc[j] - zc[j - 1]
                add(row, pid[i, j], 1.0 / hdz)
                add(row, pid[i, j - 1], -1.0 / hdz)
            if j + 1 < Nz and uzid[i, j + 1] >= 0:
                zp, up = zf[j + 1], uzid[i, j + 1]
            elif j + 1 == Nz:
                zp, up = zf[Nz], ("fest", w0)
            else:
                zp, up = zf[j + 1], ("fest", 0.0)
            if j == 0:
                zm, um = zf[0] - dz[0], "gleich"   # voll entwickelt
            elif uzid[i, j - 1] >= 0:
                zm, um = zf[j - 1], uzid[i, j - 1]
            else:
                zm, um = zf[j - 1], ("fest", 0.0)
            Dp, Dm = zp - zf[j], zf[j] - zm
            cp, cm = 2.0 / (Dp * (Dp + Dm)), 2.0 / (Dm * (Dp + Dm))
            add(row, row, mu * (cp + cm))
            for cc, u in ((cp, up), (cm, um)):
                if isinstance(u, tuple):
                    rhs[row] += mu * cc * u[1]
                elif u == "gleich":
                    add(row, row, -mu * cc)
                else:
                    add(row, u, -mu * cc)
            # radial: (1/r)·d/dr(r·du/dr); Achse und Symmetrierand ohne Fluss
            for ff, nbi in ((i + 1, i + 1), (i, i - 1)):
                if ff == 0 or ff == Nr:
                    continue
                if 0 <= nbi < Nr and uzid[nbi, j] >= 0:
                    g = rf[ff] / (abs(rc[nbi] - rc[i]) * rc[i] * dr[i])
                    add(row, row, mu * g)
                    add(row, uzid[nbi, j], -mu * g)
                else:                              # Wand bei rf[ff]
                    g = rf[ff] / (abs(rf[ff] - rc[i]) * rc[i] * dr[i])
                    add(row, row, mu * g)
    A = coo_matrix((np.array(vals, complex), (np.array(rows), np.array(cols))),
                   shape=(N, N)).tocsr()
    x = spsolve(A, rhs)
    p_top = x[pid[:, Nz - 1]]
    w_area = rf[1:]**2 - rf[:-1]**2
    p_mem = np.sum(p_top * w_area) / np.sum(w_area)
    Q = np.pi * R_c**2
    return dict(Z=p_mem / Q,
                Z_ref=_referenz_diskret(rf, zf, i_a, j0, R_c, L, omega, rho, mu),
                N=N)


def _referenz_diskret(rf, zf, i_a, j0, R_c, L, omega, rho, mu):
    """Reynolds-Zelle + Rohr auf demselben Gitter wie die Zelle."""
    jw = 1j * omega * rho
    # Schlitz: Leitwert K je Breite aus dem diskreten z-Profil
    z = zf[j0:]
    zc, dz = 0.5 * (z[1:] + z[:-1]), np.diff(z)
    n = zc.size
    A = np.zeros((n, n), complex)
    for j in range(n):
        zp = zc[j + 1] if j + 1 < n else 2 * z[-1] - zc[j]
        zm = zc[j - 1] if j > 0 else 2 * z[0] - zc[j]
        Dp, Dm = zp - zc[j], zc[j] - zm
        cp, cm = 2.0 / (Dp * (Dp + Dm)), 2.0 / (Dm * (Dp + Dm))
        A[j, j] += jw + mu * (cp + cm)
        if j + 1 < n:
            A[j, j + 1] -= mu * cp
        else:
            A[j, j] += mu * cp
        if j > 0:
            A[j, j - 1] -= mu * cm
        else:
            A[j, j] += mu * cm
    K = np.sum(np.linalg.solve(A, np.ones(n, complex)) * dz)
    # Rohr: Impedanz je Länge aus dem diskreten r-Profil
    r = rf[:i_a + 1]
    rc, dr = 0.5 * (r[1:] + r[:-1]), np.diff(r)
    m = rc.size
    B = np.zeros((m, m), complex)
    for i in range(m):
        B[i, i] += jw
        c = r[i + 1] / (((rc[i + 1] if i + 1 < m else r[i + 1]) - rc[i])
                        * rc[i] * dr[i])
        B[i, i] += mu * c
        if i + 1 < m:
            B[i, i + 1] -= mu * c
        if i > 0:
            c = r[i] / ((rc[i] - rc[i - 1]) * rc[i] * dr[i])
            B[i, i] += mu * c
            B[i, i - 1] -= mu * c
    w = np.linalg.solve(B, np.ones(m, complex))
    Z_rohr = L / np.sum(w * 2 * np.pi * rc * dr)
    # radiale FV-Reynolds-Gleichung a..R_c, p(a) = 0, Membran v = 1
    r = rf[i_a:]
    rc, dr = 0.5 * (r[1:] + r[:-1]), np.diff(r)
    k = rc.size
    C = np.zeros((k, k), complex)
    for i in range(k):
        if i + 1 < k:
            c = K * r[i + 1] / (rc[i + 1] - rc[i])
            C[i, i] += c
            C[i, i + 1] -= c
        c = K * r[i] / (rc[i] - (rc[i - 1] if i > 0 else r[0]))
        C[i, i] += c
        if i > 0:
            C[i, i - 1] -= c
    p = np.linalg.solve(C, rc * dr)
    Z_film = np.sum(p * 2 * np.pi * rc * dr) / (np.pi * R_c**2)**2
    return Z_film + Z_rohr


ZELLEN = (3.0, 5.0, 10.0)      # R_c/a der drei Bestimmungszellen


def belaege(a, h, omega):
    """Film- und Rohrbelag des Modells: Z'_f = 1/(2π·a·K) (Film am
    Lochrand je radiale Länge), Z'_t = jωρ0/(π a² F_v) (Rohr je Länge)."""
    al = 0.5 * h * np.sqrt(1j * omega * RHO0 / MU_AIR)
    K = h / (1j * omega * RHO0) * (1.0 - np.tanh(al) / al)
    Zt1 = MicrophoneCapsule._hole_impedance(np.array([omega]), a, 1.0, 1,
                                            end_correction=False)[0]
    return 1.0 / (2 * np.pi * a * K), Zt1


def dreitor(h_a, a_delta, fein=1.5, a=0.5e-3, zellen=ZELLEN):
    """Dreitor der Mündung aus den Zellen ``zellen`` (R_c/a; drei bestimmen
    es, mehr werden ausgeglichen). Rückgabe (z, c, ω): z = (z_ff, z_fm,
    z_mm) komplex, c = die sechs Zusatzlängen der Tabelle
    (c_ff^R, c_ff^X, c_fm^R, c_fm^X, c_mm^R, c_mm^X), s.
    MicrophoneCapsule._muendung_spalt."""
    h = h_a * a
    omega = 2.0 * MU_AIR / (RHO0 * (a / a_delta)**2)
    q, dZ = [], []
    for Rca in zellen:
        s = loese_zelle(a, h, Rca * a, 4.0 * a, omega, fein=fein)
        q.append(1.0 / Rca**2)
        dZ.append(s["Z"] - s["Z_ref"])
    q = np.array(q)
    M = np.c_[(1 - q)**2, 2 * q * (1 - q), q**2].astype(complex)
    z = np.linalg.lstsq(M, np.array(dZ), rcond=None)[0]
    Zf1, Zt1 = belaege(a, h, omega)
    basis = (h * Zf1, a * a / h * Zt1, a * Zt1)
    c = []
    for zk, bk in zip(z, basis):
        c += [zk.real / bk.real, zk.imag / bk.imag]
    return z, np.array(c), omega


TAB_H_A = (0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0,
           1.5, 2.0)
TAB_A_DELTA = (0.3, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _tabellenpunkt(arg):
    h_a, a_d = arg
    return dreitor(h_a, a_d, fein=1.5)[1]


def tabelle(prozesse=4):
    """Rechnet die Tabelle MicrophoneCapsule._MUENDUNG_TAB neu, Form
    (h/a, a/δ, 6). Gut zehn Minuten auf vier Kernen; BLAS dabei einfädig
    halten, sonst überbuchen sich die Prozesse:

        OMP_NUM_THREADS=1 PYTHONPATH=. python tests/stokes_zelle.py
    """
    from multiprocessing import Pool
    args = [(ha, ad) for ha in TAB_H_A for ad in TAB_A_DELTA]
    with Pool(prozesse) as P:
        res = P.map(_tabellenpunkt, args)
    return np.array(res).reshape(len(TAB_H_A), len(TAB_A_DELTA), 6)


def tabelle_literal(C):
    """Die Tabelle als Quelltext für microphone_capsule.py."""
    zeilen = ["    _MUENDUNG_TAB = ("]
    for ih, ha in enumerate(TAB_H_A):
        zeilen.append(f"        (   # h/a = {ha:g}")
        for k in range(6):
            zeilen.append('            "' + " ".join(
                f"{0.0 if abs(v) < 5e-4 else v:.3f}"
                for v in C[ih, :, k]) + '",')
        zeilen.append("        ),")
    zeilen.append("    )")
    return "\n".join(zeilen)


if __name__ == "__main__":
    import sys
    C = tabelle()
    if len(sys.argv) > 1:
        np.save(sys.argv[1], C)
    print(tabelle_literal(C))
