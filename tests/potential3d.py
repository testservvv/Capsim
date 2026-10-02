"""3D-Gegenlöser: Front-Rück-Weg im Tiefton für nicht achsensymmetrische
Montagen (Offene Punkte 9, Gegenprobe 71).

Das Modell rechnet den äußeren Front-Rück-Transfer achsensymmetrisch
(Kugel, Sphäroid, BEM mit KOAXIALEM Körper). Die U87 ist aber seiten-
besprochen: der Körper sitzt UNTER der Kapsel, quer zu ihrer Achse. Für
den Tiefton-Grenzfall (ka → 0) genügt die Potentialströmung: einfallend
U·x, Störpotential φ_s mit ∂φ_s/∂n = −U·n_x auf den starren Flächen;
der wirksame Front-Rück-Weg ist L = (⟨φ⟩_hinten − ⟨φ⟩_vorn)/U, gemittelt
über die Membran mit dem Gewicht der Grundmode (1 − r²/a²).

Direkte BEM mit konstanten Viereckelementen (2×2-Gauß), Kollokation in
den Schwerpunkten, Selbstglied der Einfachschicht über die flächengleiche
Scheibe:  ½·φ_i − Σ_j φ_j ∫∂G/∂n_y dS = −Σ_j q_j ∫G dS,  G = 1/(4π r).

Geprüft (``python tests/potential3d.py``): Kugel gegen die exakte
Lösung 1,5·D (0,5 %), freie Kapselscheibe und Körper koaxial dahinter
gegen das achsensymmetrische BEM des Modells (≤ 1,2 %). Ergebnis für
die K67 (34 × 12,28 mm, Membran 27,2 mm, Körper Ø 56 mm): mit dem Körper
10–15 mm unter dem Kapselrand 37,4–38,6 mm statt 34,7 mm frei. Für die
Laufzeitfrage der K67 ist das nicht die Erklärung — die Kapsel hat ihre
Niere auch frei (Offene Punkte 9).
"""
import numpy as np

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)          # 2×2-Gauß


def _elemente(fn, u_kanten, v_kanten):
    """Viereckelemente einer Parameterfläche fn(u, v) -> (Punkt, Normale,
    Jacobi): Gaußpunkte (n,4,3), Gewichte (n,4), Schwerpunkte, Normalen,
    Flächen."""
    u0, v0 = np.meshgrid(u_kanten[:-1], v_kanten[:-1], indexing="ij")
    u1, v1 = np.meshgrid(u_kanten[1:], v_kanten[1:], indexing="ij")
    u0, v0, u1, v1 = (a.ravel() for a in (u0, v0, u1, v1))
    qp, qw = [], []
    for a in GP:
        for b in GP:
            p, _, jac = fn(0.5 * (u0 + u1) + 0.5 * (u1 - u0) * a,
                           0.5 * (v0 + v1) + 0.5 * (v1 - v0) * b)
            qp.append(p)
            qw.append(0.25 * (u1 - u0) * (v1 - v0) * jac)
    qp, qw = np.stack(qp, axis=1), np.stack(qw, axis=1)
    pc, nc, _ = fn(0.5 * (u0 + u1), 0.5 * (v0 + v1))
    return qp, qw, pc, nc, qw.sum(axis=1)


def kugel(R, n):
    def fn(u, v):
        p = R * np.stack([np.cos(v), np.sin(v) * np.cos(u),
                          np.sin(v) * np.sin(u)], -1)
        return p, p / R, R * R * np.sin(v)
    return [_elemente(fn, np.linspace(0, 2 * np.pi, 2 * n + 1),
                      np.linspace(0, np.pi, n + 1))]


def zylinder(achse, R, kanten, n_umf, n_ring):
    """Geschlossener Kreiszylinder um die Achse 'x' oder 'z' durch den
    Ursprung; ``kanten`` sind die Mantelteilungen längs der Achse (Enden
    = Stirnflächen). Stirnflächen in Ringen, zum Rand hin feiner."""
    e = {"x": np.array([1.0, 0, 0]), "z": np.array([0, 0, 1.0])}[achse]
    a1 = np.array([0, 1.0, 0]) if achse == "x" else np.array([1.0, 0, 0])
    a2 = np.cross(e, a1)

    def mantel(u, v):
        n = np.outer(np.cos(u), a1) + np.outer(np.sin(u), a2)
        return np.outer(v, e) + R * n, n, R * np.ones_like(u)
    teile = [_elemente(mantel, np.linspace(0, 2 * np.pi, n_umf + 1),
                       np.asarray(kanten))]
    rr = R * (1 - (1 - np.linspace(0, 1, n_ring + 1)) ** 1.5)
    for seite, xk in ((-1, kanten[0]), (+1, kanten[-1])):
        for k in range(n_ring):
            ns = max(6, int(round(n_umf * 0.5 * (rr[k] + rr[k + 1]) / R)))

            def kappe(u, v, xk=xk, seite=seite):
                p = (xk * e[None] + np.outer(v * np.cos(u), a1)
                     + np.outer(v * np.sin(u), a2))
                return p, np.repeat((seite * e)[None], u.size, 0), v
            teile.append(_elemente(kappe, np.linspace(0, 2 * np.pi, ns + 1),
                                   np.array([rr[k], rr[k + 1]])))
    return teile


def potential(teile, U=np.array([1.0, 0, 0])):
    """Gesamtpotential U·x + φ_s in den Elementschwerpunkten."""
    qp, qw, pc, nc, A = (np.concatenate([t[i] for t in teile])
                         for i in range(5))
    N = pc.shape[0]
    H, G = np.zeros((N, N)), np.zeros((N, N))
    for i0 in range(0, N, 400):
        d = qp[None] - pc[i0:i0 + 400, None, None, :]
        r = np.linalg.norm(d, axis=-1)
        r = np.where(r == 0, np.inf, r)
        G[i0:i0 + 400] = np.sum(qw[None] / (4 * np.pi * r), axis=-1)
        H[i0:i0 + 400] = np.sum(qw[None] * np.einsum("bnqk,nk->bnq", d, nc)
                                / (4 * np.pi * r ** 3), axis=-1)
    i = np.arange(N)
    G[i, i] = np.sqrt(A / np.pi) / 2.0
    H[i, i] = 0.0
    phi = np.linalg.solve(0.5 * np.eye(N) + H, G @ (nc @ U))
    return pc, nc, A, phi + pc @ U


def front_rueck_weg(teile, L_kapsel, a_mem):
    """Wirksamer Front-Rück-Weg [m] einer Kapsel mit Stirnflächen bei
    x = ∓L/2 (Membranradius a_mem, Gewicht der Grundmode)."""
    pc, nc, A, phi = potential(teile)
    r = np.hypot(pc[:, 1], pc[:, 2])
    w = A * np.clip(1 - (r / a_mem) ** 2, 0, None)
    vorn = (nc[:, 0] < -0.99) & np.isclose(pc[:, 0], -L_kapsel / 2)
    hinten = (nc[:, 0] > 0.99) & np.isclose(pc[:, 0], L_kapsel / 2)
    return (np.sum((phi * w)[hinten]) / w[hinten].sum()
            - np.sum((phi * w)[vorn]) / w[vorn].sum())


def k67(spalt=None, lage="unten", R_k=17e-3, L_k=12.28e-3, a_mem=13.6e-3,
        R_b=28e-3, L_b=80e-3, n_umf=64, n_ring=10):
    """K67-Kopf (Zylinder 34 × 12,28 mm) frei oder mit Körper (Ø 56 mm)
    im Abstand ``spalt``: 'unten' quer zur Achse (seitenbesprochen),
    'hinten' koaxial hinter der Rückmembran (Endbesprechung)."""
    s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 7)))
    teile = zylinder("x", R_k, -L_k / 2 + L_k * s, n_umf, n_ring)
    if spalt is not None:
        g = 1 - np.cos(np.linspace(0, np.pi / 2, 21))
        if lage == "hinten":
            teile += zylinder("x", R_b, L_k / 2 + spalt + L_b * g,
                              n_umf, n_ring)
        else:
            teile += zylinder("z", R_b, -(R_k + spalt) - L_b * g[::-1],
                              n_umf, n_ring)
    return front_rueck_weg(teile, L_k, a_mem)


if __name__ == "__main__":
    mm = 1e-3
    pc, nc, A, phi = potential(kugel(17 * mm, 36))
    pol = np.abs(pc[:, 0]) > 0.95 * 17 * mm
    print(f"Kugel: φ/(1,5·x) an den Polen {np.mean(phi[pol] / (1.5 * pc[pol, 0])):.4f}"
          " (exakt 1)")
    print(f"K67 frei: {k67() / mm:.2f} mm (Modell-BEM 100 Hz: 34,38 mm)")
    for g, soll in ((5, 17.42), (15, 23.93), (30, 29.26)):
        print(f"Körper koaxial, Spalt {g} mm: {k67(g * mm, 'hinten') / mm:.2f} mm"
              f" (Modell-BEM: {soll} mm)")
    for g in (5, 10, 15, 25, 40):
        print(f"Körper unter der Kapsel, Spalt {g} mm: "
              f"{k67(g * mm) / mm:.2f} mm (Körper 80 mm), "
              f"{k67(g * mm, L_b=200 * mm) / mm:.2f} mm (200 mm)")
