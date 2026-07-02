# -*- coding: utf-8 -*-
"""
Capsim — Physikalisches Modell einer Kondensatormikrofonkapsel
===============================================================

Schritt 1 des Projekts: die reine Berechnungsklasse ``MicrophoneCapsule``
(Lumped-Element-Modell / elektroakustisches Ersatzschaltbild).
Die GUI mit Plots und Datenexport folgt in einem späteren Schritt.

Modellierungsansatz
-------------------
Die Kapsel wird in der elektroakustischen Impedanz-Analogie beschrieben:

    Schalldruck p        ~  elektrische Spannung   [Pa]
    Volumenfluss q (U)   ~  elektrischer Strom     [m^3/s]
    akust. Impedanz Z_A  ~  elektrische Impedanz   [Pa*s/m^3]

Das gesamte akustische Netzwerk (Strahlung -> Gewebe -> (ggf. vordere
Backplate) -> Membran -> Luftspalt -> Backplate-Löcher -> Gewebe ->
Laufzeitglied -> Hohlraum -> Hohlraumlöcher -> rückwärtiges Schallfeld)
wird als Kette von Zweitoren (ABCD-/Kettenmatrizen) aufgebaut und für
jede Frequenz gelöst. Die beiden "Quellen" sind der Schalldruck am
vorderen Einlass und der Schalldruck an den rückwärtigen Einlassöffnungen;
ihre Phasenbeziehung hängt vom Schalleinfallswinkel ab und erzeugt so die
Richtcharakteristik (Druckgradienten-Prinzip).

Konventionen der Kettenmatrix:  [p_in; q_in] = T * [p_out; q_out]
    Serienimpedanz Z:   T = [[1, Z], [0, 1]]
    Shunt-Admittanz Y:  T = [[1, 0], [Y, 1]]
    Leitung (Länge L):  T = [[cosh(gL), Zc*sinh(gL)],
                             [sinh(gL)/Zc, cosh(gL)]]

Alle Größen in SI-Einheiten, sofern nicht anders angegeben.
"""

from functools import reduce

import numpy as np

try:
    # Für die exakte thermoviskose Rohrimpedanz (Zwikker–Kosten) werden
    # Besselfunktionen mit komplexem Argument benötigt; für die Beugung
    # am Kapselkörper sphärische Besselfunktionen.
    from scipy.special import jv as _besselj
    from scipy.special import spherical_jn as _sph_jn
    from scipy.special import spherical_yn as _sph_yn

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — Fallback auf Näherungsformeln
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Stoffwerte Luft bei 20 °C, 1013 hPa
# ---------------------------------------------------------------------------
RHO0 = 1.204          # Dichte der Luft                      [kg/m^3]
C_AIR = 343.4         # Schallgeschwindigkeit                [m/s]
MU_AIR = 1.82e-5      # dynamische Viskosität                [Pa*s]
P_ATM = 101325.0      # statischer Luftdruck                 [Pa]
GAMMA = 1.402         # Adiabatenexponent                    [-]
PRANDTL = 0.71        # Prandtl-Zahl                         [-]
EPS0 = 8.8541878128e-12  # elektrische Feldkonstante         [F/m]


class MicrophoneCapsule:
    """Lumped-Element-Modell einer Kondensatormikrofonkapsel.

    Die Klasse nimmt alle physikalischen Parameter der Kapsel auf und
    stellt Methoden für Frequenzgang (:meth:`frequency_response`) und
    Richtdiagramm (:meth:`directivity`) bereit. Beide geben reine
    NumPy-Arrays zurück, damit eine spätere GUI sie direkt plotten kann.

    Parameter
    ---------
    Membran
        membrane_material : str oder dict
            Materialname aus :attr:`MATERIALS` (z. B. ``"PET"``, ``"nickel"``)
            oder eigenes dict ``{"rho": ..., "E": ..., "nu": ...}``.
        membrane_resonance_hz : float oder None
            Resonanzfrequenz der Membran [Hz]. Ist sie angegeben, wird die
            akustische Nachgiebigkeit der Membran so gewählt, dass genau
            diese Resonanz entsteht (die Vorspannung dient dann als
            Konsistenz-Check, s. :meth:`summary`). Bei ``None`` wird die
            Resonanz aus Vorspannung + Biegesteifigkeit berechnet.
        membrane_diameter : float   — Membrandurchmesser [m]
        membrane_thickness : float  — Membrandicke [m]
        membrane_tension : float    — mechanische Vorspannung [N/m]

    Backplate-System
        air_gap : float             — Luftspalt Membran/Backplate [m]
        backplate_diameter : float  — Backplate-Durchmesser [m]
        backplate_thickness : float — Backplate-Dicke [m]
        bias_voltage : float        — Polarisationsspannung [V]
        architecture : str          — ``"single"`` (eine Backplate hinter der
                                      Membran) oder ``"dual"`` (symmetrische
                                      Backplates vor UND hinter der Membran,
                                      Gegentakt-Wandlung).
        n_through_holes, through_hole_diameter
            Anzahl/Durchmesser der Durchgangslöcher in der Backplate.
            Die Durchgangslöcher sind der einzige Weg von der Membran-
            rückseite in die dahinterliegende Struktur: ``0`` verschließt
            die Backplate — die Kapsel ist dann hermetisch dicht
            (Druckempfänger), egal was dahinter montiert ist.
        n_blind_holes, blind_hole_diameter, blind_hole_depth
            Anzahl/Durchmesser/Tiefe der Blindlöcher (Sacklöcher) auf der
            Membranseite der Backplate. ``blind_hole_depth=None`` ->
            halbe Backplate-Dicke.

    Akustische Netzwerke & Rückseite
        rear_network_enabled : bool
            ``True``: hinter der Backplate sitzt die rückwärtige Baugruppe
            (Gewebe -> Laufzeitglied -> Hohlraum -> Einlasslöcher).
            ``False``: keine Baugruppe — die Durchgangslöcher der Backplate
            münden (durch das rückwärtige Gewebe) DIREKT ins rückwärtige
            Schallfeld; die Kapsel wird zum einfachen Gradientenempfänger
            mit der äußeren Wegdifferenz Spalt + Backplate-Dicke.
        delay_length : float
            Länge des akustischen Laufzeitglieds hinter der Membran [m].
            Modelliert als (schwach verlustbehaftete) akustische Leitung;
            Laufzeit tau = L / c.
        cavity_length, cavity_wall_thickness
            Zylindrischer Hohlraum hinter dem Laufzeitglied (zur Backplate
            offen, nach hinten geschlossen). Innendurchmesser = Backplate-
            Durchmesser (Annahme). Die Wandstärke bestimmt die Länge der
            Einlasslöcher.
        cavity_hole_position : str
            ``"circumference"`` — Löcher radial in der Zylinderwand,
            ``"end"``           — Löcher axial in der hinteren Stirnfläche.
        n_cavity_holes : int
            Anzahl der Hohlraumlöcher. ``0`` -> Rückseite geschlossen,
            die Kapsel wird zum reinen Druckempfänger (Kugelcharakteristik).
        cavity_hole_diameter : float
        cavity_hole_axial_position : float
            Axialer Abstand der Umfangslöcher vom Hohlraumeingang [m]
            (nur für ``"circumference"`` relevant; Löcher gleichmäßig am
            Umfang verteilt).
        fabric_front_rayl : float
            Strömungswiderstand des Gewebes VOR der Membran [Pa*s/m] (Rayl).
        fabric_rear_rayl : float
            Strömungswiderstand des Gewebes HINTER der Backplate [Pa*s/m].

    Gehäuse & Beugung (Druckstau)
        body_diameter : float oder None
            Durchmesser des kugelförmigen Ersatz-Gehäuses für die Beugungs-
            rechnung [m]. ``None`` -> 1.2 * max(Membran-, Backplate-Ø).
        include_diffraction : bool
            ``True`` (Standard): Druckstau/Abschattung am Kapselkörper wird
            über die exakte Streuung der ebenen Welle an einer starren
            Kugel berechnet (s. :meth:`_diffraction_factors`) — dadurch
            richtet auch ein reiner Druckempfänger zu hohen Frequenzen hin
            und die Freifeldempfindlichkeit steigt frontal um bis zu +6 dB.
            ``False``: einfache ebene-Welle-Phasen (nur zu Vergleichs-
            zwecken; ohne SciPy automatisch dieser Fallback).
    """

    # Membranmaterialien: Dichte rho [kg/m^3], E-Modul E [Pa],
    # Poissonzahl nu [-] (für die Biegesteifigkeit).
    MATERIALS = {
        "pet":       {"rho": 1390.0, "E": 4.9e9,   "nu": 0.37},  # PET/Mylar
        "mylar":     {"rho": 1390.0, "E": 4.9e9,   "nu": 0.37},
        "nickel":    {"rho": 8908.0, "E": 200.0e9, "nu": 0.31},
        "titan":     {"rho": 4506.0, "E": 116.0e9, "nu": 0.32},
        "aluminium": {"rho": 2700.0, "E": 70.0e9,  "nu": 0.35},
        "gold":      {"rho": 19320.0, "E": 79.0e9, "nu": 0.42},
    }

    # Interne Materialgüte der Membran (Verlustfaktor der Folie selbst;
    # die dominante Dämpfung kommt aus dem Luftspalt, dieser Wert stellt
    # nur numerische Gutartigkeit ohne Spaltdämpfung sicher).
    _Q_MEMBRANE_INTERNAL = 100.0

    def __init__(
        self,
        # --- Membran -------------------------------------------------------
        membrane_material="PET",
        membrane_resonance_hz=8000.0,
        membrane_diameter=22e-3,
        membrane_thickness=6e-6,
        membrane_tension=400.0,
        # --- Backplate-System ---------------------------------------------
        air_gap=40e-6,
        backplate_diameter=20e-3,
        backplate_thickness=3e-3,
        bias_voltage=60.0,
        architecture="single",
        n_through_holes=60,
        through_hole_diameter=1.0e-3,
        n_blind_holes=30,
        blind_hole_diameter=1.2e-3,
        blind_hole_depth=None,
        # --- Akustische Netzwerke & Rückseite -------------------------------
        rear_network_enabled=True,
        delay_length=3e-3,
        cavity_length=12e-3,
        cavity_wall_thickness=1.5e-3,
        cavity_hole_position="circumference",
        n_cavity_holes=200,
        cavity_hole_diameter=0.2e-3,
        cavity_hole_axial_position=6e-3,
        fabric_front_rayl=10.0,
        fabric_rear_rayl=25.0,
        # --- Gehäuse & Beugung ----------------------------------------------
        body_diameter=None,
        include_diffraction=True,
    ):
        # ------------------------- Membran ---------------------------------
        if isinstance(membrane_material, dict):
            mat = membrane_material
        else:
            key = str(membrane_material).strip().lower()
            if key not in self.MATERIALS:
                raise ValueError(
                    f"Unbekanntes Membranmaterial '{membrane_material}'. "
                    f"Verfügbar: {sorted(self.MATERIALS)} oder dict mit rho/E/nu."
                )
            mat = self.MATERIALS[key]
        self.mat_rho = float(mat["rho"])
        self.mat_E = float(mat["E"])
        self.mat_nu = float(mat.get("nu", 0.35))
        self.membrane_material = membrane_material

        self.f_res_user = membrane_resonance_hz
        self.a_mem = 0.5 * float(membrane_diameter)      # Membranradius [m]
        self.t_mem = float(membrane_thickness)
        self.tension = float(membrane_tension)
        if self.a_mem <= 0 or self.t_mem <= 0 or self.tension <= 0:
            raise ValueError("Membrangeometrie und Vorspannung müssen > 0 sein.")

        # ---------------------- Backplate-System ---------------------------
        self.h_gap = float(air_gap)
        self.a_bp = 0.5 * float(backplate_diameter)
        self.t_bp = float(backplate_thickness)
        self.u_bias = float(bias_voltage)

        arch = str(architecture).strip().lower()
        if arch.startswith("dual"):
            self.architecture = "dual"
        elif arch.startswith("single"):
            self.architecture = "single"
        else:
            raise ValueError("architecture muss 'single' oder 'dual' sein.")
        # Anzahl der Backplates (Gegentakt bei 'dual')
        self.n_bp = 2 if self.architecture == "dual" else 1

        self.n_th = int(n_through_holes)
        self.r_th = 0.5 * float(through_hole_diameter)
        if self.n_th < 0:
            raise ValueError("Anzahl der Durchgangslöcher darf nicht negativ sein.")
        if self.n_th > 0 and self.r_th <= 0:
            raise ValueError("Durchgangslochdurchmesser muss > 0 sein.")
        if self.n_th == 0 and self.architecture == "dual":
            raise ValueError(
                "Dual-Architektur ohne Durchgangslöcher: die vordere "
                "Backplate würde die Membran vollständig vom Schallfeld "
                "isolieren."
            )

        self.n_bh = int(n_blind_holes)
        self.r_bh = 0.5 * float(blind_hole_diameter)
        if blind_hole_depth is None:
            blind_hole_depth = 0.5 * self.t_bp
        self.d_bh = float(blind_hole_depth)
        if self.n_bh > 0 and not (0 < self.d_bh < self.t_bp):
            raise ValueError("Blindlochtiefe muss zwischen 0 und Backplate-Dicke liegen.")

        # ------------------ Akustische Netzwerke & Rückseite ----------------
        self.rear_network_enabled = bool(rear_network_enabled)
        self.l_delay = float(delay_length)
        self.l_cav = float(cavity_length)
        self.t_cav_wall = float(cavity_wall_thickness)

        pos = str(cavity_hole_position).strip().lower()
        if pos not in ("circumference", "end"):
            raise ValueError("cavity_hole_position muss 'circumference' oder 'end' sein.")
        self.cavity_hole_position = pos
        self.n_ch = int(n_cavity_holes)
        self.r_ch = 0.5 * float(cavity_hole_diameter)
        self.x_ch = float(cavity_hole_axial_position)
        if self.n_ch > 0 and pos == "circumference":
            if not (0.0 < self.x_ch <= self.l_cav):
                raise ValueError(
                    "cavity_hole_axial_position muss im Bereich (0, cavity_length] liegen."
                )

        self.rayl_front = float(fabric_front_rayl)
        self.rayl_rear = float(fabric_rear_rayl)

        self.body_diameter = (None if body_diameter is None
                              else float(body_diameter))
        self.include_diffraction = bool(include_diffraction)

        # ------------------------ abgeleitete Größen ------------------------
        self._derive_parameters()

    # ======================================================================
    # Abgeleitete Lumped-Element-Parameter
    # ======================================================================
    def _derive_parameters(self):
        a, t = self.a_mem, self.t_mem
        self.S_mem = np.pi * a**2                 # Membranfläche [m^2]
        self.S_bp = np.pi * self.a_bp**2          # Backplate-Fläche [m^2]
        rho_s = self.mat_rho * t                  # Flächendichte [kg/m^2]

        # ------------------------------------------------------------------
        # AKUSTISCHE MASSE DER MEMBRAN
        # Für eine am Rand eingespannte, gleichmäßig druckbelastete Membran
        # ist das Auslenkungsprofil näherungsweise parabolisch:
        #     w(r) = w0 * (1 - r^2/a^2)
        # Gleichsetzen der kinetischen Energie des realen Profils mit der
        # eines Ersatzkolbens gleicher VOLUMEN-Schnelle liefert die
        # effektive akustische Masse (Standard-Lumped-Element-Resultat):
        #     M_A = (4/3) * rho_s / (pi * a^2)        [kg/m^4]
        # ------------------------------------------------------------------
        self.M_A_mem = (4.0 / 3.0) * rho_s / self.S_mem

        # ------------------------------------------------------------------
        # AKUSTISCHE NACHGIEBIGKEIT DER MEMBRAN
        # 1) Anteil der Vorspannung T [N/m] (Beranek):
        #        C_T = pi * a^4 / (8 * T)             [m^5/N = m^3/Pa]
        #    (statische Durchbiegung w0 = p*a^2/(4T), Volumen = p*pi*a^4/(8T))
        # 2) Anteil der Biegesteifigkeit der Folie (eingespannte Platte):
        #        D   = E * t^3 / (12 * (1 - nu^2))    [N*m]
        #        C_B = pi * a^6 / (192 * D)
        # Beide Federn wirken parallel (Steifigkeiten addieren sich):
        #        1/C_phys = 1/C_T + 1/C_B
        # ------------------------------------------------------------------
        C_T = np.pi * a**4 / (8.0 * self.tension)
        D_plate = self.mat_E * t**3 / (12.0 * (1.0 - self.mat_nu**2))
        C_B = np.pi * a**6 / (192.0 * D_plate)
        self.C_A_phys = 1.0 / (1.0 / C_T + 1.0 / C_B)

        # Aus Vorspannung/Steifigkeit resultierende Resonanz (Diagnose):
        self.f_res_from_tension = 1.0 / (
            2.0 * np.pi * np.sqrt(self.M_A_mem * self.C_A_phys)
        )

        # Ist eine Soll-Resonanzfrequenz vorgegeben, wird die Nachgiebigkeit
        # so skaliert, dass f_res exakt getroffen wird (die Vorspannung
        # bleibt als Plausibilitäts-Referenz in summary() sichtbar).
        if self.f_res_user is not None:
            w0 = 2.0 * np.pi * float(self.f_res_user)
            self.C_A_mem = 1.0 / (w0**2 * self.M_A_mem)
            self.f_res = float(self.f_res_user)
        else:
            self.C_A_mem = self.C_A_phys
            self.f_res = self.f_res_from_tension

        # ------------------------------------------------------------------
        # ELEKTROSTATIK: ELEKTRODENGEOMETRIE, ARBEITSPUNKT UND PULL-IN
        #
        # Die Backplate wirkt NICHT mit ihrer vollen Fläche als Elektrode:
        #   * Durchgangslöcher tragen gar kein Feld,
        #   * über Blindlöchern beträgt der Feldweg h + Tiefe >> h — ihr
        #     Beitrag zu Kraft und Kapazität ist um (h/(h+d))^2..3 kleiner
        #     (genau so gehen sie unten in die Integrale ein).
        # Zusätzlich ist die elektrostatische Last VERTEILT und die Membran
        # am Rand eingespannt: alle Größen werden mit dem Auslenkungsprofil
        # phi(r) = 1 - r^2/a^2 (Galerkin-Ansatz, gleiche Mode wie M_A/C_A)
        # über die Elektrodenfläche integriert.
        #
        # Statischer Arbeitspunkt (konstante Spannung — der Bias-Widerstand
        # hält U0 statisch fest; das Luftpolster entweicht statisch durch
        # die Löcher und trägt NICHT):
        #     k_gen * w0 = F_es(w0),  F_es = (eps0 U0^2 / 2) Int phi/g(r)^2
        # mit lokalem Spalt g(r) = h - w0*phi(r) und der generalisierten
        # Membransteifigkeit k_gen = S^2/(4 C_A) (konsistent zu C_A).
        # PULL-IN: existiert keine stabile Lösung (oder ist die tangentiale
        # Steifigkeit k_gen - dF/dw0 <= 0), kollabiert die Membran — das
        # passiert bereits VOR dem Kleinsignal-Kriterium am Ruhespalt.
        #
        # FEDER-ERWEICHUNG am Arbeitspunkt ("spring softening"):
        #     k_neg = dF/dw0 = eps0 U0^2 Int phi^2/g(r)^3
        # akustisch: 1/C_eff = 1/C_A - 4*k_neg/S^2   (w0 -> V_disp: Faktor 2/S)
        # Bei Dual-Backplates heben sich die statischen Kräfte auf (w0 = 0),
        # die Erweichung beider Seiten addiert sich.
        # ------------------------------------------------------------------
        self.phi_th = self.n_th * np.pi * self.r_th**2 / self.S_bp
        self.phi_bh = self.n_bh * np.pi * self.r_bh**2 / self.S_bp
        if self.phi_th + self.phi_bh >= 0.9:
            raise ValueError(
                "Durchgangs- und Blindlöcher bedecken >= 90 % der "
                "Backplate — keine wirksame Elektrode mehr."
            )
        # Elektrodenrand in Modenkoordinate u = r^2/a_mem^2
        self._ub = min((self.a_bp / self.a_mem) ** 2, 1.0)
        self._k_gen = self.S_mem**2 / (4.0 * self.C_A_mem)

        eq = self._solve_static_deflection(self.u_bias)
        if eq is None:
            u_pi = self.pullin_voltage()
            raise ValueError(
                "Elektrostatischer Kollaps (Pull-in): die statische "
                "Anziehung der Backplate übersteigt die Rückstellkraft der "
                "Membran. Maximal stabile Polarisationsspannung für diese "
                f"Konfiguration: ca. {u_pi:.1f} V. Abhilfe: Spannung senken, "
                "Luftspalt vergrößern oder Membran steifer (höhere "
                "Resonanzfrequenz/Vorspannung). Hinweis: statisch trägt nur "
                "die Membran-Vorspannung — das Luftpolster entweicht durch "
                "die Löcher; gemessene Kapselresonanzen enthalten dagegen "
                "die Luftpolster-Steifigkeit und liegen deshalb unter der "
                "hier maßgeblichen Vorspannungs-Resonanz."
            )
        self.w0_static, k_neg_eq = eq
        self.h_min_static = self.h_gap - self.w0_static  # Restspalt Mitte

        inv_C_eff = 1.0 / self.C_A_mem - 4.0 * k_neg_eq / self.S_mem**2
        if inv_C_eff <= 0.0:  # durch Stabilitätsprüfung praktisch abgedeckt
            raise ValueError("Elektrostatischer Kollaps (Feder-Erweichung).")
        self.C_A_eff = 1.0 / inv_C_eff
        # relative Steifigkeitsreduktion durch die Vorspannung (Diagnose)
        self.softening_ratio = 1.0 - self.C_A_mem / self.C_A_eff
        # maximal stabile Polarisationsspannung (Diagnose)
        self.U_pullin = self.pullin_voltage()

        # kleine interne Membrandämpfung (s. _Q_MEMBRANE_INTERNAL)
        self.R_A_mem = (
            np.sqrt(self.M_A_mem / self.C_A_eff) / self._Q_MEMBRANE_INTERNAL
        )

        # ------------------------------------------------------------------
        # WANDLERKOEFFIZIENT UND RUHEKAPAZITÄT AM ARBEITSPUNKT
        # Betrieb mit konstanter Ladung (hochohmig): e = U0 * dC/C0.
        # Membranmode w = dw*phi(r) moduliert die Kapazität:
        #     dC = eps0 * I_F(w0) * dw,   I_F = Int phi/g(r)^2 dS
        # (gleiches Integral wie die Kraft). Mit dw = 2*V_disp/S folgt
        #     e = Theta * V_disp,  Theta = 2 U0 eps0 I_F / (S * C0).
        # Grenzfall w0=0, keine Löcher: Theta = U0/(h*S) * (2 - (b/a)^2) —
        # die frühere Flächengewichtung kappa. Die Lochporosität kürzt
        # sich in erster Ordnung (dC und C0 skalieren gleich), Blindlöcher
        # senken C0 geringfügig. Dual-Backplates: Gegentakt addiert beide
        # Seiten (bei w0 = 0 exakt Faktor 2). Streu-/Kabelkapazität und
        # Verstärkerlast sind nicht modelliert (Leerlauf an der Kapsel).
        # ------------------------------------------------------------------
        theta = 0.0
        C0_rear = None
        signs = (+1.0,) if self.architecture == "single" else (+1.0, -1.0)
        for sign in signs:
            I_F, _, I_C = self._electrode_integrals(sign * self.w0_static)
            C0 = EPS0 * I_C
            if C0_rear is None:
                C0_rear = C0
            theta += 2.0 * self.u_bias * EPS0 * I_F / (self.S_mem * C0)
        self.C_elec_0 = C0_rear   # Ruhekapazität der (hinteren) Backplate
        self._theta = theta

        # ------------------------------------------------------------------
        # LUFTSPALT: SQUEEZE-FILM-WIDERSTAND NACH ŠKVOR
        # Die Luft im dünnen Spalt muss beim Schwingen der Membran lateral
        # zu den Durchgangslöchern strömen (Poiseuille-Strömung zwischen
        # Platten). Mit dem Lochflächenanteil q = n * r_h^2 / a_bp^2 gilt
        # (Škvor 1967, Standardformel der Mikrofon-/MEMS-Literatur):
        #     R_gap = 12*mu / (n * pi * h^3) * B(q)
        #     B(q)  = q/2 - q^2/8 - ln(q)/4 - 3/8
        # Bemerkenswert: R_gap hängt nicht vom Zellradius ab, nur von h und q.
        # Die (kleine) Masse der lateral bewegten Spaltluft wird vernachlässigt.
        # ------------------------------------------------------------------
        if self.n_th > 0:
            q = self.n_th * self.r_th**2 / self.a_bp**2
            if not (0.0 < q < 1.0):
                raise ValueError(
                    f"Lochflächenanteil q={q:.3f} der Durchgangslöcher "
                    "muss in (0, 1) liegen."
                )
            B_q = q / 2.0 - q**2 / 8.0 - np.log(q) / 4.0 - 3.0 / 8.0
            self.R_A_gap = (12.0 * MU_AIR
                            / (self.n_th * np.pi * self.h_gap**3) * B_q)
        else:
            # Geschlossene Backplate: es existiert kein Strömungspfad zu
            # Löchern, also auch keine laterale Škvor-Strömung (die Formel
            # divergiert für q -> 0). Das Spaltvolumen wirkt als reine
            # Nachgiebigkeit direkt an der Membran.
            self.R_A_gap = None

        # ------------------------------------------------------------------
        # NACHGIEBIGKEIT DES SPALTVOLUMENS
        # Der Spalt ist dünner als die thermische Grenzschicht
        # (delta_t ~ 0.1 mm bei 1 kHz) -> Kompression verläuft ISOTHERM:
        #     C_gap = V / P_atm      (statt adiabatisch V/(rho0*c^2))
        # ------------------------------------------------------------------
        V_gap = self.S_bp * self.h_gap
        self.C_A_gap = V_gap / P_ATM

        # Blindloch-Volumen (isotherme Nachgiebigkeit, Löcher sind eng):
        V_blind = self.n_bh * np.pi * self.r_bh**2 * self.d_bh
        self.C_A_blind = V_blind / P_ATM if self.n_bh > 0 else 0.0

        # Ist die Rückseite akustisch offen (Gradientenempfänger)?
        # Die Durchgangslöcher sind der einzige Weg durch die Backplate:
        if self.n_th == 0:
            # Backplate geschlossen -> hermetisch dicht, unabhängig von
            # allem, was dahinter montiert ist.
            self.rear_open = False
        elif not self.rear_network_enabled:
            # Keine rückwärtige Baugruppe: die Durchgangslöcher münden
            # direkt ins rückwärtige Schallfeld.
            self.rear_open = True
        else:
            # Baugruppe vorhanden: offen nur über deren Einlasslöcher.
            self.rear_open = self.n_ch > 0 and self.r_ch > 0

        # ------------------------------------------------------------------
        # ÄUSSERE WEGDIFFERENZ FÜR DIE RICHTWIRKUNG
        # Eine ebene Welle aus Richtung theta erreicht die rückwärtigen
        # Einlasslöcher um  dt = d_ext * cos(theta) / c  später als die
        # Membranvorderseite. d_ext = axialer Abstand Vorderseite ->
        # Rückeinlass (Beugung um den Kapselkörper wird in dieser
        # 1.-Ordnung-Näherung vernachlässigt).
        # ------------------------------------------------------------------
        d_ax = self.h_gap + self.t_bp
        if self.rear_network_enabled:
            d_ax += self.l_delay
            if self.cavity_hole_position == "circumference":
                d_ax += self.x_ch
            else:
                d_ax += self.l_cav + self.t_cav_wall
        # axiale Einbautiefe der rückwärtigen Einlässe (für die Beugung)
        self.d_rear_ax = d_ax
        d = d_ax
        if self.architecture == "dual":
            # vordere Backplate verschiebt den vorderen Einlass nach vorn
            d += self.h_gap + self.t_bp
        self.d_ext = d

        # ------------------------------------------------------------------
        # ERSATZ-GEHÄUSE FÜR DIE BEUGUNGSRECHNUNG (starre Kugel)
        # Der Kapselkörper wird für Druckstau/Abschattung als starre Kugel
        # mit Radius R_body modelliert. Die Membran liegt als Kalotte am
        # vorderen Pol (Halbwinkel aus a_mem), die rückwärtigen Einlässe
        # als Ring beim Polarwinkel ihrer axialen Einbautiefe.
        # ------------------------------------------------------------------
        if self.body_diameter is None:
            self.R_body = 1.2 * max(self.a_mem, self.a_bp)
        else:
            self.R_body = 0.5 * self.body_diameter
            if self.R_body < max(self.a_mem, self.a_bp):
                raise ValueError(
                    "Gehäusedurchmesser muss mindestens so groß sein wie "
                    "Membran- und Backplate-Durchmesser."
                )
        # cos des Kalotten-Halbwinkels der Membran
        self._cap_cos = float(np.cos(np.arcsin(
            min(self.a_mem / self.R_body, 1.0))))
        # cos des Ring-Polarwinkels der rückwärtigen Einlässe; liegt die
        # Einbautiefe hinter dem Kugeläquivalent, wird auf den hinteren
        # Pol geklammert.
        self._ring_cos = float(np.clip(
            (self.R_body - self.d_rear_ax) / self.R_body, -1.0, 1.0))

    # ======================================================================
    # Elektrostatik: Integrale, statischer Arbeitspunkt, Pull-in
    # ======================================================================
    def _electrode_integrals(self, w0):
        """Elektrodenintegrale über das Membranprofil phi(r) = 1 - r²/a².

        In Modenkoordinate u = r²/a_mem² (dS = S_mem·du, Elektrode bis
        u <= ub) mit lokalem Spalt g = h - w0·phi:
            I_F = Int phi/g²  dS   (Kraft-/Kapazitätsmodulation)
            I_k = Int phi²/g³ dS   (negative Steifigkeit)
            I_C = Int 1/g     dS   (Ruhekapazität)
        Solide Elektrodenfläche wiegt mit (1 - phi_th - phi_bh); über
        Blindlöchern gilt der vergrößerte Feldweg g + Tiefe (Durchgangs-
        löcher tragen nichts). w0 < 0 beschreibt die von der Platte weg
        ausgelenkte Membran (vordere Backplate der Dual-Architektur).
        """
        u = np.linspace(0.0, self._ub, 401)
        v = 1.0 - u                               # Modenprofil phi
        g_s = self.h_gap - w0 * v                 # Spalt, solide Elektrode
        g_b = self.h_gap + self.d_bh - w0 * v     # Feldweg über Blindloch
        c_s = 1.0 - self.phi_th - self.phi_bh
        c_b = self.phi_bh
        S = self.S_mem
        I_F = S * np.trapezoid(c_s * v / g_s**2 + c_b * v / g_b**2, u)
        I_k = S * np.trapezoid(c_s * v**2 / g_s**3 + c_b * v**2 / g_b**3, u)
        I_C = S * np.trapezoid(c_s / g_s + c_b / g_b, u)
        return I_F, I_k, I_C

    def _static_residual(self, u_bias, w_grid):
        """k_gen·w0 − F_es(w0) für ein Array von Auslenkungen (vektorisiert)."""
        u = np.linspace(0.0, self._ub, 401)
        v = 1.0 - u
        w2 = np.atleast_1d(w_grid)[:, None]
        g_s = self.h_gap - w2 * v[None, :]
        g_b = self.h_gap + self.d_bh - w2 * v[None, :]
        c_s = 1.0 - self.phi_th - self.phi_bh
        I_F = self.S_mem * np.trapezoid(
            c_s * v / g_s**2 + self.phi_bh * v / g_b**2, u, axis=1)
        F = 0.5 * EPS0 * u_bias**2 * I_F
        return self._k_gen * np.atleast_1d(w_grid) - F

    def _solve_static_deflection(self, u_bias):
        """Statischer Arbeitspunkt der Membran unter Polarisationsspannung.

        Rückgabe: (w0_statisch, k_neg_gesamt) oder ``None`` bei Pull-in.
        Gesucht wird die ERSTE Nullstelle von k_gen·w0 − F_es(w0) (das
        stabile Gleichgewicht); danach wird die tangentiale Stabilität
        k_gen − k_neg(w0) > 0 geprüft. Dual: statische Kräfte symmetrisch
        -> w0 = 0, aber beide Seiten erweichen.
        """
        if u_bias <= 1e-9:
            return 0.0, 0.0
        if self.architecture == "dual":
            _, I_k, _ = self._electrode_integrals(0.0)
            k_neg = 2.0 * EPS0 * u_bias**2 * I_k
            return (0.0, k_neg) if self._k_gen > k_neg else None
        w_max = 0.98 * self.h_gap
        grid = np.linspace(0.0, w_max, 240)
        res = self._static_residual(u_bias, grid)
        idx = None
        for i in range(1, grid.size):
            if res[i - 1] < 0.0 <= res[i]:
                idx = i
                break
        if idx is None:
            return None                            # kein Gleichgewicht
        lo, hi = grid[idx - 1], grid[idx]
        for _ in range(60):                        # Bisektion
            mid = 0.5 * (lo + hi)
            if self._static_residual(u_bias, mid)[0] < 0.0:
                lo = mid
            else:
                hi = mid
        w0 = 0.5 * (lo + hi)
        _, I_k, _ = self._electrode_integrals(w0)
        k_neg = EPS0 * u_bias**2 * I_k
        if self._k_gen <= k_neg:                   # tangential instabil
            return None
        return w0, k_neg

    def pullin_voltage(self, u_max=20000.0):
        """Maximal stabile Polarisationsspannung (Pull-in) per Bisektion."""
        hi = max(2.0 * self.u_bias, 100.0)
        while self._solve_static_deflection(hi) is not None:
            hi *= 2.0
            if hi > u_max:
                return float("inf")
        lo = 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self._solve_static_deflection(mid) is None:
                hi = mid
            else:
                lo = mid
        return lo

    # ======================================================================
    # Elementare akustische Impedanzen
    # ======================================================================
    @staticmethod
    def _hole_impedance(omega, radius, length, count,
                        end_correction=True, radiates=False):
        """Akustische Impedanz von ``count`` parallelen Zylinderlöchern.

        THERMOVISKOSE ROHRIMPEDANZ (Zwikker–Kosten / Crandall)
        ------------------------------------------------------
        Für ein zylindrisches Rohr (Radius r, Länge L) mit oszillierender
        Strömung lautet die exakte Lösung der linearisierten Navier-Stokes-
        Gleichung mit Haftbedingung an der Wand:

            k_v = sqrt(-j * omega * rho0 / mu)     (viskose Wellenzahl)
            F_v = 1 - 2*J1(k_v r) / (k_v r * J0(k_v r))
            Z   = j * omega * rho0 * L / (pi r^2 * F_v)

        Grenzfälle (zur Plausibilisierung, im Fallback auch verwendet):
          * omega -> 0 :  Z -> 8*mu*L/(pi r^4) + j*omega*(4/3)*rho0*L/(pi r^2)
                          (Hagen-Poiseuille-Widerstand + Masse mit
                          Profilfaktor 4/3)
          * omega -> oo:  Z -> rho0*L/(pi r^2) * [ sqrt(2*mu*omega/rho0)/r
                          + j*omega*(1 + sqrt(2*mu/(rho0*omega))/r) ]
                          (Kirchhoff-Grenzschichtverluste an der Wand)

        MÜNDUNGSKORREKTUR: an jedem offenen Ende schwingt eine zusätzliche
        Luftmasse mit; für angeflanschte Mündungen delta_L = 0.85*r pro Seite.

        STRAHLUNGSWIDERSTAND (radiates=True, Öffnung ins Freifeld):
            R_rad = rho0*c/(pi r^2) * (k r)^2 / 2    (Kolben in Schallwand,
                                                      auf rho0*c/S begrenzt)
        """
        S = np.pi * radius**2
        omega = np.asarray(omega, dtype=float)

        if _HAS_SCIPY:
            k_v = np.sqrt(-1j * omega * RHO0 / MU_AIR)
            arg = k_v * radius
            F_v = 1.0 - 2.0 * _besselj(1, arg) / (arg * _besselj(0, arg))
            Z = 1j * omega * RHO0 * length / (S * F_v)
        else:
            # Fallback: Überblendung der beiden Grenzfälle über die
            # Schubzahl s = r * sqrt(rho0*omega/mu)
            s = radius * np.sqrt(RHO0 * omega / MU_AIR)
            Z_lo = (8.0 * MU_AIR * length / (np.pi * radius**4)
                    + 1j * omega * (4.0 / 3.0) * RHO0 * length / S)
            Z_hi = (RHO0 * length / S) * (
                np.sqrt(2.0 * MU_AIR * omega / RHO0) / radius
                + 1j * omega * (1.0 + np.sqrt(2.0 * MU_AIR / (RHO0 * omega)) / radius)
            )
            w = np.clip((s - 1.0) / 9.0, 0.0, 1.0)
            Z = (1.0 - w) * Z_lo + w * Z_hi

        if end_correction:
            Z = Z + 1j * omega * RHO0 * (2.0 * 0.85 * radius) / S

        if radiates:
            k = omega / C_AIR
            Z = Z + (RHO0 * C_AIR / S) * np.minimum((k * radius) ** 2 / 2.0, 1.0)

        return Z / count  # parallele Löcher

    def _blind_hole_impedance(self, omega):
        """Shunt-Impedanz der Blindlöcher (Sacklöcher) in der Backplate.

        Blindlöcher vergrößern das wirksame Luftvolumen unter der Membran
        und entlasten so den Squeeze-Film (weniger Dämpfung, klassischer
        Trick bei Großmembran-Backplates). Modell: viskose Rohrimpedanz
        über die HALBE Tiefe (mittlere Eindringtiefe der Strömung in ein
        geschlossenes Loch) in Serie mit der isothermen Nachgiebigkeit des
        vollen Lochvolumens:
            Z_blind = Z_tube(r, d/2) + 1/(j*omega*C_blind)
        Einseitige Mündungskorrektur (nur membranseitige Öffnung).
        """
        omega = np.asarray(omega, dtype=float)
        Z_visc = self._hole_impedance(
            omega, self.r_bh, 0.5 * self.d_bh, self.n_bh, end_correction=False
        )
        S = np.pi * self.r_bh**2
        Z_end = 1j * omega * RHO0 * (0.85 * self.r_bh) / (S * self.n_bh)
        Z_comp = 1.0 / (1j * omega * self.C_A_blind)
        return Z_visc + Z_end + Z_comp

    def _radiation_impedance_membrane(self, omega):
        """Strahlungsimpedanz der Membranvorderseite.

        KOLBEN IN UNENDLICHER SCHALLWAND (Niederfrequenz-Näherung ka < 2):
            Z_rad = rho0*c/S * [ (k a)^2 / 2  +  j * 8 k a / (3 pi) ]
        Realteil = Strahlungswiderstand (auf rho0*c/S begrenzt),
        Imaginärteil = mitschwingende Luftmasse M_rad = 8 rho0/(3 pi^2 a).
        Die Annahme "unendliche Schallwand" überschätzt die Strahlungslast
        einer frei stehenden Kapsel bei tiefen Frequenzen leicht — für das
        Lumped-Modell ist der Einfluss vernachlässigbar klein.
        """
        omega = np.asarray(omega, dtype=float)
        a = self.a_mem
        S = self.S_mem
        k = omega / C_AIR
        R = (RHO0 * C_AIR / S) * np.minimum((k * a) ** 2 / 2.0, 1.0)
        X = (RHO0 * C_AIR / S) * (8.0 * k * a) / (3.0 * np.pi)
        return R + 1j * X

    def _membrane_impedance(self, omega):
        """Serienimpedanz der Membran: Z = R + j*omega*M + 1/(j*omega*C_eff)."""
        omega = np.asarray(omega, dtype=float)
        return (
            self.R_A_mem
            + 1j * omega * self.M_A_mem
            + 1.0 / (1j * omega * self.C_A_eff)
        )

    # ======================================================================
    # Beugung / Druckstau am Kapselkörper
    # ======================================================================
    def _diffraction_factors(self, omega, theta):
        """Druckfaktoren an Membran und Rückeinlässen inkl. Beugung.

        DRUCKSTAU UND ABSCHATTUNG AM KAPSELKÖRPER
        -----------------------------------------
        Sobald die Wellenlänge in die Größenordnung des Gehäuses kommt
        (ka >~ 1), verändert der Kapselkörper das Schallfeld: frontal
        staut sich der Druck auf (bis +6 dB an der starren Wand), seitlich
        und rückwärtig wird abgeschattet. Deshalb richtet JEDES Mikrofon —
        auch ein idealer Druckempfänger — zu hohen Frequenzen hin immer
        stärker. Grundlage ist die klassische Reihenlösung der Streuung
        einer ebenen Welle an der STARREN KUGEL (Morse, "Vibration and
        Sound"; Konvention e^{-i omega t}):

            p(a, psi)/p0 = i/(ka)^2 * Sum_n (2n+1)(-i)^n P_n(cos psi)
                                              / h'_n^(1)(ka)

        psi = Winkel zwischen Aufpunktrichtung und Einfallsrichtung
        (Vereinfachung über die Wronski-Identität j h' - j' h = i/x^2).
        Die Reihe enthält automatisch:
          * den Druckstau am vorderen Pol (|p| -> 2 fuer ka -> oo),
          * die Abschattung inkl. Antipoden-Hellfleck (kriechende Wellen),
          * die verlaengerte Beugungslaufzeit um den Koerper (fuer ka -> 0
            wird die effektive Front-Rueck-Distanz 1.5 * 2R — der bekannte
            Dipol-Streufaktor 3/2 der starren Kugel).

        APERTUREFFEKT DER MEMBRAN: die ausgedehnte Membran mittelt die
        Druckverteilung über ihre Fläche — bei schrägem Einfall löschen
        sich Beiträge hoher Frequenzen teilweise aus. Modelliert als
        flächengemittelte Kugelkalotte am vorderen Pol; die azimutale
        Mittelung ist über das Legendre-Additionstheorem exakt:
            <P_n(cos psi)>_Ring    = P_n(cos alpha) * P_n(cos theta)
            <P_n(cos psi)>_Kalotte = C_n * P_n(cos theta)
            C_n = [P_{n-1}(u0) - P_{n+1}(u0)] / ((2n+1)(1-u0)),
            u0 = cos(Kalotten-Halbwinkel)

        Rückgabe: (F_front, F_rear) komplex, Form (len(omega), len(theta)),
        konjugiert in die hier verwendete e^{+j omega t}-Konvention.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        ka = omega * self.R_body / C_AIR
        n_max = int(np.max(ka)) + 12
        ct = np.cos(theta)

        # Legendre-Polynome per Aufwärtsrekurrenz:
        # (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
        def _legendre_table(x, nmax):
            x = np.atleast_1d(x)
            tab = [np.ones_like(x), x.copy()]
            for n in range(1, nmax):
                tab.append(((2 * n + 1) * x * tab[n] - n * tab[n - 1])
                           / (n + 1))
            return tab

        P_t = _legendre_table(ct, n_max + 1)                # an cos(theta)
        P_u0 = _legendre_table(np.array([self._cap_cos]), n_max + 2)
        P_ur = _legendre_table(np.array([self._ring_cos]), n_max + 2)
        u0 = self._cap_cos

        F_f = np.zeros((omega.size, theta.size), dtype=complex)
        F_r = np.zeros_like(F_f)
        for n in range(n_max + 1):
            if n == 0 or (1.0 - u0) < 1e-9:   # Punktmembran -> C_n = P_n(1)
                C_n = 1.0
            else:
                C_n = float(P_u0[n - 1][0] - P_u0[n + 1][0]) \
                    / ((2 * n + 1) * (1.0 - u0))
            h1p = (_sph_jn(n, ka, derivative=True)
                   + 1j * _sph_yn(n, ka, derivative=True))
            base = (2 * n + 1) * (-1j) ** n / h1p           # (N_omega,)
            F_f += np.outer(base, C_n * P_t[n])
            F_r += np.outer(base, float(P_ur[n][0]) * P_t[n])
        pref = 1j / ka**2
        F_f *= pref[:, None]
        F_r *= pref[:, None]
        # Konvention e^{-i omega t} -> e^{+j omega t}: konjugieren
        return np.conj(F_f), np.conj(F_r)

    def _source_pressures(self, omega, theta):
        """Effektive Quelldrücke p_front/p_rear für Einfallswinkel theta.

        Mit Beugung: Kugelstreufaktoren (s. :meth:`_diffraction_factors`).
        Ohne (include_diffraction=False oder kein SciPy): ebene Welle mit
        geometrischer Wegdifferenz d_ext (Verhalten der Vorversionen).
        Rückgabeform jeweils (len(omega), len(theta)).
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if self.include_diffraction and _HAS_SCIPY:
            return self._diffraction_factors(omega, theta)
        k = omega / C_AIR
        p_front = np.ones((omega.size, theta.size), dtype=complex)
        p_rear = np.exp(-1j * np.outer(k * self.d_ext, np.cos(theta)))
        return p_front, p_rear

    # ======================================================================
    # ABCD-Zweitor-Bausteine (vektorisiert über omega, Form (2, 2, N))
    # ======================================================================
    @staticmethod
    def _abcd_series(Z, omega):
        one = np.ones_like(omega, dtype=complex)
        return np.array([[one, Z * one], [np.zeros_like(one), one]])

    @staticmethod
    def _abcd_shunt(Y, omega):
        one = np.ones_like(omega, dtype=complex)
        return np.array([[one, np.zeros_like(one)], [Y * one, one]])

    @staticmethod
    def _mmul(A, B):
        """Produkt zweier (2,2,N)-Kettenmatrizen."""
        return np.array([
            [A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0],
             A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]],
            [A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0],
             A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]],
        ])

    def _duct_propagation(self, omega, radius_duct):
        """Ausbreitungskonstante und Wellenwiderstand einer weiten Leitung.

        AKUSTISCHE LEITUNG MIT WANDVERLUSTEN (Kirchhoff, weites Rohr):
            gamma = alpha + j*k,   k = omega/c,   Zc = rho0*c/S
            alpha = sqrt(mu*omega/(2*rho0)) * (1 + (gamma_ad - 1)/sqrt(Pr))
                    / (r * c)
        Der Dämpfungsbelag alpha erfasst viskose UND thermische Grenz-
        schichtverluste an der Rohrwand und verhindert zugleich unphysika-
        lisch scharfe Stehwellenresonanzen des Hohlraums im Modell.
        """
        omega = np.asarray(omega, dtype=float)
        k = omega / C_AIR
        alpha = (
            np.sqrt(MU_AIR * omega / (2.0 * RHO0))
            * (1.0 + (GAMMA - 1.0) / np.sqrt(PRANDTL))
            / (radius_duct * C_AIR)
        )
        Zc = RHO0 * C_AIR / (np.pi * radius_duct**2)
        return alpha + 1j * k, Zc

    def _abcd_line(self, omega, length, radius_duct):
        """Kettenmatrix einer akustischen Leitung (Laufzeitglied/Hohlraum).

        Die Leitung wirkt als Laufzeitglied (tau = L/c) und bildet zugleich
        Nachgiebigkeit + Masse der eingeschlossenen Luft sowie Stehwellen
        oberhalb der Lumped-Grenze korrekt ab:
            T = [[cosh(gL), Zc*sinh(gL)], [sinh(gL)/Zc, cosh(gL)]]
        """
        g, Zc = self._duct_propagation(omega, radius_duct)
        gl = g * length
        ch, sh = np.cosh(gl), np.sinh(gl)
        return np.array([[ch, Zc * sh], [sh / Zc, ch]])

    def _closed_stub_impedance(self, omega, length, radius_duct):
        """Eingangsimpedanz eines hart abgeschlossenen Leitungsstücks.

            Z_in = Zc * coth(gamma * L)
        (verlustfrei: -j*Zc*cot(kL); bei tiefen Frequenzen die reine
        Volumen-Nachgiebigkeit 1/(j*omega*V/(rho0 c^2)) des Reststücks).
        """
        g, Zc = self._duct_propagation(omega, radius_duct)
        gl = g * length
        return Zc * np.cosh(gl) / np.sinh(gl)

    # ======================================================================
    # Netzwerk-Zusammenbau
    # ======================================================================
    def _backplate_gap_abcd(self, omega, outside_to_membrane,
                            holes_radiate=False):
        """Kettenmatrix des Backplate/Luftspalt-Netzwerks.

        Topologie von der Membran aus gesehen:
            Membran --[Shunt: Blindlöcher]--[Serie: R_gap (Škvor)]--
            --[Shunt: C_gap]--[Serie: Durchgangslöcher]-- außen
        ``outside_to_membrane=True`` liefert die umgekehrte Kettenrichtung
        (für die vordere Backplate der Dual-Architektur).
        ``holes_radiate=True``: die Durchgangslöcher münden direkt ins
        Freifeld (keine rückwärtige Baugruppe) -> Strahlungswiderstand.

        Sonderfall geschlossene Backplate (n_th = 0): kein Strömungspfad
        durch die Platte und keine laterale Škvor-Strömung — Spaltvolumen
        und Blindlöcher wirken als reine Shunt-Nachgiebigkeiten an der
        Membran; die Kette endet dahinter blockiert.
        """
        mats = []  # Reihenfolge: Membranseite -> Außenseite
        if self.n_bh > 0:
            mats.append(self._abcd_shunt(1.0 / self._blind_hole_impedance(omega), omega))
        if self.n_th == 0:
            mats.append(self._abcd_shunt(1j * omega * self.C_A_gap, omega))
            return reduce(self._mmul, mats)
        Z_holes = self._hole_impedance(
            omega, self.r_th, self.t_bp, self.n_th,
            end_correction=True, radiates=holes_radiate,
        )
        mats.append(self._abcd_series(self.R_A_gap, omega))
        mats.append(self._abcd_shunt(1j * omega * self.C_A_gap, omega))
        mats.append(self._abcd_series(Z_holes, omega))
        if outside_to_membrane:
            mats = mats[::-1]
        return reduce(self._mmul, mats)

    def _assemble_network(self, omega):
        """Baut die Kettenmatrizen des Gesamtnetzwerks für alle omega auf.

        Rückgabe:
            T_total : Kettenmatrix vom vorderen Einlass zum rückwärtigen Port
            T_rear  : Kettenmatrix von der Membran-Rückseite zum Port
                      (Zeile [1,:] liefert daraus den Membran-Volumenfluss)
        """
        omega = np.asarray(omega, dtype=float)

        # ---------------- vorderer Zweig: Quelle -> Membran ----------------
        front = [
            # Strahlungsimpedanz der Membran-/Einlassöffnung
            self._abcd_series(self._radiation_impedance_membrane(omega), omega),
            # Gewebe vor der Membran: Z = Rayl-Wert / durchströmte Fläche
            self._abcd_series(self.rayl_front / self.S_mem, omega),
        ]
        if self.architecture == "dual":
            # vordere Backplate (identisch zur hinteren, gespiegelt)
            front.append(self._backplate_gap_abcd(omega, outside_to_membrane=True))
        T_front = reduce(self._mmul, front)

        # ---------------------------- Membran ------------------------------
        T_mem = self._abcd_series(self._membrane_impedance(omega), omega)

        # ------------- hinterer Zweig: Membran -> rückwärtiger Port --------
        # Die Durchgangslöcher der Backplate sind der Zugang zur Rückseite.
        # Drei Fälle:
        #   n_th = 0                  -> hermetisch dicht direkt am Spalt
        #   keine rückwärtige Baugruppe -> Löcher münden (durchs Gewebe)
        #                                direkt ins rückwärtige Schallfeld
        #   Baugruppe vorhanden       -> Gewebe -> Laufzeitglied -> Hohlraum
        vents_directly = self.n_th > 0 and not self.rear_network_enabled
        rear = [self._backplate_gap_abcd(omega, outside_to_membrane=False,
                                         holes_radiate=vents_directly)]

        if self.n_th == 0:
            # geschlossene Backplate: Port unmittelbar blockiert
            # (Gewebe/Laufzeitglied/Hohlraum sind akustisch unerreichbar)
            T_rear = reduce(self._mmul, rear)
            T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
            return T_total, T_rear

        # Gewebe hinter der Backplate (überspannt die Backplate-Fläche,
        # liegt auch im Direktbelüftungsfall über den Öffnungen)
        rear.append(self._abcd_series(self.rayl_rear / self.S_bp, omega))

        if vents_directly:
            # kein Laufzeitglied/Hohlraum: Port = rückwärtiges Schallfeld
            T_rear = reduce(self._mmul, rear)
            T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
            return T_total, T_rear

        # Laufzeitglied als akustische Leitung (tau = L/c)
        if self.l_delay > 0.0:
            rear.append(self._abcd_line(omega, self.l_delay, self.a_bp))

        if self.rear_open:
            if self.cavity_hole_position == "circumference":
                # Leitung bis zur axialen Lochposition ...
                if self.x_ch > 0.0:
                    rear.append(self._abcd_line(omega, self.x_ch, self.a_bp))
                # ... dahinter wirkt das geschlossene Reststück als Shunt
                l_rest = self.l_cav - self.x_ch
                if l_rest > 1e-9:
                    Z_stub = self._closed_stub_impedance(omega, l_rest, self.a_bp)
                    rear.append(self._abcd_shunt(1.0 / Z_stub, omega))
                hole_len = self.t_cav_wall  # radiale Löcher durch die Wand
            else:  # "end": Löcher in der hinteren Stirnfläche
                rear.append(self._abcd_line(omega, self.l_cav, self.a_bp))
                hole_len = self.t_cav_wall
            # Einlasslöcher: thermoviskoses Rohr + Strahlung ins Freifeld
            Z_ch = self._hole_impedance(
                omega, self.r_ch, hole_len, self.n_ch,
                end_correction=True, radiates=True,
            )
            rear.append(self._abcd_series(Z_ch, omega))
        else:
            # geschlossene Rückseite: gesamter Hohlraum, Port ist "blockiert"
            rear.append(self._abcd_line(omega, self.l_cav, self.a_bp))

        T_rear = reduce(self._mmul, rear)
        T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
        return T_total, T_rear

    def _membrane_volume_velocity(self, omega, T_total, T_rear, p_front, p_rear):
        """Löst das Netzwerk für den Volumenfluss der Membran.

        Kettengleichung: [p_front; q_front] = T_total * [p_rear; q_rear]
          * offene Rückseite:  p_front, p_rear bekannt ->
                q_rear = (p_front - A*p_rear) / B
          * geschlossene Rückseite: q_rear = 0, Druck am Abschluss unbekannt ->
                p_end = p_front / A
        Der Membran-Volumenfluss ist der Eingangsstrom der hinteren Kette:
            q_mem = T_rear[1,0]*p_port + T_rear[1,1]*q_port
        (Serien-Zweitore ändern den Strom nicht, daher gilt q durch die
        Membran = q am Eingang von T_rear.)
        """
        A, B = T_total[0, 0], T_total[0, 1]
        if self.rear_open:
            q_rear = (p_front - A * p_rear) / B
            q_mem = T_rear[1, 0] * p_rear + T_rear[1, 1] * q_rear
        else:
            p_end = p_front / A
            q_mem = T_rear[1, 0] * p_end
        return q_mem

    def _output_voltage(self, omega, q_mem):
        """Elektrostatische Wandlung: Volumenfluss -> Leerlaufspannung.

        e = Theta * V_disp mit dem in _derive_parameters am statischen
        Arbeitspunkt berechneten Wandlerkoeffizienten Theta (Herleitung
        dort: e = U0*dC/C0 bei konstanter Ladung, dC über das Membran-
        profil und die poröse Elektrode integriert; Dual-Backplates im
        Gegentakt). V_disp = q_mem/(j*omega) ist die Volumenverschiebung.
        """
        v_disp = q_mem / (1j * np.asarray(omega, dtype=float))
        return self._theta * v_disp

    # ======================================================================
    # Öffentliche Auswertemethoden
    # ======================================================================
    def transfer_function(self, frequencies_hz, angle_deg=0.0):
        """Komplexe Übertragungsfunktion e/p0 [V/Pa] für gegebene Frequenzen.

        angle_deg: Schalleinfallswinkel (0° = frontal). Referenz ist der
        ungestörte Freifelddruck p0 = 1 Pa; mit aktiver Beugung ist das
        Ergebnis also die FREIFELD-Übertragungsfunktion inkl. Druckstau.
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * f
        T_total, T_rear = self._assemble_network(omega)
        theta = np.array([np.deg2rad(angle_deg)])
        p_f, p_r = self._source_pressures(omega, theta)
        q_mem = self._membrane_volume_velocity(omega, T_total, T_rear,
                                               p_f[:, 0], p_r[:, 0])
        return self._output_voltage(omega, q_mem)

    def frequency_response(self, f_min=10.0, f_max=25000.0, n_points=500,
                           angle_deg=0.0):
        """Frequenzgang (Amplitude und Phase) der Kapsel.

        Rückgabe: dict mit NumPy-Arrays
            'frequency_hz'        — Frequenzachse (logarithmisch)
            'sensitivity_v_pa'    — komplexe Empfindlichkeit e/p0 [V/Pa]
            'amplitude_db'        — 20*log10(|e/p0|)  [dB re 1 V/Pa]
            'amplitude_db_norm'   — auf den Wert bei 1 kHz normiert [dB]
            'phase_deg'           — Phase, über der Frequenz entrollt [Grad]
        """
        f = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))
        H = self.transfer_function(f, angle_deg=angle_deg)
        amp_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-30))
        # Normierung auf 1 kHz (Interpolation auf der log-Achse)
        ref_db = np.interp(np.log10(1000.0), np.log10(f), amp_db)
        phase = np.rad2deg(np.unwrap(np.angle(H)))
        return {
            "frequency_hz": f,
            "sensitivity_v_pa": H,
            "amplitude_db": amp_db,
            "amplitude_db_norm": amp_db - ref_db,
            "phase_deg": phase,
        }

    def directivity(self, frequencies_hz=(100.0, 1000.0, 5000.0, 10000.0),
                    n_angles=361):
        """Richtdiagramm für die angegebenen Frequenzen.

        Für jede Frequenz wird das Netzwerk einmal aufgebaut; die
        winkelabhängigen Quelldrücke an Membran und Rückeinlässen liefert
        :meth:`_source_pressures` — mit aktiver Beugung inklusive
        Druckstau/Abschattung am Kapselkörper und Apertureffekt der
        Membran, wodurch auch ein Druckempfänger zu hohen Frequenzen hin
        richtet.

        Rückgabe: dict
            'angles_deg' — Winkelachse 0..360°
            'patterns'   — {f: {'linear': |e(theta)|/|e(0)|,
                                'db': 20*log10(...), auf -40 dB begrenzt}}
        """
        angles = np.linspace(0.0, 360.0, int(n_angles))
        theta = np.deg2rad(angles)
        patterns = {}
        for f in frequencies_hz:
            omega = np.array([2.0 * np.pi * float(f)])
            T_total, T_rear = self._assemble_network(omega)
            p_f2, p_r2 = self._source_pressures(omega, theta)
            p_front, p_rear = p_f2[0], p_r2[0]
            if self.rear_open:
                A, B = T_total[0, 0][0], T_total[0, 1][0]
                q_rear = (p_front - A * p_rear) / B
                q_mem = T_rear[1, 0][0] * p_rear + T_rear[1, 1][0] * q_rear
            else:
                # Druckempfänger: winkelabhängig nur über den Druckstau
                # an der Membran (ohne Beugung: exakte Kugel)
                q_mem = T_rear[1, 0][0] * (p_front / T_total[0, 0][0])
            e = self._output_voltage(np.full_like(theta, omega[0]), q_mem)
            mag = np.abs(e)
            ref = mag[0] if mag[0] > 0 else np.max(mag)
            lin = mag / ref
            db = np.maximum(20.0 * np.log10(np.maximum(lin, 1e-30)), -40.0)
            patterns[float(f)] = {"linear": lin, "db": db}
        return {"angles_deg": angles, "patterns": patterns}

    # ======================================================================
    # Diagnose
    # ======================================================================
    def summary(self):
        """Mehrzeilige Übersicht der abgeleiteten Modellparameter."""
        sens = self.transfer_function(1000.0)[0]
        lines = [
            "MicrophoneCapsule — abgeleitete Parameter",
            "-" * 55,
            f"Architektur:                  {self.architecture} "
            f"({self.n_bp} Backplate(s))",
            f"Membranfläche:                {self.S_mem * 1e6:9.2f} mm²",
            f"akust. Masse Membran M_A:     {self.M_A_mem:9.2f} kg/m⁴",
            f"akust. Nachgiebigkeit C_A:    {self.C_A_mem:9.3e} m³/Pa",
            f"  dto. effektiv (mit Bias):   {self.C_A_eff:9.3e} m³/Pa",
            f"Feder-Erweichung durch Bias:  {self.softening_ratio * 100:9.2f} %",
            f"statische Durchbiegung w0:    {self.w0_static * 1e6:9.2f} µm "
            f"(Restspalt Mitte {self.h_min_static * 1e6:.1f} µm)",
            ("Pull-in-Spannung U_PI:        "
             + (f"{self.U_pullin:9.1f} V" if np.isfinite(self.U_pullin)
                else "     > 20 kV")),
            f"Elektroden-Porosität:         {100 * (self.phi_th + self.phi_bh):9.1f} % "
            f"(Durchgang {100 * self.phi_th:.1f} %, Blind {100 * self.phi_bh:.1f} %)",
            f"Resonanz (Modell):            {self.f_res:9.1f} Hz",
            f"Resonanz aus Vorspannung/E:   {self.f_res_from_tension:9.1f} Hz",
            f"Ruhekapazität C0 (je BP):     {self.C_elec_0 * 1e12:9.2f} pF",
            ("Squeeze-Film-Widerst. R_gap:  "
             + (f"{self.R_A_gap:9.3e} Pa·s/m³" if self.R_A_gap is not None
                else "        — (Backplate geschlossen)")),
            f"Nachgiebigkeit Spalt C_gap:   {self.C_A_gap:9.3e} m³/Pa",
            f"Nachgiebigkeit Blindl. C_bh:  {self.C_A_blind:9.3e} m³/Pa",
            f"rückwärtige Baugruppe:        {self.rear_network_enabled}",
            f"Rückseite offen (Gradient):   {self.rear_open}",
            f"äußere Wegdifferenz d_ext:    {self.d_ext * 1e3:9.2f} mm",
            f"Beugung am Gehäuse:           "
            f"{self.include_diffraction and _HAS_SCIPY}",
            f"Ersatz-Gehäuseradius R_body:  {self.R_body * 1e3:9.2f} mm "
            f"(ka=1 bei {C_AIR / (2 * np.pi * self.R_body):.0f} Hz)",
            f"Empfindlichkeit @ 1 kHz:      {abs(sens) * 1e3:9.2f} mV/Pa "
            f"({20 * np.log10(abs(sens)):.1f} dB re 1 V/Pa)",
        ]
        return "\n".join(lines)


# ===========================================================================
# Testlauf mit realistischen Dummy-Werten
# ===========================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # 1"-Großmembrankapsel, Nieren-artig, einzelne Backplate
    capsule = MicrophoneCapsule(
        # Membran
        membrane_material="PET",          # Mylar-Folie
        membrane_resonance_hz=8000.0,     # [Hz]
        membrane_diameter=22e-3,          # [m]
        membrane_thickness=6e-6,          # [m]
        membrane_tension=400.0,           # [N/m]
        # Backplate
        air_gap=40e-6,                    # [m]
        backplate_diameter=20e-3,         # [m]
        backplate_thickness=3e-3,         # [m]
        bias_voltage=60.0,                # [V]
        architecture="single",
        n_through_holes=60, through_hole_diameter=1.0e-3,
        n_blind_holes=30, blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        # Rückseite
        delay_length=3e-3,                # Laufzeitglied [m]
        cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
        # Feiner Lochkranz am Umfang: viele Mikrolöcher wirken als
        # WIDERSTANDSDOMINIERTER Einlass (aperiodisch, Q < 1) — das
        # akustische Pendant zum Gewebe über Schlitzen realer Kapseln.
        # Wenige große Löcher wären massedominiert und bildeten mit dem
        # Hohlraum einen unbedämpften Helmholtz-Resonator, der die
        # Richtwirkung im Mittenband zerstört.
        cavity_hole_position="circumference",
        n_cavity_holes=200, cavity_hole_diameter=0.2e-3,
        cavity_hole_axial_position=6e-3,
        fabric_front_rayl=10.0, fabric_rear_rayl=25.0,
    )

    print(capsule.summary())
    print()

    # ------------------------- Frequenzgang --------------------------------
    fr = capsule.frequency_response(10.0, 25000.0, n_points=400)
    f = fr["frequency_hz"]
    assert f.shape == (400,)
    assert fr["amplitude_db"].shape == f.shape
    assert fr["phase_deg"].shape == f.shape
    assert np.all(np.isfinite(fr["amplitude_db"]))
    assert np.all(np.isfinite(fr["phase_deg"]))

    print("Frequenzgang (Auszug, 0° Einfall):")
    print(f"{'f [Hz]':>10} {'Ampl. [dB re 1V/Pa]':>22} {'norm. [dB]':>12} "
          f"{'Phase [°]':>11}")
    for f_show in (20, 100, 1000, 5000, 10000, 20000):
        i = int(np.argmin(np.abs(f - f_show)))
        print(f"{f[i]:10.0f} {fr['amplitude_db'][i]:22.2f} "
              f"{fr['amplitude_db_norm'][i]:12.2f} {fr['phase_deg'][i]:11.1f}")
    print()

    # ------------------------- Richtdiagramm -------------------------------
    di = capsule.directivity(frequencies_hz=(100.0, 1000.0, 5000.0, 10000.0),
                             n_angles=361)
    angles = di["angles_deg"]
    assert angles.shape == (361,)
    print("Richtdiagramm |e(θ)|/|e(0°)| (linear):")
    print(f"{'f [Hz]':>10} {'0°':>7} {'90°':>7} {'135°':>7} {'180°':>7}")
    for f_pat, pat in di["patterns"].items():
        lin = pat["linear"]
        assert lin.shape == angles.shape
        assert np.all(np.isfinite(lin))
        print(f"{f_pat:10.0f} {lin[0]:7.3f} {lin[90]:7.3f} "
              f"{lin[135]:7.3f} {lin[180]:7.3f}")
    print()

    # --------- Gegenprobe 1: geschlossene Rückseite -> Kugel ---------------
    # (ohne Beugung: exakte Kugel als Netzwerk-Konsistenzprüfung)
    omni = MicrophoneCapsule(n_cavity_holes=0, include_diffraction=False)
    di_o = omni.directivity(frequencies_hz=(1000.0,))
    lin_o = di_o["patterns"][1000.0]["linear"]
    assert np.allclose(lin_o, 1.0, atol=1e-9), "Druckempfänger muss Kugel sein"
    print("Gegenprobe geschlossene Rückseite (ohne Beugung): Kugel  OK")

    # --------- Gegenprobe 2: Dual-Backplate (Gegentakt) --------------------
    # Im steifigkeitskontrollierten Bereich (deutlich unterhalb der
    # Resonanz) muss der Gegentakt der beiden Backplates ~+6 dB liefern.
    # Dazu kommt hier ~+2 dB, weil die vordere Backplate den vorderen
    # Einlass nach vorn verschiebt (größeres d_ext -> mehr Gradienten-
    # antrieb). Nahe der Resonanz wird ein Teil des Gewinns durch die
    # zusätzliche Squeeze-Film-Dämpfung des VORDEREN Spalts aufgezehrt.
    dual = MicrophoneCapsule(architecture="dual")
    fr_d = dual.frequency_response(n_points=100)
    assert np.all(np.isfinite(fr_d["amplitude_db"]))
    e_s = 20 * np.log10(abs(capsule.transfer_function(100.0)[0]))
    e_d = 20 * np.log10(abs(dual.transfer_function(100.0)[0]))
    assert 4.0 < e_d - e_s < 9.5, "Gegentakt-Gewinn außerhalb Erwartung"
    print(f"Dual-Backplate @100 Hz: {e_d:.2f} dB re 1 V/Pa "
          f"(single: {e_s:.2f} dB) — Gegentakt-Gewinn "
          f"{e_d - e_s:+.2f} dB  OK")

    # --------- Gegenprobe 3: Durchgangslöcher & Rückseiten-Baugruppe -------
    # a) Ohne rückwärtige Baugruppe münden die Durchgangslöcher direkt ins
    #    Schallfeld -> Gradientenempfänger mit kurzer Wegdifferenz.
    vented = MicrophoneCapsule(rear_network_enabled=False)
    di_v = vented.directivity(frequencies_hz=(500.0,))
    lin_v = di_v["patterns"][500.0]["linear"]
    assert lin_v[90] < 0.9 or lin_v[180] < 0.9, \
        "direkt belüftete Backplate muss Richtwirkung zeigen"
    # b) Ohne Durchgangslöcher ist die Kapsel hermetisch dicht — auch mit
    #    montierter Rückseite (die dann akustisch unerreichbar ist).
    sealed = MicrophoneCapsule(n_through_holes=0, include_diffraction=False)
    di_s = sealed.directivity(frequencies_hz=(1000.0,))
    assert np.allclose(di_s["patterns"][1000.0]["linear"], 1.0, atol=1e-9), \
        "geschlossene Backplate muss Kugel sein"
    fr_s = sealed.frequency_response(n_points=50)
    assert np.all(np.isfinite(fr_s["amplitude_db"]))
    print(f"Belüftete Backplate (ohne Baugruppe): 90°={lin_v[90]:.2f}, "
          f"180°={lin_v[180]:.2f} — Richtwirkung  OK")
    print("Geschlossene Backplate (0 Durchgangslöcher): Kugel, "
          "hermetisch dicht  OK")

    # --------- Gegenprobe 4: Druckstau/Beugung am Kapselkörper -------------
    # Ein Druckempfänger MIT Beugung muss bei tiefen Frequenzen praktisch
    # kugelförmig sein, zu hohen Frequenzen hin aber zunehmend richten;
    # frontal steigt die Freifeldempfindlichkeit durch den Druckstau.
    omni_d = MicrophoneCapsule(n_through_holes=0)   # Beugung standardmäßig an
    di_hf = omni_d.directivity(frequencies_hz=(100.0, 4000.0, 16000.0))
    p100 = di_hf["patterns"][100.0]["linear"]
    p4k = di_hf["patterns"][4000.0]["linear"]
    p16k = di_hf["patterns"][16000.0]["linear"]
    assert abs(p100[135] - 1.0) < 0.03, "100 Hz muss nahezu Kugel sein"
    assert p16k[135] < p4k[135] < 1.0, \
        "Richtwirkung muss mit der Frequenz zunehmen"
    e_lo = abs(omni_d.transfer_function(200.0)[0])
    e_hi = abs(omni_d.transfer_function(16000.0, angle_deg=0.0)[0])
    e_hi_nod = abs(MicrophoneCapsule(
        n_through_holes=0, include_diffraction=False
    ).transfer_function(16000.0)[0])
    boost_db = 20 * np.log10(e_hi / e_hi_nod)
    assert 1.0 < boost_db < 7.0, "Druckstau frontal: erwarte ~+2..6 dB"
    print(f"Druckstau/Beugung: Kugel @100 Hz, 135°-Pegel "
          f"{p4k[135]:.2f} (4 kHz) -> {p16k[135]:.2f} (16 kHz), "
          f"frontaler Druckstau @16 kHz: +{boost_db:.1f} dB  OK")

    # --------- Gegenprobe 5: Elektrostatik / Pull-in -----------------------
    # Eine sehr weiche Membran (f_res = 800 Hz -> T ~ 4 N/m bei 22 mm/6 µm)
    # kann 60 V bei 40 µm Spalt statisch nicht tragen (das Luftpolster
    # entweicht durch die Löcher) -> Pull-in. Bei reduzierter Spannung
    # existiert ein stabiler Arbeitspunkt mit statischer Durchbiegung.
    try:
        MicrophoneCapsule(membrane_resonance_hz=800.0)
        raise AssertionError("800 Hz / 40 µm / 60 V müsste kollabieren")
    except ValueError as exc:
        assert "Pull-in" in str(exc)
    soft = MicrophoneCapsule(membrane_resonance_hz=800.0, bias_voltage=20.0)
    assert 20.0 < soft.U_pullin < 60.0, "U_PI muss zwischen 20 und 60 V liegen"
    assert 0.0 < soft.w0_static < soft.h_gap
    fr_soft = soft.frequency_response(n_points=50)
    assert np.all(np.isfinite(fr_soft["amplitude_db"]))
    print(f"Pull-in-Gegenprobe: 800 Hz kollabiert bei 60 V (U_PI = "
          f"{soft.U_pullin:.1f} V), stabil bei 20 V mit statischer "
          f"Durchbiegung {soft.w0_static * 1e6:.1f} µm  OK")

    print("\nAlle Testläufe erfolgreich — Arrays werden korrekt berechnet.")
