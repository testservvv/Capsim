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
    # Besselfunktionen mit komplexem Argument benötigt.
    from scipy.special import jv as _besselj

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
        # ELEKTROSTATISCHE FEDER-ERWEICHUNG ("spring softening")
        # Die Anziehungskraft der geladenen Backplate wächst mit kleiner
        # werdendem Spalt: F = eps0*A*U0^2 / (2 h^2). Ihre Ableitung wirkt
        # als NEGATIVE Steifigkeit auf die Membran:
        #     k_e = eps0 * A_bp * U0^2 / h^3           [N/m]
        # In der akustischen Domäne (K_A = k / S^2):
        #     1/C_eff = 1/C_A - n_bp * k_e / S_mem^2
        # Bei symmetrischen Dual-Backplates wirken beide Seiten (n_bp = 2).
        # Wird 1/C_eff <= 0, kollabiert die Membran auf die Backplate.
        # ------------------------------------------------------------------
        self.k_elec = EPS0 * self.S_bp * self.u_bias**2 / self.h_gap**3
        inv_C_eff = 1.0 / self.C_A_mem - self.n_bp * self.k_elec / self.S_mem**2
        if inv_C_eff <= 0.0:
            raise ValueError(
                "Elektrostatischer Kollaps: Polarisationsspannung zu hoch "
                "für Spalt/Membransteifigkeit."
            )
        self.C_A_eff = 1.0 / inv_C_eff
        # relative Steifigkeitsreduktion durch die Vorspannung (Diagnose)
        self.softening_ratio = 1.0 - self.C_A_mem / self.C_A_eff

        # kleine interne Membrandämpfung (s. _Q_MEMBRANE_INTERNAL)
        self.R_A_mem = (
            np.sqrt(self.M_A_mem / self.C_A_eff) / self._Q_MEMBRANE_INTERNAL
        )

        # Ruhekapazität pro Backplate (Plattenkondensator):
        self.C_elec_0 = EPS0 * self.S_bp / self.h_gap  # [F]

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
        d = self.h_gap + self.t_bp
        if self.rear_network_enabled:
            d += self.l_delay
            if self.cavity_hole_position == "circumference":
                d += self.x_ch
            else:
                d += self.l_cav + self.t_cav_wall
        if self.architecture == "dual":
            # vordere Backplate verschiebt den vorderen Einlass nach vorn
            d += self.h_gap + self.t_bp
        self.d_ext = d

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

        ELEKTROSTATISCHE WANDLUNG (Betrieb mit konstanter Ladung):
        Bei hochohmiger Beschaltung bleibt die Ladung Q = C0*U0 konstant;
        eine mittlere Spaltänderung <dh> erzeugt die Leerlaufspannung
            e = U0 * <dh> / h.
        <dh> ist die über die BACKPLATE gemittelte Membranauslenkung. Mit
        parabolischem Auslenkungsprofil und Backplate-Radius b gilt
            <dh>_bp = kappa * (V_disp / S_mem),  kappa = 2 - (b/a)^2
        (kappa = 1 für b = a; die Backplate "sieht" bevorzugt die stärker
        ausgelenkte Membranmitte). V_disp = q_mem/(j*omega) ist die
        Volumenverschiebung. Dual-Backplates arbeiten im Gegentakt ->
        doppelte Spannung (n_bp = 2).
        """
        kappa = 2.0 - (self.a_bp / self.a_mem) ** 2 if self.a_bp < self.a_mem else 1.0
        v_disp = q_mem / (1j * np.asarray(omega, dtype=float))
        return self.n_bp * (self.u_bias / self.h_gap) * kappa * v_disp / self.S_mem

    # ======================================================================
    # Öffentliche Auswertemethoden
    # ======================================================================
    def transfer_function(self, frequencies_hz, angle_deg=0.0):
        """Komplexe Übertragungsfunktion e/p0 [V/Pa] für gegebene Frequenzen.

        angle_deg: Schalleinfallswinkel (0° = frontal).
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * f
        T_total, T_rear = self._assemble_network(omega)
        k = omega / C_AIR
        theta = np.deg2rad(angle_deg)
        p_front = np.ones_like(omega, dtype=complex)          # Referenz: 1 Pa
        # ebene Welle: Rückeinlass wird um d_ext*cos(theta)/c später erreicht
        p_rear = np.exp(-1j * k * self.d_ext * np.cos(theta))
        q_mem = self._membrane_volume_velocity(omega, T_total, T_rear,
                                               p_front, p_rear)
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

        Für jede Frequenz wird das Netzwerk einmal aufgebaut; nur die
        Phasenlage der rückwärtigen Druckquelle hängt vom Winkel ab:
            p_rear(theta) = exp(-j*k*d_ext*cos(theta))
        (1.-Ordnung-Gradientenmodell; Beugung am Kapselkörper und die
        Richtwirkung der Membran selbst sind vernachlässigt.)

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
            k = omega[0] / C_AIR
            p_rear = np.exp(-1j * k * self.d_ext * np.cos(theta))
            if self.rear_open:
                A, B = T_total[0, 0][0], T_total[0, 1][0]
                q_rear = (1.0 - A * p_rear) / B
                q_mem = T_rear[1, 0][0] * p_rear + T_rear[1, 1][0] * q_rear
            else:
                # Druckempfänger: Antwort winkelunabhängig (Kugel)
                p_end = 1.0 / T_total[0, 0][0]
                q_mem = np.full_like(theta, T_rear[1, 0][0] * p_end,
                                     dtype=complex)
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
    omni = MicrophoneCapsule(n_cavity_holes=0)
    di_o = omni.directivity(frequencies_hz=(1000.0,))
    lin_o = di_o["patterns"][1000.0]["linear"]
    assert np.allclose(lin_o, 1.0, atol=1e-9), "Druckempfänger muss Kugel sein"
    print("Gegenprobe geschlossene Rückseite: Kugelcharakteristik  OK")

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
    sealed = MicrophoneCapsule(n_through_holes=0)
    di_s = sealed.directivity(frequencies_hz=(1000.0,))
    assert np.allclose(di_s["patterns"][1000.0]["linear"], 1.0, atol=1e-9), \
        "geschlossene Backplate muss Kugel sein"
    fr_s = sealed.frequency_response(n_points=50)
    assert np.all(np.isfinite(fr_s["amplitude_db"]))
    print(f"Belüftete Backplate (ohne Baugruppe): 90°={lin_v[90]:.2f}, "
          f"180°={lin_v[180]:.2f} — Richtwirkung  OK")
    print("Geschlossene Backplate (0 Durchgangslöcher): Kugel, "
          "hermetisch dicht  OK")

    print("\nAlle Testläufe erfolgreich — Arrays werden korrekt berechnet.")
