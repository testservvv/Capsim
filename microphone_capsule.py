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
    from scipy.linalg import solve_banded as _solve_banded
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
        architecture : str
            ``"single"``          — eine Backplate hinter der Membran;
            ``"dual"``            — symmetrische Backplates vor UND hinter
                                    der Membran (Gegentakt-Wandlung);
            ``"dual_diaphragm"``  — K67-Bauform: ZWEI Membranen außen,
                                    zwei innenliegende Backplates, nur
                                    durch ``center_gap`` getrennt. Die
                                    hintere Membran ist passiv (Nieren-
                                    modus) und bildet zusammen mit den
                                    Spalt-/Lochwiderständen das Phasen-
                                    schiebernetzwerk; Laufzeitglied und
                                    Hohlraum werden ignoriert.
        center_gap : float
            Nur ``"dual_diaphragm"``: Luftspalt zwischen den beiden
            Backplate-Hälften (Spacer) [m].
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
        through_hole_rings, blind_hole_rings
            Optional: Verteilung der Löcher eines Typs auf MEHRERE
            Lochkreise als Liste ``[(anzahl, lochkreis_durchmesser_m),
            ...]``; Lochkreis ``None`` -> dieser Anteil ist gleichmäßig
            über die Elektrode verteilt. Wenn gesetzt, ersetzen sie
            ``n_*_holes`` und ``*_pcd`` (Gesamtzahl = Summe der
            Anzahlen). Der Lochdurchmesser gilt weiterhin je Lochtyp.
            Die radiale Sitzverteilung wirkt im 2D-Feldmodell und (über
            die Porositätsprofile) in der Elektrostatik.
        through_holes_stepped : bool
            ``True``: STUFENBOHRUNG wie bei K67/K87 — jedes Durchgangs-
            loch sitzt konzentrisch am GRUND einer Senkung mit Sackloch-
            Geometrie (Durchmesser ``blind_hole_diameter``, Tiefe
            ``blind_hole_depth``); nur die Restdicke
            ``backplate_thickness − blind_hole_depth`` ist mit
            ``through_hole_diameter`` eng durchbohrt. Zählweise:
            ``n_through_holes`` = Anzahl der gestuften Bohrungen,
            ``n_blind_holes`` = nur die REINEN (nicht durchbohrten)
            Sacklöcher. Die Senkungen zählen als Sackvolumen, in der
            Elektrostatik als Stirnöffnung (feldfreier Kern + Blindloch-
            Ring) und im Spaltfilm als je EINE weite Senke. Erfordert
            Sackloch-Ø > Durchgangsloch-Ø.

    Akustische Netzwerke & Rückseite
        rear_network_enabled : bool
            ``True``: hinter der Backplate sitzt die rückwärtige Baugruppe
            (Gewebe -> Laufzeitglied -> Hohlraum -> Einlasslöcher).
            ``False``: keine Baugruppe — die Durchgangslöcher der Backplate
            münden (durch das rückwärtige Gewebe) DIREKT ins rückwärtige
            Schallfeld; die Kapsel wird zum einfachen Gradientenempfänger
            mit der äußeren Wegdifferenz Spalt + Backplate-Dicke.
        rear_spacer_height : float
            Höhe des Distanzrings (Spacer) DIREKT hinter der Backplate [m]:
            eine dünne laterale Luftschicht zwischen Backplate und
            Rückplatte, wie bei K103-artigen Bauformen (TLM 103 — K87-
            Front, Rückseite durch Platte statt Rückmembran abgeschlossen).
            ``0`` = kein Spacer. Bei ``"dual_diaphragm"`` ignoriert.
        rear_plate_thickness, n_rear_plate_holes, rear_plate_hole_diameter
            Massive Rückplatte hinter dem Spacer: Dicke [m] (``0`` = keine
            Rückplatte), Anzahl und Durchmesser ihrer Durchgangslöcher.
            Ohne Löcher verschließt die Platte die Rückseite (die Kapsel
            wird zum Druckempfänger). Ist hinter der Platte nichts mehr
            konfiguriert (Laufzeitglied 0, Hohlraum 0, keine Einlass-
            löcher), münden die Plattenlöcher DIREKT ins rückwärtige
            Schallfeld — der Normalfall der K103-Bauform.
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

    Spaltfilm-Modell
        squeeze_model : str
            ``"1d"`` (Standard): der Luftspalt wird als ein einziges
            Lumped-Element modelliert (Škvor-Widerstand + Nachgiebigkeit
            + Lochimpedanz in einer festen Topologie). Schnell, für dichte
            gleichmäßige Lochmuster ausreichend.
            ``"2d"``: das Druckfeld im Spalt wird als modifizierte
            REYNOLDS-GLEICHUNG (Homentcovschi & Miles, JASA 2004; Bao)
            axialsymmetrisch als Feldgleichung gelöst — s.
            :meth:`_gap_field_2port`. Trennt korrekt den Nachgiebigkeits-
            Rückweg (Spaltvolumen + Blindlöcher, lokal) vom Rückkopplungs-
            weg (nur durch die Durchgangslöcher) und erfasst den radialen
            Druckaufbau. Wichtig bei WENIGEN, engen Durchgangslöchern
            (z. B. Braunmühl-Weber-Platten), wo das 1D-Modell die interne
            Übertragung überschätzt. Braucht SciPy (sonst Fallback 1D).
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
        center_gap=50e-6,
        n_through_holes=60,
        through_hole_diameter=1.0e-3,
        through_hole_pcd=None,
        through_hole_rings=None,
        n_blind_holes=30,
        blind_hole_diameter=1.2e-3,
        blind_hole_depth=None,
        blind_hole_pcd=None,
        blind_hole_rings=None,
        through_holes_stepped=False,
        # --- Akustische Netzwerke & Rückseite -------------------------------
        rear_network_enabled=True,
        rear_spacer_height=0.0,
        rear_plate_thickness=0.0,
        n_rear_plate_holes=0,
        rear_plate_hole_diameter=1.0e-3,
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
        # --- Spaltfilm-Modell -----------------------------------------------
        squeeze_model="1d",
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
        if arch.startswith(("dual_d", "k67", "doppelmembran")):
            self.architecture = "dual_diaphragm"
        elif arch.startswith("dual"):
            self.architecture = "dual"
        elif arch.startswith("single"):
            self.architecture = "single"
        else:
            raise ValueError("architecture muss 'single', 'dual' oder "
                             "'dual_diaphragm' sein.")
        # Anzahl der wandelnden Backplates (Gegentakt nur bei 'dual';
        # bei der K67-Bauform ist im Nierenmodus nur die vordere Seite
        # polarisiert)
        self.n_bp = 2 if self.architecture == "dual" else 1
        self.h_center = float(center_gap)
        if self.architecture == "dual_diaphragm" and self.h_center < 0:
            raise ValueError("center_gap darf nicht negativ sein.")
        # center_gap = 0 ist zulässig und beschreibt eine EINZELNE, komplett
        # durchbohrte Mittelelektrode (Braunmühl-Weber-Bauform, z. B.
        # Debenham/Robinson/Stebbings): die Durchgangslöcher beider Seiten
        # fluchten, es existiert keine laterale Zwischenschicht.
        # backplate_thickness ist dann die HALBE Plattendicke (je Seite).

        # Lochmuster: jeder Lochtyp (Durchgang/Blind) sitzt auf einem oder
        # MEHREREN Lochkreisen. Intern wird alles auf eine Ringliste
        # [(anzahl, lochkreisRADIUS oder None), ...] normiert; None = dieser
        # Anteil ist gleichmäßig über die Elektrode verteilt. Die Skalar-
        # Parameter n_*_holes/*_pcd sind der Ein-Ring-Sonderfall. Der
        # radiale Sitz steuert im 2D-Feldmodell die Verteilung der Loch-
        # leitwerte (der Versatz zwischen beiden Lochtypen bildet dort die
        # Laufzeitstrecke der Niere ab) und geht über die Porositäts-
        # profile in die Elektrostatik ein.
        def _normalize_rings(rings, n_scalar, pcd_scalar, label):
            if rings is None:
                rings = [(n_scalar, pcd_scalar)]
            out, n_tot = [], 0
            for entry in rings:
                try:
                    cnt, pcd = entry
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{label}: jeder Lochkreis braucht das Paar "
                        "(Anzahl, Lochkreis-Durchmesser)."
                    )
                cnt = int(cnt)
                if cnt < 0:
                    raise ValueError(
                        f"{label}: Lochanzahl darf nicht negativ sein.")
                r = None if pcd is None else 0.5 * float(pcd)
                if r is not None and not (0.0 <= r <= self.a_bp):
                    raise ValueError(
                        f"{label}: Lochkreisradius muss zwischen 0 und "
                        "Backplate-Radius liegen."
                    )
                out.append((cnt, r))
                n_tot += cnt
            return out, n_tot

        self._th_rings, self.n_th = _normalize_rings(
            through_hole_rings, n_through_holes, through_hole_pcd,
            "Durchgangslöcher")
        self.r_th = 0.5 * float(through_hole_diameter)
        if self.n_th > 0 and self.r_th <= 0:
            raise ValueError("Durchgangslochdurchmesser muss > 0 sein.")
        if self.n_th == 0 and self.architecture == "dual":
            raise ValueError(
                "Dual-Architektur ohne Durchgangslöcher: die vordere "
                "Backplate würde die Membran vollständig vom Schallfeld "
                "isolieren."
            )

        self._bh_rings, self.n_bh = _normalize_rings(
            blind_hole_rings, n_blind_holes, blind_hole_pcd, "Blindlöcher")
        self.r_bh = 0.5 * float(blind_hole_diameter)
        if blind_hole_depth is None:
            blind_hole_depth = 0.5 * self.t_bp
        self.d_bh = float(blind_hole_depth)
        if self.n_bh > 0 and not (0 < self.d_bh < self.t_bp):
            raise ValueError("Blindlochtiefe muss zwischen 0 und Backplate-Dicke liegen.")

        # Stufenbohrung (K67/K87): jedes Durchgangsloch sitzt konzentrisch
        # am Grund einer Senkung mit Sackloch-Geometrie (r_bh, d_bh); nur
        # die Restdicke t_bp − d_bh ist mit r_th eng durchbohrt.
        # n_through_holes zählt die gestuften Bohrungen, n_blind_holes nur
        # die reinen Sacklöcher.
        self.stepped = bool(through_holes_stepped) and self.n_th > 0
        if self.stepped:
            if not (0 < self.d_bh < self.t_bp):
                raise ValueError(
                    "Stufenbohrung: die Senkungstiefe (= Blindlochtiefe) "
                    "muss zwischen 0 und Backplate-Dicke liegen."
                )
            if self.r_bh <= self.r_th:
                raise ValueError(
                    "Stufenbohrung: die Senkung muss weiter sein als der "
                    "Kern (Sackloch-Ø > Durchgangsloch-Ø)."
                )
        # wirksame Länge der engen Durchgangsbohrung
        self.t_th_eff = self.t_bp - self.d_bh if self.stepped else self.t_bp

        # ------------------ Akustische Netzwerke & Rückseite ----------------
        self.rear_network_enabled = bool(rear_network_enabled)

        # Spacer + massive Rückplatte (K103-Bauform) direkt hinter der
        # Backplate; Teil der rückwärtigen Baugruppe.
        self.h_sp = float(rear_spacer_height)
        self.t_rp = float(rear_plate_thickness)
        self.n_rp = int(n_rear_plate_holes)
        self.r_rp = 0.5 * float(rear_plate_hole_diameter)
        if self.h_sp < 0 or self.t_rp < 0 or self.n_rp < 0:
            raise ValueError(
                "Spacer-Höhe, Rückplatten-Dicke und -Lochzahl dürfen "
                "nicht negativ sein."
            )
        if self.t_rp > 0 and self.n_rp > 0 and self.r_rp <= 0:
            raise ValueError("Rückplatten-Lochdurchmesser muss > 0 sein.")

        self.l_delay = float(delay_length)
        self.l_cav = float(cavity_length)
        self.t_cav_wall = float(cavity_wall_thickness)

        pos = str(cavity_hole_position).strip().lower()
        if pos not in ("circumference", "end"):
            raise ValueError("cavity_hole_position muss 'circumference' oder 'end' sein.")
        self.cavity_hole_position = pos
        self.n_ch = int(n_cavity_holes)
        self.r_ch = 0.5 * float(cavity_hole_diameter)
        # Die axiale Lochposition ergibt nur innerhalb des Hohlraums Sinn;
        # sie wird still in [0, cavity_length] geklammert (0 = am Eingang,
        # cavity_length = am geschlossenen Ende). Bei l_cav = 0 (kein
        # Hohlraum) fällt sie auf 0.
        if self.l_cav > 0.0:
            self.x_ch = min(max(float(cavity_hole_axial_position), 0.0),
                            self.l_cav)
        else:
            self.x_ch = 0.0

        self.rayl_front = float(fabric_front_rayl)
        self.rayl_rear = float(fabric_rear_rayl)

        self.body_diameter = (None if body_diameter is None
                              else float(body_diameter))
        self.include_diffraction = bool(include_diffraction)

        sm = str(squeeze_model).strip().lower()
        if sm not in ("1d", "2d"):
            raise ValueError("squeeze_model muss '1d' oder '2d' sein.")
        # Das 2D-Feldmodell (modifizierte Reynolds-Gleichung) braucht SciPy.
        self.squeeze_model = sm if (sm == "1d" or _HAS_SCIPY) else "1d"

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
        # RADIALGITTER FÜR DAS 2D-SPALTFILM-MODELL (modifizierte Reynolds-
        # Gleichung). Zellzentriertes Finite-Volumen-Gitter auf [0, a_bp];
        # die Zellzentren r_i = (i+1/2)*dr vermeiden die 1/r-Singularität
        # bei r=0. Die Löcher werden über die Elektrodenfläche homogenisiert,
        # das Membranprofil ist die Grundmode phi. (Steht VOR der Elektro-
        # statik, weil deren Porositätsprofile dieselben Dichten nutzen.)
        # ------------------------------------------------------------------
        N = 60
        dr = self.a_bp / N
        r_c = (np.arange(N) + 0.5) * dr
        self._fld_area = 2.0 * np.pi * r_c * dr           # Zellflächen [m^2]
        self._fld_phi = np.maximum(1.0 - (r_c / self.a_mem) ** 2, 0.0)
        self._fld_Sphi = float(np.sum(self._fld_phi * self._fld_area))
        # Geometriefaktor der lateralen Flächenleitwerte: Gface = 2*pi*k*K
        # (Fläche k liegt bei r = k*dr; Randflächen 0 = kein Fluss -> Neumann)
        gg = np.zeros(N + 1)
        gg[1:N] = 2.0 * np.pi * np.arange(1, N)
        self._fld_gface_geom = gg
        self._fld_S_elec = np.pi * self.a_bp**2
        self._fld_N = N

        # Radiale Dichteverteilungen der Löcher (normiert: Σ dens·A = 1),
        # damit die Gesamt-Lochleitwerte erhalten bleiben. Jeder Lochkreis
        # wird als schmales Ringband um seinen Radius konzentriert (ohne
        # Lochkreis: gleichmäßig über die Elektrode) und mit seinem Anteil
        # an der Gesamt-Lochzahl gewichtet — ein einzelner Ring reproduziert
        # exakt das bisherige Ein-PCD-Verhalten.
        def _hole_density(rings, n_total):
            if n_total <= 0:
                return np.ones(N) / float(np.sum(self._fld_area))
            width = max(0.10 * self.a_bp, 1.5 * dr)
            dens = np.zeros(N)
            for cnt, r_pcd in rings:
                if cnt <= 0:
                    continue
                band = (np.ones(N) if r_pcd is None
                        else np.exp(-0.5 * ((r_c - r_pcd) / width) ** 2))
                dens += cnt * band / float(np.sum(band * self._fld_area))
            return dens / n_total

        self._fld_dens_th = _hole_density(self._th_rings, self.n_th)
        self._fld_dens_bh = _hole_density(self._bh_rings, self.n_bh)

        # Elektrodenrand in Modenkoordinate u = r^2/a_mem^2
        self._ub = min((self.a_bp / self.a_mem) ** 2, 1.0)

        # Porositätsprofile für die ELEKTROSTATIK auf der Modenkoordinate:
        # lokale Lochflächenanteile aus denselben radialen Dichten wie im
        # Feldmodell — damit sind Lochkreise (PCD) auch in Pull-in, Feder-
        # Erweichung, Wandlerkoeffizient und C0 konsistent berücksichtigt.
        # Ohne PCD ergeben sich exakt die bisherigen konstanten Anteile.
        u_es = np.linspace(0.0, self._ub, 401)
        r_es = self.a_mem * np.sqrt(u_es)
        dth_es = np.interp(r_es, r_c, self._fld_dens_th)
        dbh_es = np.interp(r_es, r_c, self._fld_dens_bh)
        # Feldfreier Anteil (Durchbruch) und Blindloch-Anteil (Feldweg
        # g + Tiefe). Bei STUFENBOHRUNG zeigt die Stirnseite die WEITE
        # Senkung: nur der enge Kern ist feldfrei, der Senkungs-RING
        # wirkt wie ein Blindloch (gleiche Tiefe d_bh).
        p_th = self.n_th * np.pi * self.r_th**2 * dth_es
        p_bh = self.n_bh * np.pi * self.r_bh**2 * dbh_es
        if self.stepped:
            p_bh = p_bh + (self.n_th * np.pi
                           * (self.r_bh**2 - self.r_th**2) * dth_es)
        tot = p_th + p_bh
        scale = np.where(tot > 0.95, 0.95 / np.maximum(tot, 1e-30), 1.0)
        self._es_u = u_es
        self._es_c_solid = 1.0 - tot * scale
        self._es_c_blind = p_bh * scale

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
        if self.stepped:
            # Senkungsringe der Stufenbohrungen zählen zur Blind-Stirnfläche
            self.phi_bh += (self.n_th * np.pi
                            * (self.r_bh**2 - self.r_th**2) / self.S_bp)
        if self.phi_th + self.phi_bh >= 0.9:
            raise ValueError(
                "Durchgangs- und Blindlöcher bedecken >= 90 % der "
                "Backplate — keine wirksame Elektrode mehr."
            )
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
        # senken C0 geringfügig. Nur die ECHTE Dual-Backplate arbeitet im
        # Gegentakt (beide Seiten polarisiert, addiert bei w0 = 0 exakt
        # Faktor 2). Die K67-Doppelmembran (dual_diaphragm) ist im Nieren-
        # modus dagegen NUR frontseitig polarisiert — die Rückmembran liegt
        # auf Backplate-Potential und trägt weder zur Wandlung noch zur
        # Ruhekapazität bei (wie 'single'). Streu-/Kabelkapazität und
        # Verstärkerlast sind nicht modelliert (Leerlauf an der Kapsel).
        # ------------------------------------------------------------------
        theta = 0.0
        C0_rear = None
        signs = (+1.0, -1.0) if self.architecture == "dual" else (+1.0,)
        for sign in signs:
            I_F, _, I_C = self._electrode_integrals(sign * self.w0_static)
            C0 = EPS0 * I_C
            if C0_rear is None:
                C0_rear = C0
            theta += 2.0 * self.u_bias * EPS0 * I_F / (self.S_mem * C0)
        self.C_elec_0 = C0_rear   # Ruhekapazität der (hinteren) Backplate
        self._theta = theta

        # ------------------------------------------------------------------
        # RÜCKWIRKUNG DES ARBEITSPUNKTS AUF DEN SPALTFILM
        # Die polarisierte (Front-)Membran ist statisch zur Backplate hin
        # durchgebogen — ihr wirksamer Filmspalt ist kleiner (flächen-
        # gemittelt über die Elektrode: <phi> = 1 - ub/2). Wegen der
        # h³-Abhängigkeit des Filmwiderstands bricht das bei der Doppel-
        # membran-Bauform die Front/Rück-Symmetrie und koppelt die
        # Polarisationsspannung in REALISTISCHEM Maß an die Richt-
        # charakteristik. Bei 'dual' (beidseitig polarisiert) ist w0 = 0.
        # ------------------------------------------------------------------
        sag = self.w0_static * (1.0 - self._ub / 2.0)
        self.h_gap_front = max(self.h_gap - sag, 0.05 * self.h_gap)

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
            # ERWEITERUNG der Škvor-Formel: auch BLINDLÖCHER wirken im
            # Spaltfilm als verteilte Druck-Sammelstellen — die laterale
            # Strömung muss nur bis zur nächstgelegenen Bohrung laufen
            # (gleich welcher Art), nicht über die ganze Platte zu den
            # wenigen Durchgangslöchern. Der wirksame Squeeze-Film-
            # Widerstand wird daher mit ALLEN Bohrungen als Senken
            # gebildet (n_drain, q_drain). Ohne Blindlöcher fällt die
            # Formel auf das klassische Škvor-Ergebnis zurück.
            # Filmseitige Senkenöffnungen: bei STUFENBOHRUNG ist jede
            # Öffnung die weite Senkung (der enge Kern liegt am Grund) —
            # jede Stufenbohrung zählt als EINE Senke mit Radius r_bh.
            if self.stepped:
                q = (self.n_th + self.n_bh) * self.r_bh**2 / self.a_bp**2
            else:
                q = (self.n_th * self.r_th**2
                     + self.n_bh * self.r_bh**2) / self.a_bp**2
            if not (0.0 < q < 1.0):
                raise ValueError(
                    f"Lochflächenanteil q={q:.3f} der Bohrungen "
                    "muss in (0, 1) liegen."
                )
            self._q_drain = q
            self.R_A_gap = self._skvor_R(self.h_gap)          # nominal
            # wirksamer Widerstand der polarisierten (Front-)Seite mit
            # statisch verkleinertem Spalt
            self.R_A_gap_front = self._skvor_R(self.h_gap_front)
        else:
            # Geschlossene Backplate: es existiert kein Strömungspfad zu
            # Löchern, also auch keine laterale Škvor-Strömung (die Formel
            # divergiert für q -> 0). Das Spaltvolumen wirkt als reine
            # Nachgiebigkeit direkt an der Membran.
            self.R_A_gap = None
            self.R_A_gap_front = None

        # ------------------------------------------------------------------
        # NACHGIEBIGKEIT DES SPALTVOLUMENS
        # Der Spalt ist dünner als die thermische Grenzschicht
        # (delta_t ~ 0.1 mm bei 1 kHz) -> Kompression verläuft ISOTHERM:
        #     C_gap = V / P_atm      (statt adiabatisch V/(rho0*c^2))
        # ------------------------------------------------------------------
        V_gap = self.S_bp * self.h_gap
        self.C_A_gap = V_gap / P_ATM

        # Blindloch-Volumen: Bohrungen im mm-Maßstab sind DEUTLICH weiter
        # als die thermische Grenzschicht (~0.07 mm bei 1 kHz) — die
        # Kompression verläuft ADIABATISCH, C = V/(rho0*c^2). (Isotherm
        # wie im Membranspalt würde die Nachgiebigkeit um den Faktor
        # gamma = 1.4 überschätzen.)
        V_blind = self.n_bh * np.pi * self.r_bh**2 * self.d_bh
        self.C_A_blind = (V_blind / (RHO0 * C_AIR**2)
                          if self.n_bh > 0 else 0.0)
        # Senkungsvolumina der Stufenbohrungen (ebenfalls adiabatisch);
        # sie shunten wie Blindlöcher an der Membranseite des Spalts,
        # während der enge Kern den Serien-Durchgang bildet.
        if self.stepped:
            V_cb = self.n_th * np.pi * self.r_bh**2 * self.d_bh
            self.C_A_cb = V_cb / (RHO0 * C_AIR**2)
        else:
            self.C_A_cb = 0.0

        # ------------------------------------------------------------------
        # ZWISCHENSPALT DER K67-BAUFORM ("dual_diaphragm")
        # Dünne Luftschicht (Spacer, ~50 µm) zwischen den beiden Backplate-
        # Hälften. Die Strömung tritt durch die Durchgangslöcher der einen
        # Hälfte ein und durch die der anderen aus — die laterale
        # Poiseuille-Strömung zwischen den Lochmustern wird wie im
        # Membranspalt mit der Škvor-Formel (Spalthöhe = center_gap)
        # modelliert; das Schichtvolumen ist isotherm nachgiebig.
        # ------------------------------------------------------------------
        if (self.architecture == "dual_diaphragm" and self.n_th > 0
                and self.h_center > 0):
            q_c = self.n_th * self.r_th**2 / self.a_bp**2
            B_qc = q_c / 2.0 - q_c**2 / 8.0 - np.log(q_c) / 4.0 - 3.0 / 8.0
            self.R_A_center = (12.0 * MU_AIR
                               / (self.n_th * np.pi * self.h_center**3)
                               * B_qc)
            self.C_A_center = self.S_bp * self.h_center / P_ATM
        else:
            # h_center = 0: einteilige durchbohrte Mittelelektrode —
            # kein lateraler Strömungswiderstand, kein Zwischenvolumen
            self.R_A_center = 0.0
            self.C_A_center = 0.0

        # ------------------------------------------------------------------
        # SPACER + MASSIVE RÜCKPLATTE (K103-BAUFORM)
        # Dünne Luftschicht (Distanzring) zwischen Backplate-Rückseite und
        # einer massiven, gelochten Rückplatte — wie beim K103 (TLM 103):
        # K87-artige Front, die Rückseite ist statt einer Rückmembran durch
        # eine Platte abgeschlossen. Die Strömung tritt durch die
        # Durchgangslöcher der Backplate in den Spacer ein und durch die
        # Löcher der Rückplatte aus; wie beim K67-Zwischenspalt wird jede
        # Seite mit ihrer HALBEN Škvor-Zelle modelliert — hier aber mit dem
        # jeweils eigenen Lochmuster (Loch­zahlen/-radien beider Platten
        # dürfen sich unterscheiden):
        #     R_half = 6*mu*B(q) / (n * pi * h_sp^3),  q = n*r^2/a_bp^2
        # Das Schichtvolumen wirkt als (thermisch relaxierende) Shunt-
        # Nachgiebigkeit, s. _film_compliance_Y.
        # ------------------------------------------------------------------
        def _half_skvor(n, r_hole, h_film):
            q = min(n * r_hole**2 / self.a_bp**2, 1.0) if n > 0 else 0.0
            if n == 0 or q <= 0.0 or q >= 1.0:
                return 0.0     # keine/berührende Löcher: keine Engstelle
            B_q = max(q / 2.0 - q**2 / 8.0 - np.log(q) / 4.0 - 3.0 / 8.0,
                      0.0)
            return 6.0 * MU_AIR / (n * np.pi * h_film**3) * B_q

        if self.h_sp > 0.0:
            self.R_A_sp_in = _half_skvor(self.n_th, self.r_th, self.h_sp)
            self.R_A_sp_out = _half_skvor(self.n_rp, self.r_rp, self.h_sp)
        else:
            self.R_A_sp_in = 0.0
            self.R_A_sp_out = 0.0

        # Ist hinter Spacer/Rückplatte nichts mehr konfiguriert, münden die
        # Rückplattenlöcher direkt ins rückwärtige Schallfeld (K103-Fall).
        self._rear_tail_empty = (self.l_delay <= 0.0 and self.l_cav <= 0.0
                                 and (self.n_ch == 0 or self.r_ch <= 0.0))
        self._plate_vents = (self.rear_network_enabled
                             and self.architecture != "dual_diaphragm"
                             and self.t_rp > 0.0 and self.n_rp > 0
                             and self._rear_tail_empty)

        # Ist die Rückseite akustisch offen (Gradientenempfänger)?
        # Die Durchgangslöcher sind der einzige Weg durch die Backplate:
        if self.n_th == 0:
            # Backplate geschlossen -> hermetisch dicht, unabhängig von
            # allem, was dahinter montiert ist.
            self.rear_open = False
        elif self.architecture == "dual_diaphragm":
            # K67-Bauform: die passive Rückmembran überträgt rückwärtigen
            # Schall immer; Laufzeitglied/Hohlraum existieren nicht.
            self.rear_open = True
        elif not self.rear_network_enabled:
            # Keine rückwärtige Baugruppe: die Durchgangslöcher münden
            # direkt ins rückwärtige Schallfeld.
            self.rear_open = True
        elif self.t_rp > 0.0 and self.n_rp == 0:
            # Massive Rückplatte ohne Löcher: verschließt die Rückseite
            # unmittelbar hinter Backplate/Spacer — Druckempfänger.
            self.rear_open = False
        elif self._plate_vents:
            # Rückplattenlöcher münden direkt ins Schallfeld (K103-Fall).
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
        if self.architecture == "dual_diaphragm":
            # K67-Bauform: rückwärtiger Einlass = Rückmembran
            d_ax = 2.0 * (self.h_gap + self.t_bp) + self.h_center
        else:
            d_ax = self.h_gap + self.t_bp
            if self.rear_network_enabled:
                # Spacer + Rückplatte verschieben den Rückeinlass nach
                # hinten; münden die Plattenlöcher direkt (K103-Fall),
                # endet der Weg dort.
                d_ax += self.h_sp + self.t_rp
                if not self._plate_vents:
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
        # Auch der Rückeinlass der K67-Bauform (Rückmembran) wird in der
        # Beugungsrechnung als Ring bei seiner axialen Einbautiefe
        # modelliert: die Kapsel ist eine dünne Scheibe, deren Rückseite
        # nur d_rear_ax (~6 mm) hinter der Front liegt — eine Kalotte am
        # hinteren Kugelpol würde den effektiven Außenweg auf ~3*R_body
        # aufblähen und die Richtwirkung zerstören.

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
    # Spaltfilm-Grundgrößen
    # ======================================================================
    def _skvor_R(self, h_film):
        """Škvor-Squeeze-Film-Widerstand für die Spalthöhe ``h_film``.

        Alle Bohrungen (Durchgang + Blind) zählen als Senken; s. Kommentar
        in :meth:`_derive_parameters` (dort wird auch der Lochflächen-
        anteil q bestimmt — bei Stufenbohrung mit den weiten Senkungs-
        öffnungen). Der Spalt der polarisierten Seite ist durch die
        statische Durchbiegung kleiner -> größeres R (h³!).
        """
        n_drain = self.n_th + self.n_bh
        q = self._q_drain
        B_q = q / 2.0 - q**2 / 8.0 - np.log(q) / 4.0 - 3.0 / 8.0
        return 12.0 * MU_AIR / (n_drain * np.pi * h_film**3) * B_q

    def _film_compliance_Y(self, omega, h_film, S):
        """Shunt-Admittanz eines dünnen Luftvolumens (Fläche S, Höhe h)
        mit THERMISCHER RELAXATION (Tijdeman/Low-Reduced-Frequency):

            Y = jω · S·h / (n_p(ω) · P_atm),
            n_p = γ / [1 + (γ−1)·tanh(α_t)/α_t],  α_t = (h/2)·sqrt(jωρ0Pr/μ)

        n_p läuft von 1 (isotherm, dünner Spalt/tiefe Frequenz) nach γ
        (adiabatisch); der komplexe Übergang enthält die thermische
        Relaxationsdämpfung.
        """
        omega = np.asarray(omega, dtype=float)
        a_t = 0.5 * h_film * np.sqrt(1j * omega * RHO0 * PRANDTL / MU_AIR)
        n_p = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t) / a_t)
        return 1j * omega * S * h_film / (n_p * P_ATM)

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
        Die Porosität geht als RADIALES Profil ein (gleiche Lochdichten
        wie im Feldmodell, s. _derive_parameters): solide Elektrodenfläche
        wiegt mit c_solid(u), über Blindlöchern gilt der vergrößerte
        Feldweg g + Tiefe (Durchgangslöcher tragen nichts). Damit sind
        auch Lochkreise (PCD) in der Elektrostatik konsistent. w0 < 0
        beschreibt die von der Platte weg ausgelenkte Membran (vordere
        Backplate der Dual-Architektur).
        """
        u = self._es_u
        v = 1.0 - u                               # Modenprofil phi
        g_s = self.h_gap - w0 * v                 # Spalt, solide Elektrode
        g_b = self.h_gap + self.d_bh - w0 * v     # Feldweg über Blindloch
        c_s = self._es_c_solid
        c_b = self._es_c_blind
        S = self.S_mem
        I_F = S * np.trapezoid(c_s * v / g_s**2 + c_b * v / g_b**2, u)
        I_k = S * np.trapezoid(c_s * v**2 / g_s**3 + c_b * v**2 / g_b**3, u)
        I_C = S * np.trapezoid(c_s / g_s + c_b / g_b, u)
        return I_F, I_k, I_C

    def _static_residual(self, u_bias, w_grid):
        """k_gen·w0 − F_es(w0) für ein Array von Auslenkungen (vektorisiert)."""
        u = self._es_u
        v = 1.0 - u
        w2 = np.atleast_1d(w_grid)[:, None]
        g_s = self.h_gap - w2 * v[None, :]
        g_b = self.h_gap + self.d_bh - w2 * v[None, :]
        I_F = self.S_mem * np.trapezoid(
            self._es_c_solid * v / g_s**2
            + self._es_c_blind * v / g_b**2, u, axis=1)
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

    def _blind_hole_impedance(self, omega, count=None, C_vol=None):
        """Shunt-Impedanz von Sacklochvolumina in der Backplate.

        Blindlöcher vergrößern das wirksame Luftvolumen unter der Membran
        und entlasten so den Squeeze-Film (weniger Dämpfung, klassischer
        Trick bei Großmembran-Backplates). Modell: viskose Rohrimpedanz
        über die HALBE Tiefe (mittlere Eindringtiefe der Strömung in ein
        geschlossenes Loch) in Serie mit der isothermen Nachgiebigkeit des
        vollen Lochvolumens:
            Z_blind = Z_tube(r, d/2) + 1/(j*omega*C_blind)
        Einseitige Mündungskorrektur (nur membranseitige Öffnung).
        Ohne ``count``/``C_vol`` die reinen Blindlöcher; mit Argumenten
        auch für die SENKUNGEN der Stufenbohrungen nutzbar (gleiche
        Geometrie r_bh/d_bh, eigene Anzahl und Nachgiebigkeit).
        """
        if count is None:
            count, C_vol = self.n_bh, self.C_A_blind
        omega = np.asarray(omega, dtype=float)
        Z_visc = self._hole_impedance(
            omega, self.r_bh, 0.5 * self.d_bh, count, end_correction=False
        )
        S = np.pi * self.r_bh**2
        Z_end = 1j * omega * RHO0 * (0.85 * self.r_bh) / (S * count)
        Z_comp = 1.0 / (1j * omega * C_vol)
        return Z_visc + Z_end + Z_comp

    def _through_hole_impedance(self, omega, count, radiates=False):
        """Serienimpedanz der Durchgangsbohrungen der Backplate.

        Normale Bohrung: Zwikker–Kosten-Rohr über die volle Plattendicke
        mit beidseitig angeflanschter Mündungskorrektur. STUFENBOHRUNG:
        eng gebohrt ist nur die Restdicke t_bp − d_bh unter der Senkung;
        die äußere Mündung ist angeflanscht (0.85·r), die innere mündet
        in die weite Senkung — ihre Mündungsmasse trägt den Karal-Faktor
        (1 − r_th/r_bh) der Querschnittsstufe. Das Senkungsvolumen selbst
        shuntet als Sacklochvolumen an der Membranseite
        (s. _blind_hole_impedance mit C_A_cb).
        """
        omega = np.asarray(omega, dtype=float)
        if not self.stepped:
            return self._hole_impedance(omega, self.r_th, self.t_bp, count,
                                        end_correction=True,
                                        radiates=radiates)
        Z = self._hole_impedance(omega, self.r_th, self.t_th_eff, count,
                                 end_correction=False, radiates=radiates)
        S = np.pi * self.r_th**2
        delta = 0.85 * self.r_th * (2.0 - self.r_th / self.r_bh)
        return Z + 1j * omega * RHO0 * delta / (S * count)

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

    def _membrane_impedance_passive(self, omega):
        """Serienimpedanz der PASSIVEN Rückmembran (K67-Bauform, Niere).

        Im Nierenmodus liegt die Rückmembran auf Backplate-Potential —
        kein Feld, keine Feder-Erweichung: es gilt die unpolarisierte
        Nachgiebigkeit C_A_mem (gleiches Material/Tuning wie vorn).
        """
        omega = np.asarray(omega, dtype=float)
        R = np.sqrt(self.M_A_mem / self.C_A_mem) / self._Q_MEMBRANE_INTERNAL
        return (
            R + 1j * omega * self.M_A_mem
            + 1.0 / (1j * omega * self.C_A_mem)
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
    @staticmethod
    def _abcd_inv(T):
        """Kehrmatrix eines reziproken (2,2,N)-Zweitors (det = 1):
        T^-1 = [[D, -B], [-C, A]]. Für die gespiegelte Kettenrichtung."""
        return np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]])

    def _gap_field_2port(self, omega, h_film=None):
        """Zweitor des Luftspalts aus der modifizierten Reynolds-Gleichung.

        ``h_film``: wirksame Spalthöhe (Standard: nomineller Luftspalt);
        die polarisierte Seite übergibt hier ihren statisch verkleinerten
        Spalt h_gap_front.

        MODIFIZIERTE REYNOLDS-GLEICHUNG (2D-Feldmodell)
        -----------------------------------------------
        Statt eines einzelnen Lumped-Widerstands wird das Druckfeld p(r)
        im dünnen Spalt als Feldgleichung gelöst (Homentcovschi & Miles,
        JASA 116, 2004; Bao et al.). Axialsymmetrisch, mit über die
        Elektrode homogenisierten Löchern:

            ∇·[(h³/12μ) ∇p] - Y(r)·p = -v(r) + g_h(r)·p_rear

        Bilanz je Ringzelle (Finite-Volumen):
          * laterale Filmströmung: frequenzabhängiger Leitwert je Breite
                K_f(ω) = h/(jωρ0) · [1 − tanh(α_v)/α_v],
                α_v = (h/2)·sqrt(jωρ0/μ)
            (Schlitz-Pendant der Zwikker–Kosten-Lösung: für ω→0 die
            Poiseuille-Leitung h³/(12μ), bei hohen Frequenzen dominiert
            die TRÄGHEIT der Spaltluft — bei 25 kHz/60 µm beträgt der
            Unterschied Faktor ~4 mit −73° Phase)
          * Speicherung im Spaltvolumen mit thermischer Relaxation:
                c_gap(ω) = h/(n_p(ω)·P_atm),
                n_p = γ / [1 + (γ−1)·tanh(α_t)/α_t],  α_t = α_v·sqrt(Pr)
            (polytroper Übergang isotherm -> adiabatisch, Tijdeman/LRF)
          * Blindlöcher: LOKALE Admittanz y_bh (Rohr + adiab. Nachgiebig-
            keit), koppeln NICHT zur Rückseite
          * Durchgangslöcher: Admittanz g_h, koppeln zum rückwärtigen Port
            (Druckdifferenz p - p_rear)
          * Membran treibt mit der Modenform v(r) = φ(r)·U/∫φ dA

        ZELL-ENGSTELLENWIDERSTAND: die axialsymmetrische Homogenisierung
        kann die azimutale Strömungskonvergenz zu den DISKRETEN Löchern
        nicht auflösen (Validierung: ohne Korrektur fehlen ~99 % des
        Škvor-Widerstands im Grenzfall dichter Löcher). Nach dem Zellen-
        ansatz der modifizierten Reynolds-Gleichung (Bao; Homentcovschi &
        Miles) erhält deshalb jede Bohrung den Engstellenwiderstand ihrer
        Zelle in Serie:
                R_cell(ω) = B(q_c) / (π·K_f(ω)),
                q_c = (n_th + n_bh)·r_loch² / a_bp²
        (B = Škvor-Zellfunktion; alle Bohrungen teilen sich die Zellen).
        Damit reproduziert das Feldmodell im Grenzfall dichter, gleich-
        verteilter Löcher exakt das Škvor-Ergebnis, und bei spärlichen
        oder ringförmig sitzenden Löchern kommt die im Gitter aufgelöste
        plattenweite Ausbreitung additiv hinzu.

        Der entscheidende Fortschritt gegenüber dem 1D-Modell: der
        Nachgiebigkeits-Rückweg (Spaltvolumen + Blindlöcher, überall lokal)
        und der Rückkopplungsweg (nur durch die wenigen Durchgangslöcher)
        sehen UNTERSCHIEDLICHE Widerstände — im 1D-Lumped teilen sie sich
        einen einzigen R_gap. Bei wenigen, engen Durchgangslöchern
        (Braunmühl-Weber-Platten) überschätzt 1D deshalb die interne
        Übertragung; das Feldmodell erfasst zudem den radialen Druckaufbau.

        Das Zweitor wird durch zwei Feldlösungen extrahiert (Membran treibt
        bei kurzgeschlossenem Rückport / Rückport treibt bei ruhender
        Membran) und ist per Konstruktion reziprok (det T = 1). Rückgabe
        als (2,2,N)-Kettenmatrix in Richtung Membran -> Außen, gleiche
        Konvention wie :meth:`_backplate_gap_abcd`.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        Nf = omega.size
        S_el = self._fld_S_elec

        A = self._fld_area                               # (N,)
        phi = self._fld_phi
        Sphi = self._fld_Sphi
        N = self._fld_N
        dens_th = self._fld_dens_th                       # Σ dens·A = 1
        dens_bh = self._fld_dens_bh

        # Filmleitwert mit viskoser Trägheit und polytrope Kompressibilität
        # (s. Docstring); beide sind komplex und frequenzabhängig.
        h = self.h_gap if h_film is None else h_film
        a_v = 0.5 * h * np.sqrt(1j * omega * RHO0 / MU_AIR)
        K_f = h / (1j * omega * RHO0) * (1.0 - np.tanh(a_v) / a_v)
        a_t = a_v * np.sqrt(PRANDTL)
        n_poly = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t) / a_t)
        c_gap = h / (n_poly * P_ATM)                     # (Nf,) komplex

        # Zell-Engstellenwiderstand je Bohrung (Škvor-Zellfunktion, alle
        # Bohrungen teilen sich die Zellen; q_c >= 1 -> Löcher berühren
        # sich, keine Engstelle mehr)
        n_wells = self.n_th + self.n_bh

        def _cell_B(r_hole):
            q_c = min(n_wells * r_hole**2 / self.a_bp**2, 1.0)
            if q_c >= 1.0:
                return 0.0
            return max(q_c / 2.0 - q_c**2 / 8.0
                       - np.log(q_c) / 4.0 - 3.0 / 8.0, 0.0)

        # pro-Loch-Impedanzen (vektorisiert über omega); die Lochleitwerte
        # werden über die radialen Dichten dens_th/dens_bh verteilt.
        # Mündungskorrekturen: die filmseitige Ausbreitung übernimmt der
        # Zell-Engstellenwiderstand R_cell — die Flansch-Mündungsmasse
        # 0.85·r wird deshalb nur EINSEITIG (Portseite) angesetzt; für
        # Blindlöcher (Öffnung nur zum Film) entfällt sie ganz.
        # STUFENBOHRUNG: enges Rohr nur über die Restdicke, innere Mündung
        # mit Karal-Stufenfaktor; die filmseitige Zelle sieht die WEITE
        # Senkungsöffnung, deren Volumen zusätzlich (auf der Durchgangs-
        # dichte) wie ein Blindloch shuntet.
        S_th = np.pi * self.r_th**2
        r_well_th = self.r_bh if self.stepped else self.r_th
        Z_th1 = (self._hole_impedance(omega, self.r_th, self.t_th_eff, 1,
                                      end_correction=False)
                 + 1j * omega * RHO0 * (0.85 * self.r_th) / S_th
                 + _cell_B(r_well_th) / (np.pi * K_f))
        if self.stepped:
            # weites Senkungssegment in Serie + Karal-Stufenmündung
            # (die filmseitige Ausbreitung deckt die Zelle mit r_bh ab)
            Z_th1 = (Z_th1
                     + self._hole_impedance(omega, self.r_bh, self.d_bh, 1,
                                            end_correction=False)
                     + 1j * omega * RHO0 * 0.85 * self.r_th
                     * (1.0 - self.r_th / self.r_bh) / S_th)
        g_tot = self.n_th / Z_th1                         # Gesamtleitwert (Nf,)
        if self.n_bh > 0:
            Z_v = self._hole_impedance(omega, self.r_bh, 0.5 * self.d_bh, 1,
                                       end_correction=False)
            Z_comp = self.n_bh / (1j * omega * self.C_A_blind)
            y_tot = self.n_bh / (Z_v + Z_comp
                                 + _cell_B(self.r_bh) / (np.pi * K_f))
        else:
            y_tot = np.zeros(Nf, dtype=complex)
        if self.stepped:
            # Senkungsvolumina der Stufenbohrungen (sitzen auf dens_th)
            Z_v_cb = self._hole_impedance(omega, self.r_bh, 0.5 * self.d_bh,
                                          1, end_correction=False)
            Z_comp_cb = self.n_th / (1j * omega * self.C_A_cb)
            y_cb = self.n_th / (Z_v_cb + Z_comp_cb
                                + _cell_B(self.r_bh) / (np.pi * K_f))
        else:
            y_cb = np.zeros(Nf, dtype=complex)

        src_a = phi * A / Sphi                            # Membran treibt (U=1)
        T = np.empty((2, 2, Nf), dtype=complex)
        ab = np.zeros((3, N), dtype=complex)
        gg = self._fld_gface_geom
        for f in range(Nf):
            Gface = gg * K_f[f]                           # (N+1,) komplex
            ab[0, 1:] = -Gface[1:N]                       # Superdiagonale
            ab[2, :-1] = -Gface[1:N]                      # Subdiagonale
            g_h = g_tot[f] * dens_th                      # (N,) verteilt
            y_bh = y_tot[f] * dens_bh + y_cb[f] * dens_th
            Y = 1j * omega[f] * c_gap[f] + y_bh + g_h
            ab[1, :] = Gface[:N] + Gface[1:N + 1] + Y * A
            rhs = np.column_stack((src_a, g_h * A))       # (N, 2)
            sol = _solve_banded((1, 1), ab, rhs)
            p_a, p_b = sol[:, 0], sol[:, 1]
            alpha = np.sum(p_a * phi * A) / Sphi
            beta = np.sum(p_b * phi * A) / Sphi
            gamma = np.sum(g_h * A * p_a)
            delta = np.sum(g_h * A * (p_b - 1.0))
            T[0, 0, f] = beta - alpha * delta / gamma
            T[0, 1, f] = alpha / gamma
            T[1, 0, f] = -delta / gamma
            T[1, 1, f] = 1.0 / gamma
        return T

    def _backplate_gap_abcd(self, omega, outside_to_membrane,
                            holes_radiate=False, polarized=False):
        """Kettenmatrix des Backplate/Luftspalt-Netzwerks.

        Topologie von der Membran aus gesehen:
            Membran --[Shunt: Blindlöcher]--[Serie: R_gap (Škvor)]--
            --[Shunt: C_gap]--[Serie: Durchgangslöcher]-- außen
        ``outside_to_membrane=True`` liefert die umgekehrte Kettenrichtung
        (für die vordere Backplate der Dual-Architektur).
        ``holes_radiate=True``: die Durchgangslöcher münden direkt ins
        Freifeld (keine rückwärtige Baugruppe) -> Strahlungswiderstand.
        ``polarized=True``: dieser Spalt gehört zur polarisierten Membran
        — es gilt der statisch verkleinerte Spalt h_gap_front (größerer
        Filmwiderstand, h³). Die Spalt-Nachgiebigkeit wird polytrop
        (isotherm -> adiabatisch) gerechnet, s. _film_compliance_Y.

        Sonderfall geschlossene Backplate (n_th = 0): kein Strömungspfad
        durch die Platte und keine laterale Škvor-Strömung — Spaltvolumen
        und Blindlöcher wirken als reine Shunt-Nachgiebigkeiten an der
        Membran; die Kette endet dahinter blockiert.
        """
        h_eff = self.h_gap_front if polarized else self.h_gap

        # 2D-Feldmodell: das komplette Spalt-/Lochnetzwerk kommt aus der
        # Reynolds-Feldlösung (nur sinnvoll, wenn Durchgangslöcher da sind).
        if self.squeeze_model == "2d" and self.n_th > 0:
            T = self._gap_field_2port(omega, h_film=h_eff)
            if holes_radiate:
                k = np.asarray(omega, float) / C_AIR
                S_holes = self.n_th * np.pi * self.r_th**2
                Z_rad = (RHO0 * C_AIR / S_holes) * np.minimum(
                    (k * self.r_th) ** 2 / 2.0, 1.0)
                T = self._mmul(T, self._abcd_series(Z_rad, omega))
            if outside_to_membrane:
                T = self._abcd_inv(T)
            return T

        Y_gap = self._film_compliance_Y(omega, h_eff, self.S_bp)
        mats = []  # Reihenfolge: Membranseite -> Außenseite
        if self.n_bh > 0:
            mats.append(self._abcd_shunt(1.0 / self._blind_hole_impedance(omega), omega))
        if self.n_th == 0:
            mats.append(self._abcd_shunt(Y_gap, omega))
            return reduce(self._mmul, mats)
        Z_holes = self._through_hole_impedance(omega, self.n_th,
                                               radiates=holes_radiate)
        mats.append(self._abcd_series(self._skvor_R(h_eff), omega))
        mats.append(self._abcd_shunt(Y_gap, omega))
        if self.stepped:
            # Senkungssegment der Stufenbohrungen in SERIE: weites Rohr
            # mit spaltseitiger Mündung, dann shuntet das Senkungsvolumen,
            # bevor der enge Kern (Z_holes) folgt. Grenzfall Senkung -> 0
            # reproduziert exakt die normale Durchgangsbohrung.
            S_cb = np.pi * self.r_bh**2
            Z_wide = (self._hole_impedance(omega, self.r_bh, self.d_bh,
                                           self.n_th, end_correction=False)
                      + 1j * omega * RHO0 * (0.85 * self.r_bh)
                      / (S_cb * self.n_th))
            mats.append(self._abcd_series(Z_wide, omega))
            mats.append(self._abcd_shunt(1j * omega * self.C_A_cb, omega))
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
            # vordere Backplate (identisch zur hinteren, gespiegelt; beide
            # Seiten polarisiert, aber w0 = 0 -> h_eff = h)
            front.append(self._backplate_gap_abcd(
                omega, outside_to_membrane=True, polarized=True))
        T_front = reduce(self._mmul, front)

        # ---------------------------- Membran ------------------------------
        T_mem = self._abcd_series(self._membrane_impedance(omega), omega)

        # ------------- hinterer Zweig: Membran -> rückwärtiger Port --------
        if self.architecture == "dual_diaphragm":
            # K67-BAUFORM: vordere Membran -> vorderer Spalt/Backplate ->
            # Zwischenspalt (Spacer) -> hintere Backplate/Spalt -> PASSIVE
            # Rückmembran -> Gewebe -> Abstrahlung -> rückwärtiges Feld.
            # Die Rückmembran ersetzt Laufzeitglied und Hohlraum: ihre
            # Nachgiebigkeit bildet mit den Spalt-/Lochwiderständen das
            # Phasenschiebernetzwerk der Niere.
            # Frontspalt: polarisierte Seite (statisch verkleinerter Spalt
            # -> Symmetriebruch, realistische Bias-Wirkung aufs Pattern)
            rear = [self._backplate_gap_abcd(omega, outside_to_membrane=False,
                                             polarized=True)]
            if self.n_th > 0:
                if self.squeeze_model == "2d" and self.h_center > 0:
                    # Zwischenspalt feldkonsistent: Ein-/Austritts-
                    # Engstelle je Durchgangsloch (Zellfunktion mit dem
                    # frequenzabhängigen Filmleitwert K_f des Spacers;
                    # versetzte Locharrays angenommen) + polytrope
                    # Nachgiebigkeit des Schichtvolumens.
                    hc = self.h_center
                    a_v = 0.5 * hc * np.sqrt(1j * omega * RHO0 / MU_AIR)
                    K_fc = hc / (1j * omega * RHO0) * (1 - np.tanh(a_v) / a_v)
                    q_cc = min(self.n_th * self.r_th**2 / self.a_bp**2, 1.0)
                    B_cc = max(q_cc / 2 - q_cc**2 / 8
                               - np.log(q_cc) / 4 - 3.0 / 8.0, 0.0) \
                        if q_cc < 1.0 else 0.0
                    Z_c_half = B_cc / (np.pi * K_fc) / self.n_th
                    Y_c = self._film_compliance_Y(omega, hc, self.S_bp)
                    center = [self._abcd_series(Z_c_half, omega),
                              self._abcd_shunt(Y_c, omega),
                              self._abcd_series(Z_c_half, omega)]
                else:
                    center = [
                        self._abcd_series(0.5 * self.R_A_center, omega),
                        self._abcd_shunt(self._film_compliance_Y(
                            omega, self.h_center, self.S_bp)
                            if self.h_center > 0 else 0.0, omega),
                        self._abcd_series(0.5 * self.R_A_center, omega),
                    ]
                rear += center + [
                    self._backplate_gap_abcd(omega, outside_to_membrane=True,
                                             polarized=False),
                    self._abcd_series(self._membrane_impedance_passive(omega),
                                      omega),
                    self._abcd_series(self.rayl_rear / self.S_mem, omega),
                    self._abcd_series(
                        self._radiation_impedance_membrane(omega), omega),
                ]
            T_rear = reduce(self._mmul, rear)
            T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
            return T_total, T_rear

        # Die Durchgangslöcher der Backplate sind der Zugang zur Rückseite.
        # Drei Fälle:
        #   n_th = 0                  -> hermetisch dicht direkt am Spalt
        #   keine rückwärtige Baugruppe -> Löcher münden (durchs Gewebe)
        #                                direkt ins rückwärtige Schallfeld
        #   Baugruppe vorhanden       -> Gewebe -> Laufzeitglied -> Hohlraum
        vents_directly = self.n_th > 0 and not self.rear_network_enabled
        # Einzel-Backplate: der (einzige) Spalt gehört zur polarisierten
        # Membran -> statisch verkleinerter effektiver Spalt
        rear = [self._backplate_gap_abcd(omega, outside_to_membrane=False,
                                         holes_radiate=vents_directly,
                                         polarized=True)]

        if self.n_th == 0:
            # geschlossene Backplate: Port unmittelbar blockiert
            # (Gewebe/Laufzeitglied/Hohlraum sind akustisch unerreichbar)
            T_rear = reduce(self._mmul, rear)
            T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
            return T_total, T_rear

        if vents_directly:
            # kein Laufzeitglied/Hohlraum: Gewebe liegt über den Öffnungen,
            # Port = rückwärtiges Schallfeld
            rear.append(self._abcd_series(self.rayl_rear / self.S_bp, omega))
            T_rear = reduce(self._mmul, rear)
            T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
            return T_total, T_rear

        # ------- Spacer + massive Rückplatte (K103-Bauform) ----------------
        # Dünner Distanzspalt hinter der Backplate; die Strömung tritt über
        # die Backplate-Durchgangslöcher ein und die Rückplattenlöcher aus
        # (je halbe Škvor-Zelle mit dem eigenen Lochmuster, s.
        # _derive_parameters), das Schichtvolumen shuntet dazwischen.
        plate = self.t_rp > 0.0
        if self.h_sp > 0.0:
            Y_sp = self._film_compliance_Y(omega, self.h_sp, self.S_bp)
            if plate and self.n_rp > 0:
                rear.append(self._abcd_series(self.R_A_sp_in, omega))
                rear.append(self._abcd_shunt(Y_sp, omega))
                rear.append(self._abcd_series(self.R_A_sp_out, omega))
            else:
                # ohne (gelochte) Rückplatte wirkt der Spacer nur als
                # zusätzliches Luftvolumen (axialer Durchtritt, kein
                # nennenswerter lateraler Widerstand)
                rear.append(self._abcd_shunt(Y_sp, omega))
        if plate:
            if self.n_rp == 0:
                # Rückplatte ohne Löcher: Rückseite hier verschlossen —
                # alles Dahinterliegende ist akustisch unerreichbar
                # (rear_open=False blockiert den Port).
                T_rear = reduce(self._mmul, rear)
                T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
                return T_total, T_rear
            # Durchgangslöcher der Rückplatte: thermoviskoses Rohr über
            # die Plattendicke; münden sie direkt ins Schallfeld
            # (K103-Fall), kommt die Strahlungsimpedanz hinzu.
            Z_rp = self._hole_impedance(omega, self.r_rp, self.t_rp,
                                        self.n_rp, end_correction=True,
                                        radiates=self._plate_vents)
            rear.append(self._abcd_series(Z_rp, omega))

        # Gewebe hinter der Backplate/Rückplatte (überspannt die Fläche,
        # liegt im Direktmündungsfall über den Plattenöffnungen)
        rear.append(self._abcd_series(self.rayl_rear / self.S_bp, omega))

        if self._plate_vents:
            # K103-Fall: hinter der Rückplatte folgt nichts mehr —
            # Port = rückwärtiges Schallfeld an den Plattenlöchern
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
    @staticmethod
    def _ring_note(rings):
        """Kurzform eines Lochmusters für summary(): Anzahl je Lochkreis."""
        parts = [(f"{cnt} gleichmäßig" if r is None
                  else f"{cnt} auf LK ⌀{2e3 * r:.1f} mm")
                 for cnt, r in rings if cnt > 0]
        return "; ".join(parts) if parts else "keine"

    def summary(self):
        """Mehrzeilige Übersicht der abgeleiteten Modellparameter."""
        sens = self.transfer_function(1000.0)[0]
        arch_note = {
            "single": "1 Backplate",
            "dual": "2 Backplates, Gegentakt",
            "dual_diaphragm": "K67-Bauform, passive Rückmembran",
        }[self.architecture]
        lines = [
            "MicrophoneCapsule — abgeleitete Parameter",
            "-" * 55,
            f"Architektur:                  {self.architecture} ({arch_note})",
            f"Spaltfilm-Modell:             {self.squeeze_model} "
            + ("(modifizierte Reynolds-Feldlösung)" if self.squeeze_model == "2d"
               else "(Lumped-Element)"),
            f"Membranfläche:                {self.S_mem * 1e6:9.2f} mm²",
            f"akust. Masse Membran M_A:     {self.M_A_mem:9.2f} kg/m⁴",
            f"akust. Nachgiebigkeit C_A:    {self.C_A_mem:9.3e} m³/Pa",
            f"  dto. effektiv (mit Bias):   {self.C_A_eff:9.3e} m³/Pa",
            f"Feder-Erweichung durch Bias:  {self.softening_ratio * 100:9.2f} %",
            f"statische Durchbiegung w0:    {self.w0_static * 1e6:9.2f} µm "
            f"(Restspalt Mitte {self.h_min_static * 1e6:.1f} µm)",
            f"wirksamer Frontspalt h_eff:   {self.h_gap_front * 1e6:9.2f} µm "
            f"(nominal {self.h_gap * 1e6:.1f} µm)",
            ("Pull-in-Spannung U_PI:        "
             + (f"{self.U_pullin:9.1f} V" if np.isfinite(self.U_pullin)
                else "     > 20 kV")),
            f"Elektroden-Porosität:         {100 * (self.phi_th + self.phi_bh):9.1f} % "
            f"(Durchgang {100 * self.phi_th:.1f} %, Blind {100 * self.phi_bh:.1f} %)",
            f"Lochmuster Durchgang:         {self.n_th:6d} × ⌀{2e3 * self.r_th:.2f} mm "
            f"({self._ring_note(self._th_rings)})"
            + (f" — Stufenbohrung: Kern {self.t_th_eff * 1e3:.2f} mm unter "
               f"⌀{2e3 * self.r_bh:.2f}-mm-Senkung" if self.stepped else ""),
            f"Lochmuster Blind:             {self.n_bh:6d} × ⌀{2e3 * self.r_bh:.2f} mm "
            f"({self._ring_note(self._bh_rings)})",
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
        ]
        if (self.architecture != "dual_diaphragm"
                and self.rear_network_enabled
                and (self.h_sp > 0.0 or self.t_rp > 0.0)):
            sp = (f"Spacer {self.h_sp * 1e6:.0f} µm" if self.h_sp > 0
                  else "kein Spacer")
            if self.t_rp > 0:
                rp = f"Rückplatte {self.t_rp * 1e3:.2f} mm"
                rp += (f", {self.n_rp} × ⌀{2e3 * self.r_rp:.2f} mm"
                       if self.n_rp > 0 else ", ohne Löcher (dicht)")
                if self._plate_vents:
                    rp += " → Schallfeld"
            else:
                rp = "keine Rückplatte"
            lines.append(f"Spacer/Rückplatte (K103):     {sp}; {rp}")
        if self.architecture == "dual_diaphragm":
            # Druckleck der Doppelmembran-Bauform: die Rückmembran liegt
            # als Nachgiebigkeit in SERIE im rückwärtigen Pfad; das innere
            # Luftpolster (Spalte, Blindlöcher, Zwischenspalt) zweigt den
            # frequenzunabhängigen Anteil δ = C_int/C_mem des Flusses ab.
            # Unterhalb von f_δ (wo 1.5·k·d_ext = δ) dominiert das Leck
            # und die Richtwirkung geht in Richtung Kugel.
            C_int = (2.0 * self.C_A_gap
                     + 2.0 * (self.C_A_blind + self.C_A_cb)
                     + self.C_A_center)
            delta = C_int / self.C_A_mem
            f_floor = delta * C_AIR / (2.0 * np.pi * 1.5 * self.d_ext)
            lines += [
                f"Druckleck δ = C_int/C_mem:    {delta:9.4f}",
                f"Pattern-Untergrenze f_δ:      {f_floor:9.1f} Hz "
                "(darunter -> Kugel)",
            ]
        lines += [
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

    # --------- Gegenprobe 6: K67-Bauform (Doppelmembran) -------------------
    # Zwei Membranen außen, Backplates innen (center_gap): die passive
    # Rückmembran bildet das Phasenschiebernetzwerk -> Nierencharakteristik
    # ohne Laufzeitglied/Hohlraum; die Gegentakt-Mode beider Membranen
    # gegen das innere Luftpolster erzeugt die K67-typische Präsenz-
    # anhebung im 10-kHz-Bereich.
    k67 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_tension=13.7, air_gap=60e-6, backplate_diameter=25e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=60, through_hole_diameter=1.2e-3,
        n_blind_holes=60, blind_hole_diameter=1.8e-3, blind_hole_depth=1.1e-3,
        fabric_front_rayl=2500.0, fabric_rear_rayl=1500.0,
        body_diameter=34e-3,
    )
    di_k = k67.directivity(frequencies_hz=(1000.0,))
    pk67 = di_k["patterns"][1000.0]["db"]
    assert pk67[180] < -15.0, "K67-Bauform muss Nierencharakteristik zeigen"
    fr_k = k67.frequency_response(n_points=150)
    assert np.all(np.isfinite(fr_k["amplitude_db"]))
    fk, ak = fr_k["frequency_hz"], fr_k["amplitude_db_norm"]
    ihf = (fk > 5000) & (fk < 16000)
    assert 1.0 < np.max(ak[ihf]) < 8.0, \
        "moderate Präsenzanhebung erwartet (U87Ai-Kurve: +2..3 dB)"
    print(f"K67-Bauform: Niere (180° = {pk67[180]:.1f} dB @1 kHz), "
          f"Präsenzanhebung +{np.max(ak[ihf]):.1f} dB bei "
          f"{fk[ihf][np.argmax(ak[ihf])]/1000:.1f} kHz  OK")

    # --------- Gegenprobe 7: Elektrostatik der Doppelmembran ---------------
    # Im Nierenmodus ist NUR die Frontmembran polarisiert: die passive
    # Rückmembran benutzt die unpolarisierte Nachgiebigkeit (keine Feder-
    # Erweichung), und Wandlung/Ruhekapazität rechnen einseitig (n_bp = 1,
    # anders als die echte Gegentakt-Dual-Backplate mit n_bp = 2).
    assert k67.n_bp == 1, "K67-Bauform: nur die Frontseite ist polarisiert"
    Z_front = k67._membrane_impedance(np.array([2 * np.pi * 1000.0]))[0]
    Z_pass = k67._membrane_impedance_passive(np.array([2 * np.pi * 1000.0]))[0]
    assert abs(Z_front - Z_pass) > 0, "Rückmembran muss unpolarisiert sein"
    # Die Polarisation wirkt NUR in realistischem Maß auf das Richtdiagramm:
    # über den statisch verkleinerten Frontspalt (h³-Filmwiderstand), nicht
    # über die Feder-Erweichung (die kürzt sich als Serienelement aus dem
    # normierten Muster). Erwartung: wenige dB Verschiebung bei 180°, kein
    # Umkippen des Patterns.
    def _p180(bias):
        c = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_tension=13.7, air_gap=60e-6, backplate_diameter=25e-3,
            bias_voltage=bias, architecture="dual_diaphragm", center_gap=50e-6,
            n_through_holes=60, through_hole_diameter=1.2e-3,
            n_blind_holes=60, blind_hole_diameter=1.8e-3,
            blind_hole_depth=1.1e-3, fabric_front_rayl=2500.0,
            fabric_rear_rayl=1500.0, body_diameter=34e-3)
        return c.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
    d20, d60 = _p180(20.0), _p180(60.0)
    assert d20 < -12.0 and d60 < -12.0, "Niere muss bei beiden Spannungen bestehen"
    assert abs(d20 - d60) < 8.0, \
        "Bias-Wirkung aufs Richtdiagramm muss im realistischen Rahmen bleiben"
    print(f"Elektrostatik Doppelmembran: nur Front polarisiert (n_bp=1); "
          f"Bias-Wirkung aufs Pattern realistisch begrenzt "
          f"(180° @1 kHz: {d20:.1f} dB @20 V -> {d60:.1f} dB @60 V)  OK")

    # --------- Gegenprobe 8: 2D-Spaltfilmmodell (Reynolds-Feld) ------------
    if _HAS_SCIPY:
        # a) Das Feld-Zweitor ist reziprok (det T = 1) und passiv
        #    (Re Z_in >= 0 über das Band).
        cap2d = MicrophoneCapsule(architecture="single", n_through_holes=12,
                                  through_hole_diameter=0.5e-3,
                                  rear_network_enabled=False,
                                  squeeze_model="2d")
        oms = 2.0 * np.pi * np.logspace(1.0, np.log10(25000.0), 40)
        Tg = cap2d._gap_field_2port(oms)
        det = Tg[0, 0] * Tg[1, 1] - Tg[0, 1] * Tg[1, 0]
        assert np.max(np.abs(det - 1.0)) < 1e-6, \
            "Reynolds-Zweitor muss reziprok sein"
        assert np.min((Tg[0, 1] / Tg[1, 1]).real) > 0.0, \
            "Reynolds-Zweitor muss passiv sein"
        # b) Grenzfall dichte, gleichverteilte Löcher: dank des Zell-
        #    Engstellenwiderstands muss das Feldmodell bei tiefer Frequenz
        #    den Škvor-Spaltwiderstand reproduzieren (das Gitter fügt die
        #    real aufgelöste radiale Ausbreitung des Membranprofils hinzu,
        #    daher Toleranz nach oben).
        cd = MicrophoneCapsule(architecture="single", n_through_holes=60,
                               through_hole_diameter=1.2e-3, n_blind_holes=0,
                               air_gap=60e-6, rear_network_enabled=False,
                               squeeze_model="2d")
        om1 = np.array([2.0 * np.pi * 20.0])
        Tg1 = cd._gap_field_2port(om1)
        Z_h = cd._hole_impedance(om1, cd.r_th, cd.t_bp, cd.n_th,
                                 end_correction=True)[0]
        ratio = (Tg1[0, 1, 0] - Z_h).real / cd.R_A_gap
        assert 0.8 < ratio < 1.6, \
            f"2D muss im dichten Grenzfall Škvor reproduzieren (ratio={ratio:.2f})"
        print(f"2D-Reynolds-Feld: reziprok/passiv über das Band; dichter "
              f"Grenzfall 2D/Škvor = {ratio:.2f}  OK")

    # --------- Gegenprobe 9: Mehrfach-Lochkreise ---------------------------
    # Die Ringlisten sind eine reine VERALLGEMEINERUNG der Skalar-Parameter:
    # a) ein einzelner Ring muss exakt dem Skalar-PCD entsprechen,
    # b) mehrere gleichmäßige Anteile exakt der Gleichverteilung.
    _base = dict(architecture="single", backplate_diameter=20e-3,
                 n_blind_holes=30, blind_hole_diameter=1.2e-3)
    _fchk = np.array([100.0, 1000.0, 10000.0])
    ref_pcd = MicrophoneCapsule(n_through_holes=12, through_hole_pcd=16e-3,
                                **_base)
    one_ring = MicrophoneCapsule(through_hole_rings=[(12, 16e-3)], **_base)
    assert np.allclose(ref_pcd.transfer_function(_fchk)[0],
                       one_ring.transfer_function(_fchk)[0], rtol=1e-12), \
        "1 Lochkreis muss dem Skalar-PCD exakt entsprechen"
    ref_uni = MicrophoneCapsule(n_through_holes=12, **_base)
    two_uni = MicrophoneCapsule(through_hole_rings=[(8, None), (4, None)],
                                **_base)
    assert np.allclose(ref_uni.transfer_function(_fchk)[0],
                       two_uni.transfer_function(_fchk)[0], rtol=1e-12), \
        "gleichmäßige Ringanteile müssen der Gleichverteilung entsprechen"
    # c) Mehrere echte Ringe: Gesamtzahlen = Summen, Dichten normiert
    #    (Σ dens·A = 1, Lochleitwerte bleiben erhalten), Ergebnis endlich —
    #    auch im 2D-Feldmodell.
    multi = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        through_hole_rings=[(6, 17e-3), (3, 12e-3), (3, 6e-3)],
        through_hole_diameter=0.7e-3,
        blind_hole_rings=[(12, 17e-3), (12, 12e-3), (6, 6e-3)],
        blind_hole_diameter=1.0e-3,
        squeeze_model="2d" if _HAS_SCIPY else "1d",
    )
    assert multi.n_th == 12 and multi.n_bh == 30
    for _dens in (multi._fld_dens_th, multi._fld_dens_bh):
        assert abs(np.sum(_dens * multi._fld_area) - 1.0) < 1e-9, \
            "Ring-Dichteprofil muss auf Σ dens·A = 1 normiert sein"
    fr_multi = multi.frequency_response(20.0, 20000.0, n_points=60)
    assert np.all(np.isfinite(fr_multi["amplitude_db"]))
    print(f"Mehrfach-Lochkreise: 1 Ring ≡ Skalar-PCD, gleichmäßige Anteile "
          f"≡ Gleichverteilung; {multi.n_th}+{multi.n_bh} Löcher auf 3+3 "
          f"Kreisen ({multi.squeeze_model}-Modell) lauffähig  OK")

    # --------- Gegenprobe 10: Spacer + Rückplatte (K103-Bauform) -----------
    # a) Spacer/Rückplatte = 0 muss exakt dem bisherigen Verhalten
    #    entsprechen (reine Erweiterung).
    ref10 = MicrophoneCapsule()
    zero10 = MicrophoneCapsule(rear_spacer_height=0.0,
                               rear_plate_thickness=0.0,
                               n_rear_plate_holes=0)
    assert np.allclose(ref10.transfer_function(_fchk)[0],
                       zero10.transfer_function(_fchk)[0], rtol=1e-12), \
        "Spacer/Rückplatte = 0 darf nichts ändern"
    # b) Rückplatte OHNE Löcher verschließt die Rückseite: Druckempfänger
    #    (ohne Beugung exakt Kugelcharakteristik), egal was dahinter kommt.
    sealed_rp = MicrophoneCapsule(rear_spacer_height=80e-6,
                                  rear_plate_thickness=2e-3,
                                  n_rear_plate_holes=0,
                                  include_diffraction=False)
    assert not sealed_rp.rear_open
    H0 = sealed_rp.transfer_function(1000.0, angle_deg=0.0)[0]
    H180 = sealed_rp.transfer_function(1000.0, angle_deg=180.0)[0]
    assert abs(abs(H180) / abs(H0) - 1.0) < 1e-9, \
        "dichte Rückplatte muss Druckempfänger (Kugel) ergeben"
    # c) K103-artige Konfiguration: Membran -> Backplate -> Spacer ->
    #    gelochte massive Rückplatte -> Schallfeld (Laufzeitglied/Hohlraum
    #    leer). Die Rückseite ist offen, die Plattenlöcher münden direkt,
    #    der Außenweg endet an der Rückplatte, das Pattern ist gerichtet.
    k103 = MicrophoneCapsule(
        membrane_resonance_hz=1500.0, membrane_diameter=25e-3,
        air_gap=50e-6, backplate_diameter=23e-3, backplate_thickness=3e-3,
        bias_voltage=45.0, architecture="single",
        n_through_holes=24, through_hole_diameter=0.8e-3,
        n_blind_holes=30, blind_hole_diameter=1.2e-3,
        rear_network_enabled=True,
        rear_spacer_height=50e-6, rear_plate_thickness=2e-3,
        n_rear_plate_holes=30, rear_plate_hole_diameter=0.5e-3,
        delay_length=0.0, cavity_length=0.0, n_cavity_holes=0,
        include_diffraction=False,
    )
    assert k103.rear_open and k103._plate_vents
    assert abs(k103.d_ext - (50e-6 + 3e-3 + 50e-6 + 2e-3)) < 1e-12, \
        "d_ext muss an der direkt mündenden Rückplatte enden"
    H0 = k103.transfer_function(1000.0, angle_deg=0.0)[0]
    H180 = k103.transfer_function(1000.0, angle_deg=180.0)[0]
    ratio_k103 = abs(H180) / abs(H0)
    assert ratio_k103 < 0.5, \
        f"K103-Konfiguration muss richten (180°/0° = {ratio_k103:.2f})"
    fr_k103 = k103.frequency_response(20.0, 20000.0, n_points=60)
    assert np.all(np.isfinite(fr_k103["amplitude_db"]))
    print(f"Spacer/Rückplatte (K103): 0-Werte ≡ Bestand, dichte Platte -> "
          f"Kugel, K103-Konfiguration richtet (180°/0° @1 kHz = "
          f"{20 * np.log10(ratio_k103):.1f} dB)  OK")

    # --------- Gegenprobe 11: Stufenbohrung (K67/K87) ----------------------
    # a) Grenzfall verschwindende Senkung (Tiefe -> 0, Senkungs-Ø knapp
    #    über Kern-Ø): muss die normale Durchgangsbohrung reproduzieren.
    base11 = dict(architecture="single", backplate_diameter=20e-3,
                  n_through_holes=24, through_hole_diameter=1.0e-3,
                  n_blind_holes=0, rear_network_enabled=False)
    plain11 = MicrophoneCapsule(**base11)
    tiny11 = MicrophoneCapsule(through_holes_stepped=True,
                               blind_hole_diameter=1.02e-3,
                               blind_hole_depth=0.02e-3, **base11)
    H_p = plain11.transfer_function(_fchk)[0]
    H_t = tiny11.transfer_function(_fchk)[0]
    assert np.max(np.abs(H_t / H_p - 1.0)) < 0.05, \
        "verschwindende Senkung muss die normale Bohrung reproduzieren"
    # b) K67-Geometrie (verifiziert: je Seite 120 Bohrungen 1.3 mm x
    #    3.7 mm in der 4-mm-Halbplatte, jede zweite mit 0.6-mm-Durchbruch
    #    am Grund -> 60 gestufte + 60 reine Sacklöcher): das enge Rohr
    #    ist nur noch t_bp − Tiefe = 0.3 mm lang -> deutlich kleinere
    #    Durchgangsimpedanz; Stirnporosität zählt die Senkungsringe;
    #    die Niere der Doppelmembran-Bauform bleibt erhalten.
    k67_kwargs = dict(
        membrane_material="pet", membrane_resonance_hz=1150.0,
        membrane_diameter=26e-3, membrane_thickness=6e-6,
        membrane_tension=13.7, air_gap=65e-6, backplate_diameter=25e-3,
        backplate_thickness=4e-3, bias_voltage=60.0,
        architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=60, blind_hole_diameter=1.3e-3,
        blind_hole_depth=3.7e-3,
        fabric_front_rayl=300.0, fabric_rear_rayl=200.0,
        body_diameter=34e-3,
    )
    k67_pl = MicrophoneCapsule(**k67_kwargs)
    k67_st = MicrophoneCapsule(through_holes_stepped=True, **k67_kwargs)
    om1k = np.array([2.0 * np.pi * 1000.0])
    Zst = abs(k67_st._through_hole_impedance(om1k, 60)[0])
    Zpl = abs(k67_pl._through_hole_impedance(om1k, 60)[0])
    assert Zst < 0.85 * Zpl, \
        "Stufenbohrung muss die Durchgangsimpedanz senken (kürzeres Rohr)"
    assert k67_st.phi_bh > k67_pl.phi_bh    # Senkungsringe in der Porosität
    assert k67_st.C_A_cb > 0.0
    # Ruhekapazität trifft den nachgemessenen K67-Wert (~50 pF)
    assert 47.0 < k67_st.C_elec_0 * 1e12 < 54.0, \
        f"K67-C0 = {k67_st.C_elec_0 * 1e12:.1f} pF (erwartet ~50 pF)"
    H0 = k67_st.transfer_function(1000.0, angle_deg=0.0)[0]
    H180 = k67_st.transfer_function(1000.0, angle_deg=180.0)[0]
    st_180 = 20.0 * np.log10(abs(H180) / abs(H0))
    assert st_180 < -12.0, \
        f"K67 mit Stufenbohrung muss Niere bleiben (180° = {st_180:.1f} dB)"
    # c) 2D-Feldmodell mit Stufenbohrung: reziprok und endlich
    if _HAS_SCIPY:
        st2d = MicrophoneCapsule(through_holes_stepped=True,
                                 squeeze_model="2d", **k67_kwargs)
        Tg2 = st2d._gap_field_2port(2.0 * np.pi
                                    * np.logspace(1.5, 4.3, 12))
        det2 = Tg2[0, 0] * Tg2[1, 1] - Tg2[0, 1] * Tg2[1, 0]
        assert np.max(np.abs(det2 - 1.0)) < 1e-6
        fr_st = st2d.frequency_response(20.0, 20000.0, n_points=40)
        assert np.all(np.isfinite(fr_st["amplitude_db"]))
    print(f"Stufenbohrung: Grenzfall ≡ normale Bohrung, K67-Stufengeometrie "
          f"|Z_th| um {100 * (1 - Zst / Zpl):.0f} % kleiner, Niere bleibt "
          f"(180° @1 kHz = {st_180:.1f} dB), 2D reziprok  OK")

    print("\nAlle Testläufe erfolgreich — Arrays werden korrekt berechnet.")
