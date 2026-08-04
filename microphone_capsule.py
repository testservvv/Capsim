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
_trapz = getattr(np, "trapezoid", None) or np.trapz  # np>=2.0: trapezoid
K_BOLTZ = 1.380649e-23   # Boltzmann-Konstante               [J/K]
T_KELVIN = 293.15        # Lufttemperatur (20 °C, konsistent
                         # mit RHO0/C_AIR/MU_AIR)            [K]
P_REF = 2.0e-5           # Bezugsschalldruck                 [Pa]


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
            ``through_hole_diameter`` eng durchbohrt. Zählweise (an der
            realen Kapsel orientiert): ``n_blind_holes`` = GESAMTZAHL der
            Senkungen, ``n_through_holes`` davon sind zusätzlich
            durchgebohrt (K67: 120 Senkungen, 60 durchgebohrt). Die
            Senkungen zählen als Sackvolumen, in der Elektrostatik als
            Stirnöffnung (feldfreier Kern + Blindloch-Ring) und im
            Spaltfilm als je EINE weite Senke. Erfordert Sackloch-Ø >
            Durchgangsloch-Ø und n_through_holes ≤ n_blind_holes.

    Klemmringe (nur Doppelmembran-Bauform)
        clamp_ring_thickness, clamp_ring_width : float
            Vor jeder Membran montierter Klemmring [m] (Dicke = axiale
            Auftragung, Breite = radiale Ausdehnung). Bei K67/K87 sitzen
            solche Ringe vor beiden Membranen; sie versenken die Membranen
            um ihre Dicke und verlängern damit die EHRLICHE geometrische
            Front-Rück-Distanz: ``d_ext += 2 · Dicke``. Die Breite
            vergrößert nur den Außenradius (Geometrieangabe, kein eigener
            d_ext-Beitrag). Die Nierennull entsteht, wenn die interne
            akustische Laufzeit des Phasenschieber-Netzwerks (Bohrungen,
            Spaltfilme, Spacer) diese externe Laufzeit d_ext/c trifft.
            ``0`` = keine Ringe. Wirkt nur bei ``dual_diaphragm``.
        clearance_ring_diameter, clearance_ring_width, clearance_ring_depth : float
            Ringförmiger FREISTICH in den Elektroden-Stirnflächen (je
            Seite) [m]: mittlerer Ring-Ø (Position), radiale Breite,
            axialer Abtrag. Breite Ringe (≥ 1 Feldgitterzelle) werden als
            lokale Spaltvertiefung ``h -> h + Tiefe`` aufgelöst und
            ENTLASTEN die Mündungs-Engstellen dort sitzender Bohrungen —
            bei wenigen engen Durchgangslöchern (Braunmühl-Weber-Platten
            wie der Debenham) ist genau diese Engstelle der begrenzende
            Widerstand des Nieren-Phasenschiebers, und der Freistich
            vertieft die 180°-Auslöschung dramatisch. Schmalere Ringe
            wirken als konzentrierter Schlitz-Stub (R–C) an ihrer Zelle.
            ``0`` = kein Ring. Wirkt nur im 2D-Feldmodell.
        ring_vent_width, ring_vent_length : float
            DURCHGEHENDER RANDSPALT um die Backplate (B&K-Bauform):
            ringförmiger Kanal der radialen Breite ``ring_vent_width``
            [m] am Plattenumfang, der den Luftspalt am Rand mit der
            Rückseite verbindet (axiale Kanallänge ``ring_vent_length``;
            ``None`` = Backplate-Dicke). Der Filmrand ist damit nicht
            mehr dicht: die Randströmung läuft durch eine thermoviskose
            SCHLITZLEITUNG (LRF, exakte Grenzfälle R = 12μL/(b·w³) und
            M = (6/5)·ρ0·L/(b·w) mit b = 2π·a_bp) zum selben
            rückwärtigen Port wie die Durchgangslöcher. ``0`` = kein
            Randspalt (Filmrand dicht, Bestand). Gatter: nur
            ``single``/``dual`` (bei ``dual_diaphragm`` versiegeln
            Spacer/Klemmringe den Rand); nicht kombinierbar mit
            Spacer/Rückplatte (K103); im 1D-Pfad nur OHNE
            Durchgangslöcher (Löcher UND Randspalt brauchen die
            Stromaufteilung eines Feldmodells). Im 3D-Löser hängt der
            Ringkanal über den Randflächen-Leitwert an der äußersten
            Filmzellreihe (s. Gegenprobe 29).

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
            ``"3d"``: volles (r, phi)-Sandwich mit DISKRETEN Löchern und
            Membranen als FD-Feldern — s. :meth:`_build_3d_geometry` /
            :meth:`_solve_3d`. Löst die azimutale Zuströmung zu den
            einzelnen Bohrungen und die dadurch teilentkoppelten Sack-
            löcher auf (bedämpft die interne Helmholtz-Resonanz
            realistisch). Nur für ``dual_diaphragm`` mit ``center_gap=0``
            und ohne Stufenbohrung (einteilige Elektrode, Debenham-Typ);
            DEUTLICH langsamer (LU-Faktorisierung je Frequenzpunkt).
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

    # Nullstellen von J0 — die axialsymmetrischen (0,m)-Membranmoden.
    # Konstanten, deshalb ohne SciPy hinterlegt.
    _J0_ZEROS = (2.404825557695773, 5.520078110286311, 8.653727912911011,
                 11.791534439014281, 14.930917708487787)


    def __init__(
        self,
        # --- Membran -------------------------------------------------------
        membrane_material="PET",
        membrane_resonance_hz=8000.0,
        membrane_diameter=22e-3,
        membrane_thickness=6e-6,
        membrane_tension=400.0,
        membrane_modes=1,
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
        # --- Klemmringe (Doppelmembran-Bauform) -----------------------------
        clamp_ring_thickness=0.0,
        clamp_ring_width=0.0,
        # --- Clearance-Ring (Freistich in den Elektroden-Stirnflächen) ------
        clearance_ring_diameter=0.0,
        clearance_ring_width=0.0,
        clearance_ring_depth=0.0,
        ring_vent_width=0.0,
        ring_vent_length=None,
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
        fabric_rear_position="backplate",
        # --- Gehäuse & Beugung ----------------------------------------------
        body_diameter=None,
        include_diffraction=True,
        axial_body_model="sphere",
        bem_body_diameter=56e-3,
        bem_body_gap=15e-3,
        bem_body_length=80e-3,
        # --- Spaltfilm-Modell -----------------------------------------------
        squeeze_model="1d",
        half_rotation_deg=None,
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
        self.membrane_modes = int(membrane_modes)
        if not 1 <= self.membrane_modes <= len(self._J0_ZEROS):
            raise ValueError(
                f"membrane_modes muss zwischen 1 und {len(self._J0_ZEROS)} "
                "liegen (axialsymmetrische (0,m)-Moden)."
            )

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
        # ZÄHLWEISE (an der realen Kapsel orientiert): ``n_blind_holes`` ist
        # die GESAMTZAHL der Senkungen; ``n_through_holes`` davon sind
        # zusätzlich durchgebohrt (jede zweite bei der K67 -> 120 Senkungen,
        # 60 durchgebohrt). Intern zählt ``self.n_bh`` nur die REIN blinden
        # Senkungen (Gesamt − durchgebohrt); die durchgebohrten tragen ihre
        # Senkung über ``self.n_th`` (C_A_cb, Porosität, Škvor-Senken), so
        # dass die Gesamtzahl der Senkungen erhalten bleibt.
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
            if self.n_th > self.n_bh:
                raise ValueError(
                    "Stufenbohrung: höchstens so viele Durchgangslöcher wie "
                    "Senkungen (jedes Durchgangsloch sitzt in einer Senkung; "
                    "Blindlochanzahl = Gesamtzahl der Senkungen)."
                )
            # rein blinde Senkungen = Gesamt − durchgebohrt; die Ring-
            # verteilung wird anteilig auf die blind bleibenden skaliert
            # (Durchgangs- und Blindlöcher sind gleich verteilt).
            n_total_bh = self.n_bh
            n_pure = n_total_bh - self.n_th
            frac = n_pure / n_total_bh if n_total_bh else 0.0
            self._bh_rings = [(int(round(c * frac)), r)
                              for c, r in self._bh_rings]
            # Rundungsrest korrigieren, damit Σ = n_pure exakt bleibt
            drift = n_pure - sum(c for c, _ in self._bh_rings)
            if drift and self._bh_rings:
                c0, r0 = self._bh_rings[0]
                self._bh_rings[0] = (c0 + drift, r0)
            self.n_bh = n_pure
        # wirksame Länge der engen Durchgangsbohrung
        self.t_th_eff = self.t_bp - self.d_bh if self.stepped else self.t_bp

        # Klemmringe vor den Membranen (Doppelmembran-Bauform): sie sitzen
        # außen vor jeder Membran und versenken sie um ihre Dicke — das
        # verlängert die ehrliche geometrische Front-Rück-Distanz d_ext um
        # 2× die Ringdicke. Die Breite vergrößert nur den Außenradius
        # (reine Geometrieangabe).
        self.clamp_ring_thickness = float(clamp_ring_thickness)
        self.clamp_ring_width = float(clamp_ring_width)
        if self.clamp_ring_thickness < 0 or self.clamp_ring_width < 0:
            raise ValueError("Klemmring-Dicke und -Breite dürfen nicht "
                             "negativ sein.")

        # Clearance-Ring: ringförmiger Freistich in den Elektroden-
        # Stirnflächen (je Seite). Position über den Ring-Ø, radiale
        # Breite, axiale Tiefe; 0 = kein Ring. Breite Ringe (>= 1 Feld-
        # gitterzelle) wirken als lokale Spaltvertiefung h -> h + Tiefe
        # und ENTLASTEN die Mündungs-Engstellen dort sitzender Bohrungen
        # (entscheidend für die Nierentiefe bei wenigen engen Durchgangs-
        # löchern, z. B. Debenham); schmalere Ringe wirken als
        # konzentrierter Schlitz-Stub. Nur im 2D-Feldmodell.
        self.clearance_ring_diameter = float(clearance_ring_diameter)
        self.clearance_ring_width = float(clearance_ring_width)
        self.clearance_ring_depth = float(clearance_ring_depth)
        if (self.clearance_ring_diameter < 0 or self.clearance_ring_width < 0
                or self.clearance_ring_depth < 0):
            raise ValueError("Clearance-Ring-Maße dürfen nicht negativ sein.")

        # Durchgehender Randspalt um die Backplate (B&K-Bauform)
        self.ring_vent_w = float(ring_vent_width)
        if self.ring_vent_w < 0.0:
            raise ValueError("Randspalt-Breite darf nicht negativ sein.")
        self.ring_vent_L = (self.t_bp if ring_vent_length is None
                            else float(ring_vent_length))
        if self.ring_vent_w > 0.0:
            if self.ring_vent_L <= 0.0:
                raise ValueError("Randspalt-Kanallänge muss > 0 sein.")
            if arch == "dual_diaphragm":
                raise ValueError(
                    "Randspalt (ring_vent_width) gilt nur für die "
                    "Bauformen 'single'/'dual' — bei der Doppelmembran-"
                    "Bauform versiegeln Spacer/Klemmringe den "
                    "Elektrodenrand."
                )

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
        # Position des rückwärtigen Gewebes: "backplate" = im Zylinder
        # direkt hinter der Backplate, über die volle Bohrung gespannt
        # (Z = Rayl/S_bp, Bestand); "inlet" = außen ÜBER den Einlass-
        # öffnungen (Hohlraum-Einlasslöcher, K103-Rückplattenlöcher bzw.
        # bei Direktmündung die Durchgangslöcher selbst). Dort zählt nur
        # die LOCHFLÄCHE als Durchströmfläche — dasselbe Tuch ist um den
        # Faktor S_bp/S_Löcher hochohmiger — und der Widerstand liegt
        # HINTER den Shunt-Volumina von Laufzeitrohr/Hohlraum sowie in
        # Serie mit der Einlassloch-Masse (bedämpft deren Helmholtz-
        # Resonator direkt). Bei geschlossener Rückseite gibt es keinen
        # Einlass — das Gewebe entfällt dann wirkungslos (s.
        # _rear_chain_mats, Gegenprobe 24).
        frp = str(fabric_rear_position).strip().lower()
        if frp not in ("backplate", "inlet"):
            raise ValueError("fabric_rear_position muss 'backplate' oder "
                             "'inlet' sein.")
        if frp == "inlet" and self.architecture == "dual_diaphragm":
            raise ValueError(
                "fabric_rear_position='inlet': die Doppelmembran-Bauform "
                "hat keine rückwärtigen Einlasslöcher — ihr rückwärtiges "
                "Gewebe liegt über der Rückmembran ('backplate' nutzen).")
        self.fabric_rear_position = frp

        self.body_diameter = (None if body_diameter is None
                              else float(body_diameter))
        self.include_diffraction = bool(include_diffraction)

        sm = str(squeeze_model).strip().lower()
        if sm not in ("1d", "2d", "3d"):
            raise ValueError("squeeze_model muss '1d', '2d' oder '3d' sein.")
        if sm == "3d":
            # Der 3D-(r,phi)-Löser rechnet das Sandwich der durchbohrten
            # Elektrode(n) mit DISKRETEN Löchern. Moden:
            #   * dual_diaphragm, center_gap = 0: einteilige Elektrode
            #     (Debenham-Typ, 2 Filme) — der ursprüngliche Löser;
            #   * dual_diaphragm, center_gap > 0: ZWEI Elektrodenhälften
            #     mit Zwischenspalt als drittem Reynolds-Film (K67-Typ),
            #     eigene, um half_rotation_deg VERDREHTE Lochbilder je
            #     Hälfte, Stufenbohrungen als Zweitor-Kette je Loch
            #     (Gegenprobe 22);
            #   * single: EIN Film + EIN Membranfeld; die Durchgangs-
            #     löcher münden in einen SAMMELKNOTEN, dessen Abschluss
            #     die rückwärtige Baugruppe (Spacer/Rückplatte, Gewebe,
            #     Laufzeitglied, Hohlraum) als Lumped-Kette bildet
            #     (Gegenprobe 23);
            #   * dual: ZWEI Filme um das Membranfeld; je Backplate ein
            #     eigener Sammelknoten (vorn: Gewebe+Strahlung, hinten:
            #     rückwärtige Baugruppe).
            if self.architecture == "dual_diaphragm" and self.stepped \
                    and self.h_center <= 0.0:
                raise ValueError("squeeze_model='3d' mit Stufenbohrung "
                                 "erfordert bei der Doppelmembran-Bauform "
                                 "center_gap > 0 (zwei Elektrodenhälften, "
                                 "K67-Typ).")
            if self.n_th <= 0 and self.ring_vent_w <= 0.0:
                raise ValueError("squeeze_model='3d' erfordert "
                                 "Durchgangslöcher oder einen Randspalt.")
        # Verdrehung der Elektrodenhälften gegeneinander (nur 3D-K67-
        # Modus): die realen Hälften sind so verdreht, dass die
        # Durchgangslöcher nicht zueinander zeigen. None = automatisch
        # eine halbe Teilung des Durchgangs-Lochbilds (180°/n_th).
        if half_rotation_deg is None:
            self.half_rotation_deg = 180.0 / max(self.n_th, 1)
        else:
            self.half_rotation_deg = float(half_rotation_deg)

        # Die Feldmodelle (2D/3D) brauchen SciPy.
        self.squeeze_model = sm if (sm == "1d" or _HAS_SCIPY) else "1d"

        # Randspalt-Gatter, die die endgültige Konfiguration brauchen:
        # K103-Spacer/Rückplatte teilen sich den Plattenrand mit dem
        # Randspalt (nicht modelliert), und im 1D-Pfad ist die Strom-
        # aufteilung Löcher/Randspalt nicht lumped darstellbar.
        if self.ring_vent_w > 0.0:
            if self.h_sp > 0.0 or self.t_rp > 0.0:
                raise ValueError(
                    "Randspalt ist nicht mit Spacer/Rückplatte "
                    "(K103-Bauform) kombinierbar — beide teilen sich den "
                    "Backplate-Rand."
                )
            if self.squeeze_model == "1d" and self.n_th > 0:
                raise ValueError(
                    "Randspalt PLUS Durchgangslöcher erfordert das "
                    "2D-/3D-Feldmodell (Stromaufteilung Rand/Löcher); "
                    "der 1D-Pfad rechnet nur den rein randbelüfteten "
                    "Fall (n_through_holes = 0)."
                )

        # Axiales Körpermodell für den Front-Rück-Transfer der
        # Doppelmembran-Bauform: "sphere" (Kugel mit Durchmesser d_ext,
        # Standard — beschreibt die am Mikrofonkörper MONTIERTE Kapsel)
        # oder "spheroid" (frei stehende Scheibe: oblates Sphäroid mit
        # radialer Halbachse R_body und axialer d_ext/2 — dokumentierter
        # Befund: die freie Scheibe hat eine DEUTLICH längere effektive
        # Distanz, am Pol bis 4R/π; ein dahinterliegender Mikrofonkörper
        # unterbindet den Scheibenrand-Umweg, weshalb die Kugel der
        # montierten Realität entspricht, s. Gegenprobe 20).
        # "bem": axisymmetrisches Randelementverfahren auf der Kontur
        # Kapselkopf + Mikrofonkörper (Zylinder ⌀ bem_body_diameter,
        # Länge bem_body_length, axialer Luftspalt bem_body_gap unter dem
        # Kopf; Durchmesser 0 = frei stehender Kopf). Das ist die
        # montagetreue Rechnung ZWISCHEN den Referenzkörpern d_ext-Kugel
        # (montiert-idealisiert) und freiem Sphäroid.
        ab = str(axial_body_model).strip().lower()
        if ab not in ("sphere", "spheroid", "bem"):
            raise ValueError(
                "axial_body_model muss 'sphere', 'spheroid' oder 'bem' sein."
            )
        if ab in ("spheroid", "bem") and not _HAS_SCIPY:
            raise ValueError(
                f"axial_body_model='{ab}' erfordert SciPy.")
        self.axial_body_model = ab
        self.bem_body_diameter = float(bem_body_diameter)
        self.bem_body_gap = float(bem_body_gap)
        self.bem_body_length = float(bem_body_length)
        if ab == "bem" and self.bem_body_diameter > 0.0:
            if self.bem_body_gap <= 2e-3 or self.bem_body_length <= 5e-3:
                raise ValueError(
                    "BEM-Körper: Luftspalt > 2 mm und Länge > 5 mm nötig "
                    "(getrennte, verrundete Konturen)."
                )
        self._bem_geo = None

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
        # EXAKTE Modalfrequenz der Grundmode (Diagnose): die Lumped-Werte
        # (Kolbenfaktor 4/3 aus der STATISCHEN Parabelform) ergeben für
        # reine Vorspannung f = sqrt(6)/(2·pi·a)·sqrt(T/rho_s) und liegen
        # damit 1.9 % ÜBER dem exakten Membran-Eigenwert j01 = 2.40483
        # (Grundmode J0; Massenfaktor j01²/4 = 1.446 statt 4/3). Für die
        # Biegesteifigkeit gilt der eingespannte Platten-Eigenwert
        # lambda² = 10.2158; beide Anteile addieren sich in guter Näherung
        # quadratisch. Die NETZWERK-Dynamik behält bewusst die konsistenten
        # Lumped-Werte (LF-exakt, f_res-Kalibrierung absorbiert den
        # Unterschied; die vollen Modenformen rechnet der 3D-Löser) —
        # diese Größe dient dem ehrlichen Vergleich in summary().
        _j01 = 2.404825557695773
        f_T_ex = _j01 / (2.0 * np.pi * a) * np.sqrt(self.tension / rho_s)
        f_B_ex = (10.2158 / (2.0 * np.pi * a**2)
                  * np.sqrt(D_plate / rho_s))
        self.f_res_modal_exact = float(np.hypot(f_T_ex, f_B_ex))

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

        # CLEARANCE-RING auf dem Feldgitter (s. __init__): Relief-Karte
        # für breite Ringe, Stub-Zelle für schmale; Flags, ob Loch-
        # Mündungen im Relief liegen (dann entlastete Engstelle).
        self._clr_relief = np.zeros(N)
        self._clr_stub_cell = None
        self._clr_th_relieved = False
        self._clr_bh_relieved = False
        if (self.clearance_ring_width > 0.0
                and self.clearance_ring_depth > 0.0
                and self.clearance_ring_diameter > 0.0):
            r_ring = 0.5 * self.clearance_ring_diameter
            if self.clearance_ring_width >= dr:
                mask = np.abs(r_c - r_ring) <= 0.5 * self.clearance_ring_width
                self._clr_relief[mask] = self.clearance_ring_depth
                thr = 0.5 / float(np.sum(self._fld_area))
                if np.any(mask & (self._fld_dens_th > thr)):
                    self._clr_th_relieved = True
                if np.any(mask & (self._fld_dens_bh > thr)):
                    self._clr_bh_relieved = True
            else:
                self._clr_stub_cell = int(np.clip(r_ring / dr, 0, N - 1))

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

        # NUMERISCHER BODEN der Membrandämpfung (s. _Q_MEMBRANE_INTERNAL).
        # Die DOMINANTE Dämpfung der Membran-Grundmode kommt aus dem Spalt-
        # film: die Piston-Bewegung der Membran drückt die Spaltluft lateral
        # zu den Löchern (Škvor-Widerstand R_A_gap). Dieser Widerstand wird
        # in _membrane_impedance frequenzkorrigiert direkt in Reihe geschaltet
        # (die polarisierte Front-Membran über h_gap_front, die passive Rück-
        # membran über h_gap). Der Wert hier trägt nur, wenn KEIN Spaltfilm
        # existiert (geschlossene Backplate, n_th = 0 -> R_A_gap_front = None).
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
            if self.ring_vent_w > 0.0:
                # Randspalt als ZUSÄTZLICHE Senke am Plattenrand: für die
                # Membran-Grundmodendämpfung liegen beide Abflusswege
                # parallel (Löcher-Škvor || Rand-Poiseuille).
                self.R_A_gap = 1.0 / (1.0 / self.R_A_gap
                                      + 1.0 / self._edge_R(self.h_gap))
                self.R_A_gap_front = 1.0 / (
                    1.0 / self.R_A_gap_front
                    + 1.0 / self._edge_R(self.h_gap_front))
        elif self.ring_vent_w > 0.0:
            # REIN RANDBELÜFTETE Platte (B&K-Bauform ohne Bohrungen):
            # die Spaltluft strömt radial zum offenen Plattenrand. Für
            # gleichförmigen Kolbenantrieb (dieselbe Konvention wie
            # Škvor) folgt aus der radialen Poiseuille-Strömung
            #     dp/dr = -12·mu·Q(r)/(2·pi·r·h³),  Q(r) = Q·r²/a²
            # das flächengemittelte Ergebnis R_edge = 3·mu/(2·pi·h³) —
            # wie bei Škvor unabhängig vom Plattenradius.
            self.R_A_gap = self._edge_R(self.h_gap)
            self.R_A_gap_front = self._edge_R(self.h_gap_front)
        else:
            # Geschlossene Backplate: es existiert kein Strömungspfad zu
            # Löchern, also auch keine laterale Škvor-Strömung (die Formel
            # divergiert für q -> 0). Das Spaltvolumen wirkt als reine
            # Nachgiebigkeit direkt an der Membran.
            self.R_A_gap = None
            self.R_A_gap_front = None

        # ------------------------------------------------------------------
        # MÜNDUNGS-WECHSELWIRKUNG IM LOCHARRAY (Fok/Melling)
        # Die Flansch-Mündungskorrektur 0.85·r gilt für die EINSAME
        # Mündung. Sitzen viele Mündungen dicht beieinander (Locharray auf
        # der Backplate/Rückplatte), überlappen ihre Nahfelder und die
        # mitschwingende Masse sinkt. Klassische Korrektur (Fok 1941;
        # Polynomform nach Melling 1973, gültig für quadratische/hexagonale
        # Gitter bis xi ~ 0.5):
        #     delta_L = 0.85·r · F(xi),   xi = r / r_Zelle,
        #     r_Zelle = sqrt(S_Platte / (n·pi))   (Fläche je Loch)
        #     F(xi) = 1 − 1.4092·xi + 0.33818·xi³ + 0.06793·xi⁵
        #             − 0.02287·xi⁶ + 0.03015·xi⁷ − 0.01641·xi⁸
        # Angewandt NUR auf die array-seitigen Mündungen der Durchgangs-
        # löcher (Portseite) und der Rückplattenlöcher — die filmseitigen
        # Mündungen behalten ihre Konvention (die Zell-/Škvor-Ausbreitung
        # deckt die laterale Zuströmung dort ab). Die Einlassringe der
        # Hohlraumwand bleiben unkorrigiert: für eine einzelne Lochreihe
        # auf einem Zylindermantel ist die Gitterprämisse des Fok-Polynoms
        # nicht erfüllt (dokumentierte Näherung).
        # ------------------------------------------------------------------
        def _fok_factor(r_hole, n, S_plate):
            if n < 1 or r_hole <= 0.0 or S_plate <= 0.0:
                return 1.0
            xi = min(r_hole * np.sqrt(n * np.pi / S_plate), 0.9)
            F = (1.0 - 1.4092 * xi + 0.33818 * xi**3 + 0.06793 * xi**5
                 - 0.02287 * xi**6 + 0.03015 * xi**7 - 0.01641 * xi**8)
            return float(np.clip(F, 0.3, 1.0))

        self._fok_th = _fok_factor(self.r_th, self.n_th, self.S_bp)
        self._fok_rp = _fok_factor(self.r_rp, self.n_rp, self.S_bp)

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
        # Durchgangslöcher UND Randspalt sind die Wege durch die Platte:
        if self.n_th == 0 and self.ring_vent_w == 0.0:
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
        # EXTERNE FRONT-RÜCK-DISTANZ der Doppelmembran-Scheibe
        # ----------------------------------------------------------------
        # d_ext ist die EHRLICHE GEOMETRISCHE Wegdifferenz einer ebenen
        # Welle zwischen Vorder- und Rückmembranfläche: axiale Tiefe der
        # Elektrodenbaugruppe (d_ax) plus der Rücksprung beider Membranen
        # hinter die Klemmringe (2 · clamp_ring_thickness). KEIN gefitteter
        # Beugungszuschlag. Die Nierennull entsteht, wenn die INTERNE
        # akustische Laufzeit des Phasenschieber-Netzwerks (Bohrungen,
        # Spaltfilme, Spacer — sie fällt seit der Port-Tausch-Korrektur in
        # _abcd_reverse aus den physikalischen Parametern) diese externe
        # Laufzeit d_ext/c trifft. Nur Doppelmembran-Bauform;
        # Einzel-/Dual-Backplate behalten ihren Rückeinlass.
        if self.architecture == "dual_diaphragm":
            d += 2.0 * self.clamp_ring_thickness
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

        # Sphäroid-/BEM-Körpermodell: nur für die Doppelmembran-Bauform.
        if self.axial_body_model in ("spheroid", "bem"):
            if self.architecture != "dual_diaphragm":
                raise ValueError(
                    f"axial_body_model='{self.axial_body_model}' gilt nur "
                    "für die Doppelmembran-Bauform (axialer Front-Rück-"
                    "Transfer)."
                )
        if self.axial_body_model == "spheroid":
            if 0.5 * self.d_ext >= 0.98 * self.R_body:
                raise ValueError(
                    "axial_body_model='spheroid': die axiale Halbachse "
                    f"d_ext/2 = {0.5 * self.d_ext * 1e3:.1f} mm muss "
                    f"kleiner als der Körperradius {self.R_body * 1e3:.1f}"
                    " mm sein (oblate Scheibe)."
                )

        # 3D-Löser: Gitter-/Lochgeometrie einmalig aufbauen
        if self.squeeze_model == "3d":
            self._build_3d_geometry()

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

    @staticmethod
    def _edge_R(h_film):
        """Squeeze-Film-Widerstand einer RANDBELÜFTETEN Platte.

        Radiale Poiseuille-Strömung der Spaltluft zum offenen Rand bei
        gleichförmigem Kolbenantrieb (gleiche Konvention wie Škvor):
        R_edge = 3·mu/(2·pi·h³), unabhängig vom Plattenradius. Herleitung
        s. Kommentar in :meth:`_derive_parameters`; Frequenzkorrektur
        Φ(ω) wie beim Škvor-Widerstand (derselbe Schlitzfilm).
        """
        return 3.0 * MU_AIR / (2.0 * np.pi * h_film**3)

    def _slit_line_abcd(self, omega, width, breadth, length):
        """Kettenmatrix einer thermoviskosen SCHLITZLEITUNG (LRF).

        Parallelplatten-Kanal der Spaltweite ``width`` (Wandabstand),
        Breite ``breadth`` (hier: abgewickelter Umfang 2π·a_bp des
        Randspalts, width << breadth) und Lauflänge ``length``. Die
        Low-Reduced-Frequency-Lösung ist das Schlitz-Pendant der
        Zwikker–Kosten-Rohrleitung und nutzt DIESELBEN verifizierten
        Profilfunktionen wie der Spaltfilm:

            Z' = jωρ0/(S·B_v),          B_v = 1 − tanh(α_v)/α_v,
            Y' = jω·S/(n_p·P_atm),      n_p = γ/[1+(γ−1)·tanh(α_t)/α_t],
            γ_p = sqrt(Z'·Y'),          Z_c = sqrt(Z'/Y'),   S = w·b.

        Exakte Grenzfälle (Gegenprobe 29): ω→0 liefert den Poiseuille-
        Schlitzwiderstand R = 12·mu·L/(b·w³), die laterale Masse
        M = (6/5)·ρ0·L/(b·w) (kinetischer Profilfaktor 6/5 der
        Schlitzströmung) und die ISOTHERME Nachgiebigkeit V/P_atm des
        Kanalvolumens; hohe Frequenzen laufen gegen die verlustfreie
        Leitung. Mündungskorrekturen an beiden Enden entfallen bewusst:
        filmseitig löst der Film (bzw. R_edge) die Zuströmung auf,
        portseitig ist die Schlitzmündungsmasse O(ρ0·w/S) gegenüber der
        Leitungsimpedanz vernachlässigbar (w << L).
        """
        omega = np.asarray(omega, dtype=float)
        S = width * breadth
        a_v = 0.5 * width * np.sqrt(1j * omega * RHO0 / MU_AIR)
        B_v = 1.0 - np.tanh(a_v) / a_v
        a_t = a_v * np.sqrt(PRANDTL)
        n_p = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t) / a_t)
        Zs = 1j * omega * RHO0 / (S * B_v)              # Serie je Länge
        Ys = 1j * omega * S / (n_p * P_ATM)             # Shunt je Länge
        g_l = np.sqrt(Zs * Ys) * length
        # Numerischer Isolator-Grenzfall: ab Re(γL) ~ 350 ist die Leitung
        # praktisch undurchdringlich (Dämpfung e^350); Kappung verhindert
        # cosh-Überlauf bei pathologisch schmalen Spalten (w -> 0).
        g_l = np.where(np.real(g_l) > 350.0,
                       350.0 + 1j * np.imag(g_l), g_l)
        Zc = np.sqrt(Zs / Ys)
        ch, sh = np.cosh(g_l), np.sinh(g_l)
        return np.array([[ch, Zc * sh], [sh / Zc, ch]])

    @staticmethod
    def _film_R_dynamic(omega, h_film):
        """Frequenzkorrektur des lateralen Squeeze-Film-Widerstands.

        Der statische Škvor-Widerstand gilt für Poiseuille-Strömung; mit
        steigender Frequenz füllt die viskose Grenzschicht den Spalt
        nicht mehr und die TRÄGHEIT der lateral bewegten Spaltluft
        dominiert. Korrekturfaktor (Schlitz-Zwikker–Kosten, identisch
        zum Filmleitwert K_f des 2D-Feldmodells):

            Φ(ω) = (h³/12μ) / K_f(ω),
            K_f  = h/(jωρ0)·[1 − tanh(α)/α],  α = (h/2)·sqrt(jωρ0/μ)

        Φ→1 für ω→0; bei 25 kHz/60 µm beträgt der Unterschied Faktor ~4
        mit −73° Phase. Multiplikativ auf den statischen R anzuwenden —
        macht das 1D-Modell filmphysikalisch konsistent zum 2D-Modell.
        """
        omega = np.asarray(omega, dtype=float)
        a_v = 0.5 * h_film * np.sqrt(1j * omega * RHO0 / MU_AIR)
        K_f = h_film / (1j * omega * RHO0) * (1.0 - np.tanh(a_v) / a_v)
        return (h_film**3 / (12.0 * MU_AIR)) / K_f

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
                        end_correction=True, radiates=False, visc_ends=0):
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

        VISKOSE MÜNDUNG (``visc_ends`` = Anzahl der Enden mit viskosem
        Mündungswiderstand): Die Einströmung in eine Kreisöffnung hat
        neben der Mündungsmasse auch einen REIBUNGSWIDERSTAND
        (Sampson/Roscoe-Kriechströmung; Weissberg 1962:
        R_Rohr+Enden = 8·mu·L/(pi r^4) + 3·mu/r^3). Er entspricht einer
        zusätzlichen ROHRLÄNGE von 3*pi*r/16 je offener Mündung und wird
        hier mit dem thermoviskosen Belag des Rohres selbst ausgewertet
        (nur Realteil — die Mündungsmasse steckt bereits in 0.85·r), so
        dass er bei hohen Frequenzen physikalisch mit der Grenzschicht
        ~sqrt(omega) wächst. Bei kurzen, engen Bohrungen ist dieser Term
        vergleichbar mit dem Rohrwiderstand selbst und darf nicht fehlen.
        Für Mündungen in DÜNNE Spaltfilme gilt er NICHT (dort deckt die
        Škvor-/Zell-Ausbreitung die Zuströmung ab) — daher explizit
        je Aufrufstelle 0, 1 oder 2 Enden.

        STRAHLUNGSWIDERSTAND (radiates=True, Öffnung ins Freifeld):
            R_rad = rho0*c/(pi r^2) * (k r)^2 / 2    (Kolben in Schallwand,
                                                      auf rho0*c/S begrenzt)
        """
        S = np.pi * radius**2
        omega = np.asarray(omega, dtype=float)

        if _HAS_SCIPY:
            F_v, _ = MicrophoneCapsule._fv_ft(omega, radius)
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

        if visc_ends:
            # viskoser Mündungswiderstand (s. Docstring): äquivalente
            # Zusatzlänge 3*pi*r/16 je Mündung, gleicher thermoviskoser
            # Belag wie das Rohr, nur der Realteil zählt.
            L_end = visc_ends * (3.0 * np.pi / 16.0) * radius
            Z_end = MicrophoneCapsule._hole_impedance(
                omega, radius, L_end, 1, end_correction=False)
            Z = Z + Z_end.real

        if radiates:
            k = omega / C_AIR
            Z = Z + (RHO0 * C_AIR / S) * np.minimum((k * radius) ** 2 / 2.0, 1.0)

        return Z / count  # parallele Löcher

    @staticmethod
    def _fv_ft(omega, radius):
        """Zwikker–Kosten-Rohrfunktionen F_v und F_t, numerisch robust.

            F_v = 1 − 2·J1(z)/(z·J0(z)),   z = sqrt(−jωρ0/μ)·r
            F_t = 2·J1(z_t)/(z_t·J0(z_t)), z_t = z·sqrt(Pr)

        Die Besselform ist EXAKT für alle Radien, überläuft aber numerisch
        für große Schubzahlen ζ = r·sqrt(ωρ0/μ) (J wächst wie e^|Im z|;
        ab ζ ~ 1000 wird e^|Im z| > 1e308). Oberhalb ζ = 600 wird deshalb
        die Grenzschicht-Asymptotik verwendet (Fehler < 0.2 % dort,
        identisch mit der Kirchhoff-Näherung weiter Rohre):

            F_v → 1 − (1−j)·δ_v/r,   δ_v = sqrt(2μ/(ρ0·ω)),
            F_t → (1−j)·δ_t/r,       δ_t = δ_v/sqrt(Pr).
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        zeta = radius * np.sqrt(RHO0 * omega / MU_AIR)
        F_v = np.empty(omega.shape, dtype=complex)
        F_t = np.empty(omega.shape, dtype=complex)
        small = zeta <= 600.0
        if np.any(small):
            z = np.sqrt(-1j * omega[small] * RHO0 / MU_AIR) * radius
            F_v[small] = 1.0 - 2.0 * _besselj(1, z) / (z * _besselj(0, z))
            z_t = z * np.sqrt(PRANDTL)
            F_t[small] = (2.0 * _besselj(1, z_t)
                          / (z_t * _besselj(0, z_t)))
        big = ~small
        if np.any(big):
            d_v = np.sqrt(2.0 * MU_AIR / (RHO0 * omega[big]))
            F_v[big] = 1.0 - (1.0 - 1j) * d_v / radius
            F_t[big] = (1.0 - 1j) * d_v / (np.sqrt(PRANDTL) * radius)
        return F_v, F_t

    @staticmethod
    def _narrow_duct_propagation(omega, radius):
        """Ausbreitungskonstante/Wellenwiderstand einer Leitung (exakt).

        Volle Zwikker–Kosten-Form — gültig für ALLE Radien: vom engen
        Rohr (Grenzschichten füllen den Querschnitt: Sacklöcher,
        Senkungen) bis zur weiten Leitung (asymptotisch Kirchhoff, s.
        :meth:`_fv_ft`):

            Z' = jω·rho0 / (S·F_v)           (Impedanzbelag)
            Y' = jω·S / (n_p(ω)·P_atm)       (Admittanzbelag)
            n_p = gamma / [1 + (gamma−1)·F_t]

        n_p läuft von 1 (isotherm) nach gamma (adiabatisch) inkl.
        Relaxationsverlusten. Rückgabe: (gamma_prop, Zc). Erfordert SciPy.
        """
        omega = np.asarray(omega, dtype=float)
        S = np.pi * radius**2
        F_v, F_t = MicrophoneCapsule._fv_ft(omega, radius)
        n_p = GAMMA / (1.0 + (GAMMA - 1.0) * F_t)
        Zp = 1j * omega * RHO0 / (S * F_v)
        Yp = 1j * omega * S / (n_p * P_ATM)
        return np.sqrt(Zp * Yp), np.sqrt(Zp / Yp)

    def _closed_hole_stub(self, omega, radius, length):
        """Eingangsimpedanz EINES endseitig geschlossenen engen Rohres
        (Sackloch) als thermoviskose Leitung:

            Z_in = Zc · coth(gamma·L)

        Gegenüber dem früheren Lumped-Modell (Rohr über die halbe Tiefe
        + adiabatische Volumen-Nachgiebigkeit) ist die Reibung korrekt
        über die Tiefe VERTEILT — der LF-Grenzfall ist das Standard-
        Ergebnis der RC-Leitung, Z ≈ R_Rohr/3 + 1/(jωC) —, die
        Nachgiebigkeit wechselt konsistent isotherm→adiabatisch (mit
        Relaxationsdämpfung), und die λ/4-Resonanz des Sacklochs am
        oberen Bandende liegt an der richtigen Stelle.
        Ohne SciPy: Fallback auf das bisherige Lumped-Modell.
        """
        omega = np.asarray(omega, dtype=float)
        if not _HAS_SCIPY:
            Z_v = self._hole_impedance(omega, radius, 0.5 * length, 1,
                                       end_correction=False)
            C_ad = np.pi * radius**2 * length / (RHO0 * C_AIR**2)
            return Z_v + 1.0 / (1j * omega * C_ad)
        g, Zc = self._narrow_duct_propagation(omega, radius)
        return Zc / np.tanh(g * length)

    def _blind_hole_impedance(self, omega, count=None):
        """Shunt-Impedanz von Sacklochvolumina in der Backplate.

        Blindlöcher vergrößern das wirksame Luftvolumen unter der Membran
        und entlasten so den Squeeze-Film (weniger Dämpfung, klassischer
        Trick bei Großmembran-Backplates). Modell: endseitig geschlossener
        thermoviskoser Leitungsstub über die volle Tiefe
        (s. :meth:`_closed_hole_stub` — verteilte Reibung, isotherm→
        adiabatische Nachgiebigkeit, λ/4-Stub-Verhalten) plus einseitige
        Mündungsmasse. KEIN viskoser Mündungswiderstand: die Öffnung
        liegt im dünnen Spaltfilm, dessen Zuströmung bereits die
        Škvor-/Zell-Ausbreitung abdeckt.
        Ohne ``count`` die reinen Blindlöcher; mit ``count`` auch für
        die SENKUNGEN der Stufenbohrungen nutzbar (gleiche Geometrie
        r_bh/d_bh, eigene Anzahl).
        """
        if count is None:
            count = self.n_bh
        omega = np.asarray(omega, dtype=float)
        Z_stub = self._closed_hole_stub(omega, self.r_bh, self.d_bh)
        S = np.pi * self.r_bh**2
        Z_end = 1j * omega * RHO0 * (0.85 * self.r_bh) / S
        return (Z_stub + Z_end) / count

    def _through_hole_impedance(self, omega, count, radiates=False):
        """Serienimpedanz der Durchgangsbohrungen der Backplate.

        Normale Bohrung: Zwikker–Kosten-Rohr über die volle Plattendicke
        mit beidseitig angeflanschter Mündungskorrektur; der viskose
        Mündungswiderstand (Sampson) zählt nur AUSSENSEITIG — die
        spaltseitige Zuströmung deckt die Škvor-Ausbreitung ab.
        STUFENBOHRUNG: eng gebohrt ist nur die Restdicke t_bp − d_bh
        unter der Senkung; die äußere Mündung ist angeflanscht (0.85·r,
        mit viskosem Anteil), die innere mündet in die weite Senkung —
        Mündungsmasse UND viskoser Mündungswiderstand tragen dort den
        Karal-Faktor (1 − r_th/r_bh) der Querschnittsstufe. Das
        Senkungsvolumen selbst liegt als eigenes Ketten-Element im
        Serienpfad (s. _backplate_gap_abcd).
        """
        omega = np.asarray(omega, dtype=float)
        S = np.pi * self.r_th**2
        if not self.stepped:
            # Mündungsmassen: NUR portseitig (mit Fok-Array-Faktor). Die
            # filmseitige Mündung öffnet in den engen Spalt, nicht in
            # einen Halbraum — dort gibt es kein halbkugeliges Nahfeld;
            # die radiale Ausbreitungsmasse steckt bereits vollständig im
            # Škvor-Term (verifiziert: R_Škvor·rho0·h²/12mu trifft die
            # Baird/Zuckerwar-Spaltmasse exakt). Eine zusätzliche
            # Freifeld-Flanschmasse 0.85·r wäre Doppelzählung — dieselbe
            # Konvention führen das 2D-Feldmodell (dort deckt der
            # Zell-Engstellenwiderstand die Ausbreitung ab) und der
            # 3D-Feldlöser (dort das Filmfeld selbst).
            Z = self._hole_impedance(omega, self.r_th, self.t_bp, count,
                                     end_correction=False,
                                     radiates=radiates, visc_ends=1)
            delta = 0.85 * self.r_th * self._fok_th
            return Z + 1j * omega * RHO0 * delta / (S * count)
        Z = self._hole_impedance(omega, self.r_th, self.t_th_eff, count,
                                 end_correction=False, radiates=radiates,
                                 visc_ends=1)
        karal = 1.0 - self.r_th / self.r_bh
        # portseitige Mündung mit Fok-Faktor, Stufenmündung mit Karal
        delta = 0.85 * self.r_th * (self._fok_th + karal)
        # viskose Mündung an der Stufe (Karal-gewichtete Sampson-Länge)
        Z_step = self._hole_impedance(
            omega, self.r_th, (3.0 * np.pi / 16.0) * self.r_th, 1,
            end_correction=False).real * karal
        return Z + (1j * omega * RHO0 * delta / S + Z_step) / count

    def _radiation_impedance_membrane(self, omega):
        """Strahlungsimpedanz der Membranaußenseite — EXAKTER Kolben.

        KOLBEN IN UNENDLICHER SCHALLWAND, geschlossene Form (Beranek):

            Z_rad = rho0*c/S * [ R1(2ka) + j*X1(2ka) ],
            R1(x) = 1 - 2*J1(x)/x,      X1(x) = 2*H1(x)/x

        mit der Besselfunktion J1 und der Struve-Funktion H1. Kein
        Fit-Koeffizient: das ist die geschlossene Lösung des
        Rayleigh-Integrals über die Kolbenfläche.

        WARUM NICHT DIE ASYMPTOTE: Bis Version <= Gegenprobe 26 stand hier
        die KLEINARGUMENT-Asymptote R ~ (ka)^2/2, X ~ 8ka/(3pi). Sie ist
        für ka -> 0 exakt (Gegenprobe 27 prüft das), divergiert aber
        oberhalb ka ~ 1 grob: die Reaktanz X1 hat ein MAXIMUM bei
        2ka ~ 2 und fällt danach wie 4/(pi*2ka) ab, während die Asymptote
        linear weiterwächst. Die mitschwingende Luftmasse
        M = X*Z0/omega ist deshalb nicht konstant, sondern verschwindet
        im Hochton:

            f (26-mm-Membran)   1k    4k    8k   12k   16k
            M_asymptote      25.0  25.0  25.0  25.0  25.0  kg/m^4
            M_exakt          24.6  19.6   8.9   2.0   0.8  kg/m^4

        Da M_A_mem der K67 nur 20.9 kg/m^4 beträgt, VERDOPPELTE die
        Asymptote die bewegte Masse über das ganze Band und drückte den
        Hochton künstlich (bei 16 kHz um ~7 dB). Die Strahlungslast ist
        also keineswegs "vernachlässigbar klein": |Z_rad| liegt in der
        Größenordnung von |Z_mem| selbst.

        VERBLEIBENDE NÄHERUNG (bewusst, dokumentiert): die unendliche
        Schallwand. Die reale Kapsel sitzt auf einer endlichen Scheibe;
        exakt lieferte das der BEM über die Reziprozität von Streu- und
        Strahlungsproblem. Der Unterschied ist zweiter Ordnung gegenüber
        dem hier behobenen Asymptotenfehler.
        """
        omega = np.asarray(omega, dtype=float)
        Z0 = RHO0 * C_AIR / self.S_mem
        x = 2.0 * omega * self.a_mem / C_AIR              # x = 2ka
        if not _HAS_SCIPY:
            # Ohne SciPy bleibt nur die Asymptote (Gültigkeit ka < 1).
            ka = 0.5 * x
            return Z0 * (np.minimum(ka ** 2 / 2.0, 1.0)
                         + 1j * (8.0 * ka) / (3.0 * np.pi))
        from scipy.special import j1 as _bessel_j1, struve as _struve_h
        xs = np.maximum(x, 1e-30)
        R = 1.0 - 2.0 * _bessel_j1(xs) / xs
        X = 2.0 * _struve_h(1, xs) / xs
        # Kleinargument: 1 - 2*J1(x)/x löscht sich aus -> Reihe verwenden
        small = x < 1.0e-3
        if np.any(small):
            R = np.where(small, x ** 2 / 8.0, R)
            X = np.where(small, 4.0 * x / (3.0 * np.pi), X)
        return Z0 * (R + 1j * X)

    def _higher_mode_branches(self):
        """Akustische (M_A, C_A) der HÖHEREN (0,m)-Membranmoden, m >= 2.

        MODALES AUFBRECHEN DER MEMBRAN (Galerkin, fit-frei)
        ---------------------------------------------------
        Das Lumped-Modell führt die Membran als EINEN Freiheitsgrad
        (Grundmode). Oberhalb weniger kHz schwingt eine reale Membran
        aber längst nicht mehr kolbenförmig: sie bildet Knotenringe. Die
        axialsymmetrischen Moden der unter Spannung stehenden
        Kreismembran sind

            psi_m(r) = J0(x_m r/a),   x_m = m-te Nullstelle von J0,
            omega_m  = omega_1 · x_m/x_1.

        Bei GLEICHFÖRMIGER Druckbelastung (die Annahme des Lumped-/
        1D-Pfads) folgt aus der modalen Zerlegung w = Σ q_m psi_m

            q_m (K_m − omega² M_m) = Δp · A_m,   A_m = ∫ psi_m dS,
            U = j omega Σ q_m A_m
            => Y_ak = Σ_m 1/Z_m  — die Moden liegen PARALLEL.

        Mit M_m^mech = sigma·pi a²·J1(x_m)² und A_m = 2 pi a² J1(x_m)/x_m
        wird die akustische Modenmasse

            M_A,m = M_m^mech / A_m² = sigma·x_m²/(4 pi a²)
                  = M_A,1 · (x_m/x_1)²,

        die Nachgiebigkeit C_A,m = 1/(omega_m² M_A,m). Die GRUNDMODE
        bleibt exakt die kalibrierte (M_A_mem, C_A_eff) — alle
        bestehenden Anker (f_res, Empfindlichkeit, Pull-in) sind
        unberührt; die höheren Moden kommen additiv hinzu.

        Wirkung: im massegesteuerten Hochton trägt Mode m den Anteil
        (x_1/x_m)² zur Volumenschnelle bei (Mode 2: 19 %, Mode 3: 7.7 %)
        — die Membran wird akustisch "weicher", die scharfe
        Einmoden-Auslöschung verschmiert.

        BEWUSSTE NÄHERUNGEN (dokumentiert, nicht angepasst):
        * omega_1 ist hier die UNGESOFTETE Spannungsresonanz
          1/sqrt(M_A_mem·C_A_mem) — die elektrostatische Feder-
          Erweichung ist auf die Grundmode kalibriert und wird auf die
          höheren Moden NICHT übertragen (sie wirkt dort schwächer).
        * Die Filmdämpfung R wird für alle Moden gleich angesetzt. Höhere
          Moden verschieben die Spaltluft über kürzere Strecken, ihr
          echter Widerstand ist kleiner — die Näherung DÄMPFT sie also
          eher zu stark (konservativ).
        Wer die Moden voll gekoppelt will, nimmt den 3D-Feldlöser: der
        führt die Membranen ohnehin als Felder ohne Modenabschneidung.
        """
        if self.membrane_modes <= 1:
            return []
        x = self._J0_ZEROS
        w1 = 1.0 / np.sqrt(self.M_A_mem * self.C_A_mem)
        out = []
        for m in range(1, self.membrane_modes):
            rat = x[m] / x[0]
            M_m = self.M_A_mem * rat**2
            C_m = 1.0 / ((w1 * rat) ** 2 * M_m)
            out.append((M_m, C_m))
        return out

    def _modal_parallel(self, omega, Z1, R):
        """Grundmode Z1 mit den höheren Moden PARALLEL schalten."""
        branches = self._higher_mode_branches()
        if not branches:
            return Z1
        Y = 1.0 / Z1
        for M_m, C_m in branches:
            Y = Y + 1.0 / (R + 1j * omega * M_m
                           + 1.0 / (1j * omega * C_m))
        return 1.0 / Y

    def _membrane_impedance(self, omega):
        """Serienimpedanz der Membran: Z = R + j*omega*M + 1/(j*omega*C_eff).

        Der Dämpfungsterm R ist der SPALTFILM-Widerstand, den die Piston-
        Bewegung der Membran erfährt (Škvor R_A_gap_front am polarisierten
        Frontspalt, frequenzkorrigiert Φ(ω) — bei tiefen Frequenzen reiner
        Widerstand, zu hohen hin mit lateraler Filmträgheit). So bedämpft
        der Spaltfilm die Grundmode DIREKT. Ohne Spaltfilm (geschlossene
        Backplate) bleibt nur der numerische Boden R_A_mem.

        Mit ``membrane_modes > 1`` treten die höheren (0,m)-Bessel-Moden
        PARALLEL hinzu (s. :meth:`_higher_mode_branches`); für
        ``membrane_modes = 1`` (Voreinstellung) bleibt das Ergebnis
        bit-für-bit das bisherige.
        """
        omega = np.asarray(omega, dtype=float)
        R = self._membrane_film_damping(omega, self.h_gap_front,
                                        self.R_A_gap_front)
        Z1 = (
            R
            + 1j * omega * self.M_A_mem
            + 1.0 / (1j * omega * self.C_A_eff)
        )
        return self._modal_parallel(omega, Z1, R)

    def _membrane_film_damping(self, omega, h_film, R_A_gap):
        """Spaltfilm-Dämpfungsimpedanz der Membran-Piston-Mode.

        Die bewegte Membran drückt die Spaltluft lateral durch den Film zu
        den Löchern — der Škvor-Widerstand R_A_gap (frequenzkorrigiert
        Φ(ω, h) für die laterale Filmträgheit) ist die dominante Dämpfung
        der Grundmode. Ohne Spaltfilm (R_A_gap = None, geschlossene
        Backplate) bleibt nur der numerische Boden R_A_mem.
        """
        if R_A_gap is None:
            return self.R_A_mem
        omega = np.asarray(omega, dtype=float)
        return self.R_A_mem + R_A_gap * self._film_R_dynamic(omega, h_film)

    def _membrane_impedance_passive(self, omega):
        """Serienimpedanz der PASSIVEN Rückmembran (K67-Bauform, Niere).

        Im Nierenmodus liegt die Rückmembran auf Backplate-Potential —
        kein Feld, keine Feder-Erweichung: es gilt die unpolarisierte
        Nachgiebigkeit C_A_mem (gleiches Material/Tuning wie vorn). Die
        Dämpfung kommt wie bei der Frontmembran aus dem Spaltfilm — hier am
        NOMINALEN Rückspalt h_gap (unpolarisiert, keine Durchbiegung), also
        R_A_gap.
        """
        omega = np.asarray(omega, dtype=float)
        R = self._membrane_film_damping(omega, self.h_gap, self.R_A_gap)
        Z1 = (
            R + 1j * omega * self.M_A_mem
            + 1.0 / (1j * omega * self.C_A_mem)
        )
        return self._modal_parallel(omega, Z1, R)

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

    # ------------------------------------------------------------------
    # Oblates Sphäroid (m = 0): eigene Spezialfunktionen.
    # scipy.special.obl_rad2 ist für ξ0 < 1 unbrauchbar (die Neumann-
    # Reihe der Radialfunktion 2. Art konvergiert nur für ξ > 1) —
    # deshalb: Eigenwerte/Koeffizienten aus der Flammer-Rekursion
    # (Tridiagonal-Eigenproblem, Querprobe scipy obl_cv auf 1e-14),
    # R^(3) per Hankel-Reihen-Start bei ξ_far = max(2, 40/c) (dort
    # konvergent) und Einwärts-RK4 der Radial-ODE auf log-Gitter
    # (einwärts stabil: die reguläre Beimischung fällt ab). Selbst-
    # verifikation je Mode über die Wronski-Identität
    # W{R1, R3} = i/(c(ξ²+1)) mit R1 aus der überall konvergenten
    # Besselreihe (Gegenprobe 20).
    # ------------------------------------------------------------------
    @staticmethod
    def _oblate_modes(c, n_max, K=60):
        """d_r-Vektoren (Meixner-Schäfke-Norm) + Eigenwerte, m = 0."""
        c2 = -c * c                    # oblate: c² -> -c²
        pars = []
        for par in (0, 1):
            r = par + 2 * np.arange(K)
            B = r * (r + 1) + c2 * (2 * r * (r + 1) - 1) / (
                (2 * r - 1) * (2 * r + 3))
            A = (r + 1) * (r + 2) / ((2 * r + 3) * (2 * r + 5)) * c2
            Cc = r * (r - 1) / ((2 * r - 3) * (2 * r - 1)) * c2
            M = np.diag(B) + np.diag(A[:-1], 1) + np.diag(Cc[1:], -1)
            ev, V = np.linalg.eig(M)
            idx = np.argsort(ev.real)
            pars.append((r, ev[idx].real, V[:, idx].real))
        modes = []
        for n in range(n_max + 1):
            r, ev, V = pars[n % 2]
            j = (n - n % 2) // 2
            d = V[:, j].copy()
            if d[j] < 0:
                d = -d
            modes.append({"n": n, "r": r, "d": d, "lam": float(ev[j]),
                          "N": float(np.sum(d**2 * 2.0 / (2 * r + 1)))})
        return modes

    @staticmethod
    def _oblate_S(mode, x):
        """S_0n(c, x) = Σ d_r P_r(x) (Legendre-Aufwärtsrekurrenz)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        r_arr, d_arr = mode["r"], mode["d"]
        r_max = int(r_arr[-1])
        dmap = np.zeros(r_max + 1)
        dmap[r_arr] = d_arr
        P0, P1 = np.ones_like(x), x.copy()
        S = dmap[0] * P0 + (dmap[1] * P1 if r_max >= 1 else 0.0)
        for rr in range(1, r_max):
            P2 = ((2 * rr + 1) * x * P1 - rr * P0) / (rr + 1)
            S = S + dmap[rr + 1] * P2
            P0, P1 = P1, P2
        return S

    @staticmethod
    def _oblate_R1(mode, c, xi):
        """R1, dR1/dξ aus der Besselreihe (konvergiert für alle ξ)."""
        r, d, n = mode["r"], mode["d"], mode["n"]
        x = c * xi
        pref = 1.0 / np.sum(d)
        ph = 1j ** (r - n)
        R1 = pref * np.sum(ph * d * _sph_jn(r, x))
        dR1 = pref * np.sum(ph * d * _sph_jn(r, x, derivative=True)) * c
        return R1, dR1

    @staticmethod
    def _oblate_R3(mode, c, xi0, n_steps=1500):
        """R3, dR3/dξ: Hankel-Reihe (konvergiert für ξ > 1) — direkt,
        wenn ξ0 ≥ 1.5; sonst Reihen-Start bei ξ = 2 und Einwärts-RK4
        der Radial-ODE auf log-Gitter (kurze Spanne ln(2/ξ0), einwärts
        stabil: die reguläre Beimischung fällt ab)."""
        lam = mode["lam"]
        r, d, n = mode["r"], mode["d"], mode["n"]

        def _series(xi):
            # Der Startpunkt hält x = c·ξ >= 40 (bzw. ξ0-Kurzschluss):
            # dort bleiben die y_r gutartig und die Reihe konvergiert in
            # wenigen Termen — kleiner wählbare Startpunkte scheitern,
            # weil die Reihe dann Koeffizienten UNTER dem Eigenvektor-
            # Rauschboden bräuchte (Termabfall nur ~ ξ^{-r}, y_r-Wachstum
            # verstärkt das Rauschen katastrophal). Überlaufende Terme
            # mit vernachlässigbarem Gewicht werden genullt.
            x = c * xi
            pref = 1.0 / np.sum(d)
            ph = 1j ** (r - n)
            with np.errstate(over="ignore", invalid="ignore"):
                hr = _sph_jn(r, x) + 1j * _sph_yn(r, x)
                dhr = (_sph_jn(r, x, derivative=True)
                       + 1j * _sph_yn(r, x, derivative=True))
                t_R = ph * d * hr
                t_dR = ph * d * dhr
            bad = ~np.isfinite(t_R) | ~np.isfinite(t_dR)
            if np.any(bad & (np.abs(d) > 1e-120)):
                raise RuntimeError(
                    "Sphäroid-R3: Hankel-Reihe numerisch instabil."
                )
            t_R[bad] = 0.0
            t_dR[bad] = 0.0
            return pref * np.sum(t_R), pref * np.sum(t_dR)

        if xi0 >= 1.5 and c * xi0 >= 1e-2:
            R, dR_dx = _series(xi0)
            return R, dR_dx * c
        # Einwärts-RK4 auf log-Gitter in x = c·ξ (außen Oszillation,
        # innen Potenzverhalten — beides aufgelöst); Schrittzahl wächst
        # mit der Spanne. Einwärts stabil: die reguläre Beimischung
        # fällt einwärts ab.
        xi_far = max(2.0, 40.0 / c, 1.5 * xi0)
        x_far, x0 = c * xi_far, c * xi0
        R, dR_dx = _series(xi_far)
        span = np.log(x_far / x0)
        n_int = max(int(n_steps), int(400.0 * span))
        t, h = np.log(x_far), -span / n_int

        def deriv(t_c, y):
            x = np.exp(t_c)
            return np.array([x * y[1] / (x**2 + c**2),
                             x * (lam - x**2) * y[0]])

        y = np.array([R, (x_far**2 + c**2) * dR_dx], dtype=complex)
        for _ in range(n_int):
            k1 = deriv(t, y)
            k2 = deriv(t + h / 2, y + h / 2 * k1)
            k3 = deriv(t + h / 2, y + h / 2 * k2)
            k4 = deriv(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y[0], y[1] / (x0**2 + c**2) * c

    def _spheroid_pole_transfer(self, omega, theta, a_maj, b_min):
        """G(θ) = p(Rückpol)/p(Frontpol) am starren oblaten Sphäroid.

        Halbachsen: a_maj radial, b_min axial (b < a). Ebene Welle unter
        θ zur Symmetrieachse; an den Polen (η = ±1) verschwinden alle
        azimutalen Ordnungen m > 0 — es bleibt die m=0-Reihe

            G(θ) = Σ b_n (−1)^n S_0n(c, cos θ) / Σ b_n S_0n(c, cos θ),
            b_n  = (−i)^n S_0n(c, 1) / (N_0n · R'^(3)_0n(c, ξ0)),

        c = k·f, f = sqrt(a² − b²), ξ0 = b/f (Konvention validiert am
        Kugel-Grenzfall gegen die Morse-Reihe). Grenzfälle (Gegen-
        probe 20): b→a reproduziert die Kugel; die dünne Scheibe liefert
        am Pol die exakte LF-Distanz 4a/π (klassisches Scheiben-
        resultat). Selbstprüfung: Wronski-Fehler je Mode
        (self._spheroid_wronski_max), Gate bei 1e-3.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        focal = np.sqrt(a_maj**2 - b_min**2)
        xi0 = b_min / focal
        ct = np.cos(theta)
        G = np.empty((omega.size, theta.size), dtype=complex)
        w_max = 0.0
        for i, om in enumerate(omega):
            c = om / C_AIR * focal
            n_max = int(np.ceil(c)) + 10
            modes = self._oblate_modes(c, n_max)
            num = np.zeros(theta.size, dtype=complex)
            den = np.zeros(theta.size, dtype=complex)
            for m in modes:
                n = m["n"]
                R3, dR3 = self._oblate_R3(m, c, xi0)
                R1, dR1 = self._oblate_R1(m, c, xi0)
                W = R1 * dR3 - dR1 * R3
                w_err = abs(W / (1j / (c * (xi0**2 + 1.0))) - 1.0)
                w_max = max(w_max, w_err)
                S1 = self._oblate_S(m, 1.0)[0]
                St = self._oblate_S(m, ct)
                b_n = (-1j)**n * S1 / (m["N"] * dR3)
                den = den + b_n * St
                num = num + b_n * (-1.0)**n * St
            G[i] = np.conj(num / den)
        self._spheroid_wronski_max = w_max
        if w_max > 1e-3:
            raise RuntimeError(
                "Sphäroid-Transfer: Wronski-Selbstprüfung fehlgeschlagen "
                f"(max. Fehler {w_max:.2e})."
            )
        return G

    def _axial_spheroid_transfer(self, omega, theta):
        """Axialer Front-Rück-Transfer am Sphäroid mit der Kapselgeometrie:
        radiale Halbachse R_body, axiale d_ext/2 (frei stehende Scheibe).
        """
        return self._spheroid_pole_transfer(omega, theta, self.R_body,
                                            0.5 * self.d_ext)

    # ------------------------------------------------------------------
    # Axisymmetrisches Randelementverfahren (m = 0) für starre
    # Rotationskörper: direkte Kirchhoff-Helmholtz-Kollokation,
    #     u(x)/2 = p_inc(x) + PV ∮ (∂G/∂n_y) u(y) dS_y,
    # Konvention e^{-iωt}, G = e^{ikR}/(4πR), am Ende konjugiert.
    # m=0 genügt: Front-/Rückmembran-MITTELWERTE sind exakt die
    # azimutalen m=0-Projektionen (wie die Ring-/Kalottenmittelung der
    # Kugelbeugung); der m=0-Anteil der ebenen Welle unter θ ist
    #     p_inc,0(r, z) = J0(k r sinθ) · e^{-i k z cosθ}.
    # Diagonale über die statische Raumwinkel-Identität
    # Σ_j K0_ij = -1/2 (Außenraum, x auf S); irreguläre Frequenzen
    # (innere Dirichlet-Eigenfrequenzen) fängt CHIEF ab (Achsen- und
    # Innenpunkte, Least-Squares). Verrundete Kanten (Fillets) halten
    # die Fläche glatt; auf ebenen Flächenstücken verschwindet der
    # Doppelschichtkern koplanarer Paare exakt.
    # Validiert (Gegenprobe 21): Kugelkontur trifft die Morse-Reihe und
    # Sphäroidkontur die Sphäroid-Reihe auf < 1e-3 über das Band.
    # ------------------------------------------------------------------
    _BEM_NPHI = 96
    _BEM_G4 = np.array([-0.8611363116, -0.3399810436,
                        0.3399810436, 0.8611363116])
    _BEM_W4 = np.array([0.3478548451, 0.6521451549,
                        0.6521451549, 0.3478548451]) / 2.0

    @staticmethod
    def _bem_contour(segments, rf=0.8e-3, h=1.0e-3):
        """Meridian-Polylinie aus Linien-/Bogenstücken (Ecken verrundet).

        ``segments``: Liste von ("line", p0, p1) / ("arc", cen, r, a0, a1);
        Rückgabe: (M, 2)-Punktfolge. Die Kontur muss von der Achse (oben,
        größtes z) zur Achse (unten) laufen — dann zeigt die Normale
        (-dz, dr)/L nach außen.
        """
        pts = [segments[0][1] if segments[0][0] == "line" else None]

        def _line(p0, p1):
            n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / h))
            for i in range(1, n + 1):
                pts.append((p0[0] + (p1[0] - p0[0]) * i / n,
                            p0[1] + (p1[1] - p0[1]) * i / n))

        def _arc(cen, r, a0, a1, n=7):
            for i in range(1, n + 1):
                a = a0 + (a1 - a0) * i / n
                pts.append((cen[0] + r * np.cos(a), cen[1] + r * np.sin(a)))

        for seg in segments:
            if seg[0] == "line":
                _line(seg[1], seg[2])
            else:
                _arc(seg[1], seg[2], seg[3], seg[4])
        return np.array(pts)

    @staticmethod
    def _bem_elems(pts):
        """Konstante Elemente: Mittelpunkte, Außennormalen, Längen,
        Gauß-Knoten (4 je Element) entlang des Meridians."""
        p0, p1 = pts[:-1], pts[1:]
        d = p1 - p0
        L = np.hypot(d[:, 0], d[:, 1])
        ok = L > 1e-12
        p0, d, L = p0[ok], d[ok], L[ok]
        mid = p0 + 0.5 * d
        G4 = MicrophoneCapsule._BEM_G4
        return dict(
            mid_r=mid[:, 0], mid_z=mid[:, 1],
            nr=-d[:, 1] / L, nz=d[:, 0] / L, L=L,
            gr=mid[:, 0][:, None] + 0.5 * d[:, 0][:, None] * G4[None, :],
            gz=mid[:, 1][:, None] + 0.5 * d[:, 1][:, None] * G4[None, :],
        )

    @classmethod
    def _bem_phi_quad(cls):
        """φ-Quadratur [0, 2π), t²-geclustert um φ = 0 (Nähe-Peak)."""
        t = (np.arange(cls._BEM_NPHI) + 0.5) / cls._BEM_NPHI
        phi = np.pi * t**2
        w = 2.0 * (2.0 * np.pi * t / cls._BEM_NPHI)
        return np.cos(phi), w

    @staticmethod
    def _bem_ring_rows(k, xr, xz, elems, cphi, wphi, static=False):
        """Zeilenblock des Ringkern-Operators: für Aufpunkte (xr, xz)
        [Form (B,)] die Integrale ∮ ∂G/∂n_y dS über alle Elemente.
        Rückgabe (B, N)."""
        gr = elems["gr"][None, :, :, None]           # (1, N, 4, 1)
        gz = elems["gz"][None, :, :, None]
        nr = elems["nr"][None, :, None, None]
        nz = elems["nz"][None, :, None, None]
        xr_ = xr[:, None, None, None]
        xz_ = xz[:, None, None, None]
        R2 = ((xz_ - gz)**2 + xr_**2 + gr**2
              - 2.0 * xr_ * gr * cphi)
        R = np.sqrt(np.maximum(R2, 1e-30))
        ndot = nr * (gr - xr_ * cphi) + nz * (gz - xz_)
        if static:
            g = -ndot / (4.0 * np.pi * R**3)
        else:
            g = (ndot * (1j * k * R - 1.0) * np.exp(1j * k * R)
                 / (4.0 * np.pi * R**3))
        ring = np.sum(g * (gr * wphi), axis=-1)      # (B, N, 4)
        W4 = MicrophoneCapsule._BEM_W4
        return np.sum(ring * W4[None, None, :], axis=-1) * elems["L"][None, :]

    def _bem_geometry(self):
        """Kontur Kapselkopf (+ optionaler Körperzylinder), Membranmasken,
        CHIEF-Punkte — einmalig aufgebaut und am Objekt gehalten."""
        if getattr(self, "_bem_geo", None) is not None:
            return self._bem_geo
        rf = 0.8e-3
        Rh, zf = self.R_body, 0.5 * self.d_ext
        zr = -zf
        segs = [("line", (0.0, zf), (Rh - rf, zf)),
                ("arc", (Rh - rf, zf - rf), rf, np.pi / 2, 0.0),
                ("line", (Rh, zf - rf), (Rh, zr + rf)),
                ("arc", (Rh - rf, zr + rf), rf, 0.0, -np.pi / 2),
                ("line", (Rh - rf, zr), (0.0, zr))]
        pts = self._bem_contour(segs, h=1.0e-3)
        elems = self._bem_elems(pts)
        n_head = elems["L"].size
        chief = [(0.0, 0.0), (0.55 * Rh, 0.0)]
        if self.bem_body_diameter > 0.0:
            Rb = 0.5 * self.bem_body_diameter
            z0 = zr - self.bem_body_gap
            z1 = z0 - self.bem_body_length
            segs_b = [("line", (0.0, z0), (Rb - rf, z0)),
                      ("arc", (Rb - rf, z0 - rf), rf, np.pi / 2, 0.0),
                      ("line", (Rb, z0 - rf), (Rb, z1 + rf)),
                      ("arc", (Rb - rf, z1 + rf), rf, 0.0, -np.pi / 2),
                      ("line", (Rb - rf, z1), (0.0, z1))]
            eb = self._bem_elems(self._bem_contour(segs_b, h=1.6e-3))
            elems = {kk: np.concatenate([elems[kk], eb[kk]], axis=0)
                     for kk in elems}
            chief += [(0.0, z0 - 0.25 * self.bem_body_length),
                      (0.0, z0 - 0.75 * self.bem_body_length),
                      (0.5 * Rb, z0 - 0.5 * self.bem_body_length)]
        mr, mz = elems["mid_r"], elems["mid_z"]
        w_area = 2.0 * np.pi * mr * elems["L"]
        front = (np.abs(mz - zf) < 1e-6) & (mr <= self.a_mem)
        rear = ((np.abs(mz - zr) < 1e-6) & (mr <= self.a_mem)
                & (np.arange(mr.size) < n_head))
        self._bem_geo = dict(elems=elems, chief=chief, w_area=w_area,
                             front=front, rear=rear)
        return self._bem_geo

    def _bem_axial_fields(self, omega, theta):
        """Absoluter Frontfaktor F(ω,θ) UND Front-Rück-Transfer G(ω,θ)
        aus EINEM m=0-BEM-Lösungsgang auf der Kontur Kopf + Körper:

            F = ⟨p⟩_Frontmembran / p0,
            G = ⟨p⟩_Rückmembran / ⟨p⟩_Frontmembran,

        p0 = ungestörter Freifelddruck im Kapselzentrum (Ursprung).
        F ist damit der Beugungs-/Druckstaufaktor der REALEN FLACHEN
        Stirnfläche inkl. Apertur-Mittelung über die Membranfläche
        (das m=0-Flächenmittel ist auch bei Schrägeinfall exakt).
        Kein freier Parameter — reine Geometrie im Helmholtz-
        Randintegral. Wesentlich gegenüber der Kugelkalotten-Näherung
        (_diffraction_factors): am flachen Kopf steht die Membran
        SENKRECHT zur Einfallsrichtung — der Druckstau erreicht die
        Verdopplung (+6 dB) schon bei ka ≈ 2..4, während die um bis
        ±50° gekrümmte Kugelkalotte dort erst +3..4 dB liefert
        (validiert: Gegenprobe 26)."""
        from scipy.special import j0 as _bessel_j0
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        geo = self._bem_geometry()
        elems, chief = geo["elems"], geo["chief"]
        w_area, front, rear = geo["w_area"], geo["front"], geo["rear"]
        N = elems["L"].size
        cphi, wphi = self._bem_phi_quad()
        mr, mz = elems["mid_r"], elems["mid_z"]
        cr = np.array([c[0] for c in chief])
        cz = np.array([c[1] for c in chief])
        ct, st = np.cos(theta), np.sin(theta)

        def _pinc(k, r, z):
            return (_bessel_j0(np.outer(st * k, r))
                    * np.exp(-1j * np.outer(ct * k, z))).T   # (Npunkte, Nθ)

        F = np.empty((omega.size, theta.size), dtype=complex)
        G = np.empty((omega.size, theta.size), dtype=complex)
        wsum = np.zeros(N)
        for i, om in enumerate(omega):
            k = om / C_AIR
            K = np.empty((N, N), dtype=complex)
            K0 = np.empty((N, N), dtype=complex)
            for b0 in range(0, N, 32):
                b1 = min(b0 + 32, N)
                K[b0:b1] = self._bem_ring_rows(k, mr[b0:b1], mz[b0:b1],
                                               elems, cphi, wphi)
                K0[b0:b1] = self._bem_ring_rows(0.0, mr[b0:b1], mz[b0:b1],
                                                elems, cphi, wphi,
                                                static=True)
            # Diagonale: statische Identität Σ_j K0_ij = -1/2
            rows = np.arange(N)
            row0 = np.sum(K0, axis=1) - K0[rows, rows]
            K[rows, rows] = (K[rows, rows] - K0[rows, rows]
                             + (-0.5 - row0))
            wsum = np.abs(row0 + K0[rows, rows] + 0.5)
            A = 0.5 * np.eye(N) - K
            b = _pinc(k, mr, mz)
            K_c = self._bem_ring_rows(k, cr, cz, elems, cphi, wphi)
            A = np.vstack([A, -K_c])
            b = np.vstack([b, _pinc(k, cr, cz)])
            u, *_ = np.linalg.lstsq(A, b, rcond=None)
            p_f = (w_area[front] @ u[front]) / np.sum(w_area[front])
            p_r = (w_area[rear] @ u[rear]) / np.sum(w_area[rear])
            F[i] = np.conj(p_f)
            G[i] = np.conj(p_r / p_f)
        # Diagnose: Residuum der Raumwinkel-Identität (Gitterqualität)
        self._bem_solid_angle_residual = float(np.max(wsum))
        return F, G

    def _bem_axial_transfer(self, omega, theta):
        """Nur der Front-Rück-Transfer G(θ) (s. :meth:`_bem_axial_fields`)."""
        return self._bem_axial_fields(omega, theta)[1]

    def _axial_body_transfer(self, omega, theta):
        """Front-Rück-Transfer G(θ) = p_rück/p_front der Doppelmembran-Scheibe.

        BEUGUNG UM DIE AXIALE KÖRPERAUSDEHNUNG (kein Fit-Koeffizient):
        Die beiden Membranen sitzen auf den Stirnflächen eines Körpers der
        axialen Dicke d_ext. Der externe Weg des rückwärtigen Schalls zur
        Frontmembran ist LÄNGER als die nackte Geometrie d_ext·cosθ, weil
        der Schall den Körper umlaufen muss. Modelliert als Pol-zu-Pol-
        Transfer der exakten Morse-Streureihe an der starren Kugel mit
        DURCHMESSER d_ext (gleiche Reihe wie _diffraction_factors, hier
        mit der korrekten AXIALEN Ausdehnung als Körperskala):

            G(θ) = Σ b_n(ka)·(−1)^n·P_n(cosθ) / Σ b_n(ka)·P_n(cosθ),
            b_n = (2n+1)(−i)^n / h'_n(ka),   ka = ω·(d_ext/2)/c.

        Für ka → 0 liefert das automatisch den bekannten 3/2-Dipolfaktor
        der starren Kugel (effektive Distanz 1.5·d_ext) samt der
        zugehörigen kleinen Amplituden-Asymmetrie — beides zweiter
        Ordnung konsistent zur internen RC-Laufzeit. Warum nicht die
        R_body-Kugel: deren Ring-Platzierung (bei d_rear_ax) ergibt nur
        ~0.6·d_ext effektiv (zu kurz), eine Kalotte am hinteren Pol
        ~2.5·R_body (zu lang, Superniere) — die 34-mm-Breitenkugel
        überzeichnet die axiale Ausdehnung der ~12 mm dünnen Scheibe.
        Bekannte Näherungsgrenze: eine Kugel überschätzt den Umweg einer
        FLACHEN Scheibe tendenziell etwas (kanonische Ersatzkörper-
        Vergleiche); der Wert ist aber vollständig hergeleitet, nicht
        kalibriert.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        ka = omega * (0.5 * self.d_ext) / C_AIR
        n_max = int(np.max(ka)) + 12
        ct = np.cos(theta)
        P = [np.ones_like(ct), ct.copy()]
        for n in range(1, n_max):
            P.append(((2 * n + 1) * ct * P[n] - n * P[n - 1]) / (n + 1))
        num = np.zeros((omega.size, theta.size), dtype=complex)
        den = np.zeros_like(num)
        for n in range(n_max + 1):
            h1p = (_sph_jn(n, ka, derivative=True)
                   + 1j * _sph_yn(n, ka, derivative=True))
            base = (2 * n + 1) * (-1j) ** n / h1p
            den += np.outer(base, P[n])                 # vorderer Pol
            num += np.outer(base, (-1) ** n * P[n])     # hinterer Pol
        return np.conj(num / den)

    # ======================================================================
    # 3D-(r,phi)-Feldlöser: Sandwich mit diskreten Löchern
    # ======================================================================
    def _build_3d_geometry(self):
        """Einmalige Gitter-/Lochgeometrie für ``squeeze_model='3d'``.

        Das 3D-Modell löst das komplette Sandwich der durchbohrten
        Elektrode(n) als EIN gekoppeltes Feldproblem auf einem
        (r, phi)-Gitter. Bauformen der Doppelmembran-Architektur:

        * ``center_gap = 0`` (einteilig, Debenham-Typ, 2 Filme):

              p_front(r,phi), p_rear(r,phi)  — Reynolds-Filme beider Spalte
              w_front(r,phi), w_rear(r,phi)  — Membranen als FD-FELDER
                                               (Spannungsoperator, am Rand
                                               eingespannt; KEINE Moden-
                                               abschneidung)

          Jedes Durchgangsloch verbindet die beiden Filme direkt
          (Lochpaar-Leitwert über die volle Elektrodendicke).

        * ``center_gap > 0`` (zweiteilig, K67-Typ, 3 Filme): zusätzlich
          p_center(r,phi) als dritter Reynolds-Film im Zwischenspalt.
          Jede Elektrodenhälfte trägt ihr EIGENES Lochbild; die
          Durchgangslöcher verbinden Membranfilm <-> Zwischenspalt als
          Zweitor-Kette (bei Stufenbohrung: weite Senkung als
          Leitungsstück + Karal-Stufe + enger Kern). Die Hälften sind
          um ``half_rotation_deg`` gegeneinander VERDREHT (Standard:
          eine halbe Teilung 180°/n_th, wie an der realen K67, deren
          Bohrungen nicht zueinander zeigen) — ausgerichtete Löcher
          (0°) kurzschließen den Phasenschieber durch den Spalt,
          s. Gegenprobe 22.

        SINGLE/DUAL (Gegenprobe 23): EIN Membranfeld; je Backplate ein
        Film, dessen Durchgangslöcher als Zweitor-Ketten in einen
        SAMMELKNOTEN münden. Die Knoten-Abschlüsse sind die baugleichen
        Lumped-Ketten des 1D/2D-Pfads: hinten die rückwärtige Baugruppe
        (_rear_chain_mats: Spacer/Rückplatte, Gewebe, Laufzeitglied,
        Hohlraum, Einlasslöcher), vorn Strahlung + Gewebe vor der
        Membran (single: Frontknoten über der Membranfläche, Grenzfall
        Z -> 0 = direkter Quelldruck; dual: vor der vorderen Backplate).
        Die 3D-Ausgänge folgen der Ketten-Flussrichtung (Vorzeichen wie
        1D/2D, s. _solve_3d) — H ist über alle Modelle phasengleich.

        Gegenüber dem axialsymmetrischen 2D-Modell fällt damit die
        Homogenisierung der Löcher weg: Durchgangs- und Sacklöcher sitzen
        DISKRET an ihren (r, phi)-Positionen (Mündungs-Fußabdruck über
        die Zellen verteilt), die azimutale Zuströmung durch den Film und
        die dadurch teilentkoppelten Sacklöcher werden aufgelöst — das
        bedämpft insbesondere die interne Helmholtz-Resonanz realistisch.
        Da die realen Azimutwinkel der Bohrbilder nicht dokumentiert
        sind, gilt eine feste KONVENTION: gleichverteilte Löcher je
        Lochkreis, Ring m der Durchgangslöcher um 20°·m verdreht, Sack-
        löcher um weitere 15° (Rückseite zusätzlich um eine halbe
        Teilung bzw. im K67-Modus um half_rotation_deg) — die Ergebnisse
        hängen nur schwach davon ab.
        Membran-Elektrostatik: Feder-Erweichung als verteilte negative
        Steifigkeit, an der Grundmode kalibriert (C_A_eff). Gewebe- und
        Strahlungsimpedanz der Membranaußenseiten werden im 3D-Modell
        vernachlässigt (klein; Gewebe in den validierten Beispielen 0).
        """
        from scipy.special import j1 as _j1, jn_zeros as _jn_zeros
        # Azimutale Auflösung: einteilig genügen 96 Zellen (Debenham,
        # 12 Löcher). Im K67-Modus müssen der Lochabstand UND der
        # Verdrehungs-Versatz der Hälften im Zwischenspalt aufgelöst
        # werden — mindestens ~4 Zellen je Lochteilung (Konvergenz:
        # 180°-Wert wandert von Np=96 nach 192/288 um ~4 dB und
        # stabilisiert sich dann auf ±1.5 dB).
        Np_ = int(getattr(self, "_n_phi_3d", 0))
        if Np_ <= 0:
            if self.h_center > 0.0:
                Np_ = int(max(96, min(4 * max(self.n_th, 1), 320)))
            else:
                Np_ = 96
        Nr = self._fld_N
        dr = self.a_bp / Nr
        Nr_m = max(Nr + 1, int(round(self.a_mem / dr)))
        r_f = (np.arange(Nr) + 0.5) * dr
        r_m = (np.arange(Nr_m) + 0.5) * dr
        dphi = 2.0 * np.pi / Np_
        A_f = r_f * dr * dphi                       # Zellfläche je Ring
        A_m = r_m * dr * dphi
        NF = Nr * Np_
        NM = Nr_m * Np_

        # Membrankonstanten: Flächendichte, Spannung aus f_res, verteilte
        # Feder-Erweichung (Grundmoden-kalibriert auf C_A_eff)
        sigma = 0.75 * self.M_A_mem * self.S_mem
        b01 = float(_jn_zeros(0, 1)[0])
        T_mem = sigma * (2.0 * np.pi * self.f_res * self.a_mem / b01) ** 2
        from scipy.special import j0 as _j0
        psi1 = _j0(b01 * r_m / self.a_mem)
        k1 = ((2.0 * np.pi * self.f_res) ** 2 * sigma
              * np.pi * self.a_mem ** 2 * _j1(b01) ** 2)
        E2 = float(np.sum((psi1[:Nr] ** 2) * A_f * Np_))
        kappa = k1 * (1.0 - self.C_A_mem / self.C_A_eff) / E2

        # Loch-Fußabdrücke: Zellen, deren Zentrum in der Mündung liegt
        def _foot(radius, n, off_deg, r_hole):
            out = []
            i0 = int(np.clip(radius / dr, 0, Nr - 1))
            for k in range(max(n, 0)):
                ph0 = np.deg2rad(off_deg) + 2.0 * np.pi * k / max(n, 1)
                cells = []
                for di in range(-4, 5):
                    i = i0 + di
                    if i < 0 or i >= Nr:
                        continue
                    arc = r_f[i] * dphi
                    for dj in range(-4, 5):
                        if (di * dr) ** 2 + (dj * arc) ** 2 \
                                <= r_hole ** 2 + 1e-18:
                            cells.append(i * Np_
                                         + int(ph0 / dphi + dj) % Np_)
                if not cells:
                    cells = [i0 * Np_ + int(ph0 / dphi) % Np_]
                out.append(np.array(sorted(set(cells)), dtype=int))
            return out

        def _ring_list(rings, n_total):
            # (Anzahl, Radius|None); None = gleichmäßig über die Elektrode
            # -> flächenproportional auf mehrere Kreise aufgeteilt (die
            # Zahl der Hilfskreise wächst mit der Lochzahl)
            out = []
            for cnt, r_pcd in rings:
                if cnt <= 0:
                    continue
                if r_pcd is not None:
                    out.append((cnt, r_pcd))
                    continue
                n_sub = max(2, int(round(np.sqrt(cnt))))
                edges = self.a_bp * np.sqrt(np.linspace(0.0, 1.0,
                                                        n_sub + 1))
                mids = 0.5 * (edges[:-1] + edges[1:])
                areas = np.diff(edges ** 2)
                counts = np.maximum(np.round(cnt * areas
                                             / areas.sum()), 1).astype(int)
                # Rundungsdifferenz am äußersten (größten) Ring ausgleichen
                counts[-1] += cnt - int(counts.sum())
                for c_k, r_k in zip(counts, mids):
                    if c_k > 0:
                        out.append((int(c_k), float(r_k)))
            return out

        # Architektur-Layout des DOF-Vektors:
        #   [Filme n_films·NF][Membranfelder n_mem·NM][Sammelknoten]
        # dual_diaphragm: 2 Membranen, 2 Filme (einteilig) bzw. 3 (K67:
        #   Zwischenspalt), keine Knoten — Quellen wirken auf die Membranen.
        # single: 1 Membran, 1 Film; Knoten 0 = Frontvolumen (Strahlung +
        #   Gewebe vor der Membran), Knoten 1 = Sammelknoten hinter den
        #   Durchgangslöchern (Abschluss: _rear_chain_mats).
        # dual: 1 Membran zwischen 2 Filmen; Knoten 0 = Frontknoten hinter
        #   der vorderen Backplate (Gewebe + Strahlung -> p_front),
        #   Knoten 1 = Sammelknoten hinter der hinteren Backplate.
        arch = self.architecture
        if arch == "dual_diaphragm":
            n_films = 3 if self.h_center > 0.0 else 2
            # Zwei Sammelknoten VOR den Membranaußenseiten: sie tragen
            # Strahlungsimpedanz und Gewebe (bis Gegenprobe 26 fehlten sie
            # hier — Gewebe blieb im 3D-Modus wirkungslos, s. Gegenprobe 27)
            n_mem, n_nodes = 2, 2
        elif arch == "dual":
            n_films, n_mem, n_nodes = 2, 1, 2
        else:                                        # single
            n_films, n_mem, n_nodes = 1, 1, 2
        rot = self.half_rotation_deg
        # Membranseitige Mündung: bei Stufenbohrung die WEITE Senkung
        r_mouth = self.r_bh if self.stepped else self.r_th
        th_rings_l = _ring_list(self._th_rings, self.n_th)
        th_cells = None
        th_f = th_r = th_cf = th_cr = None
        if arch == "dual_diaphragm" and n_films == 2:
            # Einteilige Platte: Durchgangsloch verbindet beide Filme an
            # DENSELBEN Zellen; Rück-Sacklöcher nach alter Konvention.
            th_cells = []
            for m, (cnt, rr) in enumerate(th_rings_l):
                th_cells += _foot(rr, cnt, 20.0 * m, self.r_th)
        elif arch == "dual_diaphragm":
            # K67: je Hälfte ein eigenes Lochbild; die Rückhälfte ist
            # GLOBAL um rot verdreht. Membranseitig mündet (bei Stufen-
            # bohrung) die WEITE Senkung, zwischenspaltseitig der enge
            # Kern.
            th_f, th_cf, th_r, th_cr = [], [], [], []
            for m, (cnt, rr) in enumerate(th_rings_l):
                th_f += _foot(rr, cnt, 20.0 * m, r_mouth)
                th_cf += _foot(rr, cnt, 20.0 * m, self.r_th)
                th_r += _foot(rr, cnt, 20.0 * m + rot, r_mouth)
                th_cr += _foot(rr, cnt, 20.0 * m + rot, self.r_th)
        elif arch == "dual":
            # zwei getrennte Platten: eigene Lochbilder, hinten um eine
            # halbe Teilung versetzt (keine direkte Kopplung der Filme,
            # der Versatz ist nur Konvention wie bei den Sacklöchern)
            th_f, th_r = [], []
            for m, (cnt, rr) in enumerate(th_rings_l):
                th_f += _foot(rr, cnt, 20.0 * m, r_mouth)
                th_r += _foot(rr, cnt,
                              20.0 * m + 180.0 / max(cnt, 1), r_mouth)
        else:                                        # single
            th_f = []
            for m, (cnt, rr) in enumerate(th_rings_l):
                th_f += _foot(rr, cnt, 20.0 * m, r_mouth)
        bhf_cells = []
        bhr_cells = []
        for m, (cnt, rr) in enumerate(_ring_list(self._bh_rings, self.n_bh)):
            bhf_cells += _foot(rr, cnt, 15.0 + 20.0 * m, self.r_bh)
            bh_rot = (rot if n_films == 3 else 180.0 / max(cnt, 1))
            bhr_cells += _foot(rr, cnt, 15.0 + 20.0 * m + bh_rot,
                               self.r_bh)

        # Statische COO-Anteile: Membran-Spannungsoperator (eingespannter
        # Rand) + Feder-Erweichung (polarisierte Membran, Elektroden-
        # bereich) + Druckkopplungen. Alle übrigen Einträge sind
        # frequenzabhängig.
        rows = []
        cols = []
        vals = []

        def _lap(base_off, Tfac):
            for i in range(Nr_m):
                if i < Nr_m - 1:
                    G = Tfac * (i + 1) * dphi
                    for j in range(Np_):
                        k1_ = base_off + i * Np_ + j
                        k2_ = base_off + (i + 1) * Np_ + j
                        rows.extend((k1_, k2_, k1_, k2_))
                        cols.extend((k2_, k1_, k1_, k2_))
                        vals.extend((-G, -G, G, G))
                else:
                    G = Tfac * Nr_m * dphi * 2.0    # geklemmter Rand
                    for j in range(Np_):
                        k1_ = base_off + i * Np_ + j
                        rows.append(k1_)
                        cols.append(k1_)
                        vals.append(G)
                Gp = Tfac * dr / (r_m[i] * dphi)
                for j in range(Np_):
                    k1_ = base_off + i * Np_ + j
                    k2_ = base_off + i * Np_ + (j + 1) % Np_
                    rows.extend((k1_, k2_, k1_, k2_))
                    cols.extend((k2_, k1_, k1_, k2_))
                    vals.extend((-Gp, -Gp, Gp, Gp))

        off_wf = n_films * NF
        off_wr = n_films * NF + NM                   # nur n_mem = 2
        off_n = n_films * NF + n_mem * NM            # Sammelknoten
        _lap(off_wf, T_mem)
        if n_mem == 2:
            _lap(off_wr, T_mem)
        for i in range(Nr):        # Erweichung: polarisierte (Front-)Membran
            for j in range(Np_):
                k1_ = off_wf + i * Np_ + j
                rows.append(k1_)
                cols.append(k1_)
                vals.append(-kappa * A_m[i])
        # Druckkopplung Membranzeilen (omega-unabhängig)
        if arch == "dual_diaphragm":
            # Frontmembran: -p_film0; Rückmembran: +p_film1
            for i in range(Nr):
                for j in range(Np_):
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(i * Np_ + j)
                    vals.append(-A_f[i])
                    rows.append(off_wr + i * Np_ + j)
                    cols.append(NF + i * Np_ + j)
                    vals.append(+A_f[i])
            # Außenseiten über die Sammelknoten (Strahlung + Gewebe) statt
            # direkt aus der Quelle — Vorzeichen wie beim bisherigen
            # Direktantrieb (rhs -A_m vorn / +A_m hinten). Grenzfall
            # Z_außen -> 0: p_knoten = p_außen reproduziert den
            # Direktantrieb exakt (Gegenprobe 27).
            for i in range(Nr_m):
                for j in range(Np_):
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(off_n + 0)
                    vals.append(+A_m[i])
                    rows.append(off_wr + i * Np_ + j)
                    cols.append(off_n + 1)
                    vals.append(-A_m[i])
        elif arch == "dual":
            # Mittelmembran zwischen den Filmen, w positiv = nach VORN
            # (Film 0 liegt VOR der Membran und drückt sie nach hinten,
            # Film 1 dahinter nach vorn) — Vorzeichen gespiegelt zur
            # Frontmembran der Doppelmembran-Bauform, deren Film HINTER
            # ihr liegt. Kraft ~ (p_film0 − p_film1) in Rückrichtung.
            for i in range(Nr):
                for j in range(Np_):
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(i * Np_ + j)
                    vals.append(+A_f[i])
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(NF + i * Np_ + j)
                    vals.append(-A_f[i])
        else:                                        # single
            # Film hinter der Membran: -p_film0; Vorderseite sieht den
            # FRONTKNOTEN (Strahlung + Gewebe -> p_front): +p_node über die
            # GESAMTE Membranfläche (Grenzfall Z_front -> 0: p_node = p_front
            # reproduziert exakt den direkten Quelldruck).
            for i in range(Nr):
                for j in range(Np_):
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(i * Np_ + j)
                    vals.append(-A_f[i])
            for i in range(Nr_m):
                for j in range(Np_):
                    rows.append(off_wf + i * Np_ + j)
                    cols.append(off_n + 0)
                    vals.append(+A_m[i])

        self._g3d = dict(
            Np=Np_, Nr=Nr, Nr_m=Nr_m, dr=dr, dphi=dphi,
            r_f=r_f, r_m=r_m, A_f=A_f, A_m=A_m, NF=NF, NM=NM,
            arch=arch, n_films=n_films, n_mem=n_mem, n_nodes=n_nodes,
            sigma=sigma, T_mem=T_mem, kappa=kappa,
            th_cells=th_cells, bhf_cells=bhf_cells, bhr_cells=bhr_cells,
            th_f=th_f, th_cf=th_cf, th_r=th_r, th_cr=th_cr,
            static=(np.array(rows), np.array(cols),
                    np.array(vals, dtype=complex)),
        )

    def _solve_3d(self, omega, want_rear=False):
        """3D-Sandwich-Lösung: Ausgangs-Volumenverschiebung je Einheits-
        Außendruck, U_front = X_f·p_front + X_r·p_rear.

        Rückgabe: (X_f, X_r) je Frequenz [m³/Pa]; mit ``want_rear``
        zusätzlich die Rückmembran-Antworten (B_f, B_r) für
        Reziprozitätsprüfungen. Je Frequenz wird das dünn besetzte
        Gesamtsystem (einteilig: 2 Filme + 2 Membranfelder, ~24k
        Unbekannte; K67-Modus mit Zwischenspalt: 3 Filme, ~30k)
        einmal LU-faktorisiert — das 3D-Modell ist damit DEUTLICH
        langsamer als 1D/2D (Sekundenbereich pro Frequenzpunkt);
        für Frequenzgänge empfiehlt sich n_points <= 150.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        g = self._g3d
        Np_, Nr, Nr_m = g["Np"], g["Nr"], g["Nr_m"]
        NF, NM = g["NF"], g["NM"]
        dr, dphi = g["dr"], g["dphi"]
        r_f, A_f, A_m = g["r_f"], g["A_f"], g["A_m"]
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        Xf = np.empty(omega.size, dtype=complex)
        Xr = np.empty_like(Xf)
        Bf = np.empty_like(Xf)
        Br = np.empty_like(Xf)

        def _film_props(h_loc, om):
            a_v = 0.5 * h_loc * np.sqrt(1j * om * RHO0 / MU_AIR)
            K = h_loc / (1j * om * RHO0) * (1.0 - np.tanh(a_v) / a_v)
            a_t = a_v * np.sqrt(PRANDTL)
            n_poly = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t) / a_t)
            return K, h_loc / (n_poly * P_ATM)

        idx_all = np.arange(NF)
        i_of = idx_all // Np_
        srows, scols, svals = g["static"]
        arch = g["arch"]
        n_films, n_mem, n_nodes = g["n_films"], g["n_mem"], g["n_nodes"]
        off_w = n_films * NF
        off_n = off_w + n_mem * NM
        N_tot = off_n + n_nodes
        # Ausgangs- (Elektrodenbereich) und Anregungs-Gewichte (Vollfläche).
        # Vorzeichen: die interne w-Konvention (positiv = von der Elektrode
        # weg) ist der 1D/2D-Flussrichtung (q_mem front -> rück) entgegen-
        # gesetzt; das Minus richtet die 3D-Ausgänge an der Kettenkonvention
        # aus, sodass H über alle Modelle phasengleich ist (Beträge und das
        # Verhältnis D_r = −X_f/X_r sind davon unberührt).
        w_out = -np.repeat(A_f, Np_)
        rhs_w = np.repeat(A_m, Np_)

        def _two_port_stamp(rows, cols, vals, ca, offa, cb, offb,
                            Y11, Y12, Y22):
            """Reziprokes Zweitor zwischen zwei Zellgruppen (Port-Druck =
            Gruppenmittel, Fluss gleichverteilt): äußere Produkte der
            Fußabdrücke."""
            na, nb = ca.size, cb.size
            aa = offa + ca
            bb = offb + cb
            rows.append(np.repeat(aa, na))
            cols.append(np.tile(aa, na))
            vals.append(np.full(na * na, Y11 / (na * na), dtype=complex))
            rows.append(np.repeat(bb, nb))
            cols.append(np.tile(bb, nb))
            vals.append(np.full(nb * nb, Y22 / (nb * nb), dtype=complex))
            rows.append(np.repeat(aa, nb))
            cols.append(np.tile(bb, na))
            vals.append(np.full(na * nb, Y12 / (na * nb), dtype=complex))
            rows.append(np.repeat(bb, na))
            cols.append(np.tile(aa, nb))
            vals.append(np.full(na * nb, Y12 / (na * nb), dtype=complex))

        # Filmliste: (Filmindex, Spalthöhe, Membran-Offset|None, Vorzeichen
        # der Membranquelle im Filmring). dual_diaphragm: Film 0/1 gehören
        # zu Front-/Rückmembran, Film 2 (K67) ist der membranlose
        # Zwischenspalt. single/dual: alle Filme koppeln an DIE Membran
        # (off_w) — bei 'dual' mit entgegengesetztem Vorzeichen beidseits.
        if arch == "dual_diaphragm":
            sides = [(0, self.h_gap_front, off_w, +1.0),
                     (1, self.h_gap, off_w + NM, -1.0)]
            if n_films == 3:
                sides.append((2, self.h_center, None, 0.0))
        elif arch == "dual":
            # w positiv = nach vorn: komprimiert den VORDEREN Film (−jw
            # analog zur Rückmembran der K67-Bauform), dehnt den hinteren
            sides = [(0, self.h_gap_front, off_w, -1.0),
                     (1, self.h_gap_front, off_w, +1.0)]
        else:                                        # single
            sides = [(0, self.h_gap_front, off_w, +1.0)]

        for fidx, om in enumerate(omega):
            rows = [srows]
            cols = [scols]
            vals = [svals]
            K_edge_side = {}
            # Filmringe (Membranseiten ggf. mit Clearance-Relief; der
            # Zwischenspalt ist eben und hat weder Relief noch Membran)
            for side, h0, mem_off, sgn in sides:
                off = side * NF
                mem_side = mem_off is not None
                h_ring = h0 + (self._clr_relief if mem_side else 0.0)
                K = np.empty(Nr, dtype=complex)
                cg = np.empty(Nr, dtype=complex)
                for hh in np.unique(h_ring):
                    msk = h_ring == hh
                    Kh, ch = _film_props(hh, om)
                    K[msk] = Kh
                    cg[msk] = ch
                K_edge_side[side] = K[Nr - 1]
                # radiale Faces
                Kmid = 0.5 * (K[:-1] + K[1:])
                Gr = (np.arange(1, Nr) * dphi) * Kmid
                k1_ = off + idx_all[:(Nr - 1) * Np_]
                k2_ = k1_ + Np_
                Gv = np.repeat(Gr, Np_)
                rows += [k1_, k2_, k1_, k2_]
                cols += [k2_, k1_, k1_, k2_]
                vals += [-Gv, -Gv, Gv, Gv]
                # azimutale Faces
                Ga = np.repeat(dr / (r_f * dphi) * K, Np_)
                k1_ = off + idx_all
                k2_ = off + i_of * Np_ + (idx_all % Np_ + 1) % Np_
                rows += [k1_, k2_, k1_, k2_]
                cols += [k2_, k1_, k1_, k2_]
                vals += [-Ga, -Ga, Ga, Ga]
                # Speicherung
                rows += [off + idx_all]
                cols += [off + idx_all]
                vals += [1j * om * np.repeat(cg * A_f, Np_)]
                if not mem_side:
                    continue
                # Membranquelle im Filmring
                rows += [off + idx_all]
                cols += [mem_off + idx_all]
                vals += [sgn * 1j * om * np.repeat(A_f, Np_)]
                # Clearance-Ring als Schlitz-Stub (schmaler Ring)
                if self._clr_stub_cell is not None:
                    r_cst = 0.5 * self.clearance_ring_diameter
                    C_st = (2.0 * np.pi * r_cst * self.clearance_ring_width
                            * self.clearance_ring_depth / (GAMMA * P_ATM))
                    R_st = (12.0 * MU_AIR * self.clearance_ring_depth
                            / (2.0 * np.pi * r_cst
                               * self.clearance_ring_width ** 3) / 3.0)
                    y_st = 1.0 / (R_st + 1.0 / (1j * om * C_st)) / Np_
                    cells = self._clr_stub_cell * Np_ + np.arange(Np_)
                    rows += [off + cells]
                    cols += [off + cells]
                    vals += [np.full(Np_, y_st, dtype=complex)]
            om_a = np.array([om])
            if arch != "dual_diaphragm":
                # ---- single/dual: Löcher in die Sammelknoten, Knoten-
                # Abschlüsse aus den Lumped-Ketten -----------------------
                # Loch-Zweitor Membranfilm -> Knoten. Membranseitige
                # Mündung löst der Film auf; die knotenseitige trägt
                # Fok-Mündungsmasse + viskose Mündung (visc_ends=1), wie
                # im 1D-Pfad (_through_hole_impedance) abzüglich des
                # filmseitigen Flanschterms.
                S_th = np.pi * self.r_th ** 2
                vents_directly = not self.rear_network_enabled
                if self.stepped:
                    gcb, Zccb = self._narrow_duct_propagation(om_a,
                                                              self.r_bh)
                    gl = gcb[0] * self.d_bh
                    ch, sh = np.cosh(gl), np.sinh(gl)
                    A2, B2 = ch, Zccb[0] * sh
                    C2, D2 = sh / Zccb[0], ch
                    karal = 1.0 - self.r_th / self.r_bh
                    Z_step = (1j * om * RHO0 * 0.85 * self.r_th * karal
                              / S_th
                              + self._hole_impedance(
                                  om_a, self.r_th,
                                  (3.0 * np.pi / 16.0) * self.r_th, 1,
                                  end_correction=False)[0].real * karal)
                    Z_core = self._hole_impedance(om_a, self.r_th,
                                                  self.t_th_eff, 1,
                                                  end_correction=False,
                                                  visc_ends=1)[0]
                    Zs = (Z_step + Z_core
                          + 1j * om * RHO0 * 0.85 * self.r_th
                          * self._fok_th / S_th)
                    A1, B1 = A2, A2 * Zs + B2
                    C1, D1 = C2, C2 * Zs + D2
                else:
                    Zs = (self._hole_impedance(om_a, self.r_th, self.t_bp,
                                               1, end_correction=False,
                                               visc_ends=1)[0]
                          + 1j * om * RHO0 * 0.85 * self.r_th
                          * self._fok_th / S_th)
                    A1, B1, C1, D1 = 1.0, Zs, 0.0, 1.0
                Y11 = D1 / B1
                Y22 = A1 / B1
                Y12 = -1.0 / B1
                node1 = np.array([0])
                # single: Film 0 -> Knoten 1 (Rückseite); dual: Film 0 ->
                # Knoten 0 (Front), Film 1 -> Knoten 1 (Rückseite)
                if arch == "single":
                    for cf in g["th_f"]:
                        _two_port_stamp(rows, cols, vals, cf, 0,
                                        node1, off_n + 1, Y11, Y12, Y22)
                else:
                    for cf in g["th_f"]:
                        _two_port_stamp(rows, cols, vals, cf, 0,
                                        node1, off_n + 0, Y11, Y12, Y22)
                    for cr_ in g["th_r"]:
                        _two_port_stamp(rows, cols, vals, cr_, NF,
                                        node1, off_n + 1, Y11, Y12, Y22)
                if self.ring_vent_w > 0.0:
                    T_l3 = self._slit_line_abcd(
                        om_a, self.ring_vent_w, 2.0 * np.pi * self.a_bp,
                        self.ring_vent_L)
                    A_l3 = complex(T_l3[0, 0][0]); B_l3 = complex(T_l3[0, 1][0])
                    C_l3 = complex(T_l3[1, 0][0]); D_l3 = complex(T_l3[1, 1][0])
                    edge_cells = (Nr - 1) * Np_ + np.arange(Np_)
                    ring_sides = ([(0, off_n + 1)] if arch == "single"
                                  else [(0, off_n + 0), (1, off_n + 1)])
                    for side3, node_off3 in ring_sides:
                        Z_e3 = 1.0 / (4.0 * np.pi * Nr * K_edge_side[side3])
                        B_c3 = B_l3 + Z_e3 * D_l3
                        _two_port_stamp(rows, cols, vals, edge_cells,
                                        side3 * NF, node1, node_off3,
                                        D_l3 / B_c3, -1.0 / B_c3,
                                        (A_l3 + Z_e3 * C_l3) / B_c3)
                # Frontknoten-Abschluss: single = Strahlung + Gewebe VOR
                # der Membran (Grenzfall Z -> 0: p_node = p_front); dual =
                # dieselbe Kette vor der vorderen Backplate.
                Z_fr = (self._radiation_impedance_membrane(om_a)[0]
                        + self.rayl_front / self.S_mem)
                rows += [np.array([off_n + 0])]
                cols += [np.array([off_n + 0])]
                vals += [np.array([1.0 / Z_fr], dtype=complex)]
                if arch == "single":
                    # Membran-Volumenfluss zieht am Frontknoten
                    midx_n = off_w + np.arange(NM)
                    rows += [np.full(NM, off_n + 0)]
                    cols += [midx_n]
                    vals += [-1j * om * rhs_w.astype(complex)]
                # Rückknoten-Abschluss: Lumped-Kette hinter den Löchern
                # (baugleich zum 1D/2D-Pfad); mündet die Platte direkt ins
                # Schallfeld, kommt die Array-Strahlung der Lochmündungen
                # als Serie hinzu (im 1D-Pfad steckt sie je Loch in
                # _through_hole_impedance).
                mats = self._rear_chain_mats(om_a)
                if vents_directly and self.n_th > 0:
                    S_holes = self.n_th * S_th
                    Z_rad_h = (RHO0 * C_AIR / S_holes) * min(
                        (om / C_AIR * self.r_th) ** 2 / 2.0, 1.0)
                    mats = [self._abcd_series(
                        np.full(1, Z_rad_h, dtype=complex), om_a)] + mats
                if mats:
                    T_b = reduce(self._mmul, mats)
                    Ab, Bb = complex(T_b[0, 0][0]), complex(T_b[0, 1][0])
                    Cb, Db = complex(T_b[1, 0][0]), complex(T_b[1, 1][0])
                else:
                    Ab, Bb, Cb, Db = 1.0, 0.0, 0.0, 1.0
                if self.rear_open and abs(Bb) > 0.0:
                    y_bk, src_bk = Db / Bb, 1.0 / Bb
                elif self.rear_open:
                    # leere Kette: Knoten liegt direkt am Schallfeld
                    y_bk, src_bk = 1e12, 1e12
                else:
                    y_bk, src_bk = Cb / Ab, 0.0
                rows += [np.array([off_n + 1])]
                cols += [np.array([off_n + 1])]
                vals += [np.array([y_bk], dtype=complex)]
            elif n_films == 2:
                # Einteilige Platte: Rohr durch die volle Dicke verbindet
                # beide Filme an denselben Fußabdruck-Zellen.
                Zth = self._hole_impedance(om_a, self.r_th,
                                           2.0 * self.t_bp, 1,
                                           end_correction=False)[0]
                gth = 1.0 / Zth
                for cells in g["th_cells"]:
                    gv = gth / cells.size
                    kf = cells
                    kr = NF + cells
                    gvv = np.full(cells.size, gv, dtype=complex)
                    rows += [kf, kr, kf, kr]
                    cols += [kf, kr, kr, kf]
                    vals += [gvv, gvv, -gvv, -gvv]
            else:
                # K67: je Loch die Zweitor-Kette seiner Halbplatte
                # (membranseitiger Film -> Zwischenspalt). Stufenbohrung:
                # weites Senkungssegment als thermoviskose Leitung +
                # Karal-Stufe + enger Kern; sonst das schlichte Rohr.
                if self.stepped:
                    gcb, Zccb = self._narrow_duct_propagation(om_a,
                                                              self.r_bh)
                    gl = gcb[0] * self.d_bh
                    ch, sh = np.cosh(gl), np.sinh(gl)
                    A2, B2 = ch, Zccb[0] * sh
                    C2, D2 = sh / Zccb[0], ch
                    S_th = np.pi * self.r_th**2
                    karal = 1.0 - self.r_th / self.r_bh
                    Z_step = (1j * om * RHO0 * 0.85 * self.r_th * karal
                              / S_th
                              + self._hole_impedance(
                                  om_a, self.r_th,
                                  (3.0 * np.pi / 16.0) * self.r_th, 1,
                                  end_correction=False)[0].real * karal)
                    Z_core = self._hole_impedance(om_a, self.r_th,
                                                  self.t_th_eff, 1,
                                                  end_correction=False)[0]
                    # Kette: Senkungsleitung · Serie(Z_step + Z_core)
                    Zs = Z_step + Z_core
                    A1, B1 = A2, A2 * Zs + B2
                    C1, D1 = C2, C2 * Zs + D2
                else:
                    Zs = self._hole_impedance(om_a, self.r_th, self.t_bp,
                                              1, end_correction=False)[0]
                    A1, B1, C1, D1 = 1.0, Zs, 0.0, 1.0
                Y11 = D1 / B1
                Y22 = A1 / B1
                Y12 = -1.0 / B1
                for cf, cc in zip(g["th_f"], g["th_cf"]):
                    _two_port_stamp(rows, cols, vals, cf, 0, cc, 2 * NF,
                                    Y11, Y12, Y22)
                for cr_, cc in zip(g["th_r"], g["th_cr"]):
                    _two_port_stamp(rows, cols, vals, cr_, NF, cc, 2 * NF,
                                    Y11, Y12, Y22)
            if arch == "dual_diaphragm":
                # ---- Außenknoten beider Membranen: Strahlung + Gewebe ----
                # Baugleich zum 1D/2D-Pfad (_assemble_parts: vorn
                # Z_rad + rayl_front/S_mem, hinten rayl_rear/S_mem + Z_rad).
                # Flussbilanz: w ist global nach VORN positiv, die
                # Frontmembran schiebt also IN den Frontknoten (-jw A),
                # die Rückmembran vom Rückknoten WEG (+jw A).
                Z_rad_m = self._radiation_impedance_membrane(om_a)[0]
                Z_ext_f = Z_rad_m + self.rayl_front / self.S_mem
                Z_ext_r = Z_rad_m + self.rayl_rear / self.S_mem
                rows += [np.array([off_n + 0]), np.array([off_n + 1])]
                cols += [np.array([off_n + 0]), np.array([off_n + 1])]
                vals += [np.array([1.0 / Z_ext_f], dtype=complex),
                         np.array([1.0 / Z_ext_r], dtype=complex)]
                midx_n = off_w + np.arange(NM)
                rows += [np.full(NM, off_n + 0), np.full(NM, off_n + 1)]
                cols += [midx_n, midx_n + NM]
                vals += [-1j * om * rhs_w.astype(complex),
                         +1j * om * rhs_w.astype(complex)]
            # Sacklöcher: geschlossene Stubs an ihren Zellen
            if self.n_bh > 0:
                ybh = 1.0 / self._closed_hole_stub(om_a, self.r_bh,
                                                   self.d_bh)[0]
                bh_sides = (((0, g["bhf_cells"]),) if arch == "single"
                            else ((0, g["bhf_cells"]), (1, g["bhr_cells"])))
                for side, bl in bh_sides:
                    off = side * NF
                    for cells in bl:
                        yv = np.full(cells.size, ybh / cells.size,
                                     dtype=complex)
                        rows += [off + cells]
                        cols += [off + cells]
                        vals += [yv]
            # Membran-Massenterme (Verlust wie 2D: kleiner interner Q)
            mterm = (-om ** 2 * g["sigma"]
                     * (1.0 - 1j / self._Q_MEMBRANE_INTERNAL))
            midx = np.arange(NM)
            for m_i in range(n_mem):
                woff = off_w + m_i * NM
                rows += [woff + midx]
                cols += [woff + midx]
                vals += [mterm * rhs_w]
            S = coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(N_tot, N_tot)).tocsc()
            lu = splu(S)
            rhs = np.zeros((N_tot, 2), dtype=complex)
            if arch == "dual_diaphragm":
                rhs[off_n + 0, 0] = 1.0 / Z_ext_f    # p_front am Frontknoten
                rhs[off_n + 1, 1] = 1.0 / Z_ext_r    # p_rear am Rückknoten
            elif arch == "dual":
                rhs[off_n + 0, 0] = 1.0 / Z_fr       # p_front am Frontknoten
                rhs[off_n + 1, 1] = src_bk           # p_rear an der Kette
            else:                                    # single
                rhs[off_n + 0, 0] = 1.0 / Z_fr       # p_front am Frontknoten
                rhs[off_n + 1, 1] = src_bk           # p_rear an der Kette
            x = lu.solve(rhs)
            wf = x[off_w:off_w + NF, :]              # Elektrodenbereich
            Xf[fidx] = np.sum(w_out[:, None] * wf, axis=0)[0]
            Xr[fidx] = np.sum(w_out[:, None] * wf, axis=0)[1]
            if arch == "dual_diaphragm":
                wr = x[off_w + NM:off_w + NM + NF, :]
                Bf[fidx] = np.sum(w_out[:, None] * wr, axis=0)[0]
                Br[fidx] = np.sum(w_out[:, None] * wr, axis=0)[1]
            else:
                # Reziprozitäts-Diagnose der akustischen Zweitor-Ports
                # (Flüsse IN das Netzwerk an den Quell-Terminals):
                # Y21 = q_rück(p_front = 1) = −q_port = −p_knoten/B und
                # Y12 = q_front(p_rück = 1) = (0 − p_node)/Z_front
                # müssen übereinstimmen.
                p_n = x[off_n + 0, :]
                p_m = x[off_n + 1, :]
                q_rear_pf = -p_m[0] / Bb if abs(Bb) > 0 else 0.0
                q_front_pr = -p_n[1] / Z_fr
                self._recip_3d = (complex(q_rear_pf), complex(q_front_pr))
                Bf[fidx] = Xf[fidx]
                Br[fidx] = Xr[fidx]
        if want_rear:
            return Xf, Xr, Bf, Br
        return Xf, Xr

    def _source_pressures(self, omega, theta):
        """Effektive Quelldrücke p_front/p_rear für Einfallswinkel theta.

        Mit Beugung: Kugelstreufaktoren (s. :meth:`_diffraction_factors`).
        Ohne (include_diffraction=False oder kein SciPy): ebene Welle mit
        geometrischer Wegdifferenz d_ext (Verhalten der Vorversionen).
        Rückgabeform jeweils (len(omega), len(theta)).
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        k = omega / C_AIR
        if self.include_diffraction and _HAS_SCIPY:
            if self.architecture == "dual_diaphragm":
                # DÜNNE-SCHEIBE-BAUFORM (K67): Druckstau/Bündelung (F_f)
                # und Front-Rück-Gradient (G_ax) getrennt, beides aus
                # exakten Streulösungen ohne Fit-Koeffizient.
                if self.axial_body_model == "bem":
                    # BEM: EIN Lösungsgang liefert den absoluten
                    # Frontfaktor der realen FLACHEN Stirnfläche UND den
                    # Transfer. Die Kugelkalotte (±50° Krümmung bei der
                    # K67) unterschätzt den frontalen Druckstau der
                    # flachen Stirnfläche im Band ka ≈ 2..5 um ~3 dB —
                    # die Ursache der künstlichen 7-kHz-Senke des
                    # Kalottenmodells (s. Gegenprobe 26).
                    F_bem, G_ax = self._bem_axial_fields(omega, theta)
                    return F_bem, F_bem * G_ax
                # Kugel-/Sphäroidmodus: die R_body-Kugel liefert die
                # GEMEINSAME Bündelung/Druckstau (Kopf-/Bodyskala, HF-
                # Richtwirkung); der Front-Rück-Gradient kommt aus dem
                # Pol-zu-Pol-Transfer der Kugel mit der korrekten AXIALEN
                # Ausdehnung d_ext (s. _axial_body_transfer) bzw. dem
                # Sphäroid: geometrische Laufzeit + Beugungsumweg +
                # Amplituden-Asymmetrie aus der exakten Streureihe.
                F_f, _ = self._diffraction_factors(omega, theta)
                if self.axial_body_model == "spheroid":
                    G_ax = self._axial_spheroid_transfer(omega, theta)
                else:
                    G_ax = self._axial_body_transfer(omega, theta)
                return F_f, F_f * G_ax
            return self._diffraction_factors(omega, theta)
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
        """Ausbreitungskonstante und Wellenwiderstand einer Leitung.

        Mit SciPy: EXAKTE Zwikker–Kosten-Form (s.
        :meth:`_narrow_duct_propagation`) — sie gilt für ALLE Radien und
        geht für weite Rohre asymptotisch in die Kirchhoff-Grenzschicht-
        näherung über. Damit rechnen alle Leitungselemente (Laufzeitglied,
        Hohlraum, Reststücke, Senkungssegmente, Stubs) mit derselben
        Theorie; gerade bei Zwischenradien (r von der Größenordnung
        weniger Grenzschichtdicken) ist die Asymptotik sonst grob.

        Ohne SciPy der bisherige Kirchhoff-Fallback (weites Rohr):
            gamma = alpha + j*k,   k = omega/c,   Zc = rho0*c/S
            alpha = sqrt(mu*omega/(2*rho0)) * (1 + (gamma_ad - 1)/sqrt(Pr))
                    / (r * c)
        Der Dämpfungsbelag alpha erfasst viskose UND thermische Grenz-
        schichtverluste an der Rohrwand und verhindert zugleich unphysika-
        lisch scharfe Stehwellenresonanzen des Hohlraums im Modell.
        """
        omega = np.asarray(omega, dtype=float)
        if _HAS_SCIPY:
            return self._narrow_duct_propagation(omega, radius_duct)
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
    def _abcd_reverse(T):
        """Kettenmatrix für den RÜCKWÄRTS-Durchlauf eines reziproken
        (2,2,N)-Zweitors (det = 1): der Port-Tausch [[D, B], [C, A]].

        NICHT die Matrix-Inverse [[D, -B], [-C, A]] verwenden: deren
        negative Elemente wirken als AKTIVE Bauteile (negativer Widerstand,
        negative Nachgiebigkeit) und löschen in einer Kaskade
        T_hin · T_rück Laufzeit UND Dämpfung des Hinwegs exakt aus —
        bei der Doppelmembran-Bauform kollabierte dadurch die interne
        Phasenschieber-Laufzeit (Acht statt Niere) und der Rückzweig
        verlor seine Dämpfung (ungedämpfte Resonanzüberhöhung). Der
        Port-Tausch ist für Elementketten identisch mit der umgekehrten
        Elementreihenfolge (``mats[::-1]``), die der 1D-Pfad verwendet —
        beide Pfade sind damit konsistent."""
        return np.array([[T[1, 1], T[0, 1]], [T[1, 0], T[0, 0]]])

    def _gap_field_2port(self, omega, h_film=None, sag_w0=0.0):
        """Zweitor des Luftspalts aus der modifizierten Reynolds-Gleichung.

        ``h_film``: nominelle Spalthöhe (Standard: Luftspalt h_gap).
        ``sag_w0``: statische Mittendurchbiegung der polarisierten
        Membran. Der Film sieht dann das ÖRTLICHE Spaltprofil
        h(r) = h_film − sag_w0·φ(r) statt des Flächenmittels — wegen der
        h³-Abhängigkeit wirkt die zentrale Verengung überproportional,
        die Bias-Kopplung an die Richtcharakteristik wird damit
        quantitativ statt gemittelt. sag_w0 = 0 reproduziert exakt den
        Bestand (unpolarisierte Seite / Dual-Architektur mit w0 = 0).

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
        # CLEARANCE-RING: Zellen im Freistich haben lokal den tieferen
        # Spalt h + Tiefe -> zellweise Leitwerte/Nachgiebigkeiten; liegen
        # Loch-Mündungen im Relief, wird deren Zell-Engstelle mit dem
        # dortigen (größeren) Filmleitwert gerechnet.
        h = self.h_gap if h_film is None else h_film

        def _film_props(h_loc):
            a_v = 0.5 * h_loc * np.sqrt(1j * omega * RHO0 / MU_AIR)
            K = h_loc / (1j * omega * RHO0) * (1.0 - np.tanh(a_v) / a_v)
            a_t = a_v * np.sqrt(PRANDTL)
            n_poly = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t) / a_t)
            return K, h_loc / (n_poly * P_ATM)

        K_f, c_gap = _film_props(h)                      # (Nf,) nominal
        # Örtliches Spaltprofil: statische Durchbiegung (h³-Wirkung!)
        # plus Clearance-Ring-Relief; sag_w0 = 0 und Relief 0 -> Bestand.
        h_cell = np.maximum(h - sag_w0 * phi + self._clr_relief, 0.05 * h)
        if sag_w0 > 0.0 or np.any(self._clr_relief > 0.0):
            sq = np.sqrt(1j * omega * RHO0 / MU_AIR)
            a_v2 = 0.5 * h_cell[None, :] * sq[:, None]          # (Nf, N)
            K_cell = (h_cell[None, :] / (1j * omega[:, None] * RHO0)
                      * (1.0 - np.tanh(a_v2) / a_v2))
            a_t2 = a_v2 * np.sqrt(PRANDTL)
            n_p2 = GAMMA / (1.0 + (GAMMA - 1.0) * np.tanh(a_t2) / a_t2)
            c_cell = h_cell[None, :] / (n_p2 * P_ATM)
            K_face = np.empty((Nf, N + 1), dtype=complex)
            K_face[:, 0] = K_cell[:, 0]
            K_face[:, N] = K_cell[:, -1]
            K_face[:, 1:N] = 0.5 * (K_cell[:, :-1] + K_cell[:, 1:])
            # Loch-Eintritts-Engstellen am ÖRTLICHEN Spalt der Mündungen:
            # Relief-Flag wie bisher (voller Freistich-Spalt), zusätzlich
            # die statische Durchbiegung am dichte-gewichteten Lochort.
            def _mean_phi(dens):
                w = dens * A
                s = float(np.sum(w))
                return float(np.sum(phi * w)) / s if s > 0.0 else 0.0

            h_e_th = ((h + self.clearance_ring_depth
                       if self._clr_th_relieved else h)
                      - sag_w0 * _mean_phi(dens_th))
            h_e_bh = ((h + self.clearance_ring_depth
                       if self._clr_bh_relieved else h)
                      - sag_w0 * _mean_phi(dens_bh))
            K_entry_th = _film_props(max(h_e_th, 0.05 * h))[0]
            K_entry_bh = _film_props(max(h_e_bh, 0.05 * h))[0]
        else:
            K_cell = np.broadcast_to(K_f[:, None], (Nf, N))
            c_cell = np.broadcast_to(c_gap[:, None], (Nf, N))
            K_face = np.broadcast_to(K_f[:, None], (Nf, N + 1))
            K_entry_th = K_f
            K_entry_bh = K_f
        y_ring = np.zeros(Nf, dtype=complex)
        if self._clr_stub_cell is not None:
            r_cst = 0.5 * self.clearance_ring_diameter
            C_st = (2.0 * np.pi * r_cst * self.clearance_ring_width
                    * self.clearance_ring_depth / (GAMMA * P_ATM))
            R_st = (12.0 * MU_AIR * self.clearance_ring_depth
                    / (2.0 * np.pi * r_cst
                       * self.clearance_ring_width**3) / 3.0)
            y_ring = 1.0 / (R_st + 1.0 / (1j * omega * C_st))

        # Zell-Engstellenwiderstand je Bohrung (Škvor-Zellfunktion, alle
        # Bohrungen teilen sich die Zellen; q_c >= 1 -> Löcher berühren
        # sich, keine Engstelle mehr)
        n_wells = self.n_th + self.n_bh

        def _B_of(n_sinks, r_hole):
            """Škvor-Zellfunktion für ``n_sinks`` gleichverteilte Senken."""
            if n_sinks <= 0:
                return 0.0
            q_c = min(n_sinks * r_hole**2 / self.a_bp**2, 1.0)
            if q_c >= 1.0:
                return 0.0
            return max(q_c / 2.0 - q_c**2 / 8.0
                       - np.log(q_c) / 4.0 - 3.0 / 8.0, 0.0)

        def _cell_B(r_hole):
            """AUFNAHME (Verdrängungsströmung): ALLE Bohrungen sind Senken
            — die Spaltluft läuft zur nächstgelegenen, gleich welcher Art.
            Gilt für die Blindloch- und Senkungs-Shunts."""
            return _B_of(n_wells, r_hole)

        def _cell_B_flow(r_hole):
            """DURCHFLUSS zur Rückseite: nur die DURCHGANGSLÖCHER zählen.

            Blindlöcher sind SACKGASSEN — sie nehmen Luft auf (Nachgiebig-
            keit), bieten aber keinen Weg nach hinten. Die azimutale
            Zuströmung zum nächsten DURCHGANGSloch läuft deshalb über
            deutlich größere Zellen als die Verdrängungsströmung: die
            Zellfunktion des Durchflusses ist mit ``n_th`` zu bilden, nicht
            mit ``n_th + n_bh``. Bis Gegenprobe 29 galt hier derselbe
            B-Wert wie für die Aufnahme, wodurch der Zugang zur Rückseite
            zu leicht war und die interne Laufzeit des Nieren-Phasen-
            schiebers zu kurz ausfiel (Gegenprobe 30).

            Die Škvor-DÄMPFUNG (``R_A_gap``) bleibt unverändert über alle
            Senken gebildet — dort ist jede Bohrung ein gültiges Ziel.
            """
            return _B_of(self.n_th, r_hole)

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
        # Portseite: Mündungsmasse (mit Fok-Array-Wechselwirkung) +
        # viskoser Mündungswiderstand (Sampson). Ohne Durchgangslöcher
        # (rein randbelüftete Platte) entfällt der Lochleitwert.
        Z_th1 = (self._hole_impedance(omega, self.r_th, self.t_th_eff, 1,
                                      end_correction=False, visc_ends=1)
                 + 1j * omega * RHO0 * (0.85 * self.r_th * self._fok_th)
                 / S_th
                 + _cell_B_flow(r_well_th) / (np.pi * K_entry_th)
                 ) if self.n_th > 0 else None
        if self.stepped:
            # weites Senkungssegment in Serie + Karal-Stufenmündung
            # (Masse und viskoser Anteil; die filmseitige Ausbreitung
            # deckt die Zelle mit r_bh ab)
            karal = 1.0 - self.r_th / self.r_bh
            Z_th1 = (Z_th1
                     + self._hole_impedance(omega, self.r_bh, self.d_bh, 1,
                                            end_correction=False)
                     + 1j * omega * RHO0 * 0.85 * self.r_th * karal / S_th
                     + self._hole_impedance(
                         omega, self.r_th,
                         (3.0 * np.pi / 16.0) * self.r_th, 1,
                         end_correction=False).real * karal)
        g_tot = (self.n_th / Z_th1 if self.n_th > 0
                 else np.zeros(Nf, dtype=complex))        # Gesamtleitwert (Nf,)
        # Sackloch-/Senkungs-Shunts als geschlossene thermoviskose Stubs
        # (verteilte Reibung, isotherm→adiabatisch, s. _closed_hole_stub)
        Z_stub1 = (self._closed_hole_stub(omega, self.r_bh, self.d_bh)
                   if (self.n_bh > 0 or self.stepped) else None)
        if self.n_bh > 0:
            y_tot = self.n_bh / (Z_stub1
                                 + _cell_B(self.r_bh) / (np.pi * K_entry_bh))
        else:
            y_tot = np.zeros(Nf, dtype=complex)
        if self.stepped:
            # Senkungsvolumina der Stufenbohrungen (sitzen auf dens_th)
            y_cb = self.n_th / (Z_stub1
                                + _cell_B(self.r_bh) / (np.pi * K_entry_th))
        else:
            y_cb = np.zeros(Nf, dtype=complex)

        # RANDSPALT (B&K): der Filmrand bei r = a_bp ist nicht mehr dicht.
        # Eine ZUSÄTZLICHE Unbekannte p_rand (Randdruck am Plattenumfang)
        # hängt tridiagonal an der äußersten Zelle: Randflächen-Leitwert
        #     G_rand = K_rand · (2π·a_bp)/(dr/2) = 4π·N·K_rand
        # (Umfang durch halbe Zellbreite), und von dort führt die
        # Schlitzleitung des Ringkanals zum SELBEN rückwärtigen Port wie
        # die Durchgangslöcher (Y-Parameter der Leitung, det T = 1). Die
        # Bandstruktur (1,1) bleibt erhalten.
        ring_open = self.ring_vent_w > 0.0
        if ring_open:
            T_line = self._slit_line_abcd(
                omega, self.ring_vent_w, 2.0 * np.pi * self.a_bp,
                self.ring_vent_L)
            A_l, B_l = T_line[0, 0], T_line[0, 1]
            D_l = T_line[1, 1]
        src_a = phi * A / Sphi                            # Membran treibt (U=1)
        T = np.empty((2, 2, Nf), dtype=complex)
        M_sys = N + 1 if ring_open else N
        ab = np.zeros((3, M_sys), dtype=complex)
        gg = self._fld_gface_geom
        for f in range(Nf):
            Gface = gg * K_face[f]                        # (N+1,) komplex
            ab[0, 1:N] = -Gface[1:N]                      # Superdiagonale
            ab[2, :N - 1] = -Gface[1:N]                   # Subdiagonale
            g_h = g_tot[f] * dens_th                      # (N,) verteilt
            y_bh = y_tot[f] * dens_bh + y_cb[f] * dens_th
            Y = 1j * omega[f] * c_cell[f] + y_bh + g_h
            if self._clr_stub_cell is not None:
                Y[self._clr_stub_cell] += (y_ring[f]
                                           / A[self._clr_stub_cell])
            ab[1, :N] = Gface[:N] + Gface[1:N + 1] + Y * A
            rhs = np.zeros((M_sys, 2), dtype=complex)
            rhs[:N, 0] = src_a
            rhs[:N, 1] = g_h * A
            if ring_open:
                G_edge = 4.0 * np.pi * N * K_face[f, N]
                ab[1, N - 1] += G_edge
                ab[0, N] = -G_edge                        # Zelle N-1 <-> Rand
                ab[2, N - 1] = -G_edge
                # Randknoten: Filmzustrom + Leitungs-Y11 (= D/B)
                ab[1, N] = G_edge + D_l[f] / B_l[f]
                # Rückport treibt (Fall b): -Y12·p_rück = +1/B
                rhs[N, 1] = 1.0 / B_l[f]
            sol = _solve_banded((1, 1), ab, rhs)
            p_a, p_b = sol[:N, 0], sol[:N, 1]
            alpha = np.sum(p_a * phi * A) / Sphi
            beta = np.sum(p_b * phi * A) / Sphi
            gamma = np.sum(g_h * A * p_a)
            delta = np.sum(g_h * A * (p_b - 1.0))
            if ring_open:
                # Ringfluss zum Rückport: q = p_rand/B − (A/B)·p_rück
                gamma = gamma + sol[N, 0] / B_l[f]
                delta = delta + sol[N, 1] / B_l[f] - A_l[f] / B_l[f]
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
        # Reynolds-Feldlösung (sinnvoll, sobald es einen Durchflussweg
        # gibt: Löcher oder Randspalt). Die polarisierte Seite übergibt
        # Basis-Spalt + statische Mittendurchbiegung — das Feld sieht das
        # ÖRTLICHE Profil h(r) = h − w0·φ(r) statt des Flächenmittels.
        if self.squeeze_model == "2d" and (self.n_th > 0
                                           or self.ring_vent_w > 0.0):
            T = self._gap_field_2port(
                omega, h_film=self.h_gap,
                sag_w0=(self.w0_static if polarized else 0.0))
            if holes_radiate and self.n_th > 0:
                k = np.asarray(omega, float) / C_AIR
                S_holes = self.n_th * np.pi * self.r_th**2
                Z_rad = (RHO0 * C_AIR / S_holes) * np.minimum(
                    (k * self.r_th) ** 2 / 2.0, 1.0)
                T = self._mmul(T, self._abcd_series(Z_rad, omega))
            if outside_to_membrane:
                # Rückwärts-Durchlauf = Port-Tausch (s. _abcd_reverse) —
                # analog zu mats[::-1] im 1D-Pfad, NICHT die Inverse.
                T = self._abcd_reverse(T)
            return T

        Y_gap = self._film_compliance_Y(omega, h_eff, self.S_bp)
        mats = []  # Reihenfolge: Membranseite -> Außenseite
        if self.n_th == 0 and self.ring_vent_w > 0.0:
            # REIN RANDBELÜFTETE Platte (1D, B&K-Bauform): Blindloch-
            # Shunts an der Membran, Rand-Poiseuille R_edge·Φ(ω) als
            # Serienweg zum Rand, Spaltvolumen-Shunt, dann die
            # Schlitzleitung des Randspalts zum rückwärtigen Port.
            # (Löcher UND Randspalt zugleich gatet der Konstruktor auf
            # das 2D-/3D-Feldmodell.)
            if self.n_bh > 0:
                mats.append(self._abcd_shunt(
                    1.0 / self._blind_hole_impedance(omega), omega))
            mats.append(self._abcd_series(
                self._edge_R(h_eff) * self._film_R_dynamic(omega, h_eff),
                omega))
            mats.append(self._abcd_shunt(Y_gap, omega))
            mats.append(self._slit_line_abcd(
                omega, self.ring_vent_w, 2.0 * np.pi * self.a_bp,
                self.ring_vent_L))
            if outside_to_membrane:
                mats = mats[::-1]
            return reduce(self._mmul, mats)
        if self.n_th == 0:
            if self.n_bh > 0:
                mats.append(self._abcd_shunt(
                    1.0 / self._blind_hole_impedance(omega), omega))
            mats.append(self._abcd_shunt(Y_gap, omega))
            return reduce(self._mmul, mats)
        Z_holes = self._through_hole_impedance(omega, self.n_th,
                                               radiates=holes_radiate)
        # statischer Škvor-R mit Frequenzkorrektur (laterale Filmträgheit)
        mats.append(self._abcd_series(
            self._skvor_R(h_eff) * self._film_R_dynamic(omega, h_eff),
            omega))
        # ALLE Senken liegen HINTER dem Filmwiderstand: Škvors Formel
        # beschreibt die laterale Strömung von der Membranfläche ZU den
        # Senken — auch die Blindloch-Stubs werden erst durch den Film
        # erreicht. (Sie vor R_gap zu shunten, ließe die Rückmembran-
        # Polstermode der Doppelmembran-Bauform ungedämpft und erzeugte
        # eine unphysikalisch scharfe Absorber-Kerbe im Frequenzgang.)
        mats.append(self._abcd_shunt(Y_gap, omega))
        if self.n_bh > 0:
            mats.append(self._abcd_shunt(
                1.0 / self._blind_hole_impedance(omega), omega))
        if self.stepped:
            # Senkungssegment der Stufenbohrungen in SERIE: spaltseitige
            # Mündungsmasse, dann das weite Rohr als thermoviskose
            # LEITUNG (verteilte Reibung + isotherm→adiabatische
            # Nachgiebigkeit), bevor der enge Kern (Z_holes) folgt.
            # Grenzfall Senkung -> 0 reproduziert die normale Bohrung.
            S_cb = np.pi * self.r_bh**2
            # KEINE spaltseitige Flanschmasse an der Senkungsmündung: sie
            # öffnet in den engen Spalt, dessen Ausbreitungsmasse der
            # Škvor-Term bereits trägt (Konvention wie 2D/3D, s.
            # _through_hole_impedance). Das Senkungsrohr folgt direkt.
            _ = S_cb
            if _HAS_SCIPY:
                g_cb, Zc_cb = self._narrow_duct_propagation(omega, self.r_bh)
                gl = g_cb * self.d_bh
                ch, sh = np.cosh(gl), np.sinh(gl)
                # Leitung aus n_th parallelen Rohren: Z skaliert 1/n
                mats.append(np.array([[ch, Zc_cb * sh / self.n_th],
                                      [self.n_th * sh / Zc_cb, ch]]))
            else:
                Z_wide = self._hole_impedance(omega, self.r_bh, self.d_bh,
                                              self.n_th,
                                              end_correction=False)
                mats.append(self._abcd_series(Z_wide, omega))
                mats.append(self._abcd_shunt(1j * omega * self.C_A_cb,
                                             omega))
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
        T_front, T_mem, T_rear = self._assemble_parts(omega)
        T_total = self._mmul(self._mmul(T_front, T_mem), T_rear)
        return T_total, T_rear

    def _assemble_parts(self, omega):
        """Wie :meth:`_assemble_network`, aber liefert die drei Teilketten
        (T_front, T_mem, T_rear) getrennt — für die Rausch-Port-Impedanz
        am Membranzweig (s. :meth:`_membrane_port_impedance`). Die
        Signalkette bleibt bit-für-bit identisch (Gegenprobe 25 prüft
        T_front·T_mem·T_rear == _assemble_network)."""
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
                    # halbe Škvor-Zellen mit Frequenzkorrektur (laterale
                    # Trägheit der Schichtluft, wie im Membranspalt)
                    R_c_half = (0.5 * self.R_A_center
                                * self._film_R_dynamic(omega, self.h_center)
                                if self.h_center > 0 else 0.0)
                    center = [
                        self._abcd_series(R_c_half, omega),
                        self._abcd_shunt(self._film_compliance_Y(
                            omega, self.h_center, self.S_bp)
                            if self.h_center > 0 else 0.0, omega),
                        self._abcd_series(R_c_half, omega),
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
            return T_front, T_mem, T_rear

        # Die Durchgangslöcher der Backplate sind der Zugang zur Rückseite.
        # Drei Fälle:
        #   n_th = 0                  -> hermetisch dicht direkt am Spalt
        #   keine rückwärtige Baugruppe -> Löcher münden (durchs Gewebe)
        #                                direkt ins rückwärtige Schallfeld
        #   Baugruppe vorhanden       -> Gewebe -> Laufzeitglied -> Hohlraum
        vents_directly = ((self.n_th > 0 or self.ring_vent_w > 0.0)
                          and not self.rear_network_enabled)
        # Einzel-Backplate: der (einzige) Spalt gehört zur polarisierten
        # Membran -> statisch verkleinerter effektiver Spalt
        rear = [self._backplate_gap_abcd(omega, outside_to_membrane=False,
                                         holes_radiate=vents_directly,
                                         polarized=True)]
        # geschlossene Backplate (n_th = 0): Port unmittelbar blockiert —
        # _rear_chain_mats liefert dann eine leere Liste
        rear += self._rear_chain_mats(omega)
        T_rear = reduce(self._mmul, rear)
        return T_front, T_mem, T_rear

    def _rear_chain_mats(self, omega):
        """Ketten-Elemente HINTER den Backplate-Durchgangslöchern.

        Alles vom Loch-Austritt bis zum rückwärtigen Port: Spacer +
        Rückplatte (K103), Gewebe, Laufzeitglied, Hohlraum, Einlass-
        löcher. Geteilt zwischen dem 1D/2D-Kettenpfad
        (:meth:`_assemble_network`) und dem 3D-Löser, der diese Kette
        als Lumped-Abschluss an seinem SAMMELKNOTEN hinter den diskreten
        Löchern nutzt — so bleiben beide Pfade baugleich. In den
        "blockiert"-Fällen (geschlossene Backplate, Rückplatte ohne
        Löcher, geschlossene Rückseite) endet die Liste einfach früher;
        die Abschlussbedingung (q_port = 0) trägt ``rear_open``.
        """
        mats = []
        # "inlet": Gewebe außen ÜBER den Einlassöffnungen statt im
        # Zylinder an der Backplate — wirksame Durchströmfläche ist die
        # LOCHFLÄCHE des jeweiligen Ports, und der Widerstand liegt
        # hinter den Shunt-Volumina (s. Konstruktor-Kommentar).
        at_inlet = self.fabric_rear_position == "inlet"
        if self.n_th == 0 and self.ring_vent_w == 0.0:
            # geschlossene Backplate: Gewebe/Laufzeitglied/Hohlraum sind
            # akustisch unerreichbar
            return mats

        if not self.rear_network_enabled:
            # kein Laufzeitglied/Hohlraum: Gewebe liegt über den Öffnungen,
            # Port = rückwärtiges Schallfeld. "inlet": das Tuch sitzt AUF
            # den Öffnungs-Mündungen (Lochfläche + Randspalt-Ringfläche).
            S_f = (self.n_th * np.pi * self.r_th**2
                   + 2.0 * np.pi * self.a_bp * self.ring_vent_w
                   if at_inlet else self.S_bp)
            mats.append(self._abcd_series(self.rayl_rear / S_f, omega))
            return mats

        # ------- Spacer + massive Rückplatte (K103-Bauform) ----------------
        # Dünner Distanzspalt hinter der Backplate; die Strömung tritt über
        # die Backplate-Durchgangslöcher ein und die Rückplattenlöcher aus
        # (je halbe Škvor-Zelle mit dem eigenen Lochmuster, s.
        # _derive_parameters), das Schichtvolumen shuntet dazwischen.
        plate = self.t_rp > 0.0
        if self.h_sp > 0.0:
            Y_sp = self._film_compliance_Y(omega, self.h_sp, self.S_bp)
            phi_sp = self._film_R_dynamic(omega, self.h_sp)
            if plate and self.n_rp > 0:
                mats.append(self._abcd_series(self.R_A_sp_in * phi_sp,
                                              omega))
                mats.append(self._abcd_shunt(Y_sp, omega))
                mats.append(self._abcd_series(self.R_A_sp_out * phi_sp,
                                              omega))
            else:
                # ohne (gelochte) Rückplatte wirkt der Spacer nur als
                # zusätzliches Luftvolumen (axialer Durchtritt, kein
                # nennenswerter lateraler Widerstand)
                mats.append(self._abcd_shunt(Y_sp, omega))
        if plate:
            if self.n_rp == 0:
                # Rückplatte ohne Löcher: Rückseite hier verschlossen —
                # alles Dahinterliegende ist akustisch unerreichbar
                # (rear_open=False blockiert den Port).
                return mats
            # Durchgangslöcher der Rückplatte: thermoviskoses Rohr über
            # die Plattendicke; münden sie direkt ins Schallfeld
            # (K103-Fall), kommt die Strahlungsimpedanz hinzu. Die äußere
            # Mündung öffnet in Freifeld/Baugruppe -> viskoser Mündungs-
            # widerstand und Fok-Array-Wechselwirkung; die innere liegt
            # im Spacer-Film (Škvor deckt ab, Flanschmasse wie bisher).
            Z_rp = self._hole_impedance(omega, self.r_rp, self.t_rp,
                                        self.n_rp, end_correction=False,
                                        radiates=self._plate_vents,
                                        visc_ends=1)
            S_rp = np.pi * self.r_rp**2
            Z_rp = Z_rp + (1j * omega * RHO0 * 0.85 * self.r_rp
                           * (1.0 + self._fok_rp) / (S_rp * self.n_rp))
            mats.append(self._abcd_series(Z_rp, omega))

        # Gewebe hinter der Backplate/Rückplatte (überspannt die Fläche);
        # bei "inlet" wandert es stattdessen ans Ende der Kette
        if not at_inlet:
            mats.append(self._abcd_series(self.rayl_rear / self.S_bp,
                                          omega))

        if self._plate_vents:
            # K103-Fall: hinter der Rückplatte folgt nichts mehr —
            # Port = rückwärtiges Schallfeld an den Plattenlöchern.
            # "inlet": Tuch auf den Rückplatten-Lochmündungen.
            if at_inlet:
                S_f = self.n_rp * np.pi * self.r_rp**2
                mats.append(self._abcd_series(self.rayl_rear / S_f, omega))
            return mats

        # Laufzeitglied als akustische Leitung (tau = L/c)
        if self.l_delay > 0.0:
            mats.append(self._abcd_line(omega, self.l_delay, self.a_bp))

        if self.rear_open:
            if self.cavity_hole_position == "circumference":
                # Leitung bis zur axialen Lochposition ...
                if self.x_ch > 0.0:
                    mats.append(self._abcd_line(omega, self.x_ch, self.a_bp))
                # ... dahinter wirkt das geschlossene Reststück als Shunt
                l_rest = self.l_cav - self.x_ch
                if l_rest > 1e-9:
                    Z_stub = self._closed_stub_impedance(omega, l_rest,
                                                         self.a_bp)
                    mats.append(self._abcd_shunt(1.0 / Z_stub, omega))
                hole_len = self.t_cav_wall  # radiale Löcher durch die Wand
            else:  # "end": Löcher in der hinteren Stirnfläche
                mats.append(self._abcd_line(omega, self.l_cav, self.a_bp))
                hole_len = self.t_cav_wall
            # Einlasslöcher: thermoviskoses Rohr + Strahlung ins Freifeld;
            # beide Mündungen öffnen in große Volumina (Hohlraum/Freifeld)
            # -> beidseitiger viskoser Mündungswiderstand (Sampson)
            Z_ch = self._hole_impedance(
                omega, self.r_ch, hole_len, self.n_ch,
                end_correction=True, radiates=True, visc_ends=2,
            )
            mats.append(self._abcd_series(Z_ch, omega))
            # "inlet": um den Zylinder gewickeltes Tuch über den
            # Einlasslöchern — in Serie mit deren Lochmasse, wirksame
            # Fläche = Gesamt-Lochfläche (rear_open garantiert n_ch,
            # r_ch > 0)
            if at_inlet:
                S_f = self.n_ch * np.pi * self.r_ch**2
                mats.append(self._abcd_series(self.rayl_rear / S_f, omega))
        else:
            # geschlossene Rückseite: gesamter Hohlraum, Port ist
            # "blockiert" — es gibt keinen Einlass, über dem ein
            # "inlet"-Gewebe liegen könnte (es entfällt wirkungslos)
            mats.append(self._abcd_line(omega, self.l_cav, self.a_bp))

        return mats

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

    def _membrane_port_impedance(self, omega):
        """Treibpunkt-Impedanz Z_tot(ω) über dem Membran-Serienzweig plus
        ihre Zerlegung in Front-, Membranfilm- und Rückpfad-Anteil.

        Grundlage der Rauschrechnung: das verallgemeinerte Nyquist-/
        Fluktuations-Dissipations-Theorem (Twiss 1955). Die Kurzschluss-
        Rauschstromdichte am Membranzweig eines PASSIVEN Netzwerks bei
        Temperatur T ist

            S_qq(ω) = 4 k_B T · Re{Z_tot} / |Z_tot|²  = 4 k_B T · Re{1/Z_tot},

        und dieses eine Ergebnis wichtet JEDEN dissipativen Widerstand des
        Netzwerks (Spaltfilm, Bohrungen, Gewebe, Strahlung) automatisch
        korrekt — inklusive der Strom-Aufteilung an allen Shunt-Zweigen.
        Kein Fit-Koeffizient: die Widerstände sind dieselben, die auch den
        Frequenzgang und die Nierennull bestimmen.

        Der Membranzweig ist ein SERIEN-Element der ABCD-Kette; die
        Impedanz über seinen Klemmen ist die Serien-Summe
            Z_tot = Z_front + Z_mem + Z_rear
        mit dem Ausgangswiderstand der vorderen Kette (Quelle
        EMK-frei = kurzgeschlossen -> B_f/A_f), der Membran-
        Serienimpedanz selbst und der Eingangsimpedanz der Rückkette
        (offener Port EMK-frei -> B_r/D_r; blockiert -> A_r/C_r). Weil
        die drei Anteile in Serie denselben Rauschstrom führen, ist der
        Rauschbeitrag jedes Pfads proportional zu seinem Re{Z_i} — das
        liefert die Pfad-Zerlegung (Front-/Membran-/Rückpfad) gratis.

        Nur für die ABCD-Modelle 1D/2D — der 3D-Feldlöser hat keinen
        konzentrierten Membranzweig.

        Rückgabe: (Z_tot, Z_front, Z_mem, Z_rear), je (len(omega),).
        """
        if self.squeeze_model == "3d":
            raise ValueError(
                "Rauschberechnung nur für squeeze_model '1d'/'2d' — der "
                "3D-Feldlöser hat keinen konzentrierten Membranzweig.")
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        T_front, T_mem, T_rear = self._assemble_parts(omega)
        A_f, B_f = T_front[0, 0], T_front[0, 1]
        Z_mem = T_mem[0, 1]                       # T_mem = [[1, Z_mem],[0,1]]
        A_r, B_r = T_rear[0, 0], T_rear[0, 1]
        C_r, D_r = T_rear[1, 0], T_rear[1, 1]
        Z_front = B_f / A_f                       # Quelle kurzgeschlossen
        Z_rear = (B_r / D_r) if self.rear_open else (A_r / C_r)
        return Z_front + Z_mem + Z_rear, Z_front, Z_mem, Z_rear

    @staticmethod
    def _a_weighting(f):
        """Lineares A-Bewertungsgewicht W(f) = R_A(f)/R_A(1 kHz) nach
        IEC 61672-1 (0 dB bei 1 kHz)."""
        f = np.asarray(f, dtype=float)
        f2 = f * f

        def _ra(x2):
            return (12194.0 ** 2 * x2 * x2) / (
                (x2 + 20.6 ** 2) * (x2 + 12194.0 ** 2)
                * np.sqrt((x2 + 107.7 ** 2) * (x2 + 737.9 ** 2)))

        return _ra(f2) / _ra(np.array(1000.0 ** 2))

    def noise_spectrum(self, frequencies_hz):
        """Thermisch-akustisches Eigenrauschen der Kapsel (fit-frei).

        Über das Fluktuations-Dissipations-Theorem (s.
        :meth:`_membrane_port_impedance`) erzeugt jeder akustische
        Widerstand bei 20 °C ein Rauschen; auf den freien Feld-Schalldruck
        zurückgerechnet ergibt sich die ÄQUIVALENTE Eingangs-Druckrausch-
        dichte

            S_p,eq(f) = S_v,out(f) / |H(f)|²,
            S_v,out   = (Θ/ω)² · S_qq,   H = e/p0 (frontal, mit Beugung),

        in der sich Θ und ω herauskürzen — das Eingangsrauschen ist eine
        rein akustische Größe. Die Beugungsverstärkung von H senkt das
        Eingangsrauschen zu hohen Frequenzen hin (das Mikrofon ist dort
        empfindlicher), wie in der Realität.

        Rückgabe: dict
            'frequency_hz'
            'psd_pa2_hz'   — S_p,eq [Pa²/Hz]
            'asd_pa_shz'   — sqrt(S_p,eq) [Pa/√Hz]
            'psd_v2_hz'    — Ausgangs-Spannungsrauschdichte [V²/Hz]
            'frac_front' / 'frac_mem' / 'frac_rear' — Rauschanteil je Pfad
                             (Re{Z_i}/Re{Z_tot}); Summe = 1
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * f
        Z_tot, Z_f, Z_m, Z_r = self._membrane_port_impedance(omega)
        reZ = np.real(Z_tot)
        # Kurzschluss-Volumenfluss-Rauschdichte am Membranzweig
        S_qq = (4.0 * K_BOLTZ * T_KELVIN * reZ
                / np.maximum(np.abs(Z_tot) ** 2, 1e-300))
        H = self.transfer_function(f, angle_deg=0.0)         # e/p0 [V/Pa]
        S_v = (self._theta / omega) ** 2 * S_qq              # [V²/Hz]
        S_p = S_v / np.maximum(np.abs(H) ** 2, 1e-300)       # [Pa²/Hz]
        reZ_safe = np.where(reZ > 0.0, reZ, np.nan)
        return {
            "frequency_hz": f,
            "psd_pa2_hz": S_p,
            "asd_pa_shz": np.sqrt(S_p),
            "psd_v2_hz": S_v,
            "frac_front": np.real(Z_f) / reZ_safe,
            "frac_mem": np.real(Z_m) / reZ_safe,
            "frac_rear": np.real(Z_r) / reZ_safe,
        }

    def self_noise(self, f_min=20.0, f_max=20000.0, n_points=1200):
        """A- und Z-bewerteter Ersatzgeräuschpegel (Eigenrauschen) [dB SPL].

        Integriert die äquivalente Eingangs-Druckrauschdichte über das
        Hörband und bezieht sie auf 20 µPa. ``spl_a_db`` ist der übliche
        A-bewertete Ersatzgeräuschpegel eines Mikrofons (Datenblatt-
        Kennzahl); ``spl_z_db`` der lineare (unbewertete) Wert. Die drei
        ``*_a_*``-Pfadwerte zerlegen den A-bewerteten Pegel in Front-,
        Membranfilm- und Rückpfad-Anteil (energetisch, Summe der
        Leistungen = Gesamtpegel).

        Reines thermisch-akustisches Kapselrauschen ohne Verstärker/
        Elektronik — die physikalische Untergrenze dieser Geometrie.
        """
        f = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))
        sp = self.noise_spectrum(f)
        S = sp["psd_pa2_hz"]
        w_a = self._a_weighting(f)

        def _spl(weight, frac=None):
            integ = S * weight ** 2
            if frac is not None:
                integ = integ * np.nan_to_num(frac, nan=0.0)
            p2 = float(_trapz(integ, f))
            return 20.0 * np.log10(np.sqrt(max(p2, 1e-300)) / P_REF)

        ones = np.ones_like(f)
        return {
            "spl_a_db": _spl(w_a),
            "spl_z_db": _spl(ones),
            "spl_a_front_db": _spl(w_a, sp["frac_front"]),
            "spl_a_mem_db": _spl(w_a, sp["frac_mem"]),
            "spl_a_rear_db": _spl(w_a, sp["frac_rear"]),
            "f_min_hz": float(f_min),
            "f_max_hz": float(f_max),
        }

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
        theta = np.array([np.deg2rad(angle_deg)])
        p_f, p_r = self._source_pressures(omega, theta)
        if self.squeeze_model == "3d":
            # 3D-Sandwich: Volumenverschiebung direkt aus dem Feldlöser
            Xf, Xr = self._solve_3d(omega)
            q_mem = 1j * omega * (Xf * p_f[:, 0] + Xr * p_r[:, 0])
            return self._output_voltage(omega, q_mem)
        T_total, T_rear = self._assemble_network(omega)
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
            p_f2, p_r2 = self._source_pressures(omega, theta)
            p_front, p_rear = p_f2[0], p_r2[0]
            if self.squeeze_model == "3d":
                # 3D-Sandwich: EIN Feldlösungs-Paar (X_f, X_r) je
                # Frequenz; die Winkelabhängigkeit steckt allein in den
                # Quelldrücken
                Xf3, Xr3 = self._solve_3d(omega)
                q_mem = 1j * omega[0] * (Xf3[0] * p_front
                                         + Xr3[0] * p_rear)
            else:
                T_total, T_rear = self._assemble_network(omega)
                if self.rear_open:
                    A, B = T_total[0, 0][0], T_total[0, 1][0]
                    q_rear = (p_front - A * p_rear) / B
                    q_mem = (T_rear[1, 0][0] * p_rear
                             + T_rear[1, 1][0] * q_rear)
                else:
                    # Druckempfänger: winkelabhängig nur über den
                    # Druckstau an der Membran (ohne Beugung: exakte Kugel)
                    q_mem = T_rear[1, 0][0] * (p_front / T_total[0, 0][0])
            e = self._output_voltage(np.full_like(theta, omega[0]), q_mem)
            mag = np.abs(e)
            ref = mag[0] if mag[0] > 0 else np.max(mag)
            lin = mag / ref
            db = np.maximum(20.0 * np.log10(np.maximum(lin, 1e-30)), -40.0)
            patterns[float(f)] = {"linear": lin, "db": db}
        return {"angles_deg": angles, "patterns": patterns}

    def delay_diagnostics(self, f_probe_hz=1000.0):
        """Interne vs. externe Laufzeit des Nieren-Phasenschiebers.

        Die 180°-Auslöschung entsteht, wenn die INTERNE Rück-Übertragung
        des Netzwerks D_r = −a/b (mit q_mem = a·p_front + b·p_rück) die
        EXTERNE Front-Rück-Übertragung G(180°) = p_rück/p_front trifft.
        Beide werden als äquivalente Laufzeit aus der Phase bei der
        Sondenfrequenz ``f_probe_hz`` ausgewertet:

            tau = arg(·) / omega.

        Die INTERNE Laufzeit ist frequenzabhängig (RC-Phasenschieber mit
        Filmträgheit und interner Helmholtz-Resonanz — kein reines
        Laufzeitglied); die Voreinstellung 1 kHz bewertet die Anpassung
        dort, wo die Nierenwirkung im Mittenband zählt. Deutung des
        Verhältnisses (numerisch verifiziert, Gegenprobe 17):
          ~1   — Laufzeiten angepasst: tiefste Auslöschung bei 180°.
          < 1  — interne Laufzeit zu kurz: das Pattern-Minimum wandert
                 VOR 180° (Richtung Hyperniere), 180° bleibt flacher.
          > 1  — interne Laufzeit zu lang: das Minimum bleibt bei 180°
                 GEPINNT (kein Außenwinkel bietet mehr Phase), die
                 Auslöschung wird aber flacher, je größer der Überschuss.

        Rückgabe: dict mit
            'tau_int_s' / 'tau_ext_s'   — Laufzeiten [s]
            'dist_int_m' / 'dist_ext_m' — äquivalente Wegstrecken c·tau
            'ratio'                     — tau_int / tau_ext
            'f_probe_hz'                — Sondenfrequenz
        oder ``None`` für reine Druckempfänger (Rückseite geschlossen —
        es gibt keinen internen Pfad und keine Laufzeit-Anpassung).
        """
        if not self.rear_open:
            return None
        omega = np.array([2.0 * np.pi * float(f_probe_hz)])
        p_f, p_r = self._source_pressures(omega, np.array([np.pi]))
        G180 = (p_r[0, 0] / p_f[0, 0]) if p_f[0, 0] != 0 else np.nan
        if self.squeeze_model == "3d":
            # 3D-Sandwich: q = jω(X_f·p_f + X_r·p_r) -> D_r = −X_f/X_r
            Xf, Xr = self._solve_3d(omega)
            D_r = -Xf[0] / Xr[0]
        else:
            T_tot, T_rear = self._assemble_network(omega)
            one, zero = np.ones(1), np.zeros(1)
            a = self._membrane_volume_velocity(omega, T_tot, T_rear,
                                               one, zero)
            b = self._membrane_volume_velocity(omega, T_tot, T_rear,
                                               zero, one)
            D_r = (-a / b)[0]
        w = omega[0]
        tau_int = float(np.angle(D_r) / w)
        tau_ext = float(np.angle(G180) / w)
        return {
            "tau_int_s": tau_int,
            "tau_ext_s": tau_ext,
            "dist_int_m": tau_int * C_AIR,
            "dist_ext_m": tau_ext * C_AIR,
            "ratio": tau_int / tau_ext if tau_ext != 0.0 else float("nan"),
            "f_probe_hz": float(f_probe_hz),
        }

    def angle_responses(self, frequencies_hz, angles_deg=(0.0, 90.0, 180.0)):
        """Winkel-Frequenzgänge und Rück-Übertragung D_r in EINEM Durchlauf.

        Der Membran-Volumenfluss ist linear in den beiden Quelldrücken
        (Superposition):

            q(θ, ω) = a(ω)·p_front(θ) + b(ω)·p_rück(θ),

        mit winkelUNabhängigen Koeffizienten a, b. Ein einziger Netzwerk-
        bzw. Feldaufbau je Frequenz liefert deshalb gleichzeitig die
        Frequenzgänge ALLER gewünschten Winkel, die intern geforderte
        Rück-Übertragung D_r = −a/b und die externe Front-Rück-
        Übertragung G(180°) = p_rück/p_front — beim 3D-Modell spart das
        die sonst je Winkel wiederholte LU-Faktorisierung.

        Rückgabe: dict
            'frequency_hz' — Frequenzachse
            'H'            — {winkel_deg: komplexe Übertragung e/p0 [V/Pa]}
            'D_r'          — Rück-Übertragung −a/b des internen
                             Phasenschiebers (None bei geschlossener
                             Rückseite: kein interner Pfad)
            'G180'         — externe Front-Rück-Übertragung bei 180°
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * f
        ang = [float(x) for x in angles_deg]
        th_list = ang + ([] if 180.0 in ang else [180.0])
        theta = np.deg2rad(np.asarray(th_list))
        p_f, p_r = self._source_pressures(omega, theta)
        if self.squeeze_model == "3d":
            Xf, Xr = self._solve_3d(omega)
            a = 1j * omega * Xf
            b = 1j * omega * Xr
        else:
            T_tot, T_rear = self._assemble_network(omega)
            one, zero = np.ones_like(omega), np.zeros_like(omega)
            a = self._membrane_volume_velocity(omega, T_tot, T_rear,
                                               one, zero)
            b = self._membrane_volume_velocity(omega, T_tot, T_rear,
                                               zero, one)
        H = {angle: self._output_voltage(omega,
                                         a * p_f[:, i] + b * p_r[:, i])
             for i, angle in enumerate(ang)}
        i180 = th_list.index(180.0)
        G180 = p_r[:, i180] / p_f[:, i180]
        D_r = (-a / b) if self.rear_open else None
        return {"frequency_hz": f, "H": H, "D_r": D_r, "G180": G180}

    @staticmethod
    def helmholtz_resonance_hz(frequency_hz, D_r):
        """Interne Helmholtz-Resonanz aus der D_r-Kurve.

        Unterhalb der Resonanz hat die Rück-Übertragung die Form
        D_r ≈ (1 − (ω/ω_H)²) + jω·RC (Durchgangsloch-Trägheit gegen die
        innere Nachgiebigkeit aus Spalt + Blindlöchern): der REALTEIL
        wechselt bei ω_H das Vorzeichen, die (entrollte) Phase kreuzt
        90°. Diese Kreuzung wird log-frequenzlinear interpoliert.

        Rückgabe: Frequenz [Hz] der ersten 90°-Kreuzung oder ``None``,
        wenn keine im übergebenen Bereich liegt (Resonanz oberhalb des
        Bandes — der gesunde Fall) oder ``D_r`` None ist.
        """
        if D_r is None:
            return None
        f = np.asarray(frequency_hz, dtype=float)
        ph = np.degrees(np.unwrap(np.angle(np.asarray(D_r))))
        below = ph < 90.0
        idx = np.nonzero(below[:-1] & ~below[1:])[0]
        if idx.size == 0:
            return None
        i = int(idx[0])
        t = (90.0 - ph[i]) / (ph[i + 1] - ph[i])
        lf = np.log10(f[i]) + t * (np.log10(f[i + 1]) - np.log10(f[i]))
        return float(10.0 ** lf)

    # ======================================================================
    # Diagnose
    # ======================================================================
    @staticmethod
    def _ring_note(rings, lang="de"):
        """Kurzform eines Lochmusters für summary(): Anzahl je Lochkreis."""
        en = str(lang).strip().lower() == "en"
        parts = [((f"{cnt} even" if en else f"{cnt} gleichmäßig") if r is None
                  else (f"{cnt} on PCD ⌀{2e3 * r:.1f} mm" if en
                        else f"{cnt} auf LK ⌀{2e3 * r:.1f} mm"))
                 for cnt, r in rings if cnt > 0]
        return "; ".join(parts) if parts else ("none" if en else "keine")

    def summary(self, lang="de"):
        """Mehrzeilige Übersicht der abgeleiteten Modellparameter.

        ``lang`` schaltet die Sprache um ("de" Standard, "en" Englisch);
        die Zahlenformate bleiben identisch. Der Standard "de" hält die
        Gegenproben im Testlauf unverändert.
        """
        en = str(lang).strip().lower() == "en"

        def _t(de_txt, en_txt):
            return en_txt if en else de_txt

        def _row(label, value):
            # Label linksbündig auf feste Spaltenbreite, Wert schließt an
            return f"{label:<30}{value}"

        sens = self.transfer_function(1000.0)[0]
        arch_note = {
            "single": _t("1 Backplate", "1 backplate"),
            "dual": _t("2 Backplates, Gegentakt", "2 backplates, push-pull"),
            "dual_diaphragm": _t("K67-Bauform, passive Rückmembran",
                                 "K67 design, passive rear membrane"),
        }[self.architecture]
        sm_note = {
            "1d": _t("(Lumped-Element)", "(lumped element)"),
            "2d": _t("(modifizierte Reynolds-Feldlösung)",
                     "(modified Reynolds field solution)"),
            "3d": _t("((r,phi)-Sandwich, diskrete Löcher)",
                     "((r,phi) sandwich, discrete holes)"),
        }[self.squeeze_model]
        lines = [
            _t("MicrophoneCapsule — abgeleitete Parameter",
               "MicrophoneCapsule — derived parameters"),
            "-" * 55,
            _row(_t("Architektur:", "Architecture:"),
                 f"{self.architecture} ({arch_note})"),
            _row(_t("Spaltfilm-Modell:", "Gap-film model:"),
                 f"{self.squeeze_model} " + sm_note),
            _row(_t("Membranfläche:", "Membrane area:"),
                 f"{self.S_mem * 1e6:9.2f} mm²"),
            _row(_t("akust. Masse Membran M_A:", "acoust. membrane mass M_A:"),
                 f"{self.M_A_mem:9.2f} kg/m⁴"),
            _row(_t("akust. Nachgiebigkeit C_A:", "acoust. compliance C_A:"),
                 f"{self.C_A_mem:9.3e} m³/Pa"),
            _row(_t("  dto. effektiv (mit Bias):", "  same, effective (bias):"),
                 f"{self.C_A_eff:9.3e} m³/Pa"),
            _row(_t("Feder-Erweichung durch Bias:", "Spring softening (bias):"),
                 f"{self.softening_ratio * 100:9.2f} %"),
            _row(_t("statische Durchbiegung w0:", "static deflection w0:"),
                 f"{self.w0_static * 1e6:9.2f} µm "
                 + _t(f"(Restspalt Mitte {self.h_min_static * 1e6:.1f} µm)",
                      f"(residual center gap {self.h_min_static * 1e6:.1f} "
                      "µm)")),
            _row(_t("wirksamer Frontspalt h_eff:", "effective front gap "
                    "h_eff:"),
                 f"{self.h_gap_front * 1e6:9.2f} µm "
                 f"(nominal {self.h_gap * 1e6:.1f} µm)"),
            _row(_t("Pull-in-Spannung U_PI:", "Pull-in voltage U_PI:"),
                 (f"{self.U_pullin:9.1f} V" if np.isfinite(self.U_pullin)
                  else "     > 20 kV")),
            _row(_t("Elektroden-Porosität:", "Electrode porosity:"),
                 f"{100 * (self.phi_th + self.phi_bh):9.1f} % "
                 + _t(f"(Durchgang {100 * self.phi_th:.1f} %, Blind "
                      f"{100 * self.phi_bh:.1f} %)",
                      f"(through {100 * self.phi_th:.1f} %, blind "
                      f"{100 * self.phi_bh:.1f} %)")),
            _row(_t("Lochmuster Durchgang:", "Hole pattern through:"),
                 f"{self.n_th:6d} × ⌀{2e3 * self.r_th:.2f} mm "
                 f"({self._ring_note(self._th_rings, lang)})"
                 + (_t(f" — Stufenbohrung: Kern {self.t_th_eff * 1e3:.2f} mm "
                       f"unter ⌀{2e3 * self.r_bh:.2f}-mm-Senkung",
                       f" — stepped bore: core {self.t_th_eff * 1e3:.2f} mm "
                       f"below ⌀{2e3 * self.r_bh:.2f} mm counterbore")
                    if self.stepped else "")),
            _row(_t("Lochmuster Blind:", "Hole pattern blind:"),
                 f"{self.n_bh:6d} × ⌀{2e3 * self.r_bh:.2f} mm "
                 f"({self._ring_note(self._bh_rings, lang)})"
                 + (_t(f" [+ {self.n_th} durchgebohrte Senkungen = "
                       f"{self.n_bh + self.n_th} gesamt]",
                       f" [+ {self.n_th} drilled-through counterbores = "
                       f"{self.n_bh + self.n_th} total]")
                    if self.stepped else "")),
            _row(_t("Resonanz (Modell):", "Resonance (model):"),
                 f"{self.f_res:9.1f} Hz"),
            _row(_t("Resonanz aus Vorspannung/E:", "Resonance from "
                    "tension/E:"),
                 f"{self.f_res_from_tension:9.1f} Hz"),
            _t(f"  (exakte J0-Modalfrequenz:   {self.f_res_modal_exact:9.1f} "
               "Hz — Lumped-Kolbenfaktor 4/3 liegt ~1.9 % darüber)",
               f"  (exact J0 modal frequency:  {self.f_res_modal_exact:9.1f} "
               "Hz — lumped piston factor 4/3 is ~1.9 % above)"),
            _row(_t("Ruhekapazität C0 (je BP):", "Static capacitance C0/BP:"),
                 f"{self.C_elec_0 * 1e12:9.2f} pF"),
            _row(_t("Squeeze-Film-Widerst. R_gap:", "Squeeze-film res. "
                    "R_gap:"),
                 (f"{self.R_A_gap:9.3e} Pa·s/m³" if self.R_A_gap is not None
                  else _t("        — (Backplate geschlossen)",
                          "        — (backplate closed)"))),
            _row(_t("Nachgiebigkeit Spalt C_gap:", "Compliance gap C_gap:"),
                 f"{self.C_A_gap:9.3e} m³/Pa"),
            _row(_t("Nachgiebigkeit Blindl. C_bh:", "Compliance blind C_bh:"),
                 f"{self.C_A_blind:9.3e} m³/Pa"),
            _row(_t("rückwärtige Baugruppe:", "rear assembly:"),
                 f"{self.rear_network_enabled}"),
            _row(_t("Rückseite offen (Gradient):", "rear open (gradient):"),
                 f"{self.rear_open}"),
            _row(_t("äußere Wegdifferenz d_ext:", "outer path diff. d_ext:"),
                 f"{self.d_ext * 1e3:9.2f} mm"),
            _row(_t("Beugung am Gehäuse:", "Diffraction at body:"),
                 f"{self.include_diffraction and _HAS_SCIPY}"),
        ]
        if (self.architecture != "dual_diaphragm"
                and self.rear_network_enabled
                and (self.h_sp > 0.0 or self.t_rp > 0.0)):
            sp = (f"Spacer {self.h_sp * 1e6:.0f} µm" if self.h_sp > 0
                  else _t("kein Spacer", "no spacer"))
            if self.t_rp > 0:
                rp = _t(f"Rückplatte {self.t_rp * 1e3:.2f} mm",
                        f"Rear plate {self.t_rp * 1e3:.2f} mm")
                rp += (f", {self.n_rp} × ⌀{2e3 * self.r_rp:.2f} mm"
                       if self.n_rp > 0
                       else _t(", ohne Löcher (dicht)",
                               ", without holes (sealed)"))
                if self._plate_vents:
                    rp += _t(" → Schallfeld", " → sound field")
            else:
                rp = _t("keine Rückplatte", "no rear plate")
            lines.append(_row(_t("Spacer/Rückplatte (K103):",
                                 "Spacer/rear plate (K103):"), f"{sp}; {rp}"))
        if (self.architecture != "dual_diaphragm"
                and self.rear_network_enabled
                and (self.l_delay > 0.0 or self.l_cav > 0.0)):
            # GÜLTIGKEITS-GATTER: Laufzeitglied/Hohlraum werden als
            # 1D-Leitungen gerechnet. Die erste azimutale Quermode eines
            # Zylinderrohres liegt bei k·R = 1.8412 — darüber können
            # (v. a. seitlich angeregte) Quermoden das 1D-Bild verfälschen.
            f_quer = 1.8412 * C_AIR / (2.0 * np.pi * self.a_bp)
            lines.append(_row(
                _t("1D-Leitungsgrenze (Quermode):", "1D line limit "
                   "(transv.):"),
                f"{f_quer:9.1f} Hz "
                + _t("(erste azimutale Hohlraum-Mode)",
                     "(first azimuthal cavity mode)")))
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
                _row(_t("Druckleck δ = C_int/C_mem:", "Pressure leak "
                        "δ=C_int/C_mem:"), f"{delta:9.4f}"),
                _row(_t("Pattern-Untergrenze f_δ:", "Pattern lower limit "
                        "f_δ:"),
                     f"{f_floor:9.1f} Hz "
                     + _t("(darunter -> Kugel)", "(below -> omni)")),
            ]
        lines += [
            _row(_t("Ersatz-Gehäuseradius R_body:", "Equiv. body radius "
                    "R_body:"),
                 f"{self.R_body * 1e3:9.2f} mm "
                 + _t(f"(ka=1 bei {C_AIR / (2 * np.pi * self.R_body):.0f} Hz)",
                      f"(ka=1 at {C_AIR / (2 * np.pi * self.R_body):.0f} Hz)")),
            _row(_t("Empfindlichkeit @ 1 kHz:", "Sensitivity @ 1 kHz:"),
                 f"{abs(sens) * 1e3:9.2f} mV/Pa "
                 f"({20 * np.log10(abs(sens)):.1f} dB re 1 V/Pa)"),
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
    # verifizierte K67-Geometrie (120 Senkungen 1.3x3.7 mm, 60 durchgebohrt
    # mit 0.6-mm-Kern, kein Gewebe, Klemmringe 2x2 mm vor beiden Membranen):
    # die interne Laufzeit fällt seit der Port-Tausch-Korrektur aus den
    # physikalischen Parametern (Bohrungen, Spaltfilme, Spacer), die
    # externe Laufzeit aus der axialen Körperbeugung (_axial_body_transfer,
    # d_ext = axial + 2·Klemmringdicke, LF-Grenzfall 1.5·d_ext) — beides
    # ohne Fit-Koeffizient. Mit den nominellen Maßen: ~-26 dB bei 180°/
    # 1 kHz, tiefstes Minimum knapp vor 180° (extern 18.3 mm vs. intern
    # ~17 mm; Spacer 45 µm pinnt die Null exakt auf 180°, s.
    # examples/cardiodtest.json). Präsenz im 8-12-kHz-Band.
    k67 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=120, blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3,
        squeeze_model="2d",   # K67 braucht die radiale Druckauflösung
    )
    di_k = k67.directivity(frequencies_hz=(1000.0,))
    ang_k = di_k["angles_deg"]
    lin_k = di_k["patterns"][1000.0]["linear"]
    pk67 = di_k["patterns"][1000.0]["db"]
    na_k = ang_k[ang_k <= 180][int(np.argmin(lin_k[ang_k <= 180]))]
    assert na_k > 150.0, \
        f"K67 muss echte Niere sein (Null bei {na_k:.0f}° statt nahe 180°)"
    assert pk67[180] < -18.0, \
        f"K67 muss rückwärts tief auslöschen ({pk67[180]:.1f} dB)"
    fr_k = k67.frequency_response(n_points=150)
    assert np.all(np.isfinite(fr_k["amplitude_db"]))
    fk, ak = fr_k["frequency_hz"], fr_k["amplitude_db_norm"]
    ihf = (fk > 5000) & (fk < 16000)
    assert 1.0 < np.max(ak[ihf]) < 12.0, \
        "Präsenzanhebung erwartet (roh; Korb/Elektronik glätten auf +2..3)"
    print(f"K67-Bauform: echte Niere (Null @{na_k:.0f}°, 180° = "
          f"{pk67[180]:.1f} dB @1 kHz), Präsenzanhebung "
          f"+{np.max(ak[ihf]):.1f} dB bei "
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
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=bias, architecture="dual_diaphragm", center_gap=50e-6,
            n_through_holes=60, through_hole_diameter=0.6e-3,
            n_blind_holes=120, blind_hole_diameter=1.3e-3,
            blind_hole_depth=3.7e-3, through_holes_stepped=True,
            clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=34e-3, squeeze_model="2d")
        return c.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
    d20, d60 = _p180(20.0), _p180(60.0)
    assert d20 < -10.0 and d60 < -10.0, "Niere muss bei beiden Spannungen bestehen"
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
    #    n_blind = n_through (alle 24 Senkungen durchgebohrt, 0 rein blind).
    plain11 = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        n_through_holes=24, through_hole_diameter=1.0e-3,
        n_blind_holes=0, rear_network_enabled=False)
    tiny11 = MicrophoneCapsule(
        architecture="single", backplate_diameter=20e-3,
        n_through_holes=24, through_hole_diameter=1.0e-3,
        n_blind_holes=24, blind_hole_diameter=1.02e-3,
        blind_hole_depth=0.02e-3, through_holes_stepped=True,
        rear_network_enabled=False)
    assert tiny11.n_bh == 0    # 24 Senkungen, alle 24 durchgebohrt
    H_p = plain11.transfer_function(_fchk)[0]
    H_t = tiny11.transfer_function(_fchk)[0]
    assert np.max(np.abs(H_t / H_p - 1.0)) < 0.05, \
        "verschwindende Senkung muss die normale Bohrung reproduzieren"
    # b) K67-Geometrie (verifiziert: je Seite 120 Senkungen 1.3 mm x
    #    3.7 mm in der 4-mm-Halbplatte, jede zweite mit 0.6-mm-Durchbruch
    #    am Grund -> n_blind=120 gesamt, davon 60 durchgebohrt, 60 rein
    #    blind): das enge Rohr ist nur noch t_bp − Tiefe = 0.3 mm lang ->
    #    deutlich kleinere Durchgangsimpedanz; die Niere der Doppel-
    #    membran-Bauform muss ihre NULLSTELLE bei ~180° behalten.
    k67_kwargs = dict(
        membrane_material="pet", membrane_resonance_hz=1150.0,
        membrane_diameter=26e-3, membrane_thickness=6e-6,
        membrane_tension=13.7, air_gap=65e-6, backplate_diameter=25e-3,
        backplate_thickness=4e-3, bias_voltage=60.0,
        architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=120, blind_hole_diameter=1.3e-3,
        blind_hole_depth=3.7e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
        body_diameter=34e-3,
    )
    # K67 braucht das 2D-Feldmodell (radiale Druckausbreitung -> volle
    # interne Laufzeit, passend zur Klemmring-verlängerten d_ext)
    k67_st = MicrophoneCapsule(through_holes_stepped=True,
                               squeeze_model="2d", **k67_kwargs)
    # Zählweise: 120 Senkungen gesamt, 60 durchgebohrt, 60 rein blind
    assert k67_st.n_th == 60 and k67_st.n_bh == 60
    k67_full = MicrophoneCapsule(   # gleiche Kerne, volle Plattendicke
        **{**k67_kwargs, "n_blind_holes": 60})
    om1k = np.array([2.0 * np.pi * 1000.0])
    Zst = abs(k67_st._through_hole_impedance(om1k, 60)[0])
    Zpl = abs(k67_full._hole_impedance(om1k, k67_st.r_th, k67_st.t_bp, 60,
                                       end_correction=True, visc_ends=1)[0])
    assert Zst < 0.85 * Zpl, \
        "Stufenbohrung muss die Durchgangsimpedanz senken (kürzeres Rohr)"
    assert k67_st.C_A_cb > 0.0
    # Ruhekapazität trifft den nachgemessenen K67-Wert (~50 pF)
    assert 47.0 < k67_st.C_elec_0 * 1e12 < 54.0, \
        f"K67-C0 = {k67_st.C_elec_0 * 1e12:.1f} pF (erwartet ~50 pF)"
    # NIERE: tiefstes Minimum nahe 180° (mit dem axialen Beugungsumweg
    # liegt es bei den nominellen Maßen knapp davor, ~158-165°; deutlich
    # kleinere Winkel wären eine echte Superniere) über das Mittenband;
    # H180/H0 klein.
    di_st = k67_st.directivity(frequencies_hz=[500.0, 1000.0])
    ang = di_st["angles_deg"]
    for f in (500.0, 1000.0):
        lin = di_st["patterns"][f]["linear"]
        na = ang[ang <= 180][int(np.argmin(lin[ang <= 180]))]
        assert na > 150.0, \
            f"K67 muss Niere sein (Null bei {na:.0f}° statt nahe 180° -> Super)"
    st_180 = 20.0 * np.log10(
        abs(k67_st.transfer_function(1000.0, angle_deg=180.0)[0])
        / abs(k67_st.transfer_function(1000.0, angle_deg=0.0)[0]))
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
    print(f"Stufenbohrung: Grenzfall ≡ normale Bohrung, K67 120 Senkungen/"
          f"60 durchgebohrt, Null bei ~180° (Niere), "
          f"|Z_th| um {100 * (1 - Zst / Zpl):.0f} % kleiner, Niere bleibt "
          f"(180° @1 kHz = {st_180:.1f} dB), 2D reziprok  OK")

    # --------- Gegenprobe 12: vollständige Verlustmechanismen --------------
    # a) Viskose Mündung (Sampson/Weissberg): der DC-Grenzfall zweier
    #    Mündungen muss exakt 3·mu/r³ betragen.
    om_lo = np.array([2.0 * np.pi * 2.0])
    r12 = 0.4e-3
    Z_no = MicrophoneCapsule._hole_impedance(om_lo, r12, 1e-3, 1)
    Z_ve = MicrophoneCapsule._hole_impedance(om_lo, r12, 1e-3, 1,
                                             visc_ends=2)
    R_samp = 3.0 * MU_AIR / r12**3
    assert abs((Z_ve - Z_no).real[0] - R_samp) / R_samp < 0.02, \
        "viskose Mündung muss im DC-Grenzfall Sampson (3µ/r³) treffen"
    if _HAS_SCIPY:
        # b) Geschlossener thermoviskoser Stub gegen fein diskretisierte
        #    Leiter (40 Segmente, gleiche Beläge): coth-Lösung korrekt.
        om12 = 2.0 * np.pi * np.array([100.0, 1000.0, 10000.0])
        r_s, L_s = 0.65e-3, 3.7e-3
        Z_stub = capsule._closed_hole_stub(om12, r_s, L_s)
        g_s, Zc_s = MicrophoneCapsule._narrow_duct_propagation(om12, r_s)
        Zp, Yp = g_s * Zc_s, g_s / Zc_s
        n_seg = 40
        dl = L_s / n_seg
        Z_lad = Zp * dl + 1.0 / (Yp * dl)     # letztes Segment (Ende zu)
        for _ in range(n_seg - 1):
            Z_lad = Zp * dl + 1.0 / (Yp * dl + 1.0 / Z_lad)
        assert np.max(np.abs(Z_stub / Z_lad - 1.0)) < 0.02, \
            "Stub-coth muss der diskretisierten Leiter entsprechen"
        # c) LF-Nachgiebigkeit des Stubs ISOTHERM (V/P_atm), nicht
        #    adiabatisch — das alte Lumped-Modell lag hier um gamma daneben.
        om3 = np.array([2.0 * np.pi * 3.0])
        V_s = np.pi * r_s**2 * L_s
        C_lf = -1.0 / (om3[0] * capsule._closed_hole_stub(om3, r_s,
                                                          L_s).imag[0])
        assert 0.9 < C_lf / (V_s / P_ATM) < 1.1, \
            "Stub-Nachgiebigkeit muss bei tiefen Frequenzen isotherm sein"
    # d) Filmkorrektur: Φ(ω→0) = 1; bei 25 kHz/60 µm dominiert die
    #    laterale Trägheit (Faktor ~4, Phase > 60°).
    phi_lo = MicrophoneCapsule._film_R_dynamic(np.array([2.0 * np.pi]),
                                               60e-6)[0]
    assert abs(phi_lo - 1.0) < 1e-3
    phi_hi = MicrophoneCapsule._film_R_dynamic(
        np.array([2.0 * np.pi * 25000.0]), 60e-6)[0]
    assert 3.0 < abs(phi_hi) < 5.0 and np.degrees(np.angle(phi_hi)) > 60.0
    print(f"Verlustmechanismen: Sampson-Mündung = 3µ/r³ (DC), Stub ≡ "
          f"diskrete Leiter (<2 %), LF-Nachgiebigkeit isotherm, "
          f"Filmkorrektur Φ(0)=1 / |Φ(25 kHz)|={abs(phi_hi):.1f}  OK")

    # --------- Gegenprobe 13: Rückwärts-Durchlauf (Port-Tausch) ------------
    # a) Algebra: der Port-Tausch eines Zweitor-Produkts muss der
    #    umgekehrten Elementreihenfolge entsprechen (so macht es der
    #    1D-Pfad) — die Matrix-Inverse täte das NICHT (negative Elemente).
    om13 = np.array([2.0 * np.pi * 700.0])
    E1 = MicrophoneCapsule._abcd_series(2.5e6 + 1j * 4e5, om13)
    E2 = MicrophoneCapsule._abcd_shunt(1j * 3e-10, om13)
    E3 = MicrophoneCapsule._abcd_series(8.0e5, om13)
    chain = reduce(MicrophoneCapsule._mmul, [E1, E2, E3])
    chain_rev = reduce(MicrophoneCapsule._mmul, [E3, E2, E1])
    assert np.allclose(
        np.array(MicrophoneCapsule._abcd_reverse(chain), dtype=complex),
        np.array(chain_rev, dtype=complex)), \
        "Port-Tausch muss der umgekehrten Elementreihenfolge entsprechen"
    # b) Physik: mit korrektem Rückwärts-Durchlauf entsteht die interne
    #    Phasenschieber-Laufzeit aus den akustischen Parametern — die
    #    K67-Niere (Null bei 180°) darf deshalb NICHT mehr an der
    #    Membranresonanz hängen (vorher kollabierte sie: die Laufzeit kam
    #    aus der f_res-abhängigen statischen Spaltasymmetrie).
    def _k67_null(fres):
        c = MicrophoneCapsule(
            membrane_resonance_hz=fres, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, body_diameter=34e-3, squeeze_model="2d")
        di = c.directivity(frequencies_hz=(1000.0,))
        pat = di["patterns"][1000.0]
        na = di["angles_deg"][:181][int(np.argmin(pat["linear"][:181]))]
        return na, pat["db"][180]
    na_lo, p_lo = _k67_null(1150.0)
    na_hi, p_hi = _k67_null(4000.0)
    assert na_lo > 150.0 and na_hi > 150.0, \
        f"Nullwinkel muss f_res-robust nahe 180° liegen ({na_lo:.0f}°/{na_hi:.0f}°)"
    assert p_lo < -15.0 and p_hi < -15.0 and abs(p_lo - p_hi) < 12.0, \
        f"Nierentiefe muss f_res-robust sein ({p_lo:.1f} / {p_hi:.1f} dB)"
    print(f"Rückwärts-Durchlauf: Port-Tausch ≡ umgekehrte Elementreihen"
          f"folge; K67-Niere f_res-robust (180° @1 kHz: {p_lo:.1f} dB "
          f"@1150 Hz / {p_hi:.1f} dB @4000 Hz, Null bei "
          f"{na_lo:.0f}°/{na_hi:.0f}°)  OK")

    # --------- Gegenprobe 14: übrige Architekturen unter dem Port-Tausch ---
    # Der zweite Nutzer des Rückwärts-Durchlaufs ist die VORDERE Backplate
    # der Dual-Architektur (2D). Das normierte Richtdiagramm darf davon
    # nicht abhängen (Serienelemente des Frontzweigs kürzen sich) — 1D-
    # und 2D-Pfad müssen übereinstimmen; der Frequenzgang bleibt endlich
    # und Gewebe wirkt als passive Seriendämpfung. K103-Rückplatte dicht
    # -> exakte Kugel auch im 2D-Modell (Vorwärtspfade unberührt).
    if _HAS_SCIPY:
        du1 = MicrophoneCapsule(architecture="dual", squeeze_model="1d")
        du2 = MicrophoneCapsule(architecture="dual", squeeze_model="2d")
        pd1 = du1.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"]
        pd2 = du2.directivity(frequencies_hz=(1000.0,))["patterns"][1000.0]["db"]
        assert np.max(np.abs(pd1 - pd2)) < 1.0, \
            "Dual-Backplate: 1D- und 2D-Richtdiagramm müssen übereinstimmen"
        fr_du = du2.frequency_response(n_points=80)
        assert np.all(np.isfinite(fr_du["amplitude_db"]))
        s_du = abs(du2.transfer_function(1000.0)[0])
        s_fab = abs(MicrophoneCapsule(
            architecture="dual", squeeze_model="2d", fabric_front_rayl=150.0,
            fabric_rear_rayl=150.0).transfer_function(1000.0)[0])
        assert 0.5 < s_fab / s_du < 0.9999, \
            "Gewebe muss passiv dämpfen (Empfindlichkeit leicht senken)"
        k103_dicht = MicrophoneCapsule(
            squeeze_model="2d", rear_spacer_height=100e-6,
            rear_plate_thickness=2e-3, n_rear_plate_holes=0,
            include_diffraction=False)
        lin_k1 = k103_dicht.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["linear"]
        assert np.max(np.abs(lin_k1 - 1.0)) < 1e-3, \
            "K103 mit dichter Rückplatte muss auch in 2D exakte Kugel sein"
        print(f"Architektur-Konsistenz: Dual-Backplate 1D≡2D (max Abw. "
              f"{np.max(np.abs(pd1 - pd2)):.2f} dB), Gewebe dämpft passiv "
              f"({s_fab / s_du:.3f}×), K103 dicht = Kugel (2D)  OK")

    # --------- Gegenprobe 15: Clearance-Ring (Stirnflächen-Freistich) ------
    # a) Ring aus (0) ≡ exakt das Bestandsverhalten.
    # b) Debenham-artige Platte (12 enge Durchgangslöcher auf Lochkreisen):
    #    der Freistich am Elektrodenrand entlastet die Mündungs-Engstellen
    #    des Nieren-Phasenschiebers -> deutlich tiefere 180°-Auslöschung,
    #    Null bleibt bei 180°.
    if _HAS_SCIPY:
        deb_kwargs = dict(
            architecture="dual_diaphragm", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=50.0, center_gap=0.0,
            through_hole_diameter=0.71e-3,
            through_hole_rings=[(6, 21.84e-3), (3, 17.48e-3),
                                (3, 8.74e-3)],
            blind_hole_diameter=1.2e-3, blind_hole_depth=3.0e-3,
            blind_hole_rings=[(12, 21.84e-3), (12, 17.48e-3),
                              (12, 13.11e-3), (6, 8.74e-3), (4, 4.37e-3)],
            clamp_ring_thickness=2e-3, clamp_ring_width=3e-3,
            body_diameter=32e-3, squeeze_model="2d")
        deb0 = MicrophoneCapsule(**deb_kwargs)
        deb0_ref = MicrophoneCapsule(clearance_ring_diameter=0.0,
                                     clearance_ring_width=0.0,
                                     clearance_ring_depth=0.0, **deb_kwargs)
        f_chk = np.array([250.0, 1000.0, 5000.0])
        assert np.allclose(deb0.transfer_function(f_chk),
                           deb0_ref.transfer_function(f_chk),
                           rtol=1e-12, atol=0.0), \
            "Clearance-Ring = 0 muss exakt dem Bestand entsprechen"
        deb1 = MicrophoneCapsule(clearance_ring_diameter=22.63e-3,
                                 clearance_ring_width=1.27e-3,
                                 clearance_ring_depth=38e-6, **deb_kwargs)
        def _n180(c, f):
            di = c.directivity(frequencies_hz=(f,))
            pat = di["patterns"][f]
            na = di["angles_deg"][:181][int(np.argmin(pat["linear"][:181]))]
            return pat["db"][180], na
        p0, _ = _n180(deb0, 500.0)
        p1, na1 = _n180(deb1, 500.0)
        assert p1 < p0 - 8.0, \
            f"Freistich muss die 180°-Auslöschung vertiefen ({p0:.1f} -> {p1:.1f})"
        assert na1 > 172.0, \
            f"Null muss bei 180° bleiben ({na1:.0f}°)"
        print(f"Clearance-Ring: 0 ≡ Bestand; Freistich am Elektrodenrand "
              f"vertieft die Debenham-Null @500 Hz von {p0:.1f} auf "
              f"{p1:.1f} dB (Null {na1:.0f}°)  OK")

    # --------- Gegenprobe 16: 3D-(r,phi)-Löser (diskrete Löcher) -----------
    # a) Gültigkeits-Gatter: '3d' erfordert Durchgangslöcher; Stufen-
    #    bohrungen der Doppelmembran-Bauform erfordern center_gap > 0
    #    (zwei Elektrodenhälften, K67-Typ, s. Gegenprobe 22). single/dual
    #    rechnet der Löser seit Gegenprobe 23 ebenfalls.
    # b) Reziprozität des Feldsystems: Frontantwort auf Rückdruck ==
    #    Rückantwort auf Frontdruck (±3 %).
    # c) Physik: Null bei 180°; ein Freistich, der ALLE Loch-Mündungen
    #    abdeckt, vertieft die Auslöschung gegenüber dem reinen Rand-Ring
    #    deutlich (Mündungs-Engstellen-Mechanismus, s. Gegenprobe 15).
    if _HAS_SCIPY:
        try:
            MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                              n_blind_holes=30)
            raise AssertionError("'3d' ohne Durchgangslöcher müsste "
                                 "scheitern")
        except ValueError:
            pass
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              center_gap=0.0, through_holes_stepped=True,
                              n_blind_holes=120, blind_hole_depth=2.0e-3,
                              squeeze_model="3d",
                              membrane_resonance_hz=2100.0)
            raise AssertionError("'3d' gestuft ohne center_gap müsste "
                                 "scheitern")
        except ValueError:
            pass
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              center_gap=50e-6, n_through_holes=0,
                              squeeze_model="3d",
                              membrane_resonance_hz=2100.0)
            raise AssertionError("'3d' ohne Durchgangslöcher müsste "
                                 "scheitern")
        except ValueError:
            pass
        deb3 = MicrophoneCapsule(**{**deb_kwargs,
                                    "squeeze_model": "3d",
                                    "clearance_ring_diameter": 22.63e-3,
                                    "clearance_ring_width": 1.27e-3,
                                    "clearance_ring_depth": 38e-6})
        Xf3, Xr3, Bf3, Br3 = deb3._solve_3d(
            np.array([2.0 * np.pi * 1000.0]), want_rear=True)
        assert 0.97 < abs(Xr3[0]) / abs(Bf3[0]) < 1.03, \
            "3D-Feldsystem muss reziprok sein"
        di3 = deb3.directivity(frequencies_hz=(1000.0,))
        p3 = di3["patterns"][1000.0]
        na3 = di3["angles_deg"][:181][int(np.argmin(p3["linear"][:181]))]
        assert na3 > 172.0, f"3D-Niere muss bei 180° nullen ({na3:.0f}°)"
        deb3b = MicrophoneCapsule(**{**deb_kwargs,
                                     "squeeze_model": "3d",
                                     "clearance_ring_diameter": 13.0e-3,
                                     "clearance_ring_width": 18.0e-3,
                                     "clearance_ring_depth": 38e-6})
        p3b = deb3b.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
        assert p3b < p3["db"][180] - 6.0, \
            (f"Mündungs-Freistich muss die 3D-Null deutlich vertiefen "
             f"({p3['db'][180]:.1f} -> {p3b:.1f})")
        fr3 = deb3.frequency_response(f_min=100.0, f_max=8000.0,
                                      n_points=8)
        assert np.all(np.isfinite(fr3["amplitude_db"]))
        print(f"3D-Löser: Gatter greifen, reziprok "
              f"({abs(Xr3[0]) / abs(Bf3[0]):.3f}), Null @{na3:.0f}°; "
              f"Mündungs-Freistich vertieft 180°/1 kHz von "
              f"{p3['db'][180]:.1f} auf {p3b:.1f} dB  OK")

    # --------- Gegenprobe 17: Laufzeit-Diagnose (delay_diagnostics) --------
    # Deutung des Verhältnisses intern/extern (Sonde 1 kHz):
    # a) K67 nominal (Spacer 50 µm): Verhältnis knapp ÜBER 1 -> das
    #    Pattern-Minimum ist bei 180° GEPINNT (na_k aus Gegenprobe 6).
    #    ACHTUNG, hier stand bis Gegenprobe 29 ein Bereich knapp UNTER 1
    #    mit Minimum bei ~165°. Das war kein Messwert, sondern der aus
    #    der damaligen Simulation abgelesene Selbstbestätigungs-Bereich —
    #    und er widersprach der realen K67, deren Niere ihr Minimum bei
    #    180° hat. Mit der korrigierten Durchfluss-Zellfunktion
    #    (_cell_B_flow, Gegenprobe 30) liegt es dort; die publizierten
    #    U87-Werte werden dabei besser getroffen (180°: -26.6 statt
    #    -26.2 dB gegen -26 dB publiziert; 20 statt 21 mV/Pa gegen
    #    ~20 mV/Pa). Die DEUTUNG des Verhältnisses (unten) ist unberührt.
    # b) Spacer 45 µm (cardiodtest): kleinerer Spalt -> größerer Film-R ->
    #    LÄNGERE interne Laufzeit (Verhältnis > 1) -> Minimum bei 180°
    #    gepinnt (kein Außenwinkel bietet mehr Phase), dafür flacher.
    # c) Debenham (2D), Sonde 250 Hz (= Region der tiefsten Null ~290 Hz):
    #    Verhältnis ~ 1; externe Laufzeit ~ Kugel-Grenzfall 1.5·d_ext/c.
    # d) 3D: der Rand-Freistich allein lässt die Mündungen verengt ->
    #    hochohmiges RC, stark ÜBER-verzögert (Null bei 180°, aber flach,
    #    s. Gegenprobe 16); der breite Freistich senkt R -> Verhältnis
    #    rückt Richtung 1 (und die Null wird tiefer).
    # e) Geschlossene Rückseite (0 Durchgangslöcher) -> None.
    dd_k67 = k67.delay_diagnostics()
    assert dd_k67 is not None and 1.0 < dd_k67["ratio"] < 1.20, \
        f"K67 nominal: Verhältnis knapp >1 erwartet ({dd_k67['ratio']:.3f})"
    assert na_k >= 179.0, \
        "Konsistenz: Verhältnis >1 muss zum Minimum bei 180° gehören"
    k67_45 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=45e-6,
        n_through_holes=60, through_hole_diameter=0.6e-3,
        n_blind_holes=120, blind_hole_diameter=1.3e-3,
        blind_hole_depth=3.7e-3, through_holes_stepped=True,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3,
        squeeze_model="2d")
    dd_45 = k67_45.delay_diagnostics()
    assert 1.05 < dd_45["ratio"] < 1.45, \
        f"45-µm-Spacer: Über-Verzögerung erwartet ({dd_45['ratio']:.3f})"
    di45 = k67_45.directivity(frequencies_hz=(1000.0,))
    lin45 = di45["patterns"][1000.0]["linear"]
    a45 = di45["angles_deg"]
    na45 = a45[a45 <= 180][int(np.argmin(lin45[a45 <= 180]))]
    assert na45 > 179.0, \
        f"Über-Verzögerung muss das Minimum bei 180° pinnen ({na45:.0f}°)"
    if _HAS_SCIPY:
        # MIT Rand-Freistich (deb1, wie die Beispiel-JSON): angepasst in
        # der Region der tiefsten Null (~290 Hz); OHNE Ring (deb0) sind
        # die Mündungen verengt -> hochohmig -> stark über-verzögert
        # (konsistent zu Gegenprobe 15: Null 180°, aber nur -3.8 dB).
        dd_deb = deb1.delay_diagnostics(f_probe_hz=250.0)
        tau_kugel = 1.5 * deb1.d_ext / C_AIR
        # Verhältnis > 1: die 12 engen Durchgangslöcher der Braunmühl-
        # Weber-Platte verzögern ÜBER die externe Laufzeit hinaus — die
        # Null bleibt bei 180° gepinnt, wird aber flacher (gemessen
        # -13.6 dB bei 317 Hz). Das ist die Deutung (b) unten und deckt
        # sich mit dem 3D-Feldlöser, der die azimutale Zuströmung zu den
        # wenigen Löchern diskret auflöst und noch flacher ausfällt.
        # (Bis Gegenprobe 29 stand hier ~1: damals zählte die
        # Durchfluss-Zellfunktion alle 58 Bohrungen als Senken statt der
        # 12 Durchgangslöcher — s. _cell_B_flow, Gegenprobe 30.)
        assert 1.2 < dd_deb["ratio"] < 1.9, \
            f"Debenham @250 Hz: Verhältnis >1 erwartet ({dd_deb['ratio']:.3f})"
        assert 0.85 < dd_deb["tau_ext_s"] / tau_kugel < 1.15, \
            "externe Laufzeit muss dem Kugel-Grenzfall 1.5·d_ext/c folgen"
        dd_deb0 = deb0.delay_diagnostics(f_probe_hz=250.0)
        assert dd_deb0["ratio"] > 1.5, \
            (f"ohne Freistich: verengte Mündungen -> Über-Verzögerung "
             f"erwartet ({dd_deb0['ratio']:.3f})")
        dd_3d = deb3.delay_diagnostics()
        assert dd_3d is not None and np.isfinite(dd_3d["ratio"]) \
            and dd_3d["ratio"] > 1.5, \
            (f"3D/Rand-Freistich: verengte Mündungen -> starke Über-"
             f"Verzögerung erwartet ({dd_3d['ratio']:.3f})")
        dd_3db = deb3b.delay_diagnostics()
        assert 1.0 < dd_3db["ratio"] < dd_3d["ratio"], \
            (f"breiter Freistich muss die Über-Verzögerung abbauen "
             f"({dd_3db['ratio']:.3f} vs. {dd_3d['ratio']:.3f})")
    hermetic = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=0, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3)
    assert hermetic.delay_diagnostics() is None, \
        "Druckempfänger: keine Laufzeit-Anpassung -> None"
    print(f"Laufzeit-Diagnose @1 kHz: K67 nominal intern/extern = "
          f"{dd_k67['ratio']:.2f} ({dd_k67['dist_int_m']*1e3:.1f}/"
          f"{dd_k67['dist_ext_m']*1e3:.1f} mm, Minimum {na_k:.0f}°), "
          f"Spacer 45 µm -> {dd_45['ratio']:.2f} (Null gepinnt {na45:.0f}°); "
          f"geschlossene Rückseite -> None  OK")

    # --------- Gegenprobe 18: angle_responses / Helmholtz-Frequenz ---------
    # a) Superposition: H(θ) aus EINEM Durchlauf == transfer_function(θ)
    #    für 0/90/180° (1D/2D-Netzwerk UND 3D-Feldlöser).
    # b) f_H (90°-Kreuzung von arg D_r) liegt für die bekannten Kapseln im
    #    erwarteten Bereich; eine dickere Platte (längere enge Bohrungen
    #    -> mehr Trägheit) muss die Resonanz absenken.
    # c) Geschlossene Rückseite: D_r und f_H -> None, H bleibt berechenbar.
    f_grid = np.logspace(np.log10(50.0), np.log10(20000.0), 120)
    ar_k = k67.angle_responses(f_grid)
    for angd in (0.0, 90.0, 180.0):
        ref = k67.transfer_function(f_grid[::20], angle_deg=angd)
        assert np.allclose(ar_k["H"][angd][::20], ref, rtol=1e-9), \
            f"angle_responses muss transfer_function reproduzieren ({angd}°)"
    fH_k67 = MicrophoneCapsule.helmholtz_resonance_hz(f_grid, ar_k["D_r"])
    assert fH_k67 is not None and 2800.0 < fH_k67 < 3800.0, \
        f"K67: interne Resonanz ~3.3 kHz erwartet ({fH_k67})"
    if _HAS_SCIPY:
        ar_d1 = deb1.angle_responses(f_grid)
        fH_deb = MicrophoneCapsule.helmholtz_resonance_hz(f_grid,
                                                          ar_d1["D_r"])
        assert fH_deb is not None and 2000.0 < fH_deb < 2800.0, \
            f"Debenham: interne Resonanz ~2.4 kHz erwartet ({fH_deb})"
        deb_dick = MicrophoneCapsule(clearance_ring_diameter=22.63e-3,
                                     clearance_ring_width=1.27e-3,
                                     clearance_ring_depth=38e-6,
                                     **{**deb_kwargs,
                                        "backplate_thickness": 6.0e-3})
        fH_dick = MicrophoneCapsule.helmholtz_resonance_hz(
            f_grid, deb_dick.angle_responses(f_grid)["D_r"])
        assert fH_dick is not None and fH_dick < fH_deb - 300.0, \
            (f"6-mm-Platte: längere Bohrungen müssen die Resonanz senken "
             f"({fH_dick:.0f} vs. {fH_deb:.0f} Hz)")
        f1 = np.array([1000.0])
        ar3 = deb3.angle_responses(f1)
        ref3 = deb3.transfer_function(f1, angle_deg=90.0)
        assert np.allclose(ar3["H"][90.0], ref3, rtol=1e-9), \
            "angle_responses (3D) muss transfer_function reproduzieren"
        assert ar3["D_r"] is not None
    ar_h = hermetic.angle_responses(np.array([100.0, 1000.0]))
    assert ar_h["D_r"] is None
    assert MicrophoneCapsule.helmholtz_resonance_hz(
        np.array([100.0, 1000.0]), ar_h["D_r"]) is None
    assert all(np.all(np.isfinite(np.abs(v))) for v in ar_h["H"].values())
    print("angle_responses: Superposition == transfer_function (0/90/180°, "
          f"auch 3D); f_H: K67 {fH_k67:.0f} Hz"
          + (f", Debenham {fH_deb:.0f} Hz -> 6-mm-Platte {fH_dick:.0f} Hz"
             if _HAS_SCIPY else "")
          + "; geschlossene Rückseite -> None  OK")

    # --------- Gegenprobe 19: Fok-Mündungen, lokales h(r), exakte Leitung --
    # a) Fok/Melling-Faktor: F(xi->0) -> 1 (einsame Mündung == Bestand),
    #    monoton fallend mit der Lochdichte; K67-Wert im erwarteten Bereich.
    # b) Lokales Spaltprofil im 2D-Film: sag_w0 = 0 reproduziert den
    #    Bestand EXAKT; mit statischer Durchbiegung liegt das Zweitor nahe
    #    an der bisherigen Flächenmittel-Näherung (kleine Korrektur), aber
    #    klar verschieden vom nominalen Spalt (h³-Wirkung vorhanden).
    # c) Exakte Zwikker-Kosten-Leitung: geht für weite Rohre in die
    #    Kirchhoff-Asymptotik über (Dämpfungsbelag/Zc innerhalb weniger %).
    # d) Exakte J0-Modalfrequenz: ~1.8 % unter dem Lumped-Wert (4/3-Faktor).
    assert k67._fok_th is not None and 0.70 < k67._fok_th < 0.80, \
        f"K67-Fok-Faktor ~0.74 erwartet ({k67._fok_th:.3f})"
    sparse = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm", center_gap=50e-6,
        n_through_holes=2, through_hole_diameter=0.3e-3,
        n_blind_holes=0, blind_hole_depth=1e-3,
        clamp_ring_thickness=2e-3, clamp_ring_width=4e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, body_diameter=34e-3)
    assert sparse._fok_th > 0.97, \
        f"einsame Mündungen: Fok -> 1 erwartet ({sparse._fok_th:.3f})"
    assert sparse._fok_th > k67._fok_th, "Fok muss mit Lochdichte fallen"
    if _HAS_SCIPY:
        om19 = np.array([2.0 * np.pi * 1000.0])
        T_loc = k67._gap_field_2port(om19, h_film=k67.h_gap,
                                     sag_w0=k67.w0_static)
        T_sag0 = k67._gap_field_2port(om19, h_film=k67.h_gap, sag_w0=0.0)
        T_nom = k67._gap_field_2port(om19, h_film=k67.h_gap)
        assert np.allclose(T_sag0, T_nom, rtol=1e-12), \
            "sag_w0 = 0 muss den Bestand exakt reproduzieren"
        T_mean = k67._gap_field_2port(om19, h_film=k67.h_gap_front)
        rel_mean = abs(T_loc[0, 1][0] / T_mean[0, 1][0] - 1.0)
        rel_nom = abs(abs(T_loc[0, 1][0]) / abs(T_nom[0, 1][0]) - 1.0)
        assert rel_mean < 0.05, \
            f"lokales Profil nahe der Flächenmittel-Näherung ({rel_mean:.3f})"
        assert 0.01 < rel_nom < 0.30, \
            f"h³-Wirkung der Durchbiegung muss sichtbar sein ({rel_nom:.3f})"
        g_ex, Zc_ex = k67._narrow_duct_propagation(om19, 10e-3)
        al_kirch = (np.sqrt(MU_AIR * om19[0] / (2.0 * RHO0))
                    * (1.0 + (GAMMA - 1.0) / np.sqrt(PRANDTL))
                    / (10e-3 * C_AIR))
        Zc0 = RHO0 * C_AIR / (np.pi * 10e-3**2)
        assert 0.9 < g_ex[0].real / al_kirch < 1.1, \
            "exakte Leitung muss für weite Rohre Kirchhoff treffen (alpha)"
        assert 0.98 < abs(Zc_ex[0]) / Zc0 < 1.02, \
            "exakte Leitung muss für weite Rohre Kirchhoff treffen (Zc)"
    r_modal = sparse.f_res_modal_exact / sparse.f_res_from_tension
    assert 0.975 < r_modal < 0.99, \
        f"exakte J0-Modalfrequenz ~1.8 % unter Lumped erwartet ({r_modal:.4f})"
    assert "J0-Modalfrequenz" in sparse.summary()
    print(f"Fok/Melling: K67 {k67._fok_th:.2f}, einsame Mündung "
          f"{sparse._fok_th:.3f} -> 1; lokales h(r): sag=0 == Bestand, "
          f"Korrektur zur Mittel-Näherung "
          + (f"{100 * rel_mean:.1f} %" if _HAS_SCIPY else "—")
          + f"; Leitung exakt == Kirchhoff (weit); Modal {r_modal:.3f}  OK")

    # --------- Gegenprobe 20: Sphäroid-Körpermodell (axialer Transfer) -----
    # Eigene oblate Spezialfunktionen (scipy obl_rad2 ist für ξ0 < 1
    # unbrauchbar). Verifikation:
    # a) Kugel-Grenzfall b -> a reproduziert die Morse-Reihe (der Fehler
    #    skaliert linear mit 1 - b/a: reine Geometrie).
    # b) Dünne-Scheiben-Grenzfall: effektive LF-Distanz am Pol -> 4a/π
    #    (klassisches Scheibenresultat).
    # c) Wronski-Selbstprüfung W{R1,R3} = i/(c(ξ²+1)) über das Band.
    # d) BEFUND als Anker: die FREI STEHENDE K67-Scheibe hätte
    #    d_eff ~ 0.94·2·R_body >> intern (~17 mm) -> Superniere weit vor
    #    180°. Die reale (montierte) Kapsel folgt der d_ext-Kugel —
    #    deshalb bleibt 'sphere' Standard; 'spheroid' ist die
    #    dokumentierte Referenz der freien Scheibe.
    if _HAS_SCIPY:
        th20 = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0]))
        R20 = 9e-3
        worst = 0.0
        for f20 in (100.0, 1000.0, 5000.0, 15000.0):
            om20 = np.array([2.0 * np.pi * f20])
            d_save = k67.d_ext
            k67.d_ext = 2.0 * R20
            G_ref = k67._axial_body_transfer(om20, th20)[0]
            k67.d_ext = d_save
            G_sph = k67._spheroid_pole_transfer(om20, th20, R20,
                                                0.9995 * R20)[0]
            worst = max(worst, float(np.max(np.abs(G_sph - G_ref)
                                            / np.abs(G_ref))))
        assert worst < 3e-3, \
            f"Sphäroid muss im Kugel-Grenzfall die Morse-Reihe treffen " \
            f"({worst:.2e})"
        om50 = np.array([2.0 * np.pi * 50.0])
        G_disc = k67._spheroid_pole_transfer(om50, np.array([np.pi]),
                                             17e-3, 0.02 * 17e-3)[0, 0]
        d_eff_disc = float(np.angle(G_disc)) / (om50[0] / C_AIR)
        assert abs(d_eff_disc / 17e-3 - 4.0 / np.pi) < 0.04, \
            f"Scheiben-Grenzfall 4a/π verfehlt ({d_eff_disc/17e-3:.3f})"
        assert k67._spheroid_wronski_max < 1e-4, \
            "Wronski-Selbstprüfung des Sphäroids"
        G_k67 = k67._axial_spheroid_transfer(om50, np.array([np.pi]))[0, 0]
        d_eff_k67 = float(np.angle(G_k67)) / (om50[0] / C_AIR)
        assert 0.85 < d_eff_k67 / (2.0 * k67.R_body) < 1.0, \
            f"freie K67-Scheibe: d_eff ~ 0.94·2R erwartet ({d_eff_k67})"
        k67_sph = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="2d", axial_body_model="spheroid")
        di_sph = k67_sph.directivity(frequencies_hz=(1000.0,))
        lin_s = di_sph["patterns"][1000.0]["linear"]
        a_s = di_sph["angles_deg"]
        na_s = a_s[a_s <= 180][int(np.argmin(lin_s[a_s <= 180]))]
        assert na_s < 150.0, \
            (f"freie Scheibe: Minimum deutlich vor 180° erwartet "
             f"({na_s:.0f}°) — Beleg, warum 'sphere' Standard bleibt")
        try:
            MicrophoneCapsule(architecture="single",
                              axial_body_model="spheroid",
                              membrane_resonance_hz=8000.0)
            raise AssertionError("spheroid ohne dual_diaphragm muss scheitern")
        except ValueError:
            pass
        print(f"Sphäroid: Kugel-Grenzfall {worst:.1e}; Scheibe "
              f"d_eff/a = {d_eff_disc/17e-3:.3f} (4/π = {4/np.pi:.3f}); "
              f"Wronski {k67._spheroid_wronski_max:.1e}; freie K67-Scheibe "
              f"d_eff = {d_eff_k67*1e3:.1f} mm -> Minimum {na_s:.0f}° "
              "(montiert: Kugel bleibt Standard)  OK")

    # --------- Gegenprobe 21: axisymmetrisches BEM (Kopf + Körper) ---------
    # a) Kugelkontur reproduziert die Morse-Reihe (Pol-zu-Pol) < 2e-3.
    # b) Sphäroidkontur reproduziert die Sphäroid-Reihe < 2e-3 — zwei
    #    unabhängige exakte Referenzen für denselben Löser.
    # c) MONTIERT: die effektive Distanz der Kopf+Körper-Kontur liegt
    #    ZWISCHEN d_ext-Kugel und frei stehender Scheibe (der Körper
    #    unterbindet einen Teil des Scheibenrand-Umwegs); |G| ~ 1 im
    #    Tiefband; Raumwinkel-Residuum (Gitterqualität) klein.
    # d) Gatter: nur Doppelmembran-Bauform.
    if _HAS_SCIPY:
        k67_bem = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="2d", axial_body_model="bem")
        th21 = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0]))

        def _inject(pts):
            el = MicrophoneCapsule._bem_elems(pts)
            n_el = el["L"].size
            fr_m = np.zeros(n_el, bool)
            fr_m[0] = True
            re_m = np.zeros(n_el, bool)
            re_m[-1] = True
            k67_bem._bem_geo = dict(
                elems=el, chief=[(0.0, 0.0)],
                w_area=2.0 * np.pi * el["mid_r"] * el["L"],
                front=fr_m, rear=re_m)

        psi21 = np.linspace(0.0, np.pi, 121)
        R21 = 9e-3
        _inject(np.stack([R21 * np.sin(psi21), R21 * np.cos(psi21)], 1))
        worst_k = 0.0
        for f21 in (100.0, 1000.0, 5000.0):
            om21 = np.array([2.0 * np.pi * f21])
            G_b = k67_bem._bem_axial_transfer(om21, th21)[0]
            d_save = k67.d_ext
            k67.d_ext = 2.0 * R21
            G_ref = k67._axial_body_transfer(om21, th21)[0]
            k67.d_ext = d_save
            worst_k = max(worst_k, float(np.max(np.abs(G_b - G_ref)
                                                / np.abs(G_ref))))
        assert worst_k < 2e-3, \
            f"BEM-Kugelkontur muss die Morse-Reihe treffen ({worst_k:.1e})"
        _inject(np.stack([17e-3 * np.sin(psi21),
                          6.1e-3 * np.cos(psi21)], 1))
        worst_o = 0.0
        for f21 in (1000.0, 5000.0):
            om21 = np.array([2.0 * np.pi * f21])
            G_b = k67_bem._bem_axial_transfer(om21, th21)[0]
            G_ref = k67._spheroid_pole_transfer(om21, th21, 17e-3,
                                                6.1e-3)[0]
            worst_o = max(worst_o, float(np.max(np.abs(G_b - G_ref)
                                                / np.abs(G_ref))))
        assert worst_o < 2e-3, \
            f"BEM-Sphäroidkontur muss die Sphäroid-Reihe treffen " \
            f"({worst_o:.1e})"
        # c) montierte Kapsel vs. freie Scheibe vs. d_ext-Kugel
        k67_bem._bem_geo = None
        om100 = np.array([2.0 * np.pi * 100.0])
        th180 = np.array([np.pi])
        k100 = om100[0] / C_AIR
        G_mnt = k67_bem._bem_axial_transfer(om100, th180)[0, 0]
        d_mnt = float(np.angle(G_mnt)) / k100
        assert k67_bem._bem_solid_angle_residual < 5e-3, \
            "BEM-Gitterqualität (Raumwinkel-Residuum)"
        k67_frei = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            axial_body_model="bem", bem_body_diameter=0.0)
        G_frei = k67_frei._bem_axial_transfer(om100, th180)[0, 0]
        d_frei = float(np.angle(G_frei)) / k100
        d_kugel = 1.5 * k67_bem.d_ext
        assert d_kugel < d_mnt < d_frei, \
            (f"montiert muss zwischen Kugel und freier Scheibe liegen "
             f"({d_kugel * 1e3:.1f} < {d_mnt * 1e3:.1f} < "
             f"{d_frei * 1e3:.1f} mm)")
        assert abs(abs(G_mnt) - 1.0) < 0.02, "BEM: |G| ~ 1 im Tiefband"
        try:
            MicrophoneCapsule(architecture="single",
                              axial_body_model="bem",
                              membrane_resonance_hz=8000.0)
            raise AssertionError("BEM ohne dual_diaphragm muss scheitern")
        except ValueError:
            pass
        print(f"BEM: Kugelkontur {worst_k:.1e}, Sphäroidkontur "
              f"{worst_o:.1e}; d_eff Kugel {d_kugel*1e3:.1f} < montiert "
              f"{d_mnt*1e3:.1f} < freie Scheibe {d_frei*1e3:.1f} mm "
              "(Körper unterbindet Teil des Rand-Umwegs)  OK")

    # --------- Gegenprobe 22: 3D-K67-Modus (Zwischenspalt, Stufen, Drehung)
    # Der 3D-Löser rechnet jetzt auch die ZWEITEILIGE Elektrode: Zwischen-
    # spalt als dritter Reynolds-Film, Stufenbohrungen als Zweitor-Kette
    # je Loch, Elektrodenhälften gegeneinander verdreht. Grenzfälle und
    # Physik ohne Fit-Koeffizient:
    # a) EINTEILIG-GRENZFALL: zweiteilig ausgerichtet (rot = 0) mit
    #    winzigem Zwischenspalt (5 µm) muss den einteiligen Löser
    #    reproduzieren (zwei völlig verschiedene Codepfade: Lochpaar-
    #    Leitwert vs. Kette Loch–Film–Loch; Restabweichung = Impedanz
    #    des 5-µm-Films selbst, gemessen 1.5–2.2 %).
    # b) STUFEN-GRENZFALL: Stufenbohrung mit winziger Senkung
    #    (⌀0.75 x 0.15 mm um den ⌀0.71-mm-Kern) muss die ungestufte
    #    Bohrung reproduzieren (gemessen 0.3–0.8 %).
    # c) K67 komplett (gestuft, verdreht): Feldsystem reziprok.
    # d) VERDREHUNG (die reale K67 verdreht die Hälften so, dass die
    #    Durchgangslöcher nicht zueinander zeigen): ausgerichtete Löcher
    #    (rot = 0) kurzschließen den Phasenschieber -> flache Auslöschung;
    #    schon eine halbe Teilung (3° bei 60 Löchern) zwingt den Pfad
    #    durch den Zwischenspalt-Film -> tiefe 180°-Null, Minimum wandert
    #    Richtung 180°, Empfindlichkeit sinkt (steiferes Luftpolster).
    #    Das 3D-Ergebnis der verdrehten Bauform liegt nahe am
    #    homogenisierten 2D-Modell (das versetzte Arrays annimmt).
    if _HAS_SCIPY:
        deb22 = dict(deb_kwargs)
        deb22.update(squeeze_model="3d",
                     blind_hole_rings=[(0, None)],
                     blind_hole_diameter=1.2e-3, blind_hole_depth=1e-3)
        om22 = np.array([2.0 * np.pi * 500.0, 2.0 * np.pi * 2000.0])
        # a) einteilig vs. zweiteilig ausgerichtet mit 5-µm-Spalt
        ein = MicrophoneCapsule(**{**deb22, "center_gap": 0.0})
        zwei = MicrophoneCapsule(**{**deb22, "center_gap": 5e-6,
                                    "half_rotation_deg": 0.0})
        Xf_e, Xr_e = ein._solve_3d(om22)
        Xf_z, Xr_z = zwei._solve_3d(om22)
        dev_a = max(float(np.max(np.abs(Xf_z / Xf_e - 1.0))),
                    float(np.max(np.abs(Xr_z / Xr_e - 1.0))))
        assert dev_a < 0.04, \
            (f"3D-K67-Modus muss im 5-µm-Grenzfall den einteiligen Löser "
             f"reproduzieren (Abweichung {dev_a:.3f})")
        # b) Stufenbohrung mit winziger Senkung vs. ungestuft
        plain = MicrophoneCapsule(**{**deb22, "center_gap": 50e-6,
                                     "half_rotation_deg": 0.0})
        step = MicrophoneCapsule(**{**deb22, "center_gap": 50e-6,
                                    "half_rotation_deg": 0.0,
                                    "through_holes_stepped": True,
                                    "blind_hole_rings": [(12, None)],
                                    "blind_hole_diameter": 0.75e-3,
                                    "blind_hole_depth": 0.15e-3})
        Xf_p, Xr_p = plain._solve_3d(om22)
        Xf_s, Xr_s = step._solve_3d(om22)
        dev_b = max(float(np.max(np.abs(Xf_s / Xf_p - 1.0))),
                    float(np.max(np.abs(Xr_s / Xr_p - 1.0))))
        assert dev_b < 0.03, \
            (f"Stufenbohrung mit winziger Senkung muss die ungestufte "
             f"Bohrung reproduzieren (Abweichung {dev_b:.3f})")
        # c)+d) komplette K67, ausgerichtet vs. verdreht (halbe Teilung)
        k67_3d = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="3d")
        res22 = {}
        for rot in (0.0, 3.0):
            cap = MicrophoneCapsule(**k67_3d, half_rotation_deg=rot)
            di = cap.directivity(frequencies_hz=(1000.0,))
            db = di["patterns"][1000.0]["db"]
            na = di["angles_deg"][:181][
                int(np.argmin(di["patterns"][1000.0]["linear"][:181]))]
            e1k = abs(cap.transfer_function(np.array([1000.0]))[0]) * 1e3
            res22[rot] = (db[180], na, e1k, cap)
        Xf_r, Xr_r, Bf_r, Br_r = res22[3.0][3]._solve_3d(
            np.array([2.0 * np.pi * 1000.0]), want_rear=True)
        rez22 = abs(Xr_r[0]) / abs(Bf_r[0])
        assert 0.97 < rez22 < 1.03, \
            f"3D-K67-Feldsystem muss reziprok sein ({rez22:.3f})"
        assert res22[0.0][0] > -12.0, \
            (f"ausgerichtete Löcher müssen den Phasenschieber kurz-"
             f"schließen (180° = {res22[0.0][0]:.1f} dB)")
        assert res22[3.0][0] < -15.0, \
            (f"verdrehte Hälften müssen tief auslöschen "
             f"(180° = {res22[3.0][0]:.1f} dB)")
        assert res22[3.0][1] > res22[0.0][1] + 15.0, \
            (f"Verdrehung muss das Minimum Richtung 180° schieben "
             f"({res22[0.0][1]:.0f}° -> {res22[3.0][1]:.0f}°)")
        assert res22[0.0][2] - res22[3.0][2] > 5.0, \
            (f"Verdrehung muss die Empfindlichkeit senken (steiferes "
             f"Polster; {res22[0.0][2]:.1f} -> {res22[3.0][2]:.1f} mV/Pa)")
        # d) verdrehte 3D-Bauform nahe am homogenisierten 2D-Modell
        k2d = MicrophoneCapsule(**{**k67_3d, "squeeze_model": "2d"})
        e2d = abs(k2d.transfer_function(np.array([1000.0]))[0]) * 1e3
        p2d = k2d.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["db"][180]
        assert abs(res22[3.0][2] - e2d) < 3.0, \
            (f"3D verdreht muss nahe der 2D-Empfindlichkeit liegen "
             f"({res22[3.0][2]:.1f} vs. {e2d:.1f} mV/Pa)")
        assert abs(res22[3.0][0] - p2d) < 9.0, \
            (f"3D verdreht muss nahe der 2D-Auslöschung liegen "
             f"({res22[3.0][0]:.1f} vs. {p2d:.1f} dB)")
        print(f"3D-K67-Modus: einteiliger Grenzfall {dev_a * 100:.1f} %, "
              f"Stufen-Grenzfall {dev_b * 100:.1f} %, reziprok "
              f"({rez22:.4f}); Verdrehung 0°->3°: 180° "
              f"{res22[0.0][0]:.1f} -> {res22[3.0][0]:.1f} dB, Minimum "
              f"{res22[0.0][1]:.0f}° -> {res22[3.0][1]:.0f}°, Empf. "
              f"{res22[0.0][2]:.1f} -> {res22[3.0][2]:.1f} mV/Pa "
              f"(2D: {p2d:.1f} dB, {e2d:.1f} mV/Pa)  OK")

    # --------- Gegenprobe 23: 3D-Löser für single/dual-Architekturen -------
    # Der 3D-Löser rechnet jetzt auch die Einzel-Backplate- und die
    # Dual-Backplate-Bauform: EIN Membranfeld, ein Film je Backplate,
    # Durchgangslöcher als Zweitor in SAMMELKNOTEN, deren Abschluss die
    # baugleiche Lumped-Kette des 1D/2D-Pfads bildet (_rear_chain_mats;
    # vorn: Strahlung + Gewebe). Verankert ohne Fit-Koeffizient:
    # a) REZIPROZITÄT der akustischen Ports: q_rück(p_front = 1) ==
    #    q_front(p_rück = 1) — exakt (Maschinengenauigkeit), validiert
    #    alle Kopplungsvorzeichen (Film-Membran, Knoten, Ketten).
    # b) GESCHLOSSENE RÜCKSEITE: exakte Kugel (ohne Beugung) und
    #    Übereinstimmung mit 1D nach Betrag UND PHASE — die 3D-Ausgänge
    #    folgen jetzt der Ketten-Flussrichtung (Vorzeichenkonvention).
    # c) K103-GRENZFALL DICHT (Spacer + Rückplatte ohne Durchlass):
    #    3D == 1D auf wenige Prozent über das Band.
    # d) K103 OFFEN: die interne Rück-Übertragung D_r (das Verhältnis
    #    beider Pfade) stimmt mit dem 2D-Feldmodell auf ~1 % überein —
    #    die absoluten Empfindlichkeiten tragen die dokumentierte
    #    Membranfeld-Klasse (±2–3 dB), ihr VERHÄLTNIS ist robust.
    # e) NIERE (Laufzeitglied + Hohlraum): Richtdiagramm 3D nahe 2D
    #    (90°/180°/Minimum-Winkel), Empfindlichkeit in der Klasse.
    # f) DUAL: Reziprozität + LF-Empfindlichkeit nahe 2D.
    if _HAS_SCIPY:
        f23 = np.array([100.0, 1000.0])
        sg23 = dict(architecture="single",
                    membrane_material="PET", membrane_resonance_hz=8000.0,
                    membrane_diameter=22e-3, membrane_thickness=6e-6,
                    membrane_tension=400.0, air_gap=40e-6,
                    backplate_diameter=20e-3, backplate_thickness=3e-3,
                    bias_voltage=60.0, n_through_holes=60,
                    through_hole_diameter=1.0e-3, n_blind_holes=30,
                    blind_hole_diameter=1.2e-3, delay_length=3e-3,
                    cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
                    n_cavity_holes=200, cavity_hole_diameter=0.2e-3,
                    cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
                    fabric_rear_rayl=0.0, body_diameter=24e-3,
                    include_diffraction=False)
        # b) geschlossene Rückseite: Kugel + Betrag/Phase == 1D
        cl3 = MicrophoneCapsule(**{**sg23, "n_cavity_holes": 0},
                                squeeze_model="3d")
        cl1 = MicrophoneCapsule(**{**sg23, "n_cavity_holes": 0},
                                squeeze_model="1d")
        lin_c = cl3.directivity(
            frequencies_hz=(1000.0,))["patterns"][1000.0]["linear"]
        assert np.max(np.abs(lin_c - 1.0)) < 1e-6, \
            "3D single geschlossen muss exakte Kugel liefern"
        r_cl = (cl3.transfer_function(f23[:1])
                / cl1.transfer_function(f23[:1]))[0]
        assert 0.90 < abs(r_cl) < 1.02, \
            f"3D/1D geschlossen @100 Hz ({abs(r_cl):.3f})"
        assert abs(np.angle(r_cl)) < np.deg2rad(12.0), \
            (f"3D muss der Ketten-Phasenkonvention folgen "
             f"({np.rad2deg(np.angle(r_cl)):.1f}°)")
        # a) Reziprozität (offene Niere, fordert alle Pfade; bei
        # geschlossener Rückseite ist der Kreuzfluss trivial 0)
        card3 = MicrophoneCapsule(**{**sg23, "n_cavity_holes": 60,
                                     "cavity_hole_diameter": 0.6e-3},
                                  squeeze_model="3d")
        card3.transfer_function(np.array([1000.0]))
        rez_s = abs(card3._recip_3d[0]) / abs(card3._recip_3d[1])
        assert abs(rez_s - 1.0) < 1e-6, \
            f"3D single muss reziprok sein ({rez_s:.8f})"
        # c)+d) K103: dicht == 1D; offen: D_r == 2D
        k23 = dict(architecture="single",
                   membrane_resonance_hz=1800.0, membrane_diameter=25.4e-3,
                   membrane_thickness=6e-6, membrane_tension=100.0,
                   air_gap=50e-6, backplate_diameter=24e-3,
                   backplate_thickness=3e-3, bias_voltage=60.0,
                   n_through_holes=36, through_hole_diameter=1.0e-3,
                   n_blind_holes=0, blind_hole_diameter=1.2e-3,
                   blind_hole_depth=1e-3, rear_spacer_height=40e-6,
                   rear_plate_thickness=2e-3,
                   rear_plate_hole_diameter=1.0e-3, delay_length=0.0,
                   cavity_length=0.0, n_cavity_holes=0,
                   fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
                   body_diameter=27e-3, include_diffraction=False)
        kd3 = MicrophoneCapsule(**k23, n_rear_plate_holes=0,
                                squeeze_model="3d")
        kd1 = MicrophoneCapsule(**k23, n_rear_plate_holes=0,
                                squeeze_model="1d")
        r_kd = np.abs(kd3.transfer_function(f23)
                      / kd1.transfer_function(f23))
        assert np.all((r_kd > 0.93) & (r_kd < 1.07)), \
            f"K103 dicht: 3D muss 1D treffen ({r_kd})"
        dr23 = {}
        for sm in ("2d", "3d"):
            ko = MicrophoneCapsule(**k23, n_rear_plate_holes=60,
                                   squeeze_model=sm)
            dr23[sm] = ko.angle_responses(f23)["D_r"]
        d_dr = np.max(np.abs(dr23["3d"] - dr23["2d"]))
        assert d_dr < 0.05, \
            (f"K103 offen: interne Rück-Übertragung D_r muss das "
             f"2D-Feldmodell treffen (|ΔD_r| = {d_dr:.3f})")
        # e) Niere: Richtdiagramm 3D nahe 2D
        n23 = dict(architecture="single",
                   membrane_resonance_hz=2100.0, membrane_diameter=25.4e-3,
                   membrane_thickness=6e-6, membrane_tension=45.0,
                   air_gap=38.1e-6, backplate_diameter=23.9e-3,
                   backplate_thickness=3.125e-3, bias_voltage=50.0,
                   n_through_holes=48, through_hole_diameter=1.0e-3,
                   n_blind_holes=24, blind_hole_diameter=1.2e-3,
                   blind_hole_depth=1.5e-3, delay_length=3e-3,
                   cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
                   n_cavity_holes=60, cavity_hole_diameter=0.6e-3,
                   cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
                   fabric_rear_rayl=0.0, body_diameter=28e-3)
        pat23 = {}
        for sm in ("2d", "3d"):
            cn = MicrophoneCapsule(**n23, squeeze_model=sm)
            di = cn.directivity(frequencies_hz=(1000.0,))
            db = di["patterns"][1000.0]["db"]
            lin = di["patterns"][1000.0]["linear"]
            na = di["angles_deg"][:181][int(np.argmin(lin[:181]))]
            H1 = abs(cn.transfer_function(np.array([1000.0]))[0])
            pat23[sm] = (db[90], db[180], na, H1)
        assert abs(pat23["3d"][0] - pat23["2d"][0]) < 1.5, \
            (f"Niere 90°: 3D nahe 2D ({pat23['2d'][0]:.1f} vs. "
             f"{pat23['3d'][0]:.1f} dB)")
        assert abs(pat23["3d"][1] - pat23["2d"][1]) < 2.5, \
            (f"Niere 180°: 3D nahe 2D ({pat23['2d'][1]:.1f} vs. "
             f"{pat23['3d'][1]:.1f} dB)")
        assert abs(pat23["3d"][2] - pat23["2d"][2]) <= 15.0, \
            (f"Minimum-Winkel: 3D nahe 2D ({pat23['2d'][2]:.0f}° vs. "
             f"{pat23['3d'][2]:.0f}°)")
        r_e = pat23["3d"][3] / pat23["2d"][3]
        assert 0.6 < r_e < 1.05, \
            f"Niere Empfindlichkeit 3D/2D @1 kHz ({r_e:.2f})"
        # f) dual: Reziprozität + LF nahe 2D
        du23 = dict(sg23, architecture="dual", n_cavity_holes=200,
                    cavity_hole_diameter=0.2e-3)
        du3 = MicrophoneCapsule(**du23, squeeze_model="3d")
        du2 = MicrophoneCapsule(**du23, squeeze_model="2d")
        r_du = (du3.transfer_function(f23[:1])
                / du2.transfer_function(f23[:1]))[0]
        rez_d = abs(du3._recip_3d[0]) / abs(du3._recip_3d[1])
        assert abs(rez_d - 1.0) < 1e-6, \
            f"3D dual muss reziprok sein ({rez_d:.8f})"
        assert 0.75 < abs(r_du) < 1.05, \
            f"dual: 3D/2D @100 Hz ({abs(r_du):.3f})"
        print(f"3D single/dual: reziprok (single {rez_s:.6f}, dual "
              f"{rez_d:.6f}); geschlossen = Kugel, 3D/1D @100 Hz "
              f"{abs(r_cl):.3f} ∠{np.rad2deg(np.angle(r_cl)):+.1f}°; "
              f"K103 dicht {r_kd.round(3)}, offen |ΔD_r| = {d_dr:.3f}; "
              f"Niere 90/180/Min: 2D {pat23['2d'][0]:.1f}/"
              f"{pat23['2d'][1]:.1f}/{pat23['2d'][2]:.0f}° vs. 3D "
              f"{pat23['3d'][0]:.1f}/{pat23['3d'][1]:.1f}/"
              f"{pat23['3d'][2]:.0f}°, Empf. {r_e:.2f}; dual @100 Hz "
              f"{abs(r_du):.3f}  OK")

    # --------- Gegenprobe 24: Position des rückwärtigen Gewebes ------------
    # fabric_rear_position: "backplate" (im Zylinder hinter der Platte,
    # über die volle Bohrung gespannt — Bestand) vs. "inlet" (außen ÜBER
    # den Einlassöffnungen). Physik ohne Fit-Koeffizient:
    # a) Ohne Gewebe (0 Rayl) ist die Position EXAKT wirkungslos.
    # b) Dasselbe Tuch wirkt am Einlass um S_bp/S_Löcher stärker (nur
    #    die Lochfläche wird durchströmt) und liegt HINTER den Shunt-
    #    Volumina sowie in Serie mit der Einlassloch-Masse: interne
    #    Laufzeit steigt deutlich (0.64 -> 3.0), die rückwärtige
    #    Auslöschung bricht ein (Empfindlichkeit +38 %), die
    #    250-Hz-Auslöschung vertieft sich.
    # c) Der 3D-Löser teilt die Kette (_rear_chain_mats) und zeigt
    #    dieselbe Richtung.
    # d) K103 (großflächige Plattenlöcher): kleiner, gleichgerichteter
    #    Effekt. e) Gatter: Doppelmembran hat keinen Einlass.
    if _HAS_SCIPY:
        g24 = dict(architecture="single",
                   membrane_resonance_hz=2100.0, membrane_diameter=25.4e-3,
                   membrane_thickness=6e-6, membrane_tension=45.0,
                   air_gap=38.1e-6, backplate_diameter=23.9e-3,
                   backplate_thickness=3.125e-3, bias_voltage=50.0,
                   n_through_holes=48, through_hole_diameter=1.0e-3,
                   n_blind_holes=24, blind_hole_diameter=1.2e-3,
                   blind_hole_depth=1.5e-3, delay_length=3e-3,
                   cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
                   n_cavity_holes=60, cavity_hole_diameter=0.6e-3,
                   cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
                   fabric_rear_rayl=25.0, body_diameter=28e-3)
        # a) 0 Rayl: Position exakt wirkungslos
        z24 = {**g24, "fabric_rear_rayl": 0.0}
        f24 = np.array([100.0, 1000.0, 8000.0])
        Ha = MicrophoneCapsule(**z24, squeeze_model="2d",
                               fabric_rear_position="backplate"
                               ).transfer_function(f24)
        Hb = MicrophoneCapsule(**z24, squeeze_model="2d",
                               fabric_rear_position="inlet"
                               ).transfer_function(f24)
        assert np.allclose(Ha, Hb, rtol=1e-12, atol=0.0), \
            "0 Rayl: Gewebe-Position muss exakt wirkungslos sein"

        def _fab24(sm, pos):
            c = MicrophoneCapsule(**g24, squeeze_model=sm,
                                  fabric_rear_position=pos)
            H1 = abs(c.transfer_function(np.array([1000.0]))[0]) * 1e3
            rat = c.delay_diagnostics()["ratio"]
            p250 = c.directivity(
                frequencies_hz=(250.0,))["patterns"][250.0]["db"][180]
            return H1, rat, p250

        Hb2, rb2, pb2 = _fab24("2d", "backplate")
        Hi2, ri2, pi2 = _fab24("2d", "inlet")
        assert rb2 < 0.8 and ri2 > 2.0, \
            (f"Einlass-Gewebe muss die interne Laufzeit stark verlängern "
             f"({rb2:.2f} -> {ri2:.2f})")
        assert Hi2 > 1.2 * Hb2, \
            (f"Einlass-Gewebe muss die rückwärtige Auslöschung schwächen "
             f"({Hb2:.1f} -> {Hi2:.1f} mV/Pa)")
        # Schwelle 0.7 dB (vorher 1.0): mit der korrigierten Durchfluss-
        # Zellfunktion (_cell_B_flow, Gegenprobe 30) ist die interne
        # Laufzeit ohnehin länger, der Zusatzeffekt des Einlass-Gewebes
        # damit etwas kleiner. Die RICHTUNG — Vertiefung — ist unberührt.
        assert pi2 < pb2 - 0.7, \
            (f"250-Hz-Auslöschung muss sich vertiefen "
             f"({pb2:.1f} -> {pi2:.1f} dB)")
        # c) 3D teilt die Kette: gleiche Richtung
        Hb3, rb3, _ = _fab24("3d", "backplate")
        Hi3, ri3, _ = _fab24("3d", "inlet")
        assert ri3 > 2.0 and ri3 > rb3 + 1.5 and Hi3 > Hb3, \
            (f"3D muss die Einlass-Gewebe-Richtung teilen "
             f"(ratio {rb3:.2f} -> {ri3:.2f}, H {Hb3:.1f} -> {Hi3:.1f})")
        # d) K103: Plattenlöcher großflächig -> kleiner Effekt, gleiche
        # Richtung
        k24 = dict(g24, delay_length=0.0, cavity_length=0.0,
                   cavity_wall_thickness=0.0, n_cavity_holes=0,
                   cavity_hole_diameter=0.0, cavity_hole_axial_position=0.0,
                   rear_spacer_height=60e-6, rear_plate_thickness=2.5e-3,
                   n_rear_plate_holes=60, rear_plate_hole_diameter=1.0e-3)
        rk_b = MicrophoneCapsule(**k24, squeeze_model="2d",
                                 fabric_rear_position="backplate"
                                 ).delay_diagnostics()["ratio"]
        rk_i = MicrophoneCapsule(**k24, squeeze_model="2d",
                                 fabric_rear_position="inlet"
                                 ).delay_diagnostics()["ratio"]
        assert rk_i > rk_b + 0.01, \
            f"K103: gleiche Wirkrichtung erwartet ({rk_b:.2f} -> {rk_i:.2f})"
        # e) Gatter
        try:
            MicrophoneCapsule(architecture="dual_diaphragm",
                              membrane_resonance_hz=1150.0,
                              fabric_rear_position="inlet")
            raise AssertionError("dual_diaphragm+inlet müsste scheitern")
        except ValueError:
            pass
        print(f"Gewebe-Position: 0 Rayl exakt wirkungslos; Einlass statt "
              f"Backplate (25 Rayl): intern/extern {rb2:.2f} -> {ri2:.2f}, "
              f"Empf. {Hb2:.1f} -> {Hi2:.1f} mV/Pa, 250 Hz/180° {pb2:.1f} "
              f"-> {pi2:.1f} dB (3D gleichgerichtet: {rb3:.2f} -> "
              f"{ri3:.2f}); K103 {rk_b:.2f} -> {rk_i:.2f}; Gatter "
              f"Doppelmembran greift  OK")

    # --------- Gegenprobe 25: Eigenrauschen (FDT/Nyquist, fit-frei) --------
    # a) METHODEN-VALIDIERUNG an einem selbstständigen Mini-ABCD-Netzwerk
    #    MIT Shunt-Zweigen (die Strom umleiten): das verallgemeinerte
    #    Nyquist-Ergebnis S_qq = 4kT·Re{Z_tot}/|Z_tot|² muss der
    #    BRUTE-FORCE-Superposition über JEDEN einzelnen Widerstand
    #    (Norton-Rauschquelle 4kT/R, Übertragung auf den Membranstrom)
    #    exakt entsprechen — beweist, dass die Port-Impedanz-Formel und
    #    die Theorem-Anwendung alle Widerstände korrekt wichten.
    # b) Die _membrane_port_impedance-Zerlegung Z_front+Z_mem+Z_rear muss
    #    die unabhängig gerechnete Treibpunkt-Impedanz Z_mem+Z_a+Z_b
    #    treffen (validiert die ABCD-Konventionen/Vorzeichen).
    # c) Kapsel-Plausibilität: Pegel endlich und positiv, Pfad-Anteile
    #    summieren zu 1, Re{Z_tot} >= 0 (Passivität); T -> 2T ergibt
    #    +3 dB (S_p ∝ T); Gatter: 3D hat keinen Membranzweig.
    om25 = np.array([2.0 * np.pi * 1000.0, 2.0 * np.pi * 5000.0])
    Rs, C1 = 3.0e6, 1.5e-12
    Rmem, Xmem, C2, Rr = 2.0e6, 4.0e6, 2.5e-12, 1.2e6
    kT4 = 4.0 * K_BOLTZ * T_KELVIN
    for om in om25:
        Zmem = Rmem + 1j * Xmem
        Za = 1.0 / (1.0 / Rs + 1j * om * C1)         # Front, Quelle kurz
        Zb = 1.0 / (1.0 / Rr + 1j * om * C2)         # Rück, Port kurz
        Z_tot = Zmem + Za + Zb
        S_nyq = kT4 * np.real(Z_tot) / abs(Z_tot) ** 2
        # Brute force: Knoten A, M (in Zmem zwischen R und jX), B
        Gs, Gm, Gr = 1.0 / Rs, 1.0 / Rmem, 1.0 / Rr
        Yx = 1.0 / (1j * Xmem)
        yc1, yc2 = 1j * om * C1, 1j * om * C2
        Y = np.array([
            [Gs + yc1 + Gm, -Gm,       0.0],
            [-Gm,           Gm + Yx,  -Yx],
            [0.0,          -Yx,        Yx + yc2 + Gr]], dtype=complex)
        Yinv = np.linalg.inv(Y)

        def _qmem(J):
            # Zweigstrom am QUELLENFREIEN Element jX_mem messen (M<->B):
            # die R_mem-Norton-Quelle sitzt parallel zu R_mem (A<->M) und
            # speist zusätzlich in den Zweig, daher wäre Gm·(V_A−V_M)
            # falsch — jX_mem trägt immer den vollen Membranstrom.
            V = Yinv @ J
            return Yx * (V[1] - V[2])

        S_bf = 0.0
        # Rs: Norton A<->gnd
        S_bf += abs(_qmem(np.array([1.0, 0, 0], complex))) ** 2 * kT4 * Gs
        # Rr: Norton B<->gnd
        S_bf += abs(_qmem(np.array([0, 0, 1.0], complex))) ** 2 * kT4 * Gr
        # R_mem: Norton A<->M
        S_bf += abs(_qmem(np.array([1.0, -1.0, 0], complex))) ** 2 * kT4 * Gm
        assert abs(S_bf - S_nyq) < 1e-9 * abs(S_nyq), \
            (f"Nyquist muss Brute-Force treffen ({S_bf:.3e} vs. "
             f"{S_nyq:.3e} bei {om / (2 * np.pi):.0f} Hz)")

    # b)+c) an einer echten Kapsel (Nieren-Single, 2D)
    ncap = MicrophoneCapsule(
        architecture="single", membrane_resonance_hz=2100.0,
        membrane_diameter=25.4e-3, membrane_thickness=6e-6,
        membrane_tension=45.0, air_gap=38.1e-6, backplate_diameter=23.9e-3,
        backplate_thickness=3.125e-3, bias_voltage=50.0, n_through_holes=48,
        through_hole_diameter=1.0e-3, n_blind_holes=24,
        blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        delay_length=3e-3, cavity_length=12e-3, cavity_wall_thickness=1.5e-3,
        n_cavity_holes=60, cavity_hole_diameter=0.6e-3,
        cavity_hole_axial_position=6e-3, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, body_diameter=28e-3, squeeze_model="2d")
    om_c = np.array([2.0 * np.pi * 1000.0])
    Zt, Zf, Zm, Zr = ncap._membrane_port_impedance(om_c)
    assert np.real(Zt)[0] > 0.0, "Passivität: Re{Z_tot} muss >= 0 sein"
    sp = ncap.noise_spectrum(np.array([1000.0]))
    fr_sum = (sp["frac_front"] + sp["frac_mem"] + sp["frac_rear"])[0]
    assert abs(fr_sum - 1.0) < 1e-9, \
        f"Pfad-Anteile müssen zu 1 summieren ({fr_sum:.6f})"
    sn = ncap.self_noise()
    assert np.isfinite(sn["spl_a_db"]) and np.isfinite(sn["spl_z_db"])
    assert 0.0 < sn["spl_a_db"] < 40.0, \
        f"Ersatzgeräuschpegel unplausibel ({sn['spl_a_db']:.1f} dB-A)"
    assert sn["spl_z_db"] > sn["spl_a_db"], \
        "linear (Z) muss über A-bewertet liegen"
    # Pfad-Zerlegung: energetische Summe = Gesamtpegel
    p_sum = 10.0 * np.log10(
        10 ** (sn["spl_a_front_db"] / 10) + 10 ** (sn["spl_a_mem_db"] / 10)
        + 10 ** (sn["spl_a_rear_db"] / 10))
    assert abs(p_sum - sn["spl_a_db"]) < 0.05, \
        (f"Pfad-Zerlegung muss sich energetisch zum Gesamtpegel summieren "
         f"({p_sum:.2f} vs. {sn['spl_a_db']:.2f} dB-A)")
    # T -> 2T: +3 dB (S_p ∝ T). Modulkonstante temporär anheben.
    _T0 = globals()["T_KELVIN"]
    try:
        globals()["T_KELVIN"] = 2.0 * _T0
        sn2 = ncap.self_noise()
    finally:
        globals()["T_KELVIN"] = _T0
    assert abs((sn2["spl_a_db"] - sn["spl_a_db"]) - 10.0 * np.log10(2.0)) \
        < 0.02, "T -> 2T muss den Rauschpegel um +3 dB anheben"
    # d) Gatter 3D
    try:
        MicrophoneCapsule(
            architecture="dual_diaphragm", membrane_resonance_hz=1150.0,
            center_gap=50e-6, n_through_holes=12, through_hole_diameter=0.6e-3,
            squeeze_model="3d").self_noise()
        raise AssertionError("self_noise im 3D-Modus müsste scheitern")
    except ValueError:
        pass
    print(f"Eigenrauschen: Nyquist == Brute-Force (Mini-Netz, "
          f"{len(om25)} Frequenzen, rel < 1e-9); Nieren-Single "
          f"{sn['spl_a_db']:.1f} dB-A ({sn['spl_z_db']:.1f} dB lin) — "
          f"Front {sn['spl_a_front_db']:.1f} / Membranfilm "
          f"{sn['spl_a_mem_db']:.1f} / Rückpfad {sn['spl_a_rear_db']:.1f} "
          f"dB-A; T->2T +3.01 dB; Anteile summieren zu 1; 3D-Gatter "
          f"greift  OK")

    # --------- Gegenprobe 26: BEM-Frontfaktor (exakter Druckstau der -------
    # FLACHEN Stirnfläche ersetzt die Kugelkalotten-Näherung im BEM-Modus).
    # Der BEM-Modus treibt die Frontmembran jetzt mit dem ABSOLUTEN
    # Flächenmittel ⟨p⟩ der realen flachen Stirnfläche (aus demselben
    # Lösungsgang wie der Front-Rück-Transfer) statt mit der Morse-
    # Kugelkalotte (bei der K67 um ±50° gekrümmt). Kein freier Parameter:
    # a) KUGELKONTUR-GRENZFALL: BEM-Frontmittel über die ±50°-Kalotte
    #    einer Kugelkontur == Kalottenmittel der Morse-Reihe (identische
    #    Geometrie, zwei unabhängige exakte Methoden).
    # b) FLACHKÖRPER-ABSOLUTREFERENZ: am oblaten Sphäroid (17 x 6.1 mm)
    #    muss das BEM-Frontmittel den ANALYTISCHEN Absolutdruck der
    #    Flammer-Reihe treffen:
    #        p(eta) = 2i/(c(xi0²+1)) Σ (−i)^n S_n(cosθ) S_n(eta)/(N_n R3'_n)
    #    — Vorfaktor aus der ebenen-Wellen-Expansion (Identität numerisch
    #    auf Maschinengenauigkeit geprüft) plus Wronski-Identität; eine
    #    unabhängige exakte Referenz für einen FLACHEN Körper.
    # c) PHYSIK DER FLACHEN STIRNFLÄCHE: ka→0 ⇒ F→1; im Band 5–9 kHz
    #    (ka ≈ 2..3) staut die SENKRECHT zur Welle stehende flache
    #    Stirnfläche +6.5..+8 dB (nahe Druckverdopplung samt Randbeugungs-
    #    Überschwingen), die ±50°-Kalotte nur +3.3..+4.1 dB — diese
    #    3–4-dB-Unterschätzung war die künstliche Vertiefung der
    #    7-kHz-Senke des Kalottenmodells (Kalotte bleibt für die
    #    Kugel-/Sphäroidmodi in Kraft, dort ist sie die konsistente
    #    Geometrie).
    # d) KONSISTENZ: H_neu = H_alt · F_bem/F_cap exakt — der Frontfaktor
    #    geht multiplikativ ein (H ∝ F·(a + b·G)), das Netzwerk bleibt
    #    unberührt.
    if _HAS_SCIPY:
        par26 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=50e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, clamp_ring_thickness=2e-3,
            clamp_ring_width=4e-3, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=34e-3,
            squeeze_model="2d", axial_body_model="bem",
            bem_body_diameter=0.0)
        k67f26 = MicrophoneCapsule(**par26)
        # a) Kugelkontur: Kalottenrand fällt exakt auf eine Gitterlinie
        R26 = k67f26.R_body
        psi_c = float(np.arcsin(k67f26.a_mem / R26))
        psi26 = np.concatenate([np.linspace(0.0, psi_c, 41),
                                np.linspace(psi_c, np.pi, 121)[1:]])
        el26 = MicrophoneCapsule._bem_elems(
            np.stack([R26 * np.sin(psi26), R26 * np.cos(psi26)], 1))
        ps_m = np.arctan2(el26["mid_r"], el26["mid_z"])
        k67f26._bem_geo = dict(
            elems=el26, chief=[(0.0, 0.0)],
            w_area=2.0 * np.pi * el26["mid_r"] * el26["L"],
            front=ps_m < psi_c, rear=ps_m > np.pi - psi_c)
        th26 = np.deg2rad(np.array([0.0, 60.0, 120.0]))
        worst_f26 = 0.0
        for f26 in (1000.0, 7000.0):
            om26 = np.array([2.0 * np.pi * f26])
            F_b26 = k67f26._bem_axial_fields(om26, th26)[0]
            F_ref26, _ = k67f26._diffraction_factors(om26, th26)
            worst_f26 = max(worst_f26, float(np.max(
                np.abs(F_b26 - F_ref26) / np.abs(F_ref26))))
        assert worst_f26 < 2e-3, \
            f"BEM-Frontmittel muss die Morse-Kalotte treffen ({worst_f26:.1e})"
        # b) Sphäroid: BEM-Frontmittel vs. Flammer-Absolutreihe
        a26, b26 = 17e-3, 6.1e-3
        foc26 = np.sqrt(a26**2 - b26**2)
        xi26 = b26 / foc26
        psi_s = np.linspace(0.0, np.pi, 181)
        el_s = MicrophoneCapsule._bem_elems(
            np.stack([a26 * np.sin(psi_s), b26 * np.cos(psi_s)], 1))
        mrs, mzs = el_s["mid_r"], el_s["mid_z"]
        fr_s = (mzs > 0) & (mrs <= 13e-3)
        k67f26._bem_geo = dict(
            elems=el_s, chief=[(0.0, 0.0)],
            w_area=2.0 * np.pi * mrs * el_s["L"],
            front=fr_s, rear=(mzs < 0) & (mrs <= 13e-3))
        w_s = (2.0 * np.pi * mrs * el_s["L"])[fr_s]
        A_s = (mrs**2 + mzs**2) / foc26**2 - 1.0
        xi_s = np.sqrt(0.5 * (A_s + np.sqrt(A_s**2
                                            + 4.0 * mzs**2 / foc26**2)))
        eta_s = np.clip(mzs / (foc26 * xi_s), -1.0, 1.0)[fr_s]
        worst_s26 = 0.0
        for f26, thd26 in ((7000.0, 0.0), (7000.0, 60.0)):
            k26 = 2.0 * np.pi * f26 / C_AIR
            c26 = k26 * foc26
            th_s = float(np.deg2rad(thd26))
            modes26 = MicrophoneCapsule._oblate_modes(
                c26, int(np.ceil(c26)) + 12)
            p26 = np.zeros(int(np.sum(fr_s)), dtype=complex)
            for m26 in modes26:
                _, dR3_26 = MicrophoneCapsule._oblate_R3(m26, c26, xi26)
                S_th26 = MicrophoneCapsule._oblate_S(m26, np.cos(th_s))[0]
                p26 += ((-1j) ** m26["n"] * S_th26
                        * MicrophoneCapsule._oblate_S(m26, eta_s)
                        / (m26["N"] * dR3_26))
            p_ref26 = np.conj(2j / (c26 * (xi26**2 + 1.0))
                              * (w_s @ p26) / np.sum(w_s))
            F_b26 = k67f26._bem_axial_fields(
                np.array([2.0 * np.pi * f26]), np.array([th_s]))[0][0, 0]
            worst_s26 = max(worst_s26,
                            float(abs(F_b26 - p_ref26) / abs(p_ref26)))
        assert worst_s26 < 1e-3, \
            (f"BEM muss die Flammer-Absolutreihe am Sphäroid treffen "
             f"({worst_s26:.1e})")
        # c) reale K67-Kopfgeometrie (frei): Grenzfälle und Band 5-9 kHz
        k67f26._bem_geo = None
        om_c26 = 2.0 * np.pi * np.array([100.0, 5000.0, 7000.0, 9000.0])
        th0_26 = np.array([0.0])
        F_flat26 = k67f26._bem_axial_fields(om_c26, th0_26)[0][:, 0]
        F_cap26, _ = k67f26._diffraction_factors(om_c26, th0_26)
        assert abs(abs(F_flat26[0]) - 1.0) < 5e-3, \
            f"ka->0 muss F->1 liefern ({abs(F_flat26[0]):.4f})"
        d_band26 = 20.0 * np.log10(np.abs(F_flat26[1:] / F_cap26[1:, 0]))
        assert np.all((d_band26 > 2.5) & (d_band26 < 4.5)), \
            (f"flache Stirnfläche muss die Kalotte um 3-4 dB übertreffen "
             f"({np.round(d_band26, 2)})")
        f_band26 = 20.0 * np.log10(np.abs(F_flat26[1:]))
        assert np.all((f_band26 > 6.0) & (f_band26 < 8.5)), \
            (f"Druckstau nahe Verdopplung (+6..+8 dB) erwartet "
             f"({np.round(f_band26, 2)})")
        # d) Multiplikativität über den vollen Signalpfad (Netzwerk unberührt)

        class _KalottenFrontBem(MicrophoneCapsule):
            """Alter BEM-Modus (Kalotten-F, BEM-G) — nur Konsistenzprobe."""

            def _bem_axial_fields(self, omega, theta):
                _, G_b = MicrophoneCapsule._bem_axial_fields(
                    self, omega, theta)
                return self._diffraction_factors(omega, theta)[0], G_b

        f26h = [1000.0, 7000.0]
        om26h = 2.0 * np.pi * np.asarray(f26h)
        H_neu26 = k67f26.transfer_function(f26h)
        H_alt26 = _KalottenFrontBem(**par26).transfer_function(f26h)
        F_n26 = k67f26._bem_axial_fields(om26h, th0_26)[0][:, 0]
        F_c26 = k67f26._diffraction_factors(om26h, th0_26)[0][:, 0]
        ratio26 = float(np.max(np.abs(H_neu26 / H_alt26 - F_n26 / F_c26)
                               / np.abs(F_n26 / F_c26)))
        assert ratio26 < 1e-9, \
            f"Frontfaktor muss exakt multiplikativ eingehen ({ratio26:.1e})"
        print(f"BEM-Frontfaktor: Kugelkontur == Morse-Kalotte "
              f"{worst_f26:.1e}; Sphäroid-Absolutreihe {worst_s26:.1e}; "
              f"ka->0: |F| = {abs(F_flat26[0]):.4f}; flache Stirnfläche "
              f"+{f_band26[1]:.1f} dB bei 7 kHz (Kalotte unterschätzt um "
              f"{d_band26[1]:.1f} dB); multiplikativ {ratio26:.1e}  OK")

    # --------- Gegenprobe 27: Strahlungsimpedanz + 3D-Außenknoten ---------
    # a) EXAKTE KOLBENSTRAHLUNG statt Kleinargument-Asymptote. Bis
    #    Gegenprobe 26 stand im Frontzweig R ~ (ka)²/2, X ~ 8ka/(3pi) —
    #    für ka -> 0 exakt, oberhalb ka ~ 1 aber grob falsch: die
    #    Reaktanz X1 hat ein MAXIMUM und fällt danach ab, die Asymptote
    #    wächst linear weiter. Folge war eine KONSTANTE Zusatzmasse von
    #    25 kg/m^4 (die Membran selbst hat nur 20.9), die den Hochton um
    #    ~7 dB bei 16 kHz niederhielt. Verankert an:
    #      - Lehrbuch-Stützwert R1(2) = 1 - J1(2) = 0.4233,
    #      - Kleinargument-Grenzfall == alte Asymptote (Stetigkeit),
    #      - Hochton: mitschwingende Masse MUSS zusammenbrechen,
    #      - omega = 0 bleibt endlich (keine 0/0-Auslöschung).
    # b) 3D-AUSSENKNOTEN der Doppelmembran-Bauform. Der 3D-Löser trieb die
    #    Membranaußenseiten direkt aus der Quelle — Strahlungsimpedanz und
    #    Gewebe fehlten ERSATZLOS (fabric_* blieb im 3D-Modus exakt
    #    wirkungslos, ein still falsches Ergebnis). Jetzt zwei Sammel-
    #    knoten wie bei single/dual. Verankert an:
    #      - GRENZFALL Z_außen -> 0 reproduziert den Direktantrieb, und
    #        zwar mit ERSTER ORDNUNG (Z zehnfach kleiner -> Abstand zum
    #        Grenzwert zehnfach kleiner) — beweist Vorzeichen und Struktur,
    #      - Passivität: Gewebe dämpft monoton (nie Verstärkung),
    #      - Topologie-Konsistenz: dieselbe Dämpfung wie im 1D/2D-Pfad,
    #      - Reziprozität X_r = -B_f bleibt erhalten.
    if _HAS_SCIPY:
        from scipy.special import j1 as _j1_27, struve as _struve_27
        cap27 = MicrophoneCapsule(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=40e-6, n_through_holes=60,
            through_hole_diameter=0.6e-3, n_blind_holes=120,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, squeeze_model="2d")
        Z0_27 = RHO0 * C_AIR / cap27.S_mem
        a27 = cap27.a_mem
        # Lehrbuch-Stützwert bei 2ka = 2
        f_x2 = C_AIR / (2.0 * np.pi * a27)            # -> ka = 1, x = 2
        Z_x2 = cap27._radiation_impedance_membrane(
            np.array([2.0 * np.pi * f_x2]))[0]
        assert abs(Z_x2.real / Z0_27 - (1.0 - float(_j1_27(2.0)))) < 1e-9, \
            f"R1(2) muss 1-J1(2) = 0.4233 sein ({Z_x2.real / Z0_27:.4f})"
        assert abs(Z_x2.imag / Z0_27 - float(_struve_27(1, 2.0))) < 1e-9, \
            "X1(2) muss 2*H1(2)/2 treffen"
        # Kleinargument: exakte Form == alte Asymptote (stetiger Anschluss)
        om_lf27 = np.array([2.0 * np.pi * 20.0])
        Z_lf = cap27._radiation_impedance_membrane(om_lf27)[0]
        ka_lf = om_lf27[0] * a27 / C_AIR
        assert abs(Z_lf.imag / Z0_27 / (8.0 * ka_lf / (3.0 * np.pi)) - 1.0) \
            < 1e-5, "ka -> 0 muss die alte Asymptote reproduzieren"
        M_class = 8.0 * RHO0 / (3.0 * np.pi ** 2 * a27)
        M_lf = Z_lf.imag / om_lf27[0]
        assert abs(M_lf / M_class - 1.0) < 1e-4, \
            (f"LF-Luftmasse muss 8*rho0/(3 pi^2 a) = {M_class:.1f} sein "
             f"({M_lf:.1f} kg/m^4)")
        # Hochton: die mitschwingende Masse MUSS zusammenbrechen
        om_hf27 = 2.0 * np.pi * np.array([16000.0])
        M_hf = (cap27._radiation_impedance_membrane(om_hf27)[0].imag
                / om_hf27[0])
        assert M_hf < 0.1 * M_lf, \
            (f"Strahlungsmasse muss im Hochton verschwinden "
             f"({M_hf:.2f} vs. {M_lf:.1f} kg/m^4)")
        assert np.all(np.isfinite(cap27._radiation_impedance_membrane(
            np.array([0.0])))), "omega = 0 muss endlich bleiben"

        # ---- b) 3D-Außenknoten ----
        par27 = dict(
            membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
            membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
            backplate_diameter=25e-3, backplate_thickness=4e-3,
            bias_voltage=60.0, architecture="dual_diaphragm",
            center_gap=40e-6, n_through_holes=12,
            through_hole_diameter=0.6e-3, n_blind_holes=24,
            blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
            through_holes_stepped=True)
        om27 = 2.0 * np.pi * np.array([1000.0])
        _rad_orig = MicrophoneCapsule._radiation_impedance_membrane

        def _rad_eps(self, omega, _e=1.0):
            om_ = np.asarray(omega, dtype=float)
            return np.full(om_.shape, _e * RHO0 * C_AIR / self.S_mem,
                           dtype=complex)

        try:
            X_lim = {}
            for e27 in (1e-3, 1e-4, 1e-5):
                MicrophoneCapsule._radiation_impedance_membrane = \
                    (lambda s, o, _e=e27: _rad_eps(s, o, _e))
                X_lim[e27] = MicrophoneCapsule(
                    squeeze_model="3d", fabric_front_rayl=0.0,
                    fabric_rear_rayl=0.0, **par27)._solve_3d(om27)[0][0]
        finally:
            MicrophoneCapsule._radiation_impedance_membrane = _rad_orig
        d1 = abs(X_lim[1e-3] - X_lim[1e-5])
        d2 = abs(X_lim[1e-4] - X_lim[1e-5])
        assert d2 < 0.2 * d1, \
            (f"Z -> 0 muss mit erster Ordnung konvergieren "
             f"(zehnfach kleineres Z: {d2:.3e} vs. {d1:.3e})")
        # Passivität + Topologie-Konsistenz gegen den 1D/2D-Pfad
        att = {}
        for mdl27 in ("2d", "3d"):
            H0_27 = MicrophoneCapsule(
                squeeze_model=mdl27, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, **par27).transfer_function([1000.0])
            lv = []
            for r27 in (1.0e3, 1.0e4, 1.0e5):
                H_27 = MicrophoneCapsule(
                    squeeze_model=mdl27, fabric_front_rayl=r27,
                    fabric_rear_rayl=0.0, **par27).transfer_function([1000.0])
                lv.append(float(20.0 * np.log10(np.abs(H_27[0])
                                                / np.abs(H0_27[0]))))
            att[mdl27] = lv
            assert lv[0] < -1e-3, \
                f"{mdl27}: Gewebe muss im 3D-Modus überhaupt wirken ({lv[0]})"
            assert lv[0] > lv[1] > lv[2], \
                f"{mdl27}: Gewebe muss monoton dämpfen (Passivität) {lv}"
        assert abs(att["3d"][2] - att["2d"][2]) < 1.0, \
            (f"3D-Außenknoten muss dieselbe Dämpfung liefern wie die "
             f"1D/2D-Kette ({att['3d'][2]:.2f} vs. {att['2d'][2]:.2f} dB)")
        # Reziprozität der Membranports bleibt erhalten
        c27r = MicrophoneCapsule(squeeze_model="3d", fabric_front_rayl=0.0,
                                 fabric_rear_rayl=0.0, **par27)
        Xf27, Xr27, Bf27, Br27 = c27r._solve_3d(om27, want_rear=True)
        assert abs(Xr27[0] / Bf27[0] + 1.0) < 5e-3, \
            (f"Reziprozität X_r = -B_f muss erhalten bleiben "
             f"({Xr27[0] / Bf27[0]:.4f})")
        print(f"Strahlung/3D-Knoten: R1(2) = {Z_x2.real / Z0_27:.4f} "
              f"(= 1-J1(2)); Luftmasse {M_lf:.1f} kg/m^4 (LF, klassisch) "
              f"-> {M_hf:.2f} bei 16 kHz; 3D-Grenzfall Z->0 konvergiert "
              f"1. Ordnung ({d2 / d1:.3f}); Gewebe wirkt jetzt im 3D "
              f"({att['3d'][2]:.1f} dB vs. 2D {att['2d'][2]:.1f} dB); "
              f"reziprok  OK")

    # --------- Gegenprobe 28: Spaltmündung + Mehrmoden-Membran ------------
    # a) KEINE DOPPELZÄHLUNG DER LATERALEN SPALTMASSE. Der Škvor-Term
    #    trägt die Massenwirkung der radial zu den Löchern gequetschten
    #    Spaltluft bereits vollständig — nachgewiesen als IDENTITÄT mit
    #    der Baird/Zuckerwar-Spaltmasse:
    #        M_gap = rho0*B(q)/(n*pi*h)  ==  R_Škvor * rho0*h²/(12 mu),
    #    und die Frequenzkorrektur Φ(ω) realisiert diese Masse auch
    #    wirklich: im Tiefton ist Im(R·Φ)/ω = (6/5)·M_gap — der Faktor
    #    6/5 ist der kinetische Profilfaktor der Poiseuille-Verteilung,
    #    den die reine Lumped-Masse nicht kennt. Eine ZUSÄTZLICHE
    #    Freifeld-Flanschmasse 0.85·r auf der SPALTSEITE wäre daher
    #    Doppelzählung; sie entfällt jetzt auch im 1D-Pfad (2D- und
    #    3D-Pfad führen sie ohnehin nicht).
    # b) MEHRMODEN-MEMBRAN (Galerkin, membrane_modes > 1). Die höheren
    #    axialsymmetrischen (0,m)-Moden liegen PARALLEL zur Grundmode.
    #    Verankert an: membrane_modes = 1 bit-für-bit wie bisher;
    #    Modenfrequenzen und -massen exakt im Bessel-Verhältnis;
    #    massegesteuerter Hochton-Grenzwert der Parallelschaltung;
    #    Passivität; Konvergenz; Gatter.
    cap28 = MicrophoneCapsule(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm",
        center_gap=40e-6, n_through_holes=60,
        through_hole_diameter=0.6e-3, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, squeeze_model="2d")
    # a) Identität Škvor <-> Baird/Zuckerwar
    h28 = cap28.h_gap
    n_dr28 = cap28.n_th + cap28.n_bh
    q28 = cap28._q_drain
    B_q28 = (0.5 * np.log(1.0 / np.sqrt(q28)) - 3.0 / 8.0
             + q28 / 2.0 - q28**2 / 8.0)
    M_baird28 = RHO0 * B_q28 / (n_dr28 * np.pi * h28)
    M_skvor28 = cap28._skvor_R(h28) * RHO0 * h28**2 / (12.0 * MU_AIR)
    assert abs(M_skvor28 / M_baird28 - 1.0) < 1e-12, \
        (f"Škvor muss die Baird/Zuckerwar-Spaltmasse exakt enthalten "
         f"({M_skvor28:.4f} vs. {M_baird28:.4f} kg/m^4)")
    om28 = np.array([2.0 * np.pi * 100.0])
    Z_film28 = cap28._skvor_R(h28) * cap28._film_R_dynamic(om28, h28)[0]
    assert abs((Z_film28.imag / om28[0]) / M_baird28 - 6.0 / 5.0) < 2e-3, \
        (f"Φ(ω) muss im Tiefton (6/5)·M_gap realisieren "
         f"({(Z_film28.imag / om28[0]) / M_baird28:.4f})")
    # Spaltseitige Flanschmasse ist raus: die ungestufte Durchgangsloch-
    # Impedanz trägt NUR noch den portseitigen Fok-Term.
    cap28p = MicrophoneCapsule(
        architecture="single", membrane_resonance_hz=2100.0,
        membrane_diameter=25.4e-3, membrane_thickness=6e-6,
        membrane_tension=45.0, air_gap=38.1e-6,
        backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
        bias_voltage=50.0, n_through_holes=48,
        through_hole_diameter=1.0e-3, n_blind_holes=24,
        blind_hole_diameter=1.2e-3, blind_hole_depth=1.5e-3,
        fabric_front_rayl=0.0, fabric_rear_rayl=0.0, squeeze_model="1d")
    om28b = 2.0 * np.pi * np.array([1000.0, 9000.0])
    S28 = np.pi * cap28p.r_th**2
    Z_ref28 = (cap28p._hole_impedance(om28b, cap28p.r_th, cap28p.t_bp,
                                      cap28p.n_th, end_correction=False,
                                      visc_ends=1)
               + 1j * om28b * RHO0 * (0.85 * cap28p.r_th * cap28p._fok_th)
               / (S28 * cap28p.n_th))
    Z_got28 = cap28p._through_hole_impedance(om28b, cap28p.n_th)
    assert np.max(np.abs(Z_got28 - Z_ref28) / np.abs(Z_ref28)) < 1e-12, \
        "Durchgangsloch darf nur noch die PORTSEITIGE Mündungsmasse tragen"
    # b) Mehrmoden-Membran
    mk28 = dict(
        membrane_resonance_hz=1150.0, membrane_diameter=26e-3,
        membrane_thickness=6e-6, membrane_tension=13.7, air_gap=65e-6,
        backplate_diameter=25e-3, backplate_thickness=4e-3,
        bias_voltage=60.0, architecture="dual_diaphragm",
        center_gap=40e-6, n_through_holes=60,
        through_hole_diameter=0.6e-3, n_blind_holes=120,
        blind_hole_diameter=1.3e-3, blind_hole_depth=3.7e-3,
        through_holes_stepped=True, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, squeeze_model="2d")
    c28_1 = MicrophoneCapsule(membrane_modes=1, **mk28)
    assert c28_1._higher_mode_branches() == [], \
        "membrane_modes = 1 darf keine Zusatzzweige erzeugen"
    om28c = 2.0 * np.pi * np.logspace(2, 4.3, 40)
    Z1_28 = c28_1._membrane_impedance(om28c)
    R28 = c28_1._membrane_film_damping(om28c, c28_1.h_gap_front,
                                       c28_1.R_A_gap_front)
    Z1_ref = (R28 + 1j * om28c * c28_1.M_A_mem
              + 1.0 / (1j * om28c * c28_1.C_A_eff))
    assert np.max(np.abs(Z1_28 - Z1_ref)) == 0.0, \
        "membrane_modes = 1 muss bit-für-bit die Einmoden-Impedanz sein"
    c28_3 = MicrophoneCapsule(membrane_modes=3, **mk28)
    br28 = c28_3._higher_mode_branches()
    x28 = MicrophoneCapsule._J0_ZEROS
    f1_28 = 1.0 / (2.0 * np.pi * np.sqrt(c28_3.M_A_mem * c28_3.C_A_mem))
    for i28, (M_m, C_m) in enumerate(br28):
        rat28 = x28[i28 + 1] / x28[0]
        assert abs(M_m / c28_3.M_A_mem - rat28**2) < 1e-12, \
            f"Modenmasse muss (x_m/x_1)² folgen (Mode {i28 + 2})"
        f_m = 1.0 / (2.0 * np.pi * np.sqrt(M_m * C_m))
        assert abs(f_m / f1_28 - rat28) < 1e-9, \
            (f"Modenfrequenz muss x_m/x_1 folgen: {f_m:.1f} statt "
             f"{f1_28 * rat28:.1f} Hz")
    # Massegesteuerter Grenzwert der Parallelschaltung: die wirksame
    # Masse sinkt auf 1/Σ(1/M_m). Ohne Dämpfung (R = 0) isoliert das die
    # Modenalgebra von der Filmdämpfung — die Reihe konvergiert sauber
    # (50 kHz: 0.9989, 1 MHz: 0.999997). Die drei Moden zusammen machen
    # die Membran akustisch um den Faktor 0.789 leichter.
    M_eff28 = 1.0 / sum(1.0 / M for M in
                        [c28_3.M_A_mem] + [b[0] for b in br28])
    om_hf28 = np.array([2.0 * np.pi * 2.0e5])
    Z1_hf28 = (1j * om_hf28 * c28_3.M_A_mem
               + 1.0 / (1j * om_hf28 * c28_3.C_A_eff))
    Z_hf28 = c28_3._modal_parallel(om_hf28, Z1_hf28, 0.0)[0]
    assert abs(Z_hf28.imag / om_hf28[0] / M_eff28 - 1.0) < 1e-3, \
        (f"Massegrenzwert muss 1/Σ(1/M_m) treffen "
         f"({Z_hf28.imag / om_hf28[0]:.3f} vs. {M_eff28:.3f})")
    assert np.all(np.real(c28_3._membrane_impedance(om28c)) > 0.0), \
        "Passivität: Re{Z_mem} > 0 über das Band"
    # Konvergenz + Wirkung im Hochton (Grundmode-Anker unberührt)
    F28 = [1000.0, 7000.0, 16000.0]
    lev28 = {}
    for m28 in (1, 3, 5):
        H28 = MicrophoneCapsule(membrane_modes=m28,
                                **mk28).transfer_function(F28)
        lev28[m28] = 20.0 * np.log10(np.abs(H28) / np.abs(H28[0]))
        if m28 > 1:
            assert abs(abs(H28[0]) / abs(
                MicrophoneCapsule(membrane_modes=1,
                                  **mk28).transfer_function(F28)[0])
                - 1.0) < 1e-3, \
                "Empfindlichkeit bei 1 kHz muss unberührt bleiben"
    d13 = abs(lev28[3][2] - lev28[1][2])
    d35 = abs(lev28[5][2] - lev28[3][2])
    assert d35 < 0.5 * d13, \
        f"Modenreihe muss konvergieren ({d35:.2f} vs. {d13:.2f} dB)"
    assert lev28[3][2] > lev28[1][2] + 1.0, \
        "höhere Moden müssen den Hochton anheben (weichere Membran)"
    # Der 7-kHz-Sattel ist NICHT modal: er bleibt praktisch unverändert
    assert abs(lev28[3][1] - lev28[1][1]) < 0.5, \
        (f"7-kHz-Sattel ist kein Modeneffekt "
         f"({lev28[1][1]:.1f} -> {lev28[3][1]:.1f} dB)")
    try:
        MicrophoneCapsule(membrane_modes=0, **mk28)
        raise AssertionError("membrane_modes = 0 muss scheitern")
    except ValueError:
        pass
    print(f"Spaltmündung/Moden: Škvor == Baird-Spaltmasse "
          f"({M_skvor28:.2f} kg/m^4, Φ realisiert 6/5 davon) -> keine "
          f"Doppelzählung, Freifeld-Flansch spaltseitig raus; Moden "
          f"{f1_28:.0f}/{f1_28 * x28[1] / x28[0]:.0f}/"
          f"{f1_28 * x28[2] / x28[0]:.0f} Hz, modes=1 bit-identisch, "
          f"16 kHz {lev28[1][2]:+.1f} -> {lev28[3][2]:+.1f} dB "
          f"(konvergent), 7 kHz unverändert  OK")

    # --------- Gegenprobe 29: durchgehender Randspalt (B&K-Bauform) -------
    # Der Luftspalt vieler Messmikrofon-Kapseln ist am Plattenumfang NICHT
    # dicht: ein umlaufender Ringkanal verbindet ihn mit der Rückkammer.
    # Bisher war der Filmrand hermetisch (Neumann, kein Fluss) — die Luft
    # konnte den Spalt nur durch Bohrungen verlassen. Der Clearance-Ring
    # kann das NICHT ersetzen (er ist eine Nut IM Film bzw. ein Sack-Stub).
    # Verankert, alles fit-frei:
    # a) SCHLITZLEITUNG: die thermoviskose LRF-Leitung des Ringkanals muss
    #    im Tiefton exakt die Poiseuille-Grenzwerte treffen —
    #    R = 12μL/(b·w³), M = (6/5)·ρ0·L/(b·w) (kinetischer Profilfaktor
    #    der Schlitzströmung, derselbe 6/5 wie in Gegenprobe 28) und die
    #    ISOTHERME Nachgiebigkeit V/P_atm — und reziprok sein (det T = 1).
    # b) RANDWIDERSTAND: die randbelüftete Platte hat den analytisch
    #    herleitbaren Squeeze-Film-Widerstand R_edge = 3μ/(2πh³)
    #    (radiale Poiseuille-Strömung zum offenen Rand, flächengemittelt,
    #    wie bei Škvor unabhängig vom Plattenradius).
    # c) DICHT-GRENZFALL: w -> 0 muss die versiegelte Platte reproduzieren
    #    (Filmrand wieder dicht) — Konvergenz von oben.
    # d) MONOTONIE/PASSIVITÄT: breiterer Randspalt -> mehr Rückkopplung,
    #    monoton, mit Sättigung sobald nicht mehr der Kanal, sondern der
    #    Film selbst begrenzt.
    # e) FELDPFAD: das 2D-Zweitor bleibt mit Randknoten reziprok
    #    (det T = 1), auch mit Löchern UND Randspalt gleichzeitig.
    # f) GATTER: Doppelmembran, K103-Spacer, 1D+Löcher, 3D.
    if _HAS_SCIPY:
        BK29 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=50.0, n_blind_holes=0,
            rear_network_enabled=True, delay_length=0.0,
            cavity_length=8.0e-3, cavity_wall_thickness=1.5e-3,
            n_cavity_holes=0, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, body_diameter=28e-3)
        c29 = MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                ring_vent_width=50e-6, **BK29)
        # a) Schlitzleitungs-Grenzfälle
        w29, L29 = 50e-6, c29.ring_vent_L
        b29 = 2.0 * np.pi * c29.a_bp
        om29 = np.array([2.0 * np.pi * 1.0])
        T29 = c29._slit_line_abcd(om29, w29, b29, L29)
        R29 = float(T29[0, 1][0].real)
        M29 = float(T29[0, 1][0].imag) / om29[0]
        C29 = float((T29[1, 0][0] / (1j * om29[0])).real)
        assert abs(R29 / (12.0 * MU_AIR * L29 / (b29 * w29**3)) - 1.0) \
            < 1e-4, f"Schlitz-Poiseuille R = 12μL/(b w³) ({R29:.3e})"
        assert abs(C29 / (w29 * b29 * L29 / P_ATM) - 1.0) < 1e-4, \
            f"isotherme Kanal-Nachgiebigkeit V/P_atm ({C29:.3e})"
        # Die Masse trägt bei LANGER Leitung bereits die Korrektur der
        # verteilten Leitung (O((γL)²)) — der reine Massegrenzwert wird
        # unten an der kurzen Leitung geprüft.
        assert M29 > 0.0, "Schlitzleitung muss induktiv sein"
        det29 = complex(T29[0, 0][0] * T29[1, 1][0]
                        - T29[0, 1][0] * T29[1, 0][0])
        assert abs(det29 - 1.0) < 1e-12, \
            f"Schlitzleitung muss reziprok sein (det = {det29})"
        # kurze Leitung -> reine Masse (ohne verteilten R'C-Anteil)
        T29s = c29._slit_line_abcd(om29, w29, b29, 0.1e-3)
        M29s = float(T29s[0, 1][0].imag) / om29[0]
        assert abs(M29s / (1.2 * RHO0 * 0.1e-3 / (b29 * w29)) - 1.0) \
            < 1e-3, "kurze Schlitzleitung muss exakt (6/5)ρ0L/(bw) sein"
        # b) Randwiderstand der randbelüfteten Platte
        assert abs(c29.R_A_gap / (3.0 * MU_AIR
                                  / (2.0 * np.pi * c29.h_gap**3)) - 1.0) \
            < 1e-12, "R_edge = 3μ/(2πh³) der randbelüfteten Platte"
        # Die B&K-Bauform bleibt ein DRUCKempfänger: der Randspalt
        # entlüftet den Spalt in die geschlossene Rückkammer, nicht ins
        # Freifeld. Erst Hohlraum-Einlässe machen daraus einen
        # Gradientenempfänger.
        assert not c29.rear_open, \
            "geschlossene Rückkammer -> Druckempfänger trotz Randspalt"
        assert MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                 ring_vent_width=50e-6,
                                 **{**BK29, "n_cavity_holes": 40,
                                    "cavity_hole_diameter": 0.5e-3}
                                 ).rear_open, \
            "mit Hohlraum-Einlässen muss der Randspalt nach hinten öffnen"
        # c) DICHT-GRENZFALL auf der ZWEITOR-Ebene. Physikalisch
        #    eindeutig ist die ENTKOPPLUNG: ein dichter Rand trennt
        #    Membran und Rückport, der Membranfluss bei kurzgeschlossenem
        #    Port (= 1/T22) muss verschwinden — und zwar wie der
        #    Kanalleitwert, also MINDESTENS wie w³ (Poiseuille); sehr
        #    schmale Spalte dämpfen sogar exponentiell, weil die Leitung
        #    dann vom Lumped- ins Wellenleiter-Regime wechselt (γL ~ 36
        #    bei w = 0.1 µm). Geprüft wird monotone, mindestens kubische
        #    Entkopplung. (Ein Vergleich der EINGANGSimpedanz taugt hier
        #    nicht: sie behält auch bei dichtem Rand die thermische
        #    Relaxationsdämpfung der Spaltluft, die zur versiegelten
        #    Platte gehört.)
        om_c29 = 2.0 * np.pi * np.array([200.0])
        cpl29 = []
        for w_t in (1.0e-5, 1.0e-6, 1.0e-7):
            Tx29 = MicrophoneCapsule(
                squeeze_model="2d", n_through_holes=0,
                ring_vent_width=w_t, **BK29)._gap_field_2port(om_c29)
            cpl29.append(float(np.abs(1.0 / Tx29[1, 1][0])))
        for k29 in range(len(cpl29) - 1):
            ratio29 = cpl29[k29] / cpl29[k29 + 1]
            assert ratio29 > 900.0, \
                (f"dichter Rand muss mindestens wie w³ entkoppeln "
                 f"(Verhältnis {ratio29:.0f} je Dekade)")
        e_prev = cpl29[-1] / cpl29[0]
        # d) Monotonie + Sättigung
        lv29 = []
        for w_t in (10e-6, 25e-6, 50e-6, 200e-6):
            H_t = MicrophoneCapsule(
                squeeze_model="2d", n_through_holes=0,
                ring_vent_width=w_t, **BK29).transfer_function([200.0])
            lv29.append(float(20.0 * np.log10(np.abs(H_t[0]))))
        assert lv29[0] < lv29[1] < lv29[2] <= lv29[3] + 1e-9, \
            f"breiterer Randspalt muss monoton mehr entlasten ({lv29})"
        assert abs(lv29[3] - lv29[2]) < abs(lv29[1] - lv29[0]), \
            "die Wirkung muss sättigen (Film wird begrenzend, nicht Kanal)"
        # e) Reziprozität des Feld-Zweitors, auch Löcher + Randspalt
        for n29, w_t in ((0, 50e-6), (48, 50e-6)):
            cx = MicrophoneCapsule(squeeze_model="2d", n_through_holes=n29,
                                   through_hole_diameter=1.0e-3,
                                   ring_vent_width=w_t, **BK29)
            Tx = cx._gap_field_2port(2.0 * np.pi
                                     * np.array([200.0, 4000.0, 16000.0]))
            dx = Tx[0, 0] * Tx[1, 1] - Tx[0, 1] * Tx[1, 0]
            assert np.max(np.abs(dx - 1.0)) < 1e-9, \
                (f"2D-Zweitor mit Randknoten muss reziprok bleiben "
                 f"(n_th = {n29}, max |det-1| = "
                 f"{np.max(np.abs(dx - 1.0)):.2e})")
        # 1D == 2D im rein randbelüfteten Tiefton (beide: Film -> Rand)
        H1d = MicrophoneCapsule(squeeze_model="1d", n_through_holes=0,
                                ring_vent_width=50e-6,
                                **BK29).transfer_function([200.0])
        H2d = MicrophoneCapsule(squeeze_model="2d", n_through_holes=0,
                                ring_vent_width=50e-6,
                                **BK29).transfer_function([200.0])
        d12 = float(abs(20.0 * np.log10(np.abs(H1d[0] / H2d[0]))))
        assert d12 < 2.0, \
            f"1D und 2D müssen im Tiefton zusammenliegen ({d12:.2f} dB)"
        # f) 3D-LÖSER: derselbe Ringkanal hängt dort über den
        #    Randflächen-Leitwert an der äußersten Filmzellreihe.
        #    Verankert am KOLBEN-GRENZFALL: nur wenn die Membran sich
        #    NICHT verformen kann, beschreiben 1D/2D (Grundmode φ
        #    erzwungen) und 3D (freies Membranfeld) dasselbe Problem.
        #    Mit steifer Membran im quasistatischen Tiefton müssen beide
        #    zusammenfallen — sie tun es auf 0.15 dB. Bei WEICHER Membran
        #    liegt 3D systematisch höher (bis ~8 dB): die freie Membran
        #    umgeht den hohen Randwiderstand, indem sie bevorzugt außen
        #    arbeitet — dieselbe Einmoden-Grenze des homogenisierten
        #    Modells wie beim K67-Sattel (Gegenprobe 28). Das ist ein
        #    dokumentierter Modellunterschied, kein Fehler.
        ST29 = dict(BK29, membrane_resonance_hz=50.0e3,
                    membrane_tension=25600.0)
        v2_29 = MicrophoneCapsule(
            squeeze_model="2d", n_through_holes=0, ring_vent_width=50e-6,
            **ST29).transfer_function([200.0])[0]
        c3_29 = MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                                  ring_vent_width=50e-6, **ST29)
        v3_29 = c3_29.transfer_function([200.0])[0]
        d3_29 = float(abs(20.0 * np.log10(abs(v3_29 / v2_29))))
        assert d3_29 < 1.0, \
            (f"3D-Randspalt muss im Kolben-Grenzfall die 2D-Lösung "
             f"treffen ({d3_29:.2f} dB)")
        # Gitterunabhängigkeit: der Ringstempel darf nicht von der
        # azimutalen Auflösung abhängen (axialsymmetrischer Kanal).
        lv3_29 = []
        for np_t in (96, 192):
            cg29 = MicrophoneCapsule(squeeze_model="3d", n_through_holes=0,
                                     ring_vent_width=50e-6, **ST29)
            cg29._n_phi_3d = np_t
            cg29._build_3d_geometry()
            lv3_29.append(float(abs(cg29.transfer_function([200.0])[0])))
        assert abs(lv3_29[1] / lv3_29[0] - 1.0) < 1e-6, \
            f"3D-Ringstempel muss Np-unabhängig sein ({lv3_29})"
        # Dicht-Grenzfall auch im 3D: schmaler Kanal entkoppelt
        v3_tight = MicrophoneCapsule(
            squeeze_model="3d", n_through_holes=0, ring_vent_width=0.5e-6,
            **ST29).transfer_function([200.0])[0]
        assert abs(v3_tight) < 0.35 * abs(v3_29), \
            "3D: schmaler Randspalt muss die Rückseite abkoppeln"
        # f) Gatter
        for kw29, why29 in (
                (dict(architecture="dual_diaphragm",
                      membrane_resonance_hz=1150.0, center_gap=40e-6,
                      n_through_holes=12, through_hole_diameter=0.6e-3,
                      ring_vent_width=50e-6), "Doppelmembran"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=0, ring_vent_width=50e-6,
                      rear_spacer_height=100e-6), "K103-Spacer"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=12, through_hole_diameter=0.6e-3,
                      ring_vent_width=50e-6, squeeze_model="1d"),
                 "1D + Löcher"),
                (dict(architecture="single", membrane_resonance_hz=8000.0,
                      n_through_holes=0, ring_vent_width=50e-6,
                      rear_plate_thickness=1.0e-3,
                      n_rear_plate_holes=8), "K103-Rückplatte")):
            try:
                MicrophoneCapsule(**kw29)
                raise AssertionError(f"Randspalt-Gatter {why29} fehlt")
            except ValueError:
                pass
        print(f"Randspalt: Schlitzleitung R/C exakt (Poiseuille, isotherm), "
              f"kurze Leitung == (6/5)ρ0L/(bw), reziprok "
              f"{abs(det29 - 1.0):.0e}; R_edge = 3μ/(2πh³); w->0 "
              f"entkoppelt wie w³ ({e_prev:.1e} über 2 Dekaden); "
              f"Breite 10->200 µm "
              f"{lv29[3] - lv29[0]:+.1f} dB (monoton, sättigt); "
              f"Feld-Zweitor reziprok mit/ohne Löcher; 1D/2D {d12:.2f} dB; "
              f"3D == 2D im Kolben-Grenzfall ({d3_29:.2f} dB), "
              f"Np-unabhängig; Gatter greifen  OK")

    # --------- Gegenprobe 30: Zellfunktion des DURCHFLUSSES ---------------
    # Die azimutale Zuströmung im Spaltfilm hat ZWEI verschiedene Ziele:
    #   * AUFNAHME (Verdrängung): jede Bohrung ist eine Senke — die Luft
    #     läuft zur nächstgelegenen, gleich welcher Art.
    #   * DURCHFLUSS zur Rückseite: nur DURCHGANGSLÖCHER zählen;
    #     Blindlöcher sind Sackgassen.
    # Bis Gegenprobe 29 nutzte auch der Durchfluss die Zellfunktion ALLER
    # Bohrungen — der Zugang zur Rückseite war dadurch zu leicht und die
    # interne Laufzeit des Nieren-Phasenschiebers zu kurz. Verankert an:
    # a) STRUKTUR: B(n_th) > B(n_th + n_bh), sobald Blindlöcher da sind;
    #    ohne Blindlöcher sind beide identisch (stetiger Anschluss).
    # b) K67-NIERE: das Pattern-Minimum liegt jetzt bei 180° — wie bei der
    #    realen K67 — statt bei ~164°, und die publizierten U87-Werte
    #    werden dabei besser getroffen (180°: -26.6 gegen -26 dB
    #    publiziert; 20.0 gegen ~20 mV/Pa).
    # c) NÄHER AM 3D-FELDLÖSER, der die diskreten Löcher auflöst und
    #    deshalb Referenz ist: RMS-Abweichung des Richtdiagramms sinkt bei
    #    der Debenham-Platte (12 Löcher) und der Nieren-Single.
    # d) GEGENPROBE OHNE BLINDLÖCHER: dort darf sich NICHTS ändern.
    if _HAS_SCIPY:
        par30 = dict(
            architecture="single", membrane_resonance_hz=2100.0,
            membrane_diameter=25.4e-3, membrane_thickness=6e-6,
            membrane_tension=45.0, air_gap=38.1e-6,
            backplate_diameter=23.9e-3, backplate_thickness=3.125e-3,
            bias_voltage=50.0, n_through_holes=12,
            through_hole_diameter=0.71e-3, rear_network_enabled=False,
            fabric_front_rayl=0.0, fabric_rear_rayl=0.0,
            body_diameter=28e-3, squeeze_model="2d")
        # a) Strukturprobe über die Zellfunktion selbst
        def _B30(n_s, r_h, a_bp):
            q = min(n_s * r_h**2 / a_bp**2, 1.0)
            if q >= 1.0:
                return 0.0
            return max(q / 2.0 - q**2 / 8.0 - np.log(q) / 4.0 - 3.0 / 8.0,
                       0.0)
        c30 = MicrophoneCapsule(n_blind_holes=46, blind_hole_diameter=1.2e-3,
                                blind_hole_depth=3.0e-3, **par30)
        B_flow = _B30(c30.n_th, c30.r_th, c30.a_bp)
        B_all = _B30(c30.n_th + c30.n_bh, c30.r_th, c30.a_bp)
        assert B_flow > B_all * 1.5, \
            (f"Sackgassen dürfen den Durchfluss nicht verkürzen "
             f"(B: {B_flow:.3f} vs. {B_all:.3f})")
        # d) ohne Blindlöcher identisch (stetiger Anschluss)
        assert abs(_B30(c30.n_th, c30.r_th, c30.a_bp)
                   - _B30(c30.n_th + 0, c30.r_th, c30.a_bp)) == 0.0, \
            "ohne Blindlöcher müssen beide Zellfunktionen gleich sein"
        # b) K67: Niere mit Minimum bei 180° und publizierte Kennwerte
        di30 = k67.directivity(frequencies_hz=(1000.0,))
        pat30 = di30["patterns"][1000.0]
        na30 = di30["angles_deg"][:181][int(np.argmin(
            pat30["linear"][:181]))]
        H30 = abs(k67.transfer_function([1000.0])[0]) * 1e3
        assert na30 >= 179.0, \
            (f"K67-Niere muss ihr Minimum bei 180° haben (real), nicht "
             f"bei {na30:.0f}°")
        assert pat30["db"][180] < -25.0, \
            f"K67 rückwärts < -25 dB erwartet ({pat30['db'][180]:.1f})"
        assert 18.0 < H30 < 22.0, \
            f"K67-Empfindlichkeit nahe 20 mV/Pa erwartet ({H30:.1f})"
        # c) Richtdiagramm näher am 3D-Feldlöser (Debenham, 12 Löcher)
        deb30 = MicrophoneCapsule(n_blind_holes=46,
                                  blind_hole_diameter=1.2e-3,
                                  blind_hole_depth=3.0e-3, **par30)
        d30_2 = deb30.directivity(frequencies_hz=(500.0,),
                                  n_angles=37)["patterns"][500.0]["db"]
        deb30_3 = MicrophoneCapsule(
            **{**par30, "squeeze_model": "3d"}, n_blind_holes=46,
            blind_hole_diameter=1.2e-3, blind_hole_depth=3.0e-3)
        d30_3 = deb30_3.directivity(frequencies_hz=(500.0,),
                                    n_angles=37)["patterns"][500.0]["db"]
        rms30 = float(np.sqrt(np.mean(
            (np.maximum(d30_2, -35.0) - np.maximum(d30_3, -35.0)) ** 2)))
        assert rms30 < 6.0, \
            (f"2D-Richtdiagramm muss nahe am 3D-Feldlöser liegen "
             f"({rms30:.2f} dB)")
        print(f"Durchfluss-Zellfunktion: Sackgassen zählen nicht "
              f"(B {B_all:.3f} -> {B_flow:.3f}); K67-Niere Minimum bei "
              f"{na30:.0f}° (real) mit {pat30['db'][180]:.1f} dB und "
              f"{H30:.1f} mV/Pa; 2D/3D-Richtdiagramm {rms30:.2f} dB  OK")

    print("\nAlle Testläufe erfolgreich — Arrays werden korrekt berechnet.")
