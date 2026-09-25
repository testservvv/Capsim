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

import types
import warnings
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
    from scipy.special import yv as _bessely

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — Fallback auf Näherungsformeln
    _HAS_SCIPY = False

    def _solve_banded(l_and_u, ab, b):
        """Thomas-Algorithmus für das tridiagonale solve_banded-Format
        (1, 1) — damit der exakte statische Arbeitspunkt auch ohne SciPy
        rechnet (ab[0, j] = A[j−1, j], ab[1, j] = A[j, j],
        ab[2, j] = A[j+1, j])."""
        if tuple(l_and_u) != (1, 1):
            raise ValueError("Fallback kann nur tridiagonal (1, 1).")
        sup, dia, sub = ab[0, 1:], ab[1].astype(complex), ab[2, :-1]
        x = np.array(b, dtype=complex)
        n = dia.size
        d = dia.copy()
        for i in range(1, n):
            m = sub[i - 1] / d[i - 1]
            d[i] = d[i] - m * sup[i - 1]
            x[i] = x[i] - m * x[i - 1]
        x[-1] = x[-1] / d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - sup[i] * x[i + 1]) / d[i]
        if np.isrealobj(ab) and np.isrealobj(b):
            return x.real
        return x


def _j0_mode(x):
    """J0(x) für das Membran-Modengewicht (0 <= x <= z_0m).

    Mit SciPy exakt. Ohne SciPy über die Potenzreihe: für die Grundmode
    (x <= 2.405) konvergiert sie nach ~12 Gliedern auf Maschinen-
    genauigkeit. Für höhere Moden reicht das Argument bis z_05 = 14.93;
    dort braucht die Reihe mehr Glieder UND verliert durch Auslöschung
    Stellen (größtes Glied ~7e4 gegen J0 ~ 1e-2). 40 Glieder halten den
    Fehler unter 1e-9 — ausreichend für ein Gewicht, aber bewusst
    dokumentiert statt stillschweigend.
    """
    x = np.asarray(x, dtype=float)
    if _HAS_SCIPY:
        return _besselj(0, x)
    t = -0.25 * x * x
    term = np.ones_like(x)
    out = np.ones_like(x)
    for m in range(1, 41):
        term = term * t / (m * m)
        out = out + term
    return out


def _j1_mode(x):
    """J1(x) für die Modenintegrale (gleiche Konvention wie _j0_mode)."""
    x = np.asarray(x, dtype=float)
    if _HAS_SCIPY:
        return _besselj(1, x)
    t = -0.25 * x * x
    term = 0.5 * x
    out = term.copy()
    for m in range(1, 41):
        term = term * t / (m * (m + 1.0))
        out = out + term
    return out


def _ring_static_shape(u, u_i):
    """Statische Auslenkungsform einer (Ring-)Membran über u = (r/a)².

    Aus T·∇²w = −p mit w(a) = 0 und — bei MITTENTERMINIERUNG — zusätzlich
    w(r_i) = 0:

        w(r) = p/(4T)·[(a²−r²) + (a²−r_i²)·ln(r/a)/ln(a/r_i)]

    In der Modenkoordinate u = (r/a)², u_i = (r_i/a)² (dS = S·du):

        φ(u) = (1−u) − (1−u_i)·ln(u)/ln(u_i).

    ``u_i = 0`` liefert exakt die Parabel 1−u, also bitgleich den Stand
    ohne Mittenterminierung. Der Logarithmus ist der Grund dafür, dass
    schon ein winziger Mittenpfosten die Membran stark versteift: er
    verschwindet nicht wie u_i, sondern nur wie 1/ln(u_i).
    """
    u = np.asarray(u, dtype=float)
    if u_i <= 0.0:
        return 1.0 - u
    su = np.clip(u, u_i, 1.0)
    return (1.0 - su) - (1.0 - u_i) * np.log(su) / np.log(u_i)


def _ring_compliance_factor(rho):
    """C_T(Ring)/C_T(Kreis) = 1 − ρ⁴ + (1−ρ²)²/ln ρ, ρ = r_i/a.

    Geschlossene Form des Volumenintegrals über :func:`_ring_static_shape`
    (Gegenprobe 45 prüft sie gegen die numerische Quadratur). Der
    Grenzwert ρ → 0 ist 1, aber NICHT stetig differenzierbar: bei
    ρ = 0.01 stehen schon 0.783, bei ρ = 0.05 nur noch 0.668.
    """
    if rho <= 0.0:
        return 1.0
    return 1.0 - rho**4 + (1.0 - rho**2) ** 2 / np.log(rho)


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
        center_post_diameter : float
            Durchmesser der MITTENTERMINIERUNG [m]; 0 (Voreinstellung) =
            klassische randgespannte Kreismembran. Ein Kontaktstift oder
            Mittenbolzen legt die Membranmitte fest — sie ist dann keine
            Kreis-, sondern eine RINGMEMBRAN, und das ist kein kleiner
            Korrekturterm. Die statische Lösung von T∇²w = −p enthält bei
            zwei Rändern einen Logarithmus:

                C_T(Ring)/C_T(Kreis) = 1 − ρ⁴ + (1−ρ²)²/ln ρ,  ρ = r_i/a,

            und ein Logarithmus verschwindet nicht wie ρ². Schon ρ = 0.01
            — ein 0.26-mm-Stift auf einer 26-mm-Membran — nimmt 22 % der
            Nachgiebigkeit weg, hebt den Grundmoden-Eigenwert von 2.4048
            auf 2.8009 (bei fester Vorspannung also die Resonanz um 16 %)
            und die Pull-in-Spannung um 18 %. Betroffen sind:
            Nachgiebigkeit, Massenfaktor, wirksame Fläche, Eigenwerte und
            Modenformen (Ringmoden statt J0), Elektrostatik (Arbeitspunkt,
            Feder-Erweichung, Pull-in, C0), Spaltfilm-Profil und die
            Aperturmittelung der Beugung. Anker: J. E. Warren, JASA 58(3),
            733–740 (1975) gibt den kritischen Antriebsparameter für
            Kreis- (0.789) und Ringmembran (1.548 bei ρ = 0.1);
            Gegenprobe 45 prüft beides. Der 3D-Feldlöser führt die
            Ringmembran ebenfalls: sein Gitter beginnt am Pfostenrand,
            und die innerste Fläche — ohne Pfosten mit Radius 0, also
            stillschweigend die Achsenbedingung — wird zur eingespannten
            Wand (Gegenprobe 47). GRENZE: der Ringfaktor gilt für den
            VORSPANNUNGSANTEIL — eine biegesteife Platte wird abgewiesen.

    Backplate-System
        air_gap : float             — Luftspalt Membran/Backplate [m]
        backplate_diameter : float  — Backplate-Durchmesser [m]
        backplate_thickness : float — Backplate-Dicke [m]
        bias_voltage : float
            Polarisationsspannung [V] JE SPALT, nicht als Gesamtversorgung.
            Bei ``"dual"`` liegt sie damit an BEIDEN Spalten voll an (die
            Backplates auf ±U gegen die Membran, Gesamtversorgung also 2·U)
            — das ist die Gegentakt-Verschaltung, die den Sinn der Bauform
            ausmacht: der Wandlerkoeffizient verdoppelt sich. Der Preis ist,
            dass sich auch die Feder-Erweichung beider Spalte ADDIERT. Die
            Pull-in-Spannung steigt trotzdem, aber nur um den Faktor 1.3464
            (Gegenprobe 39): der Gewinn kommt allein daraus, dass sich die
            statischen Kräfte aufheben und die Membran im Ruhepunkt bleibt,
            statt wie bei ``"single"`` bis auf 0.44·h zu kriechen, wo der
            verkleinerte Spalt die Erweichung hochtreibt. Wer stattdessen
            EINE Versorgung symmetrisch auf beide Spalte teilen will (U/2
            je Spalt), setzt ``bias_voltage`` auf die Hälfte: die
            Empfindlichkeit fällt dann auf den Wert der Einzel-Backplate
            zurück, die Pull-in-Spannung liegt doppelt so hoch.
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
        body_length : float oder None
            AXIALE Länge des Kapselkörpers [m]. Wird NUR von
            ``axial_body_model='bem'`` gebraucht: dort ist der Körper ein
            verrundeter Zylinder ``body_diameter × body_length``, und die
            Membran liegt auf seiner FLACHEN Stirnfläche. Die Kugel-
            rechnung braucht sie nicht (eine Kugel hat nur einen
            Durchmesser), deshalb ``None`` als Voreinstellung. Für die
            Doppelmembran-Bauform wird stattdessen ``d_ext`` benutzt (die
            beiden Membranen sitzen auf den beiden Stirnflächen).
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
            realistisch). Alle Architekturen mit Durchgangslöchern oder
            Randspalt; Stufenbohrung bei ``dual_diaphragm`` nur mit
            ``center_gap > 0`` (K67-Typ). DEUTLICH langsamer (LU-
            Faktorisierung je Frequenzpunkt).
        grid_3d : str
            Auflösung des 3D-Gitters (nur ``squeeze_model="3d"``):
            ``"coarse"`` (Standard) oder ``"fine"`` — s.
            :meth:`_grid_3d_size`.
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

    # 3D-Löser: Kurzschlussleitwert der Lochmündungen relativ zum größten
    # Film-Flächenleitwert (Äquipotential-Mündung, s. _build_3d_geometry).
    # Numerischer Parameter, kein physikalischer: das Ergebnis muss davon
    # unabhängig sein (Gegenprobe 48 prüft 10³ gegen 10⁵).
    _EQUI_SHORT = 1.0e4

    # Feines 3D-Gitter (grid_3d='fine', s. _grid_3d_size, Gegenprobe 50):
    # Zellen je kleinstem Mündungsradius und Obergrenze der Zellen je Feld
    # (K67-Typ fein: rund 43 000 Zellen je Feld, 5 Felder).
    _GRID_FINE_CELLS = 2.0
    _GRID_FINE_MAX = 50000

    # Konturtreue Mündungen im 3D-Löser (Shortley–Weller, Gegenprobe 51):
    # Filmflächen am Mündungsrand mit dem wahren Abstand zur Kreiskontur.
    # _SW_CAP begrenzt den Faktor Δ/ℓ (Filmzellmitte fast auf der Kontur,
    # sich fast berührende Mündungen); der Kurzschluss G_s wird mit dem
    # größten Faktor mitskaliert. Abschaltbar nur für den Vergleich.
    _SHORTLEY_WELLER = True
    _SW_CAP = 20.0

    # Membranring außerhalb der Platte (a_bp < r < a_mem) im 3D-Löser an
    # den Ringraum gekoppelt (Gegenprobe 52). Abschaltbar nur für den
    # Vergleich.
    _ANNULUS_COUPLED = True

    # Statischer Versatz 2D/3D (Gegenprobe 54): der 3D-Löser wandelt sein
    # Auslenkungsfeld mit demselben Elektrodenintegral wie Θ der Kette in
    # Spannung (s. _output_weight_3d), und seine Membranspannung ist bei
    # vorgegebener Vorspannung die physikalische (s. _membrane_tension_3d).
    # Abschaltbar nur für den Vergleich.
    _OUTPUT_EXACT_3D = True
    _TENSION_EXACT_3D = True

    # Wiederverwendung der 3D-Feldlösung je Frequenz (Gegenprobe 56):
    # directivity, transfer_function, delay_diagnostics und die Diagnosen
    # lösen bei gleicher Frequenz dasselbe System — es wird nur einmal
    # faktorisiert. Abschaltbar nur für den Vergleich.
    _REUSE_3D = True

    # Massenfaktor der Kette (Gegenprobe 55): 8/(z1²·g) statt des
    # Rayleigh-Werts der statischen Form (Parabel 4/3). Damit sind
    # statische Nachgiebigkeit UND Grundresonanz exakt, s. _derive_
    # parameters. Abschaltbar nur für den Vergleich.
    _MASS_EXACT = True

    # Homogenisierungsgrenze der 1D/2D-Modelle (Gegenproben 48/53):
    # kritische lokale Kennzahl über dem größten lochfreien Bereich
    # (Radius ρ), das Verhältnis der Filmkraft zur Membransteifigkeit auf
    # dieser Skala. Im Tiefton Π = ω·12μ·ρ⁴/(h³·T·j01²); allgemein mit der
    # VOLLEN Filmleitfähigkeit K(ω) (Reibung + Trägheit der Spaltluft) und
    # der dynamischen Steifigkeit der Beule (Membranmasse, Beulresonanz
    # f_ρ = f_res·a_mem/ρ), s. homogenization_limit. Gegen den 3D-Löser
    # (konturtreue Mündungen, gekoppelter Membranring) setzt die 1-dB-
    # Mehrabweichung bei gleichverteilten Lochbildern oberhalb von Π = 10
    # ein — zwei Kapseln (T ≈ 45 und 109 N/m), Spalte 20/25/38/65 µm, 42
    # Fälle; knappster Fall ½"-Kapsel, 20 µm, 32 Löcher: Warnung ab
    # 11,4 kHz, Einsatz 19,2 kHz. Π = 10 ist der VORSICHTIGE Rand.
    # Gemessen als Mehrabweichung gegen das dichte
    # Raster gleicher Lochfläche, |2D/3D| − |2D/3D dicht|: die frühere
    # vorzeichenrichtige Differenz schob beim weiten Spalt die eigene
    # Resonanzabweichung des dichten Rasters (±1,5 dB) in den Befund.
    # Die frühere Streuung der steifen Kapsel (Π ≈ 3…9) war der unbelastete
    # Membranring (Gegenprobe 52), die Lücke beim weiten Spalt (65 µm:
    # bis 107 kHz statt 18 kHz) die fehlende Filmträgheit und Beulresonanz.
    # Oberhalb von _F_BAND_TOP interessiert die Grenze nicht mehr.
    _PI_HOM = 10.0
    _F_BAND_TOP = 20.0e3

    # Lochkreis-Darstellung (Gegenprobe 53): das Radialfeld verschmiert
    # jeden Lochkreis zu einem Gaußband der Breite _RING_BAND·a_bp — eine
    # Darstellungswahl ohne eindeutigen physikalischen Wert. Bei Kreisen
    # mit vielen Löchern (Liniensenke) hängt das Ergebnis davon um mehrere
    # dB ab, und dort weicht es auch vom 3D-Löser ab (Filmwiderstand bei
    # erzwungener Form dagegen auf 3 % gleich: Formanpassung an das radial
    # stark gegliederte Druckfeld). Gewarnt wird, wo das Ergebnis mit dem
    # schmalsten darstellbaren Band (Liniensenke, 1,5 Zellen) um mehr als
    # _RING_REPR_DB abweicht — die halbe 1-dB-Toleranz, weil die Wahrheit
    # nicht zwischen beiden Darstellungen liegen muss (3D-Abweichung bis
    # zum Doppelten der Spanne). Damit kam die Warnung in allen 72
    # Lochkreis-Fällen der Sweeps vor dem 1-dB-Einsatz; knappster Fall
    # steife Kapsel, 65 µm, 48 Löcher auf einem Kreis: Warnung ab 2,23 kHz,
    # Mehrabweichung dort 0,95 dB, 1 dB erst bei 2,26 kHz (direkt nach-
    # gerechnet, Stand Gegenprobe 55; vorher 2,20 kHz und 0,91 dB). Der
    # Abstand ist knapp. Die Prüfung ist sonst vorsichtig — bei zwei
    # Lochkreisen warnt sie bis zu 40-fach zu früh.
    _RING_BAND = 0.10
    _RING_REPR_DB = 0.5

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
        modal_source=0,
        center_post_diameter=0.0,
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
        body_length=None,
        include_diffraction=True,
        axial_body_model="sphere",
        bem_body_diameter=56e-3,
        bem_body_gap=15e-3,
        bem_body_length=80e-3,
        # --- Spaltfilm-Modell -----------------------------------------------
        squeeze_model="1d",
        half_rotation_deg=None,
        grid_3d="coarse",
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
        self.modal_source = int(modal_source)
        if self.modal_source not in (0, 1):
            raise ValueError(
                "modal_source ist ein Schalter: 0 (aus, Voreinstellung) "
                "oder 1 (modenabhängiger Quelldruck)."
            )
        # ---------------------- MITTENTERMINIERUNG -------------------------
        # Eine in der Mitte festgelegte Membran (Kontaktstift, Mittenbolzen)
        # ist keine Kreis-, sondern eine RINGMEMBRAN. Das ist kein kleiner
        # Korrekturterm: die statische Lösung enthält einen Logarithmus, und
        # der macht schon einen winzigen Pfosten zu einer Größe erster
        # Ordnung (r_i/a = 1 % -> 22 % weniger Nachgiebigkeit).
        self.r_post = 0.5 * float(center_post_diameter)
        if self.r_post < 0.0:
            raise ValueError("center_post_diameter darf nicht negativ sein.")
        self.rho_post = self.r_post / self.a_mem
        if self.rho_post >= 0.6:
            raise ValueError(
                f"Mittenterminierung: r_i/a = {self.rho_post:.3f} — über 0.6 "
                "ist die Membran ein schmaler Ring, für den die hier "
                "benutzten Lumped-Formen (ein Freiheitsgrad, Kolbenmasse) "
                "nicht mehr sinnvoll sind."
            )
        if self.r_post > 0.0 and not _HAS_SCIPY:
            raise ValueError(
                "Mittenterminierung braucht SciPy (Besselfunktionen zweiter "
                "Art für die Ringmoden)."
            )
        self.u_post = self.rho_post ** 2

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
        self.body_length = (None if body_length is None
                            else float(body_length))
        if self.body_length is not None and self.body_length <= 0.0:
            raise ValueError("body_length muss > 0 sein (oder None).")
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
        # eine halbe Teilung des Durchgangs-Lochbilds — JE LOCHKREIS
        # (s. _hole_positions). Bis Gegenprobe 47 galt pauschal 180°/n_th;
        # das ist nur für EINEN Kreis mit allen Löchern eine halbe
        # Teilung, auf einem Mehrkreis-Raster lagen die Kerne der beiden
        # Hälften damit im Zwischenspalt fast übereinander (Gegenprobe 48).
        # Der gespeicherte Wert ist dann der Anzeigewert für diesen
        # Einkreis-Fall; gerechnet wird kreisweise.
        self._half_rot_auto = half_rotation_deg is None
        if half_rotation_deg is None:
            self.half_rotation_deg = 180.0 / max(self.n_th, 1)
        else:
            self.half_rotation_deg = float(half_rotation_deg)

        # Die Feldmodelle (2D/3D) brauchen SciPy.
        self.squeeze_model = sm if (sm == "1d" or _HAS_SCIPY) else "1d"
        # Auflösung des 3D-Gitters (s. _grid_3d_size); in 1D/2D ohne
        # Wirkung, wird aber immer geprüft (Projektdateien).
        g3 = str(grid_3d).strip().lower()
        if g3 not in ("coarse", "fine"):
            raise ValueError("grid_3d muss 'coarse' oder 'fine' sein.")
        self.grid_3d = g3

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
        # "bem" gilt AUSSERDEM für Ein-Membran-Kapseln mit dichter
        # Rückseite (Druckempfänger): dort liefert es den absoluten
        # FRONTFAKTOR der realen flachen Stirnfläche. Die Kopflänge ist
        # dann body_length; bem_body_diameter = 0 heißt frei stehende
        # Kapsel ohne Mikrofonkörper dahinter.
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
        if self.r_post >= 0.9 * self.a_bp:
            # Spaltfilm und Elektrode leben auf dem Ring r_i..a_bp — der
            # muss auch einer sein.
            raise ValueError(
                f"Mittenterminierung: der Pfosten (r_i = "
                f"{self.r_post * 1e3:.2f} mm) lässt von der Backplate "
                f"(a_bp = {self.a_bp * 1e3:.2f} mm) keinen brauchbaren "
                "Ring übrig.")
        self.S_mem = np.pi * a**2                 # Membranfläche [m^2]
        self.S_bp = np.pi * self.a_bp**2          # Backplate-Fläche [m^2]
        rho_s = self.mat_rho * t                  # Flächendichte [kg/m^2]

        # ------------------------------------------------------------------
        # STATISCHE FORM, KOLBENFAKTOR UND WIRKSAME FLÄCHE
        # Alles Folgende hängt an EINER Funktion: dem statischen Profil
        # φ(u) über u = (r/a)² (s. _ring_static_shape). Ohne Mitten-
        # terminierung ist das die Parabel 1−u und alle Momente sind die
        # bekannten Zahlen; mit Mittenterminierung dieselben Formeln,
        # nur andere Momente. Die Normierung ist φ_max = 1, damit w0 die
        # MAXIMALE Auslenkung bleibt (bei einer Ringmembran liegt sie
        # nicht in der Mitte, sondern bei r/a ≈ 0.33…0.46).
        # ------------------------------------------------------------------
        if self.u_post <= 0.0:
            # Parabel: die Momente sind exakt bekannt — keine Quadratur,
            # damit der Bestand BITGLEICH bleibt.
            self._phi_max, self._phi_umax = 1.0, 0.0
            m1, m2 = 0.5, 1.0 / 3.0
        else:
            # Maximum in geschlossener Form: dφ/du = −1 − (1−u_i)/(u·ln u_i)
            self._phi_umax = float(np.clip(
                -(1.0 - self.u_post) / np.log(self.u_post),
                self.u_post, 1.0))
            self._phi_max = float(_ring_static_shape(self._phi_umax,
                                                     self.u_post))
            # Momente per Gauss–Legendre (glatter Integrand auf [u_i, 1])
            _xq, _wq = np.polynomial.legendre.leggauss(512)
            _uq = 0.5 * (1.0 + self.u_post) + 0.5 * (1.0 - self.u_post) * _xq
            _wq = 0.5 * (1.0 - self.u_post) * _wq
            _pq = _ring_static_shape(_uq, self.u_post) / self._phi_max
            m1 = float(np.dot(_wq, _pq))
            m2 = float(np.dot(_wq, _pq**2))
        self._phi_m1, self._phi_m2 = m1, m2
        # wirksame Fläche: Volumenverschiebung je Maximalauslenkung
        self.S_eff_mem = self.S_mem * m1
        # ------------------------------------------------------------------
        # AKUSTISCHE MASSE DER MEMBRAN
        #     M_A = μ · rho_s / S                      [kg/m^4]
        # Ein Freiheitsgrad kann nur zwei Dinge exakt treffen. Die Kette
        # nimmt die statische Nachgiebigkeit C_T = S·a²·g/(8T) (exakt, s.
        # unten) und wählt die Masse so, dass auch die Grundresonanz der
        # Membran exakt ist, ω1² = z1²·T/(rho_s·a²):
        #     μ = 8/(z1²·g)      (ohne Pfosten 8/j01² = 1.3833).
        # Der Rayleigh-Wert der statischen Form, <φ²>/<φ>² (Parabel 4/3,
        # Ring ρ = 0.1: 1.242), ist eine obere Schranke der Frequenz: mit
        # ihm lag die Resonanz 1.9 % zu hoch, oder — bei vorgegebener
        # Resonanz — die statische Nachgiebigkeit 3.75 % zu hoch
        # (+0.32 dB, Gegenprobe 55). Die Modenmasse der J0-Form allein,
        # j01²/4 = 1.446, wäre ebenso falsch: sie gehört zur Moden-
        # Nachgiebigkeit 32/j01⁴ = 95.7 % der statischen. Die Biege-
        # steifigkeit der Folie trägt Promille und steckt in C_A_phys; ihr
        # Eigenwert wird nicht eigens getroffen.
        # ------------------------------------------------------------------
        self.sigma_mem = rho_s
        self._ring_g = _ring_compliance_factor(self.rho_post)
        self._mass_factor_rayleigh = m2 / m1**2
        _z1 = float(self._ring_modes()["z"][0])
        self._piston_factor = (8.0 / (_z1 ** 2 * self._ring_g)
                               if self._MASS_EXACT
                               else self._mass_factor_rayleigh)
        self.M_A_mem = self._piston_factor * rho_s / self.S_mem

        # ------------------------------------------------------------------
        # AKUSTISCHE NACHGIEBIGKEIT DER MEMBRAN
        # 1) Anteil der Vorspannung T [N/m] (Beranek):
        #        C_T = pi * a^4 / (8 * T)             [m^5/N = m^3/Pa]
        #    (statische Durchbiegung w0 = p*a^2/(4T), Volumen = p*pi*a^4/(8T))
        #    MIT MITTENTERMINIERUNG kommt der geschlossene Ringfaktor
        #        g(ρ) = 1 − ρ⁴ + (1−ρ²)²/ln ρ
        #    dazu (s. _ring_compliance_factor) — bei ρ = 0.05 sind das
        #    schon 33 % weniger Nachgiebigkeit.
        # 2) Anteil der Biegesteifigkeit der Folie (eingespannte Platte):
        #        D   = E * t^3 / (12 * (1 - nu^2))    [N*m]
        #        C_B = pi * a^6 / (192 * D)
        # Beide Federn wirken parallel (Steifigkeiten addieren sich):
        #        1/C_phys = 1/C_T + 1/C_B
        # ------------------------------------------------------------------
        C_T = np.pi * a**4 / (8.0 * self.tension) * self._ring_g
        D_plate = self.mat_E * t**3 / (12.0 * (1.0 - self.mat_nu**2))
        C_B = np.pi * a**6 / (192.0 * D_plate)
        self.C_A_phys = 1.0 / (1.0 / C_T + 1.0 / C_B)
        # GRENZE, dokumentiert: der Ringfaktor gilt für die VORSPANNUNG.
        # Die Biegesteifigkeit einer eingespannten Ringplatte hat einen
        # anderen Formfaktor, der hier nicht gerechnet wird. Bei Folien-
        # membranen trägt sie Promille — wird sie relevant, ist das ein
        # Fehler und keine Feinheit.
        if self.r_post > 0.0 and C_T > 0.02 * C_B:
            raise ValueError(
                "Mittenterminierung: die Biegesteifigkeit der Folie trägt "
                f"{100.0 * C_T / (C_T + C_B):.1f} % der Gesamtsteifigkeit. "
                "Der Ringfaktor ist nur für den VORSPANNUNGSANTEIL "
                "hergeleitet — für eine so steife Platte gilt er nicht."
            )

        # Aus Vorspannung/Steifigkeit resultierende Resonanz (Diagnose):
        self.f_res_from_tension = 1.0 / (
            2.0 * np.pi * np.sqrt(self.M_A_mem * self.C_A_phys)
        )
        # EXAKTE Modalfrequenz der Grundmode (Diagnose): Membran-Eigenwert
        # j01 = 2.40483 (mit Mittenterminierung der der RINGmembran, z_1 =
        # 2.80 schon bei r_i/a = 1 %), für die Biegesteifigkeit der
        # eingespannte Platten-Eigenwert lambda² = 10.2158; beide Anteile
        # addieren sich in guter Näherung quadratisch. Mit dem Massenfaktor
        # 8/(z1²·g) trifft f_res_from_tension den Vorspannungsanteil exakt
        # (vorher, mit dem Rayleigh-Wert 4/3, lag er 1.9 % darüber); ein
        # Unterschied bleibt nur über die Biegesteifigkeit.
        f_T_ex = _z1 / (2.0 * np.pi * a) * np.sqrt(self.tension / rho_s)
        f_B_ex = (10.2158 / (2.0 * np.pi * a**2)
                  * np.sqrt(D_plate / rho_s))
        self.f_res_modal_exact = float(np.hypot(f_T_ex, f_B_ex))

        # Ist eine Soll-Resonanzfrequenz vorgegeben, wird die Nachgiebigkeit
        # so skaliert, dass f_res exakt getroffen wird (die Vorspannung
        # bleibt als Plausibilitäts-Referenz in summary() sichtbar). Mit
        # dem Massenfaktor 8/(z1²·g) ist das die statische Nachgiebigkeit
        # der Membran, deren Grundresonanz f_res ist.
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
        # Mit MITTENTERMINIERUNG beginnt das Gitter am Pfostenrand: dort
        # sitzt eine Wand, kein Fluss, und das Membranprofil ist die
        # Ringform (s. _ring_static_shape). Ohne Pfosten ist r0 = 0 und
        # alles unten reduziert sich bitgleich auf den bisherigen Stand.
        N = 60
        r0 = self.r_post
        dr = (self.a_bp - r0) / N
        r_c = r0 + (np.arange(N) + 0.5) * dr
        r_f = r0 + np.arange(N + 1) * dr
        self._fld_area = 2.0 * np.pi * r_c * dr           # Zellflächen [m^2]
        self._fld_phi = _ring_static_shape(
            np.minimum((r_c / self.a_mem) ** 2, 1.0), self.u_post)
        self._fld_phi = np.maximum(self._fld_phi, 0.0) / self._phi_max
        self._fld_Sphi = float(np.sum(self._fld_phi * self._fld_area))
        # Geometriefaktor der lateralen Flächenleitwerte: Gface = 2*pi*r_f/dr*K
        # (Randflächen 0 = kein Fluss -> Neumann, innen die Pfostenwand)
        gg = np.zeros(N + 1)
        gg[1:N] = 2.0 * np.pi * r_f[1:N] / dr
        self._fld_gface_geom = gg
        self._fld_gedge_geom = 4.0 * np.pi * r_f[N] / dr
        self._fld_S_elec = np.pi * (self.a_bp**2 - r0**2)
        self._fld_N = N
        self._fld_r_c = r_c

        # Radiale Dichteverteilungen der Löcher (normiert: Σ dens·A = 1),
        # damit die Gesamt-Lochleitwerte erhalten bleiben. Jeder Lochkreis
        # wird als schmales Ringband um seinen Radius konzentriert (ohne
        # Lochkreis: gleichmäßig über die Elektrode) und mit seinem Anteil
        # an der Gesamt-Lochzahl gewichtet — ein einzelner Ring reproduziert
        # exakt das bisherige Ein-PCD-Verhalten.
        def _hole_density(rings, n_total, line=False):
            if n_total <= 0:
                return np.ones(N) / float(np.sum(self._fld_area))
            # line: schmalstes darstellbares Band (Liniensenke) — nur für
            # die Prüfung der Lochkreis-Darstellung (Gegenprobe 53)
            width = (1.5 * dr if line
                     else max(self._RING_BAND * self.a_bp, 1.5 * dr))
            dens = np.zeros(N)
            for cnt, r_pcd in rings:
                if cnt <= 0:
                    continue
                band = (np.ones(N) if r_pcd is None
                        else np.exp(-0.5 * ((r_c - r_pcd) / width) ** 2))
                dens += cnt * band / float(np.sum(band * self._fld_area))
            return dens / n_total

        self._fld_dens_th = _hole_density(self._th_rings, self.n_th)
        self._fld_dens_th_line = _hole_density(self._th_rings, self.n_th,
                                               line=True)
        self._fld_dens_bh = _hole_density(self._bh_rings, self.n_bh)

        # CLEARANCE-RING auf dem Feldgitter (s. __init__): Relief-Karte
        # für breite Ringe, Stub-Zelle für schmale; Flags, ob Loch-
        # Mündungen im Relief liegen (dann entlastete Engstelle).
        self._clr_relief, self._clr_stub_cell = self._clearance_on_grid(
            r_c, dr, r0)
        mask = self._clr_relief > 0.0
        thr = 0.5 / float(np.sum(self._fld_area))
        self._clr_th_relieved = bool(np.any(mask & (self._fld_dens_th > thr)))
        self._clr_bh_relieved = bool(np.any(mask & (self._fld_dens_bh > thr)))

        # Elektrodenrand in Modenkoordinate u = r^2/a_mem^2
        self._ub = min((self.a_bp / self.a_mem) ** 2, 1.0)

        # Porositätsprofile für die ELEKTROSTATIK auf der Modenkoordinate:
        # lokale Lochflächenanteile aus denselben radialen Dichten wie im
        # Feldmodell — damit sind Lochkreise (PCD) auch in Pull-in, Feder-
        # Erweichung, Wandlerkoeffizient und C0 konsistent berücksichtigt.
        # Ohne PCD ergeben sich exakt die bisherigen konstanten Anteile.
        # Mit Mittenterminierung beginnt die Elektrodenfläche am
        # Pfostenrand: unter dem Pfosten gibt es weder Luftspalt noch Feld.
        u_es = np.linspace(self.u_post, self._ub, 401)
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
        # Modenprofil an denselben Stützstellen (Parabel bzw. Ringform),
        # normiert auf 1 — die dynamische Modenform der Kette.
        self._es_phi = (_ring_static_shape(u_es, self.u_post)
                        / self._phi_max)
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
        # am Rand eingespannt. Seit Gegenprobe 49 wird der statische
        # Arbeitspunkt als EXAKTE Randwertaufgabe gelöst (s. _static_setup):
        # konstante Spannung (der Bias-Widerstand hält U0 statisch fest;
        # das Luftpolster entweicht statisch durch die Löcher und trägt
        # NICHT),
        #     T·∇²w = −(eps0 U0²/2)·[c_s/(h−w)² + c_b/(h+d−w)²].
        # PULL-IN ist der Faltpunkt dieses Lösungsasts. Die Feder-Erweichung
        # kommt aus dem linearisierten Operator am Arbeitspunkt; Kraft-
        # modulation und Ruhekapazität (Wandlerkoeffizient, C0) werden
        # über dem exakten Spaltprofil g(r) = h − w(r) integriert. Das
        # frühere Ein-Moden-Bild (Galerkin mit der statischen Form, k_gen =
        # S_eff²/C_A) war für kleine Lasten exakt, lag aber am Pull-in
        # Warrens Ā um +5.0 % (Kreis) bzw. +2.6 % (Ring) zu hoch.
        #
        # DUAL-BACKPLATES (Gegentakt): u_bias liegt an BEIDEN Spalten voll
        # an (Backplates auf ±U, Gesamtversorgung 2·U). Die statischen
        # Kräfte heben sich auf -> w0 = 0, die Erweichung beider Seiten
        # ADDIERT sich. Pull-in ist deshalb das Eigenwertkriterium am
        # Ruhespalt, nicht das Verschwinden eines Gleichgewichts — für die
        # volle, lochfreie Elektrode geschlossen Ā = j01²/4 = 1.4458, und
        # mit Warrens 0.789 der Einzel-Backplate
        #     U_PI(dual)/U_PI(single) = sqrt(1.4458/0.789) = 1.3537
        # (Ein-Moden-Bild: 1.3464, Gegenprobe 39). Der Gewinn kommt NICHT
        # aus einer kleineren Feldstärke, sondern allein daraus, dass die
        # Membran im Ruhepunkt bleibt.
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
        # generalisierte Steifigkeit zur Koordinate w0 (Maximalauslenkung):
        # E = (w0·S_eff)²/(2C_A)  ->  k_gen = S_eff²/C_A. Ohne Mitten-
        # terminierung ist S_eff = S/2, also S²/(4C_A). Seit Gegenprobe 49
        # trägt sie den Arbeitspunkt nicht mehr (das tut die exakte
        # Randwertaufgabe), bleibt aber die Referenz des Ein-Moden-Bilds.
        self._k_gen = self.S_eff_mem**2 / self.C_A_mem

        # EXAKTER ARBEITSPUNKT (Gegenprobe 49, s. _static_setup)
        self._static_setup()
        self._st_branch_cache = None
        st = self._st
        lam_b = EPS0 * self.u_bias**2
        if self.architecture == "dual":
            # Gegentakt: statisch symmetrisch (w0 = 0), beide Spalte
            # erweichen; Pull-in = Eigenwertkriterium am Ruhespalt
            lam_pi = self._st_dual_pullin_lambda()
            w_st = np.zeros(st["u"].size)
            soft = lam_b * 2.0 * (st["cs"] / self.h_gap**3
                                  + st["cb"] / (self.h_gap + self.d_bh)**3)
            stable = lam_b < lam_pi
        else:
            lam_pi = self._st_branch()[1]
            if lam_b <= 0.0:
                w_st = np.zeros(st["u"].size)
            else:
                w_st = self._st_equilibrium(lam_b)
            stable = w_st is not None
            soft = self._st_system(w_st, lam_b)[3] if stable else None
        self.U_pullin = (float(np.sqrt(lam_pi / EPS0))
                         if np.isfinite(lam_pi) else float("inf"))
        if not stable:
            raise ValueError(
                "Elektrostatischer Kollaps (Pull-in): die statische "
                "Anziehung der Backplate übersteigt die Rückstellkraft der "
                "Membran. Maximal stabile Polarisationsspannung für diese "
                f"Konfiguration: ca. {self.U_pullin:.1f} V. Abhilfe: "
                "Spannung senken, Luftspalt vergrößern oder Membran steifer "
                "(höhere Resonanzfrequenz/Vorspannung). Hinweis: statisch "
                "trägt nur die Membran-Vorspannung — das Luftpolster "
                "entweicht durch die Löcher; gemessene Kapselresonanzen "
                "enthalten dagegen die Luftpolster-Steifigkeit und liegen "
                "deshalb unter der hier maßgeblichen Vorspannungs-Resonanz."
            )
        self._w_static = w_st
        self._st_soft = soft
        self.w0_static = float(np.max(w_st))       # MAXIMALE Auslenkung
        self.h_min_static = self.h_gap - self.w0_static  # engster Restspalt

        # Kleinsignal-Nachgiebigkeit aus dem LINEARISIERTEN Operator. Das
        # Verhältnis zweier Lösungen desselben diskreten Systems kürzt den
        # Diskretisierungsfehler heraus: ohne Bias ist es exakt 1.
        C_st0 = self._st_compliance(np.zeros_like(soft))
        C_st1 = self._st_compliance(soft)
        if not (np.isfinite(C_st1) and C_st1 > 0.0):
            raise ValueError("Elektrostatischer Kollaps (Feder-Erweichung).")
        self.C_A_eff = self.C_A_mem * C_st1 / C_st0
        # relative Steifigkeitsreduktion durch die Vorspannung (Diagnose)
        self.softening_ratio = 1.0 - self.C_A_mem / self.C_A_eff
        # statische Auslenkung auf dem Elektrodengitter der Integrale
        self._es_w = np.interp(self._es_u, st["u"], w_st)

        # NUMERISCHER BODEN der Membrandämpfung (s. _Q_MEMBRANE_INTERNAL).
        # Die DOMINANTE Dämpfung der Membran-Grundmode kommt aus dem Spalt-
        # film: die Piston-Bewegung der Membran drückt die Spaltluft lateral
        # zu den Löchern (Škvor-Widerstand R_A_gap). Dieser Widerstand sitzt
        # AUSSCHLIESSLICH im Spalt-Zweitor (_backplate_gap_abcd bzw. das
        # 2D-Feld) — Gegenprobe 31 hat die frühere zusätzliche Reihenschaltung
        # in _membrane_impedance als Doppelzählung entfernt. Der Wert hier ist
        # nur der Materialverlust der Folie und trägt praktisch allein dann,
        # wenn KEIN Spaltfilm existiert (geschlossene Backplate, n_th = 0
        # und kein Randspalt -> R_A_gap_front = None); über Q = 20…1e5
        # ändert er die Resonanzüberhöhung um ≤ 0.07 dB.
        self.R_A_mem = (
            np.sqrt(self.M_A_mem / self.C_A_eff) / self._Q_MEMBRANE_INTERNAL
        )

        # ------------------------------------------------------------------
        # WANDLERKOEFFIZIENT UND RUHEKAPAZITÄT AM ARBEITSPUNKT
        # Betrieb mit konstanter Ladung (hochohmig): e = U0 * dC/C0.
        # Membranmode w = dw*phi(r) moduliert die Kapazität:
        #     dC = eps0 * I_F(w0) * dw,   I_F = Int phi/g(r)^2 dS
        # (gleiches Integral wie die Kraft). Mit dw = V_disp/S_eff
        # (S_eff = S·m1, wirksame Fläche der Mode) folgt
        #     e = Theta * V_disp,  Theta = U0 eps0 I_F / (S_eff * C0),
        # für die Parabel (m1 = 1/2) Theta = 2 U0 eps0 I_F / (S * C0). Bis
        # Gegenprobe 54 stand hier auch mit Mittenterminierung 2/S — bei
        # der Ringmembran ist m1 > 1/2 (1 mm Pfosten auf 26 mm: 0.620), die
        # Spannung lag um 2·m1 zu hoch (+1.9 dB).
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
            # exaktes statisches Profil (Gegenprobe 49); bei 'dual' null
            I_F, _, I_C = self._electrode_integrals(profile=sign * self._es_w)
            C0 = EPS0 * I_C
            if C0_rear is None:
                C0_rear = C0
            theta += ((1.0 / self._phi_m1) * self.u_bias * EPS0 * I_F
                      / (self.S_mem * C0))
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
        # Seit Gegenprobe 49 das Flächenmittel des EXAKTEN Profils.
        # ------------------------------------------------------------------
        span_es = self._es_u[-1] - self._es_u[0]
        sag = (float(np.trapezoid(self._es_w, self._es_u)) / span_es
               if span_es > 0.0 else 0.0)
        self.h_gap_front = max(self.h_gap - sag, 0.05 * self.h_gap)
        # normierte statische FORM auf dem 2D-Feldgitter (Spaltprofil der
        # polarisierten Seite, s. _gap_field_2port); ohne Bias die Mode
        if self.w0_static > 0.0:
            self._fld_sag_shape = np.interp(
                np.minimum((self._fld_r_c / self.a_mem) ** 2, 1.0),
                st["u"], w_st) / self.w0_static
        else:
            self._fld_sag_shape = self._fld_phi.copy()

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
            # BEWUSSTE GRENZE DES 1D-PFADS: Škvor rechnet die Spaltluft
            # unter der GANZEN Platte zu den Löchern; die Membranfläche
            # AUSSERHALB des Plattenrands (a_bp < a_mem) erzeugt aber
            # Volumenfluss, der den Film über den tieferen Ringraum am
            # Rand umgeht. Das 2D-Feldmodell führt das seit Gegenprobe 37
            # explizit (Faktor f_in² mit f_in = u(2−u), u = (a_bp/a_mem)²);
            # hier bleibt es absichtlich weg, weil der Faktor von der
            # Antriebskonvention abhängt — Škvor und _edge_R rechnen mit
            # KOLBEN-Antrieb (dann wäre es u²), das Feld mit der Parabel-
            # mode. Der 1D-Pfad ist der grobe Lumped-Pfad (er weicht auch
            # sonst um bis zu 13 dB vom Feldmodell ab); wer a_bp < a_mem
            # quantitativ rechnen will, nimmt squeeze_model='2d'.
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
        #
        # GEMESSEN AB DEM VORDEREN EINLASS, also ab der Kalotte am
        # vorderen Pol — und das ist d_ext, nicht d_rear_ax. Bei EINER
        # Backplate sind beide gleich (die Membran IST der vordere
        # Einlass); bei ZWEI symmetrischen Backplates liegt der vordere
        # Einlass um Spalt + Plattendicke weiter vorn, und genau darum
        # ist d_ext um diesen Betrag größer. Mit d_ext ist der Ring auch
        # mit der beugungsfreien Rechnung konsistent, die den Rückdruck
        # als exp(-j·k·d_ext·cosθ) ansetzt (Gegenprobe 44 g).
        self._ring_cos = float(np.clip(
            (self.R_body - self.d_ext) / self.R_body, -1.0, 1.0))

        # ------------------------------------------------------------------
        # KÖRPERMODELL FÜR DIE BEUGUNG
        # 'sphere' (Standard) und 'spheroid' sind Ersatzkörper für den
        # AXIALEN Front-Rück-Transfer der Doppelmembran-Bauform. 'bem'
        # rechnet dagegen die reale Kontur und liefert damit auch den
        # absoluten FRONTFAKTOR der flachen Stirnfläche — der ist auch für
        # eine Ein-Membran-Kapsel die eigentlich interessante Größe, weil
        # die Kugelkalotte dort systematisch falsch liegt
        # (s. _bem_axial_fields und Gegenprobe 41). Ist die Kapsel
        # rückseitig offen, kommt der Druck am Rückeinlass aus demselben
        # Lösungsgang (_bem_rear_inlet_weights, Gegenprobe 44).
        # Kopflänge: bei der Doppelmembran spannen die beiden Membranen
        # die Stirnflächen auf, also d_ext; sonst ist sie eine eigene
        # Geometrieangabe (body_length).
        # ------------------------------------------------------------------
        self._bem_head_len = self.d_ext
        if self.axial_body_model == "spheroid":
            if self.architecture != "dual_diaphragm":
                raise ValueError(
                    "axial_body_model='spheroid' gilt nur für die "
                    "Doppelmembran-Bauform (axialer Front-Rück-Transfer). "
                    "Für den Frontfaktor einer Ein-Membran-Kapsel 'bem' "
                    "nehmen."
                )
        elif self.axial_body_model == "bem":
            if self.architecture != "dual_diaphragm":
                # GRADIENTENEMPFÄNGER: der rückwärtige Einlass hat seit
                # Gegenprobe 44 einen eigenen Patch auf der Kontur
                # (Stirnfläche oder Mantel, je nach cavity_hole_position).
                # Der Fall ist damit gerechnet statt gesperrt — was bleibt,
                # ist eine Modellgrenze und deshalb eine Warnung: die
                # Kontur ist ein glatter Zylinder, sie kennt weder Korb
                # noch Kapselgitter, und der Bohrungskranz wird als
                # idealer Ring bei seiner Einbautiefe angesetzt.
                if self.body_length is None:
                    raise ValueError(
                        "axial_body_model='bem' braucht die axiale "
                        "Körperlänge body_length (die Kugelrechnung kennt "
                        "nur body_diameter)."
                    )
                self._bem_head_len = self.body_length
                if self.rear_open:
                    # Tiefe ab der Stirnfläche = vorderer Einlass, also
                    # d_ext (s. _bem_rear_inlet_weights).
                    if self.cavity_hole_position == "end":
                        wo = "in die hintere Stirnfläche"
                        stimmig = (abs(self.d_ext - self.body_length)
                                   <= 0.25 * self.d_ext)
                        warum = ("bei 'end' münden die Löcher am "
                                 "Hohlraumende, also MUSS body_length "
                                 "ungefähr d_ext sein")
                    else:
                        wo = "als Bohrungskranz in den Mantel"
                        stimmig = self.body_length > self.d_ext
                        warum = ("bei 'circumference' münden die Löcher "
                                 "radial, also MUSS body_length größer "
                                 "als d_ext sein; sonst wird der Ring "
                                 "auf die hintere Stirnfläche geklemmt")
                    hinweis = (
                        f"axial_body_model='bem' mit offener Rückseite: der "
                        f"rückwärtige Einlass wird {wo} gelegt, "
                        f"d_ext = {self.d_ext * 1e3:.1f} mm hinter dem "
                        f"vorderen Einlass (Gegenprobe 44). Die Kontur ist "
                        f"ein glatter Zylinder — Korb, Kapselgitter und "
                        f"die endliche Lochteilung sind darin nicht "
                        f"enthalten."
                    )
                    if not stimmig:
                        hinweis += (
                            f" ACHTUNG: body_length = "
                            f"{self.body_length * 1e3:.1f} mm passt nicht "
                            f"dazu ({warum})."
                        )
                    warnings.warn(hinweis, UserWarning, stacklevel=2)
            elif self.body_length is not None:
                raise ValueError(
                    "body_length gilt nicht für die Doppelmembran-Bauform "
                    "— dort spannen die beiden Membranen die Stirnflächen "
                    "auf, die axiale Länge ist d_ext."
                )
        if self.axial_body_model == "bem" and self._bem_head_len < 2.0e-3:
            # Verrundung und Elementlänge skalieren mit dem Körper
            # (s. _bem_geometry), darunter wird die Scheibe aber so dünn,
            # dass die m=0-Kollokation auf dem Mantel entartet.
            raise ValueError(
                "BEM-Kontur: die axiale Körperlänge muss >= 2 mm sein "
                f"({self._bem_head_len * 1e3:.2f} mm).")
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
        elif self.n_th > 0 or self.ring_vent_w > 0.0:
            # HOMOGENISIERUNGSGRENZE (Gegenproben 48/53): 1D/2D verschmieren
            # die Löcher. Liegt die Grenze im Hörband, wird gerechnet,
            # aber gewarnt — der 3D-Löser löst den Fall auf.
            lim = self.homogenization_limit()
            if lim["f_limit"] < self._F_BAND_TOP and lim["cause"] == "ring":
                warnings.warn(
                    f"squeeze_model='{self.squeeze_model}': die Lochkreis-"
                    f"Darstellung ist oberhalb von etwa "
                    f"{lim['f_ring'] / 1e3:.2f} kHz nicht belastbar. Das "
                    f"Radialfeld verschmiert jeden Lochkreis zu einem Band; "
                    f"ab dort hängt das Ergebnis um mehr als "
                    f"{self._RING_REPR_DB:.1f} dB von dieser Darstellungs"
                    f"wahl ab (Liniensenke statt Band), und gegen den 3D-"
                    f"Löser erreicht die Abweichung 1 dB und mehr — bei "
                    f"Kreisen mit vielen Löchern bis 5 dB (Gegenprobe 53). "
                    f"Für diesen Bereich squeeze_model='3d' verwenden.",
                    UserWarning, stacklevel=3)
            elif lim["f_hom"] < self._F_BAND_TOP:
                warnings.warn(
                    f"squeeze_model='{self.squeeze_model}': das Lochbild "
                    f"ist für die Homogenisierung zu spärlich — lochfreie "
                    f"Bereiche bis {lim['rho'] * 1e3:.1f} mm Abstand zur "
                    f"nächsten Durchgangsbohrung. Oberhalb von etwa "
                    f"{lim['f_hom'] / 1e3:.1f} kHz beult sich die Membran "
                    f"über diesen Bereichen örtlich aus, was 1D/2D nicht "
                    f"abbilden; die Abweichung gegen den 3D-Löser kann "
                    f"dort 1 dB übersteigen, bei sehr spärlichen Lochbildern "
                    f"10 dB (Gegenprobe 48). Für diesen Bereich "
                    f"squeeze_model='3d' verwenden.",
                    UserWarning, stacklevel=3)

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
    # Elektrostatik: EXAKTER statischer Arbeitspunkt (Gegenprobe 49)
    # ======================================================================
    # Die Membran unter Polarisationsspannung ist eine nichtlineare
    # Randwertaufgabe, kein Ein-Freiheitsgrad-Problem:
    #
    #     T·(4/a²)·(u·w_u)_u = −p_es(u, w),        u = r²/a_mem²,
    #     p_es = (ε0U²/2)·[c_s/(h−w)² + c_b/(h+d_bh−w)²]   (Elektrode),
    #     w(1) = 0;  Kreis: u·w_u = 0 bei u = 0;  Ring: w(u_i) = 0.
    #
    # In der Koordinate u ist der Operator an der Achse regulär und das
    # Finite-Volumen-System tridiagonal. T ist die Spannung, die die
    # statische Nachgiebigkeit C_A_mem trägt: T = π·a⁴·g(ρ)/(8·C_A_mem).
    # Bis Gegenprobe 48 stand hier ein Ein-Moden-Galerkin mit der
    # statischen Form φ — für kleine Lasten exakt, zum Pull-in hin aber
    # zu steif: Warrens kritisches Ā (0.789 Kreis, 1.548 Ring ρ = 0.1)
    # lag um +5.0 % bzw. +2.6 % zu hoch, U_PI also um ~2.5 %/1.3 %.
    #
    # PULL-IN ist der FALTPUNKT des Lösungsasts. Der Ast wird über das
    # verdrängte Volumen V = ∫w dS parametrisiert (monoton längs des Asts,
    # auch über die Falte hinweg); λ = ε0·U² ist dann eine Unbekannte, und
    # ihr Maximum über V ist λ_PI. Die Kleinsignal-Nachgiebigkeit folgt aus
    # dem LINEARISIERTEN Operator L + λ·∂p/∂w am Arbeitspunkt (statt
    # 1/C_A − 4k_neg/S²) und divergiert genau im Faltpunkt.
    #
    # 'dual' (Gegentakt): w0 = 0, beide Spalte erweichen; Pull-in ist der
    # kleinste Eigenwert von −L·w = λ·2(c_s/h³ + c_b/(h+d)³)·w — für die
    # volle, lochfreie Elektrode geschlossen Ā = j01²/4 = 1.4458.
    _N_STATIC = 400

    def _static_setup(self):
        """Gitter, Porosität und Membranoperator der exakten Statik."""
        N = self._N_STATIC
        u = np.linspace(self.u_post, 1.0, N + 1)
        du = u[1] - u[0]
        vol = np.full(u.size, du)
        vol[0] *= 0.5
        vol[-1] *= 0.5
        on = u <= self._ub + 1e-12
        cs = np.where(on, np.interp(u, self._es_u, self._es_c_solid), 0.0)
        cb = np.where(on, np.interp(u, self._es_u, self._es_c_blind), 0.0)
        tension = (np.pi * self.a_mem**4 * self._ring_g
                   / (8.0 * self.C_A_mem))
        g = (4.0 * tension / self.a_mem**2) * 0.5 * (u[:-1] + u[1:]) / du
        L = np.zeros((3, u.size))                  # solve_banded-Format
        L[0, 1:] = g                               # A[k, k+1]
        L[2, :-1] = g                              # A[k+1, k]
        L[1, :-1] -= g
        L[1, 1:] -= g
        # dynamische Modenform (die der Kette) auf demselben Gitter
        psi = _ring_static_shape(u, self.u_post) / self._phi_max
        self._st = dict(u=u, vol=vol, cs=cs, cb=cb, L=L, tension=tension,
                        ring=self.u_post > 0.0, psi=psi,
                        S=np.pi * self.a_mem**2)

    def _st_system(self, w, lam, sides=1.0):
        """Residuum F und Jacobi-Band J von L·w + λ·p̂(w)·vol = 0 samt
        Dirichlet-Zeilen; außerdem ∂F/∂λ und die örtliche Erweichung
        λ·∂p̂/∂w (sides = 2: Gegentakt bei w = 0)."""
        st = self._st
        L, vol = st["L"], st["vol"]
        g1 = self.h_gap - w
        g2 = self.h_gap + self.d_bh - w
        f = 0.5 * (st["cs"] / g1**2 + st["cb"] / g2**2)
        fp = sides * (st["cs"] / g1**3 + st["cb"] / g2**3)
        Lw = L[1] * w
        Lw[:-1] += L[0, 1:] * w[1:]
        Lw[1:] += L[2, :-1] * w[:-1]
        F = Lw + lam * f * vol
        J = L.copy()
        J[1] = J[1] + lam * fp * vol
        dFdl = f * vol
        # Dirichlet: Einspannung außen, Ring zusätzlich am Pfosten
        F[-1] = w[-1]
        J[1, -1], J[2, -2], J[0, -1] = 1.0, 0.0, 0.0
        dFdl[-1] = 0.0
        if st["ring"]:
            F[0] = w[0]
            J[1, 0], J[0, 1] = 1.0, 0.0
            dFdl[0] = 0.0
        return F, J, dFdl, lam * fp

    def _st_on_volume(self, V, w, lam):
        """Punkt des Lösungsasts mit vorgegebenem Volumen Σw·vol = V
        (Newton mit Randbordierung, zwei Bandlösungen je Schritt)."""
        vol = self._st["vol"]
        for _ in range(60):
            F, J, dFdl, _ = self._st_system(w, lam)
            x1 = _solve_banded((1, 1), J, -F)
            x2 = _solve_banded((1, 1), J, dFdl)
            dl = (np.dot(vol, x1) + np.dot(vol, w) - V) / np.dot(vol, x2)
            dw = x1 - dl * x2
            w = w + dw
            lam = lam + dl
            if not np.all(np.isfinite(w)) or np.max(w) >= 0.999 * self.h_gap:
                return None, None
            if np.max(np.abs(dw)) < 1e-13 * self.h_gap:
                return w, lam
        return None, None

    def _st_branch(self):
        """Stabiler Ast bis zum Faltpunkt: [(V, λ, w), ...] und λ_PI.
        Einmal je Kapsel (gecacht)."""
        cached = getattr(self, "_st_branch_cache", None)
        if cached is not None:
            return cached
        vol = self._st["vol"]
        V_full = self.h_gap * float(np.sum(vol))
        w = np.zeros(vol.size)
        lam = 0.0
        pts = [(0.0, 0.0, w.copy())]
        dV = V_full / 80.0
        V = 0.0
        while V < 0.95 * V_full:
            V += dV
            w_n, lam_n = self._st_on_volume(V, w, lam)
            if w_n is None:
                break
            w, lam = w_n, lam_n
            pts.append((V, lam, w.copy()))
            if len(pts) >= 3 and pts[-1][1] < pts[-2][1]:
                break                              # Falte überschritten
        lams = np.array([p[1] for p in pts])
        k = int(np.argmax(lams))
        lam_pi = float(lams[k])
        if 0 < k < len(pts) - 1:
            # Goldener Schnitt um das diskrete Maximum
            a_, b_ = pts[k - 1][0], pts[k + 1][0]
            wg = pts[k][2]
            gr = 0.5 * (np.sqrt(5.0) - 1.0)
            c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
            wc, lc = self._st_on_volume(c_, wg, pts[k][1])
            wd, ld = self._st_on_volume(d_, wg, pts[k][1])
            for _ in range(40):
                if wc is None or wd is None:
                    break
                if lc > ld:
                    b_, d_, wd, ld = d_, c_, wc, lc
                    c_ = b_ - gr * (b_ - a_)
                    wc, lc = self._st_on_volume(c_, wd, ld)
                else:
                    a_, c_, wc, lc = c_, d_, wd, ld
                    d_ = a_ + gr * (b_ - a_)
                    wd, ld = self._st_on_volume(d_, wc, lc)
            lam_pi = max(lam_pi, *(x for x in (lc, ld) if x is not None))
        self._st_branch_cache = (pts[:k + 1], lam_pi)
        return self._st_branch_cache

    def _st_equilibrium(self, lam_t):
        """Stabiles Gleichgewicht bei λ = ε0·U² oder None (Pull-in)."""
        pts, lam_pi = self._st_branch()
        if lam_t >= lam_pi:
            return None
        lams = np.array([p[1] for p in pts])
        k = int(np.searchsorted(lams, lam_t))
        k = min(max(k, 1), len(pts) - 1)
        V0, l0, w0_ = pts[k - 1]
        V1, l1, w1_ = pts[k]
        t = (lam_t - l0) / (l1 - l0) if l1 > l0 else 0.0
        w = w0_ + t * (w1_ - w0_)
        lam = lam_t
        # Newton bei festem λ vom interpolierten Startwert
        for _ in range(60):
            F, J, _, _ = self._st_system(w, lam)
            dw = _solve_banded((1, 1), J, -F)
            w = w + dw
            if np.max(np.abs(dw)) < 1e-13 * self.h_gap:
                break
        # Sekanten-Absicherung über das Volumen, falls Newton nahe der
        # Falte abwandert (dort ist J fast singulär)
        F, _, _, _ = self._st_system(w, lam)
        if (not np.all(np.isfinite(w)) or np.max(np.abs(F[:-1])) > 1e-6
                or np.max(w) >= 0.999 * self.h_gap):
            lo, hi = V0, V1
            wv, lv = w0_, l0
            for _ in range(80):
                Vm = 0.5 * (lo + hi)
                wm, lm = self._st_on_volume(Vm, wv, lv)
                if wm is None:
                    return None
                if lm < lam_t:
                    lo, wv, lv = Vm, wm, lm
                else:
                    hi = Vm
                if abs(lm - lam_t) < 1e-12 * lam_pi:
                    break
            w = wm
        return w

    def _st_compliance(self, soft):
        """Volumen-Nachgiebigkeit unter gleichförmigem Druck für den
        linearisierten Operator L + soft·vol (Dirichlet wie oben)."""
        st = self._st
        _, J, _, _ = self._st_system(np.zeros(st["u"].size), 0.0)
        free = np.ones(st["u"].size)               # keine Dirichlet-Zeile
        free[-1] = 0.0
        if st["ring"]:
            free[0] = 0.0
        J[1] = J[1] + soft * st["vol"] * free
        dw = _solve_banded((1, 1), J, -st["vol"] * free)
        return float(np.dot(st["vol"], dw)) * st["S"]

    def _st_dual_pullin_lambda(self):
        """Gegentakt: kleinstes λ mit −L·w = λ·2(c_s/h³ + c_b/(h+d)³)·w
        (Potenziteration auf (−L)⁻¹·K̂, K̂ ≥ 0)."""
        st = self._st
        _, J, _, _ = self._st_system(np.zeros(st["u"].size), 0.0)
        Khat = 2.0 * (st["cs"] / self.h_gap**3
                      + st["cb"] / (self.h_gap + self.d_bh)**3) * st["vol"]
        Khat[-1] = 0.0
        if st["ring"]:
            Khat[0] = 0.0
        x = st["psi"].copy()
        nu = 0.0
        for _ in range(200):
            y = _solve_banded((1, 1), -J, Khat * x)
            nu_new = float(np.dot(x, Khat * y) / np.dot(x, Khat * x))
            x = y / np.max(np.abs(y))
            if abs(nu_new - nu) < 1e-13 * abs(nu_new):
                nu = nu_new
                break
            nu = nu_new
        return 1.0 / nu if nu > 0.0 else float("inf")

    def _st_softening_density(self, r):
        """Örtliche elektrostatische Erweichung [Pa/m] am Arbeitspunkt,
        λ·∂p/∂w — für den 3D-Löser (statt einer gleichförmigen, an der
        Grundmode kalibrierten negativen Steifigkeit)."""
        u = np.minimum((np.asarray(r, float) / self.a_mem) ** 2, 1.0)
        return np.interp(u, self._st["u"], self._st_soft)

    def _electrode_integrals(self, w0=0.0, profile=None):
        """Elektrodenintegrale über dem statischen Spaltprofil.

        In Modenkoordinate u = r²/a_mem² (dS = S_mem·du, Elektrode bis
        u <= ub) mit lokalem Spalt g = h − w(u):
            I_F = Int psi/g²  dS   (Kraft-/Kapazitätsmodulation)
            I_k = Int psi²/g³ dS   (negative Steifigkeit, Ein-Moden-Bild)
            I_C = Int 1/g     dS   (Ruhekapazität)
        psi ist die dynamische Modenform der Kette (statische Form, max 1).
        ``profile`` ist die statische Auslenkung auf ``_es_u`` (seit
        Gegenprobe 49 die EXAKTE Lösung, s. _static_setup); ohne Profil
        gilt das Ein-Moden-Bild w = w0·psi. Die Porosität geht als
        RADIALES Profil ein (gleiche Lochdichten wie im Feldmodell):
        solide Elektrodenfläche wiegt mit c_solid(u), über Blindlöchern
        gilt der vergrößerte Feldweg g + Tiefe (Durchgangslöcher tragen
        nichts). Negative Auslenkung beschreibt die von der Platte weg
        gebogene Membran (vordere Backplate der Dual-Architektur).
        """
        u = self._es_u
        v = self._es_phi                          # Modenprofil psi (max 1)
        w = w0 * v if profile is None else np.asarray(profile, float)
        g_s = self.h_gap - w                      # Spalt, solide Elektrode
        g_b = self.h_gap + self.d_bh - w          # Feldweg über Blindloch
        c_s = self._es_c_solid
        c_b = self._es_c_blind
        S = self.S_mem
        I_F = S * np.trapezoid(c_s * v / g_s**2 + c_b * v / g_b**2, u)
        I_k = S * np.trapezoid(c_s * v**2 / g_s**3 + c_b * v**2 / g_b**3, u)
        I_C = S * np.trapezoid(c_s / g_s + c_b / g_b, u)
        return I_F, I_k, I_C

    def _output_weight_3d(self, r):
        """Θ-konsistentes Ausgangsgewicht je Fläche für den 3D-Löser
        (Gegenprobe 54).

        Die Kette wandelt die Volumenverschiebung V ihrer Grundmode mit
        e = Θ·V; Θ enthält das Elektrodenintegral über der Modenform
        (s. _derive_parameters). Der 3D-Löser hat stattdessen das volle
        Auslenkungsfeld w(r, φ), und für jede Form gilt physikalisch

            e = Σ_Seiten (U0/I_C) · ∫_El w·(c_s/g_s² + c_b/g_b²) dS

        (dasselbe Spaltprofil, dieselbe Porosität wie Θ und C0). Ausgegeben
        wird die Volumenverschiebung V_eq = e/Θ = ∫ W(r)·w dS, die mit Θ
        genau diese Spannung ergibt. Für die Grundmode ist V_eq exakt die
        Volumenverschiebung der GANZEN Membran, wie in der Kette;
        außerhalb der Elektrode ist W = 0. Bis Gegenprobe 54 gab der
        3D-Löser die Volumenverschiebung über der Elektrode aus (W = 1
        dort) und war damit bei a_bp < a_mem um u·(2 − u), u = (a_bp/a_mem)²,
        zu leise (B&K 4134: −1,08 dB).
        """
        u = np.minimum((np.asarray(r, dtype=float) / self.a_mem) ** 2, 1.0)
        inside = (u >= self._es_u[0]) & (u <= self._es_u[-1])
        signs = (+1.0, -1.0) if self.architecture == "dual" else (+1.0,)
        num = np.zeros_like(u)
        den = 0.0
        for sign in signs:
            prof = sign * self._es_w
            I_F, _, I_C = self._electrode_integrals(profile=prof)
            dens = (self._es_c_solid / (self.h_gap - prof) ** 2
                    + self._es_c_blind / (self.h_gap + self.d_bh - prof) ** 2)
            num += np.where(inside, np.interp(u, self._es_u, dens), 0.0) / I_C
            den += (1.0 / self._phi_m1) * I_F / (self.S_mem * I_C)
        return num / den

    def _membrane_tension_3d(self):
        """Membranspannung des 3D-Felds und der Homogenisierungsgrenze
        [N/m] (Gegenprobe 54).

        Mit vorgegebener Resonanz ist es die Spannung, deren EXAKTE
        Grundmode f_res trifft. Mit vorgegebener Vorspannung ist es die
        physikalische, über die exakte Modalfrequenz (Vorspannung plus
        Biegeanteil der Folie, s. f_res_modal_exact). Bis Gegenprobe 54
        wurde sie auch dann aus der Resonanz der Kette zurückgerechnet.
        Die lag mit dem Kolbenfaktor 4/3 der statischen Form 1,9 % zu
        hoch, die Spannung also 3,75 % und die Nachgiebigkeit war
        entsprechend zu klein (B&K 4134: −0,3 dB im Tiefton). Seit dem
        Massenfaktor 8/(z1²·g) (Gegenprobe 55) trifft die Kette den
        Vorspannungsanteil selbst exakt; beide Wege unterscheiden sich
        nur noch über die Biegesteifigkeit der Folie (Promille).
        """
        sigma = self.sigma_mem
        z1 = float(self._ring_modes()["z"][0])
        f_t = (self.f_res_modal_exact
               if self.f_res_user is None and self._TENSION_EXACT_3D
               else self.f_res)
        return sigma * (2.0 * np.pi * f_t * self.a_mem / z1) ** 2

    def pullin_voltage(self, u_max=20000.0):
        """Maximal stabile Polarisationsspannung (Pull-in) — der Faltpunkt
        der exakten statischen Randwertaufgabe bzw. bei 'dual' das
        Eigenwertkriterium am Ruhespalt (Gegenprobe 49)."""
        return self.U_pullin if self.U_pullin <= u_max else float("inf")

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
        w1 = 1.0 / np.sqrt(self.M_A_mem * self.C_A_mem)
        out = []
        if self.r_post <= 0.0:
            x = self._J0_ZEROS
            for m in range(1, self.membrane_modes):
                rat = x[m] / x[0]
                M_m = self.M_A_mem * rat**2
                C_m = 1.0 / ((w1 * rat) ** 2 * M_m)
                out.append((M_m, C_m))
            return out
        # RINGMEMBRAN: das Massenverhältnis ist nicht mehr (z_m/z_1)²,
        # sondern der exakte Quotient der Modenintegrale I2/I1² — die
        # Frequenzstaffelung bleibt z_m/z_1 (s. _ring_modes).
        md = self._ring_modes()
        mu = md["I2"] / md["I1"] ** 2
        for m in range(1, self.membrane_modes):
            rat = md["z"][m] / md["z"][0]
            M_m = self.M_A_mem * float(mu[m] / mu[0])
            C_m = 1.0 / ((w1 * rat) ** 2 * M_m)
            out.append((M_m, C_m))
        return out

    def _modal_internal_Z(self, omega, h_film):
        """Innere Umverteilungsimpedanz der HÖHEREN Membranmoden.

        WAS HIER FEHLTE
        ---------------
        Das Ketten-Zweitor trägt den NETTO-Volumenfluss der Membran zu den
        Bohrungen — für die Kolbenmode ist das vollständig. Eine höhere
        Mode J0(z_m·r/a) schiebt die Spaltluft zusätzlich ZWISCHEN ihren
        Knotenringen hin und her, ohne dass dabei netto Luft die Platte
        verlässt. Diese Umverteilung ist dissipativ und träge, und sie kam
        im Modell bisher überhaupt nicht vor: die Modenzweige trugen seit
        Gegenprobe 31 nur noch die Materialdämpfung und hatten Güten um
        1e4 (Gegenprobe 34 hielt das als Grenze fest).

        HERLEITUNG (kein Fit)
        ---------------------
        Reynolds im Spalt mit der Filmleitfähigkeit K_f(ω) und der
        homogenisierten Lochadmittanz Y_h (Volumenfluss je Fläche und Pa):

            K_f·∇²p − Y_h·p = v(r),   v = v̂·ψ_m,  ∇²ψ_m = −k_m²·ψ_m
            =>  p = −v̂·ψ_m / (K_f·k_m² + Y_h)

        Die Modenimpedanz folgt aus Kraft je NETTO-Fluss, also mit
        ∫ψ_m² dA = S·J1(z_m)² und ∫ψ_m dA = S·2J1(z_m)/z_m:

            Z_m = [S·J1²/(K_f k_m² + Y_h)] / [S²·4J1²/z_m²]
                = 1 / (4π·(K_f + Y_h·a²/z_m²))          mit k_m = z_m/a.

        Bemerkenswert: OHNE Löcher ist das MODENUNABHÄNGIG — das z_m² der
        Bezugsgröße kürzt sich exakt gegen das 1/k_m² des Drucks. Mit
        K_f = (h³/12μ)/Φ(ω) ist der lochfreie Grenzfall
        Z = 3μ/(π·h³)·Φ(ω), also genau die schon verifizierte
        Frequenzkorrektur des Spaltfilms — Poiseuille bei tiefen
        Frequenzen, laterale Trägheit zu hohen hin.

        Die LOCHENTLASTUNG Y_h·a²/z_m² ist unverzichtbar: bei dicht
        gelochten Platten drainiert jede Zelle lokal, dann gibt es kaum
        modenweite Umverteilung. Sie verschwindet wie 1/z_m² — feine
        Moden „sehen" die Bohrungen nicht mehr. Der lochfreie Grenzfall
        ist also genau dort scharf, wo der Term gebraucht wird.

        NUR für m >= 2. Die Grundmode ist im Ketten-Zweitor vollständig
        enthalten (Škvor bzw. das 2D-Feld, das ihr Profil ohnehin
        auflöst) — sie hier nochmals zu belasten wäre die Doppelzählung,
        die Gegenprobe 31 beseitigt hat.

        Näherung, bewusst: der Modenradius ist a_mem, der Film reicht nur
        bis a_bp. Für a_bp < a_mem ist die Umverteilung damit leicht
        überschätzt.
        """
        omega = np.asarray(omega, dtype=float)
        K_f = ((h_film**3 / (12.0 * MU_AIR))
               / self._film_R_dynamic(omega, h_film))
        if self.n_th > 0 and self.S_bp > 0.0:
            Z_h = self._hole_impedance(omega, self.r_th, self.t_bp,
                                       self.n_th, end_correction=True)
            Y_h = 1.0 / (Z_h * self.S_bp)
        else:
            Y_h = 0.0
        out = []
        if self.r_post <= 0.0:
            for m in range(1, self.membrane_modes):
                z_m = self._J0_ZEROS[m]
                out.append(1.0 / (4.0 * np.pi
                                  * (K_f + Y_h * self.a_mem**2 / z_m**2)))
            return out
        # RINGMEMBRAN: allgemein Z_m = (I2/I1²)/(S·(K_f·k_m² + Y_h)); für
        # die Vollmembran ist I2/I1² = z²/4 und das reduziert sich exakt
        # auf die Zeile darüber.
        md = self._ring_modes()
        for m in range(1, self.membrane_modes):
            z_m = md["z"][m]
            fac = float(md["I2"][m] / md["I1"][m] ** 2)
            out.append(fac / (self.S_mem
                              * (K_f * (z_m / self.a_mem) ** 2 + Y_h)))
        return out

    def _modal_split_factor(self):
        """Normierung der Modenaufteilung, s_N = Σ_{j<=N} (z_1/z_j)^4.

        WARUM DAS NÖTIG IST
        -------------------
        Die statische Nachgiebigkeit einer Membran ist eine FESTE Zahl,
        πa⁴/(8T), unabhängig davon, mit wie vielen Moden man sie
        beschreibt. In der exakten Modalzerlegung verteilt sie sich als

            C_m = πa⁴/(8T) · 32/z_m⁴,   Σ_m C_m = πa⁴/(8T)

        (Rayleigh-Summe Σ 1/z_m⁴ = 1/32, s. Gegenprobe 42). Die
        Grundmode allein trägt davon nur 32/z_1⁴ = 95.68 %.

        Unser Ein-Freiheitsgrad-Modell ist aber KEINE Mode-1 dieser Reihe,
        sondern die klassische Lumped-Näherung: Nachgiebigkeit = exakter
        statischer Wert, Masse so, dass die Grundresonanz exakt ist
        (8/j01², Gegenprobe 55). Für einen Freiheitsgrad ist das richtig. Legt man die höheren
        Moden ADDITIV daneben (jede mit C_A_mem·(z_1/z_m)^4), zählt die
        Reihe die Nachgiebigkeit doppelt: gemessen +0.30/+0.35/+0.36/+0.37
        dB Tiefton bei 2/3/4/5 Moden — ein Effekt, den es nicht gibt.

        Der Ausweg ohne Sprung bei N = 1: ALLE Zweigimpedanzen mit s_N
        multiplizieren. Die Admittanzen sinken damit um 1/s_N, die Summe
        der Nachgiebigkeiten wird für JEDES N exakt die des Ein-Moden-
        Modells, und weil Masse und Nachgiebigkeit jedes Zweigs im
        gleichen Verhältnis wandern, bleiben alle Modenresonanzen
        unverändert. N = 1 -> s_1 = 1: bitgleich zum Bestand.

        Hochtongrenzwert: die exakten Modenmassen erfüllen Σ 1/M_m = S/σ —
        eine vielmodige Membran verhält sich weit oberhalb aller
        Resonanzen wie ein freier KOLBEN. Ohne Normierung liefe unsere
        Reihe über diesen Wert hinaus (5 Moden −0.31 dB, 10 Moden +0.03,
        50 Moden +0.31 — sie wird mit mehr Termen SCHLECHTER); mit
        Normierung liegt sie bei 5/10/50 Moden 0.69/0.35/0.07 dB darunter
        und konvergiert monoton EXAKT auf den Kolben: mit dem Massenfaktor
        μ = 8/j01² ist s_∞·μ = (j01⁴/32)·(8/j01²) = j01²/4, jeder Zweig
        trägt dann für N -> ∞ genau die exakte Modenmasse und
        -nachgiebigkeit (Gegenprobe 55). Mit dem früheren 4/3 lief die
        Reihe 0.32 dB ÜBER den Kolben hinaus.

        Die elektrostatische Feder-Erweichung wirkt weiterhin nur auf die
        Grundmode; durch die Normierung wird ihr Beitrag um 1/s_N
        verdünnt (bei 13.5 % Erweichung und 5 Moden 0.05 dB).
        """
        if self.membrane_modes <= 1:
            return 1.0
        if self.r_post <= 0.0:
            x = self._J0_ZEROS
            return float(sum((x[0] / x[m]) ** 4
                             for m in range(self.membrane_modes)))
        # RINGMEMBRAN: C_m ∝ I1²/(z² I2) statt 1/z⁴ (s. _ring_modes).
        md = self._ring_modes()
        c = md["I1"] ** 2 / (md["z"] ** 2 * md["I2"])
        return float(np.sum(c[:self.membrane_modes]) / c[0])

    def _modal_parallel(self, omega, Z1, R, h_film=None):
        """Grundmode Z1 mit den höheren Moden PARALLEL schalten.

        Die höheren Zweige tragen zusätzlich ihre innere Umverteilung im
        Spaltfilm (s. :meth:`_modal_internal_Z`); ohne sie wären sie
        praktisch ungedämpft. Die Aufteilung ist nachgiebigkeits-erhaltend
        normiert (s. :meth:`_modal_split_factor`).
        """
        branches = self._higher_mode_branches()
        if not branches:
            return Z1
        Z_int = (self._modal_internal_Z(omega, h_film)
                 if h_film is not None else [0.0] * len(branches))
        Y = 1.0 / Z1
        for (M_m, C_m), Zi in zip(branches, Z_int):
            Y = Y + 1.0 / (R + Zi + 1j * omega * M_m
                           + 1.0 / (1j * omega * C_m))
        return self._modal_split_factor() / Y

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
        return self._modal_parallel(omega, Z1, R, self.h_gap_front)

    def _membrane_film_damping(self, omega, h_film, R_A_gap):
        """Innere Materialdämpfung der Membran — OHNE Spaltfilm.

        KEINE DOPPELZÄHLUNG DES SPALTFILMS (Gegenprobe 31)
        --------------------------------------------------
        Bis Gegenprobe 30 stand hier ``R_A_mem + R_A_gap·Φ(ω)``: die
        Spaltfilm-Dämpfung wurde zusätzlich der Membranimpedanz
        zugeschlagen. Sie steckt aber bereits VOLLSTÄNDIG im
        Backplate/Spalt-Zweitor, das in derselben Kette in Serie folgt —
        nachweisbar an dessen Eingangsimpedanz bei kurzgeschlossenem Port
        und widerstandsarmen Bohrungen:

            Z_in = T12/T22 = R_A_gap   (1D: Verhältnis 1.0004;
                                        2D-Feld: 1.2007 = 6/5, der
                                        kinetische Profilfaktor)

        Der Membranfluss sah damit 2·R_gap statt R_gap. Physikalisch ist
        es EIN Weg — die Piston-Bewegung drückt die Spaltluft lateral zu
        den Senken —, also einmal zu zählen. Hier bleibt nur die
        Eigendämpfung der Folie (Materialgüte _Q_MEMBRANE_INTERNAL).

        BELEG am DRUCKEMPFÄNGER (der einzige unverfälschte Leitfall:
        Gradientenbauformen hängen an einer Auslöschung und reagieren auf
        jede Phasenänderung überempfindlich). Empfindlichkeit bei 4 kHz
        gegen den 3D-Feldlöser, der die Löcher diskret auflöst, bei
        konstanter Lochfläche:

            n_th      12     24     48     96    192
            einfach  -5.5   -1.3   +0.4   +0.4   +2.5   dB
            doppelt -10.8   -6.5   -4.5   -3.7   -0.5   dB

        Im Gültigkeitsbereich der Homogenisierung (48-96 Bohrungen) trifft
        die einfache Zählung den Feldlöser auf 0.4 dB, die doppelte liegt
        4 dB daneben. Bei sehr spärlichen Rastern (12-24) versagen beide —
        dort ist die azimutale Auflösung des 3D-Lösers nötig.
        (Stand Gegenprobe 31, 50 V. Mit dem in Gegenprobe 48 korrigierten
        3D-Löser und dem exakten Arbeitspunkt aus Gegenprobe 49 — der
        Prüfling läuft jetzt mit 45 V, weil 50 V genau auf seinem Pull-in
        liegen — lautet die Zeile „einfach“ −5.8 / −1.2 / +0.2 / +0.3 /
        −0.5 dB. Die Entscheidung bleibt dieselbe, und bei 192 Bohrungen
        fällt die frühere Unstimmigkeit von +2.5 dB weg.)

        ``h_film``/``R_A_gap`` bleiben in der Signatur, damit die
        Aufrufstellen unverändert lesbar sind.
        """
        return self.R_A_mem

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
        return self._modal_parallel(omega, Z1, R, self.h_gap)

    # ======================================================================
    # Beugung / Druckstau am Kapselkörper
    # ======================================================================
    def _modal_source_scale(self, omega, theta, F_mode1):
        """Korrekturfaktor p_eff/p_1 des MODENABHÄNGIGEN Quelldrucks.

        Jede Membranmode wird von ihrer eigenen Galerkin-Projektion
        <p_f · psi_0m> getrieben, nicht von einem gemeinsamen Skalar
        (Šimonová/Honzík, JASA 159, 4512 (2026), Gl. 5 + 24). Im
        Kettenmodell liegen die Moden PARALLEL an demselben Spaltknoten,
        deshalb lässt sich das exakt auf eine Ersatzquelle zusammenziehen:

            Σ_m Y_m (p_m − p_gap) = (Σ_m Y_m)·(p_eff − p_gap),
            p_eff = Σ_m Y_m p_m / Σ_m Y_m.

        Das ist keine Näherung, solange alle Moden an denselben
        nachgelagerten Knoten koppeln — im 1D/2D-Pfad ist das der Fall.
        Zurückgegeben wird p_eff/p_1, weil die Kette bereits p_1 führt.

        Die Modenfaktoren p_m/p_1 kommen aus derselben Projektion wie die
        Grundmode:
          * mit Beugung aus der Kalotten-/BEM-Oberflächenmittelung mit
            dem Gewicht J0(z_0m·r/a_mem),
          * ohne Beugung (freie ebene Welle) in geschlossener Form
                D_m(u) = z_0m²·J0(u)/(z_0m² − u²),   u = k·a_mem·sin(theta),
            dem Bessel-Produktintegral (Gl. A2).

        Bei ``membrane_modes = 1`` bleibt nur die Grundmode: der Faktor ist
        dann mit Beugung exakt 1 (die Kette führt bereits die richtige
        Projektion) und ohne Beugung D_1(u) — der APERTUREFFEKT der freien
        Membran, den der Pfad ohne Körper bisher gar nicht kannte.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        z = self._J0_ZEROS
        nm = self.membrane_modes

        # Modenfaktoren p_m/p_1, Form (nm, Nomega, Ntheta)
        if self.include_diffraction and _HAS_SCIPY:
            if nm == 1:
                return np.ones((omega.size, theta.size), dtype=complex)
            if self.axial_body_model == "bem":
                # Die Kette führt hier F_1 aus dem BEM — die Verhältnisse
                # müssen aus DEMSELBEN Körper kommen, sonst mischen sich
                # flache Stirnfläche und Kugelkalotte in einer Größe.
                # Derselbe Lösungsgang, nur andere Projektion.
                F_all, _ = self._bem_front_modes(omega, theta)
                rel = F_all / F_all[0]
            else:
                rel = [np.ones((omega.size, theta.size), dtype=complex)]
                for m in range(1, nm):
                    F_m, _ = self._diffraction_factors(omega, theta,
                                                       mode=m + 1)
                    rel.append(F_m / F_mode1)
                rel = np.array(rel)
        elif self.r_post > 0.0:
            # RINGMEMBRAN im Freifeld: dasselbe Lommel-Integral, nur mit
            # der Ringmodenform. Mit C0(z) = 0 und C1(zρ) = 2/(πzρ) ist
            #     ∫ψ_m J0(u r/a) dA / ∫ψ_m dA
            #       = 2[z·C1(z)·J0(u) − (2/π)·J0(uρ)] / ((z²−u²)·I1),
            # und der Zähler verschwindet bei u = z exakt mit dem Nenner
            # (Wronski) — die Singularität ist hebbar wie im Vollkreis.
            md = self._ring_modes()
            rho = self.rho_post
            u = np.outer(omega / C_AIR * self.a_mem, np.sin(theta))
            D = []
            for m in range(nm):
                zm = md["z"][m]
                C1 = (_besselj(1, zm) * _bessely(0, zm * rho)
                      - _bessely(1, zm) * _besselj(0, zm * rho))
                num = (zm * C1 * _besselj(0, u)
                       - (2.0 / np.pi) * _besselj(0, u * rho))
                den = zm**2 - u**2
                lim = ((zm * C1 * _besselj(1, zm)
                        - (2.0 * rho / np.pi) * _besselj(1, zm * rho))
                       / (2.0 * zm))
                safe = np.where(np.abs(den) < 1e-9, 1.0, den)
                Dm = np.where(np.abs(den) < 1e-9, lim, num / safe)
                D.append((2.0 * Dm / md["I1"][m]).astype(complex))
            rel = np.array(D)                      # absolut, wie unten
        else:
            u = np.outer(omega / C_AIR * self.a_mem, np.sin(theta))
            D = []
            for m in range(nm):
                den = z[m] ** 2 - u**2
                # Hebbare Singularität bei u = z_0m: Grenzwert z*J1(z)/2
                safe = np.where(np.abs(den) < 1e-9, 1.0, den)
                Dm = z[m] ** 2 * _j0_mode(u) / safe
                if _HAS_SCIPY:
                    from scipy.special import j1 as _j1_ms
                    lim = 0.5 * z[m] * _j1_ms(z[m])
                    Dm = np.where(np.abs(den) < 1e-9, lim, Dm)
                D.append(Dm.astype(complex))
            # ABSOLUT, nicht auf die Grundmode normiert: der Pfad ohne
            # Beugung führt in der Kette den UNIFORMEN Druck p = 1 und
            # kennt den Aperturfaktor bisher gar nicht. Im Beugungspfad
            # ist es umgekehrt — dort trägt die Kette bereits F_1.
            rel = np.array(D)

        if nm == 1:
            return rel[0] if rel.ndim == 3 else rel

        # Modenadmittanzen Y_m(omega) — es müssen GENAU dieselben Zweige
        # sein, die _modal_parallel danach parallel schaltet, sonst
        # gewichtet die Ersatzquelle anders als das Netzwerk rechnet.
        # Insbesondere gehört die innere Umverteilung im Spaltfilm
        # (_modal_internal_Z) dazu: ohne sie sind die höheren Zweige
        # praktisch ungedämpft, ihre Admittanz schießt an der eigenen
        # Resonanz hoch und zieht p_eff dort auf p_m — im Frequenzgang
        # als scharfe Senke bei der zweiten Modenfrequenz sichtbar, die
        # es in der Messung nicht gibt (Gegenprobe 43).
        # Der gemeinsame Normierungsfaktor s_N von _modal_split_factor
        # kürzt sich hier heraus (Zähler und Nenner), deshalb steht er
        # nicht dabei.
        R = self._membrane_film_damping(omega, self.h_gap_front,
                                        self.R_A_gap_front)
        R = np.asarray(R, dtype=complex) * np.ones_like(omega, dtype=complex)
        Y = [1.0 / (R + 1j * omega * self.M_A_mem
                    + 1.0 / (1j * omega * self.C_A_eff))]
        Z_int = self._modal_internal_Z(omega, self.h_gap_front)
        for (M_m, C_m), Zi in zip(self._higher_mode_branches(), Z_int):
            Y.append(1.0 / (R + Zi + 1j * omega * M_m
                            + 1.0 / (1j * omega * C_m)))
        Y = np.array(Y)[:, :, None]                 # (nm, Nomega, 1)
        return np.sum(Y * rel, axis=0) / np.sum(Y, axis=0)

    def _cap_mode_quad(self, n_nodes=48, mode=1):
        """Knoten u und NORMIERTE Modengewichte der Membrankalotte.

        Der Antrieb einer Membranmode ist nicht der Flächenmittelwert des
        Frontdrucks, sondern die Galerkin-Projektion <p_f · psi> / <psi>
        (Šimonová/Honzík, JASA 159, 4512 (2026), Gl. 5; klassisch bereits
        bei Lavergne et al.). Für die Grundmode ist das Gewicht die
        Modenform selbst,

            w(r) = J0(z01 · r / a_mem),

        auf die Kalotte abgebildet über r = R_body · sin(psi), also
        w(u) = J0(z01 · sqrt(1−u²) · R_body / a_mem) mit u = cos(psi).
        Die Integration läuft über dA = 2π R² du, deshalb ist die
        Gauss–Legendre-Quadratur direkt in u exakt richtig.

        ``mode`` wählt die Bessel-Mode (1 = Grundmode); die höheren
        brauchen dasselbe Gewicht mit ihrer eigenen Nullstelle z_0m und
        werden für den modenabhängigen Quelldruck benötigt
        (s. :meth:`_modal_source_scale`).

        Rückgabe: (u, w) mit Σ w = 1, sodass <f>_Mode = Σ w_i f(u_i).
        Für eine Punktmembran (u0 → 1) leere Arrays — dort ist C_n = 1.

        Grenzfall w ≡ 1 wäre das bisherige flächengleiche Mittel; der
        Unterschied ist im Freifeld A(u) = 2J1(u)/u gegen
        D(u) = z01²J0(u)/(z01²−u²) und wird in Gegenprobe 33 geprüft.
        """
        u0 = self._cap_cos
        if (1.0 - u0) < 1e-9:
            return np.empty(0), np.empty(0)
        x, w = np.polynomial.legendre.leggauss(int(n_nodes))
        u = 0.5 * (1.0 + u0) + 0.5 * (1.0 - u0) * x        # -> [u0, 1]
        wq = 0.5 * (1.0 - u0) * w
        s = np.sqrt(np.clip(1.0 - u * u, 0.0, None))       # sin(psi)
        # Klammerung am EIGENEN Membranrand (und, bei Mitten-
        # terminierung, am Pfostenrand): dort ist die Modenform null.
        wq = wq * self._membrane_mode_weight(s * self.R_body,
                                             max(int(mode), 1))
        tot = float(np.sum(wq))
        if not np.isfinite(tot) or abs(tot) < 1e-300:
            return np.empty(0), np.empty(0)
        return u, wq / tot

    def _ring_modes(self):
        """Eigenwerte und Modenintegrale der Membran (gecacht).

        OHNE Mittenterminierung sind das die bekannten J0-Nullstellen mit
        ψ_m(r) = J0(z_m·r/a). MIT Mittenterminierung ist die Membran ein
        RING, und die Modenform ist die Zylinderfunktions-Kombination, die
        an BEIDEN Rändern verschwindet:

            ψ_m(r) = J0(k r)·Y0(k r_i) − Y0(k r)·J0(k r_i) =: C0(k r),
            Eigenwertgleichung  C0(k a) = 0.

        Zurückgegeben wird ein dict mit

            z   — z_m = k_m·a_mem,
            I1  — ∫ψ_m dA / S_mem,
            I2  — ∫ψ_m² dA / S_mem,

        beides in GESCHLOSSENER Form. Mit C1(x) = J1(x)Y0(kr_i) −
        Y1(x)J0(kr_i), C0(z) = 0 und der Wronski-Identität
        C1(z·ρ) = 2/(π·z·ρ):

            I1 = 2·[C1(z)/z − 2/(π z²)],
            I2 = C1(z)² − 4/(π² z²).

        Daraus folgen Modenmasse und -nachgiebigkeit exakt:

            M_m = ρ_s·(I2/I1²)/S,   C_m = S·I1²·a²/(T·z²·I2),

        und die Summe Σ C_m trifft die statische Nachgiebigkeit der
        Ringmembran auf acht Stellen (Gegenprobe 45) — das ist die
        Verallgemeinerung der Rayleigh-Summe Σ1/z⁴ = 1/32.
        """
        cached = getattr(self, "_ring_mode_cache", None)
        if cached is not None:
            return cached
        n = max(self.membrane_modes, len(self._J0_ZEROS))
        if self.r_post <= 0.0:
            z = np.array(self._J0_ZEROS[:n], dtype=float)
            j1z = _j1_mode(z)
            self._ring_mode_cache = dict(z=z, I1=2.0 * j1z / z, I2=j1z**2)
        else:
            self._ring_mode_cache = self._ring_eigen(self.rho_post, n)
        return self._ring_mode_cache

    @staticmethod
    def _ring_eigen(rho, n):
        """Erste ``n`` Ringmoden zu rho = r_i/a: dict(z, I1, I2).

        Herleitung und Bedeutung s. :meth:`_ring_modes`. Als eigene
        Funktion, damit Gegenprobe 45 die Reihe über HUNDERTE Moden
        summieren kann, ohne dafür eine Kapsel bauen zu müssen.
        """
        f = (lambda x: _besselj(0, x) * _bessely(0, x * rho)
             - _bessely(0, x) * _besselj(0, x * rho))
        z, x0 = [], 1e-6
        step = 0.02 * np.pi / max(1.0 - rho, 1e-3)
        prev = f(x0)
        while len(z) < n and x0 < 1.0e5:
            x1 = x0 + step
            cur = f(x1)
            if np.isfinite(prev) and np.isfinite(cur) and prev * cur < 0:
                lo, hi = x0, x1
                for _ in range(80):              # Bisektion, robust
                    mid = 0.5 * (lo + hi)
                    if f(lo) * f(mid) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                z.append(0.5 * (lo + hi))
            x0, prev = x1, cur
        if len(z) < n:
            raise ValueError(
                "Mittenterminierung: Ring-Eigenwerte nicht gefunden.")
        z = np.array(z, dtype=float)
        C1 = (_besselj(1, z) * _bessely(0, z * rho)
              - _bessely(1, z) * _besselj(0, z * rho))
        return dict(z=z, I1=2.0 * (C1 / z - 2.0 / (np.pi * z**2)),
                    I2=C1**2 - 4.0 / (np.pi**2 * z**2))

    def _membrane_mode_weight(self, r, mode=1):
        """Modengewicht ψ_m(r), außerhalb der Membran 0.

        Gewicht der Galerkin-Projektion des Frontdrucks auf die
        (0,m)-Membranmode (s. :meth:`_cap_mode_quad`); für Flächenstücke
        jenseits des Membranrandes — und, bei Mittenterminierung, INNERHALB
        des Pfostens — null, weil dort keine Mode sitzt. ``mode`` ist
        1-basiert, ``mode=1`` ist die Grundmode. Die Normierung ist
        beliebig: überall, wo das Gewicht auftritt, steht es in einem
        Quotienten mit seiner eigenen Summe.
        """
        r = np.asarray(r, dtype=float)
        if self.r_post <= 0.0:
            x = self._J0_ZEROS[mode - 1] * np.clip(r / self.a_mem, 0.0, 1.0)
            return _j0_mode(x)
        z = self._ring_modes()["z"][mode - 1]
        x = z * np.clip(r / self.a_mem, self.rho_post, 1.0)
        return (_besselj(0, x) * _bessely(0, z * self.rho_post)
                - _bessely(0, x) * _besselj(0, z * self.rho_post))

    def _diffraction_factors(self, omega, theta, mode=1):
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
        gemittelte Kugelkalotte am vorderen Pol; die azimutale
        Mittelung ist über das Legendre-Additionstheorem exakt:
            <P_n(cos psi)>_Ring    = P_n(cos alpha) * P_n(cos theta)
            <P_n(cos psi)>_Kalotte = C_n * P_n(cos theta)

        Die Kalottenmittelung ist MODENGEWICHTET, nicht flächengleich
        (s. :meth:`_cap_mode_quad`): der Antrieb einer Membranmode ist die
        Galerkin-Projektion <p_f · psi> und nicht der schlichte Flächen-
        mittelwert. Für die Grundmode J0(z01·r/a) unterscheiden sich die
        beiden im Freifeld-Grenzfall als
            Flächenmittel  A(u) = 2·J1(u)/u
            Projektion     D(u) = z01²·J0(u)/(z01² − u²),   u = k·a·sin θ
        A(u) hat bei u = 3.83 eine Nullstelle, die die Grundmode gar nicht
        hat — ihre erste liegt bei u = 5.52. Das flächengleiche Mittel
        erzeugt dort also eine Auslöschung, die es physikalisch nicht
        gibt (Gegenprobe 33).

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
        P_ur = _legendre_table(np.array([self._ring_cos]), n_max + 2)
        u0 = self._cap_cos
        u_q, w_q = self._cap_mode_quad(mode=mode)  # Knoten + Modengewichte
        P_q = _legendre_table(u_q, n_max + 1) if u_q.size else None

        F_f = np.zeros((omega.size, theta.size), dtype=complex)
        F_r = np.zeros_like(F_f)
        for n in range(n_max + 1):
            if n == 0 or (1.0 - u0) < 1e-9 or P_q is None:
                C_n = 1.0                     # Punktmembran -> C_n = P_n(1)
            else:
                C_n = float(np.dot(w_q, P_q[n]))
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
    # Obergrenze der Meridian-Elementlänge; in _bem_geometry zusätzlich mit
    # der Körpergröße skaliert. Als Klassenkonstante, damit die Gitter-
    # konvergenz prüfbar ist (Gegenprobe 41).
    _BEM_H_MAX = 1.0e-3
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
        Rh, zf = self.R_body, 0.5 * self._bem_head_len
        zr = -zf
        # Kantenverrundung und Elementlänge SKALIEREN mit dem Körper, damit
        # die Kontur für jede Baugröße gültig bleibt (rf < R und
        # 2·rf < Länge sind damit automatisch erfüllt). Für Körper ab
        # ~4 mm Radius/Länge greifen die alten festen Werte unverändert.
        rf = min(0.8e-3, 0.2 * Rh, 0.2 * self._bem_head_len)
        h_el = min(self._BEM_H_MAX, 0.35 * Rh)
        segs = [("line", (0.0, zf), (Rh - rf, zf)),
                ("arc", (Rh - rf, zf - rf), rf, np.pi / 2, 0.0),
                ("line", (Rh, zf - rf), (Rh, zr + rf)),
                ("arc", (Rh - rf, zr + rf), rf, 0.0, -np.pi / 2),
                ("line", (Rh - rf, zr), (0.0, zr))]
        pts = self._bem_contour(segs, rf=rf, h=h_el)
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
        if self.architecture == "dual_diaphragm":
            # Rückseite ist die zweite MEMBRAN: Galerkin-Projektion mit
            # demselben Modengewicht wie vorn, sonst wäre G nicht das
            # Verhältnis zweier gleichartiger Antriebe.
            rear = ((np.abs(mz - zr) < 1e-6) & (mr <= self.a_mem)
                    & (np.arange(mr.size) < n_head))
            w_rear = w_area * rear * self._membrane_mode_weight(mr)
        else:
            w_rear = self._bem_rear_inlet_weights(elems, n_head, zf, zr, rf)
        self._bem_geo = dict(elems=elems, chief=chief, w_area=w_area,
                             front=front, w_rear=w_rear)
        return self._bem_geo

    def _bem_rear_inlet_weights(self, elems, n_head, zf, zr, rf):
        """Gewichte des RÜCKWÄRTIGEN EINLASSES auf der Kapselkontur.

        Gegenstück zum Ring der Kugelrechnung (_diffraction_factors,
        ``_ring_cos``), aber auf der realen Kontur: bei
        ``cavity_hole_position='end'`` in der hinteren Stirnfläche, sonst
        als Bohrungskranz radial im Mantel.

        TIEFE: gemessen wird ab der STIRNFLÄCHE der Kontur, und die ist
        der vordere Schalleinlass — bei ``architecture='dual'`` also die
        Außenseite der VORDEREN Backplate, nicht die Membranebene. Der
        Abstand dorthin ist genau ``d_ext``, die äußere Wegdifferenz, mit
        der das Modell ohnehin rechnet (bei einer Backplate ist sie
        gleich ``d_rear_ax``, bei zweien um Spalt + Plattendicke größer).
        Mit ``d_rear_ax`` gerechnet läge der Ring bei der symmetrischen
        Bauform um genau diesen Betrag zu weit vorn.

        Warum ein Ring und kein Flächenmittel: die m=0-Formulierung löst
        bereits den azimutal gemittelten Oberflächendruck — genau das,
        was ein Kranz gleichmäßig verteilter Bohrungen akustisch
        abgreift. Am Mantel wird zwischen den beiden benachbarten
        Elementringen LINEAR in z interpoliert, damit das Ergebnis nicht
        an der Elementteilung hängt (Gegenprobe 44). Kein Modengewicht:
        dort sitzt keine Membran, sondern Löcher.

        Rückgabe: Gewichtsvektor über ALLE Elemente (Summe > 0).
        """
        mr, mz = elems["mid_r"], elems["mid_z"]
        N = mr.size
        kopf = np.arange(N) < n_head
        w = np.zeros(N)
        z_in = zf - self.d_ext
        if self.cavity_hole_position == "end" or z_in <= zr + rf:
            # Hintere Stirnfläche: Flächenmittel über die Lochfläche.
            stirn = kopf & (np.abs(mz - zr) < 1e-6)
            sel = stirn & (mr <= self.a_bp)
            if not np.any(sel):
                sel = stirn
            w[sel] = 2.0 * np.pi * mr[sel] * elems["L"][sel]
            return w
        # Mantel: linear in z zwischen den beiden Nachbarringen
        wand = np.where(kopf & (mr > 0.9 * self.R_body)
                        & (mz < zf - 1e-9) & (mz > zr + 1e-9))[0]
        zs = mz[wand]
        o = np.argsort(zs)
        wand, zs = wand[o], zs[o]
        j = int(np.searchsorted(zs, z_in))
        if j == 0:
            w[wand[0]] = 1.0
        elif j >= zs.size:
            w[wand[-1]] = 1.0
        else:
            t = (z_in - zs[j - 1]) / (zs[j] - zs[j - 1])
            w[wand[j - 1]] = 1.0 - t
            w[wand[j]] = t
        return w

    def _bem_front_modes(self, omega, theta):
        """Frontfaktoren ALLER gebrauchten Membranmoden UND der Transfer.

        Aus EINEM m=0-BEM-Lösungsgang auf der Kontur Kopf + Körper:

            F_m = ⟨p⟩_Frontmembran,Mode m / p0,   Form (n_mod, Nω, Nθ)
            G   = ⟨p⟩_Rückeinlass / ⟨p⟩_Frontmembran,Mode 1,

        wobei der Rückeinlass bei der Doppelmembran-Bauform die zweite
        MEMBRAN ist und bei einer Ein-Membran-Kapsel der Bohrungskranz
        des rückwärtigen Einlasses (s. :meth:`_bem_rear_inlet_weights`).

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
        (validiert: Gegenprobe 26).

        MODENFAKTOREN: mit ``modal_source`` wird jede Membranmode von
        ihrer EIGENEN Galerkin-Projektion getrieben. Das Gewicht ist
        J0(z_0m·r/a_mem), die Randintegralgleichung selbst hängt davon
        nicht ab — deshalb wird EINMAL gelöst und n_mod-fach projiziert.
        Vorher kam die Grundmode aus dem BEM, die Verhältnisse p_m/p_1
        aber weiter aus der Kugelkalotte; das war die Mischung zweier
        Körpermodelle in einer Größe (s. Gegenprobe 43).

        WAS GERECHNET WIRD UND WAS NICHT (Laufzeit): teuer ist der Aufbau
        der Matrix, und der hängt NUR von ω ab — die Einfallsrichtungen
        sind bloß weitere rechte Seiten. Deshalb liegt am Geometrie-Objekt
        ein Ergebnisspeicher je (ω, θ): ein zweiter Aufruf mit denselben
        Frequenzen und einer Teilmenge der Winkel kostet nichts mehr. Das
        ist kein Luxus — der Frequenzgang läuft mit θ = 0/90/180°, das
        Eigenrauschen danach mit θ = 0° über dasselbe Frequenzraster, und
        ohne diesen Speicher rechnete der BEM alles ein zweites Mal.
        Ebenso wird der STATISCHE Kern K0 (frequenzunabhängig) einmal je
        Geometrie gebaut statt je Frequenz — das halbiert den Aufbau.

        Der letzte Satz wird zusätzlich identisch zurückgegeben, weil
        :meth:`_source_pressures` ihn zweimal braucht (Grundmode und
        Modenverhältnisse)."""
        from scipy.special import j0 as _bessel_j0
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        key = (omega.tobytes(), theta.tobytes())
        cached = getattr(self, "_bem_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        geo = self._bem_geometry()
        elems, chief = geo["elems"], geo["chief"]
        w_area, front, w_rear = geo["w_area"], geo["front"], geo["w_rear"]
        N = elems["L"].size
        cphi, wphi = self._bem_phi_quad()
        mr, mz = elems["mid_r"], elems["mid_z"]
        cr = np.array([c[0] for c in chief])
        cz = np.array([c[1] for c in chief])
        ct, st = np.cos(theta), np.sin(theta)

        def _pinc(k, r, z):
            return (_bessel_j0(np.outer(st * k, r))
                    * np.exp(-1j * np.outer(ct * k, z))).T   # (Npunkte, Nθ)

        # Die BEM-LÖSUNG hängt nicht von der Membranmode ab — nur die
        # Projektion danach. Deshalb wird einmal gelöst und auf alle
        # Moden projiziert, die dieses Objekt braucht (bei modal_source
        # sind das membrane_modes, sonst nur die Grundmode).
        n_mod = self.membrane_modes if self.modal_source else 1
        # Speicher und statischer Kern hängen am GEOMETRIE-Objekt: wer
        # eine andere Kontur injiziert (Gegenproben 21/26/44), bekommt
        # automatisch einen frischen Speicher.
        store = geo.setdefault("_cache", {})
        if len(store) > 60000:                    # Speicher begrenzen
            store.clear()
        if "_K0" not in geo:
            K0 = np.empty((N, N), dtype=complex)
            for b0 in range(0, N, 32):
                b1 = min(b0 + 32, N)
                K0[b0:b1] = self._bem_ring_rows(0.0, mr[b0:b1], mz[b0:b1],
                                                elems, cphi, wphi,
                                                static=True)
            rows0 = np.arange(N)
            row0 = np.sum(K0, axis=1) - K0[rows0, rows0]
            geo["_K0"] = (K0, row0,
                          float(np.max(np.abs(row0 + K0[rows0, rows0] + 0.5))))
        K0, row0, res0 = geo["_K0"]
        self._bem_solid_angle_residual = res0
        rows = np.arange(N)
        # Welche (ω, θ) fehlen noch?
        todo = []
        for i, om in enumerate(omega):
            miss = [j for j in range(theta.size)
                    if (float(om), float(theta[j])) not in store]
            if miss:
                todo.append((i, miss))
        wf_all = [w_area[front] * self._membrane_mode_weight(mr[front], mm + 1)
                  for mm in range(n_mod)]
        self._bem_solves = getattr(self, "_bem_solves", 0) + len(todo)
        for i, miss in todo:
            k = omega[i] / C_AIR
            K = np.empty((N, N), dtype=complex)
            for b0 in range(0, N, 32):
                b1 = min(b0 + 32, N)
                K[b0:b1] = self._bem_ring_rows(k, mr[b0:b1], mz[b0:b1],
                                               elems, cphi, wphi)
            # Diagonale: statische Identität Σ_j K0_ij = -1/2
            K[rows, rows] = (K[rows, rows] - K0[rows, rows]
                             + (-0.5 - row0))
            A = np.vstack([0.5 * np.eye(N) - K,
                           -self._bem_ring_rows(k, cr, cz, elems, cphi, wphi)])
            b = np.vstack([_pinc(k, mr, mz), _pinc(k, cr, cz)])[:, miss]
            # LAPACK setzt auf manchen Plattformen (Apple Accelerate)
            # Gleitkomma-Flags, die numpy erst beim NÄCHSTEN ufunc meldet
            # — als "divide by zero encountered in matmul" mitten in der
            # Projektion. Deshalb hier gekapselt und das ERGEBNIS geprüft,
            # statt sich auf Warnungen zu verlassen.
            with np.errstate(all="ignore"):
                u, *_ = np.linalg.lstsq(A, b, rcond=None)
                if not np.all(np.isfinite(u)):
                    raise ValueError(
                        f"BEM: die Randintegralgleichung ist bei "
                        f"{omega[i] / (2 * np.pi):.0f} Hz nicht lösbar "
                        f"({N} Elemente, Kondition "
                        f"{np.linalg.cond(A):.2e}). Körpermaße prüfen.")
                # FRONT: Galerkin-Projektion auf die Membranmode (Fläche ×
                # Modengewicht, s. _cap_mode_quad). RÜCK: was dort steht,
                # entscheidet die Geometrie — bei der Doppelmembran
                # dieselbe Projektion auf der zweiten Membran, bei einer
                # Ein-Membran-Kapsel der Bohrungskranz des Rückeinlasses
                # (s. _bem_rear_inlet_weights). Nur so ist G das
                # Verhältnis der beiden Antriebe, die die Kette sieht.
                p_f = np.array([(wf @ u[front]) / np.sum(wf)
                                for wf in wf_all])         # (n_mod, n_miss)
                p_r = (w_rear @ u) / np.sum(w_rear)         # (n_miss,)
            for jj, j in enumerate(miss):
                store[(float(omega[i]), float(theta[j]))] = (
                    np.conj(p_f[:, jj]), complex(np.conj(p_r[jj] / p_f[0, jj])))
        F = np.empty((n_mod, omega.size, theta.size), dtype=complex)
        G = np.empty((omega.size, theta.size), dtype=complex)
        for i, om in enumerate(omega):
            for j in range(theta.size):
                Fc, Gv = store[(float(om), float(theta[j]))]
                F[:, i, j] = Fc
                G[i, j] = Gv
        self._bem_cache = (key, F, G)
        return F, G

    def _bem_axial_fields(self, omega, theta):
        """Frontfaktor der GRUNDMODE und Front-Rück-Transfer.

        Dünner Aufsatz auf :meth:`_bem_front_modes` — dieselbe Rückgabe
        wie bisher, damit alle bestehenden Aufrufer unverändert bleiben.
        """
        F, G = self._bem_front_modes(omega, theta)
        return F[0], G

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
    def _hole_ring_layout(self, rings):
        """Lochkreisliste [(Anzahl, Radius), ...] mit aufgelösten
        Gleichverteilungen — die Lagekonvention des 3D-Lösers.

        ``(Anzahl, None)`` = gleichmäßig über die Elektrode. Das wird als
        ISOTROPES Raster aus Hilfskreisen gelegt (Gegenprobe 48): gleicher
        Ringabstand Δr, Lochzahl je Kreis proportional zu seinem Umfang,
        so dass die Teilung auf dem Kreis ≈ Δr ist. Das ist die Zelle, die
        Škvor und die 2D-Homogenisierung voraussetzen (rund, Fläche S/n).
        Bis Gegenprobe 47 lagen die Hilfskreise flächengleich mit GLEICHER
        Lochzahl — außen entstanden so Zellen von 0.9 × 10 mm, die den
        Film künstlich versteiften.
        """
        r0 = self.r_post
        out = []
        for cnt, r_pcd in rings:
            if cnt <= 0:
                continue
            if r_pcd is not None:
                out.append((cnt, r_pcd))
                continue
            span = self.a_bp - r0
            n_sub = max(1, int(round(np.sqrt(
                cnt * span / (np.pi * (self.a_bp + r0))))))
            mids = r0 + (np.arange(n_sub) + 0.5) * span / n_sub
            counts = np.maximum(np.round(cnt * mids / mids.sum()),
                                1).astype(int)
            # Rundungsdifferenz am äußersten (größten) Ring ausgleichen
            counts[-1] += cnt - int(counts.sum())
            for c_k, r_k in zip(counts, mids):
                if c_k > 0:
                    out.append((int(c_k), float(r_k)))
        return out

    def _hole_positions(self):
        """Lochmitten der (vorderen) Elektrode: {'th': [...], 'bh': [...]},
        je Eintrag (Radius [m], Winkel [°], Rückversatz [°]) — die Lage-
        konvention des 3D-Lösers. Um den Rückversatz ist die Gegen-
        elektrode kreisweise verdreht, damit die Durchgangslöcher beider
        Seiten nicht zueinander zeigen (automatische K67-Verdrehung,
        zweite Backplate bei 'dual'): auf expliziten Lochkreisen die
        halbe eigene Teilung, auf dem gemeinsamen Raster ein Platz (dann
        liegen die Durchgangslöcher der Gegenseite über den Sacklöchern)
        bzw. ein halber, wenn mehr als jeder zweite Platz durchgebohrt ist.

        * Explizite Lochkreise: gleichverteilt je Kreis, Durchgangs-
          kreis m um 20°·m verdreht, Sacklochkreis m um 15° + 20°·m
          (die realen Azimutwinkel sind selten dokumentiert).
        * GLEICHVERTEILTE Anteile beider Lochtypen bilden EIN gemeinsames
          isotropes Raster (s. _hole_ring_layout), auf dem sich Durch-
          gangs- und Sacklöcher gleichmäßig ABWECHSELN — wie an der realen
          K67 (120 Senkungen, jede zweite durchgebohrt). Bis Gegenprobe 47
          lagen beide Typen als getrennte Raster auf denselben Hilfskreisen,
          nur 15° gegeneinander verdreht: auf einem Kreis mit 26 Löchern
          (Teilung 13.8°) saß das Sackloch dann 1.2° neben dem Durchgangs-
          loch — die Mündungen überlappten (Gegenprobe 48).
        """
        cached = getattr(self, "_hole_pos_cache", None)
        if cached is not None:
            return cached
        th, bh = [], []
        m_th = 0
        for cnt, r_pcd in self._th_rings:
            if cnt > 0 and r_pcd is not None:
                for k in range(cnt):
                    th.append((r_pcd, 20.0 * m_th + 360.0 * k / cnt,
                               180.0 / cnt))
                m_th += 1
        m_bh = 0
        for cnt, r_pcd in self._bh_rings:
            if cnt > 0 and r_pcd is not None:
                for k in range(cnt):
                    bh.append((r_pcd, 15.0 + 20.0 * m_bh + 360.0 * k / cnt,
                               180.0 / cnt))
                m_bh += 1
        u_th = sum(c for c, r in self._th_rings if r is None and c > 0)
        u_bh = sum(c for c, r in self._bh_rings if r is None and c > 0)
        n_u = u_th + u_bh
        if n_u > 0:
            rings = self._hole_ring_layout([(n_u, None)])
            # Durchgangsanteil je Kreis, Rundungsrest am größten Kreis
            t_k = [int(round(c * u_th / n_u)) for c, _ in rings]
            t_k[-1] += u_th - sum(t_k)
            for k, ((c_k, r_k), t) in enumerate(zip(rings, t_k)):
                t = min(max(t, 0), c_k)
                # Rückversatz: belegen Durchgangslöcher höchstens jeden
                # zweiten Platz, liegen nie zwei nebeneinander — dann EIN
                # Platz weiter, und die Durchgangslöcher der Gegenseite
                # sitzen über den Sacklöchern dieser Seite. Sonst (mehr als
                # die Hälfte, z. B. 2 von 3) muss irgendwo ein Paar kollidieren;
                # dann ein halber Platz, zwischen die Löcher.
                turn = 360.0 / c_k if 2 * t <= c_k else 180.0 / c_k
                for j in range(c_k):
                    deg = 20.0 * k + 360.0 * j / c_k
                    # Bresenham-Verteilung: t von c_k Plätzen gleichmäßig
                    is_th = ((j + 1) * t) // c_k - (j * t) // c_k == 1
                    (th if is_th else bh).append((r_k, deg, turn))
        self._hole_pos_cache = {"th": th, "bh": bh}
        return self._hole_pos_cache

    def _drain_coverage_radius(self):
        """Überdeckungsradius der Durchgangslöcher [m]: der größte Abstand,
        den ein Punkt der Elektrode (r_post … a_bp) bis zur NÄCHSTEN
        Durchgangsbohrung hat — Lage wie im 3D-Löser (Ring m um 20°·m
        verdreht, Gleichverteilungen als isotropes Raster). Ein offener
        Randspalt zählt als Senke am Plattenrand.

        Das ist die Größe, an der die Homogenisierung der 1D/2D-Modelle
        scheitert (Gegenprobe 48): beide zwingen der Membran über jeder
        Lochzelle die GLOBALE Modenform auf. Reicht der lochfreie Bereich
        weit, verformt sich die gespannte Membran dort örtlich (sie weicht
        dem gestauten Film aus) — das kann nur der 3D-Löser. Bei
        gleichverteilten Löchern ist der Radius ≈ 1.3·a_bp/√n (Quadrat-
        raster: √(π/2) = 1.25), bei einem einzelnen Lochkreis der Abstand
        zur Mitte oder zum Rand.
        """
        cached = getattr(self, "_rho_cov", None)
        if cached is not None:
            return cached
        r0 = self.r_post
        if self.n_th == 0:
            if self.ring_vent_w > 0.0:
                self._rho_cov = self.a_bp - r0
            else:
                self._rho_cov = float("inf")
            return self._rho_cov
        pos = np.array([(r, d) for r, d, _ in self._hole_positions()["th"]])
        hx = pos[:, 0] * np.cos(np.deg2rad(pos[:, 1]))
        hy = pos[:, 0] * np.sin(np.deg2rad(pos[:, 1]))
        # Abtastung der Elektrode (Randpunkte eingeschlossen: dort liegt
        # das Maximum bei einzelnen Lochkreisen)
        rs = np.linspace(r0, self.a_bp, 121)
        ph = np.linspace(0.0, 2.0 * np.pi, 361)[:-1]
        best = 0.0
        for r_s in rs:
            px = r_s * np.cos(ph)
            py = r_s * np.sin(ph)
            d2 = ((px[:, None] - hx[None, :]) ** 2
                  + (py[:, None] - hy[None, :]) ** 2)
            dmin = np.sqrt(np.min(d2, axis=1))
            if self.ring_vent_w > 0.0:
                dmin = np.minimum(dmin, self.a_bp - r_s)
            best = max(best, float(np.max(dmin)))
        self._rho_cov = best
        return best

    def _polarized_gap_profile(self, r):
        """Örtlicher Spalt der polarisierten Seite h(r) = h − w(r) mit der
        EXAKTEN statischen Auslenkung (Gegenprobe 49) — dieselbe, die das
        2D-Feld über _fld_sag_shape benutzt. Ohne Boden; den setzt der
        Aufrufer."""
        u = np.minimum((np.asarray(r, float) / self.a_mem) ** 2, 1.0)
        return self.h_gap - np.interp(u, self._st["u"], self._w_static)

    def homogenization_limit(self):
        """Obere Frequenz, bis zu der die Loch-Homogenisierung der 1D/2D-
        Modelle gegen den 3D-Löser abgesichert ist (Gegenproben 48/53).

        PHYSIK. 1D und 2D verschmieren die Bohrungen zu einer Senken-
        dichte und zwingen der Membran über jeder Lochzelle die GLOBALE
        Modenform auf. Über einem lochfreien Bereich vom Radius ρ (der
        Überdeckungsradius, s. _drain_coverage_radius) staut sich aber
        der Film, und die gespannte Membran weicht ihm örtlich aus — sie
        beult sich zwischen den Löchern. Maßgeblich ist das Verhältnis der
        Filmkraft zur Steifigkeit der Beule auf dieser Skala (k = j01/ρ):

            Π(ω) = j01² · (ω / (|K(ω)|·k²)) / |T·k² − ω²·σ|.

        K(ω) ist die volle Filmleitfähigkeit (Reibung UND Trägheit der
        Spaltluft, dieselbe wie im Feld- und 3D-Modell), σ die Flächen-
        masse der Membran. Im Tiefton bleibt die bisherige Form

            Π = ω · 12μ·ρ⁴ / (h³ · T · j01²);

        beim weiten Spalt wächst |K0/K| mit der Schubzahl (die Luft im
        Spalt wird träge), und nahe der Beulresonanz

            f_ρ = (j01/(2π·ρ)) · √(T/σ)   (= f_res · (j01/z1) · a_mem/ρ)

        verschwindet die Steifigkeit der Beule — dort reicht jede Film-
        kraft (steife ½"-Kapsel, 65 µm, 16 Löcher: 2D weicht ab etwa
        19 kHz um mehr als 1 dB ab, f_ρ = 25,9 kHz; die rein viskose
        Grenze lag bei 107 kHz, die neue bei 17,6 kHz).
        Der Atmosphärendruck kürzt sich heraus; die Kompressibilität senkt
        die Filmkraft und bleibt deshalb auf der sicheren Seite weg. f_hom
        ist die Frequenz mit Π = _PI_HOM (10, vorsichtiger Rand, s. dort);
        Π wächst unterhalb f_ρ monoton, die Grenze ist eindeutig.

        LOCHKREISE: zusätzlich die Grenze der Lochkreis-Darstellung
        (s. _ring_repr_limit); ``f_limit`` ist die kleinere der beiden.

        GEPRÜFT UND VERWORFEN: (a) die Kompressibilität INNERHALB der
        Škvor-Zelle — die exakte Lösung (modifizierte Besselfunktionen)
        ändert B bis zur Zell-Squeeze-Zahl 1 um < 1 %; (b) Modenabbruch
        der Membran — membrane_modes = 3 ändert die 2D-Rechnung um
        < 0.01 dB; (c) Sacklöcher als Entlastung — 36 tiefe Sacklöcher
        zwischen 12 Durchgangslöchern senken die Abweichung nur von 10
        auf 7.6 dB. Es zählen deshalb nur die Durchgangslöcher.

        T ist die Membranspannung des 3D-Felds (s. _membrane_tension_3d),
        h der wirksame Frontspalt.

        Rückgabe: dict mit ``rho`` [m], ``tension`` [N/m],
        ``pi_per_omega`` [s] (Tieftonform Π/ω), ``f_rho`` [Hz]
        (Beulresonanz), ``f_hom`` [Hz] (lokale Grenze), ``f_ring`` [Hz]
        (Lochkreis-Darstellung), ``f_limit`` [Hz] (die kleinere) und
        ``cause`` ('local' oder 'ring'); inf, wo nichts die Grenze setzt.
        """
        rho = self._drain_coverage_radius()
        tension = self._membrane_tension_3d()
        f_ring = self._ring_repr_limit()
        inf = float("inf")
        if not np.isfinite(rho) or rho <= 0.0:
            return dict(rho=rho, tension=tension, pi_per_omega=0.0,
                        f_rho=inf, f_hom=inf, f_ring=f_ring,
                        f_limit=f_ring,
                        cause="ring" if np.isfinite(f_ring) else "local")
        h = self.h_gap_front
        j01 = 2.404825557695773
        coef = 12.0 * MU_AIR * rho ** 4 / (h ** 3 * tension * j01 ** 2)
        # Beulresonanz aus derselben Spannung und Flächenmasse
        sigma = self.sigma_mem
        f_rho = j01 / (2.0 * np.pi * rho) * np.sqrt(tension / sigma)

        def _pi(f):
            om = 2.0 * np.pi * f
            a_v = 0.5 * h * np.sqrt(1j * om * RHO0 / MU_AIR)
            K = h / (1j * om * RHO0) * (1.0 - np.tanh(a_v) / a_v)
            k_rel = (h ** 3 / (12.0 * MU_AIR)) / abs(K)
            return om * coef * k_rel / abs(1.0 - (f / f_rho) ** 2)

        # Klammer: die rein viskose Grenze ist eine obere Schranke
        # (|K| ≤ K0, Beulfaktor ≥ 1 unterhalb f_ρ), f_ρ ebenso
        f_hi = min(self._PI_HOM / (2.0 * np.pi * coef),
                   f_rho * (1.0 - 1e-9))
        while _pi(f_hi) < self._PI_HOM and f_hi < f_rho * (1.0 - 1e-6):
            f_hi = min(2.0 * f_hi, f_rho * (1.0 - 1e-9))
        f_lo = 1e-6 * f_hi
        for _ in range(80):                      # Bisektion in log f
            f_mid = np.sqrt(f_lo * f_hi)
            if _pi(f_mid) < self._PI_HOM:
                f_lo = f_mid
            else:
                f_hi = f_mid
        f_hom = float(f_hi)
        return dict(rho=rho, tension=tension, pi_per_omega=coef,
                    f_rho=float(f_rho), f_hom=f_hom, f_ring=f_ring,
                    f_limit=min(f_hom, f_ring),
                    cause="ring" if f_ring < f_hom else "local")

    def _ring_repr_limit(self):
        """Grenze der LOCHKREIS-Darstellung der 1D/2D-Modelle [Hz]
        (Gegenprobe 53).

        Das Radialfeld verschmiert jeden Lochkreis zu einem Gaußband der
        Breite _RING_BAND·a_bp. Der reale Kreis aus vielen Löchern wirkt
        aber als Liniensenke; die Bandbreite ist eine Darstellungswahl
        ohne eindeutigen Wert, und bei Kreisen mit vielen Löchern hängt
        das Ergebnis davon um mehrere dB ab — genau dort weicht es auch
        vom 3D-Löser ab (Formanpassung an das radial stark gegliederte
        Druckfeld; der Filmwiderstand bei erzwungener Form stimmt dagegen
        auf 3 %). Geprüft wird die Selbstkonsistenz: der Membran-Volumen-
        fluss auf Achse (Quelldrücke wie in transfer_function) mit dem
        Standardband gegen das schmalste darstellbare Band (1,5 Zellen,
        Liniensenke); die Grenze ist die erste Frequenz mit einer Spanne
        über _RING_REPR_DB. Immer über das 2D-Feld — auch für 1D, das
        die Lochkreise noch gröber zusammenfasst.

        inf ohne explizite Lochkreise (gleichverteilte Löcher) und im 3D.
        """
        cached = getattr(self, "_f_ring_cache", None)
        if cached is not None:
            return cached
        f_ring = float("inf")
        if (self.squeeze_model != "3d" and self.n_th > 0
                and any(c > 0 and r is not None for c, r in self._th_rings)):
            ff = np.logspace(np.log10(20.0), np.log10(self._F_BAND_TOP), 64)
            om = 2.0 * np.pi * ff
            p_f, p_r = self._source_pressures(om, np.array([0.0]))
            keep = (self.squeeze_model, self._fld_dens_th)
            q = []
            try:
                self.squeeze_model = "2d"
                for dens in (keep[1], self._fld_dens_th_line):
                    self._fld_dens_th = dens
                    T_total, T_rear = self._assemble_network(om)
                    q.append(self._membrane_volume_velocity(
                        om, T_total, T_rear, p_f[:, 0], p_r[:, 0]))
            finally:
                self.squeeze_model, self._fld_dens_th = keep
            d = np.abs(20.0 * np.log10(np.abs(q[0] / q[1])))
            over = np.nonzero(d > self._RING_REPR_DB)[0]
            if over.size:
                k = over[0]
                if k == 0:
                    f_ring = float(ff[0])
                else:
                    x0, x1 = d[k - 1], d[k]
                    f_ring = float(ff[k - 1] * (ff[k] / ff[k - 1]) ** (
                        (self._RING_REPR_DB - x0) / (x1 - x0)))
        self._f_ring_cache = f_ring
        return f_ring

    def _grid_3d_mouths(self):
        """(kleinster Mündungsradius, größter Lochmitten-Radius) der
        Mündungen, die ein 3D-Film sieht — oder None ohne Löcher.
        Membranseitig mündet die weite Senkung (Stufenbohrung) bzw. das
        Loch selbst, dazu die Sacklöcher; im K67-Zwischenspalt der enge
        Kern."""
        radii = []
        if self.n_th > 0:
            radii.append(self.r_bh if self.stepped else self.r_th)
            if self.architecture == "dual_diaphragm" and self.h_center > 0.0:
                radii.append(self.r_th)
        if self.n_bh > 0:
            radii.append(self.r_bh)
        hp = self._hole_positions()
        centers = [p[0] for p in hp["th"] + hp["bh"]]
        if not radii or not centers:
            return None
        return min(radii), max(centers)

    def _grid_3d_size(self):
        """(Nr, Np) des 3D-Gitters: radiale Zellen über der Elektrode,
        azimutale Zellen je Umlauf. ``_n_r_3d``/``_n_phi_3d`` > 0 setzen
        die Werte direkt (Konvergenzprüfungen)."""
        # GROB (Standard): radial das 2D-Feldgitter (60). Azimutal
        # 96 Zellen; mit center_gap > 0 (K67-Modus — und bei single/dual,
        # deren center_gap-Standardwert ebenfalls > 0 ist) 4 Zellen je
        # Durchgangsloch, 96…320: der Lochabstand UND der Verdrehungs-
        # Versatz der Hälften im Zwischenspalt müssen aufgelöst werden.
        Nr = self._fld_N
        if self.h_center > 0.0:
            Np = int(max(96, min(4 * max(self.n_th, 1), 320)))
        else:
            Np = 96
        if self.grid_3d == "fine":
            # FEIN (Gegenprobe 50): Der Gitterfehler sitzt an den
            # Mündungen; mit konturtreuen Randflächen (Gegenprobe 51)
            # fällt er quadratisch mit der Zellweite, eine Mündung kleiner
            # als eine Zelle bleibt aber unkorrigiert. Welche Richtung
            # zählt, hängt von der Bauform ab (K67: radial, 96er-Umfangs-
            # raster: azimutal) — deshalb nach dem kleinsten Mündungs-
            # radius r_m statt mit festem Faktor: Zellweite radial und
            # azimutal (am äußersten Lochmittenkreis) höchstens r_m/2,
            # dazu mindestens 1.5-mal feiner als grob. Obergrenze
            # _GRID_FINE_MAX Zellen je Feld (Speicher, Rechenzeit) —
            # greift sie, warnt _build_3d_geometry.
            Nr_c, Np_c = Nr, Np
            Nr = int(np.ceil(1.5 * Nr_c))
            Np = 2 * int(np.ceil(0.75 * Np_c))
            m = self._grid_3d_mouths()
            if m is not None:
                r_m, R = m
                k = self._GRID_FINE_CELLS
                Nr = max(Nr, int(np.ceil(k * (self.a_bp - self.r_post)
                                         / r_m)))
                Np = max(Np, 2 * int(np.ceil(k * np.pi * R / r_m)))
            if Nr * Np > self._GRID_FINE_MAX:
                s = np.sqrt(self._GRID_FINE_MAX / float(Nr * Np))
                Nr = max(Nr_c, int(Nr * s))
                Np = max(Np_c, 2 * int(0.5 * Np * s))
        n_r = int(getattr(self, "_n_r_3d", 0))
        n_p = int(getattr(self, "_n_phi_3d", 0))
        return (n_r if n_r > 0 else Nr), (n_p if n_p > 0 else Np)

    def _clearance_on_grid(self, r_c, dr, r0):
        """Clearance-Ring auf einem Radialgitter (Zellmitten r_c, Zell-
        breite dr, Gitterbeginn r0): Relief-Karte (Zusatztiefe je Zelle),
        wenn der Ring mindestens eine Zelle breit ist, sonst die Zelle
        des Schlitz-Stubs. Gemeinsam für das 2D-Feld und das 3D-Gitter,
        das mit ``grid_3d='fine'`` eigene Zellbreiten hat."""
        relief = np.zeros(r_c.size)
        stub = None
        if (self.clearance_ring_width > 0.0
                and self.clearance_ring_depth > 0.0
                and self.clearance_ring_diameter > 0.0):
            r_ring = 0.5 * self.clearance_ring_diameter
            if self.clearance_ring_width >= dr:
                relief[np.abs(r_c - r_ring)
                       <= 0.5 * self.clearance_ring_width] = \
                    self.clearance_ring_depth
            else:
                # Zelle, die den Ringradius enthält — gezählt ab r0 (mit
                # Mittenterminierung beginnt das Gitter am Pfostenrand;
                # bis Gegenprobe 50 wurde ab 0 gezählt)
                stub = int(np.clip((r_ring - r0) / dr, 0, r_c.size - 1))
        return relief, stub

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
        DISKRET an ihren (r, phi)-Positionen, die azimutale Zuströmung
        durch den Film, die dadurch teilentkoppelten Sacklöcher und die
        örtliche Verformung der Membran zwischen den Löchern werden
        aufgelöst — das bedämpft insbesondere die interne Helmholtz-
        Resonanz realistisch und ist die Referenz für die Homogenisierungs-
        grenze der 1D/2D-Modelle (Gegenprobe 48).
        MÜNDUNGEN (Gegenprobe 48): Fußabdruck = alle Zellen, deren Mitte
        in der Mündung liegt (exakter Abstand, Fenster nach Lochgröße),
        über G_s·(I − 11ᵀ/k) zur Äquipotentialfläche kurzgeschlossen —
        über dem Lochquerschnitt gibt es keinen Film. KONTURTREU
        (Shortley–Weller, Gegenprobe 51): die Filmflächen am Mündungsrand
        rechnen mit dem wahren Abstand zur Kreiskontur statt zur Mitte
        der Randzelle; damit konvergiert der Löser quadratisch statt
        linear-unregelmäßig (grobes Gitter ~0.1 dB statt ~1 dB neben dem
        Grenzwert). Mündungen kleiner als eine Zelle bleiben unkorrigiert.
        Da die realen Azimutwinkel der Bohrbilder nicht dokumentiert
        sind, gilt eine feste KONVENTION (s. _hole_positions): explizite
        Lochkreise gleichverteilt, Durchgangskreis m um 20°·m, Sack-
        lochkreis m um 15° + 20°·m verdreht; gleichverteilte Anteile
        beider Typen bilden EIN isotropes Raster, auf dem sie sich
        abwechseln. Die Gegenelektrode (zweite Backplate, K67-Rückhälfte)
        ist kreisweise um die halbe Durchgangsteilung verdreht, im K67-
        Modus mit vorgegebenem half_rotation_deg global um diesen Winkel.
        Membran-Elektrostatik: Feder-Erweichung als verteilte negative
        Steifigkeit, an der Grundmode kalibriert (C_A_eff); der Film der
        polarisierten Seite sieht das örtliche Spaltprofil h − w0·φ(r)
        wie im 2D-Modell. Strahlungsimpedanz und Gewebe vor den
        Membranaußenseiten tragen die Sammelknoten (seit Gegenprobe 27).
        RANDUMGEHUNG (Gegenprobe 52): der Membranring außerhalb der
        Backplate (a_bp < a_mem) liegt über dem Ringraum, spürt dessen
        Druck und speist dort ein — mit Randspalt über einen eigenen
        Ringknoten, sonst in die äußerste Filmzelle (bis Gegenprobe 51
        war er hinten unbelastet). Gegen einen unabhängigen axial-
        symmetrischen Löser auf 0.001 dB.
        HOCHTON (Gegenprobe 52): oberhalb der Membranresonanz liegt der
        3D-Löser über dem 2D-Modell — kein Fehler: die Membran weicht dem
        Filmdruck aus (das Einmodenbild kann das nicht), und für Löcher
        auf einem Lochkreis überschätzt das 2D-Feld den Filmwiderstand.
        OFFENER PUNKT: an der gemessenen B&K 4134 (Gegenprobe 38) liegt
        der 3D-Löser bei 13…20 kHz 2.2…3.5 dB über der Messung, das 2D-
        Modell höchstens 0.6 dB — die reale Kapsel dämpft also stärker
        als der Reynolds-Film. Die Aktuatormessung erklärt es nicht: ihre
        Zusatzlast hebt die 4134 um höchstens 0.6 dB an (Gegenprobe 52 d).
        Das Gitter ist es auch nicht (grid_3d='fine' ändert höchstens
        0.2 dB).
        AUSGANG UND SPANNUNG (Gegenprobe 54): die Spannung entsteht aus dem
        Auslenkungsfeld über dasselbe Elektrodenintegral wie Θ der Kette
        (s. _output_weight_3d; 'weight'), und die Membranspannung ist bei
        vorgegebener Vorspannung die physikalische (s. _membrane_tension_3d)
        — vorher lag 3D an der B&K 4134 statisch 1.38 dB unter 2D.
        GITTER (Gegenprobe 50): _grid_3d_size — grob (Standard) oder fein
        (≥ 2 Zellen je kleinstem Mündungsradius); der Membranring außerhalb
        der Elektrode hat eine eigene Zellweite, damit die Einspannung auf
        jedem Gitter exakt bei a_mem liegt.
        """
        Nr, Np_ = self._grid_3d_size()
        # MITTENTERMINIERUNG: beide Gitter beginnen am Pfostenrand r0.
        # Film und Membran teilen sich dr und den Startradius, weil die
        # ersten Nr Membranzellen mit den Filmzellen gekoppelt werden.
        # r0 = 0 liefert bitgleich den bisherigen Stand.
        r0 = self.r_post
        dr = (self.a_bp - r0) / Nr
        q0 = r0 / dr                      # Pfostenrand in Zellbreiten
        r_f = r0 + (np.arange(Nr) + 0.5) * dr
        # MEMBRANRING außerhalb der Elektrode (a_bp … a_mem) mit EIGENER
        # Zellbreite dr_o ≈ dr, damit die Einspannung GENAU bei a_mem
        # liegt. Bis Gegenprobe 50 lief das Elektrodengitter einfach
        # weiter, und die Einspannung rastete auf das nächste Vielfache
        # von dr ein (mit a_bp = a_mem sogar eine ganze Zelle zu weit):
        # die Membran war je nach Gitter zu groß oder zu klein, ihre
        # Nachgiebigkeit (∝ a⁴) sprang mit der Auflösung — bei der K67
        # zwischen 60 und 90 Radialzellen um 4 %. Ohne Überstand (a_bp
        # >= a_mem) endet die Membran am Elektrodenrand.
        a_out = max(self.a_mem - self.a_bp, 0.0)
        n_out = (max(1, int(round(a_out / dr)))
                 if a_out > 1e-9 * self.a_mem else 0)
        dr_o = a_out / n_out if n_out else dr
        Nr_m = Nr + n_out
        drm = np.concatenate([np.full(Nr, dr), np.full(n_out, dr_o)])
        r_m = np.concatenate([r_f, self.a_bp + (np.arange(n_out) + 0.5)
                              * dr_o])
        dphi = 2.0 * np.pi / Np_
        A_f = r_f * dr * dphi                       # Zellfläche je Ring
        A_m = r_m * drm * dphi
        NF = Nr * Np_
        NM = Nr_m * Np_
        # Clearance-Ring auf DIESEM Gitter (fein: eigene Zellbreite)
        relief, stub_cell = self._clearance_on_grid(r_f, dr, r0)
        # Auflösung der kleinsten Mündung in Zellen je Radius (die gröbere
        # Richtung zählt, azimutal am äußersten Lochmittenkreis)
        mouth = self._grid_3d_mouths()
        cells_rm = (None if mouth is None
                    else min(mouth[0] / dr, mouth[0] / (mouth[1] * dphi)))
        fine_capped = (self.grid_3d == "fine" and cells_rm is not None
                       and cells_rm < self._GRID_FINE_CELLS * (1.0 - 1e-9))
        if fine_capped:
            warnings.warn(
                f"grid_3d='fine': Obergrenze von {self._GRID_FINE_MAX} "
                f"Zellen je Feld erreicht — die kleinste Mündung "
                f"(r = {mouth[0] * 1e3:.2f} mm) ist nur mit "
                f"{cells_rm:.1f} statt {self._GRID_FINE_CELLS:.0f} Zellen "
                f"je Radius aufgelöst.", UserWarning, stacklevel=3)

        # Membrankonstanten: Flächendichte, Spannung (s. _membrane_tension_
        # 3d: aus f_res, wenn vorgegeben, sonst die physikalische). Mit
        # Mittenterminierung sind Eigenwert und Modenform die der
        # RINGmembran (s. _ring_modes).
        sigma = self.sigma_mem
        T_mem = self._membrane_tension_3d()
        # Feder-Erweichung ÖRTLICH aus dem exakten Arbeitspunkt (λ·∂p/∂w
        # über dem Spaltprofil, Gegenprobe 49). Bis dahin eine über die
        # Elektrode gleichförmige negative Steifigkeit, an der Grundmode
        # auf C_A_eff kalibriert — das Feld sah damit weder, dass die
        # Erweichung in der Mitte (engster Spalt) am größten ist, noch
        # die Porosität.
        kappa = self._st_softening_density(r_f)

        # Loch-Fußabdrücke: Zellen, deren Zentrum in der Mündung liegt
        # (EXAKTER kartesischer Abstand Zellmitte–Lochmitte). Bis
        # Gegenprobe 47 war das Suchfenster fest ±4 Zellen breit: sobald
        # die Mündung mehr als vier Zellen überdeckte — bei feinem Gitter
        # oder azimutal auf inneren Lochkreisen, wo die Zellen schmal
        # sind —, wurde sie abgeschnitten. Die Löcher waren dann kleiner
        # als eingegeben (12 × Ø1.4 mm auf 3 mm Radius: wirksam Ø0.8 mm),
        # und das Ergebnis wanderte mit der Gitterfeinheit statt zu
        # konvergieren (Gegenprobe 48).
        def _foot(radius, n, off_deg, r_hole):
            out = []
            i_lo = max(0, int(np.floor((radius - r_hole - r0) / dr)) - 1)
            i_hi = min(Nr - 1, int(np.floor((radius + r_hole - r0) / dr)) + 1)
            for k in range(max(n, 0)):
                ph0 = np.deg2rad(off_deg) + 2.0 * np.pi * k / max(n, 1)
                j0 = int(np.floor(ph0 / dphi))
                # die Zelle, die die Lochmitte enthält, gehört immer dazu
                # (Mündung kleiner als eine Zelle: dann ist sie es allein)
                i0 = int(np.clip((radius - r0) / dr, 0, Nr - 1))
                cells = [i0 * Np_ + j0 % Np_]
                for i in range(i_lo, i_hi + 1):
                    ri = r_f[i]
                    dj = min(Np_ // 2,
                             int(np.ceil(r_hole / (ri * dphi))) + 1)
                    jj = np.arange(j0 - dj, j0 + dj + 1)
                    d2 = (ri**2 + radius**2 - 2.0 * ri * radius
                          * np.cos((jj + 0.5) * dphi - ph0))
                    sel = jj[d2 <= r_hole**2 * (1.0 + 1e-12)]
                    cells.extend((i * Np_ + sel % Np_).tolist())
                out.append(np.array(sorted(set(cells)), dtype=int))
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
        hp = self._hole_positions()

        mouth_geo = {}                   # id(Fußabdruck) -> (R, φ, r_m)

        def _feet(plist, r_hole, rear=False, turn=None):
            # Fußabdrücke an den Lochmitten. rear: Gegenelektrode, um den
            # kreisweisen Rückversatz (halbe Durchgangsteilung) verdreht —
            # oder, wenn turn gesetzt ist, global um turn Grad (K67 mit
            # vorgegebenem half_rotation_deg). Die Kreiskontur jeder
            # Mündung wird für die konturtreuen Randflächen gemerkt.
            out = []
            for rr, deg, ring_turn in plist:
                if rear:
                    deg = deg + (ring_turn if turn is None else turn)
                cells = _foot(rr, 1, deg, r_hole)[0]
                mouth_geo[id(cells)] = (rr, np.deg2rad(deg), r_hole)
                out.append(cells)
            return out

        k67_turn = None if self._half_rot_auto else rot

        th_cells = None
        th_f = th_r = th_cf = th_cr = None
        if arch == "dual_diaphragm" and n_films == 2:
            # Einteilige Platte: Durchgangsloch verbindet beide Filme an
            # DENSELBEN Zellen; Rück-Sacklöcher um eine halbe Teilung
            # versetzt (Konvention).
            th_cells = _feet(hp["th"], self.r_th)
            bhr_cells = _feet(hp["bh"], self.r_bh, rear=True)
        elif arch == "dual_diaphragm":
            # K67: je Hälfte ein eigenes Lochbild; die Rückhälfte ist
            # verdreht (automatisch kreisweise um die halbe Durchgangs-
            # teilung, sonst global um half_rotation_deg). Membranseitig
            # mündet (bei Stufenbohrung) die WEITE Senkung, zwischen-
            # spaltseitig der enge Kern.
            th_f = _feet(hp["th"], r_mouth)
            th_cf = _feet(hp["th"], self.r_th)
            th_r = _feet(hp["th"], r_mouth, rear=True, turn=k67_turn)
            th_cr = _feet(hp["th"], self.r_th, rear=True, turn=k67_turn)
            bhr_cells = _feet(hp["bh"], self.r_bh, rear=True, turn=k67_turn)
        elif arch == "dual":
            # zwei getrennte Platten: eigene Lochbilder, hinten um eine
            # halbe Teilung versetzt (keine direkte Kopplung der Filme,
            # der Versatz ist nur Konvention)
            th_f = _feet(hp["th"], r_mouth)
            th_r = _feet(hp["th"], r_mouth, rear=True)
            bhr_cells = _feet(hp["bh"], self.r_bh, rear=True)
        else:                                        # single
            th_f = _feet(hp["th"], r_mouth)
            bhr_cells = []
        bhf_cells = _feet(hp["bh"], self.r_bh)

        # Statische COO-Anteile: Membran-Spannungsoperator (eingespannter
        # Rand) + Feder-Erweichung (polarisierte Membran, Elektroden-
        # bereich) + Druckkopplungen. Alle übrigen Einträge sind
        # frequenzabhängig.
        rows = []
        cols = []
        vals = []

        def _lap(base_off, Tfac):
            # Radialer Flächenleitwert einer Fläche bei r: T·r·dphi/dr.
            # Ohne Mittenterminierung ist r0 = 0, die innerste Fläche hat
            # den Radius 0 und trägt nichts — genau die Achsenbedingung.
            # MIT Pfosten ist dieselbe Fläche eine EINGESPANNTE Wand: sie
            # liegt eine halbe Zelle vor der ersten Zellmitte, also mit
            # dem doppelten Leitwert auf der Diagonalen. Beide Fälle
            # fallen aus derselben Formel. Außerhalb der Elektrode haben
            # die Ringe die Breite dr_o: Fläche bei r_i + drm_i/2, Abstand
            # der Zellmitten r_{i+1} − r_i, Einspannung eine halbe
            # Randzelle hinter der letzten Mitte (= a_mem).
            for i in range(Nr_m):
                if i < Nr_m - 1:
                    G = (Tfac * (r_m[i] + 0.5 * drm[i]) * dphi
                         / (r_m[i + 1] - r_m[i]))
                    for j in range(Np_):
                        k1_ = base_off + i * Np_ + j
                        k2_ = base_off + (i + 1) * Np_ + j
                        rows.extend((k1_, k2_, k1_, k2_))
                        cols.extend((k2_, k1_, k1_, k2_))
                        vals.extend((-G, -G, G, G))
                else:
                    G = (Tfac * (r_m[i] + 0.5 * drm[i]) * dphi
                         / (0.5 * drm[i]))
                    for j in range(Np_):                # geklemmter Rand
                        k1_ = base_off + i * Np_ + j
                        rows.append(k1_)
                        cols.append(k1_)
                        vals.append(G)
                if i == 0 and q0 > 0.0:                 # Pfostenrand
                    G = Tfac * q0 * dphi * 2.0
                    for j in range(Np_):
                        k1_ = base_off + j
                        rows.append(k1_)
                        cols.append(k1_)
                        vals.append(G)
                Gp = Tfac * drm[i] / (r_m[i] * dphi)
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
                vals.append(-kappa[i] * A_m[i])
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

        # ÄQUIPOTENTIALE MÜNDUNGEN (Gegenprobe 48). Über dem Lochquerschnitt
        # gibt es keinen Spaltfilm, sondern das offene Loch: der Druck ist
        # dort (bis auf den vernachlässigbaren Eigenwiderstand der Mündung)
        # EINHEITLICH. Bis Gegenprobe 47 blieben die Fußabdruck-Zellen
        # gewöhnliche Filmzellen mit gleichverteiltem Zufluss — die Luft
        # musste dann auch INNERHALB der Mündung lateral durch einen Film
        # strömen, der dort gar nicht existiert. Das ist der Unterschied
        # zwischen einer Scheibe mit gleichverteilter Quelle und einer
        # Äquipotentialscheibe, in Škvor-Einheiten +1/8 auf B (bei q = 0.04
        # rund +28 % Zellwiderstand). Hier wird jede Mündung über
        # G_s·(I − 11ᵀ/k) kurzgeschlossen: das zieht alle k Zellen auf
        # ihren gemeinsamen Mittelwert, ohne Nettofluss einzuspeisen.
        # G_s liegt 10⁴-fach über dem größten Flächenleitwert des Gitters
        # (statisch — der dynamische Filmleitwert ist betragsmäßig
        # kleiner), der Restwiderstand 2/G_s ist also vernachlässigbar.
        #
        # KONTURTREUE MÜNDUNGEN (Shortley–Weller, Gegenprobe 51). Eine
        # Zelle gehört zur Mündung, wenn ihre Mitte darin liegt — die
        # Mündung ist damit eine Treppe, deren wirksamer Rand um einen
        # Bruchteil der Zellweite neben der Kreiskontur liegt. Der Fehler
        # fiel nur linear und unregelmäßig mit dem Gitter (Gegenprobe 50).
        # Korrektur: auf jeder Filmfläche zwischen einer Mündungszelle
        # und einer Filmzelle strömt die Luft nicht über den vollen
        # Mittenabstand Δ durch Film, sondern nur über das Stück ℓ
        # außerhalb der Kontur (Schnittpunkt radial auf dem Strahl,
        # azimutal auf dem Bogen); die Mündungszelle liegt auf dem
        # Mündungsdruck. Der Flächenleitwert wird mit Δ/ℓ skaliert
        # (zwei Mündungen beiderseits einer Fläche: ℓ = Steg dazwischen).
        # Ohne Kontur in der Fläche bleibt der Faktor exakt 1.
        film_mouths = [[] for _ in range(n_films)]

        def _reg(cells_list, film):
            for cells in (cells_list or []):
                film_mouths[film].append((cells,) + mouth_geo[id(cells)])

        if th_cells is not None:
            _reg(th_cells, 0)
            _reg(th_cells, 1)
        else:
            _reg(th_f, 0)
            if arch != "single":
                _reg(th_r, 1)
            if n_films == 3:
                _reg(th_cf, 2)
                _reg(th_cr, 2)
        _reg(bhf_cells, 0)
        if arch != "single":
            _reg(bhr_cells, 1)

        def _sw_factors(mouths):
            # b_*: Mündungsanteil des Mittenabstands, von der Zelle der
            # jeweiligen Seite aus gemessen (radial: Fläche i|i+1,
            # azimutal: Fläche j|j+1)
            b_rlo = np.zeros((Nr - 1, Np_))
            b_rhi = np.zeros((Nr - 1, Np_))
            b_alo = np.zeros((Nr, Np_))
            b_ahi = np.zeros((Nr, Np_))
            owner = np.full(NF, -1)
            for m, mo in enumerate(mouths):
                owner[mo[0]] = m
            for m, (cells, R, ph, rm) in enumerate(mouths):
                i = cells // Np_
                j = cells % Np_
                ri = r_f[i]
                a = (j + 0.5) * dphi - ph
                ins = ri**2 + R**2 - 2.0 * ri * R * np.cos(a) < rm**2
                i, j, ri, a = i[ins], j[ins], ri[ins], a[ins]
                if i.size == 0:
                    continue             # Mündung kleiner als eine Zelle
                sq = np.sqrt(np.maximum(rm**2 - (R * np.sin(a))**2, 0.0))
                # radial nach außen / innen
                o = i + 1 < Nr
                o &= owner[np.minimum(i + 1, Nr - 1) * Np_ + j] != m
                np.maximum.at(b_rlo, (i[o], j[o]), np.clip(
                    R * np.cos(a[o]) + sq[o] - ri[o], 0.0, dr))
                o = i >= 1
                o &= owner[np.maximum(i - 1, 0) * Np_ + j] != m
                np.maximum.at(b_rhi, (i[o] - 1, j[o]), np.clip(
                    ri[o] - (R * np.cos(a[o]) - sq[o]), 0.0, dr))
                # azimutal: halber Öffnungswinkel der Kontur auf r_i
                den = 2.0 * ri * R
                ca = np.where(den > 0.0, (ri**2 + R**2 - rm**2)
                              / np.where(den > 0.0, den, 1.0), -1.0)
                half = np.arccos(np.clip(ca, -1.0, 1.0))
                jn = (j + 1) % Np_
                o = owner[i * Np_ + jn] != m
                t = np.mod(half - a, 2.0 * np.pi)
                np.maximum.at(b_alo, (i[o], j[o]),
                              np.clip(t[o], 0.0, dphi) * ri[o])
                jp = (j - 1) % Np_
                o = owner[i * Np_ + jp] != m
                t = np.mod(a + half, 2.0 * np.pi)
                np.maximum.at(b_ahi, (i[o], jp[o]),
                              np.clip(t[o], 0.0, dphi) * ri[o])
            cap = self._SW_CAP
            f_r = dr / np.maximum(dr - b_rlo - b_rhi, dr / cap)
            da = (r_f * dphi)[:, None]
            f_a = da / np.maximum(da - b_alo - b_ahi, da / cap)
            return f_r, f_a

        if self._SHORTLEY_WELLER:
            sw = [_sw_factors(film_mouths[fi]) for fi in range(n_films)]
        else:
            sw = [(np.ones((Nr - 1, Np_)), np.ones((Nr, Np_)))
                  for _ in range(n_films)]
        sw_max = max(max(float(fr_.max(initial=1.0)),
                         float(fa_.max(initial=1.0))) for fr_, fa_ in sw)

        h_ref = max(self.h_gap, self.h_gap_front) + float(np.max(relief))
        if n_films == 3:
            h_ref = max(h_ref, self.h_center)
        # (mit dem größten Konturfaktor: der Kurzschluss bleibt 10⁴-fach
        # über jedem Flächenleitwert)
        g_geo = max((q0 + Nr) * dphi, 1.0 / ((q0 + 0.5) * dphi)) * sw_max
        G_s = self._EQUI_SHORT * g_geo * h_ref ** 3 / (12.0 * MU_AIR)
        eq_r, eq_c, eq_v = [], [], []

        def _short(cells_list, film):
            for cells in (cells_list or []):
                k = cells.size
                if k < 2:
                    continue
                aa = film * NF + cells
                rr_ = np.repeat(aa, k)
                cc_ = np.tile(aa, k)
                vv_ = np.full(k * k, -G_s / k, dtype=complex)
                vv_[rr_ == cc_] += G_s
                eq_r.append(rr_)
                eq_c.append(cc_)
                eq_v.append(vv_)

        if th_cells is not None:                 # einteilig: beide Filme
            _short(th_cells, 0)
            _short(th_cells, 1)
        else:
            _short(th_f, 0)
            if arch != "single":
                _short(th_r, 1)
            if n_films == 3:
                _short(th_cf, 2)
                _short(th_cr, 2)
        _short(bhf_cells, 0)
        if arch != "single":
            _short(bhr_cells, 1)

        self._g3d = dict(
            Np=Np_, Nr=Nr, Nr_m=Nr_m, dr=dr, drm=drm, dphi=dphi,
            r_f=r_f, r_m=r_m, A_f=A_f, A_m=A_m, NF=NF, NM=NM, q0=q0,
            arch=arch, n_films=n_films, n_mem=n_mem, n_nodes=n_nodes,
            sigma=sigma, T_mem=T_mem, kappa=kappa,
            th_cells=th_cells, bhf_cells=bhf_cells, bhr_cells=bhr_cells,
            th_f=th_f, th_cf=th_cf, th_r=th_r, th_cr=th_cr, G_s=G_s,
            relief=relief, stub_cell=stub_cell, cells_rm=cells_rm,
            fine_capped=fine_capped, sw=sw,
            static=(np.concatenate([np.array(rows, dtype=int)] + eq_r),
                    np.concatenate([np.array(cols, dtype=int)] + eq_c),
                    np.concatenate([np.array(vals, dtype=complex)]
                                   + eq_v)),
        )

    def _solve_3d_state(self):
        """Alles, was die 3D-Lösung außer Geometrie und Frequenz bestimmt
        und sich nach dem Bau einer Kapsel noch ändern kann (Gegenprobe
        56): die Schalter und Methoden der Klasse samt Basisklassen
        (Gegenproben tauschen z. B. die Strahlungsimpedanz aus oder
        schalten Vergleichsmodelle), auf der Instanz ersetzte Methoden und
        die Stoffwerte der Luft. Die übrigen Parameter der Kapsel liegen
        mit dem Bau fest — nachträglich geändert würden auch ihre
        abgeleiteten Größen (_derive_parameters) nicht nachgeführt."""
        einfach = (bool, int, float, complex, str, type(None),
                   types.FunctionType, staticmethod, classmethod, property)
        teile = [(klass.__qualname__, k, v)
                 for klass in type(self).__mro__ if klass is not object
                 for k, v in vars(klass).items()
                 if not k.startswith("__") and isinstance(v, einfach)]
        teile += [("Instanz", k, v) for k, v in vars(self).items()
                  if callable(v)]
        teile.append(("Luft", RHO0, C_AIR, MU_AIR, P_ATM, GAMMA, PRANDTL))
        return tuple(teile)

    def _solve_3d_cache(self):
        """Lösungsspeicher der 3D-Feldlösung {ω: Ausgaben} (Gegenprobe 56).

        Er hängt an der Geometrie (_g3d): ein neu gebautes Gitter bringt
        einen leeren Speicher mit. Ändert sich der Zustand aus
        _solve_3d_state, wird er verworfen. Gespeichert sind je Frequenz
        nur die Ausgaben für beide Gewichte ('output', 'volume') und die
        Reziprozitäts-Diagnose, nicht die Feldvektoren (die wären bei 150
        Frequenzen über 100 MB)."""
        if not self._REUSE_3D:
            return None
        zustand = self._solve_3d_state()
        eintrag = self._g3d.get("_loesungen")
        if eintrag is None or eintrag[0] != zustand:
            eintrag = (zustand, {})
            self._g3d["_loesungen"] = eintrag
        return eintrag[1]

    def _solve_3d(self, omega, want_rear=False, weight="output"):
        """3D-Sandwich-Lösung: Ausgangs-Volumenverschiebung je Einheits-
        Außendruck, U_front = X_f·p_front + X_r·p_rear.

        ``weight``: 'output' (Standard) gibt die Θ-konsistente Volumen-
        verschiebung aus, die mit dem Wandlerkoeffizienten der Kette die
        Spannung ergibt (s. _output_weight_3d, Gegenprobe 54); 'volume'
        die reine Volumenverschiebung über der Elektrode (für Validierungen
        gegen geschlossene Formen und Reziprozität).

        Rückgabe: (X_f, X_r) je Frequenz [m³/Pa]; mit ``want_rear``
        zusätzlich die Rückmembran-Antworten (B_f, B_r) für
        Reziprozitätsprüfungen. Je Frequenz wird das dünn besetzte
        Gesamtsystem (einteilig: 2 Filme + 2 Membranfelder, ~24k
        Unbekannte; K67-Modus mit Zwischenspalt: 3 Filme, ~30k)
        einmal LU-faktorisiert — das 3D-Modell ist damit DEUTLICH
        langsamer als 1D/2D (Sekundenbereich pro Frequenzpunkt);
        für Frequenzgänge empfiehlt sich n_points <= 150.

        WIEDERVERWENDUNG (Gegenprobe 56): eine schon gelöste Frequenz wird
        nicht erneut faktorisiert, sondern aus dem Lösungsspeicher
        (_solve_3d_cache) bedient — bitgleich, für beide Gewichte und mit
        denselben Rückmembran-Antworten. ``_lu_3d_count`` zählt die
        tatsächlichen Faktorisierungen.
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
        # RINGRAUM (Gegenprobe 52): mit Randspalt je Filmseite ein eigener
        # Knoten p_ring zwischen Filmrand und Schlitzleitung (nur single/
        # dual; die Doppelmembran hat keinen Randspalt)
        ring_sides = []
        if self.ring_vent_w > 0.0 and arch != "dual_diaphragm":
            ring_sides = ([(0, off_n + 1)] if arch == "single"
                          else [(0, off_n + 0), (1, off_n + 1)])
        off_ring = off_n + n_nodes
        ring_of = {s_: off_ring + k_ for k_, (s_, _) in enumerate(ring_sides)}
        N_tot = off_ring + len(ring_sides)
        # Ausgangs- (Elektrodenbereich) und Anregungs-Gewichte (Vollfläche).
        # Vorzeichen: die interne w-Konvention (positiv = von der Elektrode
        # weg) ist der 1D/2D-Flussrichtung (q_mem front -> rück) entgegen-
        # gesetzt; das Minus richtet die 3D-Ausgänge an der Kettenkonvention
        # aus, sodass H über alle Modelle phasengleich ist (Beträge und das
        # Verhältnis D_r = −X_f/X_r sind davon unberührt).
        if weight not in ("output", "volume"):
            raise ValueError("weight muss 'output' oder 'volume' sein.")
        # beide Gewichte je Lösung, damit der Speicher jede spätere
        # Anfrage bedienen kann
        w_vol = -np.repeat(A_f, Np_)
        w_outs = {"volume": w_vol,
                  "output": (-np.repeat(A_f * self._output_weight_3d(r_f),
                                        Np_)
                             if self._OUTPUT_EXACT_3D else w_vol)}
        rhs_w = np.repeat(A_m, Np_)
        speicher = self._solve_3d_cache()

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
        # ÖRTLICHES Spaltprofil der polarisierten Seite, wie im 2D-Feld
        # (s. _gap_field_2port): h(r) = h − w0·φ(r) statt des Flächen-
        # mittels h_gap_front. Der Filmleitwert geht mit h³ — das Mittel
        # des Spalts ist nicht das Mittel des Leitwerts, und die Strömung
        # zu einem Lochkreis muss gerade durch die engste Stelle (Mitte).
        # Bis Gegenprobe 47 rechnete der 3D-Löser mit dem Flächenmittel
        # und war damit nahe am Pull-in systematisch zu wenig bedämpft
        # (Gegenprobe 48). Bei 'dual' ist w0 = 0 -> h_gap wie bisher.
        h_pol = self._polarized_gap_profile(r_f)
        if arch == "dual_diaphragm":
            sides = [(0, h_pol, off_w, +1.0),
                     (1, self.h_gap, off_w + NM, -1.0)]
            if n_films == 3:
                sides.append((2, self.h_center, None, 0.0))
        elif arch == "dual":
            # w positiv = nach vorn: komprimiert den VORDEREN Film (−jw
            # analog zur Rückmembran der K67-Bauform), dehnt den hinteren
            sides = [(0, h_pol, off_w, -1.0),
                     (1, h_pol, off_w, +1.0)]
        else:                                        # single
            sides = [(0, h_pol, off_w, +1.0)]

        for fidx, om in enumerate(omega):
            treffer = (speicher.get(float(om)) if speicher is not None
                       else None)
            if treffer is not None:
                Xf[fidx], Xr[fidx], Bf[fidx], Br[fidx] = treffer[weight]
                if treffer["recip"] is not None:
                    self._recip_3d = treffer["recip"]
                continue
            rows = [srows]
            cols = [scols]
            vals = [svals]
            K_edge_side = {}
            # Filmringe (Membranseiten ggf. mit Clearance-Relief; der
            # Zwischenspalt ist eben und hat weder Relief noch Membran)
            for side, h0, mem_off, sgn in sides:
                off = side * NF
                mem_side = mem_off is not None
                h_ring = h0 + (g["relief"] if mem_side else 0.0)
                if mem_side:
                    h_ring = np.maximum(h_ring, 0.05 * self.h_gap)
                h_ring = np.broadcast_to(h_ring, (Nr,))
                K = np.empty(Nr, dtype=complex)
                cg = np.empty(Nr, dtype=complex)
                for hh in np.unique(h_ring):
                    msk = h_ring == hh
                    Kh, ch = _film_props(hh, om)
                    K[msk] = Kh
                    cg[msk] = ch
                K_edge_side[side] = K[Nr - 1]
                # konturtreue Randflächen der Mündungen (Faktor Δ/ℓ,
                # s. _build_3d_geometry; sonst exakt 1)
                sw_r, sw_a = g["sw"][side]
                # radiale Faces
                Kmid = 0.5 * (K[:-1] + K[1:])
                Gr = ((g["q0"] + np.arange(1, Nr)) * dphi) * Kmid
                k1_ = off + idx_all[:(Nr - 1) * Np_]
                k2_ = k1_ + Np_
                Gv = np.repeat(Gr, Np_) * sw_r.ravel()
                rows += [k1_, k2_, k1_, k2_]
                cols += [k2_, k1_, k1_, k2_]
                vals += [-Gv, -Gv, Gv, Gv]
                # azimutale Faces
                Ga = np.repeat(dr / (r_f * dphi) * K, Np_) * sw_a.ravel()
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
                # MEMBRANRING außerhalb der Platte (a_bp < r < a_mem,
                # Gegenprobe 52): er liegt über dem tiefen Ringraum
                # zwischen Plattenrand und Einspannung, dessen Druck der
                # Randdruck ist — er spürt diesen Druck und speist seinen
                # Volumenfluss dort ein (wie die RANDUMGEHUNG des 2D-Felds,
                # Gegenprobe 37). Mit Randspalt ist das der Ringknoten,
                # sonst die äußerste Filmzelle desselben Winkels (der
                # Ringraum ist dann eine Sackgasse, die über den Filmrand
                # entleert). Bis hier war die Ringrückseite UNBELASTET —
                # als läge Vakuum hinter ihr.
                if Nr_m > Nr and self._ANNULUS_COUPLED:
                    I_, J_ = np.meshgrid(np.arange(Nr, Nr_m), np.arange(Np_),
                                         indexing="ij")
                    I_, J_ = I_.ravel(), J_.ravel()
                    tgt = (np.full(I_.size, ring_of[side]) if side in ring_of
                           else off + (Nr - 1) * Np_ + J_)
                    mw = mem_off + I_ * Np_ + J_
                    rows += [tgt, mw]
                    cols += [mw, tgt]
                    vals += [sgn * 1j * om * A_m[I_].astype(complex),
                             -sgn * A_m[I_].astype(complex)]
                # Clearance-Ring als Schlitz-Stub (schmaler Ring)
                if g["stub_cell"] is not None:
                    r_cst = 0.5 * self.clearance_ring_diameter
                    C_st = (2.0 * np.pi * r_cst * self.clearance_ring_width
                            * self.clearance_ring_depth / (GAMMA * P_ATM))
                    R_st = (12.0 * MU_AIR * self.clearance_ring_depth
                            / (2.0 * np.pi * r_cst
                               * self.clearance_ring_width ** 3) / 3.0)
                    y_st = 1.0 / (R_st + 1.0 / (1j * om * C_st)) / Np_
                    cells = g["stub_cell"] * Np_ + np.arange(Np_)
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
                if ring_sides:
                    T_l3 = self._slit_line_abcd(
                        om_a, self.ring_vent_w, 2.0 * np.pi * self.a_bp,
                        self.ring_vent_L)
                    A_l3 = complex(T_l3[0, 0][0]); B_l3 = complex(T_l3[0, 1][0])
                    D_l3 = complex(T_l3[1, 1][0])
                    edge_cells = (Nr - 1) * Np_ + np.arange(Np_)
                    for side3, node_off3 in ring_sides:
                        # Filmrand -> Ringknoten: je Randzelle die halbe
                        # Randzelle bei r = a_bp = (q0 + Nr)·dr (Summe wie
                        # _fld_gedge_geom im 2D-Feld); der Ringraum ist
                        # druckgleich. Ringknoten -> Sammelknoten: die
                        # Schlitzleitung (reziprokes Zweitor, det T = 1).
                        kr = np.array([ring_of[side3]])
                        ke = side3 * NF + edge_cells
                        Ge = np.full(Np_, 2.0 * (g["q0"] + Nr) * dphi
                                     * K_edge_side[side3], dtype=complex)
                        krr = np.full(Np_, kr[0])
                        rows += [ke, krr, ke, krr]
                        cols += [krr, ke, ke, krr]
                        vals += [-Ge, -Ge, Ge, Ge]
                        rows += [kr, kr, np.array([node_off3]),
                                 np.array([node_off3])]
                        cols += [kr, np.array([node_off3]), kr,
                                 np.array([node_off3])]
                        vals += [np.array([D_l3 / B_l3]),
                                 np.array([-1.0 / B_l3]),
                                 np.array([-1.0 / B_l3]),
                                 np.array([A_l3 / B_l3])]
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
            self._lu_3d_count = getattr(self, "_lu_3d_count", 0) + 1
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
            wr = (x[off_w + NM:off_w + NM + NF, :]
                  if arch == "dual_diaphragm" else None)
            neu = {"recip": None}
            for w_name, w_v in w_outs.items():
                xf_ = np.sum(w_v[:, None] * wf, axis=0)
                if wr is not None:
                    bf_ = np.sum(w_v[:, None] * wr, axis=0)
                    neu[w_name] = (xf_[0], xf_[1], bf_[0], bf_[1])
                else:
                    neu[w_name] = (xf_[0], xf_[1], xf_[0], xf_[1])
            if arch != "dual_diaphragm":
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
                neu["recip"] = self._recip_3d
            Xf[fidx], Xr[fidx], Bf[fidx], Br[fidx] = neu[weight]
            if speicher is not None:
                speicher[float(om)] = neu
        if want_rear:
            return Xf, Xr, Bf, Br
        return Xf, Xr

    def _source_pressures(self, omega, theta):
        """Quelldrücke p_front/p_rear, optional MODENABHÄNGIG.

        Ohne ``modal_source`` (Voreinstellung 0) exakt
        :meth:`_source_pressures_fundamental` — bit-für-bit das bisherige
        Verhalten. Mit ``modal_source = 1`` wird der Antrieb jeder
        Membranmode einzeln projiziert und über die Modenadmittanzen zu
        einer Ersatzquelle zusammengezogen (s. :meth:`_modal_source_scale`).

        Der Faktor wirkt auf BEIDE Membranen der Doppelmembran-Bauform:
        sie sind gleich groß, tragen dieselben Modenformen, und der
        Unterschied ihrer Oberflächenfelder steckt bereits im Transfer
        G_ax bzw. im rückseitigen BEM-Mittel. Beim Einzelmembran-Pfad
        bleibt p_rear unberührt — dort ist es ein EINLASSPORT ohne
        Membranmode.
        """
        p_f, p_r = self._source_pressures_fundamental(omega, theta)
        if not self.modal_source:
            return p_f, p_r
        scale = self._modal_source_scale(omega, theta, p_f)
        if self.architecture == "dual_diaphragm":
            return p_f * scale, p_r * scale
        return p_f * scale, p_r

    def _source_pressures_fundamental(self, omega, theta):
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
            if self.axial_body_model == "bem":
                # EIN-MEMBRAN-KAPSEL: der Frontfaktor kommt aus derselben
                # BEM-Lösung, jetzt aber auf der REALEN flachen
                # Stirnfläche statt auf einer Kugelkalotte. Bei DICHTER
                # Rückseite liest _membrane_volume_velocity p_rear gar
                # nicht; dort wird derselbe Faktor zurückgegeben, damit
                # keine stille Mischung zweier Körpermodelle entsteht.
                # Bei OFFENER Rückseite (Gradientenempfänger) kommt der
                # rückwärtige Druck aus demselben Lösungsgang — der
                # Bohrungskranz liegt als Ring auf der Kontur
                # (s. _bem_rear_inlet_weights, Gegenprobe 44).
                F_bem, G_bem = self._bem_axial_fields(omega, theta)
                if self.rear_open:
                    return F_bem, F_bem * G_bem
                return F_bem, F_bem
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
          * Membran treibt mit der Modenform v(r) = φ(r)·U/∫φ dA, wobei
            ∫φ dA über die GANZE Membran läuft — der außerhalb der Platte
            erzeugte Anteil umgeht den Film über den Rand (s. u.,
            RANDUMGEHUNG)

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
        # (Form der statischen Auslenkung: seit Gegenprobe 49 die EXAKTE
        # Lösung, auf ihr Maximum normiert; sag_w0 skaliert sie)
        shape = self._fld_sag_shape
        h_cell = np.maximum(h - sag_w0 * shape + self._clr_relief, 0.05 * h)
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
                return float(np.sum(shape * w)) / s if s > 0.0 else 0.0

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
        # ------------------------------------------------------------------
        # RANDUMGEHUNG (a_bp < a_mem): die Membran erzeugt ihren Volumenfluss
        # über ihrer GANZEN Fläche, der Quetschfilm liegt aber nur unter der
        # Backplate. Der außerhalb erzeugte Anteil läuft NICHT unter die
        # Platte — er tritt über den Ringraum zwischen Plattenrand und
        # Membraneinspannung ein, der um die Absatztiefe (B&K: 0.3 mm gegen
        # 21 µm Spalt) tiefer ist und dessen Schmierwiderstand deshalb um
        # (h/h_ring)³ ~ 1e-4 kleiner ist. Er wird am FILMRAND eingespeist;
        # umgekehrt sieht die Membran dort den Randdruck statt des
        # Filmdrucks. Mit φ = 1 − r²/a_mem² ist der Flussanteil ÜBER der
        # Platte geschlossen bekannt:
        #     f_in = ∫_0^a_bp φ dA / ∫_0^a_mem φ dA = u·(2 − u),
        #     u = (a_bp/a_mem)²,
        # und für einen quasi drucklosen Rand skaliert die Filmimpedanz
        # exakt mit f_in² (Quelle UND Projektion schrumpfen). Gegenprobe 37
        # hält die lochfreie Platte gegen die geschlossene Form, Gegenprobe
        # 38 die ganze Kette gegen zwei gemessene B&K-Kapseln.
        # a_bp >= a_mem -> f_in = 1, q_by = 0: Bestand.
        Sphi_tot = max(self.S_eff_mem, Sphi)
        f_in = Sphi / Sphi_tot
        q_by = 1.0 - f_in                                 # Umgehungsfluss
        S_out = Sphi_tot - Sphi                           # Modengewicht außen
        src_a = phi * A / Sphi_tot                        # Membran treibt (U=1)
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
            if q_by > 0.0 and not ring_open:
                # kein Randspalt: der Ringraum ist eine Sackgasse, die nur
                # über den Filmrand entleert -> Eintritt in die äußerste Zelle
                rhs[N - 1, 0] += q_by
            if ring_open:
                G_edge = self._fld_gedge_geom * K_face[f, N]
                ab[1, N - 1] += G_edge
                ab[0, N] = -G_edge                        # Zelle N-1 <-> Rand
                ab[2, N - 1] = -G_edge
                # Randknoten: Filmzustrom + Leitungs-Y11 (= D/B)
                ab[1, N] = G_edge + D_l[f] / B_l[f]
                # Rückport treibt (Fall b): -Y12·p_rück = +1/B
                rhs[N, 1] = 1.0 / B_l[f]
                rhs[N, 0] = q_by              # Umgehung in den Randknoten
            sol = _solve_banded((1, 1), ab, rhs)
            p_a, p_b = sol[:N, 0], sol[:N, 1]
            # Druck im Ringraum außerhalb der Platte = Randdruck
            p_out_a = sol[N, 0] if ring_open else p_a[-1]
            p_out_b = sol[N, 1] if ring_open else p_b[-1]
            alpha = (np.sum(p_a * phi * A) + S_out * p_out_a) / Sphi_tot
            beta = (np.sum(p_b * phi * A) + S_out * p_out_b) / Sphi_tot
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

    def self_noise(self, f_min=20.0, f_max=20000.0, n_points=1200,
                   spectrum=None, refine=True):
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

        ``spectrum``: ein bereits gerechnetes :meth:`noise_spectrum`. Dann
        wird dessen Frequenzraster benutzt (auf [f_min, f_max] beschnitten)
        statt ein eigenes mit ``n_points`` aufzubauen. Das ist keine
        Bequemlichkeit, sondern Laufzeit: ``noise_spectrum`` ruft
        ``transfer_function`` auf, und mit ``axial_body_model='bem'``
        steckt darin je Frequenz ein Randelementsystem. Ein zweites
        Raster verdoppelt die Rechenzeit für nichts.

        ``refine``: eine lokale NACHVERFEINERUNG um das Maximum des
        gewichteten Integranden. Sie ist nötig, weil der Integrand
        S_p = S_v/|H|² genau dort Spitzen hat, wo die Kapsel TAUB ist —
        an einer Antiresonanz von |H|. Die Beispielkapsel hat eine bei
        11.2 kHz mit 30 Hz Halbwertsbreite (Q ≈ 375); ein logarithmisches
        Raster über drei Dekaden trifft die nicht. Ohne Verfeinerung lag
        der A-Pegel mit 1200 Punkten 0.53 dB zu tief (0.21 statt 0.74 dB);
        mit einer einzigen Runde von 129 Punkten über ±4 % bleiben 0.05 dB
        (Gegenprobe 46). Abschalten nur, um genau das zu zeigen.
        """
        if spectrum is None:
            f = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))
            sp = self.noise_spectrum(f)
        else:
            f0 = np.asarray(spectrum["frequency_hz"], dtype=float)
            m = (f0 >= f_min) & (f0 <= f_max)
            if int(np.sum(m)) < 50:
                raise ValueError(
                    f"self_noise: das übergebene Spektrum hat im Band "
                    f"{f_min:.0f}…{f_max:.0f} Hz nur {int(np.sum(m))} "
                    "Stützstellen.")
            f = f0[m]
            sp = {k: (v[m] if isinstance(v, np.ndarray) and v.shape == f0.shape
                      else v) for k, v in spectrum.items()}
        _kk = ("psd_pa2_hz", "frac_front", "frac_mem", "frac_rear")
        if refine and f.size >= 8:
            fp = float(f[int(np.argmax(sp["psd_pa2_hz"]
                                       * self._a_weighting(f) ** 2))])
            f_ref = np.logspace(np.log10(max(fp / 1.04, f_min)),
                                np.log10(min(fp * 1.04, f_max)), 129)
            sp_ref = self.noise_spectrum(f_ref)
            f = np.concatenate([f, f_ref])
            sp = {k: np.concatenate([sp[k], sp_ref[k]]) for k in _kk}
            o = np.argsort(f)
            f = f[o]
            sp = {k: v[o] for k, v in sp.items()}
            f, uq = np.unique(f, return_index=True)
            sp = {k: v[uq] for k, v in sp.items()}
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

    def _homogenization_note(self, lang="de"):
        """summary()-Zeile zur Homogenisierungsgrenze (Gegenprobe 48)."""
        en = str(lang).strip().lower() == "en"
        if self.squeeze_model == "3d":
            return ("— (3D resolves the holes)" if en
                    else "— (3D löst die Löcher auf)")
        if self.n_th == 0 and self.ring_vent_w <= 0.0:
            return ("— (closed backplate)" if en
                    else "— (Backplate geschlossen)")
        lim = self.homogenization_limit()
        f_txt = (f"{lim['f_limit'] / 1e3:9.1f} kHz"
                 if np.isfinite(lim["f_limit"]) else "        ∞")
        if lim["cause"] == "ring":
            note = (f"{f_txt} (hole-circle representation, local limit "
                    f"{lim['f_hom'] / 1e3:.1f} kHz)" if en
                    else f"{f_txt} (Lochkreis-Darstellung, lokale Grenze "
                    f"{lim['f_hom'] / 1e3:.1f} kHz)")
        else:
            note = (f"{f_txt} (ρ_hole {lim['rho'] * 1e3:.2f} mm, "
                    f"T {lim['tension']:.0f} N/m)" if en
                    else f"{f_txt} (ρ_Loch {lim['rho'] * 1e3:.2f} mm, "
                    f"T {lim['tension']:.0f} N/m)")
        if lim["f_limit"] < self._F_BAND_TOP:
            note += (" — ABOVE: use 3D" if en else " — DARÜBER: 3D nehmen")
        return note

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
        if self.squeeze_model == "3d":
            g_nr, g_np = self._g3d["Nr"], self._g3d["Np"]
            sm_note = sm_note[:-1] + _t(
                f"; Gitter {'fein' if self.grid_3d == 'fine' else 'grob'} "
                f"{g_nr} × {g_np})",
                f"; {self.grid_3d} grid {g_nr} × {g_np})")
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
            _row(_t("Massenfaktor 8/(z₁²·g):", "Mass factor 8/(z₁²·g):"),
                 f"{self._piston_factor:9.4f}"),
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
        ]
        if self.r_post > 0.0:
            # Ringmembran: die drei Zahlen, die den Unterschied ausmachen.
            _md = self._ring_modes()
            lines += [
                _row(_t("Mittenterminierung r_i/a:",
                        "Center termination r_i/a:"),
                     f"{self.rho_post:9.4f} "
                     + _t(f"(⌀{2e3 * self.r_post:.2f} mm — Ringmembran)",
                          f"(⌀{2e3 * self.r_post:.2f} mm — annular)")),
                _row(_t("  Nachgiebigkeitsfaktor g:",
                        "  compliance factor g:"),
                     f"{self._ring_g:9.4f} "
                     + _t("(1 = Kreismembran)", "(1 = circular)")),
                _row(_t("  Eigenwert z_1 / Maximum:",
                        "  eigenvalue z_1 / maximum:"),
                     f"{float(_md['z'][0]):9.4f}"
                     + _t(f" (Kreis 2.4048), w_max bei r/a = "
                          f"{np.sqrt(self._phi_umax):.3f}",
                          f" (circular 2.4048), w_max at r/a = "
                          f"{np.sqrt(self._phi_umax):.3f}")),
            ]
        lines += [
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
            _row(_t("Loch-Homogenisierung bis:",
                    "Hole homogenization up to:"),
                 self._homogenization_note(lang)),
            _row(_t("Resonanz (Modell):", "Resonance (model):"),
                 f"{self.f_res:9.1f} Hz"),
            _row(_t("Resonanz aus Vorspannung/E:", "Resonance from "
                    "tension/E:"),
                 f"{self.f_res_from_tension:9.1f} Hz"),
            _t(f"  (exakte Modalfrequenz:      {self.f_res_modal_exact:9.1f} "
               f"Hz — Abweichung der Kette "
               f"{100.0 * (self.f_res_from_tension / self.f_res_modal_exact - 1.0):+.2f} %, "
               "nur über die Biegesteifigkeit)",
               f"  (exact modal frequency:     {self.f_res_modal_exact:9.1f} "
               f"Hz — lumped model deviates "
               f"{100.0 * (self.f_res_from_tension / self.f_res_modal_exact - 1.0):+.2f} %, "
               "via bending stiffness only)"),
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
        if (self.include_diffraction and _HAS_SCIPY
                and self.axial_body_model == "bem"
                and self.architecture != "dual_diaphragm"
                and self.rear_open):
            # Wo der BEM den Rückeinlass auf die Kontur legt
            # (s. _bem_rear_inlet_weights, Gegenprobe 44).
            wo = (_t("hintere Stirnfläche", "rear end face")
                  if self.cavity_hole_position == "end"
                  else _t("Bohrungskranz im Mantel",
                          "ring of holes in side wall"))
            eng = self.body_length is not None and (
                abs(self.d_ext - self.body_length) > 0.25 * self.d_ext
                if self.cavity_hole_position == "end"
                else self.body_length <= self.d_ext)
            lines.append(_row(
                _t("BEM-Rückeinlass:", "BEM rear inlet:"),
                f"{wo}, {self.d_ext * 1e3:.2f} mm"
                + (_t("  ← passt nicht zu body_length",
                      "  <- inconsistent with body_length") if eng else "")))
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
# Selbsttest: startet die Gegenproben in tests/ (pytest)
# ===========================================================================
if __name__ == "__main__":
    # Selbsttest: die Gegenproben liegen in tests/ und laufen mit pytest,
    # parallel über alle Kerne, wenn pytest-xdist installiert ist.
    # Weitere Argumente gehen an pytest, z. B.:
    #     python microphone_capsule.py -m "not slow"   # schnelle Stufe
    #     python microphone_capsule.py -k gp48         # eine Gegenprobe
    #     python microphone_capsule.py --lf            # nur die zuletzt
    #                                                  # gescheiterten
    import importlib.util
    import os
    import sys
    from pathlib import Path

    try:
        import pytest
    except ImportError:
        sys.exit("Der Selbsttest braucht pytest: "
                 "pip install -r requirements-dev.txt")
    _args = sys.argv[1:]
    # xdist nur nachsehen, nicht importieren (pytest will es selbst laden).
    # worksteal: ein freier Worker übernimmt wartende Tests eines belegten.
    if importlib.util.find_spec("xdist") is not None:
        if not any(a.startswith(("-n", "--numprocesses")) for a in _args):
            _args = ["-n", "auto", *_args]
        if not any(a.startswith(("--dist", "-d")) for a in _args):
            _args = ["--dist", "worksteal", *_args]
    # testpaths greift nur im Projektverzeichnis
    os.chdir(Path(__file__).resolve().parent)
    sys.exit(pytest.main(_args))
