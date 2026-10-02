"""Gegenproben: Externe Referenzen (FEM, Messung)."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp32_externe_referenz_fem_veroffentlicht(stand):
    """Gegenprobe 32: EXTERNE Referenz (FEM, veröffentlicht)."""
    # Erste Verankerung des Modells an einer fremden, in sich konsistenten
    # Quelle: Šimonová/Honzík, J. Acoust. Soc. Am. 159(5), 4512–4523 (2026),
    # doi:10.1121/10.0043902. Deren FEM-Referenz (COMSOL 6.3, 3D thermo-
    # viskos, ~6·10⁶ Freiheitsgrade) liefert die mittlere Membranauslenkung
    # über die Gesamtfläche für eine Kapsel, deren Parameter VOLLSTÄNDIG in
    # derselben Arbeit stehen — anders als bei den üblichen Datenblättern,
    # wo Geometrie und Kurve aus verschiedenen Quellen stammen.
    #
    # Der Prüfling ist bewusst exotisch (R = 18 mm, h_g = 230 µm, nur VIER
    # Bohrungen, f_res = 1040 Hz) und stresst damit genau die Filmphysik:
    # bei der Resonanz ist die viskose Grenzschicht 95 µm dick gegen 115 µm
    # halben Spalt — der Film sitzt mitten im Übergang von Poiseuille zu
    # Trägheit. Ein rein statisches Škvor-Modell müsste hier scheitern; die
    # Frequenzkorrektur Φ(ω) (s. _film_R_dynamic) trifft die Güte.
    #
    # Verankert wird DREIERLEI, mit unterschiedlichem Anspruch:
    # a) TIEFTON: bis 300 Hz < 1 dB — die quasistatische Nachgiebigkeit.
    # b) GÜTE: die Resonanzüberhöhung ist die eigentliche Dämpfungsprobe
    #    und wird auf < 1 dB getroffen.
    # c) RESONANZLAGE und Verlauf: als SPERRKLINKEN (Abstand zur FEM,
    #    2D und 3D), damit sie nur besser werden können.
    #
    # GESCHICHTE DER RESONANZLAGE. Lange lag das Modell rund 10 % zu TIEF
    # (2D 497, 3D 495 Hz gegen 550 Hz), und der Reihe nach wurden
    # Zellterm (Gegenprobe 36), Mündungsmasse, Membranmasse,
    # Spaltnachgiebigkeit, Homogenisierung (Gegenproben 48/51) und
    # Lochkreis-Darstellung (Gegenprobe 60) ausgeschlossen oder behoben.
    # AUFGEKLÄRT (Gegenprobe 67) — zweiteilig:
    #  * PRÜFAUFBAU: der Klassen-Standard delay_length = 3 mm hängte
    #    unbemerkt ein Laufzeitglied an (3.05 cm³, +39 % Rückvolumen); die
    #    FEM-Kapsel hat keins, alle anderen Referenzproben setzen 0 schon
    #    ausdrücklich. Ohne es lag das Modell 6 % zu HOCH (2D 584, 3D
    #    581 Hz) und war um 1.1…1.3 dB zu schwach bedämpft — der
    #    Volumenfehler hatte beides verdeckt.
    #  * PHYSIK: zwischen Reynolds-Film und Bohrungsrohr fehlte die
    #    Umlenkung selbst (Mündung Spalt → Bohrung, s. _muendung_spalt).
    #    Bei h/a = 0.46 trägt sie 13 % des Widerstands und 7 % der Masse
    #    des Lochpfads. Hergeleitet aus der instationären Stokes-Zelle,
    #    nicht an der FEM abgeglichen.
    #    Mit beiden: 2D 561, 3D 558 Hz (+2.0/+1.4 %), Überhöhung
    #    +7.32/+7.18 dB gegen +6.74 dB, RMS über die sechs FEM-Punkte
    #    0.43/0.23 dB (vorher 0.95/0.72 dB).
    # REST: der Film ist noch um ~0.5 dB zu schwach bedämpft (die FEM
    # rechnet thermoviskos, Gegenprobe 58), und das Dublett der vier
    # Bohrungen (FEM 3500/4200 Hz) liegt im 3D-Löser bei 3350/4110 Hz —
    # die Mündungsmasse hat es um 2 % gesenkt (vorher 3421/4127 Hz). Im
    # Kerbenband trägt der Lochzweig damit 4…9 % zu viel Masse oder zu
    # wenig Nachgiebigkeit. An der Mündung liegt das nicht: ihre Masse
    # ist im Trägheitsgrenzfall die der Potentialströmung (Gegenprobe
    # 67). Offen.
    #
    # DUBLETT. Die FEM zeigt im Kerbenband ZWEI Minima; der homogeni-
    # sierende 2D-Pfad kann nur eines haben, der 3D-Feldlöser, der die
    # vier Bohrungen diskret auflöst, zeigt beide (Prüfung d).
    #
    # MEMBRAN (Gegenprobe 62): die Arbeit beschreibt die Membran mit
    # Spannung und Resonanz nach der MEMBRANformel (116.27 N/m ->
    # j01/(2πa)·√(T/ρt) = 1040 Hz); mit Biegesteife (λ = 0.0125) läge die
    # Resonanz bei 1053 Hz. Verglichen wird deshalb ohne Randschicht der
    # Folie.
    MicrophoneCapsule._RANDSCHICHT = False
    if _HAS_SCIPY:
        # COMSOL-Referenz, auf 100 Hz normiert (Fig. 4 der Arbeit)
        ref32 = ((100.0, 0.00), (200.0, 0.70), (300.0, 1.98), (500.0, 6.20),
                 (550.0, 6.74), (1000.0, -7.36), (2000.0, -25.20))
        f32 = np.array([p[0] for p in ref32])
        a32_ref = np.array([p[1] for p in ref32])
        c32 = MicrophoneCapsule(
            membrane_material={"rho": 1944.0, "E": 4.0e9, "nu": 0.35},
            membrane_resonance_hz=1040.0, membrane_diameter=36.0e-3,
            membrane_thickness=25e-6, membrane_tension=116.27,
            air_gap=230e-6, backplate_diameter=36.0e-3,
            backplate_thickness=1.6e-3, bias_voltage=1.0,
            architecture="single", n_through_holes=4,
            through_hole_diameter=1.0e-3, through_hole_pcd=2 * 8.4853e-3,
            n_blind_holes=0, rear_network_enabled=True,
            cavity_length=7.6e-3, delay_length=0.0, n_cavity_holes=0,
            fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, include_diffraction=False,
            squeeze_model="2d")
        a32 = 20.0 * np.log10(np.abs(c32.transfer_function(f32)))
        a32 = a32 - a32[0]
        # a) Tiefton
        assert np.max(np.abs((a32 - a32_ref)[:3])) < 1.0, \
            (f"Tiefton muss die FEM treffen "
             f"({np.max(np.abs((a32 - a32_ref)[:3])):.2f} dB)")
        # b) Güte — die eigentliche Dämpfungsprobe
        fs32 = np.geomspace(250.0, 900.0, 400)
        A32 = 20.0 * np.log10(np.abs(c32.transfer_function(fs32)))
        A32 = A32 - 20.0 * np.log10(
            np.abs(c32.transfer_function(np.array([100.0]))[0]))
        peak32, fpk32 = float(A32.max()), float(fs32[int(np.argmax(A32))])
        assert abs(peak32 - 6.74) < 1.0, \
            (f"Resonanzüberhöhung muss die FEM treffen — das ist die "
             f"Dämpfungsprobe ({peak32:+.2f} statt +6.74 dB)")
        # c) Resonanzlage und Verlauf gegen die FEM als Sperrklinken
        det32 = fpk32 / 550.0
        stand.sperrklinke("verstimmung_2d", abs(det32 - 1.0), "",
                          "|f_Res(2D)/550 Hz − 1| gegen die FEM",
                          besser="kleiner", toleranz=0.01)
        rms32 = float(np.sqrt(np.mean((a32 - a32_ref)[1:] ** 2)))
        stand.sperrklinke("rms_fem_2d", rms32, "dB",
                          "RMS 2D gegen die sechs FEM-Punkte 200…2000 Hz",
                          besser="kleiner", toleranz=0.05)
        # ... und der diskret rechnende 3D-Löser (konturtreue Mündungen)
        c32_3d = MicrophoneCapsule(**{**dict(
            membrane_material={"rho": 1944.0, "E": 4.0e9, "nu": 0.35},
            membrane_resonance_hz=1040.0, membrane_diameter=36.0e-3,
            membrane_thickness=25e-6, membrane_tension=116.27,
            air_gap=230e-6, backplate_diameter=36.0e-3,
            backplate_thickness=1.6e-3, bias_voltage=1.0,
            architecture="single", n_through_holes=4,
            through_hole_diameter=1.0e-3, through_hole_pcd=2 * 8.4853e-3,
            n_blind_holes=0, rear_network_enabled=True,
            cavity_length=7.6e-3, delay_length=0.0, n_cavity_holes=0,
            fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, include_diffraction=False),
            "squeeze_model": "3d"})
        fs32_3 = np.linspace(400.0, 600.0, 81)
        H32_3 = np.abs(c32_3d.transfer_function(fs32_3))
        fpk32_3 = float(fs32_3[int(np.argmax(H32_3))])
        # Überhöhung des 3D-Lösers gegen die FEM: die FEM rechnet Navier–
        # Stokes statt Reynolds — der Abstand zeigt, wie viel Dämpfung dem
        # Reynolds-Film fehlt (Gegenprobe 58)
        peak32_3 = float(20.0 * np.log10(np.max(H32_3) / np.abs(
            c32_3d.transfer_function(np.array([100.0]))[0])))
        stand.wert("ueberhoehung_3d_minus_fem", peak32_3 - 6.74, "dB",
                   "Resonanzüberhöhung 3D minus FEM (+6.74 dB)")
        stand.sperrklinke("verstimmung_3d", abs(fpk32_3 / 550.0 - 1.0), "",
                          "|f_Res(3D)/550 Hz − 1| gegen die FEM",
                          besser="kleiner", toleranz=0.01)
        stand.wert("f_res_3d_zu_2d", fpk32_3 / fpk32, "",
                   "Resonanzlage 3D/2D (Anteil der diskreten Bohrungen)")
        a32_3 = 20.0 * np.log10(np.abs(c32_3d.transfer_function(f32)))
        rms32_3 = float(np.sqrt(np.mean(((a32_3 - a32_3[0])
                                         - a32_ref)[1:] ** 2)))
        stand.sperrklinke("rms_fem_3d", rms32_3, "dB",
                          "RMS 3D gegen die sechs FEM-Punkte 200…2000 Hz",
                          besser="kleiner", toleranz=0.05)
        # d) DUBLETT: die FEM hat im Kerbenband ZWEI Minima (3500 und
        #    4200 Hz). Das ist ein Effekt der vier DISKRETEN Bohrungen —
        #    der homogenisierende 2D-Pfad kann prinzipiell nur eines
        #    haben, der 3D-Feldlöser beide. Genau das wird hier geprüft;
        #    es ist die Strukturaussage hinter der Schranke oben.
        def _minima32(sm):
            cc = MicrophoneCapsule(**{**dict(
                membrane_material={"rho": 1944.0, "E": 4.0e9, "nu": 0.35},
                membrane_resonance_hz=1040.0, membrane_diameter=36.0e-3,
                membrane_thickness=25e-6, membrane_tension=116.27,
                air_gap=230e-6, backplate_diameter=36.0e-3,
                backplate_thickness=1.6e-3, bias_voltage=1.0,
                architecture="single", n_through_holes=4,
                through_hole_diameter=1.0e-3,
                through_hole_pcd=2 * 8.4853e-3, n_blind_holes=0,
                rear_network_enabled=True, cavity_length=7.6e-3,
                delay_length=0.0,
                n_cavity_holes=0, fabric_front_rayl=0.0,
                fabric_rear_rayl=0.0, include_diffraction=False),
                "squeeze_model": sm})
            ff = np.geomspace(2800.0, 5000.0, 140)
            aa = 20.0 * np.log10(np.abs(cc.transfer_function(ff)))
            return [float(ff[i]) for i in range(1, len(ff) - 1)
                    if aa[i] < aa[i - 1] and aa[i] < aa[i + 1]]
        m2_32, m3_32 = _minima32("2d"), _minima32("3d")
        assert len(m2_32) == 1, \
            (f"der homogenisierende 2D-Pfad kann nur EIN Minimum haben "
             f"({np.round(m2_32)})")
        assert len(m3_32) == 2, \
            (f"der 3D-Feldlöser muss das Dublett der diskreten Bohrungen "
             f"zeigen ({np.round(m3_32)} gegen FEM 3500/4200 Hz)")
        print(f"Externe FEM-Referenz (Šimonová/Honzík 2026, COMSOL): "
              f"Tiefton {np.max(np.abs((a32 - a32_ref)[:3])):.2f} dB, "
              f"Resonanzüberhöhung {peak32:+.2f} gegen +6.74 dB "
              f"(Güte getroffen); Dublett der vier Bohrungen: 2D "
              f"{len(m2_32)} Minimum, 3D {np.round(m3_32).astype(int)} Hz "
              f"gegen FEM 3500/4200; Lage {fpk32:.0f} gegen 550 Hz "
              f"({100 * (det32 - 1):+.1f} %, 3D {fpk32_3:.0f} Hz, "
              f"{100 * (fpk32_3 / 550.0 - 1):+.1f} %); RMS gegen die FEM "
              f"2D {rms32:.2f} dB, 3D {rms32_3:.2f} dB  OK")


# Zuckerwar, JASA 64, 1278 (1978): B&K 4134 und 4146 — Tab. I (Geometrie),
# Tab. II (Ersatzelemente), Fig. 6/7 (digitalisiert). Gegenproben 38, 58.
_NI38 = {"rho": 8900.0, "E": 200.0e9, "nu": 0.31}   # Tab. I: Nickel
_BK38 = {
    # (Parameter, Tab.-II-Ersatzelemente, Fig.-6/7-Punkte)
    "4134": (dict(
        membrane_material=_NI38, membrane_resonance_hz=None,
        membrane_diameter=2 * 4.445e-3, membrane_thickness=5.0e-6,
        membrane_tension=3162.3, air_gap=2.077e-5,
        backplate_diameter=2 * 3.607e-3,
        backplate_thickness=0.843e-3, bias_voltage=28.0,
        architecture="single", n_through_holes=6,
        through_hole_diameter=2 * 5.080e-4,
        through_hole_pcd=2 * 2.032e-3, n_blind_holes=0,
        ring_vent_width=0.838e-3, ring_vent_length=3.048e-4,
        rear_network_enabled=True, delay_length=0.0,
        cavity_length=1.264e-7 / (np.pi * 3.607e-3**2),
        n_cavity_holes=0, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, include_diffraction=False,
        squeeze_model="2d"),
        dict(M=955.0, C_M=0.485e-13, C_A=9.01e-13, R=18.9e7),
        (1e3, 2e3, 3e3, 5e3, 7e3, 1e4, 1.3e4, 1.6e4, 2e4),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.14, -0.71, -1.11, -3.06),
        (3.87, 5.68, 8.01, 14.50, 23.76, 37.86, 51.12, 65.65, 81.75),
        (0.60, 2.5, 0.20)),                # Schranken: dB/Grad/C_A
    "4146": (dict(
        membrane_material=_NI38, membrane_resonance_hz=None,
        membrane_diameter=2 * 8.890e-3, membrane_thickness=5.0e-6,
        membrane_tension=2140.9, air_gap=2.655e-5,
        backplate_diameter=2 * 6.617e-3,
        backplate_thickness=1.6633e-3, bias_voltage=28.0,
        architecture="single",
        through_hole_rings=[(12, 2 * 4.763e-3), (6, 2 * 2.375e-3),
                            (1, 0.0)],
        through_hole_diameter=2 * 4.763e-4, n_blind_holes=0,
        ring_vent_width=1.473e-3, ring_vent_length=3.556e-4,
        rear_network_enabled=True, delay_length=0.0,
        cavity_length=6.736e-7 / (np.pi * 6.617e-3**2),
        n_cavity_holes=0, fabric_front_rayl=0.0,
        fabric_rear_rayl=0.0, include_diffraction=False,
        squeeze_model="2d"),
        dict(M=239.0, C_M=11.45e-13, C_A=48.6e-13, R=1.91e7),
        (1e3, 2e3, 3e3, 5e3, 7e3, 1e4, 1.3e4),
        (0.00, 0.32, 0.57, 1.42, 1.59, -4.67, -8.02),
        (6.43, 12.30, 22.27, 42.15, 72.63, 122.22, 132.75),
        (1.60, 10.0, 0.10)),
}


def _messpruefling(par):
    """Der GEMESSENE Prüfling zu einer Membranspannung aus Zuckerwars
    Tabelle bzw. dem COMSOL-Modell (Gegenprobe 62).

    Zuckerwar bestimmt die Spannung aus der gemessenen ersten Vakuum-
    resonanz mit der MEMBRANformel T = 6.825·a²·f_R1²·ρt (NASA-Bericht
    PGSTR-PH77-48, 1977, Gl. 2-34); COMSOLs Membraninterface rechnet mit
    diesem Wert als reine Membran. Die Zahl ist also eine membran-
    äquivalente Spannung, in der die Randschicht der Folie schon steckt.
    Gemessen ist die Resonanz: der physikalische Prüfling bekommt sie
    vorgegeben, j01/(2πa)·√(T/ρt), und das Modell rechnet seine
    Randschicht selbst (bei vorgegebener Spannung zählte es sie doppelt,
    die Resonanz läge 0.6 % zu hoch)."""
    a = 0.5 * par["membrane_diameter"]
    sig = par["membrane_material"]["rho"] * par["membrane_thickness"]
    f_vak = (2.404825557695773 / (2.0 * np.pi * a)
             * np.sqrt(par["membrane_tension"] / sig))
    return dict(par, membrane_resonance_hz=f_vak)


@pytest.mark.feld3d
def test_gp38_externe_referenz_b_k_4134(stand):
    """Gegenprobe 38: EXTERNE Referenz B&K 4134/4146."""
    # Zweite fremde Verankerung, und die erste gegen eine MESSUNG:
    # A. J. Zuckerwar, "Theoretical response of condenser microphones",
    # J. Acoust. Soc. Am. 64(5), 1278–1285 (1978), doi:10.1121/1.382112.
    # Die Arbeit ist als Referenz ungewöhnlich vollständig: Tabelle I
    # enthält BEIDE Kapseln komplett (Membranradius, Dicke, Dichte,
    # Vakuumresonanz, Vorspannung, Spalt, Backplate-Radius, Rückkammer-
    # volumen, Lochkreise mit Zahl/Radius/Tiefe und den Randschlitz),
    # Tabelle II die Ersatzelemente, Fig. 6/7 Amplitude UND Phase gegen
    # Messwerte (elektrostatischer Aktuator, also gleichförmiger Antrieb
    # ohne Beugung; Polarisation nur 28 V, damit die statische Auslenkung
    # vernachlässigbar bleibt — bei uns 0.07 bzw. 0.19 % Feder-Erweichung).
    # Der übliche Fallstrick, Geometrie und Kurve aus verschiedenen
    # Quellen zu mischen, entfällt damit. Die Tabelle ist in sich
    # konsistent: aus Tabelle I folgen f_vak, M = (4/3)ρt/S (Rayleigh;
    # die Kette nimmt seit Gegenprobe 55 8/j01² statt 4/3),
    # C = S²/(8πT) und Q = √(M/C_ser)/R auf vier Stellen genau.
    #
    # ABBILDUNG. Der Randschlitz ist genau unser ``ring_vent_*``: beim
    # 4134 liegt er bei 4.026 ± 0.419 mm, reicht also exakt vom Platten-
    # rand (3.607) bis zur Membraneinspannung (4.445). Beim 4146 hat die
    # Platte DREI Lochkreise (12/6/1) mit zwei Radien und drei Tiefen; wir
    # kennen einen Durchmesser und eine Tiefe. Zusammengefasst wird auf
    # 19 Bohrungen mit r = 0.4763 mm; die Ersatztiefe aus der Parallel-
    # schaltung ist für Widerstand (~n r⁴/l) und Masse (~n r²/l) praktisch
    # gleich (1.6658 / 1.6608 mm). Eine exakte diskrete Reynolds-Rechnung
    # zeigt, dass diese Zusammenfassung nur 1.2 % kostet.
    #
    # GENAUIGKEIT DER REFERENZ. M, C_M, C_A und R stehen als ZAHLEN in den
    # Tabellen. Die Punkte aus Fig. 6/7 sind dagegen von der gedruckten
    # Kurve abgenommen (Achsen über die Teilstriche kalibriert, Kurve
    # spaltenweise verfolgt, Messsymbole per gleitendem Median entfernt);
    # die Restunsicherheit liegt bei etwa ±0.15 dB und ±2°. Die Schranken
    # unten sind entsprechend gesetzt und nicht enger. Der SPALTWIDERSTAND
    # war bis Gegenprobe 59 eine Sperrklinke (Streuung der Škvor-Zell-
    # regel, s. u.); seit dem Makroelement (Gegenprobe 60) rechnet das
    # 2D-Modell den Film am Lochkreis exakt, und Tab. II ist Zuckerwars
    # eigene Näherung — die Abweichung ist ein Stand-Wert.
    #
    # WAS SIE GEZEIGT HAT. Vor der Randumgehung (Gegenprobe 37) lag der
    # Spaltwiderstand um +39 % (4134) bzw. +106 % (4146) zu hoch und der
    # Frequenzgang um 0.76 bzw. 3.22 dB RMS daneben — der 4146 verlor
    # seine Resonanzüberhöhung ganz. Eine unabhängige, gitterkonvergente
    # Lösung der inkompressiblen Reynolds-Gleichung mit DISKRETEN
    # Bohrungen in (r,φ) hat den Fehler zerlegt: das Radialfeld selbst ist
    # exakt (0.1 % gegen die analytische Lösung), die Filmberandung war es
    # nicht. Was bleibt, ist die Streuung der Škvor-Zellregel
    # q = n·r²/a_bp²: über 16 Ringgeometrien liegt sie zwischen 0.78 und
    # 1.44 gegen die exakte Lösung. Der 4146 sitzt mit +35 % am oberen
    # Ende; die Regel selbst bleibt damit ein offener Punkt (die
    # naheliegende Alternative „Zellfläche = Lochabstand²" wurde geprüft
    # und ist SCHLECHTER: RMS(log) 0.33 gegen 0.19).
    # NACHTRAG (Gegenproben 58, 59): der 2D-Treffer am 4134 ist kein Beleg
    # für das 2D-Modell. Auf B&Ks Originalgeometrie überschätzt es den
    # Filmwiderstand gegen die thermoviskose FEM um rund 60 %, und
    # Zuckerwars Prüfling war stärker gedämpft als heutige 4134 — beides
    # gleicht sich hier aus. Die Probe bleibt als Referenz gegen die
    # Messung stehen; eine bessere Lochkreis-Darstellung wird sie
    # verschieben.
    # NACHTRAG (Gegenprobe 60): so ist es gekommen. Mit dem Makroelement
    # rechnet 2D den Film am Lochkreis exakt (Gegenprobe 60) und liegt am
    # 4134 wie 3D über der Messung (2.1 gegen 1.7 dB RMS, vorher 0.30 dB,
    # Tab. II-Widerstand −27 % statt +10 %); die Messschranken dort gelten
    # deshalb nicht mehr dem 2D-Modell, sondern dem Befund: 2D und 3D
    # liegen gleich, beide im Hochton über der Messung. Am 4146 trifft 2D
    # die Messung jetzt BESSER (0.63 statt 1.09 dB RMS, Phase 4.8 statt
    # 7.2°, Tab. II-Widerstand +9 % statt +35 %) — dort gibt es keinen
    # Ausgleich zu verdecken.
    if _HAS_SCIPY:
        bk38 = _BK38
        res38 = {}
        for nm38, (par38, tab38, f38, a38, p38, lim38) in bk38.items():
            # a) Ersatzelemente der Membran: analytisch, müssen exakt sein.
            #    Tab. II ist Zuckerwars MEMBRANmodell — verglichen wird die
            #    Kette ohne Randschicht der Folie mit seiner Spannung.
            #    Zuckerwar rechnet mit dem Rayleigh-Wert 4/3; die Kette
            #    seit Gegenprobe 55 mit 8/j01² (3.75 % schwerer, damit die
            #    Resonanz der Membran exakt ist). Geprüft wird ρt/S.
            MicrophoneCapsule._RANDSCHICHT = False
            try:
                cm38 = MicrophoneCapsule(**par38)
            finally:
                MicrophoneCapsule._RANDSCHICHT = True
            M38 = cm38.M_A_mem * cm38._mass_factor_rayleigh / cm38._piston_factor
            assert abs(M38 / tab38["M"] - 1.0) < 5e-3, \
                (f"{nm38}: (4/3)ρt/S muss Tab. II treffen "
                 f"({M38:.1f} gegen {tab38['M']:.0f})")
            assert abs(cm38.C_A_mem / tab38["C_M"] - 1.0) < 5e-3, \
                (f"{nm38}: C_A der Membran muss S²/(8πT) sein "
                 f"({cm38.C_A_mem:.4e} gegen {tab38['C_M']:.4e})")
            #    Der gemessene Prüfling (Vakuumresonanz vorgegeben, mit
            #    Randschicht, s. _messpruefling) ist bei gleicher Resonanz
            #    um 2·√(D/T)/a steifer.
            c38 = MicrophoneCapsule(**_messpruefling(par38))
            # b) Luftzweig bei 250 Hz gegen Tabelle II
            w38 = np.array([2.0 * np.pi * 250.0])
            Zr38 = c38._membrane_port_impedance(w38)[3][0]
            CA38 = -1.0 / (w38[0] * Zr38.imag)
            eC38 = CA38 / tab38["C_A"] - 1.0
            eR38 = Zr38.real / tab38["R"] - 1.0
            assert abs(eC38) < lim38[2], \
                (f"{nm38}: Luftnachgiebigkeit gegen Tab. II "
                 f"({100 * eC38:+.1f} %)")
            stand.wert(f"spaltwiderstand_{nm38}_makro", eR38, "",
                       "R/R(Tab. II) − 1 (Tab. II: Zuckerwars Näherung)")
            # c) Frequenzgang gegen Fig. 6/7 (Amplitude UND Phase)
            fa38 = np.asarray(f38, dtype=float)
            Hn38 = (c38.transfer_function(fa38)
                    / c38.transfer_function(np.array([250.0]))[0])
            am38 = 20.0 * np.log10(np.abs(Hn38)) - np.asarray(a38)
            ph38 = (-np.rad2deg(np.unwrap(np.angle(Hn38)))
                    - np.asarray(p38))
            rms_a38 = float(np.sqrt(np.mean(am38**2)))
            rms_p38 = float(np.sqrt(np.mean(ph38**2)))
            if nm38 == "4146":
                assert rms_a38 < lim38[0], \
                    (f"{nm38}: Amplitude gegen Fig. 7 "
                     f"({rms_a38:.2f} dB RMS)")
                assert rms_p38 < lim38[1], \
                    (f"{nm38}: Phase gegen Fig. 7 ({rms_p38:.2f}° RMS)")
                # Neu festgelegt mit Gegenprobe 65 (0.63 -> 0.87 dB): bis
                # dahin lag die Freifeld-Strahlungslast auch im Druckgang.
                # Sie wirkte hier als Ersatz für die Last des Aktuators und
                # verdeckte den Hochtonüberschuss des 2D-Modells (wie gegen
                # die FEM, Gegenprobe 59). 3D trifft dieselbe Messung ohne
                # sie etwas besser (0.744 -> 0.734 dB).
                stand.sperrklinke("rms_2d_4146_db", rms_a38, "dB",
                                  "2D gegen Fig. 7, Amplitude",
                                  toleranz=0.05)
            else:
                # 4134: der Prüfling war stärker gedämpft als der exakte
                # Film (Gegenproben 58, 59) — 2D muss dort liegen, wo 3D
                # liegt: im Hochton über der Messung, im Mittel gleich
                c3_38 = MicrophoneCapsule(**_messpruefling(
                    {**par38, "squeeze_model": "3d"}))
                H3_38 = (c3_38.transfer_function(fa38)
                         / c3_38.transfer_function(np.array([250.0]))[0])
                am3_38 = 20.0 * np.log10(np.abs(H3_38)) - np.asarray(a38)
                hoch = fa38 >= 13e3
                assert np.all(am38[hoch] > 0.0) and np.all(am3_38[hoch] > 0.0), \
                    (f"4134: 2D und 3D müssen im Hochton über der Messung "
                     f"liegen (2D {np.round(am38[hoch], 2)}, 3D "
                     f"{np.round(am3_38[hoch], 2)} dB)")
                # (Abstand 2D–3D unter der früheren Messschranke)
                d23 = float(np.sqrt(np.mean((am38 - am3_38) ** 2)))
                assert d23 < lim38[0], \
                    (f"4134: 2D muss gegen die Messung wie 3D liegen "
                     f"({d23:.2f} dB RMS Abstand)")
                stand.wert("rms_2d_4134_db", rms_a38, "dB",
                           "2D gegen Fig. 6, Amplitude (Prüfling gedämpfter)")
                stand.wert("rms_2d_4134_grad", rms_p38, "°",
                           "2D gegen Fig. 6, Phase")
            res38[nm38] = (rms_a38, rms_p38, eC38, eR38)
        print(f"Externe Messreferenz (Zuckerwar 1978, B&K 4134/4146): "
              f"M und C_M analytisch getroffen (<0.2 %); 4134 "
              f"{res38['4134'][0]:.2f} dB / {res38['4134'][1]:.2f}° RMS "
              f"gegen Fig. 6, C_A {100 * res38['4134'][2]:+.1f} %, "
              f"R {100 * res38['4134'][3]:+.1f} %; 4146 "
              f"{res38['4146'][0]:.2f} dB / {res38['4146'][1]:.2f}° RMS "
              f"gegen Fig. 7, C_A {100 * res38['4146'][2]:+.1f} %, "
              f"R {100 * res38['4146'][3]:+.1f} % gegen Tab. II (Zuckerwars "
              f"Näherung); 4134 liegt wie 3D über der Messung  OK")


@pytest.mark.feld3d
def test_gp58_daempfung_am_lochkreis_befund(stand):
    """Gegenprobe 58: Dämpfung am Lochkreis — was dem 3D-Modell fehlt."""
    # Offener Punkt (README, „Offene Punkte" 1): an der B&K 4134 liegt der
    # 3D-Löser bei 13–20 kHz 2–3.5 dB über Zuckerwars Messung; das 2D-
    # Modell trifft sie, weil es den Filmwiderstand am Lochkreis 1.7-fach
    # überschätzt (Gegenprobe 52). Die Recherche grenzt ein, WAS fehlt und
    # ob es allgemeine Physik ist. Die Referenz sind Zuckerwars Messungen
    # (Fig. 6/7, Amplitude UND Phase) an beiden Kapseln derselben Arbeit.
    # a) PHASE: schon weit unter der Resonanz (2–10 kHz, 4134-Resonanz
    #    ~23 kHz) fehlt dem 3D-Modell Nacheilung, etwa ein fester Anteil
    #    der gemessenen. Dort wirken Masse und Steife nicht auf die Phase,
    #    nur Widerstand: es fehlt WIDERSTAND, keine Resonanzverschiebung.
    # b) EIN frequenzunabhängiger Serienwiderstand vor der Membran
    #    (5000 Rayl = 8.1e7 Pa·s/m³, rund 2/3 des exakten Filmwiderstands)
    #    bringt am 4134 Amplitude UND Phase zugleich zur Messung; ebenso
    #    ein fast dichter Randschlitz (wirksam 15 µm statt 0.838 mm). Beide
    #    Deutungen scheitern am 4146 derselben Arbeit: dort verschlechtert
    #    jede zusätzliche Dämpfung die Übereinstimmung — der unveränderte
    #    3D-Löser trifft ihn besser als 2D (Gegenprobe 38).
    # c) Die naheliegenden Geometrie-Deutungen passen schon am 4134
    #    schlechter als der Serienwiderstand: der Spalt, mit dem das
    #    COMSOL-Anwendungsmodell der 4134 auf B&Ks Originalgeometrie rechnet
    #    („around 19 µm" statt 20.77 µm in Zuckerwars Tab. I; R ∝ h⁻³),
    #    halbiert die Amplitudenabweichung, lässt aber das Phasendefizit an
    #    der Resonanz stehen; eine geschlossene Ringnut statt des Schlitzes
    #    trifft die Phase noch schlechter.
    #    Der Aktuator scheidet nach B&K aus (Microphone Handbook BE 1447:
    #    perforierte Platte 0.4–0.8 mm vor der Membran; für 1/2"-Kapseln
    #    sind keine Korrekturen Aktuator -> Druck nötig).
    #    Homentcovschi & Miles (JASA 130, 3698, 2011) rechnen dieselbe
    #    Geometrie mit den Stokes-Gleichungen: ihre Phase liegt bis 10 kHz
    #    fast auf diesem 3D-Modell (28° gegen 26°, gemessen 32–38°), ihre
    #    Amplitude ist nach eigener Aussage überdämpft (−7.5 dB bei
    #    15 kHz) — die Messung liegt zwischen beiden Rechnungen.
    # d) Gegen die volle thermoviskose FEM (Gegenprobe 32, Navier–Stokes
    #    statt Reynolds, vier Löcher auf einem Kreis) liegt die 3D-
    #    Überhöhung nur 0.2–0.3 dB über der FEM (Stand-Wert dort).
    # e) B&Ks ORIGINALGEOMETRIE (COMSOL-Anwendungsmodell, Geometrie
    #    „courtesy of Brüel and Kjær", aus der mphtxt-Datei abgelesen):
    #    Spalt 18.6 µm, sechs Löcher r = 0.5 mm auf dem Kreis r = 1.70 mm
    #    (Tab. I: 2.03 mm), Platte r = 3.6 mm, am Rand 0.3 mm dick mit
    #    kegeliger Unterseite (1.03 mm an der Lochmitte), Membran r =
    #    4.5 mm; der Ring zwischen Plattenrand und Einspannung ist
    #    0.9 mm breit offen zur kegelförmigen Rückkammer (131 mm³) — es
    #    gibt KEINEN gedrosselten Randweg. Mit dieser Geometrie bleibt das
    #    3D-Modell unterdämpft wie mit Tab. I: der kleinere Spalt dämpft
    #    mehr, der weiter innen liegende Lochkreis weniger.
    # Folgerung: dem 3D-Film fehlt keine allgemeine Physik; die Abweichung
    # ist 4134-spezifisch. Gegenprobe 59 (COMSOL-FEM derselben
    # B&K-Geometrie) bestätigt das: 3D trifft die FEM auf 0.1 dB, und
    # Zuckerwars Prüfling war stärker gedämpft als heutige 4134.
    # NACHTRAG (Gegenprobe 60): das 2D-Modell überschätzt den Film am
    # Lochkreis nicht mehr (Makroelement) und liegt jetzt wie 3D über
    # Zuckerwars 4134-Messung (Gegenprobe 38).
    if not _HAS_SCIPY:
        return

    def _fig(nm, **kw):
        par, _, f, a, p, _ = _BK38[nm]
        # gemessener Prüfling: Vakuumresonanz vorgegeben (Gegenprobe 62)
        c = MicrophoneCapsule(**_messpruefling(
            {**par, "squeeze_model": "3d", **kw}))
        fa = np.asarray(f, dtype=float)
        H = c.transfer_function(fa) / c.transfer_function(np.array([250.0]))[0]
        da = 20.0 * np.log10(np.abs(H)) - np.asarray(a)
        dp = -np.rad2deg(np.unwrap(np.angle(H))) - np.asarray(p)
        return (fa, da, dp, float(np.sqrt(np.mean(da**2))),
                float(np.sqrt(np.mean(dp**2))))

    # a) Phasendefizit unter der Resonanz
    f34, da34, dp34, ra34, rp34 = _fig("4134")
    tief = (f34 >= 2000.0) & (f34 <= 10000.0)
    p34 = np.asarray(_BK38["4134"][4])
    assert np.all(dp34[tief] < 0.0), \
        (f"4134: dem 3D-Modell muss unter der Resonanz Phase fehlen "
         f"({np.round(dp34[tief], 1)}°)")
    anteil58 = float(np.mean((p34[tief] + dp34[tief]) / p34[tief]))
    stand.wert("phase_3d_anteil_4134", anteil58, "",
               "3D-Nacheilung / gemessene, Mittel 2–10 kHz")
    stand.wert("rms_3d_4134_db", ra34, "dB", "3D gegen Fig. 6, Amplitude")
    stand.wert("rms_3d_4134_grad", rp34, "°", "3D gegen Fig. 6, Phase")

    # b) Serienwiderstand bzw. gedrosselter Schlitz: am 4134 besser ...
    _, _, _, raR, rpR = _fig("4134", fabric_front_rayl=5000.0)
    _, _, _, raS, rpS = _fig("4134", ring_vent_width=15e-6)
    assert raR < ra34 and rpR < rp34, \
        (f"4134: ein Serienwiderstand muss Amplitude UND Phase verbessern "
         f"({ra34:.2f} -> {raR:.2f} dB, {rp34:.2f} -> {rpR:.2f}°)")
    assert raS < ra34 and rpS < rp34, \
        (f"4134: ein gedrosselter Schlitz muss beides verbessern "
         f"({ra34:.2f} -> {raS:.2f} dB, {rp34:.2f} -> {rpS:.2f}°)")
    stand.wert("rms_4134_mit_serien_r_db", raR, "dB",
               "3D + 5000 Rayl gegen Fig. 6, Amplitude")
    stand.wert("rms_4134_mit_serien_r_grad", rpR, "°",
               "3D + 5000 Rayl gegen Fig. 6, Phase")
    stand.wert("rms_4134_schlitz_15um_db", raS, "dB",
               "3D, Randschlitz 15 µm, gegen Fig. 6")
    # ... am 4146 derselben Arbeit schlechter
    _, _, _, ra46, rp46 = _fig("4146")
    _, _, _, ra46R, _ = _fig("4146", fabric_front_rayl=1000.0)
    _, _, _, ra46S, _ = _fig("4146", ring_vent_width=15e-6)
    assert ra46R > ra46 and ra46S > ra46, \
        (f"4146: zusätzliche Dämpfung muss die Übereinstimmung "
         f"verschlechtern ({ra46:.2f} dB; +1000 Rayl {ra46R:.2f}, "
         f"Schlitz 15 µm {ra46S:.2f})")
    stand.wert("rms_3d_4146_db", ra46, "dB", "3D gegen Fig. 7, Amplitude")
    stand.wert("rms_3d_4146_grad", rp46, "°", "3D gegen Fig. 7, Phase")
    stand.wert("rms_4146_mit_serien_r_db", ra46R, "dB",
               "3D + 1000 Rayl gegen Fig. 7, Amplitude")

    # c) Geometrie-Deutungen am 4134: Phase schlechter als mit R
    _, _, _, raH, rpH = _fig("4134", air_gap=19e-6)
    _, _, _, raN, rpN = _fig("4134", ring_vent_width=0.0,
                             ring_vent_length=None,
                             backplate_diameter=2 * 4.445e-3,
                             clearance_ring_diameter=2 * 4.026e-3,
                             clearance_ring_width=0.838e-3,
                             clearance_ring_depth=0.305e-3)
    assert raH < ra34, \
        (f"4134: der kleinere COMSOL-Spalt muss die Amplitude besser "
         f"treffen ({ra34:.2f} -> {raH:.2f} dB)")
    assert rpH > rpR and rpN > rpR, \
        (f"4134: Spalt 19 µm ({rpH:.2f}°) und geschlossene Ringnut "
         f"({rpN:.2f}°) müssen die Phase schlechter treffen als der "
         f"Serienwiderstand ({rpR:.2f}°)")
    stand.wert("rms_4134_spalt_19um_db", raH, "dB",
               "3D, Spalt 19 µm (COMSOL), gegen Fig. 6, Amplitude")
    stand.wert("rms_4134_spalt_19um_grad", rpH, "°",
               "3D, Spalt 19 µm (COMSOL), gegen Fig. 6, Phase")
    stand.wert("rms_4134_ringnut_grad", rpN, "°",
               "3D, geschlossene Ringnut, gegen Fig. 6, Phase")

    # e) B&K-Originalgeometrie: das 3D-Modell bleibt im Hochton über der
    #    Messung (Richtung), die Beträge sind Stand-Werte
    a_bp58 = 3.6e-3
    bk58 = dict(membrane_diameter=9.0e-3, air_gap=18.6e-6,
                backplate_diameter=2 * a_bp58, backplate_thickness=1.029e-3,
                through_hole_diameter=1.0e-3, through_hole_pcd=2 * 1.70e-3,
                ring_vent_width=0.86e-3, ring_vent_length=0.30e-3,
                cavity_length=131e-9 / (np.pi * a_bp58**2))
    fB, daB, dpB, raB, rpB = _fig("4134", **bk58)
    assert np.all(daB[fB >= 13000.0] > 0.0), \
        (f"4134 mit B&K-Geometrie: das 3D-Modell muss im Hochton über der "
         f"Messung bleiben ({np.round(daB[fB >= 13000.0], 2)} dB)")
    stand.wert("rms_4134_bk_geometrie_db", raB, "dB",
               "3D, B&K-Originalgeometrie (COMSOL), gegen Fig. 6")
    stand.wert("rms_4134_bk_geometrie_grad", rpB, "°",
               "3D, B&K-Originalgeometrie (COMSOL), gegen Fig. 6, Phase")
    print(f"Dämpfung am Lochkreis: 4134 3D {ra34:.2f} dB / {rp34:.1f}° RMS "
          f"gegen Fig. 6, Nacheilung unter der Resonanz nur "
          f"{100 * anteil58:.0f} % der gemessenen (fehlender Widerstand); "
          f"+8.1e7 Pa·s/m³ seriell {raR:.2f} dB / {rpR:.1f}°, Schlitz 15 µm "
          f"{raS:.2f} dB / {rpS:.1f}° — am 4146 beides schlechter "
          f"({ra46:.2f} -> {ra46R:.2f} / {ra46S:.2f} dB); Spalt 19 µm "
          f"(COMSOL) {raH:.2f} dB / {rpH:.1f}°, Ringnut {rpN:.1f}° Phase; "
          f"B&K-Originalgeometrie {raB:.2f} dB / {rpB:.1f}° — "
          f"4134-spezifisch, keine allgemeine Filmphysik  OK")


# B&K-4134-Originalgeometrie, wie im COMSOL-Anwendungsmodell (Geometrie
# „courtesy of Brüel and Kjær", aus dem mphtxt-Export abgelesen; s. README,
# „Dämpfung am Lochkreis"). 200 V wie im Modell, Rückseite geschlossen.
_BK4134_COMSOL = dict(
    membrane_material={"rho": 8900.0, "E": 221e9, "nu": 0.31},
    membrane_resonance_hz=None, membrane_diameter=9.0e-3,
    membrane_thickness=5.0e-6, membrane_tension=3160.0, air_gap=18.6e-6,
    backplate_diameter=7.2e-3, backplate_thickness=1.029e-3,
    bias_voltage=200.0, architecture="single", n_through_holes=6,
    through_hole_diameter=1.0e-3, through_hole_pcd=3.4e-3, n_blind_holes=0,
    ring_vent_width=0.86e-3, ring_vent_length=0.30e-3,
    rear_network_enabled=True, delay_length=0.0,
    cavity_length=131e-9 / (np.pi * 3.6e-3**2), n_cavity_holes=0,
    fabric_front_rayl=0.0, fabric_rear_rayl=0.0, include_diffraction=False)
_EXTERN = __import__("pathlib").Path(__file__).with_name("extern")


@pytest.mark.feld3d
def test_gp59_comsol_referenz_b_k_4134(stand):
    """Gegenprobe 59: COMSOL-Referenz B&K 4134 (Originalgeometrie)."""
    # Die entscheidende Probe zu Gegenprobe 58: dieselbe Geometrie, einmal
    # mit der vollen thermoviskosen FEM (COMSOL-Anwendungsmodell
    # bk_4134_microphone, Navier–Stokes und Wärmeleitung im ganzen
    # Luftraum, Membran, Elektrostatik mit Randfeld) und einmal mit Capsim.
    # Dazu die drei B&K-Messkurven desselben Modells (Mittel, untere,
    # obere; heutige 4134, 200 V).
    # a) Das 3D-Modell trifft die FEM über 1–20 kHz auf 0.1 dB RMS — dem
    #    Reynolds-Film mit diskreten Löchern fehlt KEINE Physik. Das
    #    2D-Modell lag mit dem Gaußband und der Škvor-Zelle 1.8 dB RMS
    #    darunter (überdämpft): diese Lochkreis-Darstellung überschätzte
    #    den Filmwiderstand um 60 %. Mit dem Makroelement (Gegenprobe 60)
    #    trifft es die FEM auf 0.26 dB RMS und liegt im Hochton höchstens
    #    0.5 dB darüber; das ist eine Sperrklinke.
    #    NACHTRAG (Gegenprobe 65): die FEM hat keine Strahlungslast, Capsim
    #    legte sie bis dahin auch im Druckgang an. Ohne sie trifft 3D die
    #    FEM besser (0.098 -> 0.066 dB RMS), 2D liegt im Hochton bis
    #    0.68 dB darüber — die Last hatte diesen Überschuss teilweise
    #    verdeckt. Die Grenze für 2D ist deshalb 0.7 statt 0.6 dB
    #    (bewusst gelockert, nicht um den Stand zu halten: der frühere
    #    Wert enthielt einen Fehler, der in die günstige Richtung wirkte).
    # b) Gegen die Messungen liegt 3D bis 12.6 kHz im Streuband; darüber
    #    liegen FEM und 3D gleichermaßen etwas über der Messung.
    # c) Äquivalenter akustischer Widerstand Re(p_in/Q_Membran), ohne
    #    Strahlungslast (die FEM hat keine; seit Gegenprobe 65 auch der
    #    Druckgang nicht mehr, abgezogen wird die tatsächlich anliegende
    #    Last _front_radiation): 2D lag mit dem Gaußband 61 %
    #    über der FEM, mit dem Makroelement 6 % darunter. Für 3D stand
    #    hier „12 % darüber" — gemessen mit der Volumenverschiebung nur
    #    über der ELEKTRODE (a_bp = 0.8·a_mem), während 2D und FEM die der
    #    ganzen Membran meinen. Mit derselben Größe (weight='membrane',
    #    Gegenprobe 68) liegt 3D auf 1 % an der FEM; beide Modelle müssen
    #    auf 10 % treffen.
    # Folgerung für Gegenprobe 38/58: Zuckerwars Prüfling von 1978 war
    # deutlich stärker gedämpft als heutige 4134 (20 kHz: −3.1 gegen
    # −1.2 dB); dass das alte 2D-Modell ihn traf, war das Zusammentreffen
    # dieser Abweichung mit der Überschätzung am Lochkreis.
    # Die COMSOL-Daten stehen unter COMSOLs Lizenz und liegen nicht im
    # Repo: Export aus dem Anwendungsmodell (Empfindlichkeit mit offener
    # Belüftung und den Messkurven, Empfindlichkeit mit abgeschirmter
    # Belüftung, „Equivalent Acoustic Resistance") nach tests/extern/
    # (README).
    if not _HAS_SCIPY:
        return
    d_s = _EXTERN / "comsol_4134_sens.txt"         # Modell (vent exposed)
    d_u = _EXTERN / "comsol_4134_unexposed.txt"    # + drei Messkurven
    d_r = _EXTERN / "comsol_4134_resis.txt"
    if not (d_s.is_file() and d_u.is_file() and d_r.is_file()):
        pytest.skip("COMSOL-Referenzdaten fehlen (tests/extern/, s. README)")

    def _lies(p):
        return np.array([z.split() for z in p.read_text().splitlines()
                         if z.strip() and not z.startswith("%")], float)
    sens = _lies(d_s).reshape(4, -1, 2)
    f = sens[0, :, 0]
    fem_offen, mittel, unten, oben = sens[:, :, 1]
    unexp = _lies(d_u)
    res = _lies(d_r)
    assert np.allclose(res[:, 0], f) and np.allclose(unexp[:, 0], f), \
        "Frequenzraster der Dateien"
    # Referenz ist die FEM mit abgeschirmter Belüftung (vent unexposed):
    # Capsim rechnet die Rückseite geschlossen. Die Belüftung wirkt nur im
    # tiefsten Bass; ab 1 kHz müssen beide Fälle zusammenfallen.
    fem = unexp[:, 1]
    sel = f >= 1000.0
    assert np.max(np.abs(fem[sel] - fem_offen[sel])) < 0.01, \
        "FEM: Belüftung offen/abgeschirmt muss ab 1 kHz gleich sein"
    fs = f[sel]
    om = 2.0 * np.pi * fs
    f_norm = np.array([f[np.argmin(np.abs(f - 250.0))]])  # wie COMSOL: 251 Hz
    # Gegen die FEM rechnet Capsim dieselbe Physik: COMSOLs Membran-
    # interface hat keine Biegesteifigkeit, also ohne Randschicht der
    # Folie. Gegen die Messungen der physikalische Prüfling (Vakuum-
    # resonanz vorgegeben, mit Randschicht; s. _messpruefling).
    pegel, pegel_m, re_z = {}, {}, {}
    for sm in ("2d", "3d"):
        MicrophoneCapsule._RANDSCHICHT = False
        try:
            c = MicrophoneCapsule(squeeze_model=sm, **_BK4134_COMSOL)
        finally:
            MicrophoneCapsule._RANDSCHICHT = True
        H = c.transfer_function(fs)
        pegel[sm] = 20.0 * np.log10(np.abs(H / c.transfer_function(f_norm)[0]))
        V = (c._solve_3d(om, weight="membrane")[0] if sm == "3d"
             else H / c._theta)
        re_z[sm] = np.real(1.0 / (1j * om * V)
                           - c._front_radiation(om))
        cm = MicrophoneCapsule(squeeze_model=sm,
                               **_messpruefling(_BK4134_COMSOL))
        pegel_m[sm] = 20.0 * np.log10(np.abs(
            cm.transfer_function(fs) / cm.transfer_function(f_norm)[0]))
    rms = {sm: float(np.sqrt(np.mean((pegel[sm] - fem[sel]) ** 2)))
           for sm in pegel}
    rms_m = {sm: float(np.sqrt(np.mean((pegel_m[sm] - mittel[sel]) ** 2)))
             for sm in pegel}
    # a) FEM derselben Geometrie
    assert rms["3d"] < rms["2d"], \
        (f"3D muss die FEM besser treffen als 2D ({rms['3d']:.2f} gegen "
         f"{rms['2d']:.2f} dB RMS)")
    d2 = float(np.max(np.abs(pegel["2d"] - fem[sel])))
    assert d2 < 0.7, \
        (f"2D muss die FEM mit dem Makroelement über 1–20 kHz auf 0.7 dB "
         f"treffen (größte Abweichung {d2:.2f} dB)")
    stand.sperrklinke("rms_3d_gegen_fem", rms["3d"], "dB",
                      "3D gegen COMSOL-FEM, 1–20 kHz", toleranz=0.05)
    stand.sperrklinke("rms_2d_gegen_fem_makro", rms["2d"], "dB",
                      "2D (Makroelement) gegen COMSOL-FEM, 1–20 kHz",
                      toleranz=0.05)
    # b) Messungen: 3D bis 12.6 kHz im Streuband der drei Kurven
    band = fs <= 12.6e3
    lo = np.minimum(unten, oben)[sel]
    hi = np.maximum(unten, oben)[sel]
    lo = np.minimum(lo, mittel[sel])
    hi = np.maximum(hi, mittel[sel])
    ausser = pegel_m["3d"][band] - np.clip(pegel_m["3d"][band],
                                            lo[band] - 0.02, hi[band] + 0.02)
    assert np.all(ausser == 0.0), \
        (f"3D muss bis 12.6 kHz im Streuband der B&K-Messungen liegen "
         f"(außerhalb um {np.round(ausser, 2)} dB)")
    stand.wert("rms_3d_gegen_messung", rms_m["3d"], "dB",
               "3D gegen B&K-Messmittel, 1–20 kHz")
    stand.wert("rms_2d_gegen_messung", rms_m["2d"], "dB",
               "2D gegen B&K-Messmittel, 1–20 kHz")
    # c) Widerstand
    q3 = float(np.mean(re_z["3d"] / res[sel, 1]))
    q2 = float(np.mean(re_z["2d"] / res[sel, 1]))
    assert abs(np.log(q2)) < np.log(1.1), \
        (f"2D-Widerstand muss mit dem Makroelement auf 10 % an der FEM "
         f"liegen (Re Z 2D/FEM {q2:.2f})")
    assert abs(np.log(q3)) < np.log(1.1), \
        (f"3D-Widerstand muss auf 10 % an der FEM liegen (Re Z 3D/FEM "
         f"{q3:.2f})")
    stand.wert("widerstand_3d_zu_fem", q3, "", "Re Z 3D / FEM, Mittel 1–20 kHz")
    stand.wert("widerstand_2d_zu_fem", q2, "", "Re Z 2D / FEM, Mittel 1–20 kHz")
    i20 = int(np.argmax(fs))
    print(f"COMSOL-Referenz B&K 4134 (Originalgeometrie, 200 V): 3D "
          f"{rms['3d']:.2f} dB RMS gegen die FEM, 2D {rms['2d']:.2f} dB "
          f"(Makroelement, höchstens {d2:.2f} dB); 20 kHz: FEM "
          f"{fem[sel][i20]:+.2f}, 3D "
          f"{pegel['3d'][i20]:+.2f}, 2D {pegel['2d'][i20]:+.2f}, Messmittel "
          f"{mittel[sel][i20]:+.2f} dB; 3D bis 12.6 kHz im Streuband der "
          f"Messungen ({rms_m['3d']:.2f} dB RMS gegen das Mittel); Re Z "
          f"3D/FEM {q3:.2f}, 2D/FEM {q2:.2f}  OK")
