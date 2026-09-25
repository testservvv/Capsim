"""Gegenproben: Externe Referenzen (FEM, Messung)."""
import numpy as np
import pytest
import warnings

from basis import *  # noqa: F401,F403


@pytest.mark.slow
@pytest.mark.feld3d
def test_gp32_externe_referenz_fem_veroffentlicht():
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
    #    und wird auf < 1 dB getroffen. Das ist die Kernaussage.
    # c) RESONANZLAGE: bekannter Restfehler, hier als OBERGRENZE
    #    festgeschrieben, damit er nur besser werden kann. Das Modell liegt
    #    13 % zu tief, weil der Lochzweig zu viel akustische Masse trägt
    #    (gemessen an der Antiresonanz Lochmasse/Spaltnachgiebigkeit:
    #    3203 Hz gegen 3500 Hz in der FEM, also Faktor 1.19 in der Masse).
    #    DIE URSACHE IST INZWISCHEN GEKLÄRT, und es ist KEIN
    #    Massenüberschuss. Der Reihe nach ausgeschlossen:
    #      * Zell-Engstelle: ihr Reaktivanteil ist exakt die kinetische
    #        Energie der Schmierfilmströmung, und sowohl die wirbelfreie
    #        Lösung (+5…16 %) als auch die Stokes-Zelle (+20…56 %)
    #        liefern MEHR, nicht weniger (Gegenprobe 36). Auch die
    #        Literatur zeigt dorthin: Reynolds UNTERschätzt die Dämpfung.
    #      * Mündungsmasse: selbst ihre völlige Streichung hebt die Kerbe
    #        nur um 5.9 % — nötig wären 9.2 %, und negativ kann sie nicht
    #        sein. Struktur geprüft: 1D und 2D setzen sie EINMAL an,
    #        portseitig mit Fok-Faktor; die Filmseite deckt der Zellterm.
    #      * Membranmasse und Rückkammervolumen: Empfindlichkeit der
    #        Kerbe exakt NULL. Sie ist die reine Loch-Spalt-Antiresonanz.
    #      * Spaltnachgiebigkeit: n_p = 1.30 (zwischen isotherm und
    #        adiabat) erklärt höchstens 4 %.
    #    Einziger starker Hebel ist der Lochradius (d ln f/d ln r =
    #    +0.61) — ein direkt tabellierter Wert.
    #
    #    DER VERGLEICH SELBST WAR SCHIEF. Die FEM zeigt in diesem Band
    #    ein DUBLETT (Minima 3500 und 4200 Hz, Maximum dazwischen bei
    #    3860 Hz); der 2D-Pfad kann nur EINE Kerbe haben, weil er die
    #    vier Bohrungen homogenisiert. Verglichen wurde also eine
    #    Einzelkerbe mit der ersten von zweien. Der 3D-Feldlöser, der die
    #    Löcher diskret auflöst, reproduziert das Dublett (3227/4025 Hz,
    #    s. Prüfung unten) und trifft die FEM insgesamt besser
    #    (RMS 3.16 gegen 3.41 dB über 10 Hz…5 kHz).
    #    Gegenprobe 31 hält bereits fest, dass die Homogenisierung bei
    #    12 Bohrungen am Ende ist — bei VIER ist sie weit darüber hinaus.
    #    Die Schranke unten misst deshalb wesentlich die
    #    Homogenisierungsgrenze, nicht einen Modellfehler der Physik.
    #    KORREKTUR (Gegenprobe 48): das gilt für das DUBLETT, nicht für
    #    die RESONANZLAGE. Der korrigierte 3D-Löser (vollständige,
    #    äquipotentiale Mündungen) setzt die Resonanz auf 482 Hz — fast
    #    genau wie 2D (477 Hz), beide 13 % unter der FEM (550 Hz). Ein
    #    Fehler, den das diskret rechnende Modell GENAUSO macht, kann
    #    keine Homogenisierungsgrenze sein; die Ursache der Resonanzlage
    #    ist damit wieder offen. Das Dublett trifft der 3D-Löser jetzt
    #    näher (3421/4127 gegen FEM 3500/4200 Hz mit konturtreuen
    #    Mündungen, Gegenprobe 51; davor 3378/4127, 3336/4042,
    #    ursprünglich 3227/4025).
    #    NACHTRAG (Gegenprobe 51): mit konturtreuen Mündungen ist der
    #    3D-Löser gitterkonvergent und legt die Resonanz auf 495 Hz — 4 %
    #    über 2D, 10 % unter der FEM. Die diskreten Bohrungen erklären
    #    damit rund ein Viertel der Verstimmung; der Rest bleibt offen.
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
            cavity_length=7.6e-3, n_cavity_holes=0, fabric_front_rayl=0.0,
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
        # c) Resonanzlage: dokumentierter Restfehler als Schranke
        det32 = fpk32 / 550.0
        assert 0.82 < det32 < 1.02, \
            (f"Resonanzlage {fpk32:.0f} Hz gegen 550 Hz (FEM) — "
             f"Verstimmung {det32:.3f} außerhalb der dokumentierten "
             f"Schranke (offener Restfehler, s. Kommentar)")
        # ... und die Homogenisierung erklärt davon nur einen kleinen Teil:
        # der diskret rechnende 3D-Löser (konturtreue Mündungen, auf grobem
        # und feinem Gitter 495 Hz; ohne Konturkorrektur wanderte er mit
        # dem Gitter, 482…488 Hz) liegt 4 % über 2D, aber 10 % unter der
        # FEM — rund ein Viertel des Abstands kommt von den diskreten
        # Bohrungen, der Rest bleibt offen (Korrektur Gegenprobe 48/51)
        c32_3d = MicrophoneCapsule(**{**dict(
            membrane_material={"rho": 1944.0, "E": 4.0e9, "nu": 0.35},
            membrane_resonance_hz=1040.0, membrane_diameter=36.0e-3,
            membrane_thickness=25e-6, membrane_tension=116.27,
            air_gap=230e-6, backplate_diameter=36.0e-3,
            backplate_thickness=1.6e-3, bias_voltage=1.0,
            architecture="single", n_through_holes=4,
            through_hole_diameter=1.0e-3, through_hole_pcd=2 * 8.4853e-3,
            n_blind_holes=0, rear_network_enabled=True,
            cavity_length=7.6e-3, n_cavity_holes=0, fabric_front_rayl=0.0,
            fabric_rear_rayl=0.0, include_diffraction=False),
            "squeeze_model": "3d"})
        fs32_3 = np.linspace(400.0, 600.0, 81)
        fpk32_3 = float(fs32_3[int(np.argmax(np.abs(
            c32_3d.transfer_function(fs32_3))))])
        assert fpk32 < fpk32_3 < 550.0 and fpk32_3 / fpk32 - 1.0 < 0.08 \
                and fpk32_3 / 550.0 < 0.95, \
            (f"3D ({fpk32_3:.0f} Hz) muss zwischen 2D ({fpk32:.0f} Hz) und "
             f"FEM (550 Hz) liegen, nahe bei 2D — die Homogenisierung "
             f"erklärt nur einen kleinen Teil der Verstimmung")
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
              f"({100 * (det32 - 1):+.0f} %, 3D {fpk32_3:.0f} Hz — die "
              f"Homogenisierung erklärt nur ein Viertel, offener "
              f"Restfehler)  OK")


def test_gp38_externe_referenz_b_k_4134():
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
    # unten sind entsprechend gesetzt und nicht enger.
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
    if _HAS_SCIPY:
        _NI38 = {"rho": 8900.0, "E": 200.0e9, "nu": 0.31}   # Tab. I: Nickel
        bk38 = {
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
                (0.60, 2.5, 0.20, 0.25)),          # Schranken: dB/Grad/C_A/R
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
                (1.60, 10.0, 0.10, 0.45)),
        }
        res38 = {}
        for nm38, (par38, tab38, f38, a38, p38, lim38) in bk38.items():
            c38 = MicrophoneCapsule(**par38)
            # a) Ersatzelemente der Membran: analytisch, müssen exakt sein
            #    Zuckerwar rechnet mit dem Rayleigh-Wert 4/3; die Kette
            #    seit Gegenprobe 55 mit 8/j01² (3.75 % schwerer, damit die
            #    Resonanz der Membran exakt ist). Geprüft wird ρt/S.
            M38 = c38.M_A_mem * c38._mass_factor_rayleigh / c38._piston_factor
            assert abs(M38 / tab38["M"] - 1.0) < 5e-3, \
                (f"{nm38}: (4/3)ρt/S muss Tab. II treffen "
                 f"({M38:.1f} gegen {tab38['M']:.0f})")
            assert abs(c38.C_A_mem / tab38["C_M"] - 1.0) < 5e-3, \
                (f"{nm38}: C_A der Membran muss S²/(8πT) sein "
                 f"({c38.C_A_mem:.4e} gegen {tab38['C_M']:.4e})")
            # b) Luftzweig bei 250 Hz gegen Tabelle II
            w38 = np.array([2.0 * np.pi * 250.0])
            Zr38 = c38._membrane_port_impedance(w38)[3][0]
            CA38 = -1.0 / (w38[0] * Zr38.imag)
            eC38 = CA38 / tab38["C_A"] - 1.0
            eR38 = Zr38.real / tab38["R"] - 1.0
            assert abs(eC38) < lim38[2], \
                (f"{nm38}: Luftnachgiebigkeit gegen Tab. II "
                 f"({100 * eC38:+.1f} %)")
            assert abs(eR38) < lim38[3], \
                (f"{nm38}: Spaltwiderstand gegen Tab. II "
                 f"({100 * eR38:+.1f} % — dokumentierte Schranke, "
                 f"s. Streuung der Zellregel im Kommentar)")
            # c) Frequenzgang gegen Fig. 6/7 (Amplitude UND Phase)
            fa38 = np.asarray(f38, dtype=float)
            Hn38 = (c38.transfer_function(fa38)
                    / c38.transfer_function(np.array([250.0]))[0])
            am38 = 20.0 * np.log10(np.abs(Hn38)) - np.asarray(a38)
            ph38 = (-np.rad2deg(np.unwrap(np.angle(Hn38)))
                    - np.asarray(p38))
            rms_a38 = float(np.sqrt(np.mean(am38**2)))
            rms_p38 = float(np.sqrt(np.mean(ph38**2)))
            assert rms_a38 < lim38[0], \
                (f"{nm38}: Amplitude gegen Fig. 6/7 "
                 f"({rms_a38:.2f} dB RMS)")
            assert rms_p38 < lim38[1], \
                (f"{nm38}: Phase gegen Fig. 6/7 ({rms_p38:.2f}° RMS)")
            res38[nm38] = (rms_a38, rms_p38, eC38, eR38)
        print(f"Externe Messreferenz (Zuckerwar 1978, B&K 4134/4146): "
              f"M und C_M analytisch getroffen (<0.2 %); 4134 "
              f"{res38['4134'][0]:.2f} dB / {res38['4134'][1]:.2f}° RMS "
              f"gegen Fig. 6, C_A {100 * res38['4134'][2]:+.1f} %, "
              f"R {100 * res38['4134'][3]:+.1f} %; 4146 "
              f"{res38['4146'][0]:.2f} dB / {res38['4146'][1]:.2f}° RMS "
              f"gegen Fig. 7, C_A {100 * res38['4146'][2]:+.1f} %, "
              f"R {100 * res38['4146'][3]:+.1f} % (Streuung der Škvor-"
              f"Zellregel, dokumentiert)  OK")
