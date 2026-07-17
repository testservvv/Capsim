# -*- coding: utf-8 -*-
"""
Capsim — Übersetzungstabelle der GUI (Englisch/Deutsch)
=======================================================

Alle sichtbaren Texte der Streamlit-Oberfläche liegen hier als
``{schlüssel: {"en": ..., "de": ...}}``. Platzhalter werden per
``str.format(**kwargs)`` gefüllt.

WICHTIG — Projektkompatibilität: Die kanonischen WERTE der Auswahl-
Widgets (Architektur, Material, Lochposition, axialer Körper) bleiben
die deutschen Bestandsschlüssel aus app.py. Projektdateien speichern
IMMER den kanonischen Wert; die Anzeige übersetzt ``format_func`` über
``LABEL_TR``. Dadurch sind Projekte zwischen beiden Sprachen (und mit
allen älteren Versionen) voll austauschbar.
"""

TR = {
    # ------------------------- Seite / Kopf -----------------------------
    "page_title": {
        "en": "Capsim — Capsule Simulation",
        "de": "Capsim — Kapselsimulation",
    },
    "app_title": {
        "en": "Condenser Microphone Capsule — Simulation",
        "de": "Kondensatormikrofonkapsel — Simulation",
    },
    "app_caption": {
        "en": "Electroacoustic equivalent circuit (lumped-element model) · "
              "10 Hz – 25 kHz",
        "de": "Elektroakustisches Ersatzschaltbild (Lumped-Element-Modell) · "
              "10 Hz – 25 kHz",
    },
    "sidebar_caption": {
        "en": "Lumped-element simulation of a condenser microphone capsule",
        "de": "Lumped-Element-Simulation einer Kondensatormikrofonkapsel",
    },
    # In beiden Sprachen IDENTISCH: das Label geht in die Widget-Identität
    # ein; ein sprachabhängiges Label würde den Umschalter bei jedem
    # Wechsel als neues Widget erscheinen lassen.
    "lang_label": {
        "en": "🌐 Language / Sprache",
        "de": "🌐 Language / Sprache",
    },

    # ------------------------- Projekt ----------------------------------
    "exp_project": {"en": "📁 Project", "de": "📁 Projekt"},
    "upload_label": {
        "en": "Load project (.json)",
        "de": "Projekt laden (.json)",
    },
    "save_btn": {
        "en": "💾 Save project (.json)",
        "de": "💾 Projekt speichern (.json)",
    },
    "project_filename": {
        "en": "capsim_project.json",
        "de": "capsim_projekt.json",
    },
    "load_ok": {
        "en": "Project loaded — {n} parameters applied.",
        "de": "Projekt geladen — {n} Parameter übernommen.",
    },
    "load_fail": {
        "en": "Could not load project: {exc}",
        "de": "Projekt konnte nicht geladen werden: {exc}",
    },
    "btn_reset": {
        "en": "🗑️ Set all values to zero",
        "de": "🗑️ Alle Werte auf null setzen",
    },
    "help_reset": {
        "en": "Sets every capsule parameter to zero (or to the smallest "
              "permitted value where zero is not allowed) after a "
              "confirmation step — a blank canvas to design from scratch. "
              "Selections (material, architecture, …), model options "
              "(2D/3D, diffraction) and simulation settings are kept.",
        "de": "Setzt nach einem Bestätigungsschritt alle Kapselparameter "
              "auf null (bzw. auf den kleinsten zulässigen Wert, wo null "
              "nicht erlaubt ist) — eine leere Leinwand für einen Entwurf "
              "von Grund auf. Auswahlen (Material, Architektur, …), "
              "Modelloptionen (2D/3D, Beugung) und Simulations-"
              "einstellungen bleiben erhalten.",
    },
    "reset_confirm": {
        "en": "Really set ALL capsule parameters to zero? The current "
              "values are lost — save the project first if needed.",
        "de": "Wirklich ALLE Kapselparameter auf null setzen? Die "
              "aktuellen Werte gehen verloren — bei Bedarf vorher das "
              "Projekt speichern.",
    },
    "btn_reset_yes": {
        "en": "✔️ Yes, set to zero",
        "de": "✔️ Ja, auf null",
    },
    "btn_reset_no": {
        "en": "✖️ Cancel",
        "de": "✖️ Abbrechen",
    },
    "reset_done": {
        "en": "All capsule parameters set to zero.",
        "de": "Alle Kapselparameter auf null gesetzt.",
    },

    # ------------------------- Membran ----------------------------------
    "exp_membrane": {"en": "Membrane", "de": "Membran"},
    "lbl_material": {"en": "Material", "de": "Material"},
    "lbl_use_fres": {
        "en": "Specify resonance frequency",
        "de": "Resonanzfrequenz vorgeben",
    },
    "help_use_fres": {
        "en": "Disabled: the resonance is computed from tension, bending "
              "stiffness and areal density.",
        "de": "Deaktiviert: Resonanz wird aus Vorspannung, "
              "Biegesteifigkeit und Flächendichte berechnet.",
    },
    "lbl_fres": {
        "en": "Resonance frequency [Hz]",
        "de": "Resonanzfrequenz [Hz]",
    },
    "lbl_mem_dia": {"en": "Diameter [mm]", "de": "Durchmesser [mm]"},
    "lbl_mem_thick": {"en": "Thickness [µm]", "de": "Dicke [µm]"},
    "lbl_mem_tension": {"en": "Tension [N/m]", "de": "Vorspannung [N/m]"},

    # ------------------------- Backplate --------------------------------
    "exp_backplate": {"en": "Backplate", "de": "Backplate"},
    "lbl_air_gap": {"en": "Air gap [µm]", "de": "Luftspalt [µm]"},
    "lbl_bp_dia": {"en": "Diameter [mm]", "de": "Durchmesser [mm]"},
    "lbl_bp_thick": {"en": "Thickness [mm]", "de": "Dicke [mm]"},
    "lbl_bias": {
        "en": "Polarization voltage [V]",
        "de": "Polarisationsspannung [V]",
    },
    "lbl_arch": {"en": "Architecture", "de": "Architektur"},
    "help_arch": {
        "en": "K67 design: two membranes on the outside, two inner "
              "backplates separated only by the backplate gap. The passive "
              "rear membrane forms the phase-shifter network (cardioid) — "
              "no delay line or cavity needed.",
        "de": "K67-Bauform: zwei Membranen außen, zwei innen-"
              "liegende Backplates, getrennt nur durch den "
              "Backplate-Spalt. Die passive Rückmembran bildet "
              "das Phasenschiebernetzwerk (Niere) — Laufzeitglied "
              "und Hohlraum entfallen.",
    },
    "lbl_center_gap": {
        "en": "Backplate gap (K67) [µm]",
        "de": "Backplate-Spalt (K67) [µm]",
    },
    "help_center_gap": {
        "en": "Spacer between the two backplate halves of the dual-"
              "diaphragm design. 0 = one-piece, fully drilled-through "
              "center electrode (Braunmühl-Weber); the backplate thickness "
              "is then HALF the plate thickness per side.",
        "de": "Spacer zwischen den beiden Backplate-Hälften "
              "der Doppelmembran-Bauform. 0 = einteilige, "
              "komplett durchbohrte Mittelelektrode "
              "(Braunmühl-Weber); die Backplate-Dicke ist "
              "dann die HALBE Plattendicke je Seite.",
    },
    "hd_holes": {"en": "**Hole pattern**", "de": "**Lochmuster**"},
    "cap_holes": {
        "en": "Each hole type can be spread over several pitch circles "
              "with ➕ (per circle: count + seat Ø; pitch circle Ø 0 = "
              "distributed evenly). The radial layout acts in the 2D field "
              "model and on the electrode porosity; the 1D gap model uses "
              "only the totals.",
        "de": "Jeder Lochtyp lässt sich mit ➕ auf mehrere Lochkreise "
              "verteilen (je Kreis: Anzahl + Sitz-Ø; Lochkreis-Ø 0 = "
              "gleichmäßig verteilt). Die radiale Anordnung wirkt im "
              "2D-Feldmodell und auf die Elektrodenporosität; das "
              "1D-Spaltmodell nutzt nur die Gesamtzahlen.",
    },
    "md_through": {"en": "Through holes", "de": "Durchgangslöcher"},
    "help_through": {
        "en": "The through holes are the only path through the backplate. "
              "0 holes in total = backplate closed → capsule hermetically "
              "sealed (pressure transducer), regardless of the rear side. "
              "The dual architecture needs at least 1 hole.",
        "de": "Die Durchgangslöcher sind der einzige Weg durch "
              "die Backplate. 0 Löcher insgesamt = Backplate "
              "geschlossen → Kapsel hermetisch dicht "
              "(Druckempfänger), unabhängig von der Rückseite. "
              "Bei Dual-Architektur ist mindestens 1 Loch nötig.",
    },
    "lbl_th_dia": {
        "en": "Through holes — Ø [mm]",
        "de": "Durchgangslöcher — Ø [mm]",
    },
    "help_hole_dia": {
        "en": "Bore diameter (applies to all pitch circles of this type).",
        "de": "Bohrungsdurchmesser (gilt für alle "
              "Lochkreise dieses Typs).",
    },
    "lbl_stepped": {
        "en": "Stepped bore (concentric inside blind hole)",
        "de": "Stufenbohrung (konzentrisch im Sackloch)",
    },
    "help_stepped": {
        "en": "K67/K87 construction: each through hole sits at the BOTTOM "
              "of a blind hole with the blind-hole Ø and depth — only the "
              "remaining plate thickness is drilled through narrowly. "
              "Counting ONLY while this switch is on: blind holes = TOTAL "
              "number of counterbores, through holes = how many of them "
              "are additionally drilled through (K67: 120 counterbores, "
              "60 drilled through). With the switch off both hole types "
              "stay independent as before. Requires blind-hole Ø > "
              "through-hole Ø and through ≤ blind count.",
        "de": "K67/K87-Bauweise: jedes Durchgangsloch sitzt am "
              "GRUND eines Sacklochs mit Sackloch-Ø und Sackloch-"
              "Tiefe — nur die Restdicke der Platte ist eng "
              "durchbohrt. Zählweise NUR bei aktivem Schalter: "
              "Blindlöcher = GESAMTZAHL aller Sacklöcher, "
              "Durchgangslöcher = wie viele davon zusätzlich "
              "durchgebohrt sind (K67: 120 Sacklöcher, davon 60 "
              "durchgebohrt). Bei ausgeschaltetem Schalter bleiben "
              "beide Lochtypen unabhängig wie bisher. Erfordert "
              "Sackloch-Ø > Durchgangsloch-Ø und Durchgangs- ≤ "
              "Blindlochzahl.",
    },
    "md_blind": {"en": "Blind holes", "de": "Blindlöcher"},
    "help_blind": {
        "en": "Blind holes on the membrane side: damping and volume "
              "bores, no path through the plate.",
        "de": "Sacklöcher auf der Membranseite: Dämpfungs- und "
              "Volumen-Bohrungen, kein Weg durch die Platte.",
    },
    "lbl_bh_dia": {
        "en": "Blind holes — Ø [mm]",
        "de": "Blindlöcher — Ø [mm]",
    },
    "lbl_bh_depth": {
        "en": "Blind holes — depth [mm]",
        "de": "Blindlöcher — Tiefe [mm]",
    },
    "hd_clr": {
        "en": "**Clearance ring (face relief)**",
        "de": "**Clearance-Ring (Freistich der Stirnflächen)**",
    },
    "help_clr": {
        "en": "Annular relief in the electrode faces (per side): position "
              "via the ring Ø, radial width, axial depth. Wide rings "
              "(≥ 1 grid cell) deepen the gap locally and RELIEVE the "
              "mouth constrictions of bores sitting there — decisive for "
              "cardioid depth with few narrow through holes (e.g. "
              "Debenham). Very narrow rings act as a slot stub. Only "
              "effective in the 2D field model. 0 = no ring.",
        "de": "Ringförmiger Freistich in den Elektroden-Stirn"
              "flächen (je Seite): Position über den Ring-Ø, "
              "radiale Breite, axiale Tiefe. Breite Ringe "
              "(≥ 1 Gitterzelle) vertiefen den Spalt lokal und "
              "ENTLASTEN die Mündungs-Engstellen dort sitzender "
              "Bohrungen — entscheidend für die Nierentiefe bei "
              "wenigen engen Durchgangslöchern (z. B. Debenham). "
              "Sehr schmale Ringe wirken als Schlitz-Stub. Nur "
              "im 2D-Feldmodell wirksam. 0 = kein Ring.",
    },
    "lbl_clr_dia": {
        "en": "Clearance ring — Ø [mm]",
        "de": "Clearance-Ring — Ø [mm]",
    },
    "help_clr_dia": {
        "en": "Mean diameter of the ring (e.g. the pitch circle of the "
              "through holes).",
        "de": "Mittlerer Durchmesser des Rings (z. B. der "
              "Lochkreis der Durchgangslöcher).",
    },
    "lbl_clr_w": {
        "en": "Clearance ring — width [mm]",
        "de": "Clearance-Ring — Breite [mm]",
    },
    "help_clr_w": {
        "en": "Radial width of the relief.",
        "de": "Radiale Breite des Freistichs.",
    },
    "lbl_clr_d": {
        "en": "Clearance ring — depth [mm]",
        "de": "Clearance-Ring — Tiefe [mm]",
    },
    "help_clr_d": {
        "en": "Axial removal (additional gap height in the ring area).",
        "de": "Axialer Abtrag (zusätzliche Spalthöhe im "
              "Ringbereich).",
    },
    "hd_clamp": {
        "en": "**Clamp rings (in front of the membranes)**",
        "de": "**Klemmringe (vor den Membranen)**",
    },
    "help_clamp": {
        "en": "Rings in front of both membranes (K67/K87). They recess "
              "the membranes by their thickness → the geometric front-to-"
              "rear distance d_ext grows by 2×thickness. The cardioid "
              "null forms when the internal delay of the phase-shifter "
              "network (bores, gap films, spacer) matches this external "
              "delay. The width only enters the outer radius. "
              "0 = no rings.",
        "de": "Ringe vor beiden Membranen (K67/K87). Sie "
              "versenken die Membranen um ihre Dicke → die "
              "geometrische Front-Rück-Distanz d_ext wächst "
              "um 2×Dicke. Die Nierennull entsteht, wenn die "
              "interne Laufzeit des Phasenschieber-Netzwerks "
              "(Bohrungen, Spaltfilme, Spacer) diese externe "
              "Laufzeit trifft. Die Breite geht nur in den "
              "Außenradius ein. 0 = keine Ringe.",
    },
    "lbl_clamp_t": {
        "en": "Clamp ring — thickness per side [mm]",
        "de": "Klemmring — Dicke je Seite [mm]",
    },
    "help_clamp_t": {
        "en": "Axial build-up in front of each membrane.",
        "de": "Axiale Auftragung vor jeder Membran.",
    },
    "lbl_clamp_w": {
        "en": "Clamp ring — width [mm]",
        "de": "Klemmring — Breite [mm]",
    },
    "help_clamp_w": {
        "en": "Radial extent of the ring.",
        "de": "Radiale Ausdehnung des Rings.",
    },

    # ------------------------- Lochkreis-Zeilen -------------------------
    "cap_ring_n": {"en": "Count", "de": "Anzahl"},
    "cap_ring_pcd": {
        "en": "Pitch circle Ø [mm]",
        "de": "Lochkreis Ø [mm]",
    },
    "lbl_ring_n": {
        "en": "Count · circle {i}",
        "de": "Anzahl · Kreis {i}",
    },
    "lbl_ring_pcd": {
        "en": "Pitch circle Ø [mm] · circle {i}",
        "de": "Lochkreis Ø [mm] · Kreis {i}",
    },
    "btn_ring_add": {"en": "➕ Pitch circle", "de": "➕ Lochkreis"},
    "help_ring_add": {
        "en": "Adds another pitch circle (one per click) to model real "
              "hole patterns.",
        "de": "Fügt einen weiteren Lochkreis hinzu (je Druck ein "
              "Kreis), um reale Lochmuster nachzubilden.",
    },
    "btn_ring_del": {"en": "➖ last circle", "de": "➖ letzter Kreis"},
    "help_ring_del": {
        "en": "Removes the last pitch circle.",
        "de": "Entfernt den letzten Lochkreis.",
    },
    "cap_ring_total": {
        "en": "total: {total} holes on {n} pitch circles",
        "de": "gesamt: {total} Löcher auf {n} Lochkreisen",
    },

    # ------------------------- Rückseite --------------------------------
    "exp_rear": {
        "en": "Rear side & delay line",
        "de": "Rückseite & Laufzeitglied",
    },
    "cap_rear_k67": {
        "en": "In the K67 design the passive rear membrane takes over "
              "this function — the following parameters are inactive.",
        "de": "Bei der K67-Bauform übernimmt die passive "
              "Rückmembran diese Funktion — die folgenden "
              "Parameter sind inaktiv.",
    },
    "lbl_rear_on": {"en": "Rear side active", "de": "Rückseite aktiv"},
    "help_rear_on": {
        "en": "Disabled: the rear assembly (delay line, cavity, inlet "
              "holes) is omitted — the backplate through holes then open "
              "(through the rear fabric) DIRECTLY into the sound field "
              "(simple gradient transducer, path difference = gap + "
              "backplate thickness). The capsule is hermetically sealed "
              "only with 0 through holes.",
        "de": "Deaktiviert: die rückwärtige Baugruppe (Laufzeit-"
              "glied, Hohlraum, Einlasslöcher) entfällt — die "
              "Durchgangslöcher der Backplate münden dann durch "
              "das rückwärtige Gewebe DIREKT ins Schallfeld "
              "(einfacher Gradientenempfänger, Wegdifferenz = "
              "Spalt + Backplate-Dicke). Hermetisch geschlossen "
              "ist die Kapsel nur mit 0 Durchgangslöchern.",
    },
    "hd_spacer": {
        "en": "**Spacer & rear plate (K103 design)**",
        "de": "**Spacer & Rückplatte (K103-Bauform)**",
    },
    "help_spacer_hd": {
        "en": "Directly behind the backplate: thin spacer ring and a "
              "massive perforated rear plate — as in the Neumann K103 "
              "(TLM 103), whose K87-style front is closed by a plate "
              "instead of a rear membrane. The narrow spacer provides the "
              "frictional resistance of the cardioid phase shifter. If "
              "delay line, cavity and inlet holes are 0, the plate holes "
              "open directly into the rear sound field. A rear plate "
              "without holes seals the capsule (pressure transducer).",
        "de": "Direkt hinter der Backplate: dünner Distanzring "
              "(Spacer) und massive, gelochte Rückplatte — wie "
              "beim Neumann K103 (TLM 103), dessen K87-artige "
              "Front statt einer Rückmembran durch eine Platte "
              "abgeschlossen ist. Der enge Spacer liefert den "
              "Reibungswiderstand des Nieren-Phasenschiebers. "
              "Sind Laufzeitglied, Hohlraum und Einlasslöcher 0, "
              "münden die Plattenlöcher direkt ins rückwärtige "
              "Schallfeld. Eine Rückplatte ohne Löcher "
              "verschließt die Kapsel (Druckempfänger).",
    },
    "lbl_spacer": {
        "en": "Spacer — height [µm]",
        "de": "Spacer — Höhe [µm]",
    },
    "help_spacer": {
        "en": "Air layer between backplate and rear plate. 0 = no spacer. "
              "Narrow gap = more friction (R ~ 1/h³) — the tuning element "
              "of the polar pattern.",
        "de": "Luftschicht zwischen Backplate und Rück-"
              "platte. 0 = kein Spacer. Enger Spalt = mehr "
              "Reibung (R ~ 1/h³) — das Abstimmelement der "
              "Richtcharakteristik.",
    },
    "lbl_rp_t": {
        "en": "Rear plate — thickness [mm]",
        "de": "Rückplatte — Dicke [mm]",
    },
    "help_rp_t": {
        "en": "Massive plate behind the spacer. 0 = no rear plate.",
        "de": "Massive Platte hinter dem Spacer. "
              "0 = keine Rückplatte.",
    },
    "lbl_rp_n": {
        "en": "Rear plate — hole count",
        "de": "Rückplatte — Löcher Anzahl",
    },
    "help_rp_n": {
        "en": "0 = rear plate without holes → rear side sealed (pressure "
              "transducer).",
        "de": "0 = Rückplatte ohne Löcher → Rückseite "
              "verschlossen (Druckempfänger).",
    },
    "lbl_rp_d": {
        "en": "Rear plate — hole Ø [mm]",
        "de": "Rückplatte — Löcher Ø [mm]",
    },
    "hd_delay": {
        "en": "**Delay line & cavity**",
        "de": "**Laufzeitglied & Hohlraum**",
    },
    "lbl_delay": {
        "en": "Delay line — length [mm]",
        "de": "Laufzeitglied — Länge [mm]",
    },
    "help_delay": {
        "en": "Acoustic line behind the membrane; delay τ = L/c.",
        "de": "Akustische Leitung hinter der Membran; "
              "Laufzeit τ = L/c.",
    },
    "lbl_cav_len": {
        "en": "Cavity — length [mm]",
        "de": "Hohlraum — Länge [mm]",
    },
    "lbl_cav_wall": {
        "en": "Cavity — wall thickness [mm]",
        "de": "Hohlraum — Wandstärke [mm]",
    },
    "lbl_hole_pos": {
        "en": "Cavity holes — position",
        "de": "Hohlraumlöcher — Position",
    },
    "lbl_cav_n": {
        "en": "Cavity holes — count",
        "de": "Hohlraumlöcher — Anzahl",
    },
    "help_cav_n": {
        "en": "0 = rear side closed → pressure transducer "
              "(omnidirectional).",
        "de": "0 = Rückseite geschlossen → Druckempfänger "
              "(Kugelcharakteristik).",
    },
    "lbl_cav_d": {
        "en": "Cavity holes — Ø [mm]",
        "de": "Hohlraumlöcher — Ø [mm]",
    },
    "help_cav_d": {
        "en": "0 = rear side closed.",
        "de": "0 = Rückseite geschlossen.",
    },
    "lbl_cav_ax": {
        "en": "Cavity holes — axial position [mm]",
        "de": "Hohlraumlöcher — axiale Position [mm]",
    },
    "help_cav_ax": {
        "en": "Distance from the cavity entrance (only for position "
              "'circumference').",
        "de": "Abstand vom Hohlraumeingang "
              "(nur bei Position 'Umfang').",
    },

    # ------------------------- Gewebe -----------------------------------
    "exp_fabric": {"en": "Acoustic fabric", "de": "Akustisches Gewebe"},
    "lbl_fab_front": {
        "en": "In front of the membrane [Rayl]",
        "de": "Vor der Membran [Rayl]",
    },
    "lbl_fab_rear": {
        "en": "Behind the backplate [Rayl]",
        "de": "Hinter der Backplate [Rayl]",
    },

    # ------------------------- Gehäuse & Beugung ------------------------
    "exp_body": {"en": "Body & diffraction", "de": "Gehäuse & Beugung"},
    "lbl_diffr": {
        "en": "Diffraction at the body (pressure build-up)",
        "de": "Beugung am Gehäuse (Druckstau)",
    },
    "help_diffr": {
        "en": "Scattering of the plane wave at the rigid equivalent "
              "sphere (Morse series): frontal pressure build-up of up to "
              "+6 dB, rear shadowing and the membrane aperture effect. "
              "This makes even a pure pressure transducer directional at "
              "high frequencies — as in reality. Disable only to compare "
              "against the idealized point model.",
        "de": "Streuung der ebenen Welle am starren Kugel-Ersatz-"
              "gehäuse (Morse-Reihe): frontaler Druckstau bis "
              "+6 dB, rückwärtige Abschattung und Apertureffekt "
              "der Membran. Dadurch richtet auch ein reiner "
              "Druckempfänger zu hohen Frequenzen hin — wie in "
              "der Realität. Deaktivieren nur zum Vergleich mit "
              "dem idealisierten Punktmodell.",
    },
    "lbl_body_dia": {
        "en": "Body diameter [mm]",
        "de": "Gehäusedurchmesser [mm]",
    },
    "help_body_dia": {
        "en": "Diameter of the spherical equivalent body; sets the "
              "frequency where pressure build-up and shadowing begin: "
              "ka = 1 at f ≈ 109/d Hz (d in m) — for Ø 26 mm from "
              "≈ 4 kHz.",
        "de": "Durchmesser des kugelförmigen Ersatz-"
              "gehäuses; bestimmt, ab welcher Frequenz "
              "Druckstau und Abschattung einsetzen: "
              "ka = 1 bei f ≈ 109/d Hz (d in m) — für "
              "Ø 26 mm also ab ≈ 4 kHz.",
    },
    "lbl_ax_body": {
        "en": "Axial body (front-to-rear transfer)",
        "de": "Axialer Körper (Front-Rück-Transfer)",
    },
    "help_ax_body": {
        "en": "Reference body for the axial front-to-rear transfer "
              "G(180°) of the dual-diaphragm design. Sphere (d_ext): "
              "default — describes the capsule MOUNTED on the microphone "
              "body (the body blocks the disc-edge detour). Spheroid: "
              "exact scattering at the FREE-standing disc (radial R_body, "
              "axial d_ext/2, self-built special functions with Wronskian "
              "self-check) — considerably longer effective distance (K67: "
              "~32 instead of 18 mm, thin disc: 4R/π at the pole), the "
              "minimum moves well before 180°. Documented reference case, "
              "see Gegenprobe 20. BEM: mounting-faithful boundary-element "
              "method on the head + microphone-body contour (m=0, "
              "validated against BOTH the sphere and spheroid series) — "
              "lies between the two reference bodies. CONSIDERABLY slower "
              "(~1–2 s per frequency point), results are cached.",
        "de": "Referenzkörper für den axialen Front-Rück-"
              "Transfer G(180°) der Doppelmembran-Bauform. "
              "Kugel (d_ext): Standard — beschreibt die am "
              "Mikrofonkörper MONTIERTE Kapsel (der Körper "
              "unterbindet den Scheibenrand-Umweg). "
              "Sphäroid: exakte Streuung an der FREI "
              "stehenden Scheibe (radial R_body, axial "
              "d_ext/2, eigene Spezialfunktionen mit "
              "Wronski-Selbstprüfung) — deutlich längere "
              "effektive Distanz (K67: ~32 statt 18 mm, "
              "dünne Scheibe: 4R/π am Pol), Minimum "
              "wandert weit vor 180°. Dokumentierter "
              "Referenzfall, s. Gegenprobe 20. BEM: "
              "montagetreues Randelementverfahren auf der "
              "Kontur Kopf + Mikrofonkörper (m=0, gegen "
              "Kugel- UND Sphäroid-Reihe validiert) — "
              "liegt zwischen beiden Referenzkörpern. "
              "DEUTLICH langsamer (~1-2 s je Frequenz-"
              "punkt), Ergebnisse werden gecacht.",
    },
    "lbl_bem_dia": {
        "en": "BEM: body Ø [mm]",
        "de": "BEM: Körper-Ø [mm]",
    },
    "help_bem_dia": {
        "en": "Microphone-body cylinder below the capsule head; 0 = "
              "free-standing head (disc reference).",
        "de": "Mikrofonkörper-Zylinder unter dem "
              "Kapselkopf; 0 = frei stehender Kopf "
              "(Scheiben-Referenz).",
    },
    "lbl_bem_gap": {
        "en": "BEM: air gap head→body [mm]",
        "de": "BEM: Luftspalt Kopf→Körper [mm]",
    },
    "help_bem_gap": {
        "en": "Axial distance between the head's rear face and the "
              "body's top face.",
        "de": "Axialer Abstand zwischen Kopf-Rück"
              "seite und Körper-Oberseite.",
    },
    "lbl_bem_len": {
        "en": "BEM: body length [mm]",
        "de": "BEM: Körperlänge [mm]",
    },
    "help_bem_len": {
        "en": "Length of the body cylinder (finite, capped with "
              "rounded fillets).",
        "de": "Länge des Körperzylinders (endlich, "
              "verrundet gekappt).",
    },
    "cap_bem": {
        "en": "⏳ BEM solves a boundary-element system (~200 elements) "
              "per frequency point. Recommendation: frequency points "
              "≤ 150.",
        "de": "⏳ BEM rechnet je Frequenzpunkt ein Rand"
              "elementsystem (~200 Elemente). Empfehlung: "
              "Frequenzpunkte ≤ 150.",
    },

    # ------------------------- Spaltfilm-Modell -------------------------
    "exp_squeeze": {"en": "Gap-film model", "de": "Spaltfilm-Modell"},
    "lbl_2d": {
        "en": "2D field model (modified Reynolds equation)",
        "de": "2D-Feldmodell (modifizierte Reynolds-Gleichung)",
    },
    "help_2d": {
        "en": "Off: the air gap is a lumped element (Škvor resistance + "
              "compliance + hole impedance). Fast and sufficient for "
              "dense, uniform hole patterns.\n\n"
              "On: the pressure field in the gap is solved axisymmetric"
              "ally as a modified Reynolds equation (Homentcovschi & "
              "Miles). Separates the compliance return path (gap volume + "
              "blind holes) from the feedback path (through holes only) "
              "and captures the radial pressure build-up. Removes the "
              "exaggerated gap resonance with few narrow holes and uses "
              "the pitch-circle radii (PCD). Somewhat slower.",
        "de": "Aus: der Luftspalt ist ein Lumped-Element (Škvor-"
              "Widerstand + Nachgiebigkeit + Lochimpedanz). Schnell "
              "und für dichte gleichmäßige Lochmuster ausreichend.\n\n"
              "An: das Druckfeld im Spalt wird als modifizierte "
              "Reynolds-Gleichung (Homentcovschi & Miles) axial-"
              "symmetrisch gelöst. Trennt den Nachgiebigkeits-"
              "Rückweg (Spaltvolumen + Blindlöcher) vom Rück-"
              "kopplungsweg (nur Durchgangslöcher) und erfasst den "
              "radialen Druckaufbau. Beseitigt die überhöhte "
              "Spaltresonanz bei wenigen engen Löchern und nutzt "
              "die Lochkreis-Radien (PCD). Etwas langsamer.",
    },
    "cap_2d": {
        "en": "The pitch circles in the Backplate section now control "
              "the radial hole distribution in the gap field.",
        "de": "Die Lochkreise im Abschnitt Backplate steuern "
              "jetzt die radiale Lochverteilung im Spaltfeld.",
    },
    "lbl_3d": {
        "en": "3D field model (discrete holes, r-φ sandwich)",
        "de": "3D-Feldmodell (diskrete Löcher, r-φ-Sandwich)",
    },
    "help_3d": {
        "en": "Full (r, φ) field model: all gap films AND the membrane(s) "
              "as fields; through and blind holes sit DISCRETELY at their "
              "positions (azimuthal inflow and partially decoupled blind "
              "holes are resolved; damps the internal Helmholtz resonance "
              "realistically). All architectures with through holes: "
              "single/dual route the discrete holes into a manifold node "
              "terminated by the same lumped rear assembly as 1D/2D "
              "(spacer/rear plate, delay line, cavity). Dual diaphragm "
              "with center gap > 0 computes the TWO-PIECE electrode (K67 "
              "type): center gap as a third film, stepped bores as a "
              "two-port chain per hole, electrode halves rotated against "
              "each other. Takes precedence over the 2D switch. "
              "CONSIDERABLY slower (~1–2 s, K67 type up to ~10 s per "
              "frequency point) — reduce frequency points!",
        "de": "Volles (r, φ)-Feldmodell: alle Spaltfilme UND "
              "die Membran(en) als Felder, Durchgangs- und "
              "Sacklöcher sitzen DISKRET an ihren Positionen "
              "(azimutale Zuströmung und teilentkoppelte Sack-"
              "löcher werden aufgelöst; bedämpft die interne "
              "Helmholtz-Resonanz realistisch). Alle Bauformen "
              "mit Durchgangslöchern: single/dual führen die "
              "diskreten Löcher in einen Sammelknoten, dessen "
              "Abschluss dieselbe Lumped-Rückbaugruppe wie bei "
              "1D/2D bildet (Spacer/Rückplatte, Laufzeitglied, "
              "Hohlraum). Doppelmembran mit Mittelabstand > 0 "
              "rechnet die ZWEITEILIGE Elektrode (K67-Typ): "
              "Zwischenspalt als dritter Film, Stufenbohrungen "
              "als Zweitor-Kette je Loch, Elektrodenhälften "
              "gegeneinander verdreht. Hat Vorrang vor dem "
              "2D-Schalter. DEUTLICH langsamer (~1–2 s, K67-Typ "
              "bis ~10 s je Frequenzpunkt) — Frequenzpunkte "
              "reduzieren!",
    },
    "lbl_rot_auto": {
        "en": "Half rotation automatic (half hole pitch)",
        "de": "Verdrehung der Hälften automatisch (halbe Lochteilung)",
    },
    "help_rot_auto": {
        "en": "K67 type only (center gap > 0): the real electrode halves "
              "are rotated so the through holes do not face each other — "
              "automatically 180°/n_through_holes (with 60 holes that is "
              "3°). Aligned holes (0°) short-circuit the cardioid phase "
              "shifter through the center gap: shallower 180° "
              "cancellation, higher sensitivity.",
        "de": "Nur K67-Typ (Mittelabstand > 0): die realen "
              "Elektrodenhälften sind so verdreht, dass die "
              "Durchgangslöcher nicht zueinander zeigen — "
              "automatisch 180°/n_Durchgangslöcher (bei 60 "
              "Löchern also 3°). Ausgerichtete Löcher (0°) "
              "kurzschließen den Nieren-Phasenschieber "
              "durch den Zwischenspalt: flachere 180°-Aus"
              "löschung, höhere Empfindlichkeit.",
    },
    "lbl_rot_deg": {
        "en": "Rotation of electrode halves [°]",
        "de": "Verdrehung der Elektrodenhälften [°]",
    },
    "help_rot_deg": {
        "en": "0° = through holes of both halves face each other (short-"
              "circuits the phase shifter); half pitch = maximum offset "
              "as on the real K67.",
        "de": "0° = Durchgangslöcher beider Hälften "
              "zeigen aufeinander (Kurzschluss des "
              "Phasenschiebers); halbe Teilung = "
              "maximaler Versatz wie an der realen "
              "K67.",
    },
    "cap_3d": {
        "en": "⏳ 3D performs one LU factorization per frequency point "
              "(one-piece ~24,000 unknowns; K67 type with third film and "
              "finer azimuthal resolution correspondingly more). "
              "Recommendation: frequency points ≤ 150 (K67 type ≤ 100). "
              "Results are cached per parameter set — reruns without "
              "changes are instant.",
        "de": "⏳ 3D rechnet je Frequenzpunkt eine LU-Faktori"
              "sierung (einteilig ~24 000 Unbekannte; K67-Typ "
              "mit drittem Film und feinerer Azimut-Auflösung "
              "entsprechend mehr). Empfehlung: Frequenzpunkte "
              "≤ 150 (K67-Typ ≤ 100). Ergebnisse werden je "
              "Parametersatz gecacht — Reruns ohne Änderung "
              "sind sofort da.",
    },

    # ------------------------- Simulation -------------------------------
    "exp_sim": {"en": "Simulation", "de": "Simulation"},
    "lbl_npts": {"en": "Frequency points", "de": "Frequenzpunkte"},
    "lbl_norm": {
        "en": "Normalize amplitude to 1 kHz",
        "de": "Amplitude auf 1 kHz normieren",
    },
    "lbl_dirf": {
        "en": "Polar pattern frequencies [Hz]",
        "de": "Richtdiagramm-Frequenzen [Hz]",
    },

    # ------------------------- Fortschritt / Fehler ---------------------
    "prog_fmt": {
        "en": "⚙️ Computing — {label} … {pct:.0f} %",
        "de": "⚙️ Berechnung — {label} … {pct:.0f} %",
    },
    "prog_build": {"en": "building model", "de": "Modell aufbauen"},
    "prog_fr": {"en": "frequency response", "de": "Frequenzgang"},
    "prog_di": {
        "en": "polar pattern {f:.0f} Hz",
        "de": "Richtdiagramm {f:.0f} Hz",
    },
    "err_params": {
        "en": "⚠️ Invalid parameter combination: {exc}",
        "de": "⚠️ Ungültige Parameterkombination: {exc}",
    },

    # ------------------------- Kennwerte --------------------------------
    "met_sens": {
        "en": "Sensitivity @ 1 kHz",
        "de": "Empfindlichkeit @ 1 kHz",
    },
    "help_met_sens": {
        "en": "{db:.1f} dB re 1 V/Pa (open circuit, without stray "
              "capacitance)",
        "de": "{db:.1f} dB re 1 V/Pa (Leerlauf, ohne Streukapazität)",
    },
    "met_fres": {"en": "Membrane resonance", "de": "Membranresonanz"},
    "help_met_fres": {
        "en": "Consistency check from tension/bending stiffness: "
              "{f:.0f} Hz",
        "de": "Konsistenz-Check aus Vorspannung/Biegesteifigkeit: "
              "{f:.0f} Hz",
    },
    "met_c0": {
        "en": "Static capacitance C₀",
        "de": "Ruhekapazität C₀",
    },
    "help_met_c0": {
        "en": "per backplate · architecture: {n} backplate(s)",
        "de": "je Backplate · Architektur: {n} Backplate(s)",
    },
    "met_soft": {
        "en": "Spring softening (bias)",
        "de": "Feder-Erweichung (Bias)",
    },
    "help_met_soft": {
        "en": "Share of the membrane stiffness consumed by the "
              "electrostatic attraction at the operating point. Pull-in "
              "voltage of this configuration: ≈ {upi}; static deflection "
              "{w0:.1f} µm (residual center gap {hmin:.1f} µm).",
        "de": "Anteil der Membransteifigkeit, den die elektrostatische "
              "Anziehung am Arbeitspunkt aufzehrt. Pull-in-Spannung "
              "dieser Konfiguration: ≈ {upi}; statische Durchbiegung "
              "{w0:.1f} µm (Restspalt Mitte {hmin:.1f} µm).",
    },
    "met_tau_ext": {
        "en": "External delay τ_ext",
        "de": "Externe Laufzeit τ_ext",
    },
    "help_tau_ext": {
        "en": "Front-to-rear transfer G(180°) of the sound field around "
              "the capsule body (axial body diffraction), evaluated as "
              "phase at {f:.0f} Hz. Equivalent path length: {d:.1f} mm.",
        "de": "Front-Rück-Übertragung G(180°) des Schallfelds um "
              "den Kapselkörper (axiale Körperbeugung), als Phase "
              "bei {f:.0f} Hz ausgewertet. "
              "Äquivalente Wegstrecke: {d:.1f} mm.",
    },
    "met_tau_int": {
        "en": "Internal delay τ_int",
        "de": "Interne Laufzeit τ_int",
    },
    "help_tau_int": {
        "en": "Reverse transfer D_r of the internal phase-shifter network "
              "(bores, gap films, spacer, rear side), evaluated as phase "
              "at {f:.0f} Hz — frequency dependent, since it is an RC "
              "phase shifter with film inertia. Equivalent path length: "
              "{d:.1f} mm.",
        "de": "Rück-Übertragung D_r des internen Phasenschieber-"
              "Netzwerks (Bohrungen, Spaltfilme, Spacer, Rückseite), "
              "als Phase bei {f:.0f} Hz "
              "ausgewertet — frequenzabhängig, da RC-Phasenschieber "
              "mit Filmträgheit. Äquivalente Wegstrecke: "
              "{d:.1f} mm.",
    },
    "met_ratio": {
        "en": "Ratio internal / external",
        "de": "Verhältnis intern / extern",
    },
    "delta_ratio": {
        "en": "{d:+.0f} % vs. matched",
        "de": "{d:+.0f} % vs. Anpassung",
    },
    "help_ratio": {
        "en": "≈ 1: delays matched → deepest cancellation at 180°. < 1: "
              "internal delay too short — the pattern minimum moves "
              "before 180° (towards hypercardioid). > 1: internal delay "
              "too long — the minimum stays pinned at 180° but the "
              "cancellation gets shallower. Only meaningful near the "
              "probe frequency if the internal Helmholtz resonance lies "
              "in the band!",
        "de": "≈ 1: Laufzeiten angepasst → tiefste Auslöschung bei "
              "180°. < 1: interne Laufzeit zu kurz — das Pattern-"
              "Minimum wandert vor 180° (Richtung Hyperniere). "
              "> 1: interne Laufzeit zu lang — das Minimum bleibt "
              "bei 180° gepinnt, die Auslöschung wird aber flacher. "
              "Nur nahe der Sondenfrequenz aussagekräftig, wenn die "
              "interne Helmholtz-Resonanz im Band liegt!",
    },
    "met_fh": {
        "en": "Internal Helmholtz resonance",
        "de": "Interne Helmholtz-Resonanz",
    },
    "fh_band": {
        "en": "inside the passband!",
        "de": "im Übertragungsband!",
    },
    "help_fh": {
        "en": "Resonance of the through-hole inertia against the inner "
              "compliance (gap + blind holes), determined as the 90° "
              "phase crossing of D_r. If it lies IN the band, |D_r| "
              "collapses below it and the pattern shape drifts with "
              "frequency (supercardioid → wide cardioid → omni) — "
              "remedies: stepped bores, thinner plate, larger/more "
              "through holes. Healthy: well above the passband.",
        "de": "Resonanz der Durchgangsloch-Trägheit gegen die innere "
              "Nachgiebigkeit (Spalt + Blindlöcher), bestimmt als "
              "90°-Phasendurchgang von D_r. Liegt sie IM Band, "
              "bricht |D_r| darunter ein und die Pattern-Form "
              "wandert über die Frequenz (Superniere → breite Niere "
              "→ Kugel) — Abhilfe: Stufenbohrung, dünnere Platte, "
              "größere/mehr Durchgangslöcher. Gesund: deutlich "
              "oberhalb des Übertragungsbands.",
    },

    # ------------------------- Hauptbereich -----------------------------
    "info_no_dr": {
        "en": "Rear side closed (pressure transducer) — there is no "
              "internal phase-shifter path and therefore no D_r.",
        "de": "Rückseite geschlossen (Druckempfänger) — es gibt keinen "
              "internen Phasenschieber-Pfad und damit kein D_r.",
    },
    "exp_diag": {
        "en": "Derived model parameters (diagnostics)",
        "de": "Abgeleitete Modellparameter (Diagnose)",
    },
    "hd_data": {"en": "Data & export", "de": "Daten & Export"},
    "tab_fr": {"en": "Frequency response", "de": "Frequenzgang"},
    "tab_di": {"en": "Polar pattern", "de": "Richtdiagramm"},
    "lbl_csv": {"en": "CSV format", "de": "CSV-Format"},
    "csv_intl": {
        "en": "Comma / point (international)",
        "de": "Komma / Punkt (international)",
    },
    "csv_excel_de": {
        "en": "Semicolon / comma (Excel DE)",
        "de": "Semikolon / Komma (Excel DE)",
    },
    "btn_csv_fr": {
        "en": "⬇️ Frequency response as CSV",
        "de": "⬇️ Frequenzgang als CSV",
    },
    "btn_csv_di": {
        "en": "⬇️ Polar pattern as CSV",
        "de": "⬇️ Richtdiagramm als CSV",
    },
    "csv_fr_name": {
        "en": "capsim_frequency_response.csv",
        "de": "capsim_frequenzgang.csv",
    },
    "csv_di_name": {
        "en": "capsim_polar_pattern.csv",
        "de": "capsim_richtdiagramm.csv",
    },
    "footer": {
        "en": "Capsim · physics model: ABCD chain matrices, "
              "Zwikker–Kosten hole impedances, Škvor squeeze film, "
              "electrostatic transduction with spring softening. "
              "Sensitivity = open-circuit voltage without stray "
              "capacitance/amplifier load.",
        "de": "Capsim · Physikmodell: ABCD-Kettenmatrizen, Zwikker–Kosten-"
              "Lochimpedanzen, Škvor-Squeeze-Film, elektrostatische "
              "Wandlung mit Feder-Erweichung. Empfindlichkeit = "
              "Leerlaufspannung ohne Streukapazität/Verstärkerlast.",
    },

    # ------------------------- Diagramme --------------------------------
    "fig_amp_norm": {
        "en": "Amplitude [dB rel. 1 kHz]",
        "de": "Amplitude [dB rel. 1 kHz]",
    },
    "fig_amp_abs": {
        "en": "Amplitude [dB re 1 V/Pa]",
        "de": "Amplitude [dB re 1 V/Pa]",
    },
    "fig_freq": {"en": "Frequency [Hz]", "de": "Frequenz [Hz]"},
    "fig_phase": {"en": "Phase [°]", "de": "Phase [°]"},
    "fig_bode_title": {
        "en": "Frequency response (0° incidence)",
        "de": "Frequenzgang (0° Einfall)",
    },
    "fig_rear_title": {
        "en": "Directivity vs. frequency (lateral/rear)",
        "de": "Richtwirkung über die Frequenz (seitlich/rückwärtig)",
    },
    "fig_rear_ann": {
        "en": "−6 dB (ideal cardioid @90°)",
        "de": "−6 dB (ideale Niere @90°)",
    },
    "fig_lvl": {
        "en": "Level rel. 0° [dB]",
        "de": "Pegel rel. 0° [dB]",
    },
    "name_90": {"en": "90° rel. 0°", "de": "90° rel. 0°"},
    "name_180": {"en": "180° rel. 0°", "de": "180° rel. 0°"},
    "fig_dr_title": {
        "en": "Phase shifter D_r — does it match the external target?",
        "de": "Phasenschieber D_r — trifft er das externe Ziel?",
    },
    "name_dr": {"en": "|D_r| internal", "de": "|D_r| intern"},
    "name_g": {
        "en": "|G(180°)| external (target)",
        "de": "|G(180°)| extern (Ziel)",
    },
    "name_arg_dr": {"en": "arg D_r internal", "de": "arg D_r intern"},
    "name_arg_g": {
        "en": "arg G(180°) external (target)",
        "de": "arg G(180°) extern (Ziel)",
    },
    "fig_mag": {"en": "Magnitude [–]", "de": "Betrag [–]"},
    "fig_polar_title": {
        "en": "Polar pattern (normalized to 0°)",
        "de": "Richtdiagramm (normiert auf 0°)",
    },

    # ------------------------- Datenspalten -----------------------------
    "col_freq": {"en": "frequency_hz", "de": "frequenz_hz"},
    "col_sens": {"en": "sensitivity_mv_pa", "de": "empfindlichkeit_mv_pa"},
    "col_amp": {
        "en": "amplitude_db_re_1v_pa",
        "de": "amplitude_db_re_1v_pa",
    },
    "col_ampn": {
        "en": "amplitude_db_norm_1khz",
        "de": "amplitude_db_norm_1khz",
    },
    "col_phase": {"en": "phase_deg", "de": "phase_deg"},
    "col_l90": {"en": "level_90_rel0_db", "de": "pegel_90_rel0_db"},
    "col_l180": {"en": "level_180_rel0_db", "de": "pegel_180_rel0_db"},
    "col_drmag": {"en": "dr_magnitude", "de": "dr_betrag"},
    "col_drph": {"en": "dr_phase_deg", "de": "dr_phase_deg"},
    "col_angle": {"en": "angle_deg", "de": "winkel_deg"},
    "col_lvl_prefix": {"en": "level_db_", "de": "pegel_db_"},
    "col_lin_prefix": {"en": "linear_", "de": "linear_"},
}

# Kanonische Auswahl-Werte -> Anzeige je Sprache. Die KANONISCHEN Werte
# (Dict-Schlüssel) sind die Bestandsschlüssel aus app.py und wandern
# unverändert in Session-State und Projektdateien.
LABEL_TR = {
    # Material
    "PET (Mylar)": {"en": "PET (Mylar)", "de": "PET (Mylar)"},
    "Nickel": {"en": "Nickel", "de": "Nickel"},
    "Titan": {"en": "Titanium", "de": "Titan"},
    "Aluminium": {"en": "Aluminium", "de": "Aluminium"},
    "Gold": {"en": "Gold", "de": "Gold"},
    # Architektur
    "Single Backplate": {
        "en": "Single backplate",
        "de": "Single Backplate",
    },
    "Dual Symmetrical Backplates": {
        "en": "Dual symmetrical backplates",
        "de": "Dual Symmetrical Backplates",
    },
    "Doppelmembran (K67-Bauform)": {
        "en": "Dual diaphragm (K67 design)",
        "de": "Doppelmembran (K67-Bauform)",
    },
    # Hohlraumloch-Position
    "Umfang": {"en": "Circumference", "de": "Umfang"},
    "Ende (Stirnfläche)": {"en": "End (face)", "de": "Ende (Stirnfläche)"},
    # Axialer Körper
    "Kugel (d_ext, montiert)": {
        "en": "Sphere (d_ext, mounted)",
        "de": "Kugel (d_ext, montiert)",
    },
    "Sphäroid (freie Scheibe)": {
        "en": "Spheroid (free disc)",
        "de": "Sphäroid (freie Scheibe)",
    },
    "BEM (Kopf + Körper)": {
        "en": "BEM (head + body)",
        "de": "BEM (Kopf + Körper)",
    },
}
