# Statistik – Referenzprogramme für den Unterricht

Dieses Repository enthält **fertige, stabile Referenzprogramme**
für den Statistikunterricht (Sek II) und die Lehrkräftefortbildung.

Die Programme sind **Werkzeuge**, keine Programmierübungen.

---

## Ziel

Nach einer Fortbildung sollen Lehrkräfte:

- statistische Grafiken **selbstständig erzeugen**
- Parameter verändern (z. B. Stichprobengröße, Sicherheitsniveau)
- Ergebnisse **zuverlässig und reproduzierbar** darstellen
- die Programme **direkt im Unterricht einsetzen**

Ohne:
- Programmieren lernen zu müssen
- Code umbauen oder debuggen zu müssen

---

## Grundprinzipien

- **Eine Aktivität = ein Notebook**
- **Eine Grafik = eine statistische Idee**
- Modell, Darstellung und Berechnung sind getrennt
- Alle Programme sind deterministisch und Binder-stabil

GeoGebra eignet sich zum **Entdecken**.  
Diese Programme dienen als **Referenz und Absicherung**.

---

# Statistik – Referenzprogramme für den Unterricht

Dieses Repository enthält **fertige, stabile Referenzprogramme**
für den Statistikunterricht (Sek II) und die Lehrkräftefortbildung.

Die Programme sind **Werkzeuge**, keine Programmierübungen
und keine interaktiven Spielumgebungen.

---
# Statistische Intervalle – Referenznotebooks

Dieses Repository enthält didaktisch aufbereitete Referenzprogramme
zur Visualisierung und Untersuchung von Konfidenz- und Prognoseintervallen.

Die Materialien richten sich an:
- Lehrkräfte (LuL)
- Studierende
- Fortbildungen im schulischen und universitären Kontext

Ziel ist es, **stabile, reproduzierbare und mathematisch saubere Werkzeuge**
bereitzustellen, die auch nach einer Fortbildung selbstständig genutzt werden können.

---

## Namensschema und Struktur

Dieses Projekt folgt einem **klaren und konsistenten Namensschema**.
Modell, Geometrie, Darstellung und Simulation werden **streng getrennt**,
um mathematische Begriffe nicht zu vermischen und didaktische Entscheidungen
sichtbar zu halten.

Die meisten Nutzer:innen müssen diese Struktur nicht aktiv reflektieren –
sie sorgt im Hintergrund für Ruhe und Verlässlichkeit.

---

### 1. Konfigurationen (fachlich / didaktisch veränderbar)

Diese Klassen bündeln **inhaltliche Setzungen** und dürfen im Unterricht
oder bei eigenen Experimenten verändert werden.

#### Konfidenzintervalle

**CIConfig**

- `h` – beobachteter Stichprobenanteil  
- `n` – Stichprobengröße  
- `gamma` – Sicherheitsniveau  

Der Name *Config* ist bewusst gewählt:
`h` ist kein Modellparameter, sondern ein beobachtetes Ergebnis.

---

#### Prognoseintervalle

**PIModel**

- `p` – fixierter Modellparameter  
- `n` – Stichprobengröße  
- `gamma` – Sicherheitsniveau  

Hier ist `p` unter der Modellannahme fixiert und wird nicht geschätzt.

---

### 2. Geometrie (Darstellungsraum)

**IntervalGeometry**

Beschreibt ausschließlich den geometrischen Zeichenraum einer Grafik:
- Achsengrenzen
- Auflösung (`points`)

Diese Klasse enthält keine Statistik und keine Modellannahmen.

---

### 3. Darstellung (rein optisch)

**CIStyle**, **CISimStyle**

Diese Klassen steuern ausschließlich:
- Farben
- Linienstärken
- Transparenzen
- Gitter
- Bildgröße

Sie haben keinen Einfluss auf die mathematischen Inhalte.

---

### 4. Simulation (Überdeckungsrate)

**CISimConfig**

- `p_true` ist ein fixierter Referenzwert des Modellparameters
- Zufälligkeit entsteht ausschließlich durch Wiederholung (`m`, `seed`)

Der Name `p_true` markiert explizit:
Der Parameter ist nicht gesetzt, sondern **fixiert unter der Modellannahme**.

---

### 5. Funktionen

Alle zentralen Funktionen tragen zeitlose, fachlich motivierte Namen:

- `plot_ci`
- `plot_ci_simulation`
- `prediction_interval`
- `wilson_ci`
- `wald_ci`

Es gibt keine Versionsnummern oder „final“-Dateien.
Versionierung erfolgt ausschließlich über Git.

---

## Wie nutze ich die Notebooks?

Jedes Notebook steht für **eine klar abgegrenzte Aktivität**.
Es ist als stabile Referenz gedacht, nicht als interaktives Spielzeug.

### Grundprinzip

- Ein Notebook – eine Idee  
- Eine Grafik – eine Aussage  

---

### Typischer Ablauf

1. Notebook öffnen  
2. Zentrale Parameter in Konfigurationsobjekten anpassen  
3. Grafik durch einen einzigen Funktionsaufruf erzeugen  
4. Optional als PDF speichern:

```python
save="dateiname.pdf"

---

## Ziel

Nach einer Fortbildung sollen Lehrkräfte:

- statistische Grafiken **selbstständig erzeugen**
- Modellparameter gezielt verändern (z. B. `n`, `γ`, `α`)
- Ergebnisse **fachlich korrekt** und **reproduzierbar** darstellen
- die Programme **direkt im Unterricht einsetzen**

Ohne:
- Programmieren lernen zu müssen
- Code umbauen oder debuggen zu müssen
- statistische Aussagen vereinfachen oder „weichzeichnen“ zu müssen

---

## Didaktische Haltung

Dieses Projekt folgt bewusst einer **präzisen mathematischen Sprache**.

- Parameter sind **nicht zufällig**
- Intervalle sind **Objekte**, keine Aussagen
- Sicherheits- und Signifikanzniveaus sind **Verfahrenseigenschaften**
- Zufälligkeit liegt **im Verfahren**, nicht im Parameter

Diese Genauigkeit dient **nicht der Formalisierung um ihrer selbst willen**,
sondern der **Vermeidung typischer Fehlvorstellungen** im Statistikunterricht.

> Präzision ist hier eine Form von Respekt  
> – gegenüber der Mathematik, den Schülerinnen und Schülern  
> und der professionellen Rolle von Lehrkräften.

---

## Grundprinzipien

- **Eine Aktivität = ein Notebook**
- **Eine Grafik = eine statistische Idee**
- Modell, Darstellung und Berechnung sind strikt getrennt
- Alle Programme sind deterministisch und Binder-stabil
- Jede Grafik ist referenzierbar und erklärbar

GeoGebra eignet sich hervorragend zum **Entdecken**.  
Diese Programme dienen als **Referenz, Klärung und Absicherung**.

---


## Ordnerstruktur


### Wichtig:
Der Ordner `lib/` muss **nicht geöffnet** werden, um mit den Notebooks zu arbeiten.

---

## Arbeiten mit den Notebooks

In den Notebooks werden nur:

- Parameter gesetzt (z. B. `n`, `p`, `gamma`, `alpha`)
- fertige Funktionen aufgerufen

Beispiel:
```python
model = CIModel(h=0.45, n=80)
plot_wilson_ci(model)


## Zufall und Reproduzierbarkeit

Simulationen verwenden bewusst einen expliziten Zufallsstartwert (`seed`).

- `seed = 42`  
  → reproduzierbare Referenzsimulation  
  → gleiche Ergebnisse bei erneutem Ausführen

- `seed = None`  
  → echte Zufallsrealisierung  
  → Ergebnisse variieren sichtbar

Der Zufall ist Teil des Modells.
Er dient nicht der Illustration, sondern der Analyse von Verfahren.

---

🎯 **Didaktischer Effekt:**  
Das README sagt sehr klar:
> *„Du darfst das benutzen. Du musst es nicht verstehen.“*

Das ist enorm entlastend.

---

