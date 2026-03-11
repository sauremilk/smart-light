# Smart Light – Emotion-gesteuerte Philips Hue Lampe

Multimodale, lokale KI-Emotionserkennung via **Webcam + Mikrofon + Körpersprache** → steuert Hue-Lampenfarbe sanft und kontextbewusst in Echtzeit.  
**Keine Cloud. Keine externen APIs.**

> **Hinweis:** Dieses Repository ist ausschließlich ein privates Hobbyprojekt.

## Features

- **EMA-Smoothing** über den vollständigen 7-Emotions-Wahrscheinlichkeitsvektor – eliminiert Flackern
- **Valence-Arousal-Modell** – kontinuierliches 2D-Farblicht statt diskreter Emotion→Farbe-Tabelle
- **Konfidenz-gewichtetes Alpha** – starke Erkennungen haben mehr Einfluss als unsichere
- **CLAHE-Beleuchtungsnormalisierung** – stabil bei Dunkelheit und Gegenlicht
- **RetinaFace-Detektor** – deutlich robuster bei Drehung, Teilverdeckung und schlechter Beleuchtung
- **Trend-Analyse** – sanftere Übergänge bei fallender Stimmung, schnellere bei Aufhellung
- **Micro-Expression Burst** – bei plötzlichen Konfidenzsprüngen wird temporär mit höherer Framerate analysiert
- **Audio-Emotionserkennung** (optional) – SpeechBrain/wav2vec2 erkennt Stimmungstonalität aus dem Mikrofon
- **Körpersprache-Analyse** (optional) – MediaPipe Pose extrahiert Arousal-Signale aus Haltung
- **Multimodale Fusion** – Video + Audio + Pose werden gewichtet zusammengeführt
- **Nutzer-Kalibrierung** – 7-Emotionen-Kalibrierung gleicht individuelle Erkennungsabweichungen aus
- **Multi-Lampen-Support** – steuert beliebig viele Hue-Lampen gleichzeitig
- **Optionale Alexa-Integration** – emotionsabhaengige Musik-/Lautstaerkesteuerung (inoffiziell via alexapy)

Weitere Details: `docs/ALEXA_INTEGRATION.md`

---

## Voraussetzungen

| Komponente         | Version                    |
| ------------------ | -------------------------- |
| Python             | 3.10+                      |
| Philips Hue Bridge | v2 (API v1)                |
| USB-Webcam         | beliebig                   |
| Mikrofon           | optional (für Audio-Modul) |
| Betriebssystem     | Windows 10/11 oder Linux   |

---

## Schnellstart

### 1. Setup (einmalig)

```powershell
# Windows PowerShell – im Projektordner ausführen:
.\setup.ps1
```

Das Setup aktiviert auch lokale Git-Hooks (`core.hooksPath=.githooks`).
Dadurch werden wichtige staged Änderungen automatisch in
`docs/IMPORTANT_CHANGES.md` dokumentiert, sobald ein Commit erstellt wird.
Ein zusaetzlicher `pre-push`-Hook blockiert Pushes, falls wichtige Aenderungen
ohne entsprechendes Update in `docs/IMPORTANT_CHANGES.md` enthalten sind.
Die Liste der als wichtig geltenden Pfade liegt zentral in
`tools/auto_doc_important_paths.txt`.

Wenn du hauptsaechlich mit Coding-Agenten arbeitest und auch ohne Commit
Zwischenstaende dokumentieren willst:

```powershell
.\venv\Scripts\python tools\auto_document_changes.py --working-tree --actor copilot-agent
```

Optional dauerhaft fuer Agent-Sessions konfigurieren:

```powershell
git config autodoc.actor copilot-agent
```

Oder manuell:

```powershell
python -m venv venv
.\venv\Scripts\pip install -e .
git config core.hooksPath .githooks
```

> **Hinweis:** Torch (~114 MB), SpeechBrain und MediaPipe werden beim ersten `pip install` heruntergeladen. Das kann einige Minuten dauern.

### 2. Bridge-IP konfigurieren

Empfohlen: `config_local.example.py` nach `config_local.py` kopieren und dort
deine lokalen Werte setzen (damit `config.py` unveraendert bleibt):

```powershell
Copy-Item .\config_local.example.py .\config_local.py
```

Dann `config_local.py` anpassen:

```python
HUE_BRIDGE_IP  = "192.168.1.100"   # ← deine Bridge-IP hier
HUE_LIGHT_IDS  = [1]               # ← Lampen-IDs (kommagetrennt)
```

Alternativ kannst du direkt `config.py` bearbeiten.

> **Bridge-IP finden:** Philips Hue App → Einstellungen → Hue Bridges → (i)-Button

### 3. Starten

```powershell
# Vollbetrieb (Video + Audio + Pose):
.\venv\Scripts\python main.py

# Ohne optionale Module (schneller Start, weniger CPU):
.\venv\Scripts\python main.py --no-audio --no-pose

# Einmalige Nutzer-Kalibrierung durchführen:
.\venv\Scripts\python main.py --calibrate

# Mock-Modus (keine Hardware nötig – zum Testen):
.\venv\Scripts\python main.py --mock --no-audio --no-pose

# Bridge-IP und Lampen per CLI überschreiben:
.\venv\Scripts\python main.py --bridge-ip 192.168.178.42 --light-ids 2,3,4
```

**Beenden:** `Q`-Taste im Kamerafenster drücken, oder `Ctrl+C` im Terminal.

---

## CLI-Optionen

| Option                    | Beschreibung                                              |
| ------------------------- | --------------------------------------------------------- |
| `--mock`                  | Simuliert Webcam und Hue-Bridge (kein Hardware nötig)     |
| `--no-audio`              | Audio-Emotionserkennung deaktivieren                      |
| `--no-pose`               | Körpersprache-Analyse deaktivieren                        |
| `--no-alexa`              | Alexa-Steuerung deaktivieren (auch wenn `USE_ALEXA=True`) |
| `--calibrate`             | Interaktive Kalibrierung starten (7 × 10s)                |
| `--calibration-file PATH` | Alternativen Kalibrierungsdatei-Pfad angeben              |
| `--bridge-ip IP`          | Bridge-IP überschreiben                                   |
| `--light-ids IDs`         | Lampen-IDs überschreiben (z.B. `2,3,4`)                   |
| `--pseudonymize-session`  | Pseudonymisiert `participant`/`session_id` im Session-Log |

---

## Nutzer-Kalibrierung

Die Kalibrierung kompensiert individuelle Erkennungsabweichungen (z.B. wenn „neutral" bei dir oft als „sad" erkannt wird):

```powershell
.\venv\Scripts\python main.py --calibrate
```

Das Programm führt dich in 2 Minuten durch 7 Emotionen (je 10 Sekunden). Die Ergebnisse werden in `calibration_default.json` gespeichert und automatisch bei jedem Start geladen.

---

## Benchmarking

Für einen bewusst sehr schwierigen, reproduzierbaren Referenztest (als fixer Vergleichspunkt für Verbesserungen):

```powershell
.\venv\Scripts\python benchmarks\extreme_reference_benchmark.py
```

Der Report wird nach `benchmarks/results/extreme_reference.json` geschrieben und enthält:

- `enhanced.index` als kompakten Referenz-Score (0-1000)
- `delta.index` als Verbesserung gegenüber Baseline
- `hardness.hardest_profile` als aktuell schwierigstes Störungsprofil

Formeln (explizit):

| Metrik           | Formel                                  | Zweck                                            |
| ---------------- | --------------------------------------- | ------------------------------------------------ |
| `weighted_score` | `0.4 * accuracy + 0.6 * macro_f1`       | Balance zwischen Genauigkeit und Klassenfairness |
| `index`          | `round(clamp01(weighted_score) * 1000)` | Kompakter Vergleichswert (0-1000)                |

Zusatz im Report:

- `relative.enhanced_percent_of_baseline.*` gibt `%` relativ zur Baseline aus (z.B. `index`).

Die Störprofile sind deterministisch definiert und im Report unter
`profile_specs` dokumentiert (inkl. Parameterbereiche):

- `low_light_noise`: Gamma-Verdunkelung + Helligkeitsabfall + Gauß-Rauschen
- `motion_blur`: Richtungsunschärfe + leichte Gauß-Unschärfe
- `jpeg_artifacts`: niedrige JPEG-Qualität + Down/Up-Sampling
- `occlusion`: Rechteckverdeckung oder Patch-Replace
- `rotation_scale`: Rotation/Skalierung/Translation
- `color_cast_shadow`: Farbkanal-Gains + Schatten-Gradient
- `mixed_extreme`: Kombination mehrerer Störarten

Schneller Smoke-Test:

```powershell
.\venv\Scripts\python benchmarks\extreme_reference_benchmark.py --limit 14 --variants-per-sample 1 --no-face-mesh
```

### Professionelle Referenz-Suite (empfohlen für Agenten + Releases)

Die Datei `benchmarks/reference_suite.py` kombiniert mehrere Säulen in einem Lauf:

- `extreme_visual_robustness` (harte Bildstörungen)
- `multi_seed_stability` (Seed-Varianz und Robustheit)
- `test_quality` (Projekt-Tests)
- `module_sanity` (Subsystem-Vertragschecks)
- `e2e_runtime` (End-to-End-Laufzeit, Stabilität, Ressourcen)

Damit sind Verbesserungen nur dann gültig, wenn das Gesamtsystem besser wird und nicht nur eine Einzelmetrik.

Scoring-Modell (transparent):

| Metrik              | Formel                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `composite_score01` | `0.40 * extreme_visual_robustness + 0.23 * multi_seed_stability + 0.17 * test_quality + 0.10 * module_sanity + 0.10 * e2e_runtime` |
| `composite_index`   | `round(clamp01(composite_score01) * 1000)`                                                                                         |

`multi_seed_stability` nutzt explizit eine Varianzstrafe:

```text
stability_score01 = clamp01(mean(enhanced_weighted_score) - 0.5 * std(enhanced_weighted_score))
```

```powershell
# Schneller Entwicklungs-Check
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick

# Noch schnellerer lokaler Loop (ohne E2E-Laufzeitmessung)
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick --skip-e2e

# Minimaler lokaler Loop (ohne E2E und ohne Test-Saeule)
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick --skip-e2e --skip-tests

# Vor Handover / Pull Request (Regression-Gate aktiv)
.\venv\Scripts\python benchmarks\reference_suite.py --profile standard --enforce-gate

# Strenger Referenzlauf
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --enforce-gate
```

Hinweis:

- `--skip-e2e` und `--skip-tests` sind nur fuer lokale Iteration gedacht.
- Mit `--enforce-gate` sind diese Flags absichtlich nicht kombinierbar.

Ausgabe:

- `benchmarks/results/reference_suite_latest.json`
- Optionales Baseline-File: `benchmarks/results/reference_suite_baseline.json`
- Historie aller Runs: `benchmarks/results/reference_suite_history.jsonl`

Der `latest`-Report enthält zusätzlich einen `trend`-Block mit Delta gegen den letzten vergleichbaren Run
(bevorzugt gleiches Profil + gleicher Detector).
Zusätzlich enthält er:

- `environment`: OS/Python/Library-Versionen + CUDA/GPU-Metadaten (falls vorhanden)
- `runtime`: Gesamtlaufzeit und Laufzeit pro Komponente
- `components.multi_seed_stability.details.*_stats`: `n`, Mittelwert, Standardabweichung, SEM, 95%-CI
- `relative_to_baseline`: `%` relativ zur Baseline (composite + pro Komponente)
- `components.e2e_runtime.details.scenarios`: p50/p95/p99-Latenz, Drop-Rate, CPU/RAM-Drift je Szenario

Interpretation kleiner Deltas:

- Wenn ein Delta kleiner ist als die `ci95_half_width` in der relevanten Statistik, ist es wahrscheinlich Rauschen.
- Für Merge-/Release-Entscheidungen sind `standard`/`strict` mit mehreren Seeds verpflichtend.

Baseline nur bewusst aktualisieren (nicht automatisch):

```powershell
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --write-baseline
```

Windows-Shortcut:

```powershell
.\benchmarks\run_reference_suite.ps1 -Mode standard -EnforceGate
```

History bei Bedarf deaktivieren:

```powershell
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick --no-history
```

Detailliertes Governance-Protokoll:

- `benchmarks/REFERENCE_BENCHMARK_PROTOCOL.md`

End-to-End-Performance-Hinweis:

- Die Referenz-Suite misst jetzt explizit End-to-End-Runtime via `e2e_runtime`-Komponente.
- Die E2E-Metrik basiert auf kontrollierten `--mock`-Szenarien und ist maschinenvergleichbar.
- Für produktionsnahe Echtzeitprüfung weiterhin zusätzlich `main.py` mit echter Kamera/Mikrofon-Last prüfen.

### Agentic Face Fine-Tuning (vollautomatisiert)

Es gibt jetzt eine autonome Pipeline fuer Face-Fine-Tuning mit Gates:

1. Dataset-Erzeugung (`agentic_dataset_gen.py`, Gate: `>=1000` Samples)
2. Training + ONNX-Export (`finetune_face_agentic.py`, Gate: `val_accuracy >= 0.82`)
3. Benchmark-Gate (`reference_suite.py --profile strict --enforce-gate`)

Einzelne Master-Ausfuehrung:

```powershell
.\venv\Scripts\python execute_agentic_face_finetune_pipeline.py --benchmark-profile strict --retry-on-fail
```

Master-Prompt-Datei fuer Agenten:

- `agentic_face_finetune_pipeline.md`

### Real-World-Evaluation (lokale Sessions)

Für reale Nutzungsbedingungen (Lichtwechsel, Occlusion, Kopfpose, Hintergrundgeräusch) gibt es jetzt ein JSONL-Schema und einen dedizierten Auswerter:

- Schema: `benchmarks/real_world_eval_schema.json`
- Auswerter: `benchmarks/real_world_eval.py`

Beispiel:

```powershell
.\venv\Scripts\python benchmarks\real_world_eval.py --glob "benchmarks/results/real_world/*.jsonl"
```

Output:

- `benchmarks/results/real_world_eval_latest.json`

Der Report enthält:

- `metrics`: Accuracy, Macro-F1, Weighted-Score, Index
- `uncertainty`: Low-Confidence-Rate, Fehlerquote bei low/high confidence, ECE, Guardrail-Aktivierungsrate
- `scenarios`: Aufschlüsselung pro Real-World-Szenario

Zusatz in der `reference_suite.py`:

- Die Suite ergänzt jetzt unter `extensions.real_world_uncertainty` automatisch Real-World- und Unsicherheitsmetriken,
  wenn JSONL-Dateien vorhanden sind (Default-Glob: `benchmarks/results/real_world/*.jsonl`).

```powershell
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick --real-world-glob "benchmarks/results/real_world/*.jsonl"
```

### Audio-Robustheit (SNR-Sweep)

Für P5.4 gibt es einen synthetischen SNR-Sweep, der Audioqualität und dynamisches Audio-Fusionsgewicht ausweist:

```powershell
.\venv\Scripts\python benchmarks\audio_noise_robustness.py
```

Output:

- `benchmarks/results/audio_noise_robustness.json`

---

## Licht-Mapping

Das System verwendet standardmäßig das **Valence-Arousal-Modell** (kontinuierliche Farbinterpolation):

| Dimension                     | Bereich     | → Lichteffekt                                      |
| ----------------------------- | ----------- | -------------------------------------------------- |
| **Valence** (positiv/negativ) | −1.0 … +1.0 | Blau (neg.) ↔ Warmweiß (neutral) ↔ Warmgelb (pos.) |
| **Arousal** (Aktivierung)     | −1.0 … +1.0 | Dunkel/gedämpft (niedrig) ↔ Hell/gesättigt (hoch)  |

| Emotion  | Valence | Arousal | Lichtcharakter           |
| -------- | ------- | ------- | ------------------------ |
| happy    | +0.9    | +0.5    | Warm, hell, lebendig     |
| sad      | −0.8    | −0.7    | Blau, dunkel, gedämpft   |
| angry    | −0.9    | +1.0    | Rot, sehr hell, intensiv |
| fear     | −0.7    | +0.8    | Violett, hell, unruhig   |
| surprise | +0.3    | +0.9    | Orange, sehr hell        |
| disgust  | −0.6    | +0.2    | Grünlich, mittelmäßig    |
| neutral  | 0.0     | 0.0     | Warmweiß, ruhig          |

Das direkte Emotion→Farbe-Mapping bleibt als Fallback erhalten (`USE_VALENCE_AROUSAL = False`).

---

## Konfigurationsparameter (`config.py`)

### Hardware & Capture

| Konstante                 | Default            | Beschreibung                             |
| ------------------------- | ------------------ | ---------------------------------------- |
| `WEBCAM_INDEX`            | `0`                | Device-Index der USB-Webcam              |
| `HUE_BRIDGE_IP`           | `"192.168.178.20"` | **Pflichtfeld ändern!**                  |
| `HUE_LIGHT_IDS`           | `[2,3,4,6]`        | Liste der zu steuernden Lampen-IDs       |
| `FRAME_WIDTH/HEIGHT`      | `640/480`          | Auflösung des Kamera-Feeds               |
| `ANALYSIS_EVERY_N_FRAMES` | `5`                | Jeden n-ten Frame analysieren            |
| `CAMERA_BUFFER_SIZE`      | `1`                | Kamera-Puffer (1 = immer neuester Frame) |

### DeepFace & Bildverarbeitung

| Konstante             | Default        | Beschreibung                                    |
| --------------------- | -------------- | ----------------------------------------------- |
| `DETECTOR_BACKEND`    | `"retinaface"` | Face-Detektor (`retinaface`, `opencv`, `mtcnn`) |
| `MIN_CONFIDENCE`      | `0.45`         | Minimale Erkennungskonfidenz (0–1)              |
| `ANALYSIS_FRAME_SIZE` | `224`          | Frame-Breite vor Analyse (px)                   |
| `CLAHE_CLIP_LIMIT`    | `2.0`          | Beleuchtungsnormalisierung (0 = aus)            |

### EMA-Smoothing

| Konstante        | Default | Beschreibung                                              |
| ---------------- | ------- | --------------------------------------------------------- |
| `EMA_ALPHA`      | `0.15`  | Basis-Gewicht neuer Messung (wird mit Konfidenz skaliert) |
| `EMA_MIN_WEIGHT` | `0.05`  | Emotionen unter diesem EMA-Wert werden ignoriert          |
| `FALLBACK_DECAY` | `0.08`  | EMA-Drift Richtung neutral bei fehlendem Gesicht          |

### Erweiterte Analyse

| Konstante                | Default | Beschreibung                                  |
| ------------------------ | ------- | --------------------------------------------- |
| `TREND_INFLUENCE`        | `0.3`   | Einfluss des Emotionstrends auf Übergangszeit |
| `BURST_CONFIDENCE_DELTA` | `0.25`  | Konfidenzsprung der Burst-Modus auslöst       |
| `BURST_FRAMES`           | `5`     | Anzahl Frames im Burst-Modus                  |
| `USE_VALENCE_AROUSAL`    | `True`  | Valence-Arousal-Modell statt direktem Mapping |

### Audio & Pose

| Konstante             | Default | Beschreibung                       |
| --------------------- | ------- | ---------------------------------- |
| `USE_AUDIO`           | `True`  | Audio-Emotionserkennung aktivieren |
| `AUDIO_DEVICE_INDEX`  | `None`  | Mikrofon-Index (None = Standard)   |
| `AUDIO_CHUNK_SECONDS` | `2.0`   | Länge eines Audio-Analyse-Chunks   |
| `AUDIO_EMA_ALPHA`     | `0.12`  | EMA-Trägheit für Audio             |
| `AUDIO_WEIGHT`        | `0.35`  | Anteil Audio in der Fusion (35%)   |
| `USE_POSE`            | `True`  | Körpersprache-Analyse aktivieren   |
| `POSE_WEIGHT`         | `0.2`   | Einfluss des Pose-Arousal-Offsets  |

### Kalibrierung & Hue

| Konstante                         | Default                      | Beschreibung                          |
| --------------------------------- | ---------------------------- | ------------------------------------- |
| `CALIBRATION_FILE`                | `"calibration_default.json"` | Pfad der Kalibrierungsdatei           |
| `CALIBRATION_SECONDS_PER_EMOTION` | `10`                         | Sekunden pro Emotion bei Kalibrierung |
| `TRANSITION_TIME`                 | `20`                         | Farbübergang in 1/10 s → 2,0 s        |
| `FALLBACK_AFTER_SECONDS`          | `8`                          | Sek. ohne Gesicht → Fallback-Licht    |

---

## Architektur

```
main.py            ← Hauptprogramm (Capture-Loop, Fusion, Overlay, Hue-Steuerung)
config.py          ← Alle Konfigurationsparameter & Emotion-Mappings
audio_analyzer.py  ← Audio-Emotionserkennung (SpeechBrain, Hintergrund-Thread)
pose_analyzer.py   ← Körpersprache-Analyse (MediaPipe Pose, Hintergrund-Thread)
calibration.py     ← Interaktive Nutzer-Kalibrierung
requirements.txt
setup.ps1          ← Windows-Einrichtungsskript
```

Threads während des Betriebs:

- **Main Thread** – OpenCV Capture + Fusion + Overlay + Hue-Steuerung
- **Analysis Thread** – DeepFace Inference (CPU/GPU)
- **Audio Thread** – SpeechBrain Mikrofon-Analyse (optional)
- **Pose Thread** – MediaPipe Pose (optional)

---

## Troubleshooting

### "Hue-Bridge-Verbindung fehlgeschlagen"

→ Bridge-Button einmal kurz drücken, dann Programm sofort neu starten (Pairing innerhalb 30 s).

### "Webcam 0 nicht verfügbar"

→ Webcam-Index anpassen: `WEBCAM_INDEX = 1` (oder 2, …) in `config.py`.

### Audio- oder Pose-Modul startet nicht

→ Die Module starten automatisch ohne Audio/Pose wenn die Abhängigkeiten fehlen. Fehler werden geloggt. Manuell testen:

```powershell
.\venv\Scripts\python -c "import sounddevice; import speechbrain; print('Audio OK')"
.\venv\Scripts\python -c "import mediapipe; print('Pose OK')"
```

### Langsame Emotionserkennung / CPU-Last hoch

→ `ANALYSIS_EVERY_N_FRAMES` erhöhen (z.B. `8`), `--no-audio --no-pose` verwenden, oder `DETECTOR_BACKEND = "opencv"` setzen.

### GPU-Beschleunigung (NVIDIA)

```powershell
.\venv\Scripts\pip install "tensorflow[and-cuda]"
```

### Erste Ausführung dauert lang

→ Normal. DeepFace lädt beim ersten Start RetinaFace-Modell (~120 MB) und das Emotions-Modell. SpeechBrain lädt beim ersten Audio-Start das wav2vec2-Modell (~400 MB).

### EMA reagiert zu träge / zu nervös

→ `EMA_ALPHA` erhöhen (reaktiver) oder verringern (träger). Empfohlener Bereich: 0.05–0.30.

---

## Lizenz

DIY-Projekt – keine Garantie. Verwende es nach eigenem Ermessen.
