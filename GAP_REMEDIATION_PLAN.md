# Gap Remediation Plan

Stand: 2026-03-08
Status: In Progress (P0 active)

## Ziel
Systematische Abarbeitung der identifizierten Luecken nach Impact, mit messbarer Verifikation ueber die Reference Suite.

## Arbeitsprinzip
1. Fuer jedes Arbeitspaket: Codeaenderung -> lokale Verifikation -> quick profile.
2. Keine Abschlussmeldung ohne aktualisierten JSON-Report in `benchmarks/results/reference_suite_latest.json`.
3. Bei Regression: sofort beheben, dann erneut verifizieren.

## Backlog Nach Prioritaet

### P0 - Governance und Gate-Haerte
- [x] P0.1 Gate darf bei Benchmark-Schema-Mismatch nicht mehr als PASS durchgehen.
- [x] P0.2 Gate-Reasons klar trennen in `warnings` vs `failures`.
- [x] P0.3 Doku fuer Baseline-Refresh-Workflow ergaenzen.

### P1 - Laufzeit-Sicherheit und Integritaet
- [x] P1.1 Runtime-Downloads absichern (Datei-Integritaet/Hash-Pruefung).
- [x] P1.2 Globale Monkeypatches in `audio_analyzer.py` eliminieren oder kapseln.
- [x] P1.3 Hue-I/O gegen Blockaden haerten (Timeout/Retry/Entkopplung).

### P2 - Observability und Fehlerbild
- [x] P2.1 Breite `except Exception`-Pfade strukturieren und Telemetrie verbessern.
- [x] P2.2 Fehlercodes/Fehlerkategorien fuer Hauptpfade vereinheitlichen.

### P3 - Testabdeckung
- [x] P3.1 Tests fuer Benchmark-Gate-Logik (Schema-Mismatch, Regression, Pass).
- [x] P3.2 Tests fuer Audio/Pose/Hue-Fehlerpfade ergaenzen.
- [x] P3.3 Integrationsnahe Main-Loop-Safeguards testen.

### P4 - Konfig-/Privacy-Hygiene
- [x] P4.1 Sensible lokale Defaults aus `config.py` in `.env` oder lokale Override-Datei verschieben.
- [x] P4.2 Session-Logs optional pseudonymisieren.

### P5 - Genauigkeit und Unsicherheitssteuerung (Neu)
- [x] P5.1 Real-World-Eval-Set aufbauen (eigene, lokale Sessions mit Einwilligung; verschiedene Lichtlagen, Kopfrotation, Teilverdeckung, Hintergrundgeraesch).
- [x] P5.2 Unsicherheitsbudget einfuehren (z. B. entropy/margin-basierte confidence quality), damit bei niedriger Sicherheit konservativere Hue-Updates erfolgen.
- [x] P5.3 Fehlreaktionen reduzieren: Guardrail-Policy fuer low-confidence (hold/slow-fade/neutral-safe-color statt harter Umspruenge).
- [x] P5.4 Audio-Robustheit gegen Geraeusch testen (SNR-Stufen, Hintergrundquellen) und fusion weight dynamisch an Qualitaet koppeln.
- [x] P5.5 Benchmark-Erweiterung: `reference_suite.py` um realnahe Szenarien und Unsicherheitsmetriken erweitern.

### P6 - Wirksamkeit, UX-Opt-out und Betrieb auf schwacher Hardware (Neu)
- [ ] P6.1 Manual-Mood-Override als Opt-out einbauen (CLI + Runtime-Toggle), damit Nutzer die automatische Erkennung jederzeit uebersteuern koennen.
- [ ] P6.2 Wirkungsmessung fuer Hobby-Evaluation definieren (kurze vor/nach Selbsteinschaetzung, optional anonymisiert) fuer nachvollziehbare Trenddaten.
- [ ] P6.3 Transparenzhinweise in README/UI erweitern (Grenzen der Emotionserkennung, keine medizinische Wirkung zugesichert).
- [ ] P6.4 Datenschutz-Controls erweitern: explizites Clear-Command fuer lokale Artefakte (Kalibrierung/Logs), klare Speicherorte dokumentieren.
- [ ] P6.5 Low-End-Profil implementieren und testen (niedrigere Aufloesung, adaptive Framerate, modulweises Auto-Disable, CPU-Schutzschwellen).

### P7 - Agentic Face Fine-Tuning Pipeline (Neu)
- [x] P7.1 Vollautomatisierte Pipeline-Datei erstellt (`agentic_face_finetune_pipeline.md`).
- [x] P7.2 Agentic Dataset-Generator erstellt (`agentic_dataset_gen.py`) mit Sample-Gate >=1000.
- [x] P7.3 Agentic Fine-Tune-Script erstellt (`finetune_face_agentic.py`) mit ONNX-Export + `config.py` Update.
- [x] P7.4 Master-Orchestrator erstellt (`execute_agentic_face_finetune_pipeline.py`) inkl. Retry mit `lr/2`.

## Akzeptanzkriterien fuer neue Arbeitspakete
- P5 gilt als abgeschlossen, wenn ein reproduzierbarer Real-World-Report in `benchmarks/results/` vorliegt und Unsicherheits-Guardrails Fehlreaktionsrate sichtbar senken, ohne `composite_index` im `standard`-Profil zu verschlechtern.
- P6 gilt als abgeschlossen, wenn Manual-Override, Clear-Command und Low-End-Profil dokumentiert und ueber Tests plus Reference-Suite-Lauf (`standard --enforce-gate`) verifiziert sind.
- Fuer alle neuen Pakete gilt: keine Behauptung ueber Verbesserung ohne neuen `reference_suite_latest.json` und dokumentierte Deltas gegen den letzten vergleichbaren Run.

## Messplan (Delta-orientiert)
- Kernmetrik: `composite_index` + Component-Deltas aus `reference_suite_latest.json`.
- Zusatzmetrik P5: Fehlreaktionsrate bei low-confidence, Anteil Guardrail-Aktivierungen, Modalitaetsqualitaet (video/audio/pose).
- Zusatzmetrik P6: User-Override-Nutzungsrate, mittlere Latenz unter Low-End-Profil, Session-Abbruchrate bei CPU-Last.
- Stabilitaet: Deltas immer gegen den letzten vergleichbaren Lauf (gleiches Profil, gleicher Detector), inkl. CI95 aus Multi-Seed-Stats.

## Ausfuehrungsprotokoll
- [x] Baseline vor Implementierung ausgefuehrt: `reference_suite.py --profile quick` (Composite 519, Gate FAIL).
- [x] Schritt 1 abgeschlossen: P0.1 + P0.2 in `benchmarks/reference_suite.py` umgesetzt.
- [x] Verifikation nach Aenderung: `reference_suite.py --profile quick` (Composite 519, Gate FAIL aus inhaltlichen Regressionsgruenden; kein Schema-Mismatch-Skip).
- [x] Schritt 2 abgeschlossen: P0.3 in `benchmarks/REFERENCE_BENCHMARK_PROTOCOL.md` dokumentiert.
- [x] Schritt 3 abgeschlossen: P1.1 mit `asset_integrity.py` + Analyzer/Benchmark-Integration.
- [x] Schritt 4 abgeschlossen: P1.2 Audio-Patches nur temporaer und rueckgesetzt.
- [x] Schritt 5 abgeschlossen: P1.3 Hue-Apply asynchron ueber Sender-Thread entkoppelt.
- [x] Schritt 6 abgeschlossen: P3.1 Gate-Tests angelegt (`tests/test_reference_suite_gate.py`).
- [x] Schritt 7 abgeschlossen: P2.1 RuntimeErrorTelemetry in `main.py` integriert (Startup/Hue/DeepFace/Kalibrierung).
- [x] Schritt 8 abgeschlossen: P3.2 Fehlerpfad-Tests (`tests/test_runtime_error_telemetry.py`) + Integration in Suite.
- [x] Schritt 9 abgeschlossen: P2.2 Shared Taxonomie in `error_taxonomy.py` + Verwendung in `main.py`.
- [x] Schritt 10 abgeschlossen: P3.3 mit `tests/test_main_loop_safeguards.py` umgesetzt.
- [x] Schritt 11 abgeschlossen: P4.2 per `--pseudonymize-session` + `SESSION_LOG_SALT` umgesetzt und getestet.
- [x] Schritt 12 abgeschlossen: P4.1 via `config_local.py`-Override + Beispiel/README/Test umgesetzt.
- [x] Schritt 13 abgeschlossen: P5.1 mit `benchmarks/real_world_eval_schema.json`, `benchmarks/real_world_eval.py` und `tests/test_real_world_eval.py` umgesetzt.
- [x] Schritt 14 abgeschlossen: P5.2/P5.3 mit Modellqualitaet (Entropy+Margin), LOW-Q-Guardrail und Session-Log-Feldern (`model_quality`, `low_quality_guardrail`) umgesetzt.
- [x] Schritt 15 abgeschlossen: P5.4 Audio-SNR-Qualitaet + dynamisches Audio-Fusionsgewicht (`audio_quality.py`, `audio_analyzer.py`, `main.py`, `benchmarks/audio_noise_robustness.py`, `tests/test_audio_quality.py`).
- [x] Schritt 16 abgeschlossen: P5.5 umgesetzt (`reference_suite.py` erweitert um `extensions.real_world_uncertainty` + CLI-Optionen + Tests).
- [ ] Schritt 17 in Arbeit: visuelle Robustheit verbessert via robuster Enhanced-Inferenz (`benchmarks/accuracy_benchmark.py`: Preprocess-Varianten + selektiver Detector-Fallback).
- [x] Schritt 18 abgeschlossen: Agentic Face-Fine-Tune-Pipeline als end-to-end Workflow (Dataset -> Train -> Benchmark Gate) integriert.
- [x] Verifikation Schritt 18 (quick): `reference_suite.py --profile quick` -> Composite 533, Gate FAIL (nur extreme_visual_robustness unter Baseline-Schwelle).
- [x] Verifikation Schritt 18 (standard gate): `reference_suite.py --profile standard --enforce-gate` -> Composite 533, Gate PASS.
- [x] Verifikation Schritt 18 (strict gate): `reference_suite.py --profile strict --enforce-gate` -> Composite 541, Gate PASS.
- [x] Verifikation Schritt 14: `pytest tests/test_main_overlay.py tests/test_uncertainty_guardrail.py` + `reference_suite.py --profile quick` (Composite 519, test_quality 1000, e2e_runtime 937).
- [x] Verifikation Schritt 15: `pytest tests/test_audio_quality.py tests/test_light_mapping.py tests/test_main_overlay.py tests/test_uncertainty_guardrail.py` + `benchmarks/audio_noise_robustness.py` (monotonic dynamic weight = true).
- [x] Verifikation Schritt 16: `pytest tests/test_reference_suite_real_world_extension.py tests/test_audio_quality.py tests/test_real_world_eval.py` + `reference_suite.py --profile quick` (Composite 519, extension `real_world_uncertainty` aktiv/reportiert).
- [x] Verifikation Schritt 17 (Zwischenstand): `pytest tests/test_accuracy_robust_fallback.py tests/test_light_mapping.py` + `reference_suite.py --profile quick` mit Delta `extreme_visual_robustness 151 -> 184` und `composite 519 -> 533`.
- [x] Schritt-17-Tuningversuch B (mehr aggressive Preprocess-Kandidaten) getestet und wegen Regression verworfen (`extreme_visual_robustness 184 -> 137`).
- [x] Schritt-17-Stand wiederhergestellt auf bestes bekanntes Ergebnis nach Rollback (`reference_suite.py --profile quick`: Composite 533, Extreme 184, Test-Quality 1000).
- [x] Benchmark-Nachlauf nach Plan-Update: `reference_suite.py --profile quick` (Composite 519, Trend +0, Gate FAIL wegen bekannter Baseline-Regressionsgrenze).
- [x] Verifikation Schritt 15 (Zwischenstand): `reference_suite.py --profile quick` mehrfach stabil bei Composite 519; `delta_score_with_minus_without_face_mesh` auf `0.0` reduziert.
- [x] Standard-Check aktualisiert: `reference_suite.py --profile standard --enforce-gate` -> Composite 527, Gate PASS, Trend +4 vs vorherigem comparable Standard-Lauf.
- [ ] Schritt 15 bleibt aktiv: `extreme_visual_robustness` weiterhin niedrig (quick idx 151 / standard idx 248), weitere gezielte Robustheitsarbeit offen.
