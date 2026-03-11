# Alexa Integration Dokumentation

## Status

Diese Integration ist implementiert und optional aktivierbar.

- Hue-Steuerung bleibt wie bisher der Hauptpfad.
- Alexa-Steuerung laeuft zusaetzlich und beeinflusst nur Audio-Verhalten (z. B. Musik/Lautstaerke), nicht die Hue-Befehle.
- Wenn Alexa nicht verfuegbar ist oder Login fehlschlaegt, laeuft die App weiter.

## Implementierter Umfang

### 1. Neuer Controller

Datei: `core/alexa_controller.py`

Enthaelt:

- `AlexaController` (asynchroner Thread mit eigenem `asyncio`-Loop)
- Valence/Arousal -> Mood-Mapping (`_va_to_mood`)
- Valence/Arousal -> Lautstaerke-Mapping (`_va_to_volume`)
- E-Mail-Masking fuer Logs (`_hide_email`)

Mood-Kategorien:

- `energetic_positive`
- `calm_positive`
- `neutral`
- `calm_negative`
- `energetic_negative`

### 2. Konfiguration

Dateien:

- `config.py`
- `config_local.example.py`

Neue Parameter:

- `USE_ALEXA`
- `ALEXA_EMAIL`
- `ALEXA_PASSWORD`
- `ALEXA_DEVICE_NAME`
- `ALEXA_AMAZON_URL`
- `ALEXA_COOLDOWN_SECONDS`
- `ALEXA_MUSIC_PROVIDER`
- `ALEXA_VOLUME_CONTROL`
- `ALEXA_MOOD_PLAYLISTS`

`config.py` unterstuetzt lokale Overrides ueber `config_local.py`.

### 3. Runtime-Integration

Datei: `main.py`

Erweitert um:

- Import der Alexa-Config-Werte
- Initialisierung des Alexa-Controllers beim Start
- Zyklisches `alexa_controller.update(fused_v, fused_a, emotion)` im Main-Loop
- Sauberes `shutdown()` im `finally`-Block
- CLI-Flag `--no-alexa`

### 4. Abhaengigkeit

Datei: `requirements.txt`

- `alexapy>=1.28.0`

## Sicherheit und Geheimnisse

### Wo Credentials liegen

Nur in `config_local.py` (lokal, nicht versioniert).

### Git-Schutz

- `config_local.py` ist in `.gitignore`.
- Die Datei ist nicht getrackt und war nicht in der Git-Historie.

### Wichtige Regeln

- Niemals Zugangsdaten in `config.py`, `README.md` oder Commits schreiben.
- Logs maskieren E-Mail-Adressen.
- Passwort nie im Klartext in Issues/PRs/Chats posten.

## Betrieb

## Aktivieren

In `config_local.py`:

```python
USE_ALEXA = True
ALEXA_EMAIL = "<deine amazon email>"
ALEXA_PASSWORD = "<dein amazon passwort>"
ALEXA_DEVICE_NAME = "<exakter Geraetename aus der Alexa-App>"
ALEXA_AMAZON_URL = "amazon.de"
```

Start:

```powershell
c:/Users/mickg/smart-light/.venv/Scripts/python.exe main.py
```

Temporar deaktivieren:

```powershell
c:/Users/mickg/smart-light/.venv/Scripts/python.exe main.py --no-alexa
```

## Verifikationsergebnisse (aktueller Stand)

Erfolgreich verifiziert:

- ONNX-Backend Start
- Webcam-Init
- Hue-Bridge-Verbindung
- Audio/Pose/FaceMesh/HRV/Breathing Start
- Alexa-Controller Start ohne Runtime-Crash
- Main-Loop stabil

Gefundene und behobene Probleme:

1. `NameError: USE_ALEXA is not defined`
   - Ursache: Alexa-Config nicht im `from config import (...)` Block.
   - Fix in `main.py` umgesetzt.

2. `AlexaLogin.__init__() got an unexpected keyword argument 'outputfiles_prefix'`
   - Ursache: alexapy API-Version weicht ab.
   - Fix in `core/alexa_controller.py`: Signatur-Erkennung via `inspect` und dynamische Parameter.

3. `AlexaLogin.__init__() missing 1 required positional argument: 'outputpath'`
   - Ursache: installierte alexapy-Version erwartet `outputpath` Callback.
   - Fix umgesetzt: `outputpath` wird auf lokale Session-Dateien gemappt.

Aktueller offener Punkt:

- `Alexa-Login fehlgeschlagen (Status: {})`
  - Typisch bei falschem Passwort, 2FA/Captcha/OAuth-Interaktion oder Account-Sicherheitsblock.
  - Kein Code-Crash, Hue-System bleibt funktionsfaehig.

## Troubleshooting

1. Geraetename exakt pruefen

- In Alexa-App den exakten Namen kopieren.
- Muss 1:1 in `ALEXA_DEVICE_NAME` stehen.

2. Passwort/2FA pruefen

- Auf `amazon.de` interaktiv im Browser einloggen.
- Falls notwendig Passwort neu setzen.
- Danach App erneut starten.

3. Test ohne Alexa

- `main.py --no-alexa` muss stabil laufen.
- So kann klar zwischen Core-Problemen und Alexa-Login-Problemen getrennt werden.

4. Session-Dateien

- Alexa-Session-Dateien werden lokal im Projektordner abgelegt (`alexa_session_*`).
- Diese Dateien nicht committen.

## Designentscheidungen

- Alexa ist optional und darf die Kernfunktion (Emotion -> Hue) nicht blockieren.
- Asynchrone Architektur vermeidet Framedrops durch Netzwerk-/Login-Latenz.
- Fehler werden geloggt, aber nicht fatal behandelt.

## Nicht-Ziele

- Keine Garantie fuer Amazon-API-Stabilitaet (alexapy ist inoffiziell).
- Keine Speicherung von Secrets im Repository.
