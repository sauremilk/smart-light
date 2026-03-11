"""Lokale, nicht versionierte Ueberschreibungen fuer sensible Werte.

Vorgehen:
1. Datei nach `config_local.py` kopieren.
2. Eigene lokale Werte setzen.
3. `config_local.py` nicht committen.
"""

# Beispiel: lokale Hue-Bridge + Lampen
HUE_BRIDGE_IP = "192.168.1.100"
HUE_LIGHT_IDS = [1]
HUE_LIGHT_ROLES = {
    1: "primary",
}

# Beispiel: Alexa-Steuerung aktivieren
# Muss ausgefuellt werden bevor USE_ALEXA = True gesetzt wird.
USE_ALEXA = False
ALEXA_EMAIL = "deine@email.de"  # Amazon-Konto E-Mail
ALEXA_PASSWORD = "deinPasswort"  # Amazon-Konto Passwort
ALEXA_DEVICE_NAME = "Michs Echo"  # Gerätename exakt wie in Alexa-App
ALEXA_AMAZON_URL = "amazon.de"  # amazon.de fuer Deutschland

# Optional: eigene Playlists pro Stimmung
# ALEXA_MOOD_PLAYLISTS = {
#     "energetic_positive": "upbeat happy pop music",
#     "calm_positive":      "focus concentration background music",
#     "calm_negative":      "calming relaxing ambient music",
#     "energetic_negative": "stress relief calming music",
#     "neutral":            "lo-fi background music",
# }
