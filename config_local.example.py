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
