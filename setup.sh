#!/usr/bin/env bash
#
# Einrichtungsskript fuer das Emotion-Light-Projekt (Linux / macOS).
# Erstellt das venv, installiert Abhaengigkeiten und fuehrt einen Syntax-Check durch.
#
# Verwendung:
#   bash setup.sh
#   bash setup.sh --force   # Loescht und erstellt das venv neu
#
set -euo pipefail

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        *) echo "Unbekannte Option: $arg"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_EXE="$VENV_PATH/bin/python"
PIP_EXE="$VENV_PATH/bin/pip"
GITHOOKS_DIR="$PROJECT_ROOT/.githooks"

echo "=== Emotion-Light Setup ==="
echo "Projektordner: $PROJECT_ROOT"

# --- Python-Version pruefen (>= 3.10) ---
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "FEHLER: Python >= 3.10 nicht gefunden." >&2
    exit 1
fi
echo "Python: $PYTHON_CMD ($("$PYTHON_CMD" --version))"

# --- Venv loeschen (wenn --force) ---
if [ "$FORCE" -eq 1 ] && [ -d "$VENV_PATH" ]; then
    echo "[1/5] Altes venv wird entfernt..."
    rm -rf "$VENV_PATH"
fi

# --- Venv erstellen ---
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/5] Erstelle virtuelles Environment..."
    "$PYTHON_CMD" -m venv "$VENV_PATH"
else
    echo "[1/5] venv bereits vorhanden, ueberspringe Erstellung."
fi

# --- Pip upgraden ---
echo "[2/5] Pip upgraden..."
"$PYTHON_EXE" -m pip install --upgrade pip --quiet

# --- Abhaengigkeiten installieren ---
echo "[3/5] Installiere Abhaengigkeiten aus pyproject.toml..."
"$PIP_EXE" install -e "$PROJECT_ROOT"

# --- Syntax-Check ---
echo "[4/5] Syntax-Check (main.py, config.py)..."
"$PYTHON_EXE" -m py_compile "$PROJECT_ROOT/main.py"
"$PYTHON_EXE" -m py_compile "$PROJECT_ROOT/config.py"

# --- Git-Hooks installieren ---
echo "[5/5] Konfiguriere lokale Git-Hooks..."
if [ -d "$GITHOOKS_DIR" ]; then
    git -C "$PROJECT_ROOT" config core.hooksPath .githooks
    echo "Git-Hooks aktiviert (core.hooksPath=.githooks)."
else
    echo "Hinweis: .githooks nicht gefunden, Hook-Installation uebersprungen."
fi

echo ""
echo "=== Setup abgeschlossen! ==="
echo ""
echo "Naechste Schritte:"
echo "  1. config_local.example.py nach config_local.py kopieren:"
echo "       cp config_local.example.py config_local.py"
echo "  2. config_local.py anpassen: HUE_BRIDGE_IP = '<deine-bridge-ip>'"
echo ""
echo "Starten (mit echter Hardware):"
echo "  .venv/bin/python main.py"
echo ""
echo "Starten (Mock-Modus, kein Hardware noetig):"
echo "  .venv/bin/python main.py --mock"
echo ""
echo "Weitere Optionen:"
echo "  .venv/bin/python main.py --help"
