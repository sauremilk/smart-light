"""Zirkadianes Licht-Modell fuer tageszeitadaptive Zielzustaende.

Passt die Regulations-Zielwerte und Hue-Interpolationsbereiche an die
aktuelle Tageszeit an. Basiert auf chronobiologischer Forschung:
Morgens kuehler/aktivierender, abends waermer/ruhiger.
"""

from datetime import datetime

# Tageszeit-Profile fuer Fokus-Kontext
# (start_hour, end_hour, target_v, target_a, hue_neg, hue_pos, bri_max,
#  ct_neg, ct_pos, label)
# ct in Mirek: 153=6500 K kalt, 500=2000 K warm
_PROFILES = [
    (6, 10, 0.60, 0.40, 9500, 13000, 220, 410, 250, "Aufwachen"),
    (10, 13, 0.65, 0.35, 9000, 11000, 240, 380, 222, "Fokus-Peak"),
    (13, 15, 0.70, 0.45, 9000, 12000, 240, 380, 235, "Anti-Mittagstief"),
    (15, 18, 0.65, 0.30, 9500, 13000, 230, 400, 250, "Nachm.-Fokus"),
    (18, 22, 0.60, 0.00, 8500, 15000, 180, 440, 310, "Entspannung"),
    (22, 24, 0.50, -0.20, 7500, 12000, 100, 470, 370, "Schlaf-Vorbereitung"),
    (0, 6, 0.50, -0.20, 7500, 12000, 100, 470, 370, "Nacht"),
]


class CircadianSchedule:
    """Liefert tageszeit-abhaengige Licht-Zielwerte."""

    def get_params(self, hour: int | None = None) -> dict:
        """Gibt Zielparameter fuer die aktuelle Stunde zurueck.

        Parameters
        ----------
        hour : int or None
            Stunde (0-23). None = aktuelle Uhrzeit.

        Returns
        -------
        dict mit Schluesseln:
          target_v, target_a  – Regulations-Zielwerte
          hue_negative        – VA_HUE_NEGATIVE Override
          hue_positive        – VA_HUE_POSITIVE Override
          bri_max             – maximale Helligkeit
          label               – menschlesbare Bezeichnung
        """
        if hour is None:
            hour = datetime.now().hour

        for start, end, tv, ta, hn, hp, bm, cn, cp, label in _PROFILES:
            if start <= hour < end:
                return {
                    "target_v": tv,
                    "target_a": ta,
                    "hue_negative": hn,
                    "hue_positive": hp,
                    "bri_max": bm,
                    "ct_negative": cn,
                    "ct_positive": cp,
                    "label": label,
                }

        # Fallback (sollte nie erreicht werden)
        return {
            "target_v": 0.60,
            "target_a": 0.00,
            "hue_negative": 9000,
            "hue_positive": 14000,
            "bri_max": 180,
            "ct_negative": 450,
            "ct_positive": 250,
            "label": "Standard",
        }
