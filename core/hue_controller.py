"""Async Philips Hue controller with role-based scene composition."""

import logging
import queue
import threading
import time

from phue import Bridge

from config import (
    HUE_BRI_QUANT,
    HUE_CT_QUANT,
    HUE_HUE_QUANT,
    HUE_MIN_UPDATE_INTERVAL,
    HUE_SAT_QUANT,
    TRANSITION_TIME,
)
from core.error_taxonomy import HUE_OFF_FAILED, HUE_SEND_FAILED
from core.light_mapping import compose_multi_light_scene
from core.telemetry import ERR_TELEMETRY, RuntimeErrorTelemetry

log = logging.getLogger("emotion-light")


class HueController:
    """Steuert mehrere Philips Hue Lampen mit rollenbasierter Szenen-Komposition."""

    def __init__(
        self,
        ip: str,
        light_ids: list,
        light_roles: dict | None = None,
        *,
        telemetry: RuntimeErrorTelemetry | None = None,
    ):
        self._telemetry = telemetry or ERR_TELEMETRY
        self.bridge = Bridge(ip)
        self.bridge.connect()
        self.lids = light_ids
        self._roles = light_roles or {lid: "primary" for lid in light_ids}
        self._last_cmd: dict = {}
        self._last_sent_ts = 0.0
        self._send_q: queue.Queue = queue.Queue(maxsize=1)
        self._sender_running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        # Alle Lampen einschalten
        for lid in self.lids:
            self.bridge.set_light(lid, "on", True)
        log.info("Hue Bridge verbunden, %d Lampen aktiviert: %s", len(self.lids), self.lids)

    def _record(self, **kwargs):
        if self._telemetry is not None:
            self._telemetry.record(**kwargs)

    def _sender_loop(self):
        """Sendet Hue-Befehle asynchron, damit der Main-Loop nicht auf I/O wartet."""
        while self._sender_running:
            try:
                item = self._send_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            all_cmds, transition = item
            for lid in self.lids:
                try:
                    cmd = all_cmds[lid].copy()
                    cmd["transitiontime"] = transition
                    self.bridge.set_light(lid, cmd)
                except Exception as e:
                    self._record(
                        component="hue",
                        code=HUE_SEND_FAILED,
                        detail=f"set_light failed for lid={lid}",
                        exc=e,
                        level=logging.ERROR,
                        cooldown_s=2.0,
                    )

    def _enqueue_latest(self, all_cmds: dict, transition: int) -> None:
        """Haelt nur den neuesten ausstehenden Hue-Befehl in der Queue."""
        try:
            while not self._send_q.empty():
                self._send_q.get_nowait()
            self._send_q.put_nowait((all_cmds, transition))
        except queue.Full:
            pass

    @staticmethod
    def _quantize(value: int, step: int, min_value: int, max_value: int) -> int:
        if step <= 1:
            return max(min_value, min(max_value, int(value)))
        q = int(round(float(value) / step) * step)
        return max(min_value, min(max_value, q))

    def apply(self, params: dict, transition: int = TRANSITION_TIME):
        """Setzt Lichtparameter auf allen Lampen mit rollenbasierter Anpassung.

        Wenn ``ct`` in *params* vorhanden ist (VA-Modell), werden
        ``ct`` + ``bri`` gesendet (White-Ambiance- und Farb-Lampen-kompatibel).
        Andernfalls wird der klassische ``hue`` + ``sat`` + ``bri`` Modus verwendet.
        """
        now = time.time()
        if now - self._last_sent_ts < HUE_MIN_UPDATE_INTERVAL:
            return

        ct_mode = "ct" in params
        if ct_mode:
            primary_cmd = {
                "ct": self._quantize(params["ct"], HUE_CT_QUANT, 153, 500),
                "bri": self._quantize(params["bri"], HUE_BRI_QUANT, 1, 254),
            }
        else:
            primary_cmd = {
                "hue": self._quantize(params["hue"], HUE_HUE_QUANT, 0, 65535),
                "bri": self._quantize(params["bri"], HUE_BRI_QUANT, 1, 254),
                "sat": self._quantize(params["sat"], HUE_SAT_QUANT, 0, 254),
            }

        # Erzeuge ein Cache-Key aus allen Lampen-Parametern
        all_cmds = {}
        for lid in self.lids:
            role = self._roles.get(lid, "primary")
            role_params = compose_multi_light_scene(primary_cmd, role)
            if ct_mode:
                all_cmds[lid] = {
                    "ct": self._quantize(role_params["ct"], HUE_CT_QUANT, 153, 500),
                    "bri": self._quantize(role_params["bri"], HUE_BRI_QUANT, 1, 254),
                }
            else:
                all_cmds[lid] = {
                    "hue": self._quantize(role_params["hue"], HUE_HUE_QUANT, 0, 65535),
                    "bri": self._quantize(role_params["bri"], HUE_BRI_QUANT, 1, 254),
                    "sat": self._quantize(role_params["sat"], HUE_SAT_QUANT, 0, 254),
                }

        if all_cmds == self._last_cmd:
            return
        self._last_cmd = all_cmds
        self._last_sent_ts = now
        self._enqueue_latest(all_cmds, transition)

    def off(self):
        """Schaltet alle Lampen aus."""
        for lid in self.lids:
            try:
                self.bridge.set_light(lid, "on", False)
            except Exception as e:
                self._record(
                    component="hue",
                    code=HUE_OFF_FAILED,
                    detail=f"off failed for lid={lid}",
                    exc=e,
                    level=logging.ERROR,
                    cooldown_s=2.0,
                )

    def shutdown(self):
        """Beendet den Sender-Thread kontrolliert."""
        self._sender_running = False
        try:
            self._send_q.put_nowait(None)
        except queue.Full:
            try:
                self._send_q.get_nowait()
                self._send_q.put_nowait(None)
            except Exception:
                pass
        try:
            self._sender_thread.join(timeout=1.5)
        except Exception:
            pass


class MockBridgeController:
    """Simuliert HueController ohne echte Hardware."""

    def __init__(self):
        log.info("[MOCK] HueController aktiv – keine echte Bridge.")

    def apply(self, params: dict, transition: int = 0):
        log.info(
            "[MOCK] Hue-Befehl: hue=%d bri=%d sat=%d",
            params["hue"],
            params["bri"],
            params["sat"],
        )

    def off(self):
        log.info("[MOCK] Lampe ausschalten simuliert.")

    def shutdown(self):
        return None
