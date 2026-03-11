import importlib
import sys
import time
import types

from config import HUE_MIN_UPDATE_INTERVAL


def import_main_with_bridge_stub(bridge_cls):
    deepface_module = types.ModuleType("deepface")

    class _DeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            return []

    setattr(deepface_module, "DeepFace", _DeepFace)

    phue_module = types.ModuleType("phue")
    setattr(phue_module, "Bridge", bridge_cls)

    sys.modules["deepface"] = deepface_module
    sys.modules["phue"] = phue_module

    # Clear main and extracted sub-modules so they pick up the stubs.
    for mod_name in list(sys.modules):
        if mod_name == "main" or mod_name.startswith(("core.", "analyzers.")):
            del sys.modules[mod_name]

    return importlib.import_module("main")


def test_hue_apply_is_non_blocking_with_slow_bridge_io():
    class SlowBridge:
        def __init__(self, ip):
            self.ip = ip

        def connect(self):
            return None

        def set_light(self, lid, payload, value=None):
            # Simulate slow network writes only for full command payloads.
            if isinstance(payload, dict):
                time.sleep(0.25)
            return None

    main = import_main_with_bridge_stub(SlowBridge)
    hue = main.HueController("127.0.0.1", [1], {1: "primary"})

    start = time.perf_counter()
    hue.apply({"hue": 14000, "bri": 120, "sat": 120}, transition=2)
    elapsed = time.perf_counter() - start

    hue.shutdown()

    # apply() should enqueue and return quickly, not block on network I/O.
    assert elapsed < 0.08


def test_hue_sender_queue_eventually_applies_latest_command():
    class RecordingBridge:
        def __init__(self, ip):
            self.ip = ip
            self.sent_hues = []

        def connect(self):
            return None

        def set_light(self, lid, payload, value=None):
            if isinstance(payload, dict):
                self.sent_hues.append(int(payload.get("hue", -1)))
                time.sleep(0.08)
            return None

    main = import_main_with_bridge_stub(RecordingBridge)
    hue = main.HueController("127.0.0.1", [1], {1: "primary"})

    hue.apply({"hue": 1000, "bri": 100, "sat": 100}, transition=1)
    time.sleep(HUE_MIN_UPDATE_INTERVAL + 0.03)
    hue.apply({"hue": 12000, "bri": 110, "sat": 110}, transition=1)
    time.sleep(HUE_MIN_UPDATE_INTERVAL + 0.03)
    hue.apply({"hue": 30000, "bri": 130, "sat": 120}, transition=1)

    time.sleep(0.35)
    hue.shutdown()

    assert len(hue.bridge.sent_hues) >= 1
    assert hue.bridge.sent_hues[-1] == 30208  # quantized by HUE_HUE_QUANT=512
