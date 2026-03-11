import importlib
import sys
import time
import types


def import_main_with_stubs():
    deepface_module = types.ModuleType("deepface")

    class _DeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            return []

    setattr(deepface_module, "DeepFace", _DeepFace)

    phue_module = types.ModuleType("phue")

    class _Bridge:
        def __init__(self, ip):
            self.ip = ip
            self.fail_on_off = False
            self.fail_on_send = False

        def connect(self):
            return None

        def set_light(self, *args, **kwargs):
            # Signature variants:
            # 1) set_light(lid, "on", False)
            # 2) set_light(lid, {"hue":..., ...})
            if len(args) >= 3 and args[1] == "on" and args[2] is False and self.fail_on_off:
                raise RuntimeError("off failed")
            if len(args) >= 2 and isinstance(args[1], dict) and self.fail_on_send:
                raise RuntimeError("send failed")
            return None

    setattr(phue_module, "Bridge", _Bridge)

    sys.modules["deepface"] = deepface_module
    sys.modules["phue"] = phue_module

    # Clear main and extracted sub-modules so they pick up the stubs.
    for mod_name in list(sys.modules):
        if mod_name == "main" or mod_name.startswith(("core.", "analyzers.")):
            del sys.modules[mod_name]

    return importlib.import_module("main")


def test_load_calibration_records_telemetry_on_invalid_json(tmp_path):
    main = import_main_with_stubs()

    bad_file = tmp_path / "bad_calibration.json"
    bad_file.write_text("{invalid json", encoding="utf-8")

    key = "calibration:CALIBRATION_LOAD_FAILED"
    before = int(main.ERR_TELEMETRY.summary().get(key, 0))
    data = main.load_calibration(str(bad_file))
    after = int(main.ERR_TELEMETRY.summary().get(key, 0))

    assert data == {}
    assert after == before + 1


def test_hue_off_failure_records_telemetry():
    main = import_main_with_stubs()

    hue = main.HueController("127.0.0.1", [1], {1: "primary"})
    hue.bridge.fail_on_off = True

    key = "hue:HUE_OFF_FAILED"
    before = int(main.ERR_TELEMETRY.summary().get(key, 0))
    hue.off()
    after = int(main.ERR_TELEMETRY.summary().get(key, 0))

    hue.shutdown()

    assert after == before + 1


def test_hue_send_failure_records_telemetry():
    main = import_main_with_stubs()

    hue = main.HueController("127.0.0.1", [1], {1: "primary"})
    hue.bridge.fail_on_send = True

    key = "hue:HUE_SEND_FAILED"
    before = int(main.ERR_TELEMETRY.summary().get(key, 0))

    hue.apply({"hue": 14000, "bri": 120, "sat": 120}, transition=1)
    time.sleep(0.15)
    after = int(main.ERR_TELEMETRY.summary().get(key, 0))

    hue.shutdown()

    assert after >= before + 1
