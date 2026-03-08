import importlib
import sys
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

        def connect(self):
            return None

        def set_light(self, *args, **kwargs):
            return None

    setattr(phue_module, "Bridge", _Bridge)

    sys.modules["deepface"] = deepface_module
    sys.modules["phue"] = phue_module

    if "main" in sys.modules:
        del sys.modules["main"]

    return importlib.import_module("main")


def test_pseudonymize_identity_is_deterministic_for_same_input():
    main = import_main_with_stubs()

    a = main._pseudonymize_identity("participant-123", "salt-A")
    b = main._pseudonymize_identity("participant-123", "salt-A")

    assert a == b
    assert isinstance(a, str)
    assert len(a) == 16


def test_pseudonymize_identity_changes_with_salt_and_handles_none():
    main = import_main_with_stubs()

    x = main._pseudonymize_identity("participant-123", "salt-A")
    y = main._pseudonymize_identity("participant-123", "salt-B")

    assert x != y
    assert main._pseudonymize_identity(None, "salt-A") is None
