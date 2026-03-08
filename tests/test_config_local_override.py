import importlib
import sys
import types


def test_config_local_overrides_sensitive_defaults():
    config = importlib.import_module("config")

    original_bridge = config.HUE_BRIDGE_IP
    original_ids = list(config.HUE_LIGHT_IDS)

    local = types.ModuleType("config_local")
    local.HUE_BRIDGE_IP = "10.0.0.9"
    local.HUE_LIGHT_IDS = [99, 100]
    local.HUE_LIGHT_ROLES = {99: "primary", 100: "accent"}

    sys.modules["config_local"] = local
    try:
        cfg = importlib.reload(config)
        assert cfg.HUE_BRIDGE_IP == "10.0.0.9"
        assert cfg.HUE_LIGHT_IDS == [99, 100]
        assert cfg.HUE_LIGHT_ROLES == {99: "primary", 100: "accent"}
    finally:
        sys.modules.pop("config_local", None)
        cfg = importlib.reload(config)
        assert cfg.HUE_BRIDGE_IP == original_bridge
        assert cfg.HUE_LIGHT_IDS == original_ids
