from error_taxonomy import (
    DEEPFACE_ANALYZE_FAILED,
    HUE_SEND_FAILED,
    category_for_error_code,
)


def test_category_for_error_code_known_values():
    assert category_for_error_code(HUE_SEND_FAILED) == "runtime.hue"
    assert category_for_error_code(DEEPFACE_ANALYZE_FAILED) == "runtime.emotion"


def test_category_for_error_code_unknown_value_falls_back():
    assert category_for_error_code("NOT_A_REAL_CODE") == "runtime.unknown"
