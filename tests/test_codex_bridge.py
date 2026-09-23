import pytest

from laya.codex_bridge import Bridge, CATALOG, execute_verified


@pytest.mark.parametrize("query,expected", [
    ("calculate: 12 * 7", {"value": "84", "decimal_precision": 50}),
    ("calculate: 0.1 + 0.2", {"value": "0.3", "decimal_precision": 50}),
    ("calculate: -(5 + 2) / 2", {"value": "-3.5", "decimal_precision": 50}),
    ("length: hello", {"code_points": 5}),
    ("length: A😀é", {"code_points": 3}),
    ("length: ", {"code_points": 0}),
    ('validate json: {"x":1}', {"valid": True}),
    ('validate json: {"x":}', {"valid": False}),
    ('validate json: NaN', {"valid": False}),
])
def test_verified_results(query, expected):
    result = Bridge().fast(query)
    assert result["route"] == "verified"
    assert result["result"] == expected
    assert not result["laya_called"]


@pytest.mark.parametrize("query", [
    "", "x" * 2001, "delete all files", "What is the best tool?",
    "calculate: __import__('os').system('echo bad')", "calculate: 2 ** 9999",
    "calculate: 1/0", "calculate: (2 +", "calculate: True", "calculate: 2; delete files",
    "calculate: " + "1+" * 50 + "1", "calculate: [1,2]", "calculate: 1 // 2",
])
def test_no_unverified_execution(query):
    result = Bridge().fast(query)
    assert result["route"] == "codex" and not result["executed"]


@pytest.mark.parametrize("choice", ["t0", "t1", "t2", "none", "garbage"])
def test_all_model_choices_gated(choice):
    bridge = Bridge(predictor=lambda *_: choice)
    result = bridge.fast("calculate: 6 * 7", "laya")
    assert result["executed"] == (choice == "t0")
    assert result["result"] == ({"value": "42", "decimal_precision": 50} if choice == "t0" else None)
    assert not bridge.fast("delete all files", "laya")["executed"]


def test_generic_proposal_never_executes_and_failure_falls_back():
    bridge = Bridge(predictor=lambda *_: "t0")
    result = bridge.route("delete all files", {"delete": "delete files"})
    assert result["proposed_tool"] == "delete"
    assert result["route"] == "codex" and not result["executed"] and not result["verified"]
    def fail(*args):
        raise RuntimeError("unavailable")
    bridge.predictor = fail
    assert bridge.fast("calculate: 2+2", "laya")["route"] == "codex"
    assert bridge.route("test", CATALOG)["proposed_tool"] is None


def test_cache_results_cannot_be_mutated_by_caller():
    execute_verified.cache_clear()
    bridge = Bridge()
    first = bridge.fast("length: safe")
    first["result"]["code_points"] = 999
    second = bridge.fast("length: safe")
    assert second["cache_hit"] and second["result"]["code_points"] == 4


def test_bad_catalog_never_calls_model():
    def unexpected(*args):
        pytest.fail("model should not be called")
    bridge = Bridge(predictor=unexpected)
    for catalog in ({}, {"x": "a" * 161}, {"bad name": "x"}, {str(i): "x" for i in range(9)}):
        assert bridge.route("test", catalog)["reason"] == "model_or_input_failure"
