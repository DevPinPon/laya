"""Validate the frozen operational evaluation without loading model weights."""
import json
from pathlib import Path

import pytest

from research.benchmarks.dom.bench_published import seal
from research.benchmarks.dom.bench_reasoning import oracle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research/benchmarks/dom/results"
PLAN = json.loads((RESULTS / "reasoning_service.v1.plan.json").read_text())
RESULT = json.loads((RESULTS / "reasoning_service.v1.json").read_text())


@pytest.mark.parametrize("family", ["startup", "threads", "auth", "errors"])
def test_frozen_labels_match_executed_service_and_ambiguity_witnesses(family):
    for case in (c for c in PLAN["cases"] if c["family"] == family):
        observed = [oracle(family, s) for s in case["oracle_scenarios"]]
        assert observed == case["oracle_witnesses"]
        if case["answerable"]:
            assert observed[0]["answer"] == case["gold"]
        else:
            assert {r["answer"] for r in observed} == {"yes", "no"}
            assert case["gold"] == "unknown"
            left, right = case["oracle_scenarios"]
            assert [k for k in left if left[k] != right[k]] == [case["missing_field"]]


def test_seals_counts_and_scores_are_consistent():
    assert PLAN["plan_sha256"] == seal({k: v for k, v in PLAN.items() if k != "plan_sha256"})
    assert RESULT["result_sha256"] == seal({k: v for k, v in RESULT.items() if k != "result_sha256"})
    assert RESULT["plan_sha256"] == PLAN["plan_sha256"]
    assert len(PLAN["cases"]) == 32
    assert len(PLAN["fixtures"]) == 96
    assert RESULT["requests"] == 288
    assert len(RESULT["rows"]) == 384  # Dom model and final decision share one call.
    gold = {c["id"]: c["gold"] for c in PLAN["cases"]}
    for arm, summary in RESULT["summary"].items():
        rows = [r for r in RESULT["rows"] if r["arm"] == arm]
        assert len({(r["id"], r["order"]) for r in rows}) == 96
        for row in rows:
            assert row["gold"] == gold[row["id"]]
            assert row["correct"] == (row["choice"] == gold[row["id"]])
            assert abs(sum(row["probabilities"].values()) - 1) < 0.001
        assert summary["overall"]["correct"] == sum(r["correct"] for r in rows)
        for label, answerable in (("answerable", True), ("insufficient", False)):
            selected = [r for r in rows if r["answerable"] == answerable]
            assert summary[label]["correct"] == sum(r["correct"] for r in selected)
            assert summary[label]["n"] == len(selected)


def test_model_inputs_exclude_labels_and_use_same_question_and_options():
    for fixture in PLAN["fixtures"]:
        assert set(fixture["dom"]) == {"query", "documents", "choices", "insufficient_choice"}
        assert all(set(d) == {"id", "text"} for d in fixture["dom"]["documents"])
        raw_question = fixture["raw_all"]["questions"]["decision"]
        assert raw_question == fixture["oracle_support"]["questions"]["decision"]
        assert list(raw_question["criteria"].items()) == list(fixture["dom"]["choices"].items())
        assert fixture["requirements_status"] == "unsupported_query"


def test_unsupported_checks_did_not_change_any_model_choice():
    rows = {(r["id"], r["order"], r["arm"]): r for r in RESULT["rows"]}
    for fixture in PLAN["fixtures"]:
        key = (fixture["id"], fixture["order"])
        model = rows[(*key, "dom_model")]
        system = rows[(*key, "dom_system")]
        assert system["choice"] == model["choice"]
        assert system["decision"]["decision_source"] == "laya"


@pytest.mark.parametrize("stem", ["reasoning_moe", "reasoning_moe_full"])
def test_router_followup_integrity_inputs_and_scores(stem):
    plan = json.loads((RESULTS / f"{stem}.v1.plan.json").read_text())
    result = json.loads((RESULTS / f"{stem}.v1.json").read_text())
    assert plan["plan_sha256"] == seal({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert result["result_sha256"] == seal({k: v for k, v in result.items() if k != "result_sha256"})
    assert result["plan_sha256"] == plan["plan_sha256"]
    assert plan["baseline_plan_sha256"] == PLAN["plan_sha256"]
    assert len(result["rows"]) == result["requests"] == 192
    cases = {c["id"]: c for c in PLAN["cases"]}
    original = {(f["id"], f["order"]): f for f in PLAN["fixtures"]}
    for fixture in plan["fixtures"]:
        route = plan["routes"][fixture["id"]]
        assert route["router"] == "dom.semantic_moe_router.v1"
        assert "reasoning_circuit" in route["expert_ids"]
        by_id = {d["id"]: d["text"] for d in cases[fixture["id"]]["documents"]}
        assert fixture["moe_context"]["state"] == "\n".join(by_id[k] for k in route["selected_ids"])
        assert fixture["moe_guided"]["state"] == route["hydration_text"]
        q = original[fixture["id"], fixture["order"]]["raw_all"]["questions"]["decision"]
        assert fixture["moe_context"]["questions"]["decision"] == q
        guided_q = fixture["moe_guided"]["questions"]["decision"]
        assert guided_q["instructions"] == q["instructions"] + " Dom expert responsibilities: " + " ".join(r["role"] for r in route["roles"])
        assert list(guided_q["criteria"].items()) == list(q["criteria"].items())
    for arm, summary in result["summary"].items():
        rows = [r for r in result["rows"] if r["arm"] == arm]
        assert len({(r["id"], r["order"]) for r in rows}) == 96
        for row in rows:
            assert row["gold"] == cases[row["id"]]["gold"]
            assert row["correct"] == (row["choice"] == row["gold"])
            assert abs(sum(row["probabilities"].values()) - 1) < 0.001
        assert summary["overall"]["correct"] == sum(r["correct"] for r in rows)
        for label, answerable in (("answerable", True), ("insufficient", False)):
            selected = [r for r in rows if r["answerable"] == answerable]
            assert summary[label]["correct"] == sum(r["correct"] for r in selected)
            assert summary[label]["n"] == len(selected)
