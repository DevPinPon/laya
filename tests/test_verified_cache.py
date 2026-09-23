"""The model can nominate any candidate; it cannot authorize reuse."""
import time

import pytest

from research.benchmarks.dom.verified_cache import ApprovalAuthority, decide, verify


@pytest.fixture
def example():
    authority = ApprovalAuthority(b"test-only-approval-key-not-a-secret")
    request = {"namespace": "one", "task": "test", "facts": {"value": True}, "revision": "v1", "query": "Is this valid?"}
    entry = authority.approve({"id": "record", "namespace": "one", "task": "test", "facts": {"value": True},
                               "revision": "v1", "question": request["query"], "approved_queries": [request["query"]],
                               "answer": "yes", "proof": "independently-checked", "status": "verified",
                               "issued_at": time.time() - 10, "expires_at": time.time() + 60})
    return authority, request, entry


def test_valid_record_passes(example):
    authority, request, entry = example
    assert verify(entry, request, authority) == (True, "verified_applicable")


@pytest.mark.parametrize("change", [
    {"namespace": "other"}, {"task": "other"}, {"facts": {"value": False}},
    {"facts": {"value": 1}}, {"facts": {}}, {"revision": "v2"},
    {"query": "Is this NOT valid?"}, {"query": "is this valid?"},
])
def test_request_differences_cannot_reuse(example, change):
    authority, request, entry = example
    assert not verify(entry, {**request, **change}, authority)[0]


@pytest.mark.parametrize("field,value", [("answer", "no"), ("proof", "replacement"), ("facts", {}), ("expires_at", 99999999999)])
def test_modified_database_record_fails_approval(example, field, value):
    authority, request, entry = example
    entry[field] = value
    assert verify(entry, request, authority) == (False, "missing_or_invalid_approval")


@pytest.mark.parametrize("change", [{"expires_at": 0}, {"issued_at": 99999999999}, {"status": "draft"}])
def test_signed_but_ineligible_records_are_rejected(example, change):
    authority, request, entry = example
    entry = authority.approve({**entry, **change})
    assert not verify(entry, request, authority)[0]


def test_revocation_and_different_authority_fail(example):
    authority, request, entry = example
    assert not verify(entry, request, ApprovalAuthority(b"different-test-only-key-0123456789"))[0]
    authority.revoked.add(entry["id"])
    assert not verify(entry, request, authority)[0]


def test_wrong_model_choice_and_model_failure_return_handoff(example):
    authority, request, entry = example
    wrong = authority.approve({**entry, "facts": {"value": False}})
    class Store:
        def __init__(self):
            self.authority = authority
        def candidates(self, req):
            return [wrong, entry]
    store = Store()
    assert decide(store, request, "laya", lambda *_: 0)["route"] == "codex"
    assert decide(store, request, "laya", lambda *_: 1)["route"] == "cache"
    for choice in (None, 5, -1, True):
        assert decide(store, request, "laya", lambda *_, c=choice: c)["route"] == "codex"
    def failure(*_):
        raise TimeoutError("offline")
    assert decide(store, request, "laya", failure)["route"] == "codex"


def test_gate_does_not_claim_to_detect_a_bad_trusted_approval(example):
    authority, request, entry = example
    # An authorized signer can attest a wrong answer: this is an explicit limit,
    # not proof that a MAC establishes truth or general semantic correctness.
    approved_wrong = authority.approve({**entry, "answer": "no"})
    assert verify(approved_wrong, request, authority)[0]


def test_indexed_cache_and_exhaustive_wrong_candidate_proposals(tmp_path):
    from types import SimpleNamespace
    from research.benchmarks.dom.bench_verified_cache import create_store, dataset
    class PayloadStore:
        def __init__(self, path):
            self.data = {}
        def put(self, **row):
            self.data[row["namespace"], row["cache_key"]] = row["body"]
        def get(self, namespace, key):
            body = self.data.get((namespace, key))
            return None if body is None else SimpleNamespace(body=body)
        def close(self):
            pass
    specs, cases = dataset()
    store = create_store(tmp_path / "store", 100, specs, PayloadStore)
    try:
        for case in cases:
            direct = decide(store, case["request"], "structured")
            assert (direct["route"] == "cache") == (case["expected"] is not None)
            if direct["route"] == "cache":
                assert direct["answer"] == case["expected"]
            # Deliberately propose every enrolled record, including wrong,
            # cross-task, expired, tampered, revoked and unverified entries.
            for spec in specs:
                entry = store.get("benchmark", spec["id"])
                valid, _ = verify(entry, case["request"], store.authority)
                if valid:
                    assert case["expected"] is not None
                    assert entry["answer"] == case["expected"]
        duplicated = store.get("benchmark", specs[0]["id"])
        duplicated = store.authority.approve({**duplicated, "id": "duplicate"})
        store.add(duplicated)
        store.finish_writes()
        first = next(c for c in cases if c["id"] == specs[0]["id"] + "-exact")
        assert decide(store, first["request"], "structured")["route"] == "codex"
    finally:
        store.close()


def test_frozen_measurements_and_outcomes_are_consistent():
    import json
    from pathlib import Path
    from research.benchmarks.dom.bench_published import seal
    root = Path(__file__).resolve().parents[1] / "research/benchmarks/dom/results"
    plan = json.loads((root / "verified_cache.v1.plan.json").read_text())
    result = json.loads((root / "verified_cache.v1.json").read_text())
    assert plan["plan_sha256"] == seal({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert result["result_sha256"] == seal({k: v for k, v in result.items() if k != "result_sha256"})
    assert result["plan_sha256"] == plan["plan_sha256"]
    assert len(plan["cases"]) == 88
    assert len(result["rows"]) == 3168
    expected = {c["id"]: c["expected"] for c in plan["cases"]}
    assert sum(r["laya_called"] for r in result["rows"]) == 1080
    for size, arms in result["summary"].items():
        for arm, summary in arms.items():
            rows = [r for r in result["rows"] if r["size"] == int(size) and r["arm"] == arm]
            assert len({(r["id"], r["round"]) for r in rows}) == 264
            accepted = [r for r in rows if r["route"] == "cache"]
            assert len(accepted) == summary["accepted"]
            assert sum(r["answer"] != expected[r["id"]] for r in accepted) == summary["wrong_accepts"] == 0
            assert all(not r["codex_called"] for r in rows)
            if arm == "structured":
                assert len(accepted) == summary["valid_hit_opportunities"] == 144
