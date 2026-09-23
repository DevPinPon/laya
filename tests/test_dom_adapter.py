import asyncio
import importlib.util
from pathlib import Path
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from laya.dom import EvidenceAgent
from laya.dom_context import select_evidence
from laya.dom_requirements import check_requirements
from laya.serve import create_app


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"
    pad_token_id = 0
    def __call__(self, text, **kwargs):
        # Stable token identities, so truncation auditing can compare sequences.
        return {"input_ids": [sum(map(ord, word)) + 4 for word in text.split()]}


class FakeAgent:
    tok = Tokenizer()
    cfg = {"max_len": 512, "head_max_len": 192}
    def __init__(self):
        self.calls = []
    def _to_internal(self, q):
        return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    def predict(self, state, questions, model=None):
        from laya.common import build_sequence
        self.calls.append((state, questions))
        seq, _ = build_sequence(self.tok, state, self._to_internal(questions["decision"]), 512, 192)
        return {"answers": {"decision": {"type": "choice", "choice": "a",
                                          "probabilities": {"a": 0.8, "b": 0.1, "unknown": 0.1}}},
                "usage": {"input_tokens": len(seq), "output_tokens": 0}}


BODY = {"query": "Which region contains the cluster hosting service SVC12?",
        "documents": [{"id": "first", "text": "Service SVC12 runs on cluster CL-9."}],
        "choices": {"a": "Coastal", "b": "Inland", "unknown": "Insufficient evidence"},
        "insufficient_choice": "unknown"}


def test_one_call_preserves_raw_answer_and_marks_dom_abstention():
    agent = FakeAgent()
    response = EvidenceAgent(agent).predict(**BODY)
    assert len(agent.calls) == 1
    assert response["prediction"]["answers"]["decision"]["choice"] == "a"
    assert response["decision"]["choice"] == "unknown"
    assert response["decision"]["decision_source"] == "dom_requirements_gate"
    assert response["prediction"]["answers"]["decision"]["probabilities"] == {"a": 0.8, "b": 0.1, "unknown": 0.1}


@pytest.mark.parametrize("change", [
    {"choices": {"a": "only one option"}}, {"insufficient_choice": "invented"},
    {"documents": ["not a record"]}, {"query": "Where is SVC12? " * 80},
    {"choices": {"a": "word " * 70, "b": "other"}},
])
def test_invalid_or_truncated_requests_never_reach_model(change):
    agent = FakeAgent()
    with pytest.raises(ValueError):
        EvidenceAgent(agent).predict(**{**BODY, **change})
    assert not agent.calls


def test_option_budget_uses_the_same_leading_space_as_laya_renderer():
    class PrefixSensitiveTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.startswith(" "):
                result["input_ids"].insert(0, 999)
            return result
    agent = FakeAgent()
    agent.tok = PrefixSensitiveTokenizer()
    with pytest.raises(ValueError, match="option token budget"):
        EvidenceAgent(agent).predict(**{**BODY, "choices": {"a": "word " * 47, "b": "other"},
                                       "insufficient_choice": None})
    assert not agent.calls


def test_optional_endpoint_auth_and_validation(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "unit-test-key")
    agent = FakeAgent()
    agent.loaded = ["english"]
    with TestClient(create_app(router=agent, evidence_agent=EvidenceAgent(agent))) as client:
        assert client.post("/v1/dom/decide", json=BODY).status_code == 401
        headers = {"Authorization": "Bearer unit-test-key"}
        assert client.post("/v1/dom/decide", json=BODY, headers=headers).status_code == 200
        assert client.post("/v1/dom/decide", content="{", headers=headers).status_code == 400
        assert client.post("/v1/dom/decide", json={**BODY, "extra": True}, headers=headers).status_code == 400
        assert client.post("/v1/dom/decide", json={**BODY, "documents": [1]}, headers=headers).status_code == 422
        assert client.post("/v1/dom/decide", content="x" * (4 * 1024 * 1024 + 1), headers=headers).status_code == 413
    with TestClient(create_app(router=agent)) as client:
        assert client.post("/v1/dom/decide", json=BODY, headers=headers).status_code == 404


def test_both_routes_share_one_queue_and_health_stays_available(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    active, maximum = 0, 0
    lock = threading.Lock()
    entered = threading.Event()
    class Slow:
        loaded = ["english"]
        def predict(self, *args, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            entered.set()
            time.sleep(0.15)
            with lock:
                active -= 1
            return {"ok": True}
    slow = Slow()
    app = create_app(router=slow, evidence_agent=slow)
    async def drive():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/systemone", json={"state": "x", "questions": {}}))
            await asyncio.to_thread(entered.wait, 2)
            second = asyncio.create_task(client.post("/v1/dom/decide", json=BODY))
            health = await client.get("/health")
            assert health.status_code == 200 and not first.done()
            assert all(r.status_code == 200 for r in await asyncio.gather(first, second))
    asyncio.run(drive())
    assert maximum == 1


def test_fork_port_preserves_all_registered_stress_case_classifications():
    import json
    from collections import Counter
    data = json.loads((Path(__file__).parents[1] / "research/benchmarks/dom/requirements_cases.v1.json").read_text())
    counts = Counter()
    for case in data["cases"]:
        packet = select_evidence(case["query"], case["documents"], count_tokens=lambda s: len(s.split()))
        check = check_requirements(case["query"], case["documents"], packet)
        counts[check["status"]] += 1
        if not case["scenario"].startswith("unsupported"):
            assert (check["status"] == "supported") == case["answerable"]
    assert counts == {"supported": 36, "insufficient": 48, "unsupported_query": 12, "unsupported_records": 12}


def test_published_workload_is_identical_to_upstream_fixture():
    from research.benchmarks.dom.bench_published import STATE, questions
    path = Path(__file__).parents[1] / "research/scripts/bench_latency.py"
    spec = importlib.util.spec_from_file_location("upstream_latency_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert STATE == module.STATE_EN
    assert all(questions(n) == module.qs(n) for n in (1, 5, 10, 50))
