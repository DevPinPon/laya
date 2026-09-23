"""Frozen operational reasoning evaluation through the real local HTTP API.

Run prepare before run. Labels come from executing production service code with
stubbed model outcomes, not from Laya or a model judge. Scenarios are authored
operational tests, not captured customer traffic or an independent benchmark.
"""
from __future__ import annotations

import argparse
import ast
import copy
import http.client
import json
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import sys
import time
from unittest.mock import patch

from .bench_http import request, write_new
from .bench_published import ROOT, MODEL_REVISION, file_hash, model_hashes, percentiles, seal

CHOICES = {"yes": "Yes", "no": "No", "unknown": "Insufficient evidence"}
RULES = {
    "boolean": "Environment Boolean settings trim spaces and ignore case. Only 1, true, yes, and on enable a setting. Any other present value disables it; an absent setting uses its default.",
    "startup": "Startup preloading defaults to enabled. Automatic task detection defaults to disabled. The preload setting and automatic task detection setting are evaluated independently.",
    "models": "If startup preloading is enabled, the comma-separated model list is stripped of spaces and empty entries. An empty or absent list preloads all checkpoints, including English. A nonempty list preloads only its listed checkpoints.",
    "threads": "A present thread limit is parsed as an integer. Only a positive integer changes the Torch thread count. An absent, empty, noninteger, zero, or negative limit leaves the thread count unchanged.",
    "auth": "An absent or empty server API key disables authentication. Otherwise the request header must exactly equal Bearer followed by a space and the configured key. An incorrect or absent header causes HTTP 401 before body validation or prediction.",
    "body": "After authentication, a request body must be a JSON object containing the questions field. Otherwise the service returns HTTP 400 without calling prediction. Presence of the field passes this check even when its value is null or empty.",
    "prediction": "After successful authentication and body validation, the service invokes the router. A successful router call returns HTTP 200. An ordinary exception from the router is translated to HTTP 422; a router HTTPException retains its own status code.",
    "noise_health": "The health endpoint returns status ok, the names of loaded checkpoints, and the configured device. Health reports do not run model prediction.",
    "noise_bind": "The server bind address defaults to 0.0.0.0 and the port defaults to 8000. Uvicorn starts the application with the configured log level.",
    "noise_queue": "Inference uses one worker and a shared asynchronous lock. Waiting requests do not run their model forward passes simultaneously.",
    "noise_model": "The public multilingual model identifier selects the multilingual checkpoint. An unrecognized client model identifier lets the router choose automatically.",
}
QUESTIONS = {
    "startup": "Given the startup settings and rules, will startup preload English AND enable automatic task detection?",
    "threads": "Given the startup settings and rules, will startup apply exactly four Torch threads AND skip preloading?",
    "auth": "Given the request settings and service rules, will this request invoke the router AND return HTTP 200?",
    "errors": "Given the request settings and service rules, will this request invoke the router AND return HTTP 422?",
}
SUPPORT = {"startup": ["boolean", "startup", "models"],
           "threads": ["boolean", "startup", "threads"],
           "auth": ["auth", "body", "prediction"],
           "errors": ["auth", "body", "prediction"]}


def oracle(family, scenario):
    """Execute actual application logic, isolating environment and side effects."""
    import laya.serve as serve
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    if family in {"startup", "threads"}:
        import torch
        env = {k: v for k, v in scenario.items() if v is not None}
        class Router:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.preloads = []
            def preload(self, names):
                self.preloads.append(names)
        with patch.dict(os.environ, env, clear=True), patch("laya.router.Router", Router), \
                patch.object(torch, "set_num_threads") as set_threads:
            router = serve.build_router()
        if family == "startup":
            has_english = any(names is None or "english" in names for names in router.preloads)
            answer = has_english and router.options["auto_task_detection"]
        else:
            answer = set_threads.call_args_list == [((4,), {})] and not router.preloads
        return {"answer": "yes" if answer else "no", "preloads": router.preloads,
                "auto_task": router.options["auto_task_detection"],
                "thread_calls": [list(c.args) for c in set_threads.call_args_list]}
    class Router:
        loaded = []
        calls = 0
        def predict(self, *args, **kwargs):
            self.calls += 1
            behavior = scenario["predictor"]
            if behavior == "ValueError":
                raise ValueError("fixed oracle test outcome")
            if behavior == "RuntimeError":
                raise RuntimeError("fixed oracle test outcome")
            if behavior == "HTTPException409":
                raise HTTPException(status_code=409, detail="fixed oracle test outcome")
            return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
    router = Router()
    env = {} if scenario["key"] is None else {"LAYA_API_KEY": scenario["key"]}
    with patch.dict(os.environ, env, clear=True):
        app = serve.create_app(router=router)
        with TestClient(app) as client:
            headers = {} if scenario["header"] is None else {"Authorization": scenario["header"]}
            response = client.post("/v1/systemone", json=scenario["body"], headers=headers)
    target = 200 if family == "auth" else 422
    answer = router.calls == 1 and response.status_code == target
    return {"answer": "yes" if answer else "no", "calls": router.calls, "http_status": response.status_code}


def settings_text(family, scenario):
    if family in {"startup", "threads"}:
        names = {"LAYA_PRELOAD": "startup preloading", "LAYA_AUTO_TASK": "automatic task detection",
                 "LAYA_MODELS": "model list", "LAYA_THREADS": "thread limit"}
        fields = [f'{names[k]} = {"absent" if v is None else json.dumps(v)}'
                  for k, v in scenario.items()]
        return "Startup settings: " + "; ".join(fields) + ". Other environment settings are absent."
    return (f'Request settings: server API key = {json.dumps(scenario["key"])}; '
            f'authorization header = {json.dumps(scenario["header"])}; '
            f'JSON body = {json.dumps(scenario["body"])}; '
            f'router behavior if called = {scenario["predictor"]}. Null key or header means absent.')


def build_cases():
    startup = [
        {"LAYA_PRELOAD": None, "LAYA_AUTO_TASK": "yes", "LAYA_MODELS": "english"},
        {"LAYA_PRELOAD": " ON ", "LAYA_AUTO_TASK": "TRUE", "LAYA_MODELS": " , "},
        {"LAYA_PRELOAD": "1", "LAYA_AUTO_TASK": "on", "LAYA_MODELS": "multilingual, english"},
        {"LAYA_PRELOAD": "off", "LAYA_AUTO_TASK": "yes", "LAYA_MODELS": "english"},
        {"LAYA_PRELOAD": "true", "LAYA_AUTO_TASK": None, "LAYA_MODELS": "english"},
        {"LAYA_PRELOAD": "yes", "LAYA_AUTO_TASK": "1", "LAYA_MODELS": "multilingual"},
    ]
    threads = [
        {"LAYA_THREADS": "4", "LAYA_PRELOAD": "0"},
        {"LAYA_THREADS": " 4 ", "LAYA_PRELOAD": "false"},
        {"LAYA_THREADS": "+4", "LAYA_PRELOAD": "disabled"},
        {"LAYA_THREADS": "4", "LAYA_PRELOAD": None},
        {"LAYA_THREADS": "0", "LAYA_PRELOAD": "0"},
        {"LAYA_THREADS": "four", "LAYA_PRELOAD": "off"},
    ]
    base = {"key": "demo", "header": "Bearer demo", "body": {"questions": {}}, "predictor": "success"}
    auth = [base, {**base, "key": None, "header": None},
            {**base, "key": "", "header": "unrelated"},
            {**base, "header": "bearer demo"}, {**base, "body": {}},
            {**base, "predictor": "ValueError"}]
    errors = [{**base, "predictor": "ValueError"},
              {**base, "predictor": "RuntimeError", "key": None, "header": None},
              {**base, "predictor": "ValueError", "body": {"questions": None}},
              {**base, "predictor": "ValueError", "body": {}},
              {**base, "predictor": "ValueError", "header": None},
              {**base, "predictor": "HTTPException409"}]
    cases = []
    for family, scenarios in {"startup": startup, "threads": threads, "auth": auth, "errors": errors}.items():
        # Two incomplete scenarios have concrete opposite-outcome witnesses.
        unknown_fields = {"startup": ["LAYA_PRELOAD", "LAYA_AUTO_TASK"],
                          "threads": ["LAYA_PRELOAD", "LAYA_THREADS"],
                          "auth": ["header", "predictor"],
                          "errors": ["header", "predictor"]}[family]
        opposite = {"LAYA_PRELOAD": "false" if family == "startup" else "true",
                    "LAYA_AUTO_TASK": "false", "LAYA_THREADS": "8", "header": "wrong",
                    "predictor": "ValueError" if family == "auth" else "success"}
        for index in range(8):
            scenario = copy.deepcopy(scenarios[index] if index < 6 else scenarios[0])
            witnesses = [oracle(family, scenario)]
            missing = None
            visible = copy.deepcopy(scenario)
            if index >= 6:
                missing = unknown_fields[index - 6]
                other = {**scenario, missing: opposite[missing]}
                witnesses.append(oracle(family, other))
                assert {w["answer"] for w in witnesses} == {"yes", "no"}
                visible[missing] = "NOT RECORDED: value could differ"
                gold = "unknown"
            else:
                gold = witnesses[0]["answer"]
                assert gold == ("yes" if index < 3 else "no")
            runtime = {"id": "runtime", "text": settings_text(family, visible)}
            docs = [{"id": name, "text": RULES[name]} for name in SUPPORT[family]] + [runtime]
            support_ids = [d["id"] for d in docs]
            docs += [{"id": name, "text": RULES[name]} for name in RULES if name.startswith("noise_")]
            random.Random(840 + index).shuffle(docs)
            cases.append({"id": f"{family}-{index}", "family": family, "answerable": index < 6,
                          "query": QUESTIONS[family], "documents": docs, "support_ids": support_ids,
                          "gold": gold, "missing_field": missing, "oracle_witnesses": witnesses,
                          "oracle_scenarios": [scenario] if index < 6 else [scenario, other]})
    return cases


def helper_for(model_dir):
    from transformers import AutoTokenizer
    from laya.dom import EvidenceAgent
    class TokenizerOnly:
        tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
        cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
        @staticmethod
        def _to_internal(q):
            return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    return EvidenceAgent(TokenizerOnly())


def audit_sequence(helper, state, q):
    from laya.common import build_sequence
    internal = helper.agent._to_internal(q)
    complete, _ = build_sequence(helper.agent.tok, state, internal, 8192, 8192)
    actual, _ = build_sequence(helper.agent.tok, state, internal,
                               helper.agent.cfg["max_len"], helper.agent.cfg["head_max_len"])
    if actual != complete:
        raise ValueError("benchmark input would truncate")
    return len(actual)


def prepare(model_dir, path):
    helper = helper_for(model_dir)
    cases = build_cases()
    fixtures = []
    for case in cases:
        for order in range(3):
            keys = list(CHOICES)
            keys = keys[order:] + keys[:order]
            body = {"query": case["query"], "documents": case["documents"],
                    "choices": {k: CHOICES[k] for k in keys}, "insufficient_choice": "unknown"}
            prepared = helper.prepare(**body)
            fixture = {"id": case["id"], "order": order, "dom": body,
                       "dom_tokens": prepared["expected_input_tokens"],
                       "packet_sha256": prepared["packet"]["packet_sha256"],
                       "selected_ids": prepared["packet"]["selected_ids"],
                       "evidence_status": prepared["packet"]["status"],
                       "requirements_status": prepared["check"]["status"]}
            for arm in ("raw_all", "oracle_support"):
                docs = case["documents"] if arm == "raw_all" else [
                    d for d in case["documents"] if d["id"] in case["support_ids"]]
                state = "\n".join(d["text"] for d in docs)
                fixture[arm] = {"state": state, "questions": {"decision": prepared["question"]}, "model": "english"}
                fixture[arm + "_tokens"] = audit_sequence(helper, state, prepared["question"])
            fixtures.append(fixture)
    files = [*sorted((ROOT / "laya").glob("*.py")), Path(__file__),
             Path(__file__).with_name("serve_local.py"), Path(__file__).with_name("bench_http.py"),
             Path(__file__).with_name("bench_published.py")]
    source = (ROOT / "laya/serve.py").read_text(encoding="utf-8")
    source_lines = source.splitlines()
    referenced = {"_env_bool", "build_router", "_apply_thread_limit", "_check_auth", "systemone", "health", "main"}
    provenance = [{"path": "laya/serve.py", "symbol": node.name, "start_line": node.lineno,
                   "end_line": node.end_lineno,
                   "excerpt": "\n".join(source_lines[node.lineno - 1:node.end_lineno])}
                  for node in ast.walk(ast.parse(source))
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in referenced]
    body = {"format": "laya.operational_reasoning_plan.v1", "model_revision": MODEL_REVISION,
            "rule_provenance": provenance,
            # List entries avoid treating filenames as credential field names in scanners.
            "model_files": [{"path": k, "sha256": v} for k, v in model_hashes(model_dir).items()],
            "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
            "protocol": {"cases": 32, "answerable": 24, "insufficient": 8, "orders": 3,
                         "model_requests": 288, "device": "cuda", "threads": 4,
                         "scope": "Authored operational scenarios grounded in actual service code, with executable oracles; not live customer traffic or independently authored validation",
                         "evidence": "Human-readable rule paraphrases of laya/serve.py; code hashes frozen; runtime fields are authored scenario inputs",
                         "oracle": "Actual build_router/create_app logic; model outcome and hardware setter stubbed only for labeling; ambiguity proven by opposite-outcome executions",
                         "arms": "raw_all, Dom selected model, Dom final system, oracle_support; Dom model/system share one request",
                         "primary": "answerable accuracy and insufficient-evidence accuracy separately; overall secondary",
                         "tuning": "No changes to integration, prompts, labels, or cases after inspecting model outputs",
                         "independence": "Same author for integration and evaluation; 4 related families; no general production accuracy claim"},
            "cases": cases, "fixtures": fixtures}
    write_new(path, {**body, "plan_sha256": seal(body)})
    print(json.dumps({"status": "frozen", "cases": len(cases), "fixtures": len(fixtures),
                      "max_raw_tokens": max(f["raw_all_tokens"] for f in fixtures),
                      "max_dom_tokens": max(f["dom_tokens"] for f in fixtures),
                      "requirements_statuses": sorted({f["requirements_status"] for f in fixtures})}), flush=True)


def verify(plan, model_dir):
    assert plan["plan_sha256"] == seal({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert {r["path"]: r["sha256"] for r in plan["model_files"]} == model_hashes(model_dir)
    assert all(file_hash(ROOT / p) == h for p, h in plan["source_sha256"].items())
    for case in plan["cases"]:
        assert [oracle(case["family"], s) for s in case["oracle_scenarios"]] == case["oracle_witnesses"]


def summarize(rows, plan):
    result = {}
    for arm in ("raw_all", "dom_model", "dom_system", "oracle_support"):
        own = [r for r in rows if r["arm"] == arm]
        def score(items):
            return {"correct": sum(r["correct"] for r in items), "n": len(items),
                    "accuracy": sum(r["correct"] for r in items) / len(items)}
        result[arm] = {"overall": score(own),
                       "answerable": score([r for r in own if r["answerable"]]),
                       "insufficient": score([r for r in own if not r["answerable"]]),
                       "by_family": {f: score([r for r in own if r["family"] == f]) for f in QUESTIONS},
                       "all_orders_correct": sum(all(r["correct"] for r in own if r["id"] == c["id"]) for c in plan["cases"]),
                       "false_abstentions": sum(r["answerable"] and r["choice"] == "unknown" for r in own),
                       "latency": percentiles([r["elapsed_ms"] for r in own])}
    return result


def run(model_dir, plan_path, out):
    if out.exists():
        raise ValueError("refusing to overwrite results")
    plan = json.loads(plan_path.read_text())
    verify(plan, model_dir)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, LAYA_API_KEY=key)
    command = [sys.executable, "-m", "research.benchmarks.dom.serve_local", "--model-dir", str(model_dir),
               "--device", "cuda", "--threads", "4", "--port", str(port)]
    rows = []
    started = time.perf_counter()
    with out.with_suffix(".server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("service startup failed")
                try:
                    health, _, _, _ = request(connection, "/health", None, key)
                    assert health["device"] == "cuda"
                    break
                except (OSError, http.client.HTTPException):
                    connection.close()
                    if time.perf_counter() - started > 120:
                        raise RuntimeError("startup timeout")
                    time.sleep(0.1)
            startup_ms = (time.perf_counter() - started) * 1000
            warmup = {"state": "The lamp is green.", "questions": {"decision": {
                "type": "choice", "instructions": "What color is the lamp?",
                "criteria": {"green": "Green", "blue": "Blue"}}}, "model": "english"}
            for _ in range(3):
                request(connection, "/v1/systemone", warmup, key)
            cases = {c["id"]: c for c in plan["cases"]}
            fixtures = list(plan["fixtures"])
            random.Random(905).shuffle(fixtures)
            for index, fixture in enumerate(fixtures):
                case = cases[fixture["id"]]
                arms = ["raw_all", "dom", "oracle_support"]
                arms = arms[index % 3:] + arms[:index % 3]
                for arm in arms:
                    response, ms, _, _ = request(connection, "/v1/dom/decide" if arm == "dom" else "/v1/systemone",
                                                 fixture[arm], key)
                    prediction = response["prediction"] if arm == "dom" else response
                    assert prediction["usage"]["input_tokens"] == fixture[arm + "_tokens"]
                    answer = prediction["answers"]["decision"]
                    row = {"id": case["id"], "family": case["family"], "order": fixture["order"],
                           "answerable": case["answerable"], "gold": case["gold"], "elapsed_ms": ms,
                           "arm": "dom_model" if arm == "dom" else arm,
                           "choice": answer["choice"], "probabilities": answer.get("probabilities"),
                           "input_tokens": prediction["usage"]["input_tokens"],
                           "correct": answer["choice"] == case["gold"]}
                    rows.append(row)
                    if arm == "dom":
                        assert response["evidence"]["packet_sha256"] == fixture["packet_sha256"]
                        assert response["requirements"]["status"] == fixture["requirements_status"]
                        choice = response["decision"]["choice"]
                        rows.append({**row, "arm": "dom_system", "choice": choice,
                                     "correct": choice == case["gold"], "decision": response["decision"],
                                     "selected_ids": response["evidence"]["selected_ids"],
                                     "requirements_status": response["requirements"]["status"]})
                if (index + 1) % 16 == 0:
                    print(json.dumps({"completed_fixtures": index + 1, "requests": (index + 1) * 3}), flush=True)
        finally:
            connection.close()
            process.terminate()
            process.wait(timeout=30)
    verify(plan, model_dir)
    body = {"format": "laya.operational_reasoning_result.v1", "plan_sha256": plan["plan_sha256"],
            "startup_ms": startup_ms, "requests": 288, "rows": rows, "summary": summarize(rows, plan)}
    write_new(out, {**body, "result_sha256": seal(body)})
    print(json.dumps({"status": "complete", "summary": {a: {k: v for k, v in s.items() if k != "latency"}
                                                             for a, s in body["summary"].items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.mode == "prepare":
        prepare(args.model_dir.resolve(), args.plan)
    else:
        if args.out is None:
            parser.error("run requires --out")
        run(args.model_dir.resolve(), args.plan, args.out)


if __name__ == "__main__":
    main()
