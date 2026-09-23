"""Follow-up: actual Dom semantic MoE router plus unchanged local Laya.

Uses the already frozen operational cases, with no answers supplied to routing.
This is an exploratory follow-up on seen cases, not a new held-out evaluation.
Requires a separate checkout of Dom via --dom-root; no Dom code is vendored.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import sys
import tempfile
import time

from .bench_http import request, write_new
from .bench_published import ROOT, file_hash, seal, percentiles
from .bench_reasoning import helper_for, audit_sequence, verify as verify_baseline


def routed(query, documents, dom_root):
    """Only query and public source records reach Dom; no gold/support IDs."""
    if str(dom_root) not in sys.path:
        sys.path.append(str(dom_root))
    from dom_agent.semantic_substrate import SemanticSubstrateTools
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="laya-moe-") as temporary:
        workspace = Path(temporary)
        cells = [{"id": d["id"], "title": "Operational record", "summary": d["text"],
                  "type": "fact" if d["id"] == "runtime" else "procedure",
                  "render": {"text": d["text"]}, "confidence": 0.5,
                  "policy": {"tenant_id": "local", "visibility": "public"}}
                 for d in documents]
        path = workspace / "cells.json"
        path.write_text(json.dumps(cells), encoding="utf-8")
        tool = SemanticSubstrateTools(workspace=workspace)
        response = tool.moe_route(query, cell_path=str(path), max_cells=6,
                                  max_prompt_tokens=180, max_experts=2)
        if not response.ok:
            raise RuntimeError(response.content)
        payload = json.loads(response.content)
        by_id = {d["id"]: d["text"] for d in documents}
        selected = payload["selected_cells"]
        # Render the actual selected source texts without consulting gold labels.
        plain = "\n".join(by_id[k] for k in selected)
        result = {"router": payload["router"], "expert_ids": payload["expert_ids"],
                  "selected_ids": selected, "plain_state": plain,
                  "hydration_text": payload["hydration_text"],
                  "roles": [{"id": e["id"], "role": e["role"]} for e in payload["selected_experts"]],
                  "merge_plan": payload["merge_plan"]}
    return result, (time.perf_counter() - started) * 1000


def imported_dom_hashes(dom_root):
    found = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("dom_agent") and filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(dom_root):
                found[path.relative_to(dom_root).as_posix()] = file_hash(path)
    return dict(sorted(found.items()))


def prepare(args):
    baseline = json.loads(args.baseline.read_text())
    verify_baseline(baseline, args.model_dir)
    helper = helper_for(args.model_dir)
    cases = {c["id"]: c for c in baseline["cases"]}
    routes = {c["id"]: routed(c["query"], c["documents"], args.dom_root)[0] for c in cases.values()}
    fixtures = []
    for original in baseline["fixtures"]:
        route = routes[original["id"]]
        q = original["raw_all"]["questions"]["decision"]
        fixture = {"id": original["id"], "order": original["order"]}
        for arm in ("moe_context", "moe_guided"):
            question = dict(q)
            state = route["plain_state"]
            if arm == "moe_guided":
                state = route["hydration_text"]
                question["instructions"] += " Dom expert responsibilities: " + " ".join(r["role"] for r in route["roles"])
            fixture[arm] = {"state": state, "questions": {"decision": question}, "model": "english"}
            fixture[arm + "_tokens"] = audit_sequence(helper, state, question)
        fixtures.append(fixture)
    body = {"format": "laya.dom_moe_reasoning_plan.v1", "baseline_plan_sha256": baseline["plan_sha256"],
            "source_sha256": {"research/benchmarks/dom/bench_reasoning_router.py": file_hash(Path(__file__))},
            "dom_source_sha256": imported_dom_hashes(args.dom_root),
            "protocol": {"stage": "exploratory follow-up on already evaluated cases; no new independent holdout",
                         "router": "unmodified SemanticSubstrateTools.moe_route with built-in expert registry",
                         "cells": "runtime is fact; all rule records including distractors are procedures; neutral title/confidence, no gold annotations",
                         "budgets": {"max_cells": 6, "max_prompt_tokens": 180, "max_experts": 2},
                         "moe_context": "actual selected records verbatim, original question",
                         "moe_guided": "native hydration_text plus selected expert role instructions, original choices",
                         "executor": "same unchanged English Laya checkpoint; no stronger model or executable rule solver",
                         "isolation": "fresh temporary Dom workspace per request; no project memories, labels, or oracle support IDs",
                         "requests": 192, "timing": "client route staging/selection plus complete Laya HTTP response; cold route workspace each request",
                         "tuning": "both arms specified before predictions; original benchmark and integration unchanged"},
            "routes": routes, "fixtures": fixtures}
    write_new(args.plan, {**body, "plan_sha256": seal(body)})
    print(json.dumps({"status": "frozen", "experts": sorted({e for r in routes.values() for e in r["expert_ids"]}),
                      "max_tokens": {a: max(f[a + "_tokens"] for f in fixtures) for a in ("moe_context", "moe_guided")},
                      "dom_modules_hashed": len(body["dom_source_sha256"])}), flush=True)


def verify(plan, baseline, args):
    verify_baseline(baseline, args.model_dir)
    assert plan["baseline_plan_sha256"] == baseline["plan_sha256"]
    assert plan["plan_sha256"] == seal({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert all(file_hash(ROOT / p) == h for p, h in plan["source_sha256"].items())
    assert all(file_hash(args.dom_root / p) == h for p, h in plan["dom_source_sha256"].items())


def run(args):
    if args.out.exists():
        raise ValueError("refusing to overwrite results")
    plan = json.loads(args.plan.read_text())
    baseline = json.loads(args.baseline.read_text())
    verify(plan, baseline, args)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, LAYA_API_KEY=key)
    command = [sys.executable, "-m", "research.benchmarks.dom.serve_local", "--model-dir", str(args.model_dir),
               "--device", "cuda", "--threads", "4", "--port", str(port)]
    rows = []
    started = time.perf_counter()
    with args.out.with_suffix(".server.log").open("x", encoding="utf-8") as log:
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
            warmup = {"state": "The lamp is green.", "questions": {"decision": {
                "type": "choice", "instructions": "What color is the lamp?",
                "criteria": {"green": "Green", "blue": "Blue"}}}, "model": "english"}
            for _ in range(3):
                request(connection, "/v1/systemone", warmup, key)
            cases = {c["id"]: c for c in baseline["cases"]}
            fixtures = list(plan["fixtures"])
            random.Random(1205).shuffle(fixtures)
            for index, fixture in enumerate(fixtures):
                case = cases[fixture["id"]]
                arms = ("moe_context", "moe_guided") if index % 2 == 0 else ("moe_guided", "moe_context")
                for arm in arms:
                    start = time.perf_counter()
                    actual_route, route_ms = routed(case["query"], case["documents"], args.dom_root)
                    assert actual_route == plan["routes"][case["id"]]
                    # The byte-equivalent live route was just verified against the frozen request.
                    response, http_ms, _, _ = request(connection, "/v1/systemone", fixture[arm], key)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    assert response["usage"]["input_tokens"] == fixture[arm + "_tokens"]
                    answer = response["answers"]["decision"]
                    rows.append({"id": case["id"], "family": case["family"], "order": fixture["order"],
                                 "arm": arm, "answerable": case["answerable"], "gold": case["gold"],
                                 "choice": answer["choice"], "correct": answer["choice"] == case["gold"],
                                 "probabilities": answer["probabilities"], "input_tokens": response["usage"]["input_tokens"],
                                 "elapsed_ms": elapsed_ms, "route_ms": route_ms, "http_ms": http_ms})
                if (index + 1) % 16 == 0:
                    print(json.dumps({"completed_fixtures": index + 1, "requests": (index + 1) * 2}), flush=True)
        finally:
            connection.close()
            process.terminate()
            process.wait(timeout=30)
    verify(plan, baseline, args)
    summary = {}
    for arm in ("moe_context", "moe_guided"):
        own = [r for r in rows if r["arm"] == arm]
        def score(items):
            return {"correct": sum(r["correct"] for r in items), "n": len(items),
                    "accuracy": sum(r["correct"] for r in items) / len(items)}
        summary[arm] = {"overall": score(own), "answerable": score([r for r in own if r["answerable"]]),
                        "insufficient": score([r for r in own if not r["answerable"]]),
                        "all_orders_correct": sum(all(r["correct"] for r in own if r["id"] == c["id"]) for c in baseline["cases"]),
                        "by_family": {f: score([r for r in own if r["family"] == f]) for f in sorted({r["family"] for r in own})},
                        "latency": percentiles([r["elapsed_ms"] for r in own]),
                        "route_latency": percentiles([r["route_ms"] for r in own]),
                        "http_latency": percentiles([r["http_ms"] for r in own])}
    body = {"format": "laya.dom_moe_reasoning_result.v1", "plan_sha256": plan["plan_sha256"],
            "requests": len(rows), "rows": rows, "summary": summary}
    write_new(args.out, {**body, "result_sha256": seal(body)})
    print(json.dumps({a: {k: v for k, v in s.items() if "latency" not in k} for a, s in summary.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--dom-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "research/benchmarks/dom/results/reasoning_service.v1.plan.json")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.dom_root = args.dom_root.resolve(strict=True)
    args.model_dir = args.model_dir.resolve(strict=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.out is None:
            parser.error("run requires --out")
        run(args)


if __name__ == "__main__":
    main()
