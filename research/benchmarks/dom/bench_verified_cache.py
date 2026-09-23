"""Measure real Dom SQLite payload reads, indexed matching, verification and Laya.

Creates isolated benchmark databases. Codex fallback is a returned packet only;
no remote Codex call, latency, answer quality, or token saving is fabricated.
"""
from __future__ import annotations
import argparse
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

from .bench_http import request as http_request, write_new
from .bench_published import ROOT, file_hash, model_hashes, seal, percentiles
from .bench_reasoning import build_cases, oracle, helper_for, audit_sequence
from .verified_cache import ApprovalAuthority, VerifiedStore, decide, digest

ALIASES = {
    "startup": "Will these settings preload English while switching on automatic task detection?",
    "threads": "Do these settings apply four Torch threads and prevent checkpoint preloading?",
    "auth": "Does this request reach prediction and finish with HTTP 200?",
    "errors": "Does this request call prediction and then return HTTP 422?",
}


def dataset():
    revision = file_hash(ROOT / "laya/serve.py")
    entries, requests = [], []
    for c in build_cases():
        req = {"namespace": "benchmark", "task": c["family"], "query": c["query"],
               "facts": copy.deepcopy(c["oracle_scenarios"][0]), "revision": revision}
        if not c["answerable"]:
            req["facts"].pop(c["missing_field"])
            requests.append({"id": c["id"], "category": "missing_facts", "request": req, "expected": None})
            continue
        proof = oracle(c["family"], req["facts"])
        assert proof["answer"] == c["gold"]
        entries.append({"id": c["id"], **{k: req[k] for k in ("namespace", "task", "facts", "revision")},
                        "question": req["query"], "approved_queries": [req["query"], ALIASES[c["family"]]],
                        "answer": proof["answer"], "proof": digest({"source": revision, "execution": proof}),
                        "status": "verified", "fault": None})
        requests.append({"id": c["id"] + "-exact", "category": "exact_hit", "request": req, "expected": c["gold"]})
        requests.append({"id": c["id"] + "-alias", "category": "approved_paraphrase",
                         "request": {**req, "query": ALIASES[c["family"]]}, "expected": c["gold"]})
    for family in ALIASES:
        original = next(e for e in entries if e["task"] == family)
        base = {"namespace": original["namespace"], "task": family, "facts": original["facts"],
                "revision": revision, "query": original["question"]}
        variants = {
            "changed_facts": {**base, "facts": {**base["facts"], "new_fact": "not enrolled"}},
            "wrong_namespace": {**base, "namespace": "other"},
            "stale_source": {**base, "revision": "changed-source-version"},
            "changed_meaning": {**base, "query": "Is the opposite true? " + base["query"]},
        }
        for category, req in variants.items():
            requests.append({"id": family + "-" + category, "category": category, "request": req, "expected": None})
        for fault in ("expired", "tampered", "unverified", "revoked"):
            facts = {**base["facts"], "cache_variant": fault}
            proof = oracle(family, facts)
            entry = {**original, "id": family + "-" + fault, "facts": facts, "fault": fault,
                     "answer": proof["answer"], "proof": digest({"source": revision, "execution": proof})}
            entries.append(entry)
            requests.append({"id": family + "-" + fault, "category": fault,
                             "request": {**base, "facts": facts}, "expected": None})
    return entries, requests


def model_body(req, candidates):
    state = "Question: " + req["query"] + "\nCurrent settings: " + json.dumps(req["facts"], sort_keys=True)
    choices = {}
    for index, entry in enumerate(candidates):
        state += f'\nRecord {index}: {entry["question"]} Settings: ' + json.dumps(entry["facts"], sort_keys=True)
        choices[f"c{index}"] = f"Stored record {index}"
    choices["none"] = "No applicable stored record"
    return {"state": state, "model": "english", "questions": {"decision": {
        "type": "choice", "instructions": "Which stored record answers the question for exactly the current settings? Choose none if none applies.",
        "criteria": choices}}}


def create_store(path, size, specs, store_class):
    path.mkdir(parents=True, exist_ok=False)
    authority = ApprovalAuthority(secrets.token_bytes(32))
    store = VerifiedStore(path, store_class, authority)
    now = time.time()
    for spec in specs:
        entry = {k: v for k, v in spec.items() if k != "fault"}
        entry.update(issued_at=now - 10, expires_at=now + 3600)
        fault = spec["fault"]
        if fault == "expired":
            entry["expires_at"] = now - 1
        if fault == "unverified":
            entry["status"] = "draft"
        entry = authority.approve(entry)
        if fault == "tampered":
            entry["answer"] = "no" if entry["answer"] == "yes" else "yes"
        if fault == "revoked":
            authority.revoked.add(entry["id"])
        store.add(entry)
    for index in range(size - len(specs)):
        store.add({"id": f"archive-{index}", "namespace": "benchmark", "task": "archived_inventory",
                   "question": f"Archived inventory asset label {index}", "facts": {"inventory_id": index},
                   "revision": "archived", "answer": "no", "status": "draft", "approved_queries": []})
    store.finish_writes()
    return store


def summarize(rows):
    out = {}
    for size in sorted({r["size"] for r in rows}):
        out[str(size)] = {}
        for arm in ("exact", "structured", "laya", "hybrid"):
            own = [r for r in rows if r["size"] == size and r["arm"] == arm]
            hits = [r for r in own if r["route"] == "cache"]
            out[str(size)][arm] = {
                "requests": len(own), "accepted": len(hits), "fallbacks": len(own) - len(hits),
                "wrong_accepts": sum(r["answer"] != r["expected"] for r in hits),
                "valid_hit_opportunities": sum(r["expected"] is not None for r in own),
                "missed_valid_hits": sum(r["expected"] is not None and r["route"] != "cache" for r in own),
                "laya_calls": sum(r["laya_called"] for r in own),
                "latency": percentiles([r["elapsed_ms"] for r in own]),
                "by_category": {c: {"n": len(sub := [r for r in own if r["category"] == c]),
                                     "accepted": sum(r["route"] == "cache" for r in sub),
                                     "wrong_accepts": sum(r["route"] == "cache" and r["answer"] != r["expected"] for r in sub)}
                                for c in sorted({r["category"] for r in own})}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dom-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.dom_root = args.dom_root.resolve()
    args.model_dir = args.model_dir.resolve()
    if args.out.exists() or args.out.with_suffix(".plan.json").exists() or args.work_dir.exists():
        parser.error("use new output files and work directory")
    sys.path.append(str(args.dom_root))
    from dom_api_cache.store import ResponseStore
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    helper = helper_for(args.model_dir)
    specs, cases = dataset()
    sizes = (100, 10000, 100000)
    stores, preparations = {}, []
    process = connection = None
    try:
        for size in sizes:
            start = time.perf_counter()
            store = create_store(args.work_dir / str(size), size, specs, ResponseStore)
            stores[size] = store
            prepared = []
            for case in cases:
                candidates = store.candidates(case["request"])
                body = model_body(case["request"], candidates) if candidates else None
                tokens = audit_sequence(helper, body["state"], body["questions"]["decision"]) if body else 0
                prepared.append({"id": case["id"], "candidate_ids": [e["id"] for e in candidates], "body": body, "input_tokens": tokens})
            preparations.append({"size": size, "build_ms": (time.perf_counter() - start) * 1000,
                                 "fixtures": prepared})
            print(json.dumps({"status": "store_prepared", "rows": size, "max_input_tokens": max(p["input_tokens"] for p in prepared)}), flush=True)
        sources = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [
            Path(__file__), Path(__file__).with_name("verified_cache.py"), Path(__file__).with_name("bench_reasoning.py"),
            Path(__file__).with_name("serve_local.py"), Path(__file__).with_name("bench_http.py"),
            Path(__file__).with_name("bench_published.py"), *sorted((ROOT / "laya").glob("*.py"))]}
        plan = {"format": "laya.verified_cache_plan.v1", "entries": specs, "cases": cases,
                "prepared": preparations, "source_sha256": sources,
                "dom_store_sha256": file_hash(args.dom_root / "dom_api_cache/store.py"),
                "model_files": [{"path": k, "sha256": v} for k, v in model_hashes(args.model_dir).items()],
                "protocol": {"sizes": sizes, "rounds": 3, "arms": ["exact", "structured", "laya", "hybrid"],
                             "candidate_limit": 3, "scope": "isolated local prototype, authored verified service cases and unrelated inventory filler; not all existing Dom stores or live customer traffic",
                             "trust": "application-supplied task ID, fresh facts/version, preapproved question aliases; trusted enrollment authority signs answers after executable oracle check",
                             "timing": "JSON request roundtrip serialization, real SQLite index/payload read, verification, optional actual Laya HTTP, response serialization; single client, warm process/OS cache",
                             "fallback": "returns Codex handoff packet only; Codex inference, answer accuracy, and completion latency unmeasured",
                             "not_guaranteed": "semantic equivalence for arbitrary text, correctness of an erroneous trusted approval, source authenticity, mutable-fact freshness, universal zero error"}}
        plan["plan_sha256"] = seal(plan)
        write_new(args.out.with_suffix(".plan.json"), plan)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        key = secrets.token_urlsafe(32)
        command = [sys.executable, "-m", "research.benchmarks.dom.serve_local", "--model-dir", str(args.model_dir),
                   "--device", "cuda", "--threads", "4", "--port", str(port)]
        with args.out.with_suffix(".server.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, LAYA_API_KEY=key), stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            start = time.perf_counter()
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            while True:
                if process.poll() is not None:
                    raise RuntimeError("service startup failed")
                try:
                    health, _, _, _ = http_request(connection, "/health", None, key)
                    assert health["device"] == "cuda"
                    break
                except (OSError, http.client.HTTPException):
                    connection.close()
                    if time.perf_counter() - start > 120:
                        raise RuntimeError("startup timeout")
                    time.sleep(0.1)
            startup_ms = (time.perf_counter() - start) * 1000
            warm = {"state": "Color is green", "questions": {"decision": {"type": "choice", "instructions": "Which color?",
                    "criteria": {"a": "Green", "b": "Blue"}}}, "model": "english"}
            for _ in range(3):
                http_request(connection, "/v1/systemone", warm, key)
            rows = []
            for size in sizes:
                store = stores[size]
                expected = {f["id"]: f for p in preparations if p["size"] == size for f in p["fixtures"]}
                for round_id in range(3):
                    trials = [(case, arm) for case in cases for arm in ("exact", "structured", "laya", "hybrid")]
                    random.Random(size + round_id).shuffle(trials)
                    for case, arm in trials:
                        calls = []
                        def match(req, candidates):
                            frozen = expected[case["id"]]
                            assert [e["id"] for e in candidates] == frozen["candidate_ids"]
                            body = model_body(req, candidates)
                            assert body == frozen["body"]
                            answer, ms, _, _ = http_request(connection, "/v1/systemone", body, key)
                            assert answer["usage"]["input_tokens"] == frozen["input_tokens"]
                            prediction = answer["answers"]["decision"]
                            picked = prediction["choice"]
                            calls.append({"choice": picked, "probabilities": prediction["probabilities"],
                                          "http_ms": ms, "candidate_ids": frozen["candidate_ids"]})
                            return int(picked[1:]) if picked.startswith("c") and picked[1:].isdigit() else None
                        start = time.perf_counter()
                        req = json.loads(json.dumps(case["request"]))
                        response = decide(store, req, arm, match)
                        response = json.loads(json.dumps(response))
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        rows.append({"id": case["id"], "category": case["category"], "expected": case["expected"],
                                     "size": size, "round": round_id, "arm": arm, "elapsed_ms": elapsed_ms,
                                     "laya_called": bool(calls), "match": calls[0] if calls else None, **response})
                    print(json.dumps({"status": "round_complete", "size": size, "round": round_id}), flush=True)
        assert all(file_hash(ROOT / p) == h for p, h in sources.items())
        assert file_hash(args.dom_root / "dom_api_cache/store.py") == plan["dom_store_sha256"]
        assert model_hashes(args.model_dir) == {e["path"]: e["sha256"] for e in plan["model_files"]}
        result = {"format": "laya.verified_cache_result.v1", "plan_sha256": plan["plan_sha256"],
                  "startup_ms": startup_ms, "rows": rows, "summary": summarize(rows)}
        result["result_sha256"] = seal(result)
        write_new(args.out, result)
        print(json.dumps({"status": "complete", "requests": len(rows), "laya_calls": sum(r["laya_called"] for r in rows),
                          "summary": {s: {a: {k: v for k, v in d.items() if k not in {"latency", "by_category"}} for a, d in arms.items()} for s, arms in result["summary"].items()}}), flush=True)
    finally:
        if connection:
            connection.close()
        if process:
            process.terminate()
            process.wait(timeout=30)
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    main()
