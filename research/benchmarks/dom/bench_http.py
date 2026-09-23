"""Measure real loopback HTTP latency, not a sum of component estimates.

Starts and stops its own preloaded service; includes a fresh-process startup,
first request, serialization, transport, queueing, evidence work and decoding.
No hosted network/TLS claim. Raw baseline receives the exact selected context
that Dom supplies to the model. Synthetic fixtures, one model, no remote calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time

from .bench_published import ROOT, UPSTREAM, MODEL_REVISION, file_hash, model_hashes, percentiles, seal


def fixtures(model_dir):
    from transformers import AutoTokenizer
    from laya.dom import EvidenceAgent
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    config = json.loads((model_dir / "rl_agent_config.json").read_text())
    class TokenizerOnly:
        tok, cfg = tokenizer, config
        @staticmethod
        def _to_internal(q):
            return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    helper = EvidenceAgent(TokenizerOnly())
    data = json.loads((ROOT / "research/benchmarks/dom/requirements_cases.v1.json").read_text())
    rows = []
    for case in sorted(data["cases"], key=lambda c: c["id"]):
        if case["cluster_id"].endswith("-1") or case["scenario"] not in {"complete_a", "partial"}:
            continue
        for size in ("small", "200_records"):
            docs = list(case["documents"])
            if size == "200_records":
                docs += [{"id": f"load-noise-{i}", "text": f"Component LOAD-{i} carries a Violet label."}
                         for i in range(200 - len(docs))]
            dom = {"query": case["query"], "documents": docs, "choices": case["options"], "insufficient_choice": "unknown"}
            prepared = helper.prepare(**dom)
            raw = {"state": prepared["packet"]["state"], "questions": {"decision": prepared["question"]}, "model": "english"}
            rows.append({"id": case["id"] + "/" + size, "size": size, "raw": raw, "dom": dom,
                         "expected_input_tokens": prepared["expected_input_tokens"],
                         "packet_sha256": prepared["packet"]["packet_sha256"],
                         "requirements_status": prepared["check"]["status"]})
    return rows


def freeze(args):
    files = [*sorted((ROOT / "laya").glob("*.py")), *sorted(Path(__file__).parent.glob("*.py")),
             ROOT / "research/benchmarks/dom/requirements_cases.v1.json"]
    body = {"format": "laya.dom_http_plan.v1", "upstream_revision": UPSTREAM, "device": args.device,
            "model_repository": "convaiinnovations/laya", "model_revision": MODEL_REVISION,
            "model_sha256": model_hashes(args.model_dir),
            "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
            "protocol": {"concurrency": [1, 4], "rounds": 4, "warmups_per_endpoint": 3,
                         "fixtures": "12 task worlds at small and 200-record candidate sizes; 24 per phase",
                         "phase_order": "raw/dom in even rounds, dom/raw in odd rounds",
                         "timing": "Client perf_counter from JSON serialization to complete response JSON decoding",
                         "transport": "HTTP/1.1 on loopback, bearer enabled, persistent connection per worker",
                         "server": "one shared inference queue; preloaded English checkpoint; upstream stock forward",
                         "startup": "one fresh service process per device, includes Python import/model load until health ready; OS file cache not cleared",
                         "scope": "Measured local service path; no hosted network, TLS, reverse proxy, WAN, sustained soak or production SLO claim"},
            "environment": {"os": platform.system(), "os_release": platform.release(), "python": platform.python_version(),
                            "threads": args.threads, "interop_threads": 1,
                            "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "fastapi", "uvicorn")}},
            "fixtures": fixtures(args.model_dir)}
    return {**body, "plan_sha256": seal(body)}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def request(connection, path, body, key):
    start = time.perf_counter()
    wire = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    connection.request("POST" if body is not None else "GET", path, body=wire, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    decoded = json.loads(payload)
    elapsed = (time.perf_counter() - start) * 1000
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {str(decoded)[:200]}")
    return decoded, elapsed, len(wire or b""), len(payload)


def drive(args, plan, port, key):
    def connection():
        return http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    first = connection()
    row = plan["fixtures"][0]
    _, raw_first, _, _ = request(first, "/v1/systemone", row["raw"], key)
    _, dom_first, _, _ = request(first, "/v1/dom/decide", row["dom"], key)
    for _ in range(3):
        request(first, "/v1/systemone", row["raw"], key)
        request(first, "/v1/dom/decide", row["dom"], key)
    health = [request(first, "/health", None, key)[1] for _ in range(20)]
    first.close()
    rows, phases = [], []
    local = threading.local()
    connections, connection_lock = [], threading.Lock()
    def job(arm, fixture, concurrency, round_id):
        if not hasattr(local, "conn"):
            local.conn = connection()
            with connection_lock:
                connections.append(local.conn)
        response, elapsed, sent, received = request(local.conn, "/v1/systemone" if arm == "raw" else "/v1/dom/decide",
                                                    fixture[arm], key)
        prediction = response if arm == "raw" else response["prediction"]
        if prediction["usage"]["input_tokens"] != fixture["expected_input_tokens"]:
            raise RuntimeError("HTTP model input differs from frozen fixture")
        if arm == "dom" and (response["evidence"]["packet_sha256"] != fixture["packet_sha256"]
                              or response["requirements"]["status"] != fixture["requirements_status"]):
            raise RuntimeError("HTTP evidence differs from frozen fixture")
        return {"id": fixture["id"], "size": fixture["size"], "arm": arm, "concurrency": concurrency,
                "round": round_id, "elapsed_ms": elapsed, "request_bytes": sent, "response_bytes": received,
                "answer_sha256": seal(prediction["answers"]), "input_tokens": prediction["usage"]["input_tokens"],
                "decision": response.get("decision"), "server_timing_ms": response.get("timing_ms")}
    try:
        for concurrency in (1, 4):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for round_id in range(4):
                    for arm in (("raw", "dom") if round_id % 2 == 0 else ("dom", "raw")):
                        started = time.perf_counter()
                        futures = [pool.submit(job, arm, fixture, concurrency, round_id) for fixture in plan["fixtures"]]
                        phase = [f.result() for f in futures]
                        elapsed = time.perf_counter() - started
                        rows.extend(phase)
                        phases.append({"arm": arm, "concurrency": concurrency, "round": round_id,
                                       "requests": len(phase), "elapsed_s": elapsed, "requests_per_s": len(phase) / elapsed})
                        print(json.dumps({"phase": arm, "concurrency": concurrency, "round": round_id,
                                          "p50_ms": percentiles([r["elapsed_ms"] for r in phase])["p50_ms"]}), flush=True)
    finally:
        for conn in connections:
            conn.close()
    # The timing treatment must not alter raw model output for the same input.
    for concurrency in (1, 4):
        for fixture in plan["fixtures"]:
            hashes = {r["answer_sha256"] for r in rows if r["id"] == fixture["id"] and r["concurrency"] == concurrency}
            if len(hashes) != 1:
                raise RuntimeError("raw model output parity failed")
    summary = {}
    for concurrency in (1, 4):
        summary[str(concurrency)] = {}
        for arm in ("raw", "dom"):
            selected = [r for r in rows if r["concurrency"] == concurrency and r["arm"] == arm]
            own_phases = [p for p in phases if p["concurrency"] == concurrency and p["arm"] == arm]
            summary[str(concurrency)][arm] = {**percentiles([r["elapsed_ms"] for r in selected]),
                "requests_per_s": len(selected) / sum(p["elapsed_s"] for p in own_phases),
                "by_size": {size: percentiles([r["elapsed_ms"] for r in selected if r["size"] == size])
                            for size in ("small", "200_records")}}
    return {"first_raw_request_ms": raw_first, "first_dom_request_ms": dom_first,
            "idle_health": percentiles(health), "summary": summary, "rows": rows, "phases": phases,
            "raw_model_output_parity": True, "failed_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve(strict=True)
    plan_path = args.out.with_suffix(".plan.json")
    if args.out.exists() or plan_path.exists():
        parser.error("refusing to overwrite plan or results")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    plan = freeze(args)
    write_new(plan_path, plan)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, LAYA_API_KEY=key)
    command = [sys.executable, "-m", "research.benchmarks.dom.serve_local", "--model-dir", str(args.model_dir),
               "--device", args.device, "--threads", str(args.threads), "--port", str(port)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = args.out.with_suffix(".server.log")
    start = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=flags)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("service startup failed; inspect local server log")
                probe = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    payload, _, _, _ = request(probe, "/health", None, key)
                    if payload.get("device") != args.device:
                        raise RuntimeError("service device mismatch")
                    break
                except (OSError, http.client.HTTPException):
                    if time.perf_counter() - start > 120:
                        raise RuntimeError("service startup timed out")
                    time.sleep(0.1)
                finally:
                    probe.close()
            startup_ms = (time.perf_counter() - start) * 1000
            print(json.dumps({"status": "ready", "startup_ms": startup_ms}), flush=True)
            result = drive(args, plan, port, key)
        finally:
            process.terminate()
            process.wait(timeout=30)
    if plan["model_sha256"] != model_hashes(args.model_dir) or any(
            file_hash(ROOT / path) != expected for path, expected in plan["source_sha256"].items()):
        raise RuntimeError("model or code changed during measurement")
    body = {"format": "laya.dom_http_latency.v1", "plan_sha256": plan["plan_sha256"],
            "environment": plan["environment"], "device": args.device, "protocol": plan["protocol"],
            "startup_ms": startup_ms, **result}
    write_new(args.out, {**body, "result_sha256": seal(body)})
    print(json.dumps({"status": "complete", "requests": len(result["rows"]), "output": args.out.name}), flush=True)


if __name__ == "__main__":
    main()
