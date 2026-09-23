"""Reproduce the published T4 latency workload on explicitly recorded hardware.

Run from the repository root: python -m research.benchmarks.dom.bench_published
--english PATH --multilingual PATH --device cuda --out NEW.json
No training, model download, fast kernels, or changes to the Agent forward.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = "d120d4ba220711b93c171973118753460310e16b"
MODEL_REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
STATE = {"ticket": {"subject": "Payout failing", "messages": [{"from": "customer",
         "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}


def questions(n):
    return {f"q{i}": Q_NOUL if i % 2 else Q_CHOICE for i in range(n)}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def model_hashes(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def percentiles(samples):
    import numpy as np
    return {"n": len(samples), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "p99_ms": float(np.percentile(samples, 99)),
            "mean_ms": float(np.mean(samples)), "max_ms": max(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--multilingual", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite results")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    import laya
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute CPU")
    files = [Path(__file__), ROOT / "research/scripts/build_benchmark_nb.py", *sorted((ROOT / "laya").glob("*.py"))]
    frozen = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
    baseline = json.loads((ROOT / "research/results/t4_colab_benchmark.json").read_text())
    result = {"format": "laya.dom_published_latency.v1", "upstream_revision": UPSTREAM,
              "model_repository": "convaiinnovations/laya", "model_revision": MODEL_REVISION,
              "source_sha256": frozen, "fixture_sha256": seal([STATE, questions(50)]),
              "environment": {"os": platform.system(), "os_release": platform.release(),
                              "python": platform.python_version(), "laya_source_version": laya.__version__,
                              "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
                              "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
              "protocol": {"warmups": 3, "repetitions": 20, "question_counts": [1, 5, 10, 50],
                           "timing": "Agent.system_one, includes tokenization; CUDA synchronized before/after each timed call",
                           "deviations": "Different hardware, OS and dependency versions from original T4 run; current upstream stock runtime; no TileLang fast path",
                           "scope": "Latency workload reproduction, not a repeat of accuracy datasets or exact T4 environment"},
              "published_t4": baseline["latency"], "models": {}}
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    for name, directory in (("english", args.english), ("multilingual", args.multilingual)):
        hashes = model_hashes(directory)
        started = time.perf_counter()
        agent = laya.load(str(directory.resolve()), device=args.device)
        sync()
        row = {"files_sha256": hashes, "load_ms": (time.perf_counter() - started) * 1000,
               "amp_dtype": str(agent.dtype), "latency": {}}
        for n in (1, 5, 10, 50):
            qs = questions(n)
            for _ in range(3):
                agent.system_one(STATE, qs)
            samples, answers = [], []
            for _ in range(20):
                sync()
                start = time.perf_counter()
                answer = agent.system_one(STATE, qs)
                sync()
                samples.append((time.perf_counter() - start) * 1000)
                if agent.device.type != args.device or len(answer["answers"]) != n:
                    raise RuntimeError("device fallback or invalid response")
                answers.append(seal(answer["answers"]))
            row["latency"][str(n)] = {**percentiles(samples), "input_tokens": answer["usage"]["input_tokens"],
                                      "answer_hashes": answers, "ms_per_question": percentiles(samples)["p50_ms"] / n}
            print(json.dumps({"model": name, "questions": n, "p50_ms": row["latency"][str(n)]["p50_ms"]}), flush=True)
        if model_hashes(directory) != hashes:
            raise RuntimeError("model files changed during benchmark")
        result["models"][name] = row
        del agent
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if frozen != {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}:
        raise RuntimeError("benchmark sources changed during measurement")
    result["result_sha256"] = seal(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "complete", "result_sha256": result["result_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
