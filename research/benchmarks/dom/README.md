# Dom + Laya latency measurements

Measured on 2026-09-23 on this machine. These are actual local measurements,
including HTTP for the service comparison. They are **not hosted-production
latency, a T4 hardware replication, or a production reliability certification**.

## Final measured results

Warm, complete HTTP round trips on the English checkpoint:

| Device | Concurrent clients | Raw p50 / p95 | Dom + Laya p50 / p95 | Raw / Dom requests per second |
|---|---:|---:|---:|---:|
| GPU | 1 | 34.9 / 43.6 ms | **35.1 / 43.2 ms** | 28.5 / 28.1 |
| GPU | 4 | 128.0 / 145.7 ms | **135.8 / 156.7 ms** | 30.7 / 28.6 |
| CPU | 1 | 235.9 / 301.7 ms | **236.6 / 290.0 ms** | 4.10 / 4.11 |
| CPU | 4 | 964.0 / 1065.9 ms | **963.9 / 1063.5 ms** | 4.11 / 4.09 |

The typical single-client latency remains close to raw Laya on this workload.
The final differences in medians are 0.2 ms on GPU and 0.7 ms on CPU; these
small differences are affected by run noise and do not isolate a fixed overhead.
The first run measured 1.6 ms and 6.5 ms respectively. Lower p95 values for Dom
are not evidence of a speedup. Under GPU concurrency four, the final Dom median
is 6.1% higher and throughput is 6.8% lower; the queue amplifies added work.
These results replace any estimate based only on adding helper-function times.

All **768 final HTTP requests succeeded**, with exact raw-answer/probability
parity between endpoints. There are also 160 final timed original-workload
calls, for **928 final scored requests**, excluding warmups and health checks.
All final source/model hash and token/packet audits passed.

| Fresh-process measurement (one sample/device) | CPU | GPU |
|---|---:|---:|
| Process launch to health-ready | 14.46 s | 15.55 s |
| First raw inference after readiness | 461 ms | 1499 ms |
| First Dom request, after that raw request | 308 ms | 53 ms |
| Warm idle health p50 | 0.47 ms | 0.47 ms |

Preload **and warm** the service before routing latency-sensitive requests.
Model-loaded health status alone does not remove first-inference costs.

## Environment

- CPU: AMD Ryzen 7 7800X3D, 8 physical cores / 16 logical processors.
- GPU: NVIDIA RTX 2000 Ada Generation, 16 GiB; driver 595.95.
- Windows, Python 3.12; Torch 2.10.0+cu128, Transformers 5.5.0.
- Four Torch intra-op threads, one inter-op thread; one model worker.
- Laya source 0.3.9 at upstream base
  [`d120d4b`](https://github.com/NandhaKishorM/laya/commit/d120d4ba220711b93c171973118753460310e16b),
  plus this fork's optional adapter and endpoint. Core Agent/model forward unchanged.
- English and multilingual weights: `convaiinnovations/laya`, pinned revision
  `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`.
- Stock inference, default checkpoint AMP configuration. No TileLang fast
  kernels, quantization, compilation, training or remote model calls.

## Original advertised workload

The [published speed table](https://github.com/NandhaKishorM/laya/blob/d120d4ba220711b93c171973118753460310e16b/BENCHMARKS.md#speed-tesla-t4)
used a Tesla T4, Torch 2.11.0+cu128 and Transformers 5.17.0. The source fixture
is section 7 of `research/scripts/build_benchmark_nb.py`. This run copies its
exact state/questions, counts 1/5/10/50, three warmups and 20 timed calls per
condition. CUDA is synchronized before and after every timed call. A regression
test verifies fixture equality with upstream's matching `bench_latency.py`.
Timing includes tokenization and `Agent.system_one`, but excludes startup and
HTTP. It reproduces the workload on different hardware/software, not the exact
original environment or the original accuracy evaluations.

The English and multilingual checkpoint comparisons use separate loaded models.
The 32.8 ms published headline belongs to multilingual; English's published
one-question figure is 39.5 ms. Upstream now also documents faster TileLang
results on an RTX 4070 Ti SUPER; this experiment does not test that optional
fast path or claim parity with it.

| Questions per call | Published T4 English p50 | This GPU English p50 / p95 | Published T4 multilingual p50 | This GPU multilingual p50 / p95 |
|---|---:|---:|---:|---:|
| 1 | 39.5 ms | **32.1 / 36.9 ms** | 32.8 ms | **24.8 / 29.5 ms** |
| 5 | 84.5 ms | 66.4 / 67.9 ms | 40.1 ms | 32.1 / 36.2 ms |
| 10 | 158.6 ms | 118.7 / 120.4 ms | 72.3 ms | 51.8 / 54.7 ms |
| 50 | 771.3 ms | 586.3 / 591.2 ms | 337.4 ms | 274.6 / 276.5 ms |

This machine completed the same workload within the published T4 timings.
Different hardware and dependency versions prevent attributing that difference
to Dom or making an exact reproduction claim. Dom is not invoked in this
original-workload baseline.

## End-to-end HTTP protocol

The client timer starts **before JSON serialization** and stops **after the
complete response is read and decoded**. The service includes FastAPI/Uvicorn,
bearer authentication, the shared inference queue, evidence selection, complete
token-budget audits, requirements checking, Laya inference, system gating and
response serialization. The raw baseline receives exactly the same preselected
state/question that Dom passes to Laya. Dom receives the full candidate records.
Returning source/proof metadata makes its response larger; that cost is included.

There are 12 synthetic worlds (six families, complete and partially missing
evidence), each at two sizes: a small candidate set and 200 candidates. Four
rounds of 24 requests per arm at concurrency 1 and 4 yield **384 scored requests
per device**, 96 per arm/concurrency. Raw/Dom phase order alternates each round.
Connections persist per client worker. All inputs, model hashes and source
hashes are frozen before measurements, then verified again afterward.

Raw model answers and probabilities must match exactly between endpoints for
the same input. A separately attributed Dom abstention may change the system
choice. The benchmark fails on HTTP errors, token/packet drift, output-parity
failure, or source/model changes. Every benchmark owns and stops its own hidden
loopback service process. It never logs or saves its temporary bearer token.

Startup is one **fresh service process** per device, from process spawn through
model load to successful health probe. OS/disk caches are not cleared. The first
raw inference is measured separately and can incur CUDA/kernel initialization.
The first Dom request follows it, so that number is not a second independent
cold start. Each endpoint then receives three warmups. Startup has n=1 per run;
no startup percentile or cold-machine claim is made.

The one-worker queue means four simultaneous clients mainly increase waiting
time. This is a short load test, not sustained saturation/SLO testing. p95/p99
are descriptive empirical percentiles from small samples, not guarantees.
Internet/TLS/reverse-proxy costs and any external data-store fetch are excluded.
All source records are supplied in the request and must already be authorized.

## Reproduce

Use Python 3.11 or newer for the benchmark scripts (the recorded run used 3.12).
Install this branch with the service extra and test tools:

```shell
pip install -e ".[serve]" pytest httpx
python -m pytest tests/test_dom_adapter.py tests/test_dom_context.py tests/test_dom_requirements.py tests/test_serve.py -q
python tests/test_router.py
python tests/test_router_memory.py
python tests/test_lazy_import.py
```

Download the pinned model files separately. No benchmark downloads weights or
contacts an inference API. English is the snapshot root; multilingual is its
`multilingual/` subfolder. Each directory needs `rl_agent_config.json`,
`model.safetensors`, `encoder/config.json`, and `tokenizer/` files.

```shell
python -m research.benchmarks.dom.bench_published --english /path/to/english --multilingual /path/to/multilingual --device cuda --out /path/to/new-published.json
python -m research.benchmarks.dom.bench_http --model-dir /path/to/english --device cuda --out /path/to/new-http-gpu.json
python -m research.benchmarks.dom.bench_http --model-dir /path/to/english --device cpu --out /path/to/new-http-cpu.json
```

Run them sequentially on an otherwise idle machine. Outputs are exclusive-create
and refuse to overwrite existing data. HTTP plans contain the exact synthetic
wire payloads and source hashes; result files contain every latency sample and
model-answer hashes. Model weights, host logs and credentials are not published.

The final results use **v2**, recorded after fixing an option-token boundary
check. The initial v1 measurements remain available for transparency. They are
not pooled with v2 or selected by which run was faster. The final code passed
63 pytest tests, 386 upstream routing checks, the upstream router-memory test
and seven lazy-import checks. All 38 packaging/link checks passed and the local
wheel built successfully. These checks do not constitute the full upstream
test suite or independent production QA.

Final artifacts:

- [Published GPU workload, raw samples](results/published_cuda.v2.json)
- [HTTP GPU results](results/http_cuda.v2.json) and [frozen input plan](results/http_cuda.v2.plan.json)
- [HTTP CPU results](results/http_cpu.v2.json) and [frozen input plan](results/http_cpu.v2.plan.json)
- [Earlier runs](results/) retained without overwriting

Canonical result seals:

| Result | SHA-256 |
|---|---|
| Published GPU v2 | `c721fcd40be23234a062d25c03cf11918c42eed23fa7f97119d8e7281505a964` |
| HTTP GPU v2 | `423ab3db94dab45fbd0e53ed8191b332882902848f62e688c0b9142c5a2d19f7` |
| HTTP CPU v2 | `ee18ab513a1f314ac75598d7b62aa4f43d67c0ad955fd14ded7a2daa83c9120a` |

Source hashes are byte-level and therefore reflect checkout line endings on
Windows. The upstream commit, package/dependency versions, exact inputs and
model hashes are also recorded. Reproduction on another environment need not
match wall-clock results or source byte hashes after line-ending conversion.

For usage and limitations of the six registered evidence grammars, see
[the integration guide](../../../docs/dom-evidence.md).
