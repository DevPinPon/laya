# Operational reasoning: current integration is not reliable

Measured 2026-09-23 through the actual local Laya HTTP service on an RTX 2000 Ada.
The lightweight Dom adapter provides a small observed improvement on this set, but
**does not establish useful operational reasoning accuracy**. It matches an
always-yes baseline and fails every insufficient-information case.

A requested follow-up also tested Dom's actual semantic MoE router. With all
evidence and expert instructions, it achieved **50.0% answerable accuracy and
4.2% correct abstention**, so the broader reasoning conclusion remains negative.

## Results

32 operational scenarios, three cyclic answer orders, 288 scored HTTP model
requests. Dom's raw model and final system scores reuse the same 96 requests.
The other arms each use 96 requests. Three unrelated warm-ups are excluded.

| Condition | Answerable questions | Insufficient information | Overall | Cases correct in all three orders |
|---|---:|---:|---:|---:|
| Laya, all supplied evidence | 30/72 (41.7%) | 0/24 (0%) | 30/96 (31.3%) | 10/32 |
| Laya, Dom-selected evidence | 36/72 (50.0%) | 0/24 (0%) | 36/96 (37.5%) | 12/32 |
| Dom + Laya final decision | 36/72 (50.0%) | 0/24 (0%) | 36/96 (37.5%) | 12/32 |
| Laya, only oracle-relevant records | 34/72 (47.2%) | 0/24 (0%) | 34/96 (35.4%) | 11/32 |
| Always answer yes, no inference | 36/72 (50.0%) | 0/24 (0%) | 36/96 (37.5%) | 12/32 |

Dom improved two cases, regressed none, and tied 30 versus all-evidence Laya.
There are only four related task families; the repeated option orders are not
96 independent tasks. The six additional correct predictions do not establish
a generalizable advantage. Dom + Laya answered yes 86 times, no 10 times, and
insufficient evidence zero times.

Overall accuracy by family (each includes six answerable and two incomplete
scenarios, each tested in three orders):

| Family | All evidence | Dom + Laya | Oracle-relevant evidence |
|---|---:|---:|---:|
| Preload English AND enable automatic task detection | 25.0% | 37.5% | 37.5% |
| Apply four Torch threads AND skip preload | 25.0% | 37.5% | 37.5% |
| Invoke the router AND return HTTP 200 | 37.5% | 37.5% | 37.5% |
| Invoke the router AND return HTTP 422 | 37.5% | 37.5% | 29.2% |

## What was tested

The user selected real Laya service behavior as the target workload. These are
**authored operational scenarios grounded in real code**, executed through the
local production API implementation. They are not captured customer traffic,
an independently authored benchmark, or proof of production readiness.

The model sees human-readable rules paraphrased from `laya/serve.py`, runtime
settings, and four unrelated but true service facts. It must combine rules,
configuration values, defaults, authentication precedence, and exception
behavior. For example, it must distinguish a predictor exception that becomes
HTTP 422 from an authentication failure that prevents prediction altogether.

Labels are established by executing the actual `build_router` and `create_app`
implementations. Startup tests substitute a recording Router and a recording
Torch thread setter to prevent checkpoint loading or hardware changes. HTTP
oracle tests run the actual application using FastAPI TestClient and a predictor
with a specified success or exception outcome. Model predictions never set the
gold labels, and no model judges the answers.

Each of the eight incomplete scenarios omits one runtime field. Two concrete
values for that field are executed and confirmed to produce opposite answers;
the visible evidence therefore cannot determine yes or no. Both executions
and their inputs are preserved as oracle witnesses outside the model payload.
The 24 complete scenarios are balanced: 12 yes and 12 no.

The oracle-relevant control supplies the same rule and runtime records with
irrelevant records removed; it supplies no answer or computed conclusion. Its
weak performance shows that missing retrieval alone does not explain the
failure. It is not a mathematical upper bound: record order and distractors can
affect this model's predictions.

## Why the earlier 96.6% does not transfer

The earlier Dom requirements result came from a different, controlled-schema
dataset and package version. In this evaluation all 96 Dom requests report
`unsupported_query`: these operational questions do not match the six registered
requirements grammars. The adapter correctly leaves the raw choice unchanged;
its specialized abstention checks provide no improvement here.

The standalone selector uses its unanchored lexical fallback on these questions.
It preserves all designated support records for 22/32 cases (18/24 answerable).
It omits the Boolean-setting rule in the thread scenarios and the runtime record
in two incomplete scenarios. These are retrieval limitations in addition to the
model's inability to reliably apply the complete supplied rules. No completeness
claim is returned for this fallback.

This result supports keeping the integration scoped to validated decision types.
It does not support presenting Dom as a general reasoning upgrade. Improving
broader reasoning would require separately evaluated capabilities, such as
explicit executable checks for known decisions or escalation to a stronger
reasoning model. None was added or tuned against this evaluation.

## Follow-up: actual Dom reasoning router

At the user's request, a separate comparison invokes the real, unmodified
`SemanticSubstrateTools.moe_route` from the Dom checkout. This activates its
built-in `reasoning_circuit` and, where runtime facts are selected,
`evidence_grounding` experts. It is not the lightweight fork selector.

The actual router selects typed semantic records and creates a hydration packet.
It does not execute a rule solver or run independent expert models. The same
unchanged English Laya checkpoint still produces the answer. The two arms use:

1. **MoE context:** selected original record texts, with the original question.
2. **MoE guided:** the router's native `hydration_text`, with its selected expert
   role instructions appended to the original question. Choices remain identical.

These are explicitly exploratory follow-ups on the same already evaluated cases,
not a new holdout. Both arms were frozen before their predictions. The compact
run used six cells, two experts, and Dom's 180 estimated-token selection budget.
After observing missing support, a separately frozen control increased the
limits to eight cells and 600 estimated tokens. Both versions are preserved.
Dom's estimate differs from the tokenizer-based budget in the lightweight
adapter; these are two operating points, not equal-budget performance claims.

| Actual Dom router condition | Answerable | Insufficient information | Overall |
|---|---:|---:|---:|
| Compact MoE context | 39/72 (54.2%) | 0/24 (0%) | 39/96 (40.6%) |
| Compact MoE guided | 35/72 (48.6%) | 0/24 (0%) | 35/96 (36.5%) |
| Full-evidence MoE context | 37/72 (51.4%) | 0/24 (0%) | 37/96 (38.5%) |
| Full-evidence MoE guided | 36/72 (50.0%) | 1/24 (4.2%) | 37/96 (38.5%) |

The expanded control retains all eight supplied records in every case. Complete
guided inputs use at most 457 of 512 tokens with no head or state truncation.
Therefore retrieval omission cannot explain its remaining failures. Its one
correct insufficient-information prediction is one answer-order variant, not
reliable abstention. The full guided result is only one prediction above the
always-yes overall baseline, and equals that baseline on answerable questions.

Each router request gets a fresh temporary Dom workspace containing only the
case's original records. Rules, including distractors, are typed as procedures;
runtime settings are typed as facts. Titles and confidence are neutral. Gold
labels, oracle support IDs, prior results, and project memories are not passed
to routing. Each live route is compared with its frozen route before inference.
The 24 imported Dom source modules are hash-checked before and after execution.

These follow-ups add 384 HTTP requests, bringing the total to **672 scored model
requests** across all three runs. Guided median route-plus-HTTP latency was
53.3 ms compact and 61.4 ms with full evidence. This includes fresh workspace
staging and routing on each request; it is an experimental client-side connector,
not a deployed Dom-routing endpoint. It does not establish cached-service latency.

Router reproduction additionally requires the matching Dom checkout:

```shell
python -m research.benchmarks.dom.bench_reasoning_router prepare --dom-root /path/to/dom --model-dir /path/to/english --plan /path/to/router.plan.json
python -m research.benchmarks.dom.bench_reasoning_router run --dom-root /path/to/dom --model-dir /path/to/english --plan /path/to/router.plan.json --out /path/to/router.result.json
```

Use module `bench_reasoning_router_full` for the expanded-budget control.
Plans reference the frozen baseline plan. Exact Dom module hashes, native
hydration packets, expert roles, choices, token audits, and all outputs are in:

- [Compact router plan](results/reasoning_moe.v1.plan.json) and [results](results/reasoning_moe.v1.json).
- [Full-evidence router plan](results/reasoning_moe_full.v1.plan.json) and [results](results/reasoning_moe_full.v1.json).

The expanded runner is a separate frozen copy so the compact runner's original
source hash remains reproducible. No routing, inference, or evidence modules
were modified during this evaluation.

## Execution and verification

- Same English checkpoint, pinned model revision
  `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`, for every arm.
- All inference ran on the local GPU through authenticated loopback HTTP, using
  the existing single-worker service and unchanged Dom integration.
- Questions, labels, option orders, evidence, code hashes, model hashes, and
  expected input token counts were frozen before inference. Hashes and executable
  oracles were rechecked after the run. No inputs were changed after seeing scores.
- Every actual input token count matched its frozen audit. The maximum complete
  input was 343 tokens with all evidence and 232 with Dom; no truncation occurred.
- 288/288 scored HTTP requests succeeded. No inference or evidence implementation
  changes were made. The temporary service was stopped after the run.
- **72 tests passed**, including nine evaluation checks that replay executable
  oracles, verify ambiguity witnesses, recompute scores, verify seals and option
  parity, and confirm that unsupported checks preserved raw model choices.
  The router checks also verify both follow-ups' seals, scores, source-text
  fidelity, expert instructions, and unchanged option definitions.

Observed warm HTTP medians were 35.6 ms for all evidence, 36.5 ms for Dom, and
33.5 ms for oracle-relevant evidence (p95: 41.7, 43.1, and 41.8 ms respectively).
These are secondary timings from one shuffled accuracy run, not a replacement
for the repeated [latency benchmark](README.md). Fresh service startup was 15.9 s.

## Reproduction and artifacts

From the repository root, using the installed serve/test dependencies and local
English checkpoint, choose new output filenames (overwrites are refused):

```shell
python -m research.benchmarks.dom.bench_reasoning prepare --model-dir /path/to/english --plan /path/to/new.plan.json
python -m research.benchmarks.dom.bench_reasoning run --model-dir /path/to/english --plan /path/to/new.plan.json --out /path/to/new.result.json
python -m pytest tests/test_dom_reasoning_benchmark.py -q
```

The run command uses CUDA and performs no downloads. The frozen plan includes
source excerpts and oracle witnesses as metadata; only the explicit `dom`,
`raw_all`, and `oracle_support` request bodies are sent to inference.

- [Runner and case builder](bench_reasoning.py)
- [Frozen plan, rule provenance, labels, and input audits](results/reasoning_service.v1.plan.json)
- [All predictions, probabilities, decisions, timings, and scores](results/reasoning_service.v1.json)

Canonical plan SHA-256:
`5fe54c79296ac26a82c7dcd1f39d40ab48f1dd1841763997c351b84abe579505`

Canonical result SHA-256:
`3b72abced9ce2bc6e2493bbc5e7aeb556ae5bb32141d3a4a4dbebec7edce11ef`

Byte-level source hashes reflect the Windows checkout used for measurement.
Other checkouts may use different line endings; reproductions create their own
source manifest rather than changing these frozen files.
