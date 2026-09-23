# Verified answer reuse: direct lookup is the useful fast path here

Measured on this machine on 2026-09-23. A verified-answer prototype used Dom's
actual SQLite `ResponseStore` for payload reads, new SQLite indexes for exact,
structured and full-text candidate lookup, and the pinned local English Laya
GPU service for optional matching. Existing user stores and Codex behavior were
not changed.

**Direct structured lookup was faster and found more valid answers than Laya
matching.** The database idea is useful for this bounded task set; the experiment
does not show that Laya accelerates store searches.

## Results at 100,000 records

Each arm processed 264 requests: 88 cases repeated three times, with 144 valid
cache-hit opportunities and 120 requests that must fall back.

| Path | Median / p95 decision latency | Valid hits found | Incorrect accepts | Laya calls |
|---|---:|---:|---:|---:|
| Exact question + verification | 0.076 / 0.374 ms | 72/144 | 0 | 0 |
| Structured lookup + verification | **0.074 / 0.292 ms** | **144/144** | **0** | 0 |
| Full-text candidates + Laya + verification | 34.341 / 37.362 ms | 78/144 | 0 | 252 |
| Structured first, then Laya on misses | 0.181 / 37.095 ms | 144/144 | 0 | 108 |

The hybrid's low median hides its slow misses: its mean was 14.47 ms versus
0.101 ms for structured lookup, and it recovered no additional answers. For
accepted hits only, structured lookup's median was 0.083 ms. Exact question
lookup missed all preapproved paraphrases; structured lookup accepted them by
checking the registered task, facts, and approved wording.

Across all sizes: **3,168 measured decisions, 1,080 actual Laya HTTP calls,
zero incorrect accepts in this test**. This is 88 authored cases with repeats,
not thousands of independent correctness examples or a zero-error guarantee.
The verifier blocked 816 Laya nominations that did not qualify for reuse.

| Store rows | Structured median / p95 | Laya path median / p95 | Structured / Laya valid hits |
|---|---:|---:|---:|
| 100 | 0.061 / 0.103 ms | 34.782 / 41.859 ms | 144 / 72 of 144 |
| 10,000 | 0.064 / 0.149 ms | 35.074 / 43.649 ms | 144 / 78 of 144 |
| 100,000 | 0.074 / 0.292 ms | 34.341 / 37.362 ms | 144 / 78 of 144 |

Full-text ranking changes with corpus statistics, so candidate lists and match
outcomes need not be identical across store sizes. Each actual candidate list
and Laya input was frozen and verified for its own store size.

## What is verified

There are 24 reusable answers to actual Laya service-behavior questions, enrolled
only after executing the existing service-code oracles. Each stores its task,
exact input facts, source revision, answer, verification-record digest, approved
question wording, validity interval, and an approval MAC. The signing authority
is separate from the database; its ephemeral key is never published.

Laya can nominate one of three retrieved records or decline. It cannot approve
a record or override these checks:

- Correct namespace and registered task.
- Exact, type-preserving facts and current source revision.
- Explicitly preapproved question wording, including registered paraphrases.
- Intact approval covering the answer and all applicability fields.
- Verified enrollment status, nonrevoked identity, and valid time interval.

The cases contain 24 exact hits, 24 approved paraphrases, eight missing-fact
requests, and four each of changed facts, wrong namespace, stale source,
changed question meaning, expired approval, tampered answer, unverified status,
and revoked approval. Some valid entries and deliberately ineligible entries
are close matches. Failed checks return a packet containing the original request,
its digest, and the rejection reason, with `route="codex"` and no cached answer.

The verifier is an **experimental application of Dom's reuse requirements**,
not a claim that the existing raw response cache already verifies semantic truth.
The actual reused Dom component is `dom_api_cache.store.ResponseStore`; the new
index, enrollment, applicability gate, and Laya matcher are prototype code in
this benchmark directory.

## Why Laya did not help this task set

The structured key already contains everything this verifier requires: task,
facts, source revision and namespace. When a valid enrolled record exists, an
indexed lookup retrieves it directly. When those conditions differ, a similar
record is not reusable. Laya adds inference and can nominate the wrong record,
but cannot create an extra valid match under this strict contract.

This is an architectural consequence of the verifier, not evidence that Laya
can never help retrieval. A broader task would need a separately validated way
to establish semantic equivalence or applicability. Laya similarity or confidence
alone cannot supply that proof. The useful policy supported here is **verified
structured hit -> reuse; otherwise -> Codex handoff**, without calling Laya for
the already-defined exact contract.

## Measurement and boundaries

- Store sizes contain 40 domain records (24 valid and 16 intentionally ineligible)
  plus unrelated, unverified inventory filler. This tests index scaling, not a
  corpus of 100,000 independently verified natural-language answers.
- The caller supplies a trusted task identity, current facts and source version.
  Automatic interpretation of arbitrary user prose is not solved by this test.
  Four paraphrases are explicitly registered; unseen wording falls back.
- Approval proves trusted enrollment and detects record changes. It cannot make
  an erroneously approved answer correct. A compromised signer, incorrect source,
  stale caller facts, or faulty contract can still cause errors.
- Single-client, warm-process/OS-cache timing starts before request JSON
  serialization and includes SQLite index/payload reads, gate work, optional real
  authenticated loopback Laya HTTP, and response JSON serialization. The direct
  paths are local function calls, not separate HTTP services. No WAN, TLS, cold
  disk, concurrent load, or parallel-store-search claim is made.
- Codex is **not invoked**. Fallback latency is time to produce its handoff packet,
  not time to complete the task. Codex accuracy, net token savings, and final
  end-to-end completion speed remain unmeasured. No desktop hook is enabled.
- Questions, cases, selected candidates, complete model inputs, source hashes and
  model hashes were frozen before inference. Maximum input: 323 tokens; actual
  token counts matched all audits, with no truncation. No tuning followed results.
- The real Laya service ran on this machine's GPU and was stopped after the run.
  Database population and service startup are excluded from warm lookup timings.

**93 tests passed** across the cache, evidence adapter and prior reasoning
evaluations. Cache tests include deliberately nominating every enrolled record
against every request, typed-fact mismatches (Boolean versus integer), namespace
separation, modified answers, expiration, revocation, duplicate-key ambiguity,
model failure, and recomputation of saved measurement counts.

## Reproduction

Requires the existing Dom checkout's `dom_api_cache` module and a local English
checkpoint. Use new output/work paths; overwrites are refused:

The exact local Dom store source hash is recorded in the plan. It is an external
dependency of this experiment, not vendored into the Laya fork; a Dom checkout
without that module cannot run the live benchmark. The saved-artifact tests do
not need that external module.

```shell
python -m research.benchmarks.dom.bench_verified_cache --dom-root /path/to/dom --model-dir /path/to/english --work-dir /path/to/new-isolated-databases --out /path/to/new-result.json
python -m pytest tests/test_verified_cache.py -q
```

- [Prototype verifier and indexes](verified_cache.py)
- [Benchmark](bench_verified_cache.py)
- [Frozen plan](results/verified_cache.v1.plan.json)
- [Measurements and nomination outputs](results/verified_cache.v1.json)

Canonical plan SHA-256:
`69e478e7ccd0050e7efdac21940710f9a684ef4217b048762e017efdd4c2605f`

Canonical result SHA-256:
`ca0b1493d532716bec8b0506c1d192325f31044e1262e342d59cd257fcff79cd`
