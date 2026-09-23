# Optional Dom evidence checks

This fork adds bounded source selection and explicit evidence checks around
Laya. It keeps the original model, single forward pass, and raw prediction.
There is no training, extra model call, network retrieval, or full Dom store
dependency. The feature is opt-in and experimental.

## Use from Python

```python
import laya
from laya.dom import EvidenceAgent

agent = laya.load("/path/to/local/english-checkpoint", device="cuda")
dom = EvidenceAgent(agent)
result = dom.predict(
    query="Which region contains the cluster hosting service SVC12?",
    documents=[
        {"id": "service", "text": "Service SVC12 runs on cluster CL-9."},
        {"id": "cluster", "text": "Cluster CL-9 is located in the Coastal region."},
    ],
    choices={"a": "Coastal", "b": "Inland", "unknown": "Insufficient evidence"},
    insufficient_choice="unknown",
)
print(result["prediction"])  # unchanged Laya answer and probabilities
print(result["decision"])    # final system choice, model_choice, decision_source
print(result["requirements"]["status"])
```

The caller must supply **already authorized, policy-filtered records**. This
adapter is not an access-control filter. Defaults: 200 candidate records, six
selected records, two link hops and 180 selected-context tokenizer tokens.
Records remain verbatim. Stable exact identifiers anchor retrieval and related
identifiers connect records. The standalone fallback for questions without
identifiers is lexical selection; it does not require Dom's semantic store and
does not certify completeness.

## What the requirements check supports

The registered question and record grammars are in
[`dom_requirements.py`](../laya/dom_requirements.py): cabinet lookup, color
label, service-to-cluster-to-region, artifact-to-train-to-machine-to-zone,
publication using an explicit AND rule, and latest dated service state.
These are controlled formats. This is not a general natural-language reasoner.

Missing intermediate/terminal facts, contradictory required facts, and explicit
unknown values trigger abstention. A single unambiguous false AND prerequisite
is sufficient evidence for a negative decision, even if the other is missing.
For latest-state questions, contradictory newest observations or a newest
unknown value cannot be replaced with an older known value.

The statuses are `supported`, `insufficient`, `unsupported_query` and
`unsupported_records`. Unsupported syntax retains the earlier absence/budget
gate and does not get a completeness claim. `supported` means sufficient
consistent records under the registered grammar, not verified source truth or
a guarantee of the model's answer. The gate does not correct a model's wrong
answer when sufficient evidence is present.

Dom abstentions use `decision_source="dom_requirements_gate"`; raw model
answers/probabilities stay in `prediction`. If no insufficient choice is
registered, abstention returns `None`. Never interpret original model
probabilities as probabilities for an overridden system choice. Source hashes
detect drift; they are not signatures or protection against forged evidence.

## HTTP service

The existing `/v1/systemone` API remains unchanged. Pass an `EvidenceAgent` to
`laya.serve.create_app(router=router, evidence_agent=dom)` to explicitly enable
`POST /v1/dom/decide`. Its JSON fields are the four Python arguments above.
Both routes share the same one-worker inference queue and `LAYA_API_KEY` bearer
authorization. The new route caps request bodies at 4 MiB. Model/token budgets
and source shape are checked before inference.

For the pinned local English service used in the measurements:

```shell
python -m research.benchmarks.dom.serve_local --model-dir /path/to/english --device cuda --port 8093
```

It binds only to `127.0.0.1`, loads one checkpoint, and does no model downloads.
The service should be preloaded **and warmed with representative requests**:
health-ready only means that model loading is complete. The first GPU request
can still pay kernel/context initialization cost. Concurrent requests queue on
the single model worker; higher concurrency does not imply linear throughput.

Use this behind a deployment's own TLS, request limits, authentication, process
supervision and operational controls as appropriate. The local benchmark is
not certification of a hosted production deployment.

## Evidence and reproduction

See [benchmark report](../research/benchmarks/dom/README.md) for the original
published workload, actual HTTP timings, environment differences, startup cost,
raw samples, input plans and hashes. The included 108 synthetic schema fixtures
verify the portable helper's supported/insufficient/unsupported classifications.
They share an author and templates with the implementation and do not establish
independent real-world accuracy. Prior Dom experiments used another package
version; their accuracy percentages are not promoted as this fork's results.

The context and requirements modules were adapted from
[DevPinPon/dom](https://github.com/DevPinPon/dom), under Apache-2.0. The context
fallback and standalone packaging were changed for this fork. Laya attribution
and its Apache-2.0 license are retained.
