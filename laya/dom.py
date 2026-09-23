"""Optional Dom evidence adapter. No store access, training or extra inference.

The caller supplies already policy-filtered records. Raw model answers remain
unchanged; system abstentions are returned separately under ``decision``.
"""
from __future__ import annotations

import time

from .dom_context import EvidenceConfig, select_evidence
from .dom_requirements import apply_requirements_gate, check_requirements


class EvidenceAgent:
    """Wrap one preloaded Laya Agent with bounded evidence selection and checks."""

    def __init__(self, agent, *, config=None):
        self.agent = agent
        self.config = config or EvidenceConfig()

    def prepare(self, query, documents, choices, insufficient_choice=None):
        if (not isinstance(choices, dict) or not 2 <= len(choices) <= 20
                or any(not isinstance(k, str) or not k or len(k) > 100
                       or not isinstance(v, str) or not v.strip() or len(v) > 500 for k, v in choices.items())):
            raise ValueError("provide 2..20 bounded choice labels and descriptions")
        if insufficient_choice is not None and insufficient_choice not in choices:
            raise ValueError("insufficient choice must be registered")
        if not isinstance(documents, list) or any(not isinstance(d, dict) for d in documents):
            raise ValueError("provide a list of policy-filtered source records")
        packet = select_evidence(query, documents,
                                 count_tokens=lambda s: len(self.agent.tok(s, add_special_tokens=False)["input_ids"]),
                                 config=self.config)
        check = check_requirements(query, documents, packet)
        instruction = query
        if insufficient_choice is not None:
            instruction += " Answer using the supplied records; choose insufficient evidence when needed."
        qdef = {"type": "choice", "instructions": instruction, "criteria": choices}
        # Reject truncation, including option truncation, before any inference.
        # Both comparisons use Laya's actual renderer/tokenizer and sequence builder.
        from .common import build_sequence, render_options
        internal = self.agent._to_internal(qdef)
        for option in render_options(internal):
            rendered = " " + option.replace(self.agent.tok.mask_token, " ")
            if len(self.agent.tok(rendered, add_special_tokens=False)["input_ids"]) > 48:
                raise ValueError("choice description exceeds Laya's option token budget")
        head, _ = build_sequence(self.agent.tok, "", internal, 8192, 8192)
        full, _ = build_sequence(self.agent.tok, packet["state"], internal, 8192, 8192)
        actual, _ = build_sequence(self.agent.tok, packet["state"], internal,
                                   self.agent.cfg["max_len"], self.agent.cfg["head_max_len"])
        if len(head) > self.agent.cfg["head_max_len"] or actual != full:
            raise ValueError("question, options or evidence would be truncated")
        return {"packet": packet, "check": check, "question": qdef,
                "choices": list(choices), "insufficient_choice": insufficient_choice,
                "expected_input_tokens": len(actual)}

    def predict(self, query, documents, choices, *, insufficient_choice=None):
        start = time.perf_counter()
        prepared = self.prepare(query, documents, choices, insufficient_choice)
        prepared_at = time.perf_counter()
        prediction = self.agent.predict(prepared["packet"]["state"], {"decision": prepared["question"]})
        inferred_at = time.perf_counter()
        if prediction["usage"]["input_tokens"] != prepared["expected_input_tokens"]:
            raise RuntimeError("model input differs from audited sequence")
        raw_choice = prediction["answers"]["decision"]["choice"]
        decision = apply_requirements_gate(prepared["packet"], prepared["check"], raw_choice,
                                            choices=prepared["choices"], insufficient_choice=insufficient_choice)
        return {"prediction": prediction, "decision": decision,
                "evidence": prepared["packet"], "requirements": prepared["check"],
                "timing_ms": {"prepare": (prepared_at - start) * 1000,
                              "inference": (inferred_at - prepared_at) * 1000,
                              "gate": (time.perf_counter() - inferred_at) * 1000}}
