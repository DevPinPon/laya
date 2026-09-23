"""Opt-in, bounded evidence selection for small decision models.

Callers supply policy-filtered rendered records, never an unfiltered store.
This module performs no I/O, model inference, policy bypass or task execution.
Exact identifiers are evidence anchors, not proof that a question is answerable.
Adapted from DevPinPon/dom (Apache-2.0); the standalone lexical fallback is new.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Callable

from ._dom_util import digest


IDENTIFIER = re.compile(r"(?<![\w-])(?:[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+|[A-Za-z]{1,12}\d+[A-Za-z0-9]*)(?![\w-])")
WORDS = re.compile(r"[a-z0-9]+")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class EvidenceConfig:
    max_tokens: int = 180
    max_records: int = 6
    max_hops: int = 2
    max_candidates: int = 200

    def __post_init__(self):
        for name, low, high in (("max_tokens", 16, 4096), ("max_records", 1, 25),
                                ("max_hops", 0, 3), ("max_candidates", 1, 1000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")


def identifiers(text: str) -> set[str]:
    return {match.group(0).casefold() for match in IDENTIFIER.finditer(text)}


def _contains(text: str, anchor: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(anchor) + r"(?![\w-])", text, re.IGNORECASE) is not None


def select_evidence(query: str, documents: list[dict], *, count_tokens: Callable[[str], int],
                    config: EvidenceConfig | None = None, anchors: list[str] | None = None) -> dict:
    """Select exact subjects, then follow co-occurring identifiers breadth-first.

    Explicit anchors are optional caller-supplied query subjects, not answer or
    relevance annotations. Without an identifier, fall back to bounded
    lexical selection and decline to make an absence determination.
    """
    config = config or EvidenceConfig()
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("provide a bounded nonempty query")
    if not isinstance(documents, list) or len(documents) > config.max_candidates:
        raise ValueError("candidate budget exceeded")
    by_id = {}
    for doc in documents:
        if (set(doc) != {"id", "text"} or not isinstance(doc["id"], str) or not doc["id"]
                or doc["id"] in by_id or not isinstance(doc["text"], str) or not doc["text"].strip()
                or len(doc["text"]) > 16000):
            raise ValueError("provide unique ids and bounded policy-filtered text only")
        by_id[doc["id"]] = doc
    if anchors is not None and (not isinstance(anchors, list) or len(anchors) > 8 or any(
            not isinstance(a, str) or not a.strip() or len(a) > 160 or not _contains(query, a)
            for a in anchors)):
        raise ValueError("explicit anchors must occur in the question")
    roots = sorted({a.casefold() for a in anchors} if anchors is not None else identifiers(query))
    if len(roots) > 8:
        raise ValueError("too many query subjects")
    query_terms = set(WORDS.findall(query.lower()))
    latest = bool(query_terms & {"latest", "newest", "recent", "current"})
    def rank(doc_id):
        text = by_id[doc_id]["text"]
        overlap = len(query_terms & set(WORDS.findall(text.lower())))
        return (max(DATE.findall(text), default="") if latest else "", overlap, doc_id)
    paths, ordered, seen = {}, [], set()
    missing_roots = []
    if not roots:
        # Standalone fork: no dependency on the full Dom semantic store. This
        # lexical fallback does not claim subject absence or complete support.
        ordered = sorted(by_id, key=rank, reverse=True)[:config.max_records]
        status = "unanchored_fallback"
    else:
        index = defaultdict(set)
        for doc in documents:
            for entity in identifiers(doc["text"]):
                index[entity].add(doc["id"])
        for root in roots:
            hits = [doc["id"] for doc in documents if _contains(doc["text"], root)]
            if not hits:
                missing_roots.append(root)
            for doc_id in sorted(hits, key=rank, reverse=True):
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered.append(doc_id)
                    paths[doc_id] = [doc_id]
        frontier = list(ordered)
        for _ in range(config.max_hops):
            next_frontier = []
            for parent in frontier:
                for entity in sorted(identifiers(by_id[parent]["text"]) - set(roots)):
                    for doc_id in sorted(index[entity], key=rank, reverse=True):
                        if doc_id not in seen:
                            seen.add(doc_id)
                            ordered.append(doc_id)
                            paths[doc_id] = paths[parent] + [doc_id]
                            next_frontier.append(doc_id)
            frontier = next_frontier
        status = "missing_subject" if missing_roots else "anchored"
    selected, text_parts, omitted = [], [], []
    for doc_id in ordered:
        path = paths.get(doc_id, [doc_id])
        # Never include a child after dropping its connecting source record.
        if any(parent not in selected for parent in path[:-1]):
            omitted.append(doc_id)
            continue
        candidate = "\n".join(text_parts + [by_id[doc_id]["text"]])
        if len(selected) >= config.max_records or count_tokens(candidate) > config.max_tokens:
            omitted.append(doc_id)
            continue
        selected.append(doc_id)
        text_parts.append(by_id[doc_id]["text"])
    if omitted and status == "anchored":
        status = "budget_limited"
    state = "\n".join(text_parts) or "No relevant records are available."
    if count_tokens(state) > config.max_tokens:
        raise ValueError("empty-state marker exceeds tokenizer budget")
    payload = {"format": "dom.laya_evidence_packet.v1", "query_sha256": digest(query),
               "corpus_sha256": digest(documents), "config": vars(config), "anchors": roots,
               "status": status, "missing_subjects": missing_roots, "selected_ids": selected,
               "state": state, "context_tokens": count_tokens(state), "omitted_ids": omitted,
               "paths": {doc_id: paths.get(doc_id, [doc_id]) for doc_id in selected},
               "source_sha256": {doc_id: digest(by_id[doc_id]["text"]) for doc_id in selected},
               "answerability": "not_established",
               "scope": "supplied policy-filtered corpus; presence does not prove completeness"}
    return {**payload, "packet_sha256": digest(payload)}


def apply_evidence_gate(packet: dict, model_choice: str, *, choices: list[str],
                        insufficient_choice: str | None = None) -> dict:
    """Explicit system abstention, kept separate from the unchanged model output."""
    if (packet.get("format") != "dom.laya_evidence_packet.v1"
            or packet.get("packet_sha256") != digest({k: v for k, v in packet.items() if k != "packet_sha256"})):
        raise ValueError("invalid evidence packet")
    if model_choice not in choices or (insufficient_choice is not None and insufficient_choice not in choices):
        raise ValueError("choices must be registered")
    abstain = packet["status"] in {"missing_subject", "budget_limited"}
    return {"choice": insufficient_choice if abstain else model_choice,
            "model_choice": model_choice, "decision_source": "dom_evidence_gate" if abstain else "laya",
            "abstained": abstain, "reason": packet["status"] if abstain else None,
            "packet_sha256": packet["packet_sha256"], "changes_model_probabilities": False}
