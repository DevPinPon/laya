"""Opt-in evidence requirements for six explicitly registered record grammars.

This checks support, not truth, and never supplies a substantive answer. Only
use with policy-filtered records whose schema the caller controls. Unsupported
questions/records pass through; arbitrary prose is not a closed-world database.
Adapted from DevPinPon/dom (Apache-2.0); imports changed for standalone packaging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import re

from .dom_context import apply_evidence_gate
from ._dom_util import digest


ID = r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*"
VALUE = r"[A-Za-z][A-Za-z0-9 ]{0,79}?"
UNKNOWN = {"unknown", "unrecorded"}


@dataclass(frozen=True)
class Requirement:
    subject: str
    kind: str
    path: tuple[str, ...]


QUERIES = (
    (rf"In which cabinet is component (?P<subject>{ID}) stored\?", "path", ("cabinet",)),
    (rf"Which color label belongs to component (?P<subject>{ID})\? Match its exact identifier\.", "path", ("label",)),
    (rf"Which region contains the cluster hosting service (?P<subject>{ID})\?", "path", ("cluster", "region")),
    (rf"Which zone contains the machine hosting the release train for artifact (?P<subject>{ID})\?", "path", ("train", "machine", "zone")),
    (rf"May build (?P<subject>{ID}) be published\? Publication is allowed exactly when its tests passed AND its review is approved\. Either failed condition forbids publication\.", "conjunction", ("tests", "review")),
    (rf"What is the most recent recorded state of service (?P<subject>{ID})\? Compare observation dates\.", "latest", ("state",)),
)
RECORDS = (
    (rf"Component (?P<entity>{ID}) is stored in the (?P<value>{VALUE}) cabinet\.", "cabinet"),
    (rf"Component (?P<entity>{ID}) carries a (?P<value>{VALUE}) label\.", "label"),
    (rf"Service (?P<entity>{ID}) runs on cluster (?P<value>{ID})\.", "cluster"),
    (rf"Cluster (?P<entity>{ID}) is located in the (?P<value>{VALUE}) region\.", "region"),
    (rf"Artifact (?P<entity>{ID}) belongs to release train (?P<value>{ID})\.", "train"),
    (rf"Release train (?P<entity>{ID}) is hosted by machine (?P<value>{ID})\.", "machine"),
    (rf"Machine (?P<entity>{ID}) is in the (?P<value>{VALUE}) zone\.", "zone"),
    (rf"Build (?P<entity>{ID}): tests (?P<value>passed|failed|unknown|unrecorded)\.", "tests"),
    (rf"Build (?P<entity>{ID}): review is (?P<value>approved|not approved|unknown|unrecorded)\.", "review"),
    (rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}}): service (?P<entity>{ID}) was (?P<value>enabled|disabled|unknown|unrecorded)\.", "state"),
)
NOTE = rf"Inventory note: (?P<entity>{ID}) is listed; purchase order archived\."


def requirements_for_query(query: str) -> Requirement | None:
    """Derive requirements from the question alone, never options or gold labels."""
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("provide a bounded nonempty query")
    for pattern, kind, path in QUERIES:
        match = re.fullmatch(pattern, query.strip(), re.IGNORECASE)
        if match:
            return Requirement(match["subject"].casefold(), kind, path)
    return None


def _parse(doc: dict) -> dict | None:
    for pattern, field in RECORDS:
        match = re.fullmatch(pattern, doc["text"].strip(), re.IGNORECASE)
        if match:
            stamp = match.groupdict().get("date")
            if stamp:
                try:
                    date.fromisoformat(stamp)
                except ValueError:
                    return None
            return {"entity": match["entity"].casefold(), "field": field,
                    "value": match["value"].casefold(), "date": stamp,
                    "source_id": doc["id"], "source_sha256": digest(doc["text"])}
    if re.fullmatch(NOTE, doc["text"].strip(), re.IGNORECASE):
        return {"source_id": doc["id"], "note": True}
    return None


def _binding(facts: list[dict], entity: str, field: str, latest: bool = False) -> dict:
    matches = [f for f in facts if f.get("entity") == entity and f.get("field") == field]
    if latest and matches:
        newest = max(f["date"] for f in matches)
        matches = [f for f in matches if f["date"] == newest]
    values = {f["value"] for f in matches}
    status = ("missing" if not values else "unknown_value" if values & UNKNOWN else
              "conflict" if len(values) > 1 else "supported")
    return {"entity": entity, "field": field, "status": status,
            "value": next(iter(values)) if status == "supported" else None,
            "sources": [{k: f[k] for k in ("source_id", "source_sha256", "date")} for f in matches]}


def check_requirements(query: str, documents: list[dict], packet: dict) -> dict:
    """Check only verbatim selected evidence; bind the result to query and corpus.

    The seals detect accidental drift, not adversarial forgery. Callers must run
    this checker on their own trusted inputs; a self-hashed packet is no signature.
    """
    apply_evidence_gate(packet, "probe", choices=["probe"])
    if (packet["query_sha256"] != digest(query) or packet["corpus_sha256"] != digest(documents)):
        raise ValueError("query or corpus differs from evidence packet")
    if (not isinstance(documents, list) or len(documents) > 1000 or any(
            not isinstance(d, dict) or set(d) != {"id", "text"}
            or not isinstance(d["id"], str) or not d["id"]
            or not isinstance(d["text"], str) or not d["text"].strip() or len(d["text"]) > 16000
            for d in documents)):
        raise ValueError("invalid bounded source records")
    by_id = {d["id"]: d for d in documents}
    selected_ids = packet["selected_ids"]
    if (len(by_id) != len(documents) or len(selected_ids) != len(set(selected_ids))
            or any(key not in by_id for key in selected_ids)):
        raise ValueError("invalid source identities")
    selected = [by_id[key] for key in selected_ids]
    state = "\n".join(d["text"] for d in selected) or "No relevant records are available."
    if (state != packet["state"] or packet["source_sha256"] != {d["id"]: digest(d["text"]) for d in selected}):
        raise ValueError("selected evidence differs from source text")
    requirement = requirements_for_query(query)
    facts = [_parse(doc) for doc in selected]
    unparsed = [d["id"] for d, fact in zip(selected, facts) if fact is None]
    bindings, gaps = [], []
    if requirement is None:
        status = "unsupported_query"
    elif packet["status"] in {"budget_limited", "missing_subject"}:
        status = "insufficient"
        gaps = [{"reason": packet["status"], "subject": requirement.subject}]
    elif unparsed or packet["status"] != "anchored":
        status = "unsupported_records"
    elif requirement.kind == "conjunction":
        bindings = [_binding(facts, requirement.subject, field) for field in requirement.path]
        # One unambiguous false premise decides an AND even if the other is
        # missing or contradictory. A contradictory false premise is not decisive.
        decisive = [b for b in bindings if b["status"] == "supported" and
                    b["value"] in {"failed", "not approved"}]
        status = "supported" if decisive or all(b["status"] == "supported" for b in bindings) else "insufficient"
        if status == "insufficient":
            gaps = [b for b in bindings if b["status"] != "supported"]
    else:
        entity = requirement.subject
        status = "supported"
        for field in requirement.path:
            binding = _binding(facts, entity, field, requirement.kind == "latest")
            bindings.append(binding)
            if binding["status"] != "supported":
                gaps.append(binding)
                status = "insufficient"
                break
            entity = binding["value"]
    body = {"format": "dom.laya_requirements_check.v1", "packet_sha256": packet["packet_sha256"],
            "query_sha256": digest(query), "corpus_sha256": digest(documents),
            "requirement": asdict(requirement) if requirement else None, "status": status,
            "bindings": bindings, "gaps": gaps, "unparsed_ids": unparsed,
            "scope": "registered question and record grammars; support in supplied corpus, not truth or general reasoning"}
    return {**body, "check_sha256": digest(body)}


def apply_requirements_gate(packet: dict, check: dict, model_choice: str, *, choices: list[str],
                            insufficient_choice: str | None = None) -> dict:
    """Return a separately attributed system abstention; preserve raw model choice."""
    old = apply_evidence_gate(packet, model_choice, choices=choices, insufficient_choice=insufficient_choice)
    if (check.get("format") != "dom.laya_requirements_check.v1"
            or check.get("check_sha256") != digest({k: v for k, v in check.items() if k != "check_sha256"})
            or check.get("packet_sha256") != packet["packet_sha256"]
            or check.get("query_sha256") != packet["query_sha256"]
            or check.get("corpus_sha256") != packet["corpus_sha256"]):
        raise ValueError("invalid or stale requirements check")
    if check["status"] != "insufficient":
        return {**old, "requirements_status": check["status"], "check_sha256": check["check_sha256"]}
    return {**old, "choice": insufficient_choice, "abstained": True,
            "decision_source": "dom_requirements_gate", "reason": "insufficient_required_evidence",
            "requirements_status": check["status"], "check_sha256": check["check_sha256"]}
