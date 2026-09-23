from __future__ import annotations

import copy

import pytest

from laya.dom_context import EvidenceConfig, apply_evidence_gate, identifiers, select_evidence



def count(text):
    return len(text.split())


def doc(key, text):
    return {"id": key, "text": text}


def test_identifier_matching_is_exact_and_case_insensitive():
    documents = [doc("near", "OBJ-17-COPY is blue."), doc("real", "obj-17 is green.")]
    packet = select_evidence("Where is OBJ-17?", documents, count_tokens=count)
    assert packet["selected_ids"] == ["real"]
    assert identifiers("OBJ-17-COPY SVC17 ITEM_17") == {"obj-17-copy", "svc17", "item_17"}


def test_absent_subject_does_not_adopt_similar_identifier_evidence():
    packet = select_evidence("Where is OBJ-17?", [doc("near", "OBJ-17-COPY is green.")], count_tokens=count)
    assert packet["status"] == "missing_subject" and packet["selected_ids"] == []
    decision = apply_evidence_gate(packet, "a", choices=["a", "b", "unknown"], insufficient_choice="unknown")
    assert decision["choice"] == "unknown" and decision["model_choice"] == "a"
    assert decision["decision_source"] == "dom_evidence_gate" and decision["changes_model_probabilities"] is False


def test_follow_two_links_preserves_sources_and_dependency_paths():
    documents = [doc("terminal", "VM-9 is in the Blue zone."), doc("first", "OBJ-17 uses RT-3."),
                 doc("middle", "RT-3 runs on VM-9."), doc("other", "OBJ-99 uses RT-99.")]
    packet = select_evidence("Which zone hosts OBJ-17?", documents, count_tokens=count)
    assert packet["selected_ids"] == ["first", "middle", "terminal"]
    assert packet["paths"]["terminal"] == ["first", "middle", "terminal"]
    assert packet["state"] == "OBJ-17 uses RT-3.\nRT-3 runs on VM-9.\nVM-9 is in the Blue zone."
    assert packet["answerability"] == "not_established"
    anchor_only = select_evidence("Which zone hosts OBJ-17?", documents, count_tokens=count, config=EvidenceConfig(max_hops=0))
    assert anchor_only["selected_ids"] == ["first"]


def test_cycles_and_duplicate_links_remain_bounded():
    documents = [doc("a", "OBJ-17 uses RT-3."), doc("b", "RT-3 links VM-9 and OBJ-17."),
                 doc("c", "VM-9 links RT-3.")]
    packet = select_evidence("Where is OBJ-17?", documents, count_tokens=count)
    assert len(packet["selected_ids"]) == len(set(packet["selected_ids"])) == 3


def test_budget_does_not_orphan_or_truncate_linked_records():
    documents = [doc("first", "OBJ-17 uses RT-3. " + "padding " * 20), doc("last", "RT-3 is blue.")]
    packet = select_evidence("Where is OBJ-17?", documents, count_tokens=count,
                             config=EvidenceConfig(max_tokens=16))
    assert packet["status"] == "budget_limited" and packet["selected_ids"] == []
    assert set(packet["omitted_ids"]) == {"first", "last"}
    assert packet["context_tokens"] <= 16


def test_recent_queries_order_whole_records_without_deleting_history():
    documents = [doc("old", "2026-01-01: SVC12 is disabled."), doc("new", "2026-05-01: SVC12 is enabled.")]
    packet = select_evidence("What is the latest state of SVC12?", documents, count_tokens=count)
    assert packet["selected_ids"] == ["new", "old"]
    assert all(d["text"] in packet["state"] for d in documents)


def test_unanchored_queries_fall_back_without_claiming_absence():
    packet = select_evidence("Where is the component?", [], count_tokens=count)
    assert packet["status"] == "unanchored_fallback"
    assert not apply_evidence_gate(packet, "a", choices=["a", "b"])["abstained"]


def test_explicit_subjects_must_come_from_query_and_cannot_be_answers():
    packet = select_evidence("Where is Golden Heron?", [doc("a", "Golden Heron lives near the coast.")],
                             anchors=["Golden Heron"], count_tokens=count)
    assert packet["selected_ids"] == ["a"]
    with pytest.raises(ValueError):
        select_evidence("Where is Golden Heron?", [], anchors=["coast"], count_tokens=count)


def test_partial_evidence_is_not_mislabeled_complete_or_forced_unknown():
    packet = select_evidence("Where does OBJ-17 run?", [doc("a", "OBJ-17 uses RT-3.")], count_tokens=count)
    assert packet["status"] == "anchored" and packet["answerability"] == "not_established"
    decision = apply_evidence_gate(packet, "b", choices=["a", "b", "unknown"], insufficient_choice="unknown")
    assert decision["choice"] == "b" and decision["decision_source"] == "laya"


def test_gate_without_unknown_option_returns_no_invented_choice():
    packet = select_evidence("Where is OBJ-17?", [], count_tokens=count)
    decision = apply_evidence_gate(packet, "a", choices=["a", "b"])
    assert decision["abstained"] and decision["choice"] is None
    with pytest.raises(ValueError):
        apply_evidence_gate(packet, "a", choices=["a", "b"], insufficient_choice="unknown")


def test_tampered_packet_cannot_force_an_abstention():
    packet = select_evidence("Where is OBJ-17?", [doc("a", "OBJ-17 is near the coast.")], count_tokens=count)
    packet["status"] = "missing_subject"
    with pytest.raises(ValueError):
        apply_evidence_gate(packet, "a", choices=["a", "b"])


@pytest.mark.parametrize("documents", [
    [doc("same", "A"), doc("same", "B")], [{"id": "a", "text": "A", "expected": "answer"}],
    [doc("a", "")], [doc("a", "x" * 16001)],
])
def test_no_label_or_unbounded_record_channel(documents):
    with pytest.raises(ValueError):
        select_evidence("Where is OBJ-17?", documents, count_tokens=count)


def test_candidate_and_config_limits_are_enforced():
    with pytest.raises(ValueError):
        select_evidence("Where is OBJ-17?", [doc("a", "A"), doc("b", "B")], count_tokens=count,
                        config=EvidenceConfig(max_candidates=1))
    with pytest.raises(ValueError):
        EvidenceConfig(max_hops=4)
