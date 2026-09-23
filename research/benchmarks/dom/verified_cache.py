"""Experimental verified-answer layer over Dom's existing ResponseStore.

Not a Codex hook. A trusted application supplies task identity and fresh facts.
Laya may nominate a record, but cannot bypass applicability or approval checks.
The HMAC approval authority is separate from the database. It attests enrollment,
not truth: enrollment must have independently verified the answer beforehand.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def normalized(text):
    # Do not collapse case/spacing inside a potentially meaningful literal.
    return text


def lookup_key(request, *, exact=False):
    value = [request["task"], request["facts"], request["revision"]]
    if exact:
        value.append(normalized(request["query"]))
    return digest(value)


class ApprovalAuthority:
    def __init__(self, key):
        if len(key) < 32:
            raise ValueError("approval key must have at least 32 bytes")
        self.key = key
        self.revoked = set()

    def approve(self, entry):
        clean = {k: v for k, v in entry.items() if k != "approval"}
        return {**clean, "approval": hmac.new(self.key, digest(clean).encode(), hashlib.sha256).hexdigest()}

    def authentic(self, entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("approval"), str):
            return False
        try:
            expected = self.approve(entry)["approval"]
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(expected, entry["approval"])


def verify(entry, request, authority, *, now=None):
    """Fail closed. No model confidence or self-declared verified flag suffices."""
    now = time.time() if now is None else now
    if not entry or not authority.authentic(entry):
        return False, "missing_or_invalid_approval"
    if entry.get("status") != "verified" or entry.get("id") in authority.revoked:
        return False, "unverified_or_revoked"
    if entry.get("namespace") != request["namespace"]:
        return False, "wrong_namespace"
    if entry.get("task") != request["task"] or digest(entry.get("facts")) != digest(request["facts"]):
        return False, "different_task_or_facts"
    if entry.get("revision") != request["revision"]:
        return False, "stale_source"
    if not entry.get("issued_at", now + 1) <= now < entry.get("expires_at", 0):
        return False, "expired_or_future"
    if normalized(request["query"]) not in {normalized(q) for q in entry.get("approved_queries", [])}:
        return False, "unverified_question_meaning"
    if not entry.get("proof") or entry.get("answer") not in {"yes", "no"}:
        return False, "invalid_verification_record"
    return True, "verified_applicable"


class VerifiedStore:
    def __init__(self, directory, store_class, authority):
        self.authority = authority
        self.payloads = store_class(directory / "answers.db")
        self.index = sqlite3.connect(directory / "search.db")
        self.index.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE catalog (id INTEGER PRIMARY KEY, namespace TEXT, record_id TEXT,
                                  structured_key TEXT, exact_key TEXT, task TEXT);
            CREATE UNIQUE INDEX record_key ON catalog(namespace, record_id);
            CREATE INDEX structured_key ON catalog(namespace, structured_key);
            CREATE INDEX exact_key ON catalog(namespace, exact_key);
            CREATE VIRTUAL TABLE search USING fts5(question, facts);
        """)

    def add(self, entry):
        req = {"task": entry["task"], "facts": entry["facts"], "revision": entry["revision"],
               "query": entry["question"]}
        cursor = self.index.execute("INSERT INTO catalog(namespace,record_id,structured_key,exact_key,task) VALUES(?,?,?,?,?)",
                                    (entry["namespace"], entry["id"], lookup_key(req), lookup_key(req, exact=True), entry["task"]))
        self.index.execute("INSERT INTO search(rowid,question,facts) VALUES(?,?,?)",
                           (cursor.lastrowid, entry["question"], json.dumps(entry["facts"], sort_keys=True)))
        self.payloads.put(namespace=entry["namespace"], cache_key=entry["id"], provider="verified-experiment",
                          model="stored-answer", harness="offline-benchmark", is_stream=False, status_code=200,
                          content_type="application/json", body=json.dumps(entry).encode())

    def finish_writes(self):
        self.index.commit()

    def get(self, namespace, record_id):
        item = self.payloads.get(namespace, record_id)
        if item is None:
            return None
        try:
            return json.loads(item.body)
        except (ValueError, UnicodeDecodeError):
            return None

    def direct(self, request, *, exact=False):
        column = "exact_key" if exact else "structured_key"
        rows = self.index.execute(f"SELECT record_id FROM catalog WHERE namespace=? AND {column}=? LIMIT 3",
                                  (request["namespace"], lookup_key(request, exact=exact))).fetchall()
        # Duplicate candidates are treated as ambiguous, rather than trusting order.
        if len(rows) != 1:
            return None, "miss_or_ambiguous"
        entry = self.get(request["namespace"], rows[0][0])
        ok, reason = verify(entry, request, self.authority)
        return (entry if ok else None), reason

    def candidates(self, request, limit=3):
        terms = list(dict.fromkeys(re.findall(r"[a-z0-9]+", (request["query"] + " " + json.dumps(request["facts"])).lower())))
        if not terms:
            return []
        match = " OR ".join('"' + t + '"' for t in terms[:80])
        rows = self.index.execute("""
            SELECT c.record_id FROM search JOIN catalog c ON c.id=search.rowid
            WHERE search MATCH ? AND c.namespace=? AND c.task=?
            ORDER BY bm25(search), c.id LIMIT ?
        """, (match, request["namespace"], request["task"], limit)).fetchall()
        return [e for row in rows if (e := self.get(request["namespace"], row[0])) is not None]

    def close(self):
        self.index.close()
        self.payloads.close()


def fallback(request, reason):
    return {"route": "codex", "answer": None, "reason": reason,
            "request": request, "request_sha256": digest(request), "codex_called": False}


def accepted(entry):
    return {"route": "cache", "answer": entry["answer"], "record_id": entry["id"],
            "proof": entry["proof"], "codex_called": False}


def decide(store, request, arm, matcher=None):
    if arm in {"exact", "structured", "hybrid"}:
        entry, reason = store.direct(request, exact=arm == "exact")
        if entry:
            return accepted(entry)
        if arm != "hybrid":
            return fallback(request, reason)
    candidates = store.candidates(request)
    if not candidates:
        return fallback(request, "no_candidates")
    try:
        picked = matcher(request, candidates)
    except (OSError, ValueError, RuntimeError):
        return fallback(request, "matcher_unavailable_or_invalid")
    if picked is None or type(picked) is not int or not 0 <= picked < len(candidates):
        return fallback(request, "no_model_match")
    entry = candidates[picked]
    ok, reason = verify(entry, request, store.authority)
    return accepted(entry) if ok else fallback(request, reason)
