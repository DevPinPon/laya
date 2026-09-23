"""Experimental local Codex MCP bridge; no interception or arbitrary execution.

Run with mcp==1.26.0, a local checkpoint and the fork on PYTHONPATH.
Model choices are proposals, never correctness or authorization evidence.
"""
from __future__ import annotations

import ast
import contextlib
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from functools import lru_cache

CATALOG = {
    "calculate": "Evaluate arithmetic with numbers and + - * / parentheses",
    "text_length": "Count Unicode code points in literal text",
    "json_validate": "Check whether literal text is valid strict JSON",
}


def parse_request(request):
    if not isinstance(request, str) or not 1 <= len(request) <= 2000:
        raise ValueError("request must contain 1..2000 characters")
    # Full matches only: no semantic assumptions about arbitrary paraphrases.
    for prefix, tool in (("calculate: ", "calculate"), ("length: ", "text_length"),
                         ("validate json: ", "json_validate")):
        if request.startswith(prefix):
            payload = request[len(prefix):]
            if tool == "calculate" and not re.fullmatch(r"[0-9.()+*/ \-]{1,200}", payload):
                raise ValueError("unsupported arithmetic")
            return tool, payload
    return None


@lru_cache(maxsize=256)
def execute_verified(tool, payload):
    """Pure functions only; cache only their successfully computed results."""
    if tool == "text_length":
        return {"code_points": len(payload)}
    if tool == "json_validate":
        def reject_constant(value):
            raise ValueError("non-JSON numeric constant")
        try:
            json.loads(payload, parse_constant=reject_constant)
            return {"valid": True}
        except (ValueError, RecursionError):
            return {"valid": False}
    if tool != "calculate":
        raise ValueError("unknown tool")
    tree = ast.parse(payload.strip(), mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("arithmetic too complex")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # Use the literal source, avoiding a binary float roundtrip.
            return Decimal(ast.get_source_segment(payload.strip(), node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            return a / b
        raise ValueError("unsupported arithmetic syntax")
    with localcontext() as ctx:
        ctx.prec = 50
        value = visit(tree.body)
    if not value.is_finite() or abs(value.adjusted()) > 1000:
        raise ValueError("result outside supported bounds")
    return {"value": str(value), "decimal_precision": 50}


class Bridge:
    def __init__(self, model_dir=None, device="cuda", predictor=None):
        self.model_dir = model_dir
        self.device = device
        self.predictor = predictor
        self.agent = None
        self.lock = threading.Lock()

    def nominate(self, request, catalog):
        if (not isinstance(request, str) or not 1 <= len(request) <= 2000
                or not isinstance(catalog, dict) or not 1 <= len(catalog) <= 8
                or any(not isinstance(k, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", k)
                       or not isinstance(v, str) or not 1 <= len(v) <= 160 for k, v in catalog.items())):
            raise ValueError("bounded request and 1..8 named tools with short descriptions required")
        # Stable synthetic labels keep model choices separate from executable names.
        names = list(catalog)
        choices = {f"t{i}": catalog[name] for i, name in enumerate(names)}
        choices["none"] = "No suitable tool or uncertain"
        with self.lock:
            if self.predictor:
                choice = self.predictor(request, choices)
            else:
                with contextlib.redirect_stdout(sys.stderr):
                    if self.agent is None:
                        if not self.model_dir or not Path(self.model_dir).is_dir():
                            raise ValueError("local checkpoint unavailable")
                        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                          TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_TORCH="1")
                        import torch
                        from laya import load
                        torch.set_num_threads(4)
                        self.agent = load(str(Path(self.model_dir).resolve()), device=self.device)
                    from laya.dom import EvidenceAgent
                    result = EvidenceAgent(self.agent).predict(
                        "Which tool best handles the user's request?", [{"id": "request", "text": request}],
                        choices, insufficient_choice="none")
                    choice = result["prediction"]["answers"]["decision"]["choice"]
        if choice not in choices or choice == "none":
            return None
        return names[int(choice[1:])]

    def fast(self, request, mode="auto"):
        start = time.perf_counter()
        response = {"route": "codex", "result": None, "executed": False,
                    "laya_called": False, "reason": "unsupported_request"}
        try:
            if mode not in {"auto", "laya"}:
                raise ValueError("mode must be auto or laya")
            parsed = parse_request(request)
            if mode == "laya":
                response["laya_called"] = True
                proposal = self.nominate(request, CATALOG)
                response["proposed_tool"] = proposal
                if parsed is None or proposal != parsed[0]:
                    response["reason"] = "proposal_not_verified"
                    return response
            if parsed:
                before = execute_verified.cache_info().hits
                result = dict(execute_verified(*parsed))
                response.update(route="verified", result=result, executed=True, tool=parsed[0],
                                reason="exact_grammar_and_pure_executor", cache_hit=execute_verified.cache_info().hits > before)
        except Exception as exc:
            # Tool or model failure must leave the original work with Codex.
            response["reason"] = "validation_or_model_failure"
            response["error_type"] = type(exc).__name__
        finally:
            response["elapsed_ms"] = (time.perf_counter() - start) * 1000
        return response

    def route(self, request, catalog):
        start = time.perf_counter()
        try:
            proposal = self.nominate(request, catalog)
            return {"route": "codex", "proposed_tool": proposal, "verified": False,
                    "executed": False, "requires_codex_review": True,
                    "elapsed_ms": (time.perf_counter() - start) * 1000}
        except Exception as exc:
            return {"route": "codex", "proposed_tool": None, "executed": False,
                    "verified": False, "reason": "model_or_input_failure", "error_type": type(exc).__name__,
                    "elapsed_ms": (time.perf_counter() - start) * 1000}


def make_server(bridge):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    server = FastMCP("dom_laya", log_level="WARNING", instructions=(
        "Experimental local helper. Use dom_laya_fast for literal arithmetic, text length or JSON validation. "
        "Pass the original request; do not silently rewrite it to obtain verification. "
        "route=codex means continue yourself. Tool nominations are unverified: check intent, arguments, "
        "permissions and freshness before invoking any real tool. This server never bypasses Codex. "
        "Use mode=laya only to test the model; auto avoids unnecessary inference."))
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=readonly)
    def dom_laya_fast(request: str, mode: str = "auto") -> dict:
        """Verified pure tasks: 'calculate: 12 * 7', 'length: hello', 'validate json: {"x":1}'.

        Exact prefixes required. Arithmetic uses 50 decimal digits; length counts Unicode code points.
        auto uses bounded verified computation/cache. laya tests model selection before verification.
        Unsupported, ambiguous, invalid, or mismatched requests return route=codex without execution.
        """
        return bridge.fast(request, mode)

    @server.tool(annotations=readonly)
    def dom_laya_route(request: str, catalog: dict[str, str]) -> dict:
        """Ask local Laya to nominate one of 1..8 available tool names and short descriptions.

        Always an unverified proposal for Codex review. Does not create arguments or execute tools.
        Model can be wrong. Pass only relevant, non-secret context and actual available tool names.
        """
        return bridge.route(request, catalog)

    @server.tool(annotations=readonly)
    def dom_laya_status() -> dict:
        """Report bridge mode and whether the local model is loaded. No inference."""
        return {"experimental": True, "transport": "stdio", "pre_model_hook": False,
                "model_loaded": bridge.agent is not None, "configured_device": bridge.device,
                "actual_device": str(bridge.agent.device) if bridge.agent else None,
                "checkpoint_available": bool(bridge.model_dir and Path(bridge.model_dir).is_dir()),
                "pure_tools": list(CATALOG), "arbitrary_execution": False}
    return server


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    # On Windows, importing NumPy/Torch after MCP's stdin reader threads start
    # can deadlock in native module initialization. Import before starting stdio.
    os.environ.update(USE_TF="0", USE_TORCH="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    bridge = Bridge(args.model_dir, args.device)
    # Finish native checkpoint initialization before MCP starts Windows input
    # threads, and remove cold inference latency from interactive tool calls.
    try:
        bridge.nominate("calculate: 1 + 1", CATALOG)
    except Exception as exc:
        print(f"Dom Laya model unavailable: {type(exc).__name__}", file=sys.stderr)
    make_server(bridge).run(transport="stdio")


if __name__ == "__main__":
    main()
