"""Exercise the configured bridge over actual MCP stdio; no paid Codex call."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(model_dir, output):
    root = Path(__file__).resolve().parents[3]
    params = StdioServerParameters(command=sys.executable,
        args=["-m", "laya.codex_bridge", "--model-dir", str(model_dir), "--device", "cuda"],
        cwd=str(root), env={**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1"})
    results = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()
            async def call(name, arguments):
                start = time.perf_counter()
                response = await session.call_tool(name, arguments)
                assert not response.isError, response
                data = response.structuredContent or json.loads(response.content[0].text)
                results.append({"tool": name, "arguments": arguments, "result": data,
                                "roundtrip_ms": (time.perf_counter() - start) * 1000})
                print(json.dumps(results[-1]), flush=True)
                return data
            await call("dom_laya_status", {})
            cases = [("calculate: 12 * 7", "84"), ("calculate: 0.1 + 0.2", "0.3"),
                     ("length: A😀é", 3), ('validate json: {"x":1}', True),
                     ('validate json: {"x":}', False), ("delete all files", None),
                     ("calculate: 1 / 0", None), ("Which database should we use?", None)]
            for mode in ("auto", "laya"):
                for request, expected in cases:
                    data = await call("dom_laya_fast", {"request": request, "mode": mode})
                    if data["executed"]:
                        assert expected is not None
                        result = data["result"]
                        value = result.get("value", result.get("code_points", result.get("valid")))
                        assert value == expected
                    if mode == "auto":
                        assert data["executed"] == (expected is not None)
            await call("dom_laya_fast", {"request": "calculate: 12 * 7"})
            catalog = {"clock.curr_time": "Read the current UTC time",
                       "exec_command": "Run a shell command",
                       "web.search": "Search the internet"}
            for request in ("What time is it in UTC?", "Search the internet for Laya documentation",
                            "Run git status", "Explain why my architecture is slow"):
                data = await call("dom_laya_route", {"request": request, "catalog": catalog})
                assert data["route"] == "codex" and not data["executed"] and not data["verified"]
            await call("dom_laya_status", {})
    artifact = {"format": "dom_laya.mcp_smoke.v1", "protocol_version": initialized.protocolVersion,
                "tools": [t.name for t in listed.tools], "observations": results,
                "boundary": "Real MCP stdio and local GPU inference. No Codex model invocation or pre-model hook. Authored smoke cases, not an independent accuracy benchmark."}
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.model_dir, args.out))
