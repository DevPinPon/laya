# Experimental Dom–Laya Codex bridge

This is a callable MCP helper, not a pre-model hook. Codex still receives the user
request and chooses whether to invoke it. No saved Codex latency or tokens are
claimed. Generic tool nominations always return to Codex for review and execution.

The four tools are:

- `dom_laya_fast(request, mode="auto")`: strict literal arithmetic, Unicode code
  point count, and JSON syntax validation. Examples: `calculate: 12 * 7`,
  `length: hello`, `validate json: {"x":1}`. It uses bounded pure functions and a
  256-entry process-local cache of their computed results. This is not the earlier
  large answer-database prototype. Decimal arithmetic uses 50 significant digits.
- `dom_laya_fast(..., mode="laya")`: runs the local Laya checkpoint to choose a
  tool, then checks the choice against the exact request grammar. Mismatches,
  invalid expressions and unsupported requests return `route=codex` without
  executing a tool. Laya never supplies the computed answer.
- `dom_laya_route(request, catalog)`: nominate one of up to eight actual available
  tools, each with a short description. The result is explicitly unverified, does
  not contain generated arguments, and never executes the nominated tool.
- `dom_laya_status()`: report model loading and actual device.

Pass original requests; rewriting arbitrary natural language into a supported
grammar is a Codex interpretation, not independently verified intent. Model
confidence is never used as proof. Unsupported tasks remain with Codex. There is
no filesystem, shell, network, credential, message-sending or write executor in
this bridge. JSON validation checks syntax, not application schema or truth.

## Installation

Use a dedicated environment with the fork and `pip install '.[codex-bridge]'`.
This pins the tested MCP 1.26 SDK. Do not combine this extra with upstream's
separate `mcp` extra, which uses the 2.x SDK. Download a checkpoint separately;
the bridge forces offline model loading and never downloads weights itself.

```text
codex mcp add dom_laya --env PYTHONPATH=/absolute/path/to/laya -- /absolute/path/to/python -m laya.codex_bridge --model-dir /absolute/path/to/english --device cuda
codex mcp get dom_laya
```

Use `--device cpu` for CPU inference. Configuration registers a stdio server;
Codex starts and stops its process. Set `startup_timeout_sec = 90` in the
`mcp_servers.dom_laya` configuration table for model preload and warm-up. On
Windows, preload also avoids native-library initialization deadlocks after MCP
starts its input threads. Reconnect MCP servers
or restart the app to refresh tools in an existing session. The configured server
is removable with `codex mcp remove dom_laya`.

The [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
describes the shared configuration and connection lifecycle.

## Validation

```text
python -m pytest tests/test_codex_bridge.py -q
python -m research.benchmarks.dom.smoke_codex_bridge --model-dir /path/to/english --out smoke.json
```

The smoke runner launches the real MCP stdio process, initializes it, lists the
tools, calls every tool, runs local GPU inference, checks returned simple answers,
and stops the subprocess. This is a protocol integration test, not an independent
accuracy benchmark or a full Codex conversation latency test.

### Local smoke result, 2026-09-23

On the same RTX 2000 Ada machine used for the earlier benchmarks, all four MCP
tools were discovered and 23 calls completed over stdio. Actual device was CUDA.
The [saved observations](../research/benchmarks/dom/results/codex_bridge_smoke.v1.json)
include every request, result, and measured round trip.

- Direct mode correctly computed all five supported cases; three unsupported or
  invalid cases fell back. Round trips were 0.67–0.94 ms for supported cases.
- Laya mode accepted three of those five cases and deferred two (Unicode length
  and malformed JSON). No incorrect answer was accepted. Eight warm model calls
  took 28.29–36.96 ms round trip.
- Tool proposals selected clock, web search, and shell for the three corresponding
  requests; the reasoning request produced no proposal. These four calls took
  32.98–35.08 ms. They did not execute those tools.
- The repeated arithmetic request used the bounded computation cache.
- 30 bridge tests plus the previous 93 adapter/reasoning/cache/HTTP tests passed;
  38 packaging checks also passed.

These are small authored smoke cases. They show that the connection and fallback
work, not general tool-selection accuracy. Initialization/model warm-up is excluded
from the per-call numbers. The test uses a real MCP client, not a newly launched
Codex model conversation. The current desktop task must refresh its MCP connection
before these newly registered tools can appear in its tool catalog.
