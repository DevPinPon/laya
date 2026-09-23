"""Serve one local pinned checkpoint with optional Dom endpoint on loopback."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.threads <= 64:
        parser.error("invalid port or thread limit")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      LAYA_DEVICE=args.device)
    import torch
    import uvicorn
    from laya import Router, load
    from laya.dom import EvidenceAgent
    from laya.serve import create_app
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    router = Router(models={"english": str(args.model_dir.resolve())}, device=args.device, max_loaded=1)
    agent = load(str(args.model_dir.resolve()), device=args.device)
    router.attach("english", agent)
    class PinnedRouter:
        loaded = ["english"]
        def predict(self, state, questions, model=None):
            if model not in {None, "english"}:
                raise ValueError("only the pinned English checkpoint is served")
            prediction = router.predict(state, questions, model="english")
            if agent.device.type != args.device:
                raise RuntimeError("unexpected device fallback")
            return prediction
    uvicorn.run(create_app(router=PinnedRouter(), evidence_agent=EvidenceAgent(agent)),
                host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
