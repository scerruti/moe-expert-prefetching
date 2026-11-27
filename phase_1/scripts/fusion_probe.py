#!/usr/bin/env python3
"""
Routing from a tiny Qwen2MoE on the GPU under one TRITON_DEFAULT_FP_FUSION value.

Used by test_fusion_routing.py (#133), which repeats #140's check: run once
with fusion on and once with it off, then compare the routing bit for bit.

    python phase_1/scripts/fusion_probe.py --fusion 0 --out DIR

Run it as a fresh process: Triton reads its settings when it starts. It sets
the variable itself (not through run_settings, which only allows 1) and uses
an empty Triton cache in DIR, so no kernel compiled under the other setting is
reused. Writes DIR/routing.npz and DIR/report.json.
"""

import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fusion", choices=["0", "1"], required=True)
parser.add_argument("--out", type=Path, required=True, help="a new directory")
args = parser.parse_args()

if "torch" in sys.modules:
    raise SystemExit("torch was imported before the fusion setting; run this as a script")
args.out.mkdir(parents=True)  # must be new, so the Triton cache starts empty
triton_cache = args.out / "triton_cache"
triton_cache.mkdir()
os.environ["TRITON_DEFAULT_FP_FUSION"] = args.fusion
os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_test_helpers import TOY_PAD_ID, left_padded_batch, make_capture, make_toy_qwen  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("Needs a CUDA GPU: Triton only compiles kernels there")

NUM_PROMPTS = 64
BATCH_SIZE = 32

# Prompts of different lengths, so batches are left-padded as in the collection
rng = np.random.default_rng(0)
prompts = [rng.integers(1, 128, size=int(rng.integers(4, 64))).tolist() for _ in range(NUM_PROMPTS)]

model = make_toy_qwen().to("cuda", torch.float16)  # fp16 on the GPU, as the collection runs (#103)
capture = make_capture(model, device="cuda")
expert_probs, selected_experts = [], []
try:
    for start in range(0, NUM_PROMPTS, BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        routing = capture.capture_batch(left_padded_batch(batch, TOY_PAD_ID), batch)
        expert_probs.append(routing.expert_probs)
        selected_experts.append(routing.selected_experts)
finally:
    capture.cleanup()

np.savez(args.out / "routing.npz",
         expert_probs=np.concatenate(expert_probs), selected_experts=np.concatenate(selected_experts))
kernels = sorted({p.stem for p in triton_cache.rglob("*.json") if not p.name.startswith("__grp__")})
(args.out / "report.json").write_text(json.dumps({
    "fusion": os.environ["TRITON_DEFAULT_FP_FUSION"],
    "triton_kernels": kernels,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}, indent=2))
