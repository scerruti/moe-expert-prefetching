"""Experiment #140, step 1: does Triton compile any kernels during a routing run?

Runs the real collection path (run_full_pipeline) with a fresh, empty Triton
cache, then counts what Triton compiled into it. If nothing was compiled, the
TRITON_DEFAULT_FP_FUSION setting cannot affect routing in this path (#133), and
the fusion-on/fusion-off comparison is not needed.

Throwaway experiment code: not merged; only the findings carry forward to #133.

A100 (Colab), from the repository root:
    python phase_1/scripts/triton_kernel_check.py --out /content/triton_check

CPU dry run with a tiny random Qwen2MoE model (checks the plumbing only;
Triton isn't used on CPU, so this run is expected to find no kernels):
    python phase_1/scripts/triton_kernel_check.py --out /tmp/triton_check --tiny

Each run must be a fresh process: Triton reads its settings when it starts.
"""

import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

# --- Settings that must be in place before torch (and Triton) are imported ---
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True, help="new or empty directory for this run")
parser.add_argument("--fusion", choices=["0", "1"], default="1", help="TRITON_DEFAULT_FP_FUSION for this run")
parser.add_argument("--per-dataset", type=int, default=100, help="prompts per dataset (GSM8K, MBPP)")
parser.add_argument("--num-runs", type=int, default=2, help="determinism orderings; 2 is the fewest that exercises that path")
parser.add_argument("--tiny", action="store_true", help="CPU dry run with a tiny random model")
args = parser.parse_args()

if "torch" in sys.modules:
    raise RuntimeError("torch was imported before the Triton settings were applied; run this as a script")

out = args.out.resolve()
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"{out} is not empty. Use a new directory, so the Triton cache starts empty.")
existing = os.environ.get("TRITON_DEFAULT_FP_FUSION")
if existing is not None and existing != args.fusion:
    raise SystemExit(
        f"TRITON_DEFAULT_FP_FUSION is already set to {existing!r} in this environment, "
        f"but this run asks for {args.fusion!r}. Unset it or pass --fusion {existing}."
    )

triton_cache = out / "triton_cache"
inductor_cache = out / "inductor_cache"
triton_cache.mkdir(parents=True)
inductor_cache.mkdir()
os.environ["TRITON_DEFAULT_FP_FUSION"] = args.fusion
os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)  # torch.compile kernels, if any

warnings.simplefilter("always")
caught = []
_showwarning = warnings.showwarning


def _record(message, category, filename, lineno, file=None, line=None):
    caught.append(f"{category.__name__}: {message}")
    _showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _record

# --- Now torch and the pipeline ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from model_loader import ModelConfig  # noqa: E402


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def make_tiny(model_dir: Path) -> str:
    """A tiny random Qwen2MoE with Qwen's real tokenizer, registered as 'tiny-qwen2moe'."""
    from transformers import AutoTokenizer, Qwen2MoeConfig, Qwen2MoeForCausalLM

    os.environ["MOE_ALLOW_CPU_MODEL_LOAD"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.get("qwen1.5-moe-a2.7b")["model_id"])
    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        shared_expert_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=4, decoder_sparse_step=1,
        norm_topk_prob=False,
    )
    Qwen2MoeForCausalLM(config).eval().save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    ModelConfig.MODELS["tiny-qwen2moe"] = {
        "model_id": str(model_dir), "router_gate_path": "mlp.gate",
        "num_experts": 8, "top_k": 4, "is_vision": False,
    }
    return "tiny-qwen2moe"


def inspect_cache(cache: Path) -> dict:
    """Triton writes one directory per compiled kernel, holding <kernel>.json and binaries."""
    files = [p for p in cache.rglob("*") if p.is_file()]
    names = sorted({p.stem for p in files if p.suffix == ".json" and not p.name.startswith("__grp__")})
    return {"files": len(files), "kernel_dirs": len([p for p in cache.iterdir() if p.is_dir()]),
            "kernel_names": names}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


report = {
    "experiment": "#140 step 1: Triton kernels compiled during a routing run",
    "commit": git_commit(),
    "settings": {"TRITON_DEFAULT_FP_FUSION": args.fusion, "per_dataset": args.per_dataset,
                 "num_runs": args.num_runs, "tiny": args.tiny},
    "versions": {p: version(p) for p in ("torch", "triton", "transformers", "accelerate")},
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}

status, error = "incomplete", None
try:
    from full_pipeline import run_full_pipeline

    model_name = make_tiny(out / "tiny_model") if args.tiny else "qwen1.5-moe-a2.7b"
    stats = run_full_pipeline(model_name=model_name, num_runs=args.num_runs,
                              subset_size=args.per_dataset, output_dir=out / "collection")
    report["pipeline"] = {k: v for k, v in vars(stats).items() if isinstance(v, (int, float, str, bool))} \
        if hasattr(stats, "__dict__") else str(stats)
    status = "ran"
except Exception:
    error = traceback.format_exc()
    print(error, file=sys.stderr)
finally:
    # The cache is inspected even if the pipeline failed, but the result is then marked incomplete.
    kernels = {"triton": inspect_cache(triton_cache), "inductor_files": sum(1 for p in inductor_cache.rglob("*") if p.is_file())}
    report["triton_imported"] = "triton" in sys.modules
    report["kernels"] = kernels
    report["triton_warnings"] = [w for w in caught if "TRITON" in w.upper()]
    compiled = kernels["triton"]["files"] > 0 or kernels["inductor_files"] > 0
    if status != "ran":
        report["result"] = "INCOMPLETE: the pipeline failed, so a kernel count of zero proves nothing"
        report["error"] = error
    elif compiled:
        report["result"] = f"KERNELS COMPILED ({len(kernels['triton']['kernel_names'])} named): run A' and B"
    else:
        report["result"] = "NO KERNELS COMPILED: the fusion setting cannot affect routing in this path"
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\n" + json.dumps({k: report[k] for k in ("result", "triton_imported", "kernels", "triton_warnings",
                                                   "versions", "device", "commit")}, indent=2, default=str))
    print(f"\nFull report: {out / 'report.json'}")

sys.exit(0 if status == "ran" else 1)
