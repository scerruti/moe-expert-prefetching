#!/usr/bin/env python3
"""
Pre-test for #124: does transformers 5.23 route Qwen the same as 4.45? (#126)

Throwaway experiment; findings go to #124, the code is not merged.

Steps (each in its own process, so a different transformers version can be
installed between them):

    prepare  Tokenize the prompts once and save the token ids, so every run
             feeds the model identical inputs whatever the tokenizer version.
    capture  Run the prompts in batch orderings and save every real token's
             router logits at every layer. A hook on each router gate also
             checks, per batch, that the logits it reads equal an independent
             recomputation (hidden states @ gate weight), and, where the
             router returns its own top-k indices (transformers 5.x), that
             they match the top 4 of a float32 softmax of those logits.
    compare  Compare two captures: top-4 agreement, bitwise-identical logits,
             and how far apart the logits are in fp16 rounding steps.

Colab (A100):
    python transformers_routing_pretest.py prepare --out /content/pretest
    python transformers_routing_pretest.py capture --out /content/pretest --tag t445
    pip install -q transformers==5.23.0      # then Runtime > Restart session
    python transformers_routing_pretest.py capture --out /content/pretest --tag t523
    python transformers_routing_pretest.py compare --out /content/pretest

CPU dry run with a tiny random model: `tiny` replaces `prepare`, and
`--model <dir>` points capture at the saved tiny model.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"  # as in model_loader.py
TOP_K = 4
BATCH_SIZE = 32
ORDERS = (0, 1)  # 0 = prompt order; 1 = a fixed shuffle (seed 1)


# ---------------------------------------------------------------- prepare

def prepare(out, per_dataset):
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loaders import GSM8KLoader, MBPPLoader
    from transformers import AutoTokenizer

    gsm = GSM8KLoader().load(split="test", num_examples=per_dataset)
    mbpp = MBPPLoader().load(split="test", num_examples=per_dataset)
    texts = [(f"gsm8k-test-{i}", ex["question"]) for i, ex in enumerate(gsm)]
    texts += [(f"mbpp-test-{i}", ex["prompt"]) for i, ex in enumerate(mbpp)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    write_prompts(out, MODEL_ID, tokenizer.pad_token_id,
                  [(pid, text, tokenizer(text)["input_ids"]) for pid, text in texts])


def make_tiny(out):
    """A tiny random Qwen2MoE model and prompts, for a CPU dry run (no downloads)."""
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        shared_expert_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=TOP_K, decoder_sparse_step=1,
        norm_topk_prob=False,
    )
    model_dir = out / "tiny_model"
    Qwen2MoeForCausalLM(config).eval().save_pretrained(model_dir)
    rng = np.random.default_rng(0)
    prompts = [(f"tiny-{i}", "", rng.integers(1, 100, int(rng.integers(3, 12))).tolist()) for i in range(40)]
    write_prompts(out, str(model_dir), 0, prompts)


def write_prompts(out, model_id, pad_id, prompts):
    out.mkdir(parents=True, exist_ok=True)
    record = {"model_id": model_id, "pad_id": int(pad_id),
              "prompts": [{"id": pid, "text": text, "ids": [int(t) for t in ids]} for pid, text, ids in prompts]}
    (out / "prompts.json").write_text(json.dumps(record))
    print(f"wrote {len(prompts)} prompts, {sum(len(p[2]) for p in prompts):,} tokens, to {out / 'prompts.json'}")


# ---------------------------------------------------------------- capture

def load_model(model_path, device):
    import transformers
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if device == "cuda" else torch.float32
    dtype_arg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    placement = {"device_map": device} if device == "cuda" else {}  # device_map needs accelerate
    model = AutoModelForCausalLM.from_pretrained(model_path, **{dtype_arg: dtype}, **placement,
                                                 trust_remote_code=False)
    return model.eval()


class GateRecorder:
    """Forward hooks on every router gate: keep the logits, and check them as they arrive."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.batch = {}
        self.output_type = None
        self.max_recompute_diff = [0.0] * len(self.layers)
        self.index_mismatches = 0
        self.index_checks = 0
        self.handles = [layer.mlp.gate.register_forward_hook(self.hook(i)) for i, layer in enumerate(self.layers)]

    def hook(self, layer):
        def record(module, inputs, output):
            if self.output_type is None:
                self.output_type = (f"tuple of {len(output)}: " + ", ".join(type(o).__name__ for o in output)
                                    if isinstance(output, tuple) else type(output).__name__)
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.reshape(-1, logits.shape[-1])
            hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
            reference = F.linear(hidden, module.weight, getattr(module, "bias", None))
            diff = (logits.float() - reference.float()).abs().max().item()
            self.max_recompute_diff[layer] = max(self.max_recompute_diff[layer], diff)
            if isinstance(output, tuple) and len(output) == 3:
                ours = torch.topk(F.softmax(logits, dim=-1, dtype=torch.float), TOP_K, dim=-1).indices
                theirs = output[2].reshape(-1, output[2].shape[-1])
                same = (ours.sort(-1).values == theirs.sort(-1).values).all(-1)
                self.index_mismatches += int((~same).sum())
                self.index_checks += int(same.numel())
            self.batch[layer] = logits.detach().to("cpu", torch.float16)
        return record


def capture(out, tag, model_path, orders, batch_size):
    import transformers

    spec = json.loads((out / "prompts.json").read_text())
    prompts = spec["prompts"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path or spec["model_id"], device)
    recorder = GateRecorder(model)
    num_layers = len(recorder.layers)

    for order in orders:
        recorder.max_recompute_diff = [0.0] * num_layers
        recorder.index_mismatches = recorder.index_checks = 0
        sequence = np.arange(len(prompts)) if order == 0 else np.random.default_rng(order).permutation(len(prompts))
        per_prompt = {}
        for start in range(0, len(sequence), batch_size):
            members = sequence[start:start + batch_size]
            width = max(len(prompts[i]["ids"]) for i in members)
            ids = torch.full((len(members), width), spec["pad_id"], dtype=torch.long)
            mask = torch.zeros((len(members), width), dtype=torch.long)
            for row, i in enumerate(members):
                n = len(prompts[i]["ids"])
                ids[row, width - n:] = torch.tensor(prompts[i]["ids"])
                mask[row, width - n:] = 1
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            recorder.batch = {}
            with torch.no_grad():
                model.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                            position_ids=positions.to(device))
            if sorted(recorder.batch) != list(range(num_layers)):
                raise RuntimeError(f"router gates reported layers {sorted(recorder.batch)}")
            stacked = torch.stack([recorder.batch[l] for l in range(num_layers)], dim=1)  # [rows, layers, experts]
            if stacked.shape[0] != len(members) * width:
                raise RuntimeError(f"expected {len(members) * width} rows, got {stacked.shape[0]}")
            for row, i in enumerate(members):
                n = len(prompts[i]["ids"])
                per_prompt[int(i)] = stacked[row * width + width - n:(row + 1) * width].numpy()
        logits = np.concatenate([per_prompt[i] for i in range(len(prompts))])
        meta = {
            "tag": tag, "order": order, "transformers": transformers.__version__, "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "gate_output": recorder.output_type,
            "max_recompute_diff_by_layer": recorder.max_recompute_diff,
            "router_index_checks": recorder.index_checks, "router_index_mismatches": recorder.index_mismatches,
            "tokens": int(logits.shape[0]), "layers": num_layers, "experts": int(logits.shape[2]),
        }
        path = out / f"{tag}_order{order}.npz"
        np.savez_compressed(path, logits=logits, meta=json.dumps(meta))
        print(f"wrote {path}: {logits.shape[0]:,} tokens x {num_layers} layers x {logits.shape[2]} experts")
        print("  " + json.dumps({k: meta[k] for k in ("transformers", "device", "gate_output",
                                                      "router_index_checks", "router_index_mismatches")}))
        print(f"  largest gap from recomputation (any layer): {max(recorder.max_recompute_diff):.3g}")


# ---------------------------------------------------------------- compare

def load_run(path):
    data = np.load(path)
    return data["logits"], json.loads(str(data["meta"]))


def top_k_sets(logits):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=-1)
    return np.sort(torch.topk(probs, TOP_K, dim=-1).indices.numpy(), axis=-1)


def compare_pair(label, a_path, b_path):
    a, meta_a = load_run(a_path)
    b, meta_b = load_run(b_path)
    if a.shape != b.shape:
        raise ValueError(f"{label}: shapes differ, {a.shape} vs {b.shape}")
    decisions = a.shape[0] * a.shape[1]
    changed = (top_k_sets(a) != top_k_sets(b)).any(-1)
    bitwise = (a == b).all(-1)
    steps = np.abs(a.astype(np.float32) - b.astype(np.float32)) / np.spacing(np.abs(a)).astype(np.float32)
    worst = steps.max(-1)  # per decision, the furthest-apart expert, in fp16 rounding steps
    print(f"{label}: {meta_a['tag']} order {meta_a['order']} (transformers {meta_a['transformers']}) vs "
          f"{meta_b['tag']} order {meta_b['order']} (transformers {meta_b['transformers']})")
    print(f"  decisions compared: {decisions:,} ({a.shape[0]:,} tokens x {a.shape[1]} layers)")
    print(f"  top-4 changed: {changed.mean():.2%}   all logits bitwise identical: {bitwise.mean():.2%}")
    print(f"  largest logit difference: {np.abs(a.astype(np.float32) - b.astype(np.float32)).max():.4g}")
    print("  furthest-apart expert per decision, in fp16 steps: "
          + ", ".join(f"<= {n}: {(worst <= n).mean():.1%}" for n in (0, 1, 2, 4, 16)))
    third = a.shape[1] // 3
    for name, layers in (("first third of layers", slice(0, third)), ("middle third", slice(third, 2 * third)),
                         ("last third", slice(2 * third, None))):
        print(f"    {name}: top-4 changed {changed[:, layers].mean():.2%}, "
              f"bitwise identical {bitwise[:, layers].mean():.2%}")
    return changed.mean()


def compare(out, old, new):
    for tag in (old, new):
        meta = load_run(out / f"{tag}_order0.npz")[1]
        print(f"{tag}: transformers {meta['transformers']} on {meta['device']}; gate output {meta['gate_output']}; "
              f"router's own indices checked {meta['router_index_checks']:,}, "
              f"mismatched {meta['router_index_mismatches']:,}; largest gap from recomputation "
              f"{max(meta['max_recompute_diff_by_layer']):.3g}")
    print()
    compare_pair("Across versions, same batches", out / f"{old}_order0.npz", out / f"{new}_order0.npz")
    print()
    for tag in (old, new):
        compare_pair(f"Batch sensitivity within {tag}", out / f"{tag}_order0.npz", out / f"{tag}_order1.npz")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "tiny", "capture", "compare"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", help="capture: name for this run, e.g. t445")
    parser.add_argument("--model", help="capture: a local model directory instead of the prompts' model")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--old", default="t445")
    parser.add_argument("--new", default="t523")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.out, args.per_dataset)
    elif args.step == "tiny":
        make_tiny(args.out)
    elif args.step == "capture":
        capture(args.out, args.tag, args.model, ORDERS, args.batch_size)
    else:
        compare(args.out, args.old, args.new)


if __name__ == "__main__":
    main()
