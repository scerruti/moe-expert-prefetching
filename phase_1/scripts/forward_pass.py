#!/usr/bin/env python3
"""
Forward pass integration with hook-based routing capture (Issue #15).

Combines:
- Tokenized prompts (from tokenizer.py, Issue #14)
- Model and hooks (from pytorch_forward_hooks.py, Issue #5)

Executes forward pass and captures routing telemetry for Phase 1 analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from model_loader import ModelLoader
from pytorch_forward_hooks import RoutingDataBuffer, register_router_hooks
from tokenizer import TokenizationPipeline

logger = logging.getLogger(__name__)


@dataclass
class TokenRoutingData:
    """Routing data for a single token across all layers."""
    sequence_id: str
    token_id: int
    token_str: str
    token_position: int
    layer_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptRoutingData:
    """Complete routing telemetry for one prompt."""
    sequence_id: str
    model: str
    num_tokens: int
    num_layers: int
    token_routing: List[TokenRoutingData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRouting:
    """
    Routing for one batch as arrays: one row per (prompt, token, layer),
    ordered by prompt, then token position, then layer.
    """
    prompt_index: np.ndarray  # [rows] index into the batch
    token_position: np.ndarray  # [rows] position within the prompt
    layer_index: np.ndarray  # [rows]
    expert_probs: np.ndarray  # [rows, num_experts] float32
    selected_experts: np.ndarray  # [rows, k] int16
    selected_probs: np.ndarray  # [rows, k] float32
    num_layers: int
    expert_logits: Optional[np.ndarray] = None  # [rows, num_experts], only if requested


class ForwardPassCapture:
    """Execute forward pass with routing data capture."""

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize forward pass capture.

        Args:
            model_name: Model name (e.g., "qwen1.5-moe-a2.7b")
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.routing_buffer = None
        self.last_timing: Dict[str, float] = {}  # seconds, for the most recent run_forward_pass

    def load_model(self) -> None:
        """Load model and tokenizer via ModelLoader."""
        logger.info(f"Loading {self.model_name}...")
        loader = ModelLoader(self.model_name)
        self.model, self.tokenizer, self.config = loader.load()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"✅ Model loaded and moved to {self.device}")

    def setup_hooks(self) -> RoutingDataBuffer:
        """Register routing hooks on all router layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.routing_buffer = RoutingDataBuffer()
        self.hooks = register_router_hooks(
            self.model,
            self.routing_buffer
        )
        logger.info(f"✅ Registered {len(self.hooks)} hooks on router layers")
        return self.routing_buffer

    def run_forward_pass(
        self,
        token_tensors: Dict[str, torch.Tensor],
        sequence_ids: List[int],
        token_ids_list: List[List[int]],
        token_strs_list: List[List[str]],
    ) -> List[PromptRoutingData]:
        """
        Run forward pass and capture routing data, one object per prompt.

        Built on capture_batch(); the collection pipeline uses capture_batch()
        directly, which avoids creating Python objects per token and layer.

        Args:
            token_tensors: Output from tokenizer (input_ids, attention_mask)
            sequence_ids: Sequence IDs for each prompt
            token_ids_list: Token IDs for each prompt
            token_strs_list: Token strings for each prompt

        Returns:
            List of PromptRoutingData with complete routing telemetry
        """
        batch = self.capture_batch(token_tensors, token_ids_list, include_logits=True)
        start = time.perf_counter()
        routing_data_list = self._routing_from_batch(batch, sequence_ids, token_ids_list, token_strs_list)
        self.last_timing["extract"] += time.perf_counter() - start
        logger.info(f"✅ Captured routing data for {len(routing_data_list)} prompts")
        return routing_data_list

    def capture_batch(
        self,
        token_tensors: Dict[str, torch.Tensor],
        token_ids_list: List[List[int]],
        include_logits: bool = False,
    ) -> BatchRouting:
        """Run the forward pass and return the batch's routing as arrays (#106)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.routing_buffer is None:
            raise RuntimeError("Hooks not setup. Call setup_hooks() first.")

        input_ids = token_tensors["input_ids"].to(self.device)
        attention_mask = token_tensors["attention_mask"].to(self.device)
        logger.debug(f"Running forward pass on {input_ids.shape[0]} prompts...")

        # Run the base transformer, not the causal-LM wrapper: the router hooks
        # are inside its layers, and the LM head's vocabulary logits (151,936
        # per token, upcast to fp32) are never used. At batch 32 they alone
        # needed 3.84 GiB and ran the A100 out of memory (#106).
        transformer = getattr(self.model, "model", self.model)
        start = time.perf_counter()
        with torch.no_grad():
            _ = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        forward_done = time.perf_counter()

        batch = self._gather_batch(attention_mask.cpu(), token_ids_list, include_logits)
        self.last_timing = {
            "forward": forward_done - start,
            "extract": time.perf_counter() - forward_done,
        }
        return batch

    def _gather_batch(
        self,
        attention_mask: torch.Tensor,
        token_ids_list: List[List[int]],
        include_logits: bool,
    ) -> BatchRouting:
        """
        Gather every prompt's real-token rows, for every layer, from the hook buffer.

        The router gate runs on flattened hidden states, so each layer's event
        holds tensors of shape [batch_size * padded_len, num_experts]: prompt b
        owns rows b * padded_len to (b + 1) * padded_len - 1. Prompts are
        left-padded, so within that block the prompt's real tokens are the
        rows where attention_mask is 1 (#99).
        """
        batch_size, padded_len = attention_mask.shape
        events = {event["layer_idx"]: event for event in self.routing_buffer.get_all()}
        layers = sorted(events)

        rows, prompt_index, token_position = [], [], []
        for b in range(batch_size):
            real_positions = attention_mask[b].nonzero(as_tuple=True)[0]
            if len(real_positions) != len(token_ids_list[b]):
                raise ValueError(
                    f"Prompt {b}: {len(token_ids_list[b])} token ids but "
                    f"{len(real_positions)} unpadded positions in the batch"
                )
            rows.append(b * padded_len + real_positions)
            prompt_index.append(torch.full((len(real_positions),), b, dtype=torch.int64))
            token_position.append(torch.arange(len(real_positions), dtype=torch.int32))
        rows = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)
        num_layers = len(layers)

        def stacked(key: str) -> torch.Tensor:
            """[tokens * layers, width]: token-major, layer-minor."""
            if not layers:
                return torch.empty(0, 0)
            per_layer = torch.stack([events[layer][key][rows] for layer in layers], dim=1)
            return per_layer.reshape(len(rows) * num_layers, -1)

        probs = stacked("probabilities").float()
        selected = stacked("topk_indices").long()
        # norm_topk_prob is false, so these unnormalised softmax probabilities
        # are the routing weights the model applies (#101)
        selected_probs = torch.gather(probs, 1, selected) if probs.numel() else probs

        return BatchRouting(
            prompt_index=torch.cat(prompt_index).repeat_interleave(num_layers).numpy()
            if prompt_index else np.empty(0, dtype=np.int64),
            token_position=torch.cat(token_position).repeat_interleave(num_layers).numpy()
            if token_position else np.empty(0, dtype=np.int32),
            layer_index=np.tile(np.asarray(layers, dtype=np.int32), len(rows)),
            expert_probs=probs.numpy(),
            selected_experts=selected.to(torch.int16).numpy(),
            selected_probs=selected_probs.numpy(),
            num_layers=num_layers,
            expert_logits=stacked("logits").float().numpy() if include_logits else None,
        )

    def _routing_from_batch(
        self,
        batch: BatchRouting,
        sequence_ids: List[int],
        token_ids_list: List[List[int]],
        token_strs_list: List[List[str]],
    ) -> List[PromptRoutingData]:
        """Build per-prompt PromptRoutingData objects from a batch's arrays."""
        routing_data_list = []
        row = 0
        for prompt_idx, (token_ids, token_strs) in enumerate(zip(token_ids_list, token_strs_list)):
            sequence_id = str(sequence_ids[prompt_idx])
            end = row + len(token_ids) * batch.num_layers
            logits = batch.expert_logits[row:end].tolist() if batch.expert_logits is not None else None
            probs = batch.expert_probs[row:end].tolist()
            selected = batch.selected_experts[row:end].tolist()
            selected_probs = batch.selected_probs[row:end].tolist()
            layer_index = batch.layer_index[row:end].tolist()

            prompt_routing = PromptRoutingData(
                sequence_id=sequence_id,
                model=self.model_name,
                num_tokens=len(token_ids),
                num_layers=batch.num_layers,
            )
            for token_pos, (token_id, token_str) in enumerate(zip(token_ids, token_strs)):
                token_routing = TokenRoutingData(
                    sequence_id=sequence_id,
                    token_id=token_id,
                    token_str=token_str,
                    token_position=token_pos,
                )
                for i in range(token_pos * batch.num_layers, (token_pos + 1) * batch.num_layers):
                    layer_data = {
                        "layer_index": layer_index[i],
                        "expert_probs": probs[i],
                        "selected_experts": selected[i],
                        "selected_probs": selected_probs[i],
                    }
                    if logits is not None:
                        layer_data["expert_logits"] = logits[i]
                    token_routing.layer_data.append(layer_data)
                prompt_routing.token_routing.append(token_routing)

            prompt_routing.metadata = {
                "num_tokens_captured": len(prompt_routing.token_routing),
                "num_layers_captured": batch.num_layers,
                "device": str(self.device),
            }
            routing_data_list.append(prompt_routing)
            row = end

        return routing_data_list

    def cleanup(self) -> None:
        """Remove hooks and clean up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("✅ Hooks removed")


def run_forward_pass_pipeline(
    model_name: str,
    prompts: List[str],
    device: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> List[PromptRoutingData]:
    """
    End-to-end forward pass with routing capture.

    Loads the model once, with its tokenizer from the same load, unless an
    already-loaded model and tokenizer are passed in. (It used to load the
    full model twice, once just for the tokenizer: about 58 GB in fp16.)

    Args:
        model_name: Model name (e.g., "qwen1.5-moe-a2.7b")
        prompts: List of text prompts
        device: Device to run on (default: cuda if available)
        model, tokenizer: An already-loaded model and its tokenizer. Pass both
            or neither.

    Returns:
        List of PromptRoutingData with complete routing telemetry
    """
    if (model is None) != (tokenizer is None):
        raise ValueError("Pass both model and tokenizer, or neither")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    capture = ForwardPassCapture(model_name, device=device)
    if model is None:
        logger.info(f"Loading {model_name}...")
        capture.load_model()
    else:
        capture.model, capture.tokenizer = model, tokenizer
    capture.setup_hooks()

    logger.info(f"Tokenizing {len(prompts)} prompts...")
    tokenization_result = TokenizationPipeline(capture.tokenizer, model_name).tokenize_batch(prompts)

    try:
        routing_data = capture.run_forward_pass(
            token_tensors=tokenization_result["token_tensors"],
            sequence_ids=tokenization_result["sequence_ids"],
            token_ids_list=tokenization_result["token_ids"],
            token_strs_list=tokenization_result["token_strs"],
        )
        return routing_data
    finally:
        capture.cleanup()
