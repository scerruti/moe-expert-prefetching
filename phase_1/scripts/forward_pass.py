#!/usr/bin/env python3
"""
Forward pass integration with hook-based routing capture (Issue #15).

Combines:
- Tokenized prompts (from tokenizer.py, Issue #14)
- Model and hooks (from pytorch_forward_hooks.py, Issue #5)

Executes forward pass and captures routing telemetry for Phase 1 analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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
        Run forward pass and capture routing data.

        Args:
            token_tensors: Output from tokenizer (input_ids, attention_mask)
            sequence_ids: Sequence IDs for each prompt
            token_ids_list: Token IDs for each prompt
            token_strs_list: Token strings for each prompt

        Returns:
            List of PromptRoutingData with complete routing telemetry
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.routing_buffer is None:
            raise RuntimeError("Hooks not setup. Call setup_hooks() first.")

        input_ids = token_tensors["input_ids"].to(self.device)
        attention_mask = token_tensors["attention_mask"].to(self.device)
        batch_size = input_ids.shape[0]

        logger.info(f"Running forward pass on {batch_size} prompts...")

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

        # Collect routing data from hooks
        routing_data_list = self._extract_routing_data(
            batch_size=batch_size,
            sequence_ids=sequence_ids,
            token_ids_list=token_ids_list,
            token_strs_list=token_strs_list,
        )

        logger.info(f"✅ Captured routing data for {len(routing_data_list)} prompts")
        return routing_data_list

    def _extract_routing_data(
        self,
        batch_size: int,
        sequence_ids: List[int],
        token_ids_list: List[List[int]],
        token_strs_list: List[List[str]],
    ) -> List[PromptRoutingData]:
        """Extract and structure routing data from hook buffer."""
        routing_data_list = []

        for prompt_idx in range(batch_size):
            sequence_id = str(sequence_ids[prompt_idx])
            token_ids = token_ids_list[prompt_idx]
            token_strs = token_strs_list[prompt_idx]
            num_tokens = len(token_ids)
            num_layers = 0

            # Collect all routing data for this prompt
            prompt_routing = PromptRoutingData(
                sequence_id=sequence_id,
                model=self.model_name,
                num_tokens=num_tokens,
                num_layers=0,  # Will be set below
            )

            # Extract routing data per token per layer
            routing_events = self.routing_buffer.get_all()

            if routing_events:
                num_layers = self._count_layers(routing_events)
                prompt_routing.num_layers = num_layers

                for token_pos in range(min(num_tokens, 2048)):  # Limit to reasonable length
                    token_routing = TokenRoutingData(
                        sequence_id=sequence_id,
                        token_id=token_ids[token_pos],
                        token_str=token_strs[token_pos],
                        token_position=token_pos,
                    )

                    # Collect layer data for this token
                    for layer_idx in range(num_layers):
                        layer_data = self._extract_layer_routing(
                            routing_events,
                            prompt_idx=prompt_idx,
                            token_pos=token_pos,
                            layer_idx=layer_idx,
                        )
                        if layer_data:
                            token_routing.layer_data.append(layer_data)

                    prompt_routing.token_routing.append(token_routing)

            prompt_routing.metadata = {
                "num_tokens_captured": len(prompt_routing.token_routing),
                "num_layers_captured": num_layers,
                "device": str(self.device),
            }

            routing_data_list.append(prompt_routing)

        return routing_data_list

    def _count_layers(self, routing_events: List[Tuple]) -> int:
        """Count total number of layers from routing events."""
        if not routing_events:
            return 0
        layer_indices = set()
        for event in routing_events:
            if len(event) >= 4 and isinstance(event[3], int):
                layer_indices.add(event[3])
        return len(layer_indices)

    def _extract_layer_routing(
        self,
        routing_events: List[Tuple],
        prompt_idx: int,
        token_pos: int,
        layer_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Extract routing data for a specific token and layer."""
        for event in routing_events:
            if len(event) >= 5:
                logits, selected_weights, selected_indices, event_layer_idx, event_batch_idx = event[:5]

                if event_batch_idx == prompt_idx and event_layer_idx == layer_idx:
                    if logits is not None and len(logits) > token_pos:
                        return {
                            "layer_index": layer_idx,
                            "expert_logits": logits[token_pos].tolist() if hasattr(logits[token_pos], "tolist") else logits[token_pos],
                            "expert_probs": F.softmax(
                                torch.tensor(logits[token_pos], device=self.device),
                                dim=-1
                            ).tolist(),
                            "selected_experts": selected_indices[token_pos].tolist() if hasattr(selected_indices[token_pos], "tolist") else selected_indices[token_pos],
                            "selected_probs": selected_weights[token_pos].tolist() if hasattr(selected_weights[token_pos], "tolist") else selected_weights[token_pos],
                        }

        return None

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
) -> List[PromptRoutingData]:
    """
    End-to-end forward pass with routing capture.

    Args:
        model_name: Model name (e.g., "qwen1.5-moe-a2.7b")
        prompts: List of text prompts
        device: Device to run on (default: cuda if available)

    Returns:
        List of PromptRoutingData with complete routing telemetry
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize
    logger.info(f"Tokenizing {len(prompts)} prompts...")
    loader = ModelLoader(model_name)
    _, tokenizer, _ = loader.load()
    tokenizer_pipeline = TokenizationPipeline(tokenizer, model_name)
    tokenization_result = tokenizer_pipeline.tokenize_batch(prompts)

    # Forward pass
    logger.info("Setting up forward pass capture...")
    capture = ForwardPassCapture(model_name, device=device)
    capture.load_model()
    capture.setup_hooks()

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
