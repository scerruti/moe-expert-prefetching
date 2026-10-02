#!/usr/bin/env python3
"""
Tokenization and prompt preparation for Phase 1 data collection pipeline.

Handles:
- Loading prompts from various sources
- Tokenizing via ModelLoader (model-agnostic)
- Tracking repeated tokens and their context
- Preparing output format for Phase 1 routing telemetry capture
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import json
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class TokenMetadata:
    """Metadata for a single token occurrence."""
    token_id: int
    token_str: str
    position: int
    sequence_id: str  # hash or ID of the prompt


@dataclass
class TokenStats:
    """Aggregate statistics for a token across all occurrences."""
    token_id: int
    token_str: str
    occurrence_count: int = 0
    positions: List[int] = field(default_factory=list)
    sequences_seen: List[str] = field(default_factory=list)


class TokenizationPipeline:
    """Tokenize prompts and track repeated tokens for routing analysis."""

    def __init__(self, tokenizer, model_name: str):
        """
        Initialize tokenization pipeline.

        Args:
            tokenizer: Hugging Face tokenizer (from ModelLoader)
            model_name: Name of the model (for logging/metadata)
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.token_stats: Dict[int, TokenStats] = {}

    def tokenize_batch(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Tokenize a batch of prompts and track repeated tokens.

        Args:
            prompts: List of text prompts

        Returns:
            Dictionary with:
            - token_ids: List of token ID sequences
            - token_strs: List of token string sequences
            - token_tensors: Stacked tensor of tokenized inputs
            - token_stats: Dict of TokenStats for each unique token
            - metadata: Batch-level metadata
        """
        all_token_ids = []
        all_token_strs = []
        all_sequences = []

        for prompt in prompts:
            sequence_id = hash(prompt) % (10**9)  # Simple hash for ID
            token_ids = self.tokenizer.encode(prompt)
            token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]

            all_token_ids.append(token_ids)
            all_token_strs.append(token_strs)

            # Track repeated tokens
            for pos, (tid, tstr) in enumerate(zip(token_ids, token_strs)):
                if tid not in self.token_stats:
                    self.token_stats[tid] = TokenStats(
                        token_id=tid,
                        token_str=tstr,
                    )

                stats = self.token_stats[tid]
                stats.occurrence_count += 1
                stats.positions.append(pos)
                stats.sequences_seen.append(str(sequence_id))

            all_sequences.append(sequence_id)

        # Prepare tensors for model input
        token_tensors = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "token_ids": all_token_ids,
            "token_strs": all_token_strs,
            "token_tensors": token_tensors,
            "token_stats": self.token_stats,
            "sequence_ids": all_sequences,
            "metadata": {
                "model": self.model_name,
                "num_prompts": len(prompts),
                "total_tokens": sum(len(ids) for ids in all_token_ids),
                "unique_tokens": len(self.token_stats),
            },
        }

    def get_repeated_tokens(self) -> List[TokenStats]:
        """Get tokens that appear multiple times across the batch."""
        return [
            stats
            for stats in self.token_stats.values()
            if stats.occurrence_count > 1
        ]

    def save_stats(self, output_path: Path):
        """Save token statistics to JSON for inspection."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats_dict = {
            tid: {
                "token_str": stats.token_str,
                "occurrence_count": stats.occurrence_count,
                "unique_sequences": len(set(stats.sequences_seen)),
            }
            for tid, stats in self.token_stats.items()
        }

        with open(output_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Token statistics saved to {output_path}")

    def reset(self):
        """Reset statistics for next batch."""
        self.token_stats.clear()
