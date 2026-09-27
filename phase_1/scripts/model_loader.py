#!/usr/bin/env python3
"""
Generic MoE Model Loader - supports multiple sparse MoE architectures.

Supports:
- Qwen1.5-MoE-A2.7B (mlp.router for router) - Small, for validation
- Mixtral 8x7B (block_sparse_moe.gate for router)
- Qwen3-VL-30B (mlp.gate for router)
- DeepSeek-V2-Lite (mlp.gate for router)

Usage:
    loader = ModelLoader("Qwen/Qwen3-VL-30B-A3B-Instruct")
    model, tokenizer, config = loader.load()

    # Access router gates for telemetry hooks
    for layer in model.model.layers:
        gate = loader.get_router_gate(layer)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for different MoE models."""

    MODELS = {
        "qwen1.5-moe-a2.7b": {
            "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "router_gate_path": "mlp.gate",
            "num_experts": 16,
            "top_k": 6,
            "is_vision": False,
            "note": "Small MoE model for validation and testing on limited GPUs (Colab A100)",
        },
        "mixtral-8x7b": {
            "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "router_gate_path": "block_sparse_moe.gate",
            "num_experts": 8,
            "top_k": 2,
            "is_vision": False,
        },
        "qwen3-vl-30b": {
            "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "router_gate_path": "mlp.gate",
            "num_experts": 128,
            "top_k": 6,
            "is_vision": True,
            "note": "Vision-language model - requires custom loading. Use qwen3-moe for text-only.",
        },
        "deepseek-v2-lite": {
            "model_id": "deepseek-ai/DeepSeek-V2-Lite",
            "router_gate_path": "mlp.gate",
            "num_experts": 64,
            "top_k": 6,
            "is_vision": False,
        },
    }

    @classmethod
    def get(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model."""
        name = model_name.lower().replace(" ", "-")
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[name]


class ModelLoader:
    """Load MoE models with hardware detection and memory optimization."""

    def __init__(self, model_name: str, device: Optional[str] = None, dtype: torch.dtype = torch.float16, load_in_4bit: bool = False):
        """
        Initialize loader for a specific model.

        Args:
            model_name: Model identifier (e.g., 'mixtral-8x7b')
            device: 'cuda', 'cpu', or None for auto-detection
            dtype: torch.float16, torch.bfloat16, or torch.float32
            load_in_4bit: Use 4-bit quantization to reduce memory (requires bitsandbytes)
        """
        self.config = ModelConfig.get(model_name)
        self.model_id = self.config["model_id"]
        self.dtype = dtype
        self.device = device or self._detect_device()
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

        logger.info(f"Initialized {model_name} loader")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {dtype}")
        logger.info(f"  Quantization: {'4-bit' if load_in_4bit else 'none'}")
        logger.info(f"  Experts: {self.config['num_experts']}, Top-K: {self.config['top_k']}")

    def _detect_device(self) -> str:
        """Auto-detect available hardware."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {device_name} ({vram_gb:.1f}GB VRAM)")
            return "cuda"
        logger.warning("No GPU detected, using CPU (slow)")
        return "cpu"

    def load(self, trust_remote_code: bool = True) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load model and tokenizer.

        Returns:
            (model, tokenizer, config)
        """
        logger.info(f"Loading {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": self.device,
        }

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs
        )

        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"  Total parameters: {self._count_parameters(self.model) / 1e9:.1f}B")

        return self.model, self.tokenizer, self.config

    def get_router_gate(self, layer: Any) -> Optional[Any]:
        """
        Extract router gate from a layer (handles different architectures).

        Args:
            layer: A model layer (e.g., model.model.layers[i])

        Returns:
            Router gate module or None if not found
        """
        gate_path = self.config["router_gate_path"]
        parts = gate_path.split(".")

        current = layer
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current if current is not None else None

    @staticmethod
    def _count_parameters(model: Any) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_experts": self.config["num_experts"],
            "top_k": self.config["top_k"],
            "is_vision": self.config["is_vision"],
            "total_params": self._count_parameters(self.model) if self.model else None,
        }

    def validate_architecture(self) -> bool:
        """
        Verify loaded model has correct architecture (num_experts, top_k, layers).

        Returns:
            True if architecture is correct, False otherwise
        """
        if self.model is None:
            logger.error("Cannot validate architecture: model not loaded")
            return False

        num_experts = self.config["num_experts"]
        top_k = self.config["top_k"]

        # Get actual values from model config
        actual_experts = getattr(self.model.config, "num_local_experts", None)
        actual_top_k = getattr(self.model.config, "num_experts_per_tok", None)

        success = True
        if actual_experts != num_experts:
            logger.error(f"Architecture mismatch: expected {num_experts} experts, got {actual_experts}")
            success = False
        else:
            logger.info(f"✅ Experts: {actual_experts}")

        if actual_top_k != top_k:
            logger.error(f"Architecture mismatch: expected top-{top_k}, got top-{actual_top_k}")
            success = False
        else:
            logger.info(f"✅ Top-K: {actual_top_k}")

        return success

    def validate_tokenizer(self, sample_text: str) -> bool:
        """
        Test tokenizer encode/decode roundtrip.

        Args:
            sample_text: Text to encode and decode

        Returns:
            True if roundtrip is lossless, False otherwise
        """
        if self.tokenizer is None:
            logger.error("Cannot validate tokenizer: tokenizer not loaded")
            return False

        token_ids = self.tokenizer.encode(sample_text)
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        if decoded_text.strip() == sample_text.strip():
            logger.info(f"✅ Tokenizer roundtrip: '{sample_text}' -> {len(token_ids)} tokens -> '{decoded_text}'")
            return True
        else:
            logger.error(f"Tokenizer roundtrip failed: '{sample_text}' != '{decoded_text}'")
            return False

    def validate_forward_pass(self, sample_text: str) -> bool:
        """
        Run a forward pass to verify model is wired correctly.

        Confirms:
        - Tokenizer works with model
        - No gradients are tracked
        - Model produces output logits

        Args:
            sample_text: Text to run through model

        Returns:
            True if forward pass succeeds, False otherwise
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Cannot validate forward pass: model or tokenizer not loaded")
            return False

        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(sample_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits_shape = outputs.logits.shape
            logger.info(f"✅ Forward pass: input '{sample_text}' -> logits shape {logits_shape}")
            return True

        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return False


def test_model_loading():
    """Test script to verify model loads correctly and passes validation."""
    try:
        loader = ModelLoader("qwen3-vl-30b")
        model, tokenizer, config = loader.load()

        logger.info("✅ Model loading test passed")
        logger.info(f"Model config: {loader.get_model_info()}")

        # Validation checks
        sample_text = "Hello, how are you?"
        all_passed = True

        all_passed &= loader.validate_architecture()
        all_passed &= loader.validate_tokenizer(sample_text)
        all_passed &= loader.validate_forward_pass(sample_text)

        # Test router gate access
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            first_layer = model.model.layers[0]
            gate = loader.get_router_gate(first_layer)
            if gate is not None:
                logger.info("✅ Router gate accessible")
            else:
                logger.warning("⚠️  Router gate not found (may need architecture-specific handling)")

        if all_passed:
            logger.info("✅ All validations passed!")
        else:
            logger.error("❌ Some validations failed")

        return all_passed

    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model_loading()
