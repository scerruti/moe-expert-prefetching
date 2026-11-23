import torch
import torch.nn.functional as F
import queue

# Phase 1 is locked to Qwen1.5-MoE-A2.7B (PHASE_1_MODEL_DECISION.md): router
# gate at layer.mlp.gate, 60 experts, top-4 routing. register_router_hooks
# checks the loaded model against this so a model change fails loudly.
QWEN_TOP_K = 4

class RoutingDataBuffer:
    """
    Thread-safe buffer to store hook outputs without blocking forward execution.
    """
    def __init__(self):
        self.buffer = queue.Queue()
        
    def add(self, layer_idx, logits, probabilities, topk_indices):
        # Detach and move to CPU immediately to prevent GPU memory leaks 
        # and ensure the main forward pass isn't blocked by subsequent processing.
        self.buffer.put({
            'layer_idx': layer_idx,
            'logits': logits.detach().cpu(),
            'probabilities': probabilities.detach().cpu(),
            'topk_indices': topk_indices.detach().cpu()
        })
        
    def get_all(self):
        results = []
        while not self.buffer.empty():
            results.append(self.buffer.get())
        return results

def create_router_hook(layer_idx, data_buffer):
    """
    Creates a forward hook to capture routing data for a specific MoE layer.

    Qwen1.5-MoE-A2.7B's router gate (layer.mlp.gate) is an nn.Linear, so its
    output is the router logits for every token in the flattened batch.

    Errors are logged with the layer number, then re-raised. That stops the
    forward pass, so a collection run fails instead of writing data with a
    layer missing (#97).
    """
    import logging
    logger = logging.getLogger(__name__)

    def hook(module, input, output):
        try:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected the router gate to return a tensor of logits, got {type(output).__name__}")

            # Match Qwen2MoE's own routing: softmax in float32, then top-k
            probabilities = F.softmax(output, dim=-1, dtype=torch.float)
            selected_indices = torch.topk(probabilities, k=QWEN_TOP_K, dim=-1).indices
            data_buffer.add(layer_idx, output, probabilities, selected_indices.int())
        except Exception:
            logger.error(f"Hook error at layer {layer_idx}", exc_info=True)
            raise

    return hook

def register_router_hooks(model, data_buffer):
    """
    Registers forward hooks on all MoE router layers in Qwen1.5-MoE-A2.7B.

    Phase 1 Decision (PHASE_1_MODEL_DECISION.md): Locked to Qwen architecture.
    Router gate is at layer.mlp.gate (Qwen2MoeSparseMoeBlock).

    For other models, update the gate path (e.g., Mixtral: layer.block_sparse_moe.gate).
    """
    import logging
    logger = logging.getLogger(__name__)

    top_k = model.config.num_experts_per_tok
    if top_k != QWEN_TOP_K:
        raise ValueError(
            f"Phase 1 hooks are locked to Qwen1.5-MoE-A2.7B (top-{QWEN_TOP_K} routing), "
            f"but this model routes top-{top_k}. See PHASE_1_MODEL_DECISION.md."
        )

    hook_handles = []
    registered_count = 0

    # Iterate through all layers
    for i, layer in enumerate(model.model.layers):
        # Qwen1.5-MoE: Router gate is at layer.mlp.gate
        router_gate = layer.mlp.gate
        logger.debug(f"Layer {i}: gate type = {type(router_gate)}, gate = {router_gate}")

        # Register the hook and store the handle for potential removal later
        handle = router_gate.register_forward_hook(create_router_hook(i, data_buffer))
        registered_count += 1
        hook_handles.append(handle)

    logger.info(f"✅ Registered {registered_count} router hooks on {len(model.model.layers)} layers")
    logger.info(f"📊 Now running forward pass - watch for 🎣 hook fire messages")
    return hook_handles
