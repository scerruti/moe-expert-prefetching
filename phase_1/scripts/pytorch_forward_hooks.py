import torch
import torch.nn.functional as F
import queue

class RoutingDataBuffer:
    """
    Thread-safe buffer to store hook outputs without blocking forward execution.
    """
    def __init__(self):
        self.buffer = queue.Queue()
        
    def add(self, layer_idx, logits, probabilities, top2_indices):
        # Detach and move to CPU immediately to prevent GPU memory leaks 
        # and ensure the main forward pass isn't blocked by subsequent processing.
        self.buffer.put({
            'layer_idx': layer_idx,
            'logits': logits.detach().cpu(),
            'probabilities': probabilities.detach().cpu(),
            'top2_indices': top2_indices.detach().cpu()
        })
        
    def get_all(self):
        results = []
        while not self.buffer.empty():
            results.append(self.buffer.get())
        return results

def create_router_hook(layer_idx, data_buffer):
    """
    Creates a forward hook to capture routing data for a specific MoE layer.
    """
    def hook(module, input, output):
        # 1. Pre-softmax logits are captured directly from the gate's output
        # Shape: [batch_size, sequence_length, num_experts]
        logits = output
        
        # 2. Post-softmax probabilities are computed
        probabilities = F.softmax(logits, dim=-1)
        
        # 3. Top-2 expert indices are extracted
        _, top2_indices = torch.topk(probabilities, k=2, dim=-1)
        
        # 4. Save to thread-safe buffer
        data_buffer.add(layer_idx, logits, probabilities, top2_indices)
        
    return hook

def register_router_hooks(model, data_buffer):
    """
    Registers forward hooks on all MoE router layers in the Mixtral model.
    """
    hook_handles = []
    
    # Iterate through all layers (32 in the full 8x7B model)
    for i, layer in enumerate(model.model.layers):
        # Find the block_sparse_moe.gate path
        router_gate = layer.block_sparse_moe.gate
        
        # Register the hook and store the handle for potential removal later
        handle = router_gate.register_forward_hook(create_router_hook(i, data_buffer))
        hook_handles.append(handle)
        
    return hook_handles

def test_hooks():
    """
    Validates hook outputs against acceptance criteria using a toy Mixtral model.
    """
    from transformers import MixtralConfig, MixtralForCausalLM
    
    print("Initializing toy Mixtral model for testing...")
    # Use a tiny config so it runs instantly without requiring the 87GB model weights
    config = MixtralConfig(
        num_hidden_layers=4, 
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2
    )
    model = MixtralForCausalLM(config)
    
    # Initialize thread-safe buffer
    data_buffer = RoutingDataBuffer()
    
    # Register hooks
    print("Registering hooks on router layers...")
    handles = register_router_hooks(model, data_buffer)
    
    # Run a dummy forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(input_ids)
        
    # Validate outputs
    collected_data = data_buffer.get_all()
    print(f"Collected data from {len(collected_data)} layers.")
    
    # Assertions to fulfill Acceptance Criteria programmatically
    assert len(collected_data) == config.num_hidden_layers, "Hooks did not fire on all layers."
    
    for data in collected_data:
        layer = data['layer_idx']
        logits = data['logits']
        probs = data['probabilities']
        top2 = data['top2_indices']
        
        # Check shapes
        assert logits.shape == (batch_size, seq_len, 8), "Pre-softmax logits shape mismatch."
        assert probs.shape == (batch_size, seq_len, 8), "Post-softmax probabilities shape mismatch."
        assert top2.shape == (batch_size, seq_len, 2), "Top-2 indices shape mismatch."
        
        # Check math
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len)), "Probabilities do not sum to 1."
        
        print(f"Layer {layer} passed validation. (Logits: {logits.shape}, Top-2: {top2.shape})")

    # Cleanup
    for handle in handles:
        handle.remove()
    print("\n✅ All acceptance criteria met. Hooks fire correctly, extract all required metrics, and utilize a thread-safe buffer.")

if __name__ == "__main__":
    test_hooks()
