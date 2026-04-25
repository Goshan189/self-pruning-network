import torch


# --------------------------------------------------
# Sparsity Loss (L1 on gates)
# --------------------------------------------------
def compute_sparsity_loss(model):
    gates = model.get_all_gates()
    return torch.mean(gates)


# --------------------------------------------------
# Compute sparsity percentage
# --------------------------------------------------
def compute_sparsity(model, threshold=0.1):
    gates = model.get_all_gates()

    total = gates.numel()
    pruned = (gates < threshold).sum().item()

    sparsity = (pruned / total) * 100
    return sparsity


# --------------------------------------------------
# Optional: gate statistics (for debugging + report)
# --------------------------------------------------
def get_gate_statistics(model):
    gates = model.get_all_gates()

    return {
        "mean": gates.mean().item(),
        "min": gates.min().item(),
        "max": gates.max().item()
    }