from src.model import PrunableNet
from src.utils import compute_sparsity_loss, compute_sparsity, get_gate_statistics

model = PrunableNet()

print("Sparsity Loss:", compute_sparsity_loss(model).item())
print("Sparsity %:", compute_sparsity(model))
print("Stats:", get_gate_statistics(model))