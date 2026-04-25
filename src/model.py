import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePrunableModule(nn.Module):
    def get_all_gates(self):
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gate_values().view(-1))
        return torch.cat(gates)

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) - 1.0
        )

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores * 5)

        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"

class PrunableNet(BasePrunableModule):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.net = nn.Sequential(
            PrunableLinear(32 * 32 * 3, 512),
            nn.ReLU(),

            PrunableLinear(512, 256),
            nn.ReLU(),

            PrunableLinear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)
    
"""
model = PrunableNet()
x = torch.randn(4, 3, 32, 32)

out = model(x)

print("Output shape:", out.shape)
print("Total gates:", model.get_all_gates().shape)"""