import os
import matplotlib.pyplot as plt

def save_gate_distribution(model, lam, path="results/plots"):
    os.makedirs(path, exist_ok=True)

    gates = model.get_all_gates().detach().cpu().numpy()

    filename = f"{path}/gate_distribution_lambda_{lam}.png"

    plt.figure()
    plt.hist(gates, bins=50)
    plt.title(f"Gate Distribution (λ = {lam})")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    plt.savefig(filename)
    plt.close()

    print(f"Saved plot: {filename}")