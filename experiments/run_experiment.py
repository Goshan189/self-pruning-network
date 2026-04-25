import torch
import json

from src.model import PrunableNet
from src.dataset import get_cifar10_loaders
from src.train import train_model


def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    lambdas = [0.01, 0.1, 0.5]

    results = []

    for lam in lambdas:
        print(f"\nRunning experiment for lambda = {lam}")

        model = PrunableNet().to(device)

        history = train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=5,
            lambda_sparse=lam
        )

        final_acc = history["test_acc"][-1]
        final_sparsity = history["sparsity"][-1]

        results.append({
            "lambda": lam,
            "test_accuracy": final_acc,
            "sparsity": final_sparsity
        })

    # Save results
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nFinal Results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    run_experiments()