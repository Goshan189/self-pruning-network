import torch

from src.model import PrunableNet
from src.dataset import get_cifar10_loaders
from src.train import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    model = PrunableNet().to(device)

    history = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,          # keep it short
        lambda_sparse=0.01
    )


if __name__ == "__main__":
    main()