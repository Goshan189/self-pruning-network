import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import compute_sparsity_loss, compute_sparsity


# --------------------------------------------------
# Train for one epoch
# --------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        ce_loss = nn.CrossEntropyLoss()(outputs, labels)
        sp_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100. * correct / total
    return accuracy


# --------------------------------------------------
# Full training loop
# --------------------------------------------------
def train_model(model, train_loader, test_loader, device,
                epochs=10, lr=1e-3, lambda_sparse=1e-5):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "sparsity": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lambda_sparse
        )

        test_acc = evaluate(model, test_loader, device)
        sparsity = compute_sparsity(model)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

    return history