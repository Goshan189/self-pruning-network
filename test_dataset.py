from src.dataset import get_cifar10_loaders

def main():
    train_loader, _ = get_cifar10_loaders()

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break


if __name__ == "__main__":
    main()