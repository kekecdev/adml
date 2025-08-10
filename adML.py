import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

torch.manual_seed(3407)


def get_dataloaders(dataset_name="MNIST", aug_type="none", batch_size=128):

    if dataset_name.upper() == "MNIST":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
        pre_transform = [transforms.Grayscale(num_output_channels=3)]
        image_size = 28
    elif dataset_name.upper() == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
        pre_transform = []  
        image_size = 32


    aug_transform = []
    if aug_type == "Rotation":
        aug_transform = [transforms.RandomRotation(10)]
    elif aug_type == "HorizontalFlip":
        aug_transform = [transforms.RandomHorizontalFlip()]
    elif aug_type == "VerticalFlip":
        aug_transform = [transforms.RandomVerticalFlip()]
    elif aug_type == "Crop":
        aug_transform = [transforms.RandomCrop(image_size, padding=4)]


    transform_train = transforms.Compose(
        pre_transform + aug_transform + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    transform_test = transforms.Compose(
        pre_transform + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    if dataset_name.upper() == "MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    else: 
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_and_evaluate(dataset_name="MNIST", aug_type="none", epochs=10, lr=0.001, patience=3):
    trainloader, valloader = get_dataloaders(dataset_name, aug_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0


    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = train_loss_sum / len(trainloader)
        train_acc = 100 * correct_train / total_train

        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        
        print(f"[{dataset_name} | Aug={aug_type}] "
              f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Validation loss improved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping")
                break


    return best_val_acc


if __name__ == "__main__":
    results = {}
    datasets = ["MNIST", "CIFAR10"]

    augmentations = ["none", "Rotation", "HorizontalFlip", "VerticalFlip", "Crop"]

    for dataset in datasets:
        for aug_type in augmentations:
            print(f"\nDataset={dataset}, Augmentation={aug_type}")

            acc = train_and_evaluate(dataset_name=dataset, aug_type=aug_type, epochs=20, patience=3)
            results[(dataset, aug_type)] = acc

    print("\n\nBest Validation Accuracy")

    print(f"{'Dataset':<10} | {'Augmentation':<15} | {'Accuracy (%)'}")
    print("-" * 50)
    for (d, a), v in results.items():
        print(f"{d:<10} | {a:<15} | {v:.2f}")
