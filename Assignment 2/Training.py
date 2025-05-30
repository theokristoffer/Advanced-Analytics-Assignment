import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import ImageData
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def do_augmentation():
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform




def do_model_training(dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = do_augmentation()
    full_dataset = ImageData.ImageData(dir, transform=transform)

    # split into training and validation dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(full_dataset.mapping)
    model = models.efficientnet_b0(pretrained=True)


    #free model parameters to increase performance
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )


    #run on gpu
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    train_losses, val_losses = [], []

    epochs = 25
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)


        #validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.3f}, Training Accuracy: {train_acc:.3f}%, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}%")

        # find the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_efficient_net_model_2_layer.pth")
            print("Model Saved")

    # plot
    import matplotlib.pyplot as plt
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.show()





if __name__=="__main__":
    #path
    # dir = "dataset/train"
    dir = "C:/Users/Msi katana/PycharmProjects/Assignment_1/dataset/train"
    do_model_training(dir)




