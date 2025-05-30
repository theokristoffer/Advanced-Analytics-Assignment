import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import ImageData
import random
import torch.nn as nn
import torch.nn.functional as F



def normalization_and_resize():
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform




def do_model_testing(dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = normalization_and_resize()
    dataset = ImageData.ImageData(dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_classes = len(loader.dataset.mapping)
    model = models.efficientnet_b0(pretrained=False)

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

    model.load_state_dict(torch.load("best_efficient_net_model_2_layer.pth", map_location=device))
    model.to(device)
    model.eval()





    correct = 0
    total = 0
    #inputs are batches of 32
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.3f}%")




    #random samples
    sample_indexes = random.sample(range(len(dataset)), 5)

    for index in sample_indexes:
        image_path = dataset.image_paths[index]
        image, true_label_index = dataset[index]
        image = image.unsqueeze(0).to(device)

        true_label = dataset.classes[true_label_index]

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_label_index = torch.argmax(probabilities, 1).item()
            predicted_prob = probabilities[0, predicted_label_index].item()

        predicted_label = dataset.classes[predicted_label_index]

        print(
            f"Image Path: {image_path} , True Label: {true_label} , Predicted Label: {predicted_label} , Probability: {predicted_prob:.4f}")




if __name__=="__main__":
    #paths
    dir = "dataset/test"
    do_model_testing(dir)






