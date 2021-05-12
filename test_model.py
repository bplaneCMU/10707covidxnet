import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import os

from sklearn.metrics import recall_score, precision_score

if __name__ == "__main__":
    assert(len(argv) == 2)
    c = argv[1]

    model = None
    if c == "a":
        model = nn.Sequential(
        nn.Conv2d(3, 48, 7, stride=2, padding=3),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 160, 1, stride=1, padding=0),
        nn.ReplicationPad2d(2),
        nn.Conv2d(160, 160, 3, stride=2),
        nn.Conv2d(160, 160, 3, stride=2),
        nn.Conv2d(160, 160, 1, stride=1, padding=0),
        nn.Flatten(),
        nn.Linear(144000, 2),
        nn.Softmax(dim=1)
        )
    if c == "b":
        model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 3, stride=2, padding=1),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 1, stride=1, padding=0),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(3600, 480),
        nn.ReLU(),
        nn.Linear(480, 2),
        nn.Softmax(dim=1)
        )
    if c == "c":
        model = nn.Sequential(
        nn.Conv2d(3, 48, 9, stride=2, padding=4),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 160, 1, stride=1, padding=0),
        nn.Conv2d(160, 160, 7, stride=2, padding=3),
        nn.MaxPool2d(2),
        nn.Conv2d(160, 160, 5, stride=2, padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7840, 480),
        nn.ReLU(),
        nn.Linear(480, 2),
        nn.Softmax(dim=1)
        )
    model.load_state_dict(torch.load("experiment_{}.pkl".format(c)))
    model.eval()


    transformations = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = datasets.ImageFolder("data/test", transform = transformations)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_loader.__len__(), shuffle=True, num_workers=os.cpu_count())


    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    print("Running on {}".format(device))
    model.to(device)

    for inputs, labels in test_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        output = model.forward(inputs)
        # Calculate Loss
        testloss = criterion(output, labels)
        test_loss = testloss.item()*inputs.size(0)
        
        # Get the top class of the output
        top_p, top_class = output.topk(1, dim=1)

        # See how many of the classes were correct?
        equals = top_class == labels.view(*top_class.shape)

        # Calculate the mean (get the accuracy for this batch)
        # and add it to the running accuracy for this epoch
        y_true, y_pred = labels.cpu().detach().numpy().reshape(-1, 1), top_class.cpu().detach().numpy()

        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
    print("Accuracy: {} | Precision: {} | Recall: {}".format(accuracy, precision, recall))
