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


#
# Syntax:
# python experiment_a.py [num_epochs] [batch_size] [learning_rate] [momentum] [out_path]
#
if __name__ == "__main__":
    assert(len(argv) == 6)
    NUM_EPOCHS = int(argv[1])
    BATCH_SIZE = int(argv[2])
    LEARNING_RATE = float(argv[3])
    MOMENTUM = float(argv[4])
    OUTFILE = open(argv[5], 'w')
    
    print("Libraries loaded...")

    # Input image transformations
    transformations = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load train input images
    train_set = datasets.ImageFolder("data/train", transform = transformations)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, \
                    num_workers=os.cpu_count())

    validation_set = datasets.ImageFolder("data/validation", transform = transformations)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=validation_loader.__len__(), shuffle=True, num_workers=os.cpu_count())
    
    test_set = datasets.ImageFolder("data/test", transform = transformations)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_loader.__len__(), shuffle=True, num_workers=os.cpu_count())


    # assert(False)
    print("Dataset loaded...")

    # Model segments

    # Model architecture
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

    print("Model initialized...")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    criterion = nn.CrossEntropyLoss()

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    model.to(device)
    OUTFILE.write("epoch,train_loss,validation_loss,accuracy,precision,recall,\n")

    best_statsum = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        accuracy   = 0

        model.train()
        counter = 0
        # Train
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(inputs)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()
            train_loss += loss.item()*inputs.size(0)

            counter += 1
            if epoch % 10 == 0 and counter % 100 == 0:
                print('[', counter, "/", len(train_loader), ']')

        # Validate
        model.eval()
        validation_loss = 0
        precision = 0
        recall = 0
        accuracy = 0
        counter = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                validation_loss += valloss.item()*inputs.size(0)
                
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)

                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)

                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                y_true, y_pred = labels.cpu().detach().numpy().reshape(-1, 1), top_class.cpu().detach().numpy().reshape(-1, 1)

                accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)

        if (recall + accuracy + precision) > best_statsum:
            best_statsum = recall + accuracy + precision
            print("New best model!")
            torch.save(model.state_dict(), "experiment_b.pkl")

        train_loss = train_loss/len(train_loader.dataset)
        validation_loss  = validation_loss/len(validation_loader.dataset)
        print('Accuracy: ', accuracy/len(validation_loader))
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, validation_loss))
        OUTFILE.write("{},{},{},{},{},{},\n".format(epoch,train_loss,validation_loss,accuracy,precision,recall))

    # Final Test
    test_loss = 0
    for inputs, labels in test_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        output = model.forward(inputs)
        # Calculate Loss
        testloss = criterion(output, labels)
        test_loss += testloss.item()*inputs.size(0)
        
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
    OUTFILE.write("test...,validation_loss,accuracy,precision,recall,\n \
                {},{},{},{},\n".format(validation_loss,accuracy,precision,recall))
    OUTFILE.close()