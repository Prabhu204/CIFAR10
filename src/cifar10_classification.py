# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tf
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
transform = tf.Compose([tf.ToTensor(), tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def load_data(path, transform, download= True, train= True,shuffle=True):
    dataset = torchvision.datasets.CIFAR10(root=path,train=train,transform= transform, download=download)
    datasetloader = DataLoader(dataset=dataset, batch_size= 256, shuffle=True)
    return dataset, datasetloader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset, trainloader = load_data(path='./data', transform= transform, train= True, download= True, shuffle=True)
testset, testloader = load_data(path='./data', transform= transform, train= False, download= True, shuffle=True)

# images, lables = iter(trainloader).__next__()  # check the size of the image.

class Cifar_model(nn.Module):
    def __init__(self):
        super(Cifar_model, self).__init__()
        self.conv1 =nn.Sequential(nn.Conv2d(3,8,5), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2  = nn.Sequential(nn.Conv2d(8,16,3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3  = nn.Sequential(nn.Conv2d(16,32,2), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(32*5*5, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, len(list(classes))))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(-1, 32*5*5)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = Cifar_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.01, momentum=0.9)

# train network
for epoch in range(50):
    model.train()
    train_loss = 0
    total_predicted = []
    total_labels = []
    for idx, dataset in enumerate(trainloader):
        traindata, lables = dataset[0].to(device), dataset[1].to(device)
        optimizer.zero_grad()
        output = model(traindata)
        # _, predicted_ = torch.max(output.data, 1)
        # print(predicted_)
        loss = criterion(output, lables)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = np.argmax(output.cpu().detach(), -1)
        total_predicted.extend(predicted)
        lables = lables.cpu()
        total_labels.extend(lables)
        print("\nIteration {}/{} for epoch {} with acc: {}, loss: {}".format(idx+1,len(trainloader),epoch+1,metrics.accuracy_score(lables,predicted),loss))
    print("\n\nTrain data statistics:\nEpoch {} with acc: {}, loss {} matrix:\n{}".format(epoch + 1,
                                                                                       metrics.accuracy_score(total_labels, total_predicted),
                                                                                       train_loss/len(trainloader),
                                                                                       metrics.confusion_matrix(total_labels,total_predicted)))


    with open('results/train.txt', 'a') as f:
        f.write("\n\nTrain data statistics:\nEpoch {} with acc: {}, loss {} matrix:\n{}".format(epoch + 1,
                                                                                       metrics.accuracy_score(total_labels, total_predicted),
                                                                                       train_loss/len(trainloader),
                                                                                       metrics.confusion_matrix(total_labels,total_predicted)))

    # test network
    model.eval()
    test_predictions = []
    test_targets = []
    test_loss = 0
    for idx, t_dataset in enumerate(testloader):
        input, target = t_dataset[0].to(device), t_dataset[1].to(device)
        predict_test = model(input)
        loss = criterion(predict_test, target)
        test_loss += loss.item()
        predict_test_ = np.argmax(predict_test.cpu().detach(), -1)
        test_predictions.extend(predict_test_)
        test_targets.extend(target.cpu())
    print("\n\nTest data statistics:\nEpoch {} with acc: {}, loss {} matrix:\n{}".format(epoch + 1,
                                                                                       metrics.accuracy_score( test_targets, test_predictions),
                                                                                       test_loss/len(testloader),
                                                                                       metrics.confusion_matrix(test_targets, test_predictions)))

    with open('results/test.txt', 'a') as f:
        f.write("\n\nTest data statistics:\nEpoch {} with acc: {}, loss {} matrix:\n{}".format(epoch + 1,
                                                                                       metrics.accuracy_score( test_targets, test_predictions),
                                                                                       test_loss/len(testloader),
                                                                                       metrics.confusion_matrix(test_targets, test_predictions)))



