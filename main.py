import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ModifiedLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 10, 1)

        self.pool = nn.MaxPool2d(2)
        self.swish = Swish()

    def forward(self, x):

        x = self.pool(self.swish(self.bn1(self.conv1(x))))
        x = self.pool(self.swish(self.bn2(self.conv2(x))))
        x = self.swish(self.bn3(self.conv3(x)))

        x = self.conv4(x)

        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)

        return F.softmax(x, dim=1)


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModifiedLeNet().to(device)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_acc = []
val_acc = []

for epoch in range(10):

    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc.append(correct/total)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc.append(correct/total)

    print("Epoch:", epoch, "Train Acc:", train_acc[-1], "Val Acc:", val_acc[-1])



import matplotlib.pyplot as plt

plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
