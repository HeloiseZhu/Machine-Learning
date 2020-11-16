import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Define hyper-parameters
BATCH_SIZE = 50
EPOCHS = 5
LR = 0.01

# Load and normalize MNIST training and test datasets
# transform = transforms.Compose([transforms.ToTensor(), 
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
test_inputs = torch.unsqueeze(test_set.test_data, dim=1).type(torch.FloatTensor)/255
test_labels = test_set.test_labels
test_inputs = test_inputs[:100]
test_labels = test_labels[:100]
training_loss_figure = []
validation_loss_figure = []


# Define a Convolutional Neural Network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # LeNet-5: CONV-->POOL-->CONV-->POOL-->FC-->FC-->OUTPUT
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=2)   # padding='Same'
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)  # padding='Valid'
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''
        # Net2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=2)   # padding='Same'
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(14 * 14 * 6, 10)
        '''
        '''
        # Net3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=2)  # padding='Same'
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=4, stride=1, padding=0)  # padding='Valid'
        self.fc1 = nn.Linear(25 * 25 * 16, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
        '''

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet5()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=LR)

# Train the network
def Train():
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print("Epoch ", epoch + 1)
        print('*' * 15)
        training_loss = 0.0
        validation_loss = 0.0
        for i, data in enumerate(train_loader):
            # data is a list of [inputs, labels]
            # len(inputs) = len(labels) = BATCH_SIZE
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            t_loss = criterion(outputs, labels)
            t_loss.backward()
            optimizer.step()
            training_loss += t_loss.item()
            training_loss_figure.append(t_loss.item())

            test_outputs = net(test_inputs)
            v_loss = criterion(test_outputs, test_labels)
            validation_loss += v_loss.item()
            validation_loss_figure.append(v_loss.item())
            
            # print every 10 mini-batches
            if (i + 1) % 10 == 0:    
                print('[%d, %5d] training loss: %.4f, validation loss: %.4f'
                      % (epoch + 1, i + 1, training_loss / 10, validation_loss / 10))
                training_loss = 0.0
                validation_loss = 0.0
    print('Finished Training')


def Test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of LeNet-5 on 10000 test images: {}%'.format(100 * correct / total))
    PATH = './mnist_net.pth'
    torch.save(net.state_dict(), PATH)




if __name__ == '__main__':
    Train()
    Test()
    # Show training loss and validation loss
    x = range(1, len(training_loss_figure) + 1)
    plt.plot(x, training_loss_figure, label='training loss')
    plt.plot(x, validation_loss_figure, label='validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig('./loss_figure.png')
    plt.legend()
    plt.show()