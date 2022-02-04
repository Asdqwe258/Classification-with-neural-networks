import numpy as np
import time
import matplotlib.pyplot as plt



train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')

##Template MLP code
def softmax(x):
    f = np.exp(x - np.max(x))  #this softmax function taken from https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
    return f / f.sum(axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

class MLP():

    def __init__(self):
        #Initialize all the parametres
        #Uncomment and complete the following lines
        self.W1= np.random.normal(scale=0.1, size=(64,784))
        self.b1= 0
        self.W2= np.random.normal(scale=0.1, size=(10,64))
        self.b2= 0
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        #Feed data through the network
        #Uncomment and complete the following lines
        self.x=x
        self.W1x= self.W1 @ x
        self.a1= self.W1x + self.b1
        self.f1= sigmoid(self.a1)
        self.W2f1=  self.W2 @ self.f1
        self.a2= self.W2f1 + self.b2
        self.y_hat= softmax(self.a2)
        #self.L = CrossEntropy(self.y_hat, self.y)
        return self.y_hat

    def update_grad(self,y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        dA2db2= np.identity(10) #identity 3
        dA2dW2= self.f1.T
        dA2dF1= self.W2 #identity 1
        dF1dA1= (sigmoid(self.a1) * (1 - sigmoid(self.a1))).reshape(1,64) #identity 4 derivitave of sigmoid
        dA1db1= np.identity(64) #identity 3
        dA1dW1= self.x.T #identity 1
        dLdA2 = (self.y_hat - y).reshape(1,10) #identity 7

        dLdW2 = np.outer(dLdA2, dA2dW2).reshape(1,640) #identity 5
        dLdb2 = dLdA2 #ommitting dA2db2
        dLdF1 = dLdA2 @ dA2dF1
        dLdA1 = dLdF1 * dF1dA1
        dLdW1 = np.outer(dLdA1, dA1dW1).reshape(1,50176) #identity 5
        dLdb1 = dLdA1 #ommitting dA1db1
        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1
        pass

    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad.reshape(10,64)
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad.reshape(64,784)
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)
    def save_arrays(self):
        np.save('W1.npy', self.W1)
        np.save('W2.npy', self.W2)
        np.save('B1.npy', self.b1)
        np.save('B2.npy', self.b2)
    def show_weight(self):
        for i in range(64):
            plt.imshow(self.W1[i].reshape(28,28))
            plt.axis('off')
            plt.show()
## Init the MLP
myNet=MLP()


learning_rate=1e-4
n_epochs=100
n_train_images = 50000
n_val_images = 5000
batch_size = 256
t_acc_arr = [0]*n_epochs
v_acc_arr = [0]*n_epochs
## Training code
for iter in range(n_epochs):
    print('working on epoch ' + str(iter))
    #Code to train network goes here
    curr = 0
    train_acc = 0
    for k in range(int(n_train_images/batch_size) + 1):
        myNet.reset_grad()
        for j in range(batch_size):
            if(curr < n_train_images):
                x = train_images_np[curr,:]
                y = np.zeros(10)
                y[train_labels_np[curr] - 1] = 1
                y_hat = myNet.forward(x)
                #the values put into from the sigmoid function werent large enough to round to 1 or 0
                #so I just found the max values in the arrays
                if np.array_equal(np.argmax(y_hat), np.argmax(y)):
                    train_acc += 1
                myNet.update_grad(y)
                curr += 1
        myNet.update_params(learning_rate)
    t_acc_arr[iter] = train_acc / n_train_images
    #Code to compute validation loss/accuracy goes here
    curr = 0
    val_acc = 0
    for k in range(n_val_images):
        if (curr < n_val_images):
            x = val_images_np[k, :]
            y = np.zeros(10)
            y[val_labels_np[k] - 1] = 1
            y_hat = myNet.forward(x)
            if np.array_equal(np.argmax(y_hat), np.argmax(y)):
                val_acc += 1
    v_acc_arr[iter] = val_acc / n_val_images
myNet.save_arrays()
myNet.show_weight()
plt.title("Training = blue, Validation = orange")
plt.plot(np.arange(n_epochs),t_acc_arr, np.arange((n_epochs)),v_acc_arr)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([.7,1])
plt.show()

n_test_images = len(test_images_np)
curr = 0
val_acc = 0
for k in range(n_test_images):
        x = test_images_np[k, :]
        y = np.zeros(10)
        y[test_labels_np[k] - 1] = 1
        y_hat = myNet.forward(x)
        if np.array_equal(np.argmax(y_hat), np.argmax(y)):
            val_acc += 1
print('accuracy when used on test dataset:')
print(val_acc / n_test_images)

## Template for ConvNet Code
#This part of the project is unfinished and is non-functional
#feel free to disable it if you want to only test the first part
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    #From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Your training and testing code goes here
newNet = ConvNet()

#from section 3
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(newNet.parameters(), lr=0.001, momentum=0.9)

#from section 4 of the tutorial

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(len(train_images_np)):
        # get the inputs; data is a list of [inputs, labels]
        inputs = train_images_np[i]
        labels = train_labels_np[i]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = newNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(newNet.state_dict(), PATH)

#from section 5
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(test_images_np)):
        images = test_images_np[i]
        labels = test_labels_np[i]
        outputs = newNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))