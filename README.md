A CNN architecture for identifying a movie's genre from posters
======

***This is a PyTorch adaptation of the code that was first described and analyzed [here](https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/)***

# 1. Introduction

In this repo, we present a CNN architecture to identify the genre of a movie based on its poster. This problem falls into the category of **multi-label** image classification, meaning each movie can belong to more than one genre, with different 'weight'.

If you are not familiar with multi-label image classification you can read more [here](https://en.wikipedia.org/wiki/Multi-label_classification) or a a very nice explanation on how convolutional neural networks (CNN) do work [here](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050?gi=1d0493b5231).

# 2. Load the data

You can download the full dataset in a structured way [here](https://drive.google.com/file/d/1dNa_lBUh4CNoBnKdf9ddoruWJgABY1br/view). This folder contains the folder *images* which includes all the images needed to train our model and the *train.csv* file which contains the true genre for each image.

At first, we load all the packeges we will need for this task

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score
#%matplotlib inline

from keras.preprocessing import image
from skimage import io, img_as_float

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

import time
import os
```

We will use solely PyTorch in this tutorial, we will only use keras to import the images in the desired shape.

We load the data as:

```
train = pd.read_csv('Multi_Label_dataset/train.csv')

train.drop_duplicates(inplace=True)
train.reset_index(inplace=True, drop=True)
```
The genre column contains the list for each image which specifies the genre of that movie. So, from the head of the .csv file, the genre of the first image is Comedy and Drama.

The remaining 25 columns are the one-hot encoded columns. So, if a movie belongs to the Action genre, its value will be 1, otherwise 0. The image can belong to 25 different genres.

As the use of GPUs is widely needed for training neural networks like that, the issue of RAM limitation arises. To solve this problem, we define a function to load data by chunks.

```
def load_data(names):

    train_image = []
    for i in names:
        #img = image.load_img(data_folder + '/' +  'Images/'+train['Id'][i]+'.jpg',target_size=(400,400,3))
        #img = io.imread(data_folder + '/' +  'Images/'+train['Id'][i]+'.jpg')
        #img = torch.tensor(img_as_float(img))
        img = image.load_img('Multi_Label_dataset/Images/'+i+'.jpg',target_size=(400,300,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    #X = torch.stack(train_image)    
    X = np.array(train_image)
    
    return(X)
```

# 3. Defining the model

We first check if there is a CUDA capable GPU to speed up the training of the model; if not, one CPU unit will be used.

```
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)
```

Then we can define the architecture of the model as:

```
class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (1, number_of_unique_kos, genome_size)

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5)).to(device)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2)).to(device)
        self.drop1 = nn.Dropout2d(p=0.25).to(device)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5)).to(device)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2)).to(device)
        self.drop2 = nn.Dropout2d(p=0.25).to(device)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5)).to(device)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2)).to(device)
        self.drop3 = nn.Dropout2d(p=0.25).to(device)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5)).to(device)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2)).to(device)
        self.drop4 = nn.Dropout2d(p=0.25).to(device)

        self.fc1 = torch.nn.Linear(64 * 15 * 21, 128).to(device)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.drop5 = nn.Dropout(p=0.5).to(device)

        self.fc2 = torch.nn.Linear(128, 64).to(device)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.drop6 = nn.Dropout(p=0.5).to(device)

        self.fc3 = torch.nn.Linear(64, 25).to(device)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))  
        x = self.pool2(x)  
        x = self.drop2(x)             

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.drop4(x) 

        x = x.view(-1, 64 * 15 * 21)
        x = self.fc1(x)
        x = self.drop5(x) 
        x = self.fc2(x)
        x = self.drop6(x) 
        x = self.fc3(x)

        return torch.sigmoid(x)
```

and print it in a Keras style summary as

```
net = SimpleCNN()
summary(net, (3,400,300))
```

# 4. Training the model

In the case of PyTorch, training a model is not that straighforward as in Keras. While it needs some more commands, it is very customizable to fit our needs.

Before we train the model, we need to define two functions: a custom function to measure accuracy and a function to split the data in batches. We define the functions below.

```
def accuracy_score(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))

def split_list(A, n):
    f = [A[i:i+n] for i in range(0, len(A), n)]
    return f
```

Now we are ready to define the forst parameters of our model; the number of epochs, the batch size and the learning rate.

```
n_epochs = 20
learning_rate = 1e-03
batch_size = 64
# Print all of the hyperparameters of the training iteration:
print("===== HYPERPARAMETERS =====")
print("epochs=", n_epochs)
print("learning_rate=", learning_rate)
print("batch size=", batch_size)
print("=" * 30)
```

We then define two other important settings: the loss function and the optimizer. 

For this kind of problems, the mostly used loss function in the Binary Cross Entropy function (more about loss functions [here](https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8)) and as an optimizer we use ADAM, am extention-variation of the Gradient Descent algorithm.

```
running_loss = 0.0
net = SimpleCNN()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss = torch.nn.BCELoss().to(device)
```

Now we are ready to train the model, printing the time needed and the loss function every once in a while. Plus, in the end of each epoch, we print the acuurance of the training data.

```
batches = split_list(X_train, batch_size)
batches_ind = split_list(range(len(X_train)), batch_size)
n_batches = len(batches)

N = len(X_train)

training_start_time = time.time()
for epoch in tqdm_notebook(range(n_epochs)):
    
    running_loss = 0.0
    print_every = 15 # N // 50
    start_time = time.time()
    total_train_loss = 0
    total_output = torch.zeros(len(X_train),25)
    
    for ind in range(n_batches):
        
        batch_elements = [el for el in batches[ind]]
        batch_ind = [el for el in batches_ind[ind]]

        input_data = torch.tensor(load_data(batch_elements))
        
        if len(input_data) != batch_size:
            batch_size = len(input_data)
        
        input_data = Variable(input_data).transpose(2,3).transpose(1,2).to(device)
        labels = torch.from_numpy(train[train.Id.isin(batch_elements)].drop(['Id','Genre'], axis=1).values).to(device)
    

        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        output = net(input_data)
        total_output[torch.tensor(batch_ind),:] = output.cpu().detach()
        loss_size = loss(output, labels.float())
        loss_size.backward()
        optimizer.step()   
        
        #Print statistics
        running_loss += loss_size.data.item()
        total_train_loss += loss_size.item()
        
        if ind % print_every == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.4f} took: {:.4f}s".format(
                epoch + 1, int(100 * (ind + 1) / n_batches), running_loss / print_every, time.time() - start_time))
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
    train_acc = accuracy_score(y_train,(total_output>0.5).data.numpy())
    print("\n" + "Train Accuracy: {:.4f}".format(train_acc))
```

Our models fits the training data with an accurace of ~90%; not bad at all! But now, let's test it into another, unseen case.

I am sure you are all familiar with Game of Thrones (too bad it's over now, eh?). We will use the Game of Thrones poster (can be downloaded [here](https://drive.google.com/file/d/1cfIE-42H4_UM-JERoctseLUpKwmd40YE/view)) to see if our models can make accurate predictions. 

We define the folowing function, which will take as an input the trained network and the location of the image and it will read it, predict the genres and print the 3 which have the biggest weight.

```
def predict(cnn, location):

    img = image.load_img(location,target_size=(400,300,3))
    img = image.img_to_array(img)
    img = img/255

    img =  Variable(torch.from_numpy(img)).transpose(0,2).transpose(1,2).to(device)
    img = img.view(1,3,400,-1)
    proba = cnn(img)

    classes = np.array(train.columns[2:])
    proba = proba.cpu().detach().numpy()
    top_3 = np.argsort(proba[0])[:-4:-1]
    for i in range(3):
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
```

By running the above function

```
predict(net, './GOT.jpg')
```

we see that our model classifies this series as

- Drama (0.479)
- Action (0.38)
- Thriler (0.328)

which in my opinion classifies the GoT prety accurately! 

# 4. Conclusion

We now have a working model that was trainied on a relatively small training dataset (~7000 images).
