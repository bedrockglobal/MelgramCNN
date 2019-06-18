import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class GetDataset(Dataset):
    """
    Class used to represent training and testing datasets for model building and validation.
    ...

    Attributes
    ----------
    train_flag : bool
        boolean indicating if representing training or testing dataset
    root_dir: str
        path of directory where images are stored
    files: list
        list of files (images) held by class
    transform: torchvision.transform
        image transformation to apply to image (convert to tensor)

    Methods
    -------
    __len__()
        get the length, or size of dataset

    __getitem__(idx)
        get data from a specified index
    """
    def __init__(self, train_set, transform):
        """
        Parameters
        ----------
        train_set : bool
            boolean indicating if representing training or testing dataset
        transform : torchvision.transform object
            image transformations to apply to image (convert to tensor)
        """
        self.train_flag = train_set
        if self.train_flag:
            self.root_dir = './melgram_dataset/trainset'
        else:
            self.root_dir = './melgram_dataset/testset'
        self.files = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        :param idx: Index specifying image to retrieve.
        :return: Dictionary of sample. If training, image and label is obtained.
                                       If testing, Image and ID is obtained.
        """
        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name).convert('L')
        image = self.transform(image)
        if self.train_flag:
            label = self.files[idx].split('_')[2]
            sample = {'image': image, 'label': float(label)}
        else:
            seg_id = self.files[idx][:-4]
            sample = {'image': image, 'seg_id': seg_id}
        return sample


class Net(nn.Module):
    """
    Class used to represent melgramCNN, a convolutional neural network used for regression
    and feature extraction is done from mel-spectrograms.
    ...

    Attributes
    ----------
    conv1 :
        first convolutional layer, takes in 1 feature map and outputs 16 feature maps
    conv2 :
        second convolutional layer, takes in 16 feature maps and outputs 32 feature maps
    conv3 :
        third convolutional layer, takes in 32 feature maps and outputs 64 feature maps
    pool:
        pooling layer used in all convolutional layers, 2x2 kernel size and stride of 2.
    conv1_bn :
        batch normalization layer used after activation in first convolutional layer
    conv2_bn :
        batch normalization layer used after activation in second convolutional layer
    conv3_bn :
        batch normalization layer used after activation in third convolutional layer
    fc1 :
        first fully connected layer (input: 64 * 8 * 20, output:256)
    fc2 :
        second fully connected layer (input: 256, output: 64)
    fc3 :
        third fully connected layer (input: 64, output: 1)
    fc1_bn :
        batch normalization for first fully connected layer, used before sigmoid activation
    fc2_bn :
        batch normalization for second fully connected layer, used before sigmoid activation

    Methods
    -------
    forward (x)
        Forward propagates batch of samples to the CNN

    """

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=1,padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2,stride=1,padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64 * 8 * 20, out_features=256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """
        Parameters
        ----------
        param x: Batches of images of shape [batch_size, num_channels, height, width]
        return: Batch of forward propagated samples of shape [batch_size, 1]
        """
        x = x.view(-1, 1, 64, 158)
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = x.view(-1, 64 * 8 * 20)
        x = torch.sigmoid(self.fc1_bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


class TrainNet:
    """
    Class used to train and optimize MelgramCNN network.
    ...

    Attributes
    ----------
    net: subclass of nn.Module
        PyTorch model to train
    train_set: str
        path of directory where images are stored
    test_set: list
        list of files (images) held by class
    num_epochs: int
        number of epochs to perform in training

    Methods
    -------
    eval_model(data_loader)
        Return the loss of the model given a dataloader

    train(learn_rate)
        Executes training steps (forward propagation and optimization)
    """

    def __init__(self, net, train_set, test_set, num_epochs):
        self.net = net
        self.train_loader = train_set
        self.test_loader = test_set
        self.criterion = nn.MSELoss()
        self.num_epochs = num_epochs

    def eval_model(self, data_loader):
        """
        Returns the loss of the model given a dataloader
        :param data_loader: torch.utils.data.DataLoader object
        :return: running loss of the model
        """
        self.net.eval()
        running_loss = 0.0
        for batch_iter, batch in enumerate(data_loader, 1):
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).float()
            labels = labels.view(-1, 1)
            labels = labels.cuda()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

        return running_loss / batch_iter

    def train(self, learn_rate=0.001, weight_decay=1e-5):
        """
        Exectutes training steps (forward propagation and optimization)
        :param learn_rate: Learning rate for optimization steps
        :param weight_decay: Weight decay for optimization steps
        :return: None. Updates MelgramCNN model parameters
        """
        optimizer = optim.Adam(self.net.parameters(), lr=learn_rate, weight_decay=weight_decay)
        train_losses, test_losses = [], []
        print('------' * 10)
        for epoch in tqdm(range(self.num_epochs)):

            train_losses.append(self.eval_model(self.train_loader))
            test_losses.append(self.eval_model(self.test_loader))

            # train step
            self.net.train()
            running_train_loss = 0.0
            for batch_iter_train, train_batch in enumerate(self.train_loader, 1):
                inputs, labels = train_batch['image'], train_batch['label']
                inputs = inputs.cuda()

                labels = torch.from_numpy(np.array(labels)).float()
                labels = labels.view(-1, 1)
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = self.net(inputs)
                train_loss = self.criterion(outputs, labels)
                running_train_loss += train_loss.item()

                train_loss.backward()
                optimizer.step()

            print('Epoch {} \n train loss: {:4.6f} ----- test loss {:4.6f} '.format(epoch,
                                                                                    train_losses[-1],
                                                                                    test_losses[-1]))