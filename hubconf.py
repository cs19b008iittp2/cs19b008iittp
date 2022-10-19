dependencies = ['torch']

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
def load_data():

    train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    )
    test_data = datasets.FashionMNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    return train_data, test_data
    
def get_dataloaders(train_data, test_data):
    loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    return loaders

