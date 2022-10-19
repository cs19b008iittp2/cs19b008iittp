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

class cs19bNN(nn.Module):
  pass
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class YourRollNumberNN(nn.Module):
  pass
