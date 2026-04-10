import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

df = pd.read_csv('dataset_simple.csv')
X = torch.tensor(df.iloc[0:, 0].values) # age as feature
y = torch.tensor(df.iloc[0:, 1].values) # income as target

input_size = 1
hidden_size = 16
output_size = 1

class IncomePredNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(IncomePredNN, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.act3 = nn.ReLU() # income cant be negative

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x
    
