import torch.nn as nn


# modify Net1T to be used when performing hyper parameter optimization.
class Net1T_HPO(nn.Module):
    
    def __init__(self, inpSize, outSize, numHiddenLayers, experiment):
        super(Net1T_HPO, self).__init__()
        
        hiddenSize = experiment.get_parameter('hiddenSize')
        dropout = experiment.get_parameter('dropout')
        
        self.hiddenLayers = nn.ModuleList()
        self.hiddenLayers.append(nn.Linear(inpSize, hiddenSize))
        for i in range(numHiddenLayers-1):
            self.hiddenLayers.append(nn.Linear(hiddenSize, hiddenSize))
        self.out = nn.Linear(hiddenSize, outSize)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        for i in range(len(self.hiddenLayers)):
            x = self.relu(self.dropout(self.hiddenLayers[i](x)))
        x = self.out(x)
        
        return x


# one task network with all fully-connected layers
class Net1T(nn.Module):
    
    def __init__(self, inpSize, outSize, numHiddenLayers, hiddenSize, dropout):
        super(Net1T, self).__init__()
        
        self.hiddenLayers = nn.ModuleList()
        self.hiddenLayers.append(nn.Linear(inpSize, hiddenSize))
        for i in range(numHiddenLayers-1):
            self.hiddenLayers.append(nn.Linear(hiddenSize, hiddenSize))
        self.out = nn.Linear(hiddenSize, outSize)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        for i in range(len(self.hiddenLayers)):
            x = self.relu(self.dropout(self.hiddenLayers[i](x)))
        x = self.out(x)
        
        return x