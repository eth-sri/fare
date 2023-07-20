import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x.squeeze(dim=-1))


class Net_Reg(nn.Module):
    def __init__(self, input_size):
        super(Net_Reg, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.squeeze(dim=-1)


class Fea(nn.Module):
    def __init__(self, input_size):
        super(Fea, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x,inplace=False)
        x = self.fc2(x)
        x = F.relu(x,inplace=False)

        return x


class Clf(nn.Module):
    def __init__(self, input_size=200):
        super(Clf, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(-1)
        z = torch.sigmoid(x)
        return z


class Reg(nn.Module):
    def __init__(self, input_size=200):
        super(Reg, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x.squeeze(dim=-1)
