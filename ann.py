import torch
from torch import nn


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN, self).__init__()
        self.input = nn.Linear(input_dim, 800)#13, 666
        self.hidden = nn.Linear(800, 800)#666, 300
        self.hidden2 = nn.Linear(800, 350)#300, 128
        self.hidden3 = nn.Linear(350, 128)  # 300, 128
        self.output = nn.Linear(128, output_dim)#128, 5
        # batchnorm
        self.input_bn = nn.BatchNorm1d(800)
        self.hidden_bn = nn.BatchNorm1d(800)
        self.hidden2_bn = nn.BatchNorm1d(350)
        self.hidden3_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x= input_data
        #x = torch.mean(input_data, dim=3)
        #x = torch.mean(x, dim=1)
        #x = torch.flatten(input_data, start_dim=1)
        #print(x)
        x = self.input(x)
        x = self.input_bn(x)
        x = self.relu(x)

        #x = self.hidden(x)
        #x = self.hidden_bn(x)
        #x = self.relu(x)

        x = self.hidden2(x)
        x = self.hidden2_bn(x)
        x = self.relu(x)

        x = self.hidden3(x)
        x = self.hidden3_bn(x)
        x = self.relu(x)

        logits = self.output(x)
        prediction = self.softmax(logits)

        return prediction
