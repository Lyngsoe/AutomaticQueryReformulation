import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,layers):
        super(LSTM, self).__init__()
        self.num_layers=layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=layers)

    def forward(self, input):
        output, hidden = self.lstm(input.float())
        return output, hidden

    def initHidden(self,device,batch_size):
        hidden = Variable(next(self.parameters()).data.new(self.num_layers,batch_size,self.hidden_size)).double()
        cell = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size)).double()
        return hidden.type(torch.double), cell.type(torch.double)