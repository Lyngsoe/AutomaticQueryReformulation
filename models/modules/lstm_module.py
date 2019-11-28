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
        return output
