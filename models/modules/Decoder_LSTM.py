import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,output_size,layers,dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=layers,dropout=dropout)
        self.lin_out = nn.Linear(hidden_size, output_size)
        self.lin_in = nn.Linear(input_size, hidden_size)
        self.drops = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input,hidden):
        input = self.lin_in(input)
        input = self.drops(input)
        input = F.relu(input)
        output, hidden = self.lstm(input,hidden)
        output = self.softmax(self.lin_out(output))
        return output, hidden