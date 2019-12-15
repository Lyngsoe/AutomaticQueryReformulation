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
        self.layers=layers
        self.lin_attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, input,hidden,enc_seq):
        input = self.lin_in(input)
        input = self.drops(input)
        input = F.relu(input)

        attn = self.calc_attn(hidden[0],enc_seq)

        output, hidden = self.lstm(input,(attn,hidden[1]))
        output = self.softmax(self.lin_out(output))
        return output, hidden



    def calc_attn(self,ht,enc_seq):
        layer_context = []

        for layer in range(self.layers):
            attn_weights = torch.zeros_like(enc_seq)
            htl = self.lin_attn(ht[layer])
            for t in range(enc_seq.size(0)):
                attn_weights[t] = torch.mul(htl,enc_seq[t])
            layer_context.append(torch.sum(attn_weights,dim=0))
        layer_context = torch.stack(layer_context)
        return layer_context