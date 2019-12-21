import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self,hidden_size,output_size,layers,embeddings=None,dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=layers,dropout=dropout)
        self.lin_out = nn.Linear(hidden_size, output_size)
        self.lin_in = nn.Embedding(output_size, hidden_size)
        self.drops = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)
        self.layers=layers
        self.lin_attn = nn.Linear(hidden_size*2, hidden_size)

        if embeddings is not None:
            self.lin_in.weight = nn.Parameter(embeddings)
            self.lin_in.weight.requires_grad = False


    def forward(self, input,hidden,enc_seq):
        input = F.relu(self.drops(self.lin_in(input)))

        attn = self.calc_attn(hidden[0],enc_seq)

        output, hidden = self.lstm(input,(attn,hidden[1]))
        output = self.softmax(self.lin_out(output))
        return output, hidden



    def calc_attn(self,ht,enc_seq):
        layer_context = []

        for layer in range(self.layers):
            htl = self.lin_attn(enc_seq)
            attn_weights = torch.zeros_like(htl).double()
            for t in range(enc_seq.size(0)):
                attn_weights[t] = torch.mul(ht,htl[t])
            layer_context.append(torch.sum(attn_weights,dim=0))
        layer_context = torch.stack(layer_context)
        return layer_context