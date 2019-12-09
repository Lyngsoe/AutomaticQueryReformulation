import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size,layers,dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size,num_layers=layers,dropout=dropout)
        self.attn_linear = nn.Linear(hidden_size,hidden_size)


    def forward(self, input,hidden,enc_seq):

        attention = self.calc_attention(input,enc_seq)
        lstm_input = torch.cat((input,attention.unsqueeze(0)),dim=2)
        output, hidden = self.lstm(lstm_input,hidden)
        return output, hidden

    def calc_attention(self,input,enc_seq):
        #print("input",input.size())
        #print("enc_seq", enc_seq.size())
        attention = torch.zeros(enc_seq.size(0),input.size(1),input.size(2)).cuda()
        #print("attention", attention.size())
        for t in range(enc_seq.size(0)):
            attn_lin = self.attn_linear(input[0])
            #print("attn_lin", attn_lin.size())
            attention[t] = torch.mul(attn_lin,enc_seq[t])

        attention = torch.softmax(attention,dim=0)
        #print("attention", attention.size())
        attention = torch.mul(enc_seq,attention)
        #print("attention", attention.size())
        attention = torch.sum(attention,dim=0)
        #print("attention", attention.size())
        return attention