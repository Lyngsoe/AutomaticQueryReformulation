import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size,layers):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn = nn.Linear(self.hidden_size * 3,1)
        self.attn_combine = nn.Linear(self.hidden_size,self.hidden_size * 2)
        self.lstm = nn.LSTM(3*self.hidden_size, self.hidden_size,num_layers=layers)
        self.out = nn.Linear(self.hidden_size,self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):

        weights = []

        for i in range(encoder_outputs.size(0)):

            weights.append(self.attn(torch.cat((hidden[0][0],encoder_outputs[i]), dim=1)))


        weights_t = torch.cat(weights,dim=1).view(encoder_outputs.size(0),-1)

        normalized_weights = F.softmax(weights_t,dim=1).unsqueeze(2)

        attn_applied = torch.mul(normalized_weights,encoder_outputs)

        attn_sum = torch.sum(attn_applied,dim=0)

        input_lstm = torch.cat((attn_sum, input), dim = 1).unsqueeze(0)
        output, hidden = self.lstm(input_lstm, hidden)
        pred = self.out(output)
        pred = pred.squeeze(0)
        return output, hidden,pred
