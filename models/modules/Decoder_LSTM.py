import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_length):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size * 3,1)
        self.attn_combine = nn.Linear(self.hidden_size,self.hidden_size * 2)
        self.lstm = nn.LSTM(3*self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        #print("dec inpt:",input.size())
        #print("hidden:", hidden[0].size())
        #print("encoder_outputs:", encoder_outputs.size())
        weights = []

        for i in range(self.max_length):
            #print("hidden0:",hidden[0][0].size())
            #print("enc0:",encoder_outputs[i].size())
            weights.append(self.attn(torch.cat((hidden[0][0],encoder_outputs[i]), dim=1)))


        weights_t = torch.cat(weights,dim=1).view(encoder_outputs.size(0),-1)
        #print("weights_t:", weights_t.size())
        normalized_weights = F.softmax(weights_t,dim=1).unsqueeze(2)
        #print("normalized_weights:",normalized_weights.size())
        attn_applied = torch.mul(normalized_weights,encoder_outputs)
        #print("attn_applied:", attn_applied.size())
        attn_sum = torch.sum(attn_applied,dim=0)
        #print("attn_sum:", attn_sum.size())
        input_lstm = torch.cat((attn_sum, input), dim = 1).unsqueeze(0)
        #print("input_lstm:", input_lstm.size())
        output, hidden = self.lstm(input_lstm, hidden)
        #print("output:", output.size())
        #pred = F.log_softmax(self.out(output), dim=1)
        #print("lin_output:", pred.size())
        pred = self.out(output)
        pred = pred.squeeze(0)
        #print("lin_output:", pred.size())
        return output, hidden,pred
