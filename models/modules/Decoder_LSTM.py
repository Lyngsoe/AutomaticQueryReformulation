import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1, max_length=20):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size * 3,1)
        self.attn_combine = nn.Linear(self.hidden_size,self.hidden_size * 2)
        self.lstm = nn.LSTM(3*self.hidden_size, self.hidden_size,batch_first=True)
        self.out = nn.Linear(self.hidden_size,self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        #print("dec inpt:",input.size())
        #print("hidden:", hidden[0].size())
        #print("encoder_outputs:", encoder_outputs.size())
        weights = []
        for i in range(self.max_length):
            #print("hidden0:",hidden[0][0].shape)
            #print("enc0:",encoder_outputs[:,i].shape)
            weights.append(self.attn(torch.cat((hidden[0][0],encoder_outputs[:,i]), dim=1)))


        normalized_weights = F.softmax(torch.cat(weights, 1), 1)
        #print("normalized_weights:",normalized_weights.size())
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),encoder_outputs)
        #print("attn_applied:", attn_applied.size())
        input_lstm = torch.cat((attn_applied, input), dim = 2) #if we are using embedding, use embedding of input here instead
        #print("input_lstm:", input_lstm.size())
        output, hidden = self.lstm(input_lstm, hidden)
        #print("output:", output.size())
        #pred = F.log_softmax(self.out(output), dim=1)
        #print("lin_output:", pred.size())
        pred = self.out(output)
        pred = pred.squeeze(1)
        #print("lin_output:", pred.size())
        return output, hidden,pred
