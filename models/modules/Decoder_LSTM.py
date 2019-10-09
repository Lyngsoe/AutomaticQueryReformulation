import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 20

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True)
        self.out = nn.Linear(self.hidden_size,self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        #print("dec inpt:",input.size())
        #print("encoder_outputs:", encoder_outputs.size())

        #print("dec inpt:", input.size())
        output, hidden = self.lstm(input, hidden)
        #print("output:", output.size())
        #pred = F.log_softmax(self.out(output), dim=1)
        #print("lin_output:", pred.size())
        pred = self.out(output)
        pred = pred.squeeze(1)
        #print("lin_output:", pred.size())
        return output, hidden,pred
