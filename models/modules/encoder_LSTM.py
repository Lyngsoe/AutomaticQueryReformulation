import torch.nn as nn
import torch
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,bidirectional=True)

    def forward(self, input, hidden):
        #print("enc input:",input.size())
        #print("enc input:", input.size())
        #print("enc emb:",embedded.size())
        output, hidden = self.lstm(input.float())
        #print("output:",output.size())
        lantent_weights = self.latent_space(output)
        #print("lantent_weights:",lantent_weights.size())
        return output, hidden

    def initHidden(self,device,batch_size):
        hidden = Variable(next(self.parameters()).data.new(self.num_layers,batch_size,self.hidden_size)).double()
        cell = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size)).double()
        return hidden.type(torch.double), cell.type(torch.double)