import torch
import random
from models.modules.encoder_LSTM import EncoderLSTM
from models.modules.Decoder_LSTM import AttnDecoderLSTM
from torch import optim
import torch.nn as nn
import json
from datetime import datetime
import numpy as np
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

teacher_forcing_ratio = 0.5
learning_rate=0.01

class LSTMAutoEncoder:
    def __init__(self,drive_path,language,hidden_size = 1024,word_emb_size=1024,vocab_size=73637,device="gpu",debug=False,exp_name=None):

        self.debug = debug
        self.base_path = drive_path + "raffle_wiki/{}/debug/".format(language) if debug else drive_path + "raffle_wiki/{}/".format(language)
        self.device = device
        self.id2bpe = json.load(open(self.base_path + "id2bpe.json", 'r'))
        self.model_name="LSTM_auto_encoder"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()


        self.vocab_size = vocab_size
        self.encoder = EncoderLSTM(word_emb_size, hidden_size).to(self.device)
        self.decoder = AttnDecoderLSTM(hidden_size, vocab_size, dropout_p=0.1).to(self.device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()


    def train(self,input_tensor, target_tensor, max_length=MAX_LENGTH):

        #print("x_in:",input_tensor.size())
        #print("y_in:", target_tensor.size())
        encoder_hidden = self.encoder.initHidden(self.device,batch_size =8)
        #print("enc hidden:",encoder_hidden.size())
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        batch_size = input_tensor.size(0)
        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)

        encoder_outputs = torch.zeros(batch_size,max_length, self.encoder.hidden_size*2, device=self.device)
        lantent_space_outputs = torch.zeros(batch_size, max_length, self.encoder.hidden_size, device=self.device)
        #print("enc_out_init:",encoder_outputs.size())
        loss = 0

        for ei in range(input_length):
            encoder_in = input_tensor[:,ei]
            encoder_output, encoder_hidden,latent_out = self.encoder(encoder_in, encoder_hidden)

            encoder_outputs[:,ei] = encoder_output[:, 0]
            lantent_space_outputs[:,ei] = latent_out[:,0]

        decoder_input = torch.mean(lantent_space_outputs,1).unsqueeze(1)
        decoder_hidden = (encoder_hidden[0][0].unsqueeze(0),encoder_hidden[1][0].unsqueeze(0))
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print("decoder_hidden:", decoder_hidden[0].size())
            #print("encoder_outputs:", encoder_outputs.size())
            #print("decoder_input:", decoder_input.size())
            decoder_output, decoder_hidden,pred = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output  # detach from history as input
            #print("decoder_output:", decoder_output.size())
            #print("pred:", pred.size())
            #print("target_tensor[di]:", target_tensor[:,di].size())
            b_target_tensor = torch.argmax(target_tensor[:,di],1).type(torch.long)
            #print("b_target_tensor:", b_target_tensor.size())
            loss += self.criterion(pred,b_target_tensor)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


        return loss.item() / target_length


    def predict(self,x, max_length=MAX_LENGTH):
        with torch.no_grad():


            batch_size = x.size(0)
            input_length = x.size(1)

            encoder_hidden = self.encoder.initHidden(self.device, batch_size=batch_size)
            encoder_outputs = torch.zeros(batch_size, max_length, self.encoder.hidden_size * 2, device=self.device)
            lantent_space_outputs = torch.zeros(batch_size, max_length, self.encoder.hidden_size, device=self.device)
            # print("enc_out_init:",encoder_outputs.size())
            loss = 0

            for ei in range(input_length):
                encoder_in = x[:, ei]
                encoder_output, encoder_hidden, latent_out = self.encoder(encoder_in, encoder_hidden)

                encoder_outputs[:, ei] = encoder_output[:, 0]
                lantent_space_outputs[:, ei] = latent_out[:, 0]

            decoder_outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=self.device)
            decoder_input = torch.mean(lantent_space_outputs, 1).unsqueeze(1)
            decoder_hidden = (encoder_hidden[0][0].unsqueeze(0), encoder_hidden[1][0].unsqueeze(0))

            # Without teacher forcing: use its own predictions as the next input
            for di in range(max_length):
                # print("decoder_hidden:", decoder_hidden[0].size())
                # print("encoder_outputs:", encoder_outputs.size())
                # print("decoder_input:", decoder_input.size())
                decoder_output, decoder_hidden, pred = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output  # detach from history as input
                # print("decoder_output:", decoder_output.size())
                # print("pred:", pred.size())
                # print("target_tensor[di]:", target_tensor[:,di].size())
                decoder_outputs[:,di] = decoder_input

        bpe_tokens = self.get_tokens(decoder_outputs.item())
        sentences = self.compress(bpe_tokens)

        return decoder_outputs.item()


    def compress(self,bpe_tokens):
        sentences = []
        for pred in bpe_tokens:
            tokens = " ".join(pred)
            tokens = tokens.replace("@@ ", "")
            tokens = tokens.replace("@@", "")
            sentences.append(tokens)
        return sentences

    def get_tokens(self,model_out):
        indices = np.argmax(model_out,2)
        # print("index", indices.shape, type(indices))
        bpe_tokens = []
        for sample in list(indices):
            sent_bpe = []
            for ind in sample:
                token = self.id2bpe.get(str(ind))
                sent_bpe.append(token)
            bpe_tokens.append(sent_bpe)

        return bpe_tokens

    def get_exp_name(self):
        now = datetime.now()
        self.exp_name = self.model_name+"__"+now.strftime("%m-%d_%H:%M")

