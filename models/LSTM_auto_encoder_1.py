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

learning_rate=0.01

class LSTMAutoEncoder:
    def __init__(self,base_path,max_length,hidden_size,word_emb_size,lantent_space_size,vocab_size,device="gpu",exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.max_length = max_length
        self.device = device
        self.model_name="LSTM_auto_encoder_1"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()


        self.vocab_size = vocab_size
        self.latent_space_size = lantent_space_size
        self.encoder = EncoderLSTM(word_emb_size, hidden_size,lantent_space_size).to(self.device)
        self.decoder = AttnDecoderLSTM(hidden_size, vocab_size,max_length=self.max_length).to(self.device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        classW = torch.ones(vocab_size,device=self.device)
        classW[1] = 0
        #print("cw:",classW.size())

        self.criterion = nn.CrossEntropyLoss(weight=classW)


    def train(self,input_tensor, target_tensor):

        #print("x_in:",input_tensor.size())
        #print("y_in:", target_tensor.size())
        batch_size = input_tensor.size(1)
        encoder_hidden = self.encoder.initHidden(self.device,batch_size =batch_size)
        #print("enc hidden:",encoder_hidden.size())
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length,batch_size, self.encoder.hidden_size*2, device=self.device)
        lantent_space_outputs = torch.zeros(self.max_length,batch_size, self.latent_space_size, device=self.device)
        #print("enc_out_init:",encoder_outputs.size())
        loss = 0

        for ei in range(input_length):
            encoder_in = input_tensor[ei]
            encoder_output, encoder_hidden,latent_out = self.encoder(encoder_in, encoder_hidden)

            encoder_outputs[ei] = encoder_output[0]
            lantent_space_outputs[ei] = latent_out[0]

        decoder_input = torch.mean(lantent_space_outputs,dim=0)
        #print("decoder_input",decoder_input.size())
        decoder_hidden = (encoder_hidden[0][0].unsqueeze(0),encoder_hidden[1][0].unsqueeze(0))
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print("decoder_hidden:", decoder_hidden[0].size())
            #print("encoder_outputs:", encoder_outputs.size())
            #print("decoder_input:", decoder_input.size())
            decoder_output, decoder_hidden,pred = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output.squeeze(0)  # detach from history as input
            #print("decoder_output:", decoder_output.size())
            #print("pred:", pred.size())
            b_target_tensor = target_tensor[di].type(torch.long)
            #print("b_target_tensor:", b_target_tensor.size())
            loss += self.criterion(pred,b_target_tensor)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


        return loss.item() / target_length


    def predict(self,x,y):
        #print("x_in:",x.size())
        with torch.no_grad():


            batch_size = x.size(1)
            input_length = x.size(0)
            target_length = y.size(0)

            encoder_hidden = self.encoder.initHidden(self.device, batch_size=batch_size)
            encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size * 2, device=self.device)
            lantent_space_outputs = torch.zeros(self.max_length, batch_size, self.latent_space_size, device=self.device)
            # print("enc_out_init:",encoder_outputs.size())
            loss = 0

            for ei in range(input_length):
                encoder_in = x[ei]
                encoder_output, encoder_hidden, latent_out = self.encoder(encoder_in, encoder_hidden)

                encoder_outputs[ei] = encoder_output[0]
                lantent_space_outputs[ei] = latent_out[0]

            decoder_pred = torch.zeros(self.max_length,batch_size, self.vocab_size, device=self.device)
            decoder_input = torch.mean(lantent_space_outputs, dim=0)
            # print("decoder_input",decoder_input.size())
            decoder_hidden = (encoder_hidden[0][0].unsqueeze(0), encoder_hidden[1][0].unsqueeze(0))

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # print("decoder_hidden:", decoder_hidden[0].size())
                # print("encoder_outputs:", encoder_outputs.size())
                # print("decoder_input:", decoder_input.size())
                decoder_output, decoder_hidden, pred = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.squeeze(0)  # detach from history as input
                # print("decoder_output:", decoder_output.size())
                #print("pred:", pred.size())
                decoder_pred[di] = pred
                b_target_tensor = y[di].type(torch.long)
                # print("b_target_tensor:", b_target_tensor.size())
                loss += self.criterion(pred, b_target_tensor)

        return loss.item()/target_length,decoder_pred.squeeze(1).cpu().numpy()


    def get_exp_name(self):
        now = datetime.now()
        return self.model_name+"__"+now.strftime("%m-%d_%H:%M")

    def save_latest(self,epoch):
        save_path = self.save_path + self.exp_name + "/latest"
        self._save(epoch,save_path)
    def save_best(self,epoch):
        save_path = self.save_path + self.exp_name + "/best"
        self._save(epoch,save_path)
    def _save(self,epoch,save_path):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.encoder_optimizer.state_dict()
            }, save_path+"/encoder.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.decoder_optimizer.state_dict()
            }, save_path+"/decoder.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"encoder.pt")
        self.encoder.load_state_dict(checkpoint["model_state_dict"])
        self.encoder_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint = torch.load(save_path+"decoder.pt")
        self.decoder.load_state_dict(checkpoint["model_state_dict"])
        self.decoder_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()