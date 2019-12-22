import torch
from models.modules.encoder_LSTM import EncoderLSTM
from models.modules.Decoder_Attn_LSTM import DecoderLSTM
from torch.nn.modules import Linear
from torch import optim
import torch.nn as nn
from datetime import datetime
import numpy as np
import random

class LSTMAutoEncoder:
    def __init__(self,base_path,hidden_size_enc=128,hidden_size_dec=128,word_emb_size=768,vocab_size=30522,lr=0.1,decoder_layers=1,encoder_layers=1,device="cpu",dropout=0.2,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name="LSTM_attn"
        self.vocab_size = vocab_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.decoder_layers = decoder_layers

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.encoder = EncoderLSTM(word_emb_size, hidden_size_enc,encoder_layers,dropout=dropout).to(self.device)
        self.decoder = DecoderLSTM(hidden_size_dec,vocab_size,decoder_layers,dropout=dropout).to(self.device)
        self.lin_latent = Linear(2 * hidden_size_enc, hidden_size_dec).to(self.device)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())+ list(self.lin_latent.parameters())

        self.optimizer = optim.Adam(parameters, lr=lr,weight_decay=l2)

        classW = torch.ones(vocab_size,device=self.device)
        classW[1] = 0

        self.criterion = nn.CrossEntropyLoss(weight=classW)
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())+ list(self.lin_latent.parameters())
        print("#parameters:", sum([np.prod(p.size()) for p in parameters]))

    def train(self,input_tensor, target_tensor, x_mask, y_mask,y_emb):

        batch_size = input_tensor.size(1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(1)

        self.encoder.train()
        self.decoder.train()
        self.lin_latent.train()
        self.optimizer.zero_grad()

        loss = 0
        enc_seq = torch.zeros(input_length, batch_size, self.hidden_size_enc * 2, device=self.device)

        for ei in range(input_length):
            encoder_in = input_tensor[ei]
            encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))
            enc_seq[ei] = encoder_output.squeeze(0)


        z = self.lin_latent(encoder_output[0])

        decoder_hidden = (torch.stack([ z for i in range(self.decoder_layers)], dim=0),
                          torch.stack([ z for i in range(self.decoder_layers)], dim=0))
        outputs = torch.zeros(target_length, batch_size, self.vocab_size, device=self.device)

        decoder_input = target_tensor[:,0].unsqueeze(0)

        for di in range(1,target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,enc_seq)

            mask = (y_mask[:, di] == 0)
            for b in range(batch_size):
                if mask[b]:
                    outputs[di,b] = decoder_output.squeeze(0)[b]

            use_teacher_forcing = True if random.random() < 0.5 else False

            if use_teacher_forcing:
                decoder_input = target_tensor[:,di].unsqueeze(0)
            else:
                decoder_input = torch.argmax(decoder_output,dim=2)


            b_target_tensor = target_tensor[:, di]
            loss += self.criterion(decoder_output.squeeze(0), b_target_tensor)

        loss/=(target_length-1)
        preds = outputs[1:]


        loss.backward()
        self.optimizer.step()

        return loss.item(),preds.detach().cpu().numpy()

    def predict(self,input_tensor,target_tensor,x_mask,y_mask,y_emb):
        with torch.no_grad():

            batch_size = input_tensor.size(1)
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(1)

            loss = 0
            enc_seq = torch.zeros(input_length, batch_size, self.hidden_size_enc * 2, device=self.device)

            for ei in range(input_length):
                encoder_in = input_tensor[ei]
                encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))
                enc_seq[ei] = encoder_output.squeeze(0)

            z = self.lin_latent(encoder_output[0])

            decoder_hidden = (torch.stack([z for i in range(self.decoder_layers)], dim=0),
                              torch.stack([z for i in range(self.decoder_layers)], dim=0))
            outputs = torch.zeros(target_length, batch_size, self.vocab_size, device=self.device)

            decoder_input = target_tensor[:, 0].unsqueeze(0)

            for di in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, enc_seq)

                mask = (y_mask[:, di] == 0)
                for b in range(batch_size):
                    if mask[b]:
                        outputs[di, b] = decoder_output.squeeze(0)[b]

                use_teacher_forcing = True if random.random() < 0.5 else False

                if use_teacher_forcing:
                    decoder_input = target_tensor[:, di].unsqueeze(0)
                else:
                    decoder_input = torch.argmax(decoder_output, dim=2)

                b_target_tensor = target_tensor[:, di]
                loss += self.criterion(decoder_output.squeeze(0), b_target_tensor)

            loss /= (target_length - 1)
            preds = outputs[1:]

        return loss.item(), preds.cpu().numpy()



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
            'optimizer_state_dict': self.optimizer.state_dict()
            }, save_path+"/encoder.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.decoder.state_dict()
            }, save_path+"/decoder.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.lin_latent.state_dict()
            }, save_path+"/lin_latent.pt")

    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"encoder.pt")
        self.encoder.load_state_dict(checkpoint["model_state_dict"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint = torch.load(save_path+"decoder.pt")
        self.decoder.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load(save_path+"lin_latent.pt")
        self.lin_latent.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint["epoch"]

        if train:
            self.encoder.train()
            self.decoder.train()
            self.lin_latent.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.lin_latent.eval()
        return epoch
