import torch
from models.modules.encoder_LSTM import EncoderLSTM
from models.modules.Decoder_Attn_LSTM import DecoderLSTM
from torch.nn.modules import Linear
from torch import optim
import torch.nn as nn
from datetime import datetime
import numpy as np

class LSTMAutoEncoder:
    def __init__(self,base_path,hidden_size=128,word_emb_size=768,vocab_size=30522,lr=0.1,decoder_layers=1,encoder_layers=1,device="cpu",dropout=0.2,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name="LSTM_attn"
        self.vocab_size = vocab_size

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.encoder = EncoderLSTM(word_emb_size, hidden_size,encoder_layers,dropout=dropout).to(self.device)
        self.decoder = DecoderLSTM(hidden_size,decoder_layers,dropout=dropout).to(self.device)

        self.output = Linear(hidden_size, vocab_size).to(self.device).to(self.device)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.output.parameters())

        #self.optimizer = optim.SGD(parameters, lr=lr)
        self.optimizer = optim.Adam(parameters, lr=lr,weight_decay=l2)

        classW = torch.ones(vocab_size,device=self.device)
        classW[1] = 0

        self.criterion = nn.CrossEntropyLoss(weight=classW)
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.output.parameters())
        self.decoder_layers = decoder_layers
        print("#parameters:", sum([np.prod(p.size()) for p in parameters]))

    def train(self,input_tensor, target_tensor,y_tok, x_mask, y_mask):
        #print("x_in:",input_tensor.size())
        #print("y_in:", target_tensor.size())

        batch_size = input_tensor.size(1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        #self.encoder.initHidden(self.device,batch_size=batch_size)

        #print("enc hidden:",encoder_hidden.size())
        self.encoder.train()
        self.decoder.train()
        self.output.train()
        self.optimizer.zero_grad()

        encoder_outputs = torch.zeros(input_length,batch_size, self.encoder.hidden_size, device=self.device)
        #print("enc_out_init:",encoder_outputs.size())

        loss = 0
        #input_tensor = input_tensor.to(self.device)
        #target_tensor = target_tensor.to(self.device)
        for ei in range(input_length):
            encoder_in = input_tensor[ei]
            encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))
            mask = x_mask[:,ei] == 0
            encoder_outputs[ei][mask] = encoder_output[0][mask]

        preds = torch.zeros(target_length,batch_size, self.vocab_size, device=self.device)
        decoder_hidden = (torch.stack([encoder_hidden[0][0] for i in range(self.decoder_layers)],dim=0),
                          torch.stack([encoder_hidden[1][0] for i in range(self.decoder_layers)],dim=0))

        decoder_input = encoder_output
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_output2 = torch.zeros_like(decoder_output)
            mask = (y_mask[:, di] == 0)
            decoder_output2.squeeze(0)[mask] = decoder_output.squeeze(0)[mask]
            decoder_output = decoder_output2
            b_target_tensor = y_tok[di].type(torch.long)
            pred = self.output(decoder_output).squeeze(0)
            preds[di] = pred
            loss += self.criterion(pred, b_target_tensor)

        loss.backward()
        self.optimizer.step()

        return loss.item()/target_length,preds.detach().cpu().numpy()

    def predict(self,input_tensor,target_tensor,x_mask,y_mask):
        #print("x_in:",x.size())
        with torch.no_grad():

            batch_size = input_tensor.size(1)
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            # self.encoder.initHidden(self.device,batch_size=batch_size)

            # print("enc hidden:",encoder_hidden.size())

            self.optimizer.zero_grad()

            encoder_outputs = torch.zeros(input_length, batch_size, self.encoder.hidden_size , device=self.device)
            # print("enc_out_init:",encoder_outputs.size())

            loss = 0

            for ei in range(input_length):
                encoder_in = input_tensor[ei]
                encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))

                mask = x_mask[:, ei] == 0
                encoder_outputs[ei][mask] = encoder_output[0][mask]

            preds = torch.zeros(target_length,self.vocab_size, device=self.device)

            decoder_hidden = (torch.stack([encoder_hidden[0][0] for i in range(self.decoder_layers)], dim=0),
                              torch.stack([encoder_hidden[1][0] for i in range(self.decoder_layers)], dim=0))
            decoder_input = encoder_output
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
                decoder_output2 = torch.zeros_like(decoder_output)
                mask = (y_mask[:, di] == 0)
                decoder_output2.squeeze(0)[mask] = decoder_output.squeeze(0)[mask]
                decoder_output = decoder_output2
                b_target_tensor = target_tensor[di].type(torch.long)
                pred = self.output(decoder_output)
                preds[di] = pred.squeeze(0)

                loss += self.criterion(pred.squeeze(0), b_target_tensor)

        return loss.item()/target_length, preds.unsqueeze(1).cpu().numpy()


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
            'model_state_dict': self.output.state_dict()
        }, save_path + "/output.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"encoder.pt")
        self.encoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint = torch.load(save_path+"decoder.pt")
        self.decoder.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load(save_path+"output.pt")
        self.output.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint["epoch"]

        if train:
            self.encoder.train()
            self.decoder.train()
            self.output.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.output.eval()

        return epoch
