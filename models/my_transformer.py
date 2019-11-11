import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer
from models.modules.transformer_encoder import MyTransformerEncoder
from models.modules.transformer_decoder import MyTransformerDecoder
from models.modules.transformer import Transformer
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from models.losses.neg import SGNS
import json
from datetime import datetime

class MyTransformer:

    def __init__(self, base_path,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        encoder_layer = TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.encoder = MyTransformerEncoder(encoder_layer, num_layers,in_size=input_size,d_model=d_model,norm=LayerNorm(d_model))

        decoder_layer = TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers,d_model=d_model,output_size=output_size,norm=LayerNorm(d_model))

        self.model = Transformer(d_model=input_size,nhead=nhead,custom_encoder=self.encoder,custom_decoder=self.decoder)

        self.vocab_size = output_size
        classW = torch.ones(output_size, device=self.device).double()
        classW[1] = 0
        self.criterion = nn.CrossEntropyLoss(weight=classW)
        self.lr = lr  # learning rate
        params = self.model.parameters()
        self.optimizer = torch.optim.SGD(params, lr=self.lr)
        #self.model.cuda()
        self.model.double()
        self.linear_output = nn.Linear(input_size,output_size)
        print("#parameter:", sum([np.prod(p.size()) for p in self.model.parameters()]))

    def train(self,x,y):
        #print("x:",x.size())
        #print("y:",y.size())
        self.model.train()  # Turn on the train mode

        self.optimizer.zero_grad()
        #x = x.double()
        #y = y.double()
        targets = y.type(torch.long).view(-1)
        tgt = self.encoder.linear(x)
        output = self.model(x,tgt)
        #print("targets:",targets.size())
        #print("targets resize:",targets.view(-1).size())
        #print("output:",output.size())
        preds = output.view(-1,self.vocab_size)
        #print("preds",preds.size())

        #print("output resize:",output.view(-1,self.vocab_size).size())
        loss = self.criterion(preds,targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def predict(self,x,y):
        self.model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            tgt = self.encoder.linear(x)
            output = self.model(x,tgt)
            output_flat = output.view(-1, self.vocab_size)
            targets = y.type(torch.long).view(-1)
            loss = self.criterion(output_flat, targets).item()


        return loss,output_flat.cpu().numpy()

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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.state_dict()
            }, save_path+"/model.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"/model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        if train:
            self.model.train()
        else:
            self.model.eval()

        return epoch
