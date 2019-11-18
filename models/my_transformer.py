import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer
from models.modules.transformer_encoder import MyTransformerEncoder
from models.modules.transformer_decoder import MyTransformerDecoder
from models.modules.transformer import Transformer
from torch.nn.modules.normalization import LayerNorm
import numpy as np
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
        self.model.cuda()
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
        #tgt = self.encoder.linear(y)
        start_token = torch.zeros(y.size(1),device=self.device).type(torch.float64).unsqueeze(0)
        tgt = torch.cat((start_token,y),dim=0)
        tgt_mask = self.model.generate_square_subsequent_mask(y.size(0))
        tgt_mask=tgt_mask.cuda()
        tgt_mask=tgt_mask.double()
        output = self.model(x,tgt,tgt_mask=tgt_mask)
        #print("targets:",targets.size())
        #print("targets resize:",targets.view(-1).size())
        #print("output:",output.size())
        preds = output.view(-1,self.vocab_size)
        #print("preds",preds.size())

        #print("output resize:",output.view(-1,self.vocab_size).size())
        loss = self.criterion(preds,targets)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def predict(self,x,y):
        self.model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            size = y.size(0)
            max_len = size
            tgt = torch.ones_like(y, device=self.device).type(torch.float64)
            tgt[0] = 2
            mask = (torch.triu(torch.ones(max_len, max_len)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.cuda()
            mask = mask.double()
            for i in range(1,max_len):

                in_x = x[:i]
                in_tgt = tgt[:i]

                output = self.model(in_x,in_tgt,tgt_mask=mask[:i,:i])
                #print(output.size())
                prediction = torch.argmax(output[-1])
                tgt[i] = prediction
                #print(prediction)


            output_flat = output.view(-1, self.vocab_size)
            targets = y.type(torch.long).view(-1)
            loss = self.criterion(output_flat, targets[:max_len-1]).item()
            #print(tgt.view(-1))


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
