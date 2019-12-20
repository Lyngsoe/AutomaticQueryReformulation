import torch
import torch.nn as nn
from torch.nn.modules.transformer import Transformer
from torch.nn.modules import Linear
import numpy as np
from datetime import datetime
import math

class MyTransformer:

    def __init__(self, base_path,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"
        self.vocab_size = output_size
        self.lr = lr
        self.d_mdodel = d_model

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.linear_in = Linear(input_size,d_model).cuda().double()
        self.model = Transformer(d_model=d_model,nhead=nhead,num_decoder_layers=num_layers,num_encoder_layers=num_layers,dim_feedforward=dff,dropout=dropout).cuda().double()
        self.linear_out = Linear(d_model,output_size).cuda().double()
        self.target_in = nn.Embedding(output_size, d_model, padding_idx=1).cuda().double()


        self.pos_enc = PositionalEncoding(d_model,dropout).cuda().double()


        classW = torch.ones(output_size, device=self.device).double()
        classW[1] = 0

        self.criterion = nn.CrossEntropyLoss(weight=classW)

        params = list(self.linear_in.parameters()) + list(self.model.parameters()) + list(self.linear_out.parameters()) + list(self.target_in.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr,weight_decay=l2)

        params = list(self.linear_in.parameters()) + list(self.model.parameters()) + list(self.linear_out.parameters()) + list(self.target_in.parameters())
        print("#parameters:", sum([np.prod(p.size()) for p in params]))

    def train(self,x,y,x_mask,y_mask,y_emb):

        self.optimizer.zero_grad()
        self.model.train()

        mask = (torch.triu(torch.ones(y_emb.size(0),y_emb.size(0))) == 1).transpose(0, 1).float()
        tgt_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
        x_mask = x_mask == 1
        y_mask = y_mask == 1
        lin_in = self.linear_in(x)
        lin_in = self.pos_enc(lin_in)
        tgt = self.target_in(y.view(y.size(1),y.size(0)))
        tgt = self.pos_enc(tgt)

        output = self.model(lin_in,tgt,tgt_mask=tgt_mask,src_key_padding_mask=x_mask,tgt_key_padding_mask=y_mask)
        preds = self.linear_out(output)[:-1]

        loss = 0
        for i in range(x.size(1)):
            p = preds[:, i]
            y_b = y[i,1:]
            loss += self.criterion(p,y_b)

        loss/=y.size(0)

        loss.backward()
        self.optimizer.step()

        return loss.item(),preds.detach().cpu().numpy()

    def predict(self,x,y,x_mask,y_mask,y_emb):

        with torch.no_grad():
            self.model.eval()
            max_len = y_emb.size(0)

            m_out = torch.zeros(max_len,x.size(1)).cuda().long()
            m_out[0] = torch.Tensor(x.size(1),1).fill_(2).cuda().long()

            lin_in = self.linear_in(x)
            lin_in = self.pos_enc(lin_in)
            x_mask = x_mask == 1
            y_mask = y_mask == 1


            for i in range(1, max_len):

                mask = (torch.triu(torch.ones(i,i)) == 1).transpose(0, 1).float()
                tgt_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
                tgt_in = self.pos_enc(self.target_in(m_out[:i]))
                out = self.model(lin_in,tgt_in, tgt_mask=tgt_mask,src_key_padding_mask=x_mask,tgt_key_padding_mask=y_mask[:,:i])
                m_out[i] = torch.argmax(self.linear_out(out[-1]))

            preds = self.linear_out(out)

            loss = 0
            for i in range(x.size(1)):
                p = preds[:, i]
                y_b = y[i,1:]
                loss += self.criterion(p, y_b)

            loss /= y.size(0)


        return loss.item(),preds.cpu().numpy()

    def predict_search(self,x):

        with torch.no_grad():
            self.model.eval()
            max_len = 20

            m_out = torch.zeros(max_len,x.size(1)).cuda().long()
            m_out[0] = torch.Tensor(x.size(1),1).fill_(2).cuda().long()

            lin_in = self.linear_in(x)
            lin_in = self.pos_enc(lin_in)


            for i in range(1, max_len):

                mask = (torch.triu(torch.ones(i,i)) == 1).transpose(0, 1).float()
                tgt_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
                tgt_in = self.pos_enc(self.target_in(m_out[:i]))
                out = self.model(lin_in,tgt_in, tgt_mask=tgt_mask)
                m_out[i] = torch.argmax(self.linear_out(out[-1]))

            preds = self.linear_out(out)

        return preds.cpu().numpy()


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
            'model_state_dict': self.linear_in.state_dict(),
        }, save_path + "/linear_in.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, save_path+"/model.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.linear_out.state_dict(),
        }, save_path + "/linear_out.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.target_in.state_dict(),
        }, save_path + "/target_in.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"/model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        checkpoint = torch.load(save_path + "/linear_in.pt")
        self.linear_in.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load(save_path + "/linear_out.pt")
        self.linear_out.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load(save_path + "/target_in.pt")
        self.target_in.load_state_dict(checkpoint["model_state_dict"])

        if train:
            self.model.train()
            self.linear_in.train()
            self.linear_out.train()
        else:
            self.model.eval()
            self.linear_in.eval()
            self.linear_out.eval()

        return epoch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
