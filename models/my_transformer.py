import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer
#from models.modules.transformer_encoder import MyTransformerEncoder
from torch.nn.modules.transformer import Transformer,TransformerDecoder,TransformerEncoder
#from models.modules.transformer_decoder import MyTransformerDecoder
#from models.modules.transformer import Transformer
from torch.nn.modules import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import Dropout
import numpy as np
from datetime import datetime
import math

class MyTransformer:

    def __init__(self, base_path,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.linear_in = Linear(input_size,d_model).cuda().double()
        #encoder_layer = TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        #self.encoder = MyTransformerEncoder(encoder_layer, num_layers,norm=LayerNorm(d_model))

        #decoder_layer = TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        #self.decoder = MyTransformerDecoder(decoder_layer, num_layers,norm=LayerNorm(d_model))

        self.model = Transformer(d_model=d_model,nhead=nhead,num_decoder_layers=num_layers,num_encoder_layers=num_layers,dim_feedforward=dff,dropout=dropout)
        self.linear_out = Linear(d_model,output_size).cuda().double()
        self.pos_enc = PositionalEncoding(d_model,dropout).cuda().double()
        self.vocab_size = output_size
        classW = torch.ones(output_size, device=self.device).double()
        classW[1] = 0
        self.criterion = nn.CrossEntropyLoss(weight=classW)
        self.lr = lr  # learning rate
        params = list(self.linear_in.parameters()) + list(self.model.parameters()) + list(self.linear_out.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr,weight_decay=l2)
        #self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=l2)
        self.model.cuda()
        self.model.double()
        self.d_mdodel=d_model

        print("#parameter:", sum([np.prod(p.size()) for p in self.model.parameters()]))

    def train(self,x,y,x_mask,y_mask,y_emb):

        self.optimizer.zero_grad()
        self.model.train()

        mask = (torch.triu(torch.ones(y_emb.size(0),y_emb.size(0))) == 1).transpose(0, 1).float()
        tgt_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
        mask = (torch.triu(torch.ones(x.size(0), x.size(0))) == 1).transpose(0, 1).float()
        src_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
        x_mask = x_mask == 1
        y_mask = y_mask == 1
        lin_in = self.linear_in(x)
        lin_in = self.pos_enc(lin_in)
        #y[0] = sos_token
        tgt = self.linear_in(y_emb)
        tgt = self.pos_enc(tgt)


        output = self.model(lin_in,tgt,src_mask=src_mask,tgt_mask=tgt_mask,src_key_padding_mask=x_mask,tgt_key_padding_mask=y_mask)
        output = output.masked_fill(torch.isnan(output), 0)
        lin_out = self.linear_out(output)[:-1]

        preds = lin_out
        #print("preds",preds.size())

        #print("output resize:",output.view(-1,self.vocab_size).size())
        loss = 0
        for i in range(x.size(1)):
            p = preds[:, i]
            y_b = y[i,1:]
            loss += self.criterion(p,y_b)

        loss/=y.size(0)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item(),lin_out.detach().cpu().numpy()

    def predict(self,x,y,x_mask,y_mask,y_emb):

        with torch.no_grad():
            self.model.eval()
            max_len = y_emb.size(0)

            m_out = torch.zeros(y_emb.size(0),y_emb.size(1),self.d_mdodel).cuda().double()


            m_out[0] = self.linear_in(y_emb[0])
            #print(y.size())
            #m_out[0] = self.linear_in(x[0])
            lin_in = self.linear_in(x)
            lin_in = self.pos_enc(lin_in)
            x_mask = x_mask == 1
            y_mask = y_mask == 1
            #cat = torch.ones(1, m_out.size(1), m_out.size(2)).cuda().double()
            #m_out = torch.cat((m_out, cat))
            for i in range(1, max_len):

                mask = (torch.triu(torch.ones(i,i)) == 1).transpose(0, 1).float()
                tgt_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
                mask = (torch.triu(torch.ones(x.size(0), x.size(0))) == 1).transpose(0, 1).float()
                src_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
                #print(y_mask.size())
                #print(y_mask[:, :i].size())
                out = self.model(lin_in,self.pos_enc(m_out[:i]),src_mask=src_mask, tgt_mask=tgt_mask,src_key_padding_mask=x_mask,tgt_key_padding_mask=y_mask[:,:i])
                #print(torch.argmax(out,dim=2))
                #m_out = torch.cat((m_out, cat))
                m_out[i] = out[-1]

            m_out = m_out.masked_fill(torch.isnan(m_out), 0)
            lin_out = self.linear_out(m_out)[1:]

            #print(torch.argmax(m_out,dim=2))
            #print(torch.argmax(out, dim=2))
            preds = lin_out
            # print("preds",preds.size())

            # print("output resize:",output.view(-1,self.vocab_size).size())
            loss = 0
            for i in range(x.size(1)):
                p = preds[:, i]
                y_b = y[i,1:]
                loss += self.criterion(p, y_b)

            loss /= y.size(0)


        return loss.item(),lin_out.cpu().numpy()

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


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"/model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        checkpoint = torch.load(save_path + "/linear_in.pt")
        self.linear_in.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load(save_path + "/linear_out.pt")
        self.linear_out.load_state_dict(checkpoint["model_state_dict"])

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
