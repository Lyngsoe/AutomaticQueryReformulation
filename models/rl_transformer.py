import torch
import torch.nn as nn
from torch.nn.modules.transformer import Transformer
import numpy as np
from datetime import datetime
from torch.nn.modules import Linear
from torch.autograd import Variable
import math

class RLTransformer:

    def __init__(self, base_path,reward_function,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"
        self.vocab_size = output_size
        self.lr = lr
        self.d_mdodel = d_model
        self.reward_function = reward_function

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

    def train_predict(self,x,q_mask):
        #print("x:",x.size())
        self.model.train()
        self.linear_out.train()
        self.linear_in.train()
        self.optimizer.zero_grad()
        max_len = 20

        m_out = torch.Tensor(1,x.size(1)).fill_(2).cuda().long()

        lin_in = self.linear_in(x)
        lin_in = self.pos_enc(lin_in)

        q_mask = q_mask == 1

        for i in range(1,max_len):

            mask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1).float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
            tgt_in = self.pos_enc(self.target_in(m_out[:i]))
            out = self.model(lin_in,tgt_in,tgt_mask=mask,src_key_padding_mask=q_mask)
            cur_pred = torch.argmax(self.linear_out(out[-1]),dim=1)
            m_out = torch.cat((m_out,cur_pred.unsqueeze(0)))


        self.preds = self.linear_out(out)
        preds_no_grad = self.preds.detach()

        return preds_no_grad.cpu().numpy()

    def predict(self,x,q_mask):
        self.optimizer.zero_grad()
        self.model.eval()
        self.linear_out.eval()
        self.linear_in.eval()
        max_len = 20

        m_out = torch.Tensor(1, x.size(1)).fill_(2).cuda().long()

        lin_in = self.linear_in(x)
        lin_in = self.pos_enc(lin_in)

        q_mask = q_mask == 1

        for i in range(1, max_len):
            mask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1).float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()
            tgt_in = self.pos_enc(self.target_in(m_out[:i]))
            out = self.model(lin_in, tgt_in, tgt_mask=mask, src_key_padding_mask=q_mask)
            cur_pred = torch.argmax(self.linear_out(out[-1]), dim=1)

            m_out = torch.cat((m_out, cur_pred.unsqueeze(0)))

        self.preds = self.linear_out(out)

        return self.preds.detach().cpu().numpy()

    def calc_reward(self,*args,**kwargs):
        return self.reward_function(*args,**kwargs)

    def update_policy(self,reward,update = True):
        # Update network weights
        reward = Variable(torch.from_numpy(np.array(reward)).double().cuda())


        self.preds = torch.softmax(self.preds,dim=2)
        weights = torch.log(torch.max(self.preds,dim=2)[0])
        weights = torch.matmul(weights,reward)

        loss = -(torch.mean(weights))

        if update:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def get_exp_name(self):
        now = datetime.now()
        return self.model_name+"__"+now.strftime("%m-%d_%H:%M")

    def save_latest(self,epoch):
        save_path = self.save_path + self.exp_name + "/latest"
        self._save(epoch,save_path)
    def save_best(self,epoch):
        save_path = self.save_path + self.exp_name + "/best"
        self._save(epoch,save_path)

    def _save(self, epoch, save_path):
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
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
