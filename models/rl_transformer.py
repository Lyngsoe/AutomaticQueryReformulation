import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer
from models.modules.transformer_encoder import MyTransformerEncoder
from models.modules.transformer_decoder import MyTransformerDecoder
from models.modules.transformer import Transformer
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from datetime import datetime
from torch.nn.modules import Linear
from torch.autograd import Variable

class RLTransformer:

    def __init__(self, base_path,reward_function,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"
        self.reward_function = reward_function

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.linear_in = Linear(input_size, d_model).cuda().double()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.encoder = MyTransformerEncoder(encoder_layer, num_layers,norm=LayerNorm(d_model))

        decoder_layer = TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers,norm=LayerNorm(d_model))

        self.model = Transformer(d_model=input_size,nhead=nhead,custom_encoder=self.encoder,custom_decoder=self.decoder)
        self.linear_out = Linear(d_model, output_size).cuda().double()

        self.vocab_size = output_size
        self.d_model = d_model
        self.lr = lr  # learning rate
        params = list(self.linear_in.parameters()) + list(self.model.parameters()) + list(self.linear_out.parameters())
        #self.optimizer = torch.optim.SGD(params, lr=self.lr,weight_decay=l2)
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=l2)
        self.model.cuda()
        self.model.double()
        print("#parameter:", sum([np.prod(p.size()) for p in self.model.parameters()]))


    def train_predict(self,x,q_mask):
        #print("x:",x.size())
        self.model.train()
        self.linear_out.train()
        self.linear_in.train()
        self.optimizer.zero_grad()
        max_len = 20

        sos_token = torch.Tensor(x.size(1), x.size(2)).fill_(0.1).cuda().double()
        m_out = self.linear_in(x[0]).unsqueeze(0)

        lin_in = self.linear_in(x)

        q_mask = q_mask == 1

        for i in range(1,max_len):

            mask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1).float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()

            out = self.model(lin_in,m_out[:i],tgt_mask=mask,src_key_padding_mask=q_mask)
            m_out = torch.cat((m_out,out[-1].unsqueeze(0)))

        m_out = m_out.masked_fill(torch.isnan(m_out), 0)

        self.preds = self.linear_out(m_out)
        preds_no_grad = self.preds.detach()

        return preds_no_grad.cpu().numpy()

    def predict(self,x,q_mask):
        #print("x:",x.size())
        #with torch.no_grad():
        self.model.eval()
        self.linear_out.eval()
        self.linear_in.eval()

        max_len = 20
        sos_token = torch.Tensor(x.size(1), x.size(2)).fill_(0.1).cuda().double()
        m_out = self.linear_in(x[0]).unsqueeze(0)

        lin_in = self.linear_in(x)

        q_mask = q_mask == 1

        for i in range(1,max_len):
            mask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1).float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda().double()

            out = self.model(lin_in, m_out, tgt_mask=mask, src_key_padding_mask=q_mask)
            m_out = torch.cat((m_out,out[-1].unsqueeze(0)))

        m_out = m_out.masked_fill(torch.isnan(m_out), 0)
        self.preds = self.linear_out(m_out)

        return self.preds.detach().cpu().numpy()

    def calc_reward(self,*args,**kwargs):
        return self.reward_function(*args,**kwargs)

    def update_policy(self,reward,base_reward,update = True):
        # Update network weights
        reward = Variable(reward[:self.preds.size(0)].unsqueeze(2))
        base_reward = Variable(base_reward[:self.preds.size(0)].unsqueeze(2))

        batch_size = reward.size(1)
        self.preds = torch.softmax(self.preds,dim=2)
        h = torch.Tensor(batch_size).zero_().to(self.device).type(torch.float64)
        for i in range(self.preds.size(0)):
            h+= torch.sum (torch.mul( self.preds[i] , torch.log(self.preds[i]) ) ,dim=1)
        regularization = torch.mean(torch.Tensor(batch_size).fill_(0.001).to(self.device).type(torch.float64) * h)

        #print("reward:", reward.size())
        q_expected = torch.log(torch.max(self.preds)) * reward
        #print("q_expected:",q_expected.size())
        q_expected = torch.mean(torch.sum(q_expected.squeeze(0), dim=0))
        #print("q_expected:", q_expected)

        q0_expected = torch.log(torch.max(self.preds)) * base_reward
        #print("q0_expected:", q0_expected.size())
        q0_expected = torch.mean(torch.sum(q0_expected.squeeze(0), dim=0))
        #print("q0_expected:", q0_expected)

        #print(regularization)
        loss = -((q_expected - q0_expected) - regularization)

        if update:
            loss.backward()
            self.optimizer.step()
        #print("loss:",loss.item())

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
        }, save_path + "/model.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.linear_out.state_dict(),
        }, save_path + "/linear_out.pt")

    def load(self, save_path, train):

        checkpoint = torch.load(save_path + "/model.pt")
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
