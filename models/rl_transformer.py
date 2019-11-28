import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer
from models.modules.transformer_encoder import MyTransformerEncoder
from models.modules.transformer_decoder import MyTransformerDecoder
from models.modules.transformer import Transformer
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from datetime import datetime
from torch.autograd import Variable

class RLTransformer:

    def __init__(self, base_path,reward_function,input_size=768,d_model=128,dff=2048,num_layers=6,output_size=30522,nhead=12,device="gpu",dropout=0.2,lr=0.05,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name = "Transformer"
        self.reward_function = reward_function

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        encoder_layer = TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.encoder = MyTransformerEncoder(encoder_layer, num_layers,in_size=input_size,d_model=d_model,norm=LayerNorm(d_model))

        decoder_layer = TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dff,dropout=dropout)
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers,d_model=d_model,output_size=output_size,norm=LayerNorm(d_model))

        self.model = Transformer(d_model=input_size,nhead=nhead,custom_encoder=self.encoder,custom_decoder=self.decoder)

        self.vocab_size = output_size

        self.lr = lr  # learning rate
        params = self.model.parameters()
        self.optimizer = torch.optim.SGD(params, lr=self.lr,weight_decay=0.1)
        self.model.cuda()
        self.model.double()
        self.linear_output = nn.Linear(input_size,output_size)
        print("#parameter:", sum([np.prod(p.size()) for p in self.model.parameters()]))

    def predict(self,x):
        #print("x:",x.size())
        batch_size = x.size(1)
        max_len = x.size(0)
        tgt = torch.ones((max_len,batch_size), device=self.device).type(torch.float64)
        #print("tgt:", tgt.size())
        tgt[0,:] = 2
        mask = (torch.triu(torch.ones(max_len, max_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        mask = mask.double()
        for i in range(1,max_len):

            in_x = x[:i]
            in_tgt = tgt[:i]
            #print("in_x:", in_x.size())
            #print("in_tgt:", in_tgt.size())
            output = self.model(in_x,in_tgt,tgt_mask=mask[:i,:i])
            #print("output:", output.size())
            prediction = torch.argmax(output[-1],dim=1)
            #print("prediction:",prediction.size())
            tgt[i,:] = prediction

        self.preds = output
        #print("output:", output.size())
        preds_no_grad = output

        preds_no_grad = preds_no_grad


        return preds_no_grad.detach().cpu().numpy()

    def calc_reward(self,search_results,target):
        return self.reward_function(search_results,target)

    def update_policy(self,reward):
        # Update network weights
        batch_size = len(reward)
        reward_t = torch.zeros(self.preds.size(0),batch_size).to(self.device).type(torch.float64)

        for i in range(batch_size):
            reward_t[:,i] = reward[i]

        probs = torch.softmax(self.preds,dim=2)
        #print("probs:",probs.size())
        #print("reward:", reward_t.size())
        #print("reward:", reward_t)
        h = torch.Tensor(batch_size).zero_().to(self.device).type(torch.float64)
        for i in range(probs.size(0)):
            h+= torch.sum (torch.mul( probs[i] , torch.log(probs[i]) ) ,dim=1)


        regularization = torch.mean( - torch.Tensor(batch_size).fill_(0.001).to(self.device).type(torch.float64) * h)
        #print("regularization:",regularization.size())
        #print("regularization:", regularization)

        l1 = torch.log(probs)
        #print("l1:", l1.size())
        l2 = l1 * Variable(reward_t.unsqueeze(2))
        #print("l2:",l2.size())
        s1 = torch.mean( torch.sum( torch.mean( l2 ,dim=2 ) , dim=0 ))
        #print("s1:", s1.size())
        #print("s1 item:", s1.item())
        loss = s1 + regularization

        self.optimizer.zero_grad()
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
