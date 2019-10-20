import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.transformer import TransformerModel
import numpy as np
import json
from datetime import datetime

class Transformer:

    def __init__(self, base_path,vocab_size, ninp, nhead, nhid, nlayers,device, dropout=0.5,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.id2bpe = json.load(open(self.base_path + "id2bpe.json", 'r'))
        self.model_name = "Transformer1"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.model = TransformerModel(vocab_size,ninp,nhead,nhid,nlayers,device,dropout)
        self.vocab_size = vocab_size
        classW = torch.ones(vocab_size, device=self.device)
        classW[0] = 0
        self.criterion = nn.CrossEntropyLoss(weight=classW)
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def train(self,x,y):
        #print("x:",x.size())
        #print("y:",y.size())
        self.model.train()  # Turn on the train mode

        self.optimizer.zero_grad()
        output = self.model(x)
        targets = torch.argmax(y,2).type(torch.long)
        #print("targets:",targets.size())
        #print("targets resize:",targets.view(-1).size())
        #print("output:",output.size())
        #print("output resize:",output.view(-1,self.vocab_size).size())
        loss = self.criterion(output.view(-1,self.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def predict(self,x,y):
        self.model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            output = self.model(x)
            output_flat = output.view(-1, self.vocab_size)
            targets = torch.argmax(y, 2).type(torch.long)
            loss = self.criterion(output_flat, targets.view(-1)).item()

            bpe_tokens = self.get_tokens(output.cpu().numpy())
            sentences = self.compress(bpe_tokens)

        return loss,sentences

    def compress(self,bpe_tokens):
        sentences = []
        for pred in bpe_tokens:
            tokens = " ".join(pred)
            tokens = tokens.replace("@@ ", "")
            tokens = tokens.replace("@@", "")
            sentences.append(tokens)
        return sentences

    def get_tokens(self,model_out):
        indices = np.argmax(model_out,2)
        # print("index", indices.shape, type(indices))
        bpe_tokens = []
        for sample in list(indices):
            sent_bpe = []
            for ind in sample:
                token = self.id2bpe.get(str(ind))
                sent_bpe.append(token)
            bpe_tokens.append(sent_bpe)

        return bpe_tokens

    def get_exp_name(self):
        now = datetime.now()
        return self.model_name+"__"+now.strftime("%m-%d_%H:%M")

    def save(self,epoch):
        save_path = self.save_path+self.exp_name

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.state_dict()
            }, save_path+"/model.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.load_state_dict(checkpoint["optimizer_state_dict"])

        if train:
            self.model.train()
        else:
            self.model.eval()