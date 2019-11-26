import torch
from models.modules.lstm_module import LSTM
from torch.nn.modules import Linear
from torch import optim
import torch.nn as nn
from datetime import datetime
import numpy as np

class LSTMSimple:
    def __init__(self,base_path,reward_function,hidden_size=128,word_emb_size=768,vocab_size=30522,lr=0.1,layers=1,device="cpu",exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name="LSTM"
        self.vocab_size = vocab_size
        self.reward_function = reward_function

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.lstm = LSTM(word_emb_size, hidden_size,layers).to(self.device)
        self.output = Linear(hidden_size, vocab_size).to(self.device).to(self.device)

        parameters = list(self.lstm.parameters()) + list(self.output.parameters())

        self.optimizer = optim.SGD(parameters, lr=lr)

        parameters = list(self.lstm.parameters()) + list(self.output.parameters())
        print("#parameters:", sum([np.prod(p.size()) for p in parameters]))


    def forward(self,input_tensor,target_tensor):

        batch_size = input_tensor.size(1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        self.optimizer.zero_grad()

        preds = torch.zeros(input_length, self.vocab_size, device=self.device)

        for i in range(target_length):
            lstm_in = input_tensor[i]
            lstm_out = self.lstm(lstm_in)
            pred = self.output(lstm_out)
            preds[i] = pred.squeeze(0)

        return preds.cpu().numpy()

    def calc_reward(self,search_results,target):
        return self.reward_function(search_results,target)

    def update_policy(self,reward):
        # Update network weights
        self.optimizer.zero_grad()
        reward.backward()
        self.optimizer.step()

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
            'model_state_dict': self.lstm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, save_path+"/lstm.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.output.state_dict()
        }, save_path + "/output.pt")


    def load(self,save_path,train):

        checkpoint = torch.load(save_path+"lstm.pt")
        self.lstm.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint = torch.load(save_path+"output.pt")
        self.output.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint["epoch"]

        if train:
            self.lstm.train()
            self.output.train()
        else:
            self.lstm.eval()
            self.output.eval()

        return epoch
