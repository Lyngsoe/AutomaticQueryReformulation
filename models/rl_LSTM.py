import torch
from models.modules.encoder_LSTM import EncoderLSTM
from models.modules.Decoder_LSTM import DecoderLSTM
from torch.nn.modules import Linear
from torch import optim
import torch.nn as nn
from datetime import datetime
import numpy as np
from torch.autograd import Variable

class LSTMAutoEncoder:
    def __init__(self,base_path,reward_function,hidden_size=128,word_emb_size=768,vocab_size=30522,lr=0.1,decoder_layers=1,encoder_layers=1,device="cpu",dropout=0.2,l2=0,exp_name=None):
        self.base_path = base_path
        self.save_path = self.base_path + "experiments/"
        self.device = device
        self.model_name="LSTM"
        self.vocab_size = vocab_size
        self.reward_function = reward_function

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()

        self.encoder = EncoderLSTM(word_emb_size, hidden_size,encoder_layers,dropout=dropout).to(self.device)
        self.decoder = DecoderLSTM(1,hidden_size,decoder_layers,dropout=dropout).to(self.device)

        self.output = Linear(hidden_size, vocab_size).to(self.device).to(self.device)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())  + list(self.output.parameters())

        self.optimizer = optim.SGD(parameters, lr=lr,weight_decay=l2)
        #self.optimizer = optim.Adam(parameters, lr=lr,weight_decay=l2)

        classW = torch.ones(vocab_size,device=self.device)
        classW[1] = 0

        self.criterion = nn.CrossEntropyLoss(weight=classW)
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())  + list(self.output.parameters())
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        print("#parameters:", sum([np.prod(p.size()) for p in parameters]))

    def train_predict(self,input_tensor,mask):
        #print("x_in:",input_tensor.size())
        #print("y_in:", target_tensor.size())

        batch_size = input_tensor.size(1)
        input_length = input_tensor.size(0)
        target_length = input_length

        #self.encoder.initHidden(self.device,batch_size=batch_size)

        #print("enc hidden:",encoder_hidden.size())
        self.encoder.train()
        self.decoder.train()
        self.output.train()
        self.optimizer.zero_grad()

        #input_tensor = input_tensor.to(self.device)
        #target_tensor = target_tensor.to(self.device)
        for ei in range(input_length):
            encoder_in = input_tensor[ei]
            encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))

        outputs = torch.zeros(target_length,batch_size, self.hidden_size, device=self.device)
        decoder_hidden = (torch.stack([encoder_hidden[0][0] for i in range(self.decoder_layers)], dim=0),
                          torch.stack([encoder_hidden[1][0] for i in range(self.decoder_layers)], dim=0))
        decoder_input = torch.Tensor(1,batch_size,1).fill_(2).cuda().float()
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            decoder_output2 = torch.zeros_like(decoder_output)
            i_mask = (mask[:, di] == 0)
            decoder_output2.squeeze(0)[i_mask] = decoder_output.squeeze(0)[i_mask]
            decoder_output = decoder_output2
            pred = self.output(decoder_output).squeeze(0)
            outputs[di] = decoder_output
            decoder_input = torch.argmax(pred,dim=1).unsqueeze(0).unsqueeze(2).float()

        self.preds = self.output(outputs)

        return self.preds.detach().cpu().numpy()

    def predict(self,input_tensor,mask):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            self.output.eval()
            batch_size = input_tensor.size(1)
            input_length = input_tensor.size(0)
            target_length = input_length

            for ei in range(input_length):
                encoder_in = input_tensor[ei]
                encoder_output, encoder_hidden = self.encoder(encoder_in.unsqueeze(0))

            outputs = torch.zeros(target_length, batch_size, self.hidden_size, device=self.device)
            decoder_hidden = (torch.stack([encoder_hidden[0][0] for i in range(self.decoder_layers)], dim=0),
                              torch.stack([encoder_hidden[1][0] for i in range(self.decoder_layers)], dim=0))
            decoder_input = torch.Tensor(1,batch_size,1).fill_(2).cuda().float()
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output2 = torch.zeros_like(decoder_output)
                i_mask = (mask[:, di] == 0)
                decoder_output2.squeeze(0)[i_mask] = decoder_output.squeeze(0)[i_mask]
                decoder_output = decoder_output2

                pred = self.output(decoder_output).squeeze(0)
                outputs[di] = decoder_output

                decoder_input = torch.argmax(pred,dim=1).unsqueeze(0).unsqueeze(2).float()

            self.preds = self.output(outputs)

        return self.preds.cpu().numpy()

    def calc_reward(self,*args,**kwargs):
        return self.reward_function(*args,**kwargs)

    def update_policy(self,reward,base_reward,update = True):
        # Update network weights
        reward = Variable(reward[:self.preds.size(0)].unsqueeze(2))
        base_reward = Variable(base_reward[:self.preds.size(0)].unsqueeze(2))

        batch_size = reward.size(1)
        self.preds = torch.softmax(self.preds, dim=2).double()
        h = torch.Tensor(batch_size).zero_().to(self.device).type(torch.float64)
        for i in range(self.preds.size(0)):
            h += torch.sum(torch.mul(self.preds[i], torch.log(self.preds[i])), dim=1)
        regularization = torch.mean(torch.Tensor(batch_size).fill_(0.001).to(self.device).type(torch.float64) * h)

        # print("reward:", reward.size())
        q_expected = torch.log(self.preds) * reward
        # print("q_expected:",q_expected.size())
        q_expected = torch.mean(torch.sum(q_expected.squeeze(0), dim=0))
        # print("q_expected:", q_expected)

        q0_expected = torch.log(self.preds) * base_reward
        # print("q0_expected:", q0_expected.size())
        q0_expected = torch.mean(torch.sum(q0_expected.squeeze(0), dim=0))
        # print("q0_expected:", q0_expected)

        # print(regularization)
        loss = -((q_expected - q0_expected) - regularization)

        if update:
            loss.backward()
            self.optimizer.step()
        # print("loss:",loss.item())

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
