from dataset.bertify import construct_sentence,prune
from training.dataloaders.rl_squad_dataloader import RLSquadDataloader
from tqdm import tqdm
import torch
import os
import jsonlines
import json
import numpy as np

class Trainer:
    def __init__(self,model,base_path,search_engine,batch_size=1,epoch=0,max_epoch=50,device="gpu",max_seq_len=300,specs=None):
        self.model = model
        self.base_path = base_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.device = device
        self.max_seq_len = max_seq_len
        self.min_test_loss = None
        self.exp_path = model.save_path + model.exp_name
        self.search_engine = search_engine
        self.loss = None
        self.make_dir()
        self.info = json.load(open(base_path+"info.json"))
        self.data_points = self.info["qas"]
        self.eval_data_points = self.info["qas_eval"]
        if specs is not None:
            specs.update({"model_name:":model.exp_name})
            json.dump(specs,open(self.exp_path+"/specs.json",'w'))

    def evaluate(self,train_loss,base_reward):
        eval_data = RLSquadDataloader(base_path=self.base_path,batch_size=1,max_length=self.max_seq_len,eval=True)
        i_eval = 0
        test_loss = 0
        test_reward = 0
        w = self.get_test_reward()
        pbar = tqdm(total=5928, desc="evaluating batches for epoch {}".format(self.epoch))
        for q_emb, relevant_documents,q_txt,q_mask in iter(eval_data):

            q_emb = torch.tensor(q_emb, device=self.device).type(torch.float64)
            q_emb = q_emb.reshape(q_emb.size(1), 1, q_emb.size(2))
            q_mask = torch.tensor(q_mask, device=self.device).type(torch.float64)

            predictions = self.model.predict(q_emb,q_mask)
            predicted_sentence = construct_sentence(predictions)
            sentence_cutoff = prune(predicted_sentence)
            search_results = self.search_engine.search(sentence_cutoff)
            reward,base_reward,normarlized_reward = self.model.reward_function(search_results,relevant_documents)
            loss = self.model.update_policy(normarlized_reward,update=False)


            w.write({"reward": reward, "sentence": sentence_cutoff})


            test_loss+=loss
            test_reward+=reward
            pbar.update()
            if i_eval < 3:
                tqdm.write("#### EVAL")
                tqdm.write("query: {}".format(q_txt))
                tqdm.write("prediction: {}".format(sentence_cutoff))
                tqdm.write("loss: {} reward: {}".format(loss,reward))
            i_eval+=1
            if i_eval > 100:
                break

        test_loss = test_loss/i_eval
        test_reward = test_reward / i_eval
        pbar.close()
        w.close()
        epoch_summary = {
            "epoch":self.epoch,
            "test_loss":test_loss,
            "test_reward": test_reward,
            "train_loss":train_loss,
            "base_reward":base_reward,
        }
        self.get_result_writer().write(epoch_summary)

        if self.min_test_loss is None or test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.model.save_best(self.epoch)

        self.model.save_latest(self.epoch)
        tqdm.write("\n\n####################")
        tqdm.write("model saved!")
        tqdm.write("epoch: {} train_loss: {} base_reward: {}  test_loss: {}  test_reward: {}".format(self.epoch,train_loss,base_reward,test_loss,test_reward))
        tqdm.write("####################\n\n")

    def train(self):

        while self.epoch < self.max_epoch:
            pbar = tqdm(total=int(self.data_points/self.batch_size), desc="training batches for epoch {}".format(self.epoch))
            train_data = RLSquadDataloader(base_path=self.base_path, batch_size=self.batch_size, max_length=self.max_seq_len, eval=False)
            train_iter = 0
            for q_emb, relevant_documents,q_mask in iter(train_data):
                x_tensor = torch.tensor((np.transpose(q_emb, (1, 0, 2))),device=self.device).type(torch.float64)
                q_mask = torch.tensor(q_mask, device=self.device).type(torch.float64)

                train_iter += 1
                predictions = self.model.train_predict(x_tensor,q_mask)

                predicted_sentence = construct_sentence(predictions)
                sentence_cutoff = prune(predicted_sentence)
                search_results = self.search_engine.search(sentence_cutoff)
                reward,base_reward,normarlized_reward = self.model.reward_function(search_results,relevant_documents)

                loss = self.model.update_policy(normarlized_reward)
                normarlized_reward = np.mean(normarlized_reward)
                self.write_reward(reward,base_reward,normarlized_reward)
                self.model.reward_function.base_line.update(reward)

                self.calc_loss(loss)

                pbar.set_description("training batches for epoch {} with training loss: {:.3f}, normalized reward: {:.6f} base reward {:.6f}".format(self.epoch, self.loss,normarlized_reward,base_reward))
                pbar.update()
                if train_iter % int(((self.data_points/self.batch_size)/10)) == 0:
                    [tqdm.write(sentence) for sentence in predicted_sentence[:4]]
                    pbar.close()
                    self.evaluate(self.loss,base_reward)
                    pbar = tqdm(total=int(self.data_points/self.batch_size),
                                desc="training batches for epoch {}".format(self.epoch))
                    pbar.update(train_iter)

            pbar.close()

            self.evaluate(self.loss,base_reward)
            self.epoch += 1

    def write_reward(self,reward,base_reward,normarlized_reward):
        w = self.get_norm_reward()
        w.write({"reward":reward,"base_reward":base_reward,"normarlized_reward":normarlized_reward})
        w.close()

    def get_result_writer(self):

        result_file_path = self.exp_path+"/results.jsonl"

        if os.path.isfile(result_file_path):
            mode = 'a'
        else:
            mode = 'w'
        return jsonlines.open(result_file_path, mode)

    def get_norm_reward(self):

        result_file_path = self.exp_path+"/norm_reward.jsonl"

        if os.path.isfile(result_file_path):
            mode = 'a'
        else:
            mode = 'w'
        return jsonlines.open(result_file_path, mode)

    def get_test_reward(self):

        result_file_path = self.exp_path+"/test_reward.jsonl"

        if os.path.isfile(result_file_path):
            mode = 'a'
        else:
            mode = 'w'
        return jsonlines.open(result_file_path, mode)

    def make_dir(self):
        os.makedirs(self.exp_path + "/latest", exist_ok=True)
        os.makedirs(self.exp_path + "/best", exist_ok=True)



    def calc_loss(self,l):
        if self.loss is None:
            self.loss = l
        else:
            g = 0.001
            self.loss = g * l + (1 - g) * self.loss