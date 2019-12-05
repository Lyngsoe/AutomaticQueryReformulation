from dataset.bertify import construct_sentence,prune
from training.dataloaders.rl_squad_dataloader import RLSquadDataloader
from tqdm import tqdm
import torch
import os
import jsonlines
import time
import numpy as np

class Trainer:
    def __init__(self,model,base_path,search_engine,batch_size=1,epoch=0,max_epoch=50,device="gpu",max_seq_len=300):
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
        self.make_dir()

    def evaluate(self,train_loss,train_reward):
        eval_data = RLSquadDataloader(base_path=self.base_path,batch_size=1,max_length=self.max_seq_len,eval=True)
        i_eval = 0
        test_loss = 0
        test_reward = 0
        pbar = tqdm(total=5928, desc="evaluating batches for epoch {}".format(self.epoch))
        for q_emb, relevant_documents,q_txt,base_reward,q_mask in iter(eval_data):

            q_emb = torch.tensor(q_emb, device=self.device).type(torch.float64)
            q_emb = q_emb.reshape(q_emb.size(1), 1, q_emb.size(2))
            q_mask = torch.tensor(q_mask, device=self.device).type(torch.float64)

            predictions = self.model.predict(q_emb,q_mask)
            #print(predictions.shape)
            predicted_sentence = construct_sentence(predictions)
            sentence_cutoff = prune(predicted_sentence)
            search_results = self.search_engine.search(sentence_cutoff)
            reward,mean_reward = self.model.reward_function(search_results, relevant_documents, base_reward,q_emb.size(0))
            loss = self.model.update_policy(reward,update=False)

            test_loss+=loss
            test_reward+=mean_reward
            pbar.update()
            if i_eval < 6:
                tqdm.write("#### EVAL")
                tqdm.write("query: {}".format(q_txt))
                tqdm.write("prediction: {}".format(predicted_sentence))
                tqdm.write("loss: {} reward: {}".format(loss,mean_reward))
            i_eval+=1
            if i_eval > 10:
                break

        test_loss = test_loss/i_eval
        test_reward = test_reward / i_eval
        pbar.close()

        epoch_summary = {
            "epoch":self.epoch,
            "test_loss":test_loss,
            "test_reward": test_reward,
            "train_loss":train_loss,
            "train_reward":train_reward
        }
        self.get_result_writer().write(epoch_summary)

        if self.min_test_loss is None or test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.model.save_best(self.epoch)

        self.model.save_latest(self.epoch)
        tqdm.write("model saved!")

        tqdm.write("epoch: {} train_loss: {} train_reward: {}  test_loss: {}  test_reward: {}".format(self.epoch,train_loss,train_reward,test_loss,test_reward))


    def train(self):

        while self.epoch < self.max_epoch:
            pbar = tqdm(total=int(10031 / self.batch_size), desc="training batches for epoch {}".format(self.epoch))
            train_data = RLSquadDataloader(base_path=self.base_path, batch_size=self.batch_size, max_length=self.max_seq_len, eval=False)
            train_iter = 0
            total_rewards = 0
            total_loss = 0
            total_train_time = 0
            total_data_load_time = 0
            start_data_load = time.time()
            for q_emb, relevant_documents,base_reward,q_mask in iter(train_data):
                x_tensor = torch.tensor(q_emb, device=self.device).type(torch.float64)
                x_tensor = x_tensor.reshape(-1,x_tensor.size(0),x_tensor.size(2))
                q_mask = torch.tensor(q_mask, device=self.device).type(torch.float64)
                total_data_load_time+= time.time() - start_data_load
                train_iter += 1
                start_train = time.time()
                predictions = self.model.train_predict(x_tensor,q_mask)
                total_train_time += time.time() - start_train

                start_data_load = time.time()
                predicted_sentence = construct_sentence(predictions)
                sentence_cutoff = prune(predicted_sentence)
                #print(sentence_cutoff)
                search_results = self.search_engine.search(sentence_cutoff)
                reward,mean_reward = self.model.reward_function(search_results,relevant_documents,base_reward,x_tensor.size(0))
                total_data_load_time += time.time() - start_data_load

                start_train = time.time()
                total_loss += self.model.update_policy(reward)
                total_train_time += time.time() - start_train

                total_rewards += mean_reward
                pbar.set_description("training batches for epoch {} with training loss: {:.3f}, reward: {:.6f} train: {:.2f} load: {:.2f}".format(self.epoch, total_loss/train_iter,total_rewards/train_iter ,total_train_time / train_iter, total_data_load_time / train_iter))
                pbar.update()
                start_data_load = time.time()
                if train_iter % 100 == 0:
                    [tqdm.write(sentence) for sentence in predicted_sentence[:4]]
                if train_iter % 1000 == 0:
                    #pbar.close()
                    train_loss = total_loss / train_iter
                    train_reward = total_rewards / train_iter
                    self.evaluate(train_loss,train_reward)
                    #tqdm.write(predicted_sentence[0])
                    #break

            pbar.close()

            train_loss = total_loss / train_iter
            train_reward = total_rewards / train_iter
            self.evaluate(train_loss,train_reward)
            self.epoch += 1

    def get_result_writer(self):

        result_file_path = self.exp_path+"/results.jsonl"

        if os.path.isfile(result_file_path):
            mode = 'a'
        else:
            mode = 'w'
        return jsonlines.open(result_file_path, mode)

    def make_dir(self):
        os.makedirs(self.exp_path + "/latest", exist_ok=True)
        os.makedirs(self.exp_path + "/best", exist_ok=True)
