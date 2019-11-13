from dataset.bertify import construct_sentence
from training.dataloaders.squad_dataloader_2 import SquadDataloader2
from tqdm import tqdm
import torch
import os
import jsonlines
import time

class Trainer:
    def __init__(self,model,base_path,batch_size=8,epoch=0,max_epoch=50,device="gpu",max_seq_len=300):
        self.model = model
        self.base_path = base_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.device = device
        self.max_seq_len = max_seq_len
        self.min_test_loss = None
        self.exp_path = model.save_path + model.exp_name
        self.make_dir()

    def evaluate(self,train_loss):
        eval_data = SquadDataloader2(base_path=self.base_path,batch_size=1,max_length=self.max_seq_len,eval=True)
        i_eval = 0
        test_loss = 0
        pbar = tqdm(total=5928, desc="evaluating batches for epoch {}".format(self.epoch))
        for eval_x, eval_y,queries,targets in iter(eval_data):
            eval_x = torch.tensor(eval_x, device=self.device).type(torch.float64).view(-1, eval_x.shape[0], eval_x.shape[2])
            eval_y = torch.tensor(eval_y, device=self.device).type(torch.float64).view(-1, eval_y.shape[0])

            loss,predictions = self.model.predict(eval_x,eval_y)

            test_loss+=loss
            pbar.update()
            if i_eval < 6:
                sentences = construct_sentence(predictions)
                tqdm.write("#### EVAL")
                tqdm.write("query: {}".format(queries))
                tqdm.write("prediction: {}".format(sentences))
                tqdm.write("target: {} loss: {}".format(targets,loss))
            i_eval+=1
            if i_eval > 6:
                break

        test_loss=test_loss/i_eval
        pbar.close()

        epoch_summary = {
            "epoch":self.epoch,
            "test_loss":test_loss,
            "train_loss":train_loss
        }
        self.get_result_writer().write(epoch_summary)

        if self.min_test_loss is None or test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.model.save_best(self.epoch)

        self.model.save_latest(self.epoch)
        tqdm.write("model saved!")

        tqdm.write("epoch: {} train_loss: {}  test_loss: {}".format(self.epoch,train_loss,test_loss))


    def train(self):

        while self.epoch < self.max_epoch:
            pbar = tqdm(total=int(86821 / self.batch_size), desc="training batches for epoch {}".format(self.epoch))
            train_data = SquadDataloader2(base_path=self.base_path, batch_size=self.batch_size, max_length=self.max_seq_len, eval=False)
            train_iter = 0
            mbl = 0
            total_train_time = 0
            total_data_load_time = 0
            start_data_load = time.time()
            for x, y in iter(train_data):
                x_tensor = torch.tensor(x, device=self.device).type(torch.double).view(-1, x.shape[0], x.shape[2])
                y_tensor = torch.tensor(y, device=self.device).type(torch.double).view(-1, y.shape[0])
                total_data_load_time+= time.time() - start_data_load
                train_iter += 1
                start_train = time.time()
                batch_loss = self.model.train(x_tensor, y_tensor)
                total_train_time += time.time() - start_train
                mbl += batch_loss
                pbar.set_description("training batches for epoch {} with training loss: {:.2f} train: {:.2f} load: {:.2f}".format(self.epoch, mbl/train_iter ,total_train_time / train_iter, total_data_load_time / train_iter))
                pbar.update()
                start_data_load = time.time()
                if train_iter % 1000 == 0:
                    #train_loss = mbl / train_iter
                    #self.evaluate(train_loss)
                    break

            pbar.close()

            train_loss = mbl / train_iter
            self.evaluate(train_loss)
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