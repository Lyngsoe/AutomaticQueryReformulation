from dataset.bertify import construct_sentence
from training.dataloaders.squad_dataloader_q2q import SquadDataloaderQ2Q
from tqdm import tqdm
import torch
import os
import jsonlines
import time
import json
import numpy as np

class TrainerQ2Q:
    def __init__(self, model, base_path, batch_size=8, epoch=0, max_epoch=50, device="gpu", max_seq_len=300,
                 specs=None):
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
        if specs is not None:
            specs.update({"model_name:": model.exp_name})
            json.dump(specs, open(self.exp_path + "/specs.json", 'w'))

    def evaluate(self, train_loss, train_iter):
        eval_data = SquadDataloaderQ2Q(base_path=self.base_path, batch_size=1, max_length=self.max_seq_len, eval=True)
        i_eval = 0
        test_loss = 0
        pbar = tqdm(total=5928, desc="evaluating batches for epoch {}".format(self.epoch))
        for eval_x, eval_y, queries, targets, x_mask, y_mask, y_emb in iter(eval_data):
            eval_x = torch.tensor(np.transpose(eval_x, (1, 0, 2)), device=self.device).type(torch.float64)
            y_emb = torch.tensor(np.transpose(y_emb, (1, 0, 2)), device=self.device).type(torch.double)
            eval_y = torch.tensor(eval_y, device=self.device).type(torch.long)
            x_mask = torch.tensor(x_mask, device=self.device).type(torch.float64)
            y_mask = torch.tensor(y_mask, device=self.device).type(torch.float64)
            loss, predictions = self.model.predict(eval_x, eval_y, x_mask, y_mask, y_emb)
            test_loss += loss
            pbar.update()
            if i_eval < 3:
                sentences = construct_sentence(predictions)
                tqdm.write("\n")
                # tqdm.write("query: {}".format(queries))
                tqdm.write("prediction: {}".format(sentences))
                tqdm.write("target: {} loss: {}".format(targets, loss))
                tqdm.write("\n")
            i_eval += 1
            if i_eval > 100:
                break

        test_loss = test_loss / i_eval
        pbar.close()

        epoch_summary = {
            "epoch": self.epoch,
            "test_loss": test_loss,
            "train_loss": train_loss,
            "train_iter": train_iter
        }
        self.get_result_writer().write(epoch_summary)

        if self.min_test_loss is None or test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.model.save_best(self.epoch)

        self.model.save_latest(self.epoch)
        tqdm.write("\n\n####################")
        tqdm.write("model saved! {}".format(self.model.exp_name))
        tqdm.write("epoch: {} train_loss: {}  test_loss: {}".format(self.epoch, train_loss, test_loss))
        tqdm.write("####################\n\n")

    def train(self):

        while self.epoch < self.max_epoch:
            pbar = tqdm(total=int(86821 / self.batch_size), desc="training batches for epoch {}".format(self.epoch))
            train_data = SquadDataloaderQ2Q(base_path=self.base_path, batch_size=self.batch_size,
                                          max_length=self.max_seq_len, eval=False)
            train_iter = 0
            temp_train_iter = 0
            mbl = 0
            total_train_time = 0
            total_data_load_time = 0
            start_data_load = time.time()
            for x, y, x_mask, y_mask, y_emb in iter(train_data):
                x_tensor = torch.tensor(np.transpose(x, (1, 0, 2)), device=self.device).type(torch.double)
                y_emb = torch.tensor(np.transpose(y_emb, (1, 0, 2)), device=self.device).type(torch.double)
                y_tensor = torch.tensor(y, device=self.device).type(torch.long)
                # print(x_mask.shape)
                x_mask = torch.tensor(x_mask, device=self.device).type(torch.float64)
                y_mask = torch.tensor(y_mask, device=self.device).type(torch.float64)

                # print(x_mask.size())
                total_data_load_time += time.time() - start_data_load
                train_iter += 1
                temp_train_iter += 1
                start_train = time.time()
                batch_loss, predictions = self.model.train(x_tensor, y_tensor, x_mask, y_mask, y_emb)
                total_train_time += time.time() - start_train
                mbl += batch_loss
                pbar.set_description(
                    "training batches for epoch {} with training loss: {:.5f} train: {:.2f} load: {:.2f}".format(
                        self.epoch, mbl / temp_train_iter, total_train_time / train_iter,
                        total_data_load_time / train_iter))
                pbar.update()
                start_data_load = time.time()
                if train_iter % int(((86821 / self.batch_size) / 10)) == 0:
                    pbar.close()
                    sentences = construct_sentence(predictions)
                    tqdm.write("\nTRAIN:\n")
                    for s in sentences[:4]:
                        tqdm.write("prediction: {}".format(s))
                    train_loss = mbl / temp_train_iter
                    temp_train_iter = 0
                    mbl = 0
                    self.evaluate(train_loss, train_iter)
                    pbar = tqdm(total=int(86821 / self.batch_size),
                                desc="training batches for epoch {}".format(self.epoch))
                    pbar.update(train_iter)
                    # break

            pbar.close()
            self.epoch += 1

    def get_result_writer(self):

        result_file_path = self.exp_path + "/results.jsonl"

        if os.path.isfile(result_file_path):
            mode = 'a'
        else:
            mode = 'w'
        return jsonlines.open(result_file_path, mode)

    def make_dir(self):
        os.makedirs(self.exp_path + "/latest", exist_ok=True)
        os.makedirs(self.exp_path + "/best", exist_ok=True)
