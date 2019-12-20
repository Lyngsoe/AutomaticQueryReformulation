import orjson
import numpy as np
from json.decoder import JSONDecodeError
import copy

class SquadDataloaderQ2Q:
    def __init__(self,base_path,max_length,eval,batch_size):
        if eval:
            file_name = "qas_eval.jsonl"
        else:
            file_name = "qas.jsonl"

        self.reader = open(base_path + file_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.eval = eval

    def __next__(self):
        input_batch = []
        output_batch = []
        sources = []
        targets = []
        y_emb = []


        for i in range(self.batch_size):
            try:
                txt = self.reader.readline()
                qa = orjson.loads(txt)
            except (EOFError,JSONDecodeError):
                if len(input_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(input_batch,output_batch,sources,targets,copy.deepcopy(input_batch))



            # X
            inp = qa["question_emb"]
            source = qa["question"]
            input_batch.append(inp)
            sources.append(source)

            #Y
            out = qa["question_token_ids"]
            y_emb.append(qa["question_emb"])
            target = qa["question"]
            output_batch.append(out)
            targets.append(target)

        return self.on_return(input_batch,output_batch,sources,targets,copy.deepcopy(input_batch))

    def on_return(self,q_batch,y_batch,queries,targets,y_emb):
        max_seq_len_y = 0
        max_seq_len_q = 0

        for q in q_batch:
            seq_len = len(q)
            if seq_len > max_seq_len_q:
                max_seq_len_q = seq_len

        for y in y_batch:
            seq_len = len(y)
            if seq_len > max_seq_len_y:
                max_seq_len_y = seq_len

        mask_q = []
        for q in q_batch:
            seq_len = len(q)
            mask_q.append(self.create_mask(seq_len,max_seq_len_q))

        mask_y = []
        for y in y_batch:
            seq_len = len(y)
            mask_y.append(self.create_mask(seq_len, max_seq_len_y))

        new_q_batch = []
        for q in q_batch:
            new_q_batch.append(self.pad_x(self.add_start_token_x(q),max_seq_len_q))

        new_y_batch = []
        for y in y_batch:
            new_y_batch.append(self.pad_y(y,max_seq_len_y))

        new_y_emb_batch = []
        for y in y_emb:
            new_y_emb_batch.append(self.pad_x(self.add_start_token_x(y),max_seq_len_y))


        y_batch = np.stack(new_y_batch)
        q_batch = np.stack(new_q_batch)
        y_emb = np.stack(new_y_emb_batch)
        q_mask = np.stack(mask_q)
        y_mask = np.stack(mask_y)

        #print("y_batch",y_batch.shape)
        #print("y_mask", y_mask.shape)
        #print("q_batch", q_batch.shape)
        #print("q_mask", q_mask.shape)

        if self.eval:
            return q_batch, y_batch,queries,targets,q_mask,y_mask,y_emb

        return q_batch,y_batch,q_mask,y_mask,y_emb

    def pad_x(self,x,max_len):
        while len(x) < max_len:
            w = np.zeros(768)
            x.append(w)
        return x[:self.max_length]

    def pad_y(self,y,max_len):
        while len(y) < max_len:
            y.append(1)
        return y[:self.max_length]

    def add_start_token_x(self,x):
        sos = np.empty(768)
        sos.fill(0.1)
        x[0] = sos
        return x
    def add_start_token_y(self, y):
        y.insert(0, 2)
        return y

    def create_mask(self,seq_len,max_seq_len):

        m = [0 for i in range(seq_len)]
        m2 = [1 for i in range(max_seq_len-seq_len)]
        m.extend(m2)
        #print(m)
        #print(len(m)," ", max_seq_len)
        return m[:self.max_length]

    def __iter__(self):
        return self