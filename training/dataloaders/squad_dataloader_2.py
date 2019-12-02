import orjson
import numpy as np
from json.decoder import JSONDecodeError

class SquadDataloader2:
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
        y_toks = []


        for i in range(self.batch_size):
            try:
                txt = self.reader.readline()
                qa = orjson.loads(txt)
            except (EOFError,JSONDecodeError):
                if len(input_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(input_batch,output_batch,sources,targets,y_toks)



            # X
            inp = qa["context_emb"]
            source = qa["context"]
            input_batch.append(inp)
            sources.append(source)

            #Y
            out = qa["question_emb"]
            y_toks.append(qa["question_token_ids"])
            target = qa["question"]
            output_batch.append(out)
            targets.append(target)

        return self.on_return(input_batch,output_batch,sources,targets,y_toks)

    def on_return(self,q_batch,y_batch,queries,targets,y_toks):
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

        new_q_batch = []
        for q in q_batch:
            new_q_batch.append(self.pad_x(q,max_seq_len_q))

        new_y_batch = []
        for y in y_batch:
            new_y_batch.append(self.pad_x(y,max_seq_len_y))

        new_y_tok_batch = []
        for y in y_toks:
            new_y_tok_batch.append(self.pad_y(y,max_seq_len_y))


        y_batch = np.stack(new_y_batch)
        q_batch = np.stack(new_q_batch)
        y_toks = np.stack(new_y_tok_batch)

        #print("y_batch",y_batch.shape)
        #print("y_mask", y_mask.shape)
        #print("q_batch", q_batch.shape)
        #print("q_mask", q_mask.shape)

        if self.eval:
            return q_batch, y_batch,queries,targets,y_toks

        return q_batch,y_batch,y_toks

    def pad_x(self,x,max_len):
        while len(x) < max_len:
            w = np.zeros(768)
            x.append(w)
        return x[:self.max_length]

    def pad_y(self,y,max_len):
        while len(y) < max_len:
            y.append(1)
        return y[:self.max_length]


    def __iter__(self):
        return self