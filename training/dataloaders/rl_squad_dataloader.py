import orjson
from json.decoder import JSONDecodeError
import numpy as np

class RLSquadDataloader:
    def __init__(self, base_path, max_length, eval, batch_size):
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
        base_rewards = []

        for i in range(self.batch_size):
            try:
                txt = self.reader.readline()
                qa = orjson.loads(txt)
            except (EOFError, JSONDecodeError):
                if len(input_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(input_batch, output_batch, sources,base_rewards)

            # X
            inp = qa["question_emb"]
            source = qa["question"]
            base_rewards.append(qa["base_reward"])
            input_batch.append(inp)
            sources.append(source)

            # Y
            out = qa["paragraphs"]
            output_batch.append(out)

        return self.on_return(input_batch, output_batch, sources,base_rewards)

    def on_return(self, q_batch, y_batch, queries,base_rewards):
        max_seq_len = 0

        for q in q_batch:
            seq_len = len(q)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        new_q_batch = []
        for q in q_batch:
            new_q_batch.append(self.pad_x(q, max_seq_len))

        q_batch = np.stack(new_q_batch)

        if self.eval:
            return q_batch, y_batch, queries, queries,base_rewards

        return q_batch, y_batch,base_rewards

    def __iter__(self):
        return self

    def pad_x(self, x, max_len):
        while len(x) < max_len:
            w = np.zeros(768)
            x.append(w)
        return x[:self.max_length]