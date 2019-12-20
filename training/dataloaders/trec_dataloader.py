import orjson
import numpy as np
from json.decoder import JSONDecodeError


class TRECDataloder:
    def __init__(self, base_path, max_length, eval, batch_size):
        if eval:
            self.fold = 4
            file_name = "fold-{}.jsonl".format(self.fold)
        else:
            self.fold = 0
            file_name = "fold-{}.jsonl".format(self.fold)


        self.reader = open(base_path + file_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.eval = eval

    def __next__(self):
        input_batch = []
        output_batch = []
        sources = []

        for i in range(self.batch_size):
            try:
                txt = self.reader.readline()
                qa = orjson.loads(txt)
            except (EOFError, JSONDecodeError):
                if len(input_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(input_batch, output_batch, sources)

            # X
            inp = qa["question_emb"]
            source = qa["txt"]
            input_batch.append(inp)
            sources.append(source)

            # Y
            out = qa["paragraphs"]
            output_batch.append(out)

        return self.on_return(input_batch, output_batch, sources)

    def on_return(self, q_embedding, relevant_documents, q_txt):
        max_seq_len = 0

        for q in q_embedding:
            seq_len = len(q)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        new_q_batch = []
        for q in q_embedding:
            new_q_batch.append(self.pad_x(q, max_seq_len))

        q_batch = np.stack(new_q_batch)

        mask_q = []
        for q in q_batch:
            seq_len = len(q)
            mask_q.append(self.create_mask(seq_len, max_seq_len))

        q_mask = np.stack(mask_q)

        if self.eval:
            return q_batch, relevant_documents, q_txt, q_mask

        return q_batch, relevant_documents, q_mask

    def __iter__(self):
        return self

    def pad_x(self, x, max_len):
        while len(x) < max_len:
            w = np.zeros(768)
            x.append(w)
        return x[:self.max_length]

    def create_mask(self, seq_len, max_seq_len):

        m = [0 for i in range(seq_len)]
        m2 = [1 for i in range(max_seq_len - seq_len)]
        m.extend(m2)
        # print(m)
        # print(len(m)," ", max_seq_len)
        return m[:self.max_length]