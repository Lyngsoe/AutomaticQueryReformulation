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

    def on_return(self, q_batch, y_batch, queries):

        if self.eval:
            return q_batch, y_batch, queries, queries

        return q_batch, y_batch

    def __iter__(self):
        return self
