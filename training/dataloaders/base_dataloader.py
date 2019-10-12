import jsonlines
import numpy as np

class BaseDataloader:
    def __init__(self,base_path,input_loader,output_loader,query_file_name="0-queries.jsonl",batch_size = 8,debug=False,eval=False):
        self.base_path = base_path
        self.batch_size = batch_size
        self.eval=eval
        self.reader = jsonlines.open(self.base_path + query_file_name)
        self.input_loader = input_loader
        self.output_loader = output_loader
    def __iter__(self):
        return self

    def __next__(self):
        input_batch = []
        output_batch = []
        sources = []
        targets = []


        for i in range(self.batch_size):
            try:
                query = self.reader.read()
            except EOFError:
                if len(input_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(input_batch,output_batch,sources,targets)

            # X
            inp,source = self.input_loader.get_embedding(query)
            input_batch.append(inp)
            sources.append(source)

            #Y
            out,target = self.output_loader.get_embedding(query)
            output_batch.append(out)
            targets.append(target)

        return self.on_return(input_batch,output_batch,sources,targets)

    def on_return(self,q_batch,y_batch,queries,targets):
        max_seq_len = 0

        for q in q_batch:
            seq_len = len(q)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        for y in y_batch:
            seq_len = len(y)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        new_q_batch = []
        for q in q_batch:
            new_q_batch.append(self.input_loader.padding(q,max_seq_len+1))

        new_y_batch = []
        for y in y_batch:
            new_y_batch.append(self.output_loader.padding(y,max_seq_len+1))


        y_batch = np.stack(new_y_batch)
        q_batch = np.stack(new_q_batch)

        if self.eval:
            return q_batch, y_batch,queries,targets

        return q_batch,y_batch