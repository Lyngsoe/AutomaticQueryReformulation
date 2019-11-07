from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import json

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def _prep(self,text):
        js = json.loads(text)

        return js

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.file_path)
        mapped_itr = map(self._prep, file_itr)
        return mapped_itr



if __name__ == '__main__':
    dataset = MyIterableDataset('/media/jonas/archive/master/data/squad/qas_eval.jsonl')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    for X in dataloader:
        print(type(X))
        print(len(X))  # 64
        print(type(X[0]))  # 64
