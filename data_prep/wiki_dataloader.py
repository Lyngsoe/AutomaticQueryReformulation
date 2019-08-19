import os
import jsonlines
from tqdm import tqdm

class WikiDataloader:
    def __init__(self,base_path = "/home/jonas/data/wiki/parsed_wiki",max_pages=-1):
        self.base_path = base_path
        self.data_paths = self.get_data_paths()
        self.current_data_file = 0
        self.pages_read = 0
        self.max_pages = max_pages
        self.pbar = tqdm(total=len(self.data_paths),desc="wiki files")
        self.json_iter = self.next_iter()

    def get_data_paths(self):
        file_paths = []
        for folder in os.listdir(self.base_path):
            for file in os.listdir(os.path.join(self.base_path, folder)):
                file_paths.append(os.path.join(self.base_path, folder, file))

        return file_paths

    def next_iter(self):
        if self.current_data_file < len(self.data_paths):
            iter = jsonlines.open(self.data_paths[self.current_data_file])
            self.current_data_file+=1
            self.pbar.update(1)
            return iter
        else:
            return None

    def get_next(self):
        if self.max_pages != -1 and self.pages_read > self.max_pages:
            return None

        try:
            data_point = self.json_iter.read()
        except EOFError:
            self.json_iter = self.next_iter()
            if self.json_iter is None:
                return None
            data_point = self.json_iter.read()

        self.pages_read+=1
        return data_point

    def get_next_batch(self):

        if self.max_pages != -1 and self.pages_read > self.max_pages:
            return None

        pages = []

        if self.json_iter is None:
            return None

        for p in self.json_iter:
            pages.append(p)
            self.pages_read+=1

            if self.max_pages != -1 and self.pages_read > self.max_pages:
                break

        self.json_iter = self.next_iter()
        return pages

