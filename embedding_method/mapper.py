import json
import os
import mmap
import numpy

class MemFetcher:
    def __init__(self,lookup_path,data_path):
        self.lookup_path = lookup_path
        self.data_path = data_path
        self.lookup = None
        self.data = None

        self._load()

    def _load(self):
        self.lookup = json.load(open(self.lookup_path,'r'))
        mfd = os.open(self.data_path, os.O_RDONLY)
        self.data = mmap.mmap(mfd,0,access=mmap.PROT_READ)

    def __call__(self,lookup_id):
        fp = self.lookup[lookup_id]
        self.data.seek(fp)
        json_data = json.loads(self.data.readline())
        return numpy.array(json_data["emb"])