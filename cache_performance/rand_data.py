
import random
import jsonlines
import json


def get_random_dict(size):
    data = {}

    for i in range(size):
        data.update({i:random.randint(0,size)})
    return data

def create_mmap(data):

    writer = jsonlines.open("data.jsonl",'w')
    lookup = {}
    for k,v in data.items():
        fp = writer._fp.tell()
        lookup.update({k:fp})

        writer.write({k:v})

    json.dump(lookup,open("lookup.json",'w'))
