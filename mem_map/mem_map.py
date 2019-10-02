import jsonlines
import json
class MemoryMapLookup:
    def __init__(self,data_path,output_path,id_key="id"):
        self.data_path = data_path
        self.output_path = output_path
        self.id_key = id_key
        self.lookup = {}

        self.create_mem_lookup()

        json.dump(self.lookup,open(output_path,'w'))

    def create_mem_lookup(self):

        reader = open(self.data_path,'r')
        while True:
            pos = reader.tell()

            data = reader.readline()

            if data == "":
                break

            data = json.loads(data)
            self.lookup.update({data[self.id_key]:pos})





if __name__ == '__main__':

    base_path = "/home/jonas/data/raffle_wiki/da/debug/"
    MemoryMapLookup(base_path+"paragraphs.jsonl",base_path+"paralookup.json")