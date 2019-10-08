import siphash
import json
from tqdm import tqdm
import jsonlines
import random

class FoldDivider:
    def __init__(self,drive_path,language,debug=False):
        self.drive_path = drive_path
        self.language = language
        self.debug = debug

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.folds = 5
        self.info = json.load(open(self.base_path+"wiki_info.json",'r'))
        self.wiki2fold = {}
        self.hashtable = {'0':[],
                         '1':[],
                         '2':[],
                         '3':[],
                         '4':[]}
        self.query_hash = {'0':[],
                         '1':[],
                         '2':[],
                         '3':[],
                         '4':[]}
        self.query_folds = [0,0,0,0,0]

        self.create_hash_table()
        self.divide_queries()

        i = 0
        for k,v in self.hashtable.items():
            self.info.update({
                "fold-{}".format(i): {"wiki":len(v),"queries":self.query_folds[i]}
            })
            i+=1

        json.dump(self.info,open(self.base_path+"wiki_info.json",'w'))


    def create_hash_table(self):

        wikis = json.load(open(self.base_path+"wiki.json",'r'))

        for wiki_id,wiki in tqdm(wikis.items(),desc="Creating hashtable bins"):
            try:
                title = wiki["title"]
            except:
                tqdm.write("{}".format(wiki))
                continue

            b = title.encode()
            key = b"FEDCBA9876543210"
            hashed_title = siphash.SipHash_2_4(key,b).hash()
            self.add_to_hashtable(hashed_title,wiki_id)

        for k,v in self.hashtable.items():
            tqdm.write("bin: {}, #wikis: {}".format(k,len(v)))

    def divide_queries(self):

        writers = self.get_writers()
        pbar = tqdm(total=self.info["queries"],desc="saving queries to folds")

        queries = [[],[],[],[],[]]

        for query in jsonlines.open(self.base_path+"queries.jsonl",'r'):
            wiki_id = query["wiki_id"]
            fold = int(self.wiki2fold.get(wiki_id))
            queries[fold].append(query)
            #writers[fold].write(query)
            self.query_folds[fold] += 1
            pbar.update()

        pbar.close()

        for fold in range(4):
            qs = queries[fold]
            random.shuffle(qs)
            for q in qs:
                writers[fold].write(q)

        for i, v in enumerate(self.query_folds):
            tqdm.write("bin: {}, #queries: {}".format(i, v))

    def add_to_hashtable(self,key,wiki_id):
        hash_key = key % self.folds
        wikis = self.hashtable.get(str(hash_key))
        if wikis is None:
            raise Exception("bin does not exist {}".format(hash_key))

        wikis.append(wiki_id)

        self.hashtable.update({str(hash_key):wikis})
        self.wiki2fold.update({wiki_id:hash_key})

    def get_writers(self):
        writers = []
        for i in range(self.folds):
            w = jsonlines.open(self.base_path+"{}-queries.jsonl".format(i),'w')
            writers.append(w)

        return writers

if __name__ == '__main__':
    drive_path = "/media/jonas/archive/master/data/"
    debug = True
    language = "da"

    FoldDivider(drive_path=drive_path,language=language,debug=debug)