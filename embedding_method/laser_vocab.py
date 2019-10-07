import json
class LaserVocab:

    def __init__(self,drive_path,language,debug=False,Laser_vocab_path="/home/jonas/.local/lib/python3.6/site-packages/laserembeddings/data/93langs.fvocab"):
        self.Laser_vocab_path = Laser_vocab_path
        self.base_path = drive_path + "raffle_wiki/{}/debug/".format(language) if debug else drive_path + "raffle_wiki/{}/".format(language)
        self.bpe2id = {"<PAD>": 0}
        self.id2bpe = {0: "<PAD>"}

        for i,line in enumerate(open(Laser_vocab_path,'r')):
            symbol = line.split(" ")[0]
            self.id2bpe.update({i+1:symbol})
            self.bpe2id.update({symbol: i+1})


        json.dump(self.id2bpe,open(self.base_path+"id2bpe.json",'w'))
        json.dump(self.bpe2id, open(self.base_path + "bpe2id.json", 'w'))
