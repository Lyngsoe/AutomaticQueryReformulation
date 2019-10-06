import json
class LaserVocab:

    def __init__(self,path="/home/jonas/.local/lib/python3.6/site-packages/laserembeddings/data/93langs.fvocab"):
        self.path = path
        self.lookup = {}

        for i,line in enumerate(open(path,'r')):
            symbol = line.split(" ")[0]
            self.lookup.update({i:symbol})


        json.dump(self.lookup,open("/home/jonas/Documents/master/models/lookup.json",'w'))
        file = open("/home/jonas/Documents/master/models/lookup.txt",'w')
        for k,v in self.lookup.items():
            file.write(str(k)+" "+v+"\n")


    def __call__(self, index):
        return self.lookup.get(index)


if __name__ == '__main__':
    vocab = LaserVocab()

    print(vocab(2))