from lucene_class import LUCENE
from car import CAR
from tqdm import tqdm

debug = False

car = CAR(debug=debug)

paras = car.get_paragraphs()

luc = LUCENE(debug=debug)

for txt,id in tqdm(paras,desc="indexing paragraphs"):
    luc.index(txt,id)