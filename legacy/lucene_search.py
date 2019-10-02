from lucene_class import LUCENE
from car import CAR

car = CAR(debug=True)

paras = car.get_paragraphs()

luc = LUCENE()

results = luc.search("crypt")

print(len(results))
print(results)