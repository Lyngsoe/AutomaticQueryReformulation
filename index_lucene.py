from lucene_class import LUCENE
from car import CAR


debug = False

car = CAR(debug=debug)

paras = car.get_paragraphs()

luc = LUCENE(debug=debug)


luc.index(paras)