from car import CAR
from tqdm import tqdm
from elastic_search import ELSearch
from models.identity_model import IdentityModel
from models.wordnet_model import WordnetModel
import numpy as np
import json
import datetime
import utils
from evaluator import Evaluator

model = WordnetModel()

debug = True

search_engine = ELSearch()
dataset = CAR(debug=debug)

output_name = "results/"+model.name+"/"+datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")+".json"
utils.create_exp_dir(model.name)

paras = dataset.get_paragraphs()
queries = dataset.get_queries()

search_engine.index_seq(paras)

queries = search_engine.search(queries)

json.dump(queries,open(output_name,'w'))

Evaluator(output_name,dataset)