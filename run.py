from car import CAR
from tqdm import tqdm
from lucene_class import LUCENE
from identity_model import IdentityModel
import numpy as np
import json
import multiprocessing as mp


def search(query,search_engine,model):
        reformed_query = model.reform(query["txt"])
        query.update({"reformed_txt":reformed_query})

        results = search_engine.search(reformed_query)

        query.update({"results":results})

        return query

debug = False

search_engine = LUCENE(debug=debug,ram=True)
dataset = CAR(debug=debug)
model = IdentityModel()

paras = dataset.get_paragraphs()
search_engine.index(paras)

queries = dataset.get_queries()
formated_results = []

pool = mp.Pool(mp.cpu_count())
new_q = [pool.apply(search, args=(q,search_engine,model)) for q in queries]
pool.close()

json.dump(new_q,open("results.json",'w'))
