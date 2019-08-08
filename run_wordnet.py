from car import CAR
from tqdm import tqdm
from lucene_class import LUCENE
from identity_model import IdentityModel
from wordnet_model import WordnetModel
import numpy as np
import json

debug = True

search_engine = LUCENE(debug=debug)
dataset = CAR(debug=debug)
model = WordnetModel()

queries = dataset.get_queries()
formated_results = []

for query in tqdm(queries,desc="queries"):

    reformed_query = model.reform(query["txt"])
    query.update({"reformed_txt":reformed_query})

    results = search_engine.search(reformed_query)

    query.update({"results":results})



para2wiki,wiki2para = dataset.load_annotation()

recall10 = []
for query in tqdm(queries,desc="calc recall"):
    retrieved_doc = set()
    for para in query.get("results"):
        if para["rank"] < 11:
            retrieved_doc.add(para["para_id"])

    relevant_docs = wiki2para.get(query["wiki_id"])

    intersec = retrieved_doc.intersection(retrieved_doc)
    recall10.append(len(intersec)/10)


print("\nrecall 10:",np.mean(recall10))

json.dump(queries,open("results_wordnet_debug.json",'w'))