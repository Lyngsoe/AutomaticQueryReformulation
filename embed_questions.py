from trec_car.read_data import *
from laserembeddings import Laser
import numpy as np
import json
embedding_save_path = "embeddings/questions/"
query_cbor="test200-train/train.pages.cbor-outlines.cbor"
pages = []
with open(query_cbor, 'rb') as f:
    pages = [p for p in iter_outlines(f)]

queries = []
for page in pages:

    for section_path in page.flat_headings_list():
        query_id = "/".join([page.page_id] + [section.headingId for section in section_path])
        query = page.page_name

        for section in section_path:
            query += " "+section.heading
        queries.append((query,query_id))

queries = queries[0:100]

laser = Laser()

embedding_lookup = {}
i = 1
for query,query_id in queries:
    embeddings = laser.embed_sentences(query,lang='en')
    np.save(embedding_save_path+str(i),embeddings)
    embedding_lookup.update({str(query_id):str(i)})
    i+=1

json.dump(embedding_lookup,open(embedding_save_path+"q_index.json",'w'))