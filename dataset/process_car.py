from trec_car.read_data import iter_outlines,iter_paragraphs,ParaText
from trec_car.format_runs import format_run,RankingEntry
from laserembeddings import Laser
import numpy as np
import json
import cbor
from tqdm import tqdm
import os
import copy

def get_queries(outline_path):
    with open(outline_path, 'rb') as f:
        pages = [p for p in iter_outlines(f)]

    queries = []
    for page in pages:

        for section_path in page.flat_headings_list():
            query_id = "/".join([page.page_id] + [section.headingId for section in section_path])
            query = page.page_name

            for section in section_path:
                query += " " + section.heading
                print(section.paragraphs)


            q = {"txt": query, "query_id": query_id, "wiki_id": page.page_id}
            queries.append(q)
        raise Exception

    return queries

debug = True
base_path = "/home/jonas/data/car/"
article_path = base_path+"test200-train/train.pages.cbor" if debug else base_path+"train/base.train.cbor"
paragraph_path = base_path+"test200-train/train.pages.cbor-paragraphs.cbor" if debug else base_path+"train/base.train.cbor-paragraphs.cbor"
outline_path = base_path+"test200-train/train.pages.cbor-outlines.cbor" if debug else base_path+"train/base.train.cbor-outlines.cbor"
qrels_path = base_path+"test200-train/train.cbor-article.qrels" if debug else base_path+"train/base.train.cbor-article.qrels"

save_path = base_path+"my_car/"
os.makedirs(save_path,exist_ok=True)

bert = None

queries = get_queries(outline_path)

qas = []

for q in queries:
    new_qa = copy.deepcopy(q)
    bert_emb = bert(new_qa["txt"])


def get_paragraphs(self):
    paras = []
    with open(self.paragraph_path, 'rb') as f:
        for p in iter_paragraphs(f):

            # Print just the text
            texts = [elem.text if isinstance(elem, ParaText)
                     else elem.anchor_text
                     for elem in p.bodies]
