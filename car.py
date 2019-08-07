from trec_car.read_data import iter_outlines,iter_paragraphs,ParaText
from trec_car.format_runs import format_run,RankingEntry
from laserembeddings import Laser
import numpy as np
import json
import cbor
from tqdm import tqdm

class CAR:
    def __init__(self,
                 article_path="/home/jonas/data/car/train/base.train.cbor",
                 paragraph_path="/home/jonas/data/car/train/base.train.cbor-paragraphs.cbor",
                 outline_path="/home/jonas/data/car/train/base.train.cbor-outlines.cbor",
                 qrels_path = "/home/jonas/data/car/train/base.train.cbor-article.qrels",
                 debug=False,
                 debug_article_path="/home/jonas/data/car/test200-train/train.pages.cbor",
                 debug_paragraph_path="/home/jonas/data/car/test200-train/train.pages.cbor-paragraphs.cbor",
                 debug_outline_path="/home/jonas/data/car/test200-train/train.pages.cbor-outlines.cbor",
                 debug_qrels_path="/home/jonas/data/car/test200-train/train.pages.cbor-article.qrels"
                 ):

        self.article_path = debug_article_path if debug  else article_path
        self.paragraph_path = debug_paragraph_path if debug else paragraph_path
        self.outline_path = debug_outline_path if debug else outline_path
        self.qrels_path = debug_qrels_path if debug else qrels_path
        self.embedding_save_path = "embeddings/car/questions/"
        self.debug = debug
    def get_queries(self):

        with open(self.outline_path, 'rb') as f:
            pages = [p for p in iter_outlines(f)]

        queries = []
        for page in pages:

            for section_path in page.flat_headings_list():
                query_id = "/".join([page.page_id] + [section.headingId for section in section_path])
                query = page.page_name

                for section in section_path:
                    query += " " + section.heading

                q = {"txt":query,"query_id":query_id,"wiki_id":page.page_id}
                queries.append(q)

        return queries

    def get_query_parts(self):
        i = 0
        with open(self.outline_path, 'rb') as f:
            pages = [p for p in iter_outlines(f)]

        queries = []
        for page in pages:
            queries.append(page.page_name)
            queries.extend([sec.heading for sec in page.outline()])
        return queries

    def embed_questions(self):
        laser = Laser()

        queries = self.get_queries()

        embedding_lookup = {}
        i = 1
        for query, query_id in queries:
            embeddings = laser.embed_sentences(query, lang='en')
            np.save(self.embedding_save_path + str(i), embeddings)
            embedding_lookup.update({str(query_id): str(i)})
            i += 1

        json.dump(embedding_lookup, open(self.embedding_save_path + "q_index.json", 'w'))

    def get_paragraphs(self):
        paras = []
        with open(self.paragraph_path, 'rb') as f:
            for p in iter_paragraphs(f):

                # Print just the text
                texts = [elem.text if isinstance(elem, ParaText)
                         else elem.anchor_text
                         for elem in p.bodies]

                paras.append((' '.join(texts),p.para_id))

        return paras

    def format_results(self,results,query_id):
        formated_results = []
        for txt,para_id,rank in results:
            ranking = RankingEntry(query_id, para_id, rank, 0.5, paragraph_content=txt)
            formated_results.append(ranking)

        return formated_results

    def write_results(self,formated_results,out_path):

        with open(out_path, mode='w', encoding='UTF-8') as f:
            writer = f
            format_run(writer, formated_results, exp_name='test')

            f.close()
            print("num queries = ", len(formated_results))

    def load_annotation(self):
        para2wiki = {}
        wiki2para = {}

        for line in open(self.qrels_path,'r'):
            data = line.split(" ")
            paraset = wiki2para.get(data[0],set())
            wiki2para.update({data[0]:paraset})
            para2wiki.update({data[2]:data[0]})

        return para2wiki,wiki2para
