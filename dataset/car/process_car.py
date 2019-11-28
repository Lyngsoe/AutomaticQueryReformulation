from trec_car.read_data import iter_outlines,iter_paragraphs,ParaText
from embedding_method.embedders import get_embedder
import os
import copy
import random
import jsonlines
from tqdm import tqdm

def write_batch(writer,qas):
    for qa in qas:
        writer.write(qa)

def get_queries(path):
    with open(path, 'rb') as f:
        pages = [p for p in iter_outlines(f)]

    queries = []
    for page in pages:

        for section_path in page.flat_headings_list():
            query_id = "/".join([page.page_id] + [section.headingId for section in section_path])
            query = page.page_name

            for section in section_path:
                query += " " + section.heading


            q = {"txt": query, "query_id": query_id, "wiki_id": page.page_id}
            queries.append(q)

    return queries

def get_annotations(path):

    wiki_paragraphs = {}

    for line in open(path,'r'):
        divided = line.split()
        wiki_id = divided[0]
        para_id = divided[2]

        annotations = wiki_paragraphs.get(wiki_id,[])
        annotations.append(para_id)

        wiki_paragraphs.update({wiki_id:annotations})

    return wiki_paragraphs

def process_fold(fold,save_path,base_path,bert):

    qas = []

    annotations = get_annotations(base_path+"fold-{}-".format(fold)+"base.train.cbor-article.qrels")
    queries = get_queries(base_path+"fold-{}-".format(fold)+"base.train.cbor-outlines.cbor")
    pbar = tqdm(total=len(queries),desc="processing quries fold {}".format(fold))
    for q in queries:
        new_qa = copy.deepcopy(q)
        q_tokens,q_emb,q_token_ids = bert([new_qa["txt"]])[0]

        paragraphs = annotations.get(new_qa["wiki_id"])
        if paragraphs is None:
            continue

        new_qa.update({"tokens": q_tokens, "question_emb": [x.tolist() for x in q_emb],
                  "question_token_ids": [int(x) for x in q_token_ids],
                       "paragraphs":paragraphs})

        qas.append(new_qa)
        pbar.update()
        if len(qas) > 100:
            break

    pbar.close()
    random.shuffle(qas)
    qa_writer = jsonlines.open(save_path + "fold-{}.jsonl".format(fold), 'w', flush=True)
    write_batch(qa_writer,qas)


debug = False
#base_path = "/home/jonas/data/car/"
base_path = "/media/jonas/archive/master/data/car/"
save_path = base_path+"my_car/"
base_path += "test200-train/" if debug else "train/"
#article_path = base_path+"train.pages.cbor" if debug else base_path+"base.train.cbor"
#paragraph_path = base_path+"train.pages.cbor-paragraphs.cbor" if debug else base_path+"base.train.cbor-paragraphs.cbor"
#outline_path = base_path+"train.pages.cbor-outlines.cbor" if debug else base_path+"base.train.cbor-outlines.cbor"
#qrels_path = base_path+"train.cbor-article.qrels" if debug else base_path+"base.train.cbor-article.qrels"


os.makedirs(save_path,exist_ok=True)

bert = get_embedder("bertsub", "en")



for fold in range(5):
    process_fold(fold,save_path,base_path,bert)
