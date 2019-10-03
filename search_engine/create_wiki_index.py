from search_engine.elastic_search import ELSearch
import jsonlines

def load_all_paras(para_path):
    paras = []

    for p in jsonlines.open(para_path,'r'):
        paras.append(p)

    return paras


language = "da"
es = ELSearch(language)
es.restart()

paras_path = "/media/jonas/archive/master/data/raffle_wiki/da/paragraphs.jsonl"

paras = load_all_paras(paras_path)

es.create_indices(paras)