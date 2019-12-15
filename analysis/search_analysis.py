import jsonlines
from search_engine.elastic_search import ELSearch
from analysis.process_results import eval

def get_documents_from_result(results):

    hits = results["hits"]["hits"]

    retrieved_documents = []
    for hit in hits:
        retrieved_documents.append(hit["_id"])
    return retrieved_documents

question_path = "/media/jonas/archive/master/data/rl_squad/"

questions = jsonlines.open(question_path+"qas_eval.jsonl")
search_engine = ELSearch("squad")

results = []
i = 0

for q in questions:
    text = q["question"]
    relevant_docs = q["paragraphs"]
    actual_doc = q["c_id"]
    search_result = search_engine.search([text])[0]
    results_docs = get_documents_from_result(search_result)
    results.append((relevant_docs,results_docs))

    if i > 10:
        break
    i+=1

eval(results)