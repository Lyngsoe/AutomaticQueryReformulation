from data_prep.wiki_dataloader import WikiDataloader
from data_prep.wiki_parser import parse_queries,parse_paragraphs_and_annotations
from data_prep.lsh2 import remove_duplicates
import concurrent.futures
import multiprocessing
import jsonlines
import json
from tqdm import tqdm

def process_page(page):
    p, a = parse_paragraphs_and_annotations(page)
    q = parse_queries(page)
    return p,a,q


def parse_wiki(debug=False,extracted_wiki_path="/home/jonas/data/wiki/extracted_wiki/da",base_save_path = "/home/jonas/data/raffle_wiki/",language = "da"):

    base_save_path+="/"+language+"/"

    if debug:
        max_page=1000
    else:
        max_page = -1


    wiki_d = WikiDataloader(max_pages=max_page,base_path=extracted_wiki_path)

    paragraphs = []
    annotations = {}
    queries = []

    while True:

        batch = wiki_d.get_next_batch()

        if batch is None:
            break

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_min_hash = {executor.submit(process_page,page): page for page in batch}
            for future in concurrent.futures.as_completed(future_to_min_hash):
                results = future_to_min_hash[future]
                try:
                    p,a,q = future.result()

                    paragraphs.extend(p)
                    annotations.update(a)
                    queries.extend(q)
                except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))



    paragraphs,annotations = remove_duplicates(paragraphs,annotations)

    count_sum=0
    for v in annotations.values():
        count_sum+= len(v)
    avg_paras_per_article = count_sum/len(annotations.keys())

    count_query_paras = 0
    for q in queries:
        count_query_paras += len(annotations.get(q["url"]))
    avg_paras_per_query = count_query_paras/len(queries)


    tqdm.write("#paragraphs:",len(paragraphs))
    tqdm.write("#queries:",len(queries))
    tqdm.write("mean #paragraphs per article:",avg_paras_per_article)
    tqdm.write("mean #paragraphs per query:",avg_paras_per_query)




    writer = jsonlines.open(base_save_path+"paragraphs.json",'w')
    for para in paragraphs:
        writer.write(para)


    writer = jsonlines.open(base_save_path+"annotations.json",'w')
    for wiki,paras in annotations.items():
        writer.write({wiki:paras})


    writer = jsonlines.open(base_save_path+"queries.json",'w')
    for q in queries:
        writer.write(q)


    wiki_info = {
        "language":language,
        "paragraphs":len(paragraphs),
        "queries":len(queries),
        "paragraph_per_art":avg_paras_per_article,
        "paragraph_per_query":avg_paras_per_query
    }

    json.dump(wiki_info,base_save_path+"wiki_info.json")