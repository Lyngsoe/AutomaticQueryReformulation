from data_prep.wiki_dataloader import WikiDataloader
from data_prep.wiki_parser import parse_queries,parse_paragraphs_and_annotations
#from data_prep.lsh import remove_duplicates
from data_prep.lsh2 import remove_duplicates
import concurrent.futures
import multiprocessing

RUN = True

def process_page(page):
    if page is None:
        RUN = False
    p, a = parse_paragraphs_and_annotations(page)
    q = parse_queries(page)
    return p,a,q

wiki_d = WikiDataloader(max_pages=-1)

paragraphs = []
annotations = {}
queries = []




with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    while run:
        future_to_min_hash = {executor.submit(process_page,page): page for page in wiki_d.get_next()}
        for future in concurrent.futures.as_completed(future_to_min_hash):
            results = future_to_min_hash[future]
            try:
                p,a,q = future.result()

                paragraphs.extend(p)
                annotations.update(a)
                queries.extend(q)
            except Exception as exc:
                print('%r generated an exception: %s' % (results, exc))

    page = wiki_d.get_next()
    if page is None:
        break


paragraphs,annotations = remove_duplicates(paragraphs,annotations)

count_sum=0
for v in annotations.values():
    count_sum+= len(v)
avg_paras_per_article = count_sum/len(annotations.keys())

count_query_paras = 0
for q in queries:
    count_query_paras += len(annotations.get(q["url"]))
avg_paras_per_query = count_query_paras/len(queries)


print("#paragraphs:",len(paragraphs))
print("#queries:",len(queries))
print("mean #paragraphs per article:",avg_paras_per_article)
print("mean #paragraphs per query:",avg_paras_per_query)

#for p in paragraphs:
    #print(p)

#for k,v in annotations.items():
    #print(k,v)