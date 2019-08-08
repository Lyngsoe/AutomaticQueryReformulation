from elasticsearch import Elasticsearch
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import re


class ELSearch:
    def __init__(self):
        self.number_of_cpus = multiprocessing.cpu_count()
        self.es = Elasticsearch(["127.0.0.1:9200"])

        try:
            self.es.indices.delete(index="paragraphs")
        except:
            pass

        request_body = {
            "settings": {
                "number_of_shards": self.number_of_cpus,
                "number_of_replicas": 1
            }
        }
        self.es.indices.create(index='paragraphs', body=request_body)

    def index(self,paras):
        all_data = []
        with tqdm(total=len(paras),desc="indexing paragraphs") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                future_to_index = {executor.submit(self.es.create,index="paragraphs",id = para["para_id"],body={"txt":para["txt"]} ): para for para in paras}
                for future in concurrent.futures.as_completed(future_to_index):
                    respone = future_to_index[future]
                    try:
                        all_data.append(future.result())
                    except Exception as exc:
                        print('%r generated an exception: %s' % (respone, exc))

                    pbar.update(1)


    def index_seq(self,paras):

        for para in tqdm(paras):
            self.es.create(index="paragraphs",id = para["para_id"],body=para)

    def index_bulk(self,paras):
        self.es.bulk(index="paragraphs",body=paras)

    def search(self,queries):
        all_results = []
        with tqdm(total=len(queries),desc="queries") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.number_of_cpus) as executor:
                future_to_query = {executor.submit(self._search,query): query for query in queries}
                for future in concurrent.futures.as_completed(future_to_query):
                    results = future_to_query[future]
                    try:
                        all_results.append(future.result())
                    except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))

                    pbar.update(1)

        return all_results

    def _search(self,query):

        txt = query["txt"]

        txt = re.escape(txt)
        txt = txt.replace("\\", "")
        txt = txt.replace("/", " ")
        txt = re.sub(r'[^\w]', ' ', txt)

        request_body = {
            "query": {
                "match": {
                    "txt": txt
                }
            }
        }

        search_results = self.es.search(index="paragraphs",body=request_body,size=40)

        query.update({"results":search_results.get("hits").get("hits")})
        return query