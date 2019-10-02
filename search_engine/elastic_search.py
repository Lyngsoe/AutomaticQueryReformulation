from elasticsearch import Elasticsearch
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import re


class ELSearch:
    def __init__(self,index):
        self.number_of_cpus = multiprocessing.cpu_count()
        self.es = Elasticsearch(["127.0.0.1:9200"])
        self.index = index

        request_body = {
            "settings": {
                "number_of_shards": self.number_of_cpus,
                "number_of_replicas": 1
            }
        }
        self.es.indices.create(index=index, body=request_body,ignore=400)



    def create_indices(self,paras):
        all_data = []
        with tqdm(total=len(paras),desc="indexing paragraphs") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                future_to_index = {executor.submit(self.es.create,index=self.index(),id = para["para_id"],body={"txt":para["txt"]} ): para for para in paras}
                for future in concurrent.futures.as_completed(future_to_index):
                    respone = future_to_index[future]
                    try:
                        all_data.append(future.result())
                    except Exception as exc:
                        print('%r generated an exception: %s' % (respone, exc))

                    pbar.update(1)


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

        #txt = re.escape(txt)
        #txt = txt.replace("\\", "")
        #txt = txt.replace("/", " ")
        #txt = re.sub(r'[^\w]', ' ', txt)

        request_body = {
            "query": {
                "match": {
                    "txt": txt
                }
            }
        }

        search_results = self.es.search(index=self.index,body=request_body,size=40)

        #query.update({"results":search_results.get("hits").get("hits")})
        return search_results

    def restart(self):
        try:
            self.es.indices.delete(index=self.index)
        except:
            pass

        request_body = {
            "settings": {
                "number_of_shards": self.number_of_cpus,
                "number_of_replicas": 1
            }
        }
        self.es.indices.create(index=self.index, body=request_body)


''' Search result object
{
  "took": 7,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 1,
    "max_score": 0.2876821,
    "hits": [
      {
        "_index": "company",
        "_type": "employees",
        "_id": "1",
        "_score": 0.2876821,
        "_source": {
          "name": "Adnan Siddiqi",
          "occupation": "Software Consultant"
        }
      }
    ]
  }
}
'''