from search_engine.elastic_search import ELSearch
import jsonlines
from models.identity_model import IdentityModel
class Evaluator:
    def __init__(self,query_path,output_path,language,model,debug = False):
        self.query_path = query_path
        self.language = language
        self.debug = debug
        self.model = model
        self.output_path = output_path
        self.queries = []

        self.search_engine = ELSearch(index=self.language)
    def evaluate(self):
        self.load_all_queries()
        result_writer = jsonlines.open(self.output_path+"eval_results.jsonl",'w')

        results = self.search_engine.search(self.queries)

        for i,res in enumerate(results):
            result_writer.write({"query":self.queries[i],"result":res})

    def load_all_queries(self):

        for query in jsonlines.open(self.query_path, 'r'):
            self.queries.append(query)

if __name__ == '__main__':
    language = "da"
    query_path = "/media/jonas/archive/master/data/raffle_wiki/da/4-queries.jsonl"
    IdentityModel()
    debug = False
    eval = Evaluator(query_path=query_path,output_path="/home/jonas/Documents/master/",language=language,model=IdentityModel(),debug=debug)
    eval.evaluate()