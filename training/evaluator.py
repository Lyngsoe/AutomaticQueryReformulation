from search_engine.elastic_search import ELSearch
import jsonlines

class Evaluator:
    def __init__(self,query_path,output_path,language,model,debug = False):
        self.query_path = query_path
        self.language = language
        self.debug = debug
        self.model = model
        self.output_path = output_path

        self.search_engine = ELSearch(index=self.language)
    def evaluate(self):

        result_writer = jsonlines.open(self.output_path+"eval_results.jsonl",'w')

        for query in jsonlines.open(self.query_path,'r'):
            result = self.search_engine.search([query])

            result_writer.write({"query_id":query["id"],"result":result})


