import jsonlines
import os
from tqdm import tqdm
import re
from data_prep.wikipedia_parsing.utils import get_wikidata_id
import json

load_path = "/media/jonas/archive/google_nq/nq/"
data_paths = [load_path+file for file in os.listdir(load_path)]

save_path = "/home/jonas/data/nq_question/"
os.makedirs(save_path,exist_ok=True)

error_log_path = save_path+"nq_queries_error.jsonl"
writer = jsonlines.open(save_path+"nq_queries.jsonl",'w')
error_writer = jsonlines.open(error_log_path,'w')

number_of_questions = 0
question_char_length = 0
number_of_errors = 0

for path in tqdm(data_paths,desc="reading nq files"):
    for line in jsonlines.open(path,'r'):
        try:
            query_txt = line["question_text"]
            wiki_url = line["document_url"]
            query_txt_tokens = line["question_tokens"]

            MATCH_TITLE = "<h1>(.+?)</h1>"
            m = re.search(MATCH_TITLE, line["document_html"])
            title = m.group(1)
            wikidata_id = get_wikidata_id(title, "en")

            query = {"question_text":query_txt,"url":wiki_url,"query_tokens":query_txt_tokens,"title":title,"wikidata_id":wikidata_id}

            for word in line["question_tokens"]:
                question_char_length+=len(word)

            writer.write(query)
            number_of_questions += 1

        except Exception as e:
            error_log = {"error": e, "data": line}
            error_writer.write(error_log)
            print("Error occoured:", error_log)
            number_of_errors+=1


print("total question char length:",question_char_length)
print("total number of questions:",number_of_questions)

meta = {"questions":number_of_questions,"language":"en","errors":number_of_errors}
json.dump(meta,open(save_path+"nq_queries_meta.json",'w'))
