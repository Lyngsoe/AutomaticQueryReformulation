import jsonlines
import os
from tqdm import tqdm

base_path = "/media/jonas/archive/google_nq/nq/"

data_paths = [base_path+file for file in os.listdir(base_path)]

save_path = "/home/jonas/data/nq_question/"
writer = jsonlines.open(save_path+"nq_queries.jsonl",'w')
question_char_length = 0

for path in tqdm(data_paths,desc="reading nq files"):
    for line in jsonlines.open(path,'r'):
        query_txt = line["question_text"]
        wiki_url = line["document_url"]
        query_txt_tokens = line["question_tokens"]
        query = {"query":query_txt,"url":wiki_url,"query_tokens":query_txt_tokens}

        for word in line["question_tokens"]:
            question_char_length+=len(word)

        writer.write(query)


print(question_char_length)

