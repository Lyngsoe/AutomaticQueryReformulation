from data_prep.google_nq.google_translate import translate_with_google
import jsonlines
import json
from tqdm import tqdm
import os

def translate_nq(target_lang):
    source_lang = "en"

    nq_base_path = "/home/jonas/data/nq_question/"
    nq_trans_base_path = "/home/jonas/data/nq_question/{}/".format(target_lang)
    os.makedirs(nq_trans_base_path,exist_ok=True)

    nq_path = nq_base_path+"nq_queries.jsonl"
    nq_meta_path = nq_base_path+"nq_queries_meta.json"
    nq_trans_path = nq_trans_base_path+"nq_{}_queries.jsonl".format(target_lang)
    nq_trans_meta_path = nq_trans_base_path+"nq_{}_queries_meta.json".format(target_lang)
    writer = jsonlines.open(nq_trans_path,'w')

    error_log_path = nq_trans_base_path+"nq_{}_queries_error.jsonl".format(target_lang)
    error_writer = jsonlines.open(error_log_path,'w')


    meta = json.load(open(nq_meta_path,'r'))


    number_of_questions = 0
    number_of_errors = 0
    pbar = tqdm(total=int(meta["questions"]),desc="translating nq to {}".format(target_lang))
    for query in jsonlines.open(nq_path,'r'):
        try:
            new_query = query

            translated_txt = translate_with_google(query["question_text"],target_lang=target_lang,source_lang=source_lang)

            new_query["question_text"] = translated_txt

            writer.write(new_query)

            number_of_questions+=1
        except Exception as e:
            error_log = {"error":e,"data":query}
            error_writer.write(error_log)
            tqdm.write("Error occoured: {}".format(error_log))
            number_of_errors+=1

        pbar.update()

    new_meta = {"questions":number_of_questions,"language":target_lang,"errors":number_of_errors}

    json.dump(new_meta,open(nq_trans_meta_path,'w'))



if __name__ == '__main__':

    target_lang = "da"

    translate_nq(target_lang)