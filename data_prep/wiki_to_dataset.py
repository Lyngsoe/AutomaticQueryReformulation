import jsonlines
from data_prep.headline_parser import extract_headline
from data_prep.wiki_id_generator import generate_paragraphs_and_annotations,generate_queries
import os

def get_data_paths(base_path):
    file_paths = []
    for folder in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path,folder)):
            file_paths.append(os.path.join(base_path,folder,file))

    return file_paths

paragraphs = []
annotations = {}
outlines = {}

base_path = "/home/jonas/data/wiki/parsed_wiki"
files = get_data_paths(base_path)

for file_path in files:
    #print(file_path)
    data = jsonlines.open(file_path)

    pages_hier = []
    for d in data:
        hier_page = extract_headline(d)
        pages_hier.append(hier_page)

    p,a,o = generate_paragraphs_and_annotations(pages_hier)

    paragraphs.extend(p)
    annotations.update(a)
    outlines.update(o)


count_sum=0
for v in annotations.values():
    count_sum+= len(v)
avg_paras_per_article = count_sum/len(annotations.keys())

count_outline_paras = 0
for out in outlines.keys():
    count_outline_paras += len(annotations.get(out))
avg_paras_per_outline = count_outline_paras/len(outlines.keys())

queries = generate_queries(outlines)

for q in queries:
    print(q)


print("#paragraphs:",len(paragraphs))
print("#outlines:",len(outlines.keys()))
print("#queries:",len(queries))
print("mean #paragraphs per article:",avg_paras_per_article)
print("mean #paragraphs per outline:",avg_paras_per_outline)