import jsonlines
from data_prep.headline_parser import extract_headline
from data_prep.wiki_id_generator import generate_paragraphs_and_annotations

data = jsonlines.open("/home/jonas/data/wiki/parsed_wiki/AA/wiki_00")

pages_hier = []
for d in data:
    hier_page = extract_headline(d)
    #for k,v in hier_page["outline"].items():
        #print("!!!KEY:",k,v)

    pages_hier.append(hier_page)

paragraphs,annotations,outlines = generate_paragraphs_and_annotations(pages_hier)

for p in paragraphs:
    print(p)