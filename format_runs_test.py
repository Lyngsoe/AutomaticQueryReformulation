from trec_car.format_runs import *
from trec_car.read_data import *
import numpy as np
import itertools
import sys

query_cbor="test200-train/train.pages.cbor-outlines.cbor"
psg_cbor="/home/jonas/data/car/test200.v2.0/test200/test200-train/train.pages.cbor-paragraphs.cbor"
out="subs.txt"

pages = []
with open(query_cbor, 'rb') as f:
    pages = [p for p in iter_outlines(f)]
pages = pages[0:10]

paragraphs = []
with open(psg_cbor, 'rb') as f:
    d = {p.para_id: p for p in iter_paragraphs(f)}
    paragraphs = d.values()

print("pages: ", len(pages))
print("paragraphs: ", len(paragraphs))

mock_ranking = []
for page in pages:
    for section_path in page.flat_headings_list():
        choices = np.random.choice(len(paragraphs),len(paragraphs))
        r = 0
        for p in paragraphs:
            query_id = "/".join([page.page_id] + [section.headingId for section in section_path])
            ranking = RankingEntry(query_id, p.para_id, choices[r], 0.5, paragraph_content=p)
            mock_ranking.append(ranking)

#mock_ranking = [(p, 1.0 / (r + 1), (r + 1)) for p, r in zip(paragraphs, range(0, 198))]

with open(out,mode='w', encoding='UTF-8') as f:
    writer = f
    format_run(writer, mock_ranking, exp_name='test')

    f.close()
    print("num queries = ", len(mock_ranking))
