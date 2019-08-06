from trec_car.read_data import *

query_cbor="test200-train/train.pages.cbor-outlines.cbor"
pages = []
with open(query_cbor, 'rb') as f:
    pages = [p for p in iter_outlines(f)]

q_len = 0
for page in pages:
    for section_path in page.flat_headings_list():
        query = page.page_name
        for section in section_path:
            query += " "+section.heading
            q_len += len(query)

print("total query character length",q_len)