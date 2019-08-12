import jsonlines

data = jsonlines.open("/home/jonas/data/wiki/parsed_wiki/AA/wiki_00")

for d in data:
    print(d.keys())
    print(d["id"])
    break