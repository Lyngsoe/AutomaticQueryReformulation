import jsonlines

ids = set()
count=0

for p in jsonlines.open("/media/jonas/archive/master/data/raffle_wiki/da/paragraphs.jsonl",'r'):

    if p["id"] in ids:
        count+=1

    ids.update([p["id"]])

print(count)