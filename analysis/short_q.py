import jsonlines

squad_path = "/media/jonas/archive/master/data/squad/qas.jsonl"

r = jsonlines.open(squad_path)

for d in r:
    print(d["question"])
    print(d["question_token_ids"])
    break