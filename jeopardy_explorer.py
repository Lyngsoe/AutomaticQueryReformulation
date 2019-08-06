import json

def print_dict(in_dict,t=0):
    tabs = ""
    for i in range(t):
        tabs+="\t"

    for data in in_dict.keys():
        print(tabs,data)
        if data is list:
            for d in data:
                if d is dict:
                    print_dict(d,t+1)
        if data is dict:
            print_dict(data,t+1)

body = json.load(open("/home/jonas/data/jeopardy/val/000475-5243_jeopardy_americanauthors_200.json",'r'))

print(body["search_results"][0].keys())
print(body["answer"])