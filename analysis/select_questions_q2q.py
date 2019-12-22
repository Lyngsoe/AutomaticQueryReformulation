#from jiwer import wer
import jsonlines
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def wer(r, h):
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


base_path = "/media/jonas/archive/master/data/squad/cluster_exp/17_12_19/experiments/"
model = "Transformer__12-16_17:44/"
reader = jsonlines.open(base_path+model+"predictions_q2q.jsonl",'r')

selected_questions = [
    "how many museums are in",
    "what is the largest city of",
    "where is  ORG 1  located?",
    "who is the president of",
    "by what means were",
    "what type of engines became popular",
    "what was an example of a type",
    "what happens to the gdp",
    "if the average  GPE 2  worker"
]

for qa in reader:
    ground_truth = qa["targets"]
    for sq in selected_questions:
        if sq in ground_truth:

            hypothesis = qa["sentence"][0]
            index = hypothesis.find("?")
            if index != -1:
                hypothesis = hypothesis[:index+1]
            error = wer(ground_truth.split(" "), hypothesis.split(" "))
            print(ground_truth, "  --->  ", hypothesis,"   WER: ",error)

