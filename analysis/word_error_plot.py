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


base_path = "/home/jonas/data/squad/"

reader = jsonlines.open(base_path+"predictions.jsonl",'r')

wers = []
pbar = tqdm(total=5928,desc="calculating WER")
for qa in reader:
    ground_truth = qa["targets"]
    hypothesis = qa["sentence"]
    index = hypothesis.find("?")
    if index != -1:
        hypothesis = hypothesis[:index+1]
        error = wer(ground_truth.lower().split(" "), hypothesis.lower().split(" "))
        wers.append(error)
        if error < 3:
            print(ground_truth)
            print(hypothesis)
    pbar.update()

pbar.close()

print("average WER: ",np.average(wers))
plt.hist(wers,rwidth=0.5,bins=30)
plt.title("Word Error Rate - Transformer Medium")
plt.xlabel("Word Error Rate")
plt.ylabel("Number of Questions")
plt.savefig("fig/worderrorrate.png")
#plt.xlim(0,40)
plt.show()