import json
import matplotlib.pyplot as plt
import numpy as np
import jsonlines

base_path = "/media/jonas/archive/master/data/squad/"

ctl = json.load(open(base_path+"context_token_length.json",'r'))
qtl = json.load(open(base_path+"question_token_length.json",'r'))
cwl = json.load(open(base_path+"context_word_length.json",'r'))



max_length = np.max(ctl)
print("max_token_length: {}".format(max_length))
print("max_token_length: {}".format(np.max(qtl)))
print("max_token_length: {}".format(np.max(cwl)))
plt.hist(qtl)
plt.show()