import json
import matplotlib.pyplot as plt
import numpy as np

base_path = "/media/jonas/archive/master/data/squad/"

ctl = json.load(open(base_path+"context_token_length.json",'r'))
qtl = json.load(open(base_path+"question_token_length.json",'r'))

max_length = np.max(ctl)
print("max_token_length: {}".format(max_length))
print("max_token_length: {}".format(np.max(qtl)))
plt.hist(ctl)
plt.show()