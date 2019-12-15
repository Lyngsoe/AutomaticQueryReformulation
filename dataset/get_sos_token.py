import json
import jsonlines
import spacy
from tqdm import tqdm
import os
import random
from embedding_method.embedders import get_embedder
import numpy as np

bert = get_embedder("bertsub", "en")


q_tokens,q_emb,q_token_ids = bert(["[unused4]"])[0]
sos = q_emb[0]
print(q_tokens,q_token_ids)
q_tokens,q_emb,q_token_ids = bert(["hello hello"])[0]
print(q_tokens,q_token_ids)
print(np.array_equal(sos,q_emb[0]))

for i in range(10):
    token = bert.bert_embedder.vocab.idx_to_token[i]
    print(token)


bert.bert_embedder.bert()