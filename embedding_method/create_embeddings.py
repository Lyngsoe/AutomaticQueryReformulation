from bert_embedding import BertEmbedding
import mxnet as mx
import torch
import numpy as np
from tqdm import tqdm

ctx = mx.gpu(0)

bert_embedder = BertEmbedding(ctx=ctx,model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased",max_seq_length=1000)


token = bert_embedder.vocab.idx_to_token[1029]

embeddings = np.zeros((30522,768))

pbar = tqdm(total=30522,desc="Creating bert embeddings")
for i in range(30522):

    toks,embs,tok_ids = bert_embedder.embedding([token])[0]

    assert len(toks) == 2

    embeddings[i] = embs[1]

    pbar.update()

pbar.close()

np.save("bert_embeddings",embeddings)