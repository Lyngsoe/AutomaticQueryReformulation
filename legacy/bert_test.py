from bert_embedding import BertEmbedding
import mxnet as mx
sentences = ["let your neural network be polyglot hej med dig","use multilingual embeddings!"]
word = ["hello"]
ctx = mx.gpu(0)
print(sentences)
bert_embedder = BertEmbedding(ctx=ctx,model="bert_12_768_12",dataset_name="wiki_multilingual_cased",)

embeddings = bert_embedder(sentences)

print(len(embeddings))
print(type(embeddings[0][0]))
print(embeddings[0][0])

print(type(embeddings[0][1]))
print(len(embeddings[0][1]))