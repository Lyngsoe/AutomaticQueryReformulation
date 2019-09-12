from bert_embedding import BertEmbedding

sentences = ['let your neural network be polyglot','use multilingual embeddings!']

bert_embedder = BertEmbedding(model="bert_12_768_12",dataset_name="wiki_multilingual_cased")

embeddings = bert_embedder(sentences)