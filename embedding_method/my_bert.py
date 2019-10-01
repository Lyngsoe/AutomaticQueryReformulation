from bert_embedding import BertEmbedding
import mxnet as mx

class MyBert:
    def __init__(self):
        ctx = mx.gpu(0)
        self.bert_embedder = BertEmbedding(ctx=ctx, model="bert_12_768_12", dataset_name="wiki_multilingual_cased")

    def __call__(self, *args, **kwargs):
        bert_embedder(sentences)