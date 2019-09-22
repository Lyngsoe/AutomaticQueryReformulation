import numpy as np

class BertEmbedder:
    def __init__(self,language):
        from bert_embedding import BertEmbedding
        import mxnet as mx
        self.language = language
        ctx = mx.gpu(0)
        self.bert_embedder = BertEmbedding(ctx=ctx, model="bert_12_768_12", dataset_name="wiki_multilingual_cased")

    def __call__(self, sentences):
        results = self.bert_embedder(sentences)
        mean = []
        for i,(tokens,embeddings) in enumerate(results):

            sentence_embedding = np.mean(embeddings,axis=0)

            if sentence_embedding.size != 768:
                print("resutls:",results)
                print("tokens:",tokens)
                print("sentece_emb:",sentence_embedding)
                print("sentece_emb:", sentences[i])
            assert sentence_embedding.size == 768
            mean.append(sentence_embedding)

        return mean

class LaserEmbedder:
    def __init__(self,language):
        from laserembeddings import Laser
        self.laser = Laser()
        self.language = language


    def __call__(self, sentences):
        return self.laser.embed_sentences(sentences,self.language)


def get_embedder(method,language):
    if method == "laser":
        return LaserEmbedder(language)
    elif method == "bert":
        return BertEmbedder(language)
    else:
        raise Exception("Embedding method {} not supported".format(method))