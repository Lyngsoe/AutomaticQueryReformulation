import numpy as np

class BertEmbedder:
    def __init__(self,language):
        from bert_embedding import BertEmbedding
        import mxnet as mx
        self.language = language
        ctx = mx.gpu(0)
        self.bert_embedder = BertEmbedding(ctx=ctx,model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased")

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

class BertEmbedderSub:
    def __init__(self,language):
        from bert_embedding import BertEmbedding
        import mxnet as mx
        self.language = language
        ctx = mx.gpu(0)
        self.bert_embedder = BertEmbedding(ctx=ctx,model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased",max_seq_length=1000)

    def __call__(self, sentences):
        results = self.bert_embedder(sentences)
        #print(results[0][0])
        #print(results[0][2])
        #print(len(results[0][0]))
        #print(len(results[0][1]))
        #print(len(results[0][2]))

        return results

class BertEmbedderToken:
    def __init__(self,language):
        from bert_embedding import BertEmbedding
        self.language = language
        bert_embedder = BertEmbedding(model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased",max_seq_length=1000)

        self.tokenizer = bert_embedder.tokenizer
        self.vocab = bert_embedder.vocab
    def __call__(self, sentences):
        tokens = self.tokenizer(sentences[0])
        token_ids = self.vocab.to_indices(tokens)
        return token_ids

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
    elif method == "bertsub":
        return BertEmbedderSub(language)
    elif method == "berttoken":
        return BertEmbedderToken(language)
    else:
        raise Exception("Embedding method {} not supported".format(method))

if __name__ == '__main__':
    bert = get_embedder("bertsub","en")