from bert_embedding import BertEmbedding
import numpy as np
#book_corpus_wiki_en_uncased
#wiki_multilingual_cased
bert = BertEmbedding(model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased")

def construct_sentence(predictions):

    tokens = get_tokens(predictions)
    sentences = compress(tokens)
    return sentences

def compress(bpe_tokens):
    sentences = []
    for sentence_tokens in bpe_tokens:
        tokens = " ".join(sentence_tokens)
        tokens = tokens.replace(" ##", "")
        tokens = tokens.replace("##", "")
        sentences.append(tokens)
    return sentences


def get_tokens(model_out):
    #print(model_out.shape)
    indices = np.argmax(model_out,2)
    #print(indices.shape)
    batch_size = indices.shape[1]
    #print(indices)
    #print("index", indices.shape, type(indices))
    bpe_tokens = [ [] for i in range(batch_size) ]
    #print(bpe_tokens)
    for ind in list(indices):
        for i in range(batch_size):
            token = bert.vocab.idx_to_token[ind[i]]
            bpe_tokens[i].append(token)
        #print(bpe_tokens)
    return bpe_tokens

def prune(sentences):
    pruned = []
    for sentence in sentences:
        index = sentence.find("?")
        if index != -1:
            sentence = sentence[:index + 1]
        pruned.append(sentence)
    return pruned