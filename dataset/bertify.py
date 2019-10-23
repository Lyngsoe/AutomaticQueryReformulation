from bert_embedding import BertEmbedding
import numpy as np

bert = BertEmbedding(model="bert_12_768_12", dataset_name="wiki_multilingual_cased")

def construct_sentence(predictions):

    tokens = get_tokens(predictions)
    sentences = compress(tokens)
    return sentences

def compress(bpe_tokens):
    tokens = " ".join(bpe_tokens)
    tokens = tokens.replace(" ##", "")
    tokens = tokens.replace("##", "")
    return tokens


def get_tokens(model_out):
    indices = np.argmax(model_out,1)
    print(indices)
    # print("index", indices.shape, type(indices))
    bpe_tokens = []
    for ind in list(indices):
        token = bert.vocab.idx_to_token[ind]
        bpe_tokens.append(token)

    return bpe_tokens
