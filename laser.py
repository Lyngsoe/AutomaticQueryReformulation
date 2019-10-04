from laserembeddings import Laser

def compress(subwords):
    subwords = subwords.replace("@@ ",'')
    return subwords

laser = Laser()

embeddings = laser.embed_sentences_bpe(
    ['let your neural network be polyglot',
     'use multilingual embeddings!'],
    lang='en')  # lang is used for tokenization

print(compress(embeddings[0]))


