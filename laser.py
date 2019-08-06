from laserembeddings import Laser

laser = Laser()

embeddings = laser.embed_sentences(
    ['let your neural network be polyglot',
     'use multilingual embeddings!'],
    lang='en')  # lang is used for tokenization

print(type(embeddings))