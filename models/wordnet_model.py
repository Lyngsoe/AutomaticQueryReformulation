from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
class WordnetModel:
    def __init__(self):
        self.name = "wordnet_model"
    def reform(self,query):
        tokens = word_tokenize(query)
        reformed = query
        for word in tokens:
            i = 0
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() not in reformed.lower():
                        sym = lemma.name()
                        sym = sym.replace("_"," ")
                        reformed+=" "+sym
                        i+=1
                    if i > 3:
                        break
        return reformed