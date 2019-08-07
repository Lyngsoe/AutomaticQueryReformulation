from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
class WordnetModel:
    def reform(self,query):
        tokens = word_tokenize(query)
        reformed = query
        for word in tokens:
            i = 0
            for syn in wordnet.synsets(word):
                if syn.name() not in reformed:
                    reformed+=" "+syn.name()
                if i > 3:
                    break