from data_prep.wikipedia_parsing.wiki_parse import WikiParser
from data_prep.wikipedia_parsing.vocab_creator import VocabCreator
from data_prep.wikipedia_parsing.wiki_id_creator import WikiIDCreator
from data_prep.wikipedia_parsing.embedder import Embedder
from data_prep.wikipedia_parsing.lsh import LSH
from data_prep.wikipedia_parsing.create_folds import FoldDivider
from tfidf.tfidf import WikiTfIdf

raw_wikis = [
    ("/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml","da")
    #("/home/jonas/data/wiki/enwiki-20190801-pages-articles.xml","en")
]

drive_path = "/home/jonas/data/"
#drive_path = "/media/jonas/archive/master/data/"
debug = True
embedding_methods = ["laser"]

for wiki_path,language in raw_wikis:
    #extracted_wiki_path = extract_wiki(input_path=wiki_path,language=language)
    WikiParser(drive_path=drive_path,language=language,debug=debug)
    VocabCreator(drive_path=drive_path,language=language,embedding_methods=embedding_methods,debug=debug)
    WikiIDCreator(drive_path=drive_path,language=language,debug=debug)
    Embedder(drive_path=drive_path,language=language,embedding_methods=embedding_methods,debug=debug)
    LSH(drive_path=drive_path,language=language,embedding_methods=embedding_methods,debug=debug)
    FoldDivider(drive_path=drive_path,language=language,debug=debug)
    WikiTfIdf(drive_path=drive_path,language=language,debug=debug)

#{"language": "en", "paragraphs": 34225788, "urlqueries": 13736428, "urlwikis": 5770325, "vocabulary_size": 0}