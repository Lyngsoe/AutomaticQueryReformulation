from data_prep.wikipedia_parsing.wiki_parse import WikiParser
from data_prep.wikipedia_parsing.vocab_creator import VocabCreator
from data_prep.wikipedia_parsing.wiki_id_creator import WikiIDCreator
from data_prep.wikipedia_parsing.embedder import Embedder
from data_prep.wikipedia_parsing.lsh import LSH

raw_wikis = [
    ("/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml","da")
]

drive_path = "/home/jonas/data/"
debug = True

for wiki_path,language in raw_wikis:
    #extracted_wiki_path = extract_wiki(input_path=wiki_path,language=language)
    WikiParser(drive_path=drive_path,language=language, debug=debug)
    VocabCreator(drive_path=drive_path,language=language,debug=debug)
    WikiIDCreator(drive_path=drive_path,language=language,debug=debug)
    Embedder(drive_path=drive_path,language=language,debug=debug)
    LSH(drive_path=drive_path,language=language,debug=debug)

