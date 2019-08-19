from data_prep.wikipedia_parsing.wiki_parser import WikiParser

raw_wikis = [
    ("/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml","da")
]

for wiki_path,language in raw_wikis:
    #extracted_wiki_path = extract_wiki(input_path=wiki_path,language=language)
    #parse_wiki(extracted_wiki_path=extracted_wiki_path,language=language,debug=True)
    parse_wiki(language=language, debug=True)