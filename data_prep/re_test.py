import re
from tqdm import tqdm
string = "<title>Amter</title>\n<ns>0</ns>\n<id>1</id>\n<redirect title=\"Amt\" />\n<revision>\n<id>7488148</id>\n<parentid>2022085</parentid>\n <timestamp>2014-02-03T21:43:11Z</timestamp>\n<contributor>\n<username>Palnatoke</username>\n<id>1053</id>\n</contributor>\n<comment>#REDIRECT [[Amt]]</comment>\n <model>wikitext</model>\n<format>text/x-wiki</format>\n<text xml:space=\"preserve\">#REDIRECT [[Amt]]</text>\n<sha1>eco50ip08cfjn6rloq8a7g76sz4xt48</sha1>\n</revision>"

sample = "<text xml:space=\"preserve\">#REDIRECT [[Amt]]</text>"
MATCH_START = "<([^/.]+?)([ .]*?)>"

sample2 = "* [http://www.uhpress.hawaii.edu/journals/jwh/jwh061p001.pdf I. C. Campbell, \"The Lateen Sail in World History\"], \'\'Journal of World History\'\', \'\'\'6\'\'\'.1 (1995), 1–23'}"

REMOVE_LINKS = "(http.*? )*"
REMOVE_SPECIAL_CHARS = "[^\w ]"
def clean_text(text):
    no_links = re.sub(REMOVE_LINKS,"",text)
    no_special_chars = re.sub(REMOVE_SPECIAL_CHARS,'', no_links)
    return no_special_chars

print(sample2)
print(re.match(REMOVE_LINKS,sample2))

print(clean_text(sample2))

def Heading_pattern(number):
    string = "="*number
    return string+"([^=])+?"+string

sample3 = "== Eksterne henvisninger ==."

print(re.match())