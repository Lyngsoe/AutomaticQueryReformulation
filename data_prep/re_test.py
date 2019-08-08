import re

string = "<page>\n<title>Amter</title>\n<ns>0</ns>\n<id>1</id>\n<redirect title=\"Amt\" />\n<revision>\n<id>7488148</id>\n<parentid>2022085</parentid>\n <timestamp>2014-02-03T21:43:11Z</timestamp>\n<contributor>\n<username>Palnatoke</username>\n<id>1053</id>\n</contributor>\n<comment>#REDIRECT [[Amt]]</comment>\n <model>wikitext</model>\n<format>text/x-wiki</format>\n<text xml:space=\"preserve\">#REDIRECT [[Amt]]</text>\n<sha1>eco50ip08cfjn6rloq8a7g76sz4xt48</sha1>\n</revision>\n</page>"

m = re.search("<(.+?)>",string).group(1)

print(m)