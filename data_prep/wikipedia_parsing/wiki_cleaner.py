import re

REMOVE_LINKS = "(http.*? )*"
REMOVE_SPECIAL_CHARS = "[^\w ]"
ONLY_ONE_SPACE = " +"
REMOVE_TABLES = "{[\n\s\S\w\W]+}"
REMOVE_IMAGE = "\[\[[^:]:[\s]+?\n"
REMOVE_LISTS = "\*[\s\S]+?\n"
REMOVE_CLAMS = "[\[\]]"
REMOVE_COMMENT = "<!--[\s ]+?-->"
REMOVE_NEWLINE = "\n"

SPLIT = "[\n]{2}"

def clean_text(text):
    text = re.sub(REMOVE_LINKS,"",text)
    #text = re.sub(REMOVE_IMAGE,"",text)
    text = re.sub(REMOVE_TABLES,"",text)
    text = re.sub(REMOVE_LISTS,"",text)
    text = re.sub(REMOVE_CLAMS,"",text)
    text = re.sub(REMOVE_COMMENT,"",text)
    #no_special_chars = re.sub(REMOVE_SPECIAL_CHARS,' ', no_links)
    #only_one_space = re.sub(ONLY_ONE_SPACE,' ', no_special_chars)
    return text

def split_paragraphs(text):
    paras = re.split(SPLIT,text)
    cleaned_paras = []
    for para in paras:
        clean_p = re.sub(REMOVE_NEWLINE,' ',para)
        clean_p = re.sub("[/$&+,:;=?@#|'<>.\"^*()%!-]", ' ', clean_p)
        cleaned_paras.append(clean_p)

    return cleaned_paras