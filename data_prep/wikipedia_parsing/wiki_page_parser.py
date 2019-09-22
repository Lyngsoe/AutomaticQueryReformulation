import re
from data_prep.wikipedia_parsing.wiki_cleaner import split_paragraphs
import hashlib

MIN_SECTION_LENGTH = 3
MAX_SECTION_LENGTH = 100
MIN_SECTIONS = 3
MIN_PARAGRAPH_LENGTH = 500
MIN_PARAGRAPH_PER_WIKI = 3

def remove_title(page_txt):
    lines = page_txt.split("\n")
    no_title_txt = "\n".join(lines[1:])
    return no_title_txt

def Heading_pattern(number):
    string = "="*number
    return "\n[^=]"+string+"([^=])+?"+string+"[^=]"

def create_paras(section_txt):

    paras = split_paragraphs(section_txt)

    new_paras = []
    for para in paras:
        if len(para) > MIN_PARAGRAPH_LENGTH:
            para_id = hashlib.sha3_256(para.encode())
            new_paras.append({"id": para_id.hexdigest(), "text": para})
    return new_paras

def extract_page_queries(page_string,heading_nr,query_base):
    matches = list(re.finditer(Heading_pattern(heading_nr), page_string))
    query_strings = set()

    if matches == None or matches == []:
        query_strings.add(query_base)
        return query_strings
    else:

        ## Add current heading to queries
        query_strings.add(query_base)

        for i in range(len(matches)):
            span = matches[i].span()
            name = matches[i].group(0).replace("=","")
            name = name.replace("\n","")
            name = name.replace(".","")
            p_start = span[1]+1
            if i == len(matches)-1:
                heading_text = page_string[p_start:]
            else:
                next_idx = i + 1
                p_end = matches[next_idx].span()[0]-1
                heading_text = page_string[p_start:p_end]

            if MIN_SECTION_LENGTH < len(name) < MAX_SECTION_LENGTH:
                query_strings.update(extract_page_queries(heading_text,heading_nr+1,query_base+" "+name))
            else:
                query_strings.update(extract_page_queries(heading_text, heading_nr + 1, query_base))
        return query_strings

def parse_queries(page):
    title = page["title"]

    if not (MIN_SECTION_LENGTH < len(title) < MAX_SECTION_LENGTH):
        return []

    wiki_id = page["url"]
    p_text = remove_title(page["text"])
    for i in range(2,6):
        qs = extract_page_queries(p_text,i,title)
        if qs is not None and qs is not set():
            queries = []
            for q in qs:
                query = {"text":q,"id":q.replace(" ","_"),"url":wiki_id,"title":title}
                queries.append(query)

            if len(queries) >= MIN_SECTIONS:
                return queries
            else:
                return []

    return []

def extract_page_paragraphs(page_string,heading_nr):
    matches = list(re.finditer(Heading_pattern(heading_nr), page_string))
    paragraphs = []
    if matches is None or matches == []:
        paragraphs.extend(create_paras(page_string))
        return paragraphs
    else:
        for i in range(len(matches)):
            span = matches[i].span()
            p_start = span[1]+1

            if i == 0: ## lead text before first section
                if span[0]-1 > MIN_PARAGRAPH_LENGTH:
                    lead_text = page_string[:span[0]-1]
                    paragraphs.extend(create_paras(lead_text))

            if i == len(matches)-1: ## last section
                heading_text = page_string[p_start:]
            else: ##middle sections
                next_idx = i + 1
                p_end = matches[next_idx].span()[0]-1
                heading_text = page_string[p_start:p_end]

            paragraphs.extend(extract_page_paragraphs(heading_text,heading_nr+1))
        return paragraphs




def extract_paragraphs_and_annotations(page):
    p_text = remove_title(page["text"])
    for i in range(2, 6):
        paragraphs = extract_page_paragraphs(p_text, i)
        if paragraphs is not None and len(paragraphs) >= MIN_PARAGRAPH_PER_WIKI:
            annotations = [para["id"] for para in paragraphs]
            return paragraphs,annotations

    return [],[]
