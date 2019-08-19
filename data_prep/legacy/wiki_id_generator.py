import hashlib
from data_prep.wikipedia_parsing.wiki_cleaner import split_paragraphs
import re
ONLY_ONE_SPACE = " +"

def create_paras(section_txt):

    #section_txt = clean_text(section_txt)
    paras = split_paragraphs(section_txt)

    new_paras = []
    for para in paras:
        if len(para) > 20:
            para_id = hashlib.sha3_256(para.encode())
            new_paras.append({"id": para_id.hexdigest(), "text": para})
    return new_paras

def extract_sub_paras(section_txt):
    if type(section_txt) is str:
        paras = create_paras(section_txt)
        outline = [para["id"] for para in paras]
        return paras,outline,1
    else:
        paras = []
        outline = {}
        n_sum = 0
        for heading,sub_sub_section in section_txt.items():
            section_paras,sub_outline,n = extract_sub_paras(sub_sub_section)
            paras.extend(section_paras)
            if sub_outline != {} and 3 < len(heading) < 100:
                outline.update({heading:sub_outline})
                n_sum+=n

        return paras,outline,n_sum

def extract_paragraphs_create_outline(headline_page):
    paras = []
    outline = {}
    n_sum = 0
    for heading,sub_section in headline_page.items():
        if heading == "lead":
            continue
        sub_paras,sub_outline,n = extract_sub_paras(sub_section)
        paras.extend(sub_paras)
        if sub_outline != {} and 3 < len(heading) < 100:
            n_sum+=n
            outline.update({heading:sub_outline})
    return paras,outline,n_sum



def generate_paragraphs_and_annotations(headline_pages):
    paragraphs = []
    annotations = {}
    outlines = {}

    for page in headline_pages:
        paras,outline,n = extract_paragraphs_create_outline(page["outline"])
        page_annotations = []
        for para in paras:
            para_id = para["id"]
            page_annotations.append(para_id)

        annotations.update({page["url"]:page_annotations})
        paragraphs.extend(paras)
        if outline != {} and n > 3:
            o = {"title":page["title"],"outline":outline}
            outlines.update({page["url"]:o})

    return paragraphs,annotations,outlines


def generate_queries(outlines):
    queries = []

    for url,outline in outlines.items():
        title = outline["title"]
        query_strings = outline_to_query(title,outline["outline"])

        for qs in query_strings:

            id = qs.replace(" ","_")
            q = {"query":qs,"id":id,"url":url,"title":title}
            queries.append(q)

    return queries


def outline_to_query(base,outline):
    query_strings = []
    for q,v in outline.items():
        if type(v) == list:
            q_string = base+" "+q
            q_string = q_string.replace("\n"," ")
            q_string = re.sub(ONLY_ONE_SPACE,' ', q_string)
            query_strings.append(q_string)
        if type(v) == dict:
            sub_query_strings = outline_to_query(base+" "+q,v)
            query_strings.extend(sub_query_strings)

    return query_strings


def delete_paras(paras,annotations,outlines,paras_to_delete):
    outlines = outlines
    paras = paras
    annotations = annotations
    for para in paras_to_delete:
        paras = delete_para_from_paras(para,paras)
        annotation = delete_para_from_annotations(para,annotations)

    return paras,annotations,outlines


def delete_para_from_paras(para,paras):
    paras.remove(para)
    return paras

def delete_para_from_annotations(para, annotations):
    wikis_to_update = {}
    for wiki,paras in annotations.items():
        try:
            p = paras.remove(para)
            wikis_to_update.update({wiki:p})
        except ValueError:
            continue

    annotations.update(wikis_to_update)
    return annotations