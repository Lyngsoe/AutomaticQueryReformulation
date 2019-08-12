import hashlib
from data_prep.wiki_cleaner import clean_text,split_paragraphs

def create_paras(section_txt):

    #section_txt = clean_text(section_txt)
    paras = split_paragraphs(section_txt)

    new_paras = []
    for para in paras:
        if para != "":
            para_id = hashlib.sha3_256(para.encode())
            new_paras.append({"id": para_id.hexdigest(), "text": para})
    return new_paras

def extract_sub_paras(section_txt):
    if type(section_txt) is str:
        paras = create_paras(section_txt)
        outline = [para["id"] for para in paras]
        return paras,outline
    else:
        paras = []
        outline = {}
        for heading,sub_sub_section in section_txt.items():
            section_paras,sub_outline = extract_sub_paras(sub_sub_section)
            paras.extend(section_paras)
            outline.update({heading:sub_outline})

        return paras,outline

def extract_paragraphs_create_outline(headline_page):
    paras = []
    outline = {}
    for heading,sub_section in headline_page.items():
        if heading == "lead":
            continue
        sub_paras,sub_outline = extract_sub_paras(sub_section)
        paras.extend(sub_paras)
        outline.update({heading:sub_outline})
    return paras,outline



def generate_paragraphs_and_annotations(headline_pages):
    paragraphs = []
    annotations = []
    outlines = []

    for page in headline_pages:
        paras,outline = extract_paragraphs_create_outline(page["outline"])
        page_annotations = []
        for para in paras:
            para_id = para["id"]
            page_annotations.append(para_id)

        annotations.append({page["url"]:page_annotations})
        paragraphs.extend(paras)
        outlines.append(outline)

    return paragraphs,annotations,outlines