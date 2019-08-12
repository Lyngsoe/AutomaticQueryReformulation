from tqdm import tqdm
import re
from data_prep.wiki_extractor_mine import extract_wiki_pages
from data_prep.wiki_id_generator import generate_paragraphs_and_annotations

def Heading_pattern(number):
    string = "="*number
    return "\n{0,2}[^=\n]"+string+"([^=])+?"+string+"[^=\n]\.{0,1}"

def extract_page_headlines(page_string,heading):
    if heading == 2:
        print(page_string,"\n###########################")
    matches = list(re.finditer(Heading_pattern(heading), page_string))
    page = {}

    if matches == None or matches == []:
        return page.update({"lead":page_string})
    else:
        span_start = matches[0].span()[0]-1
        lead_txt = page_string[0:span_start]
        if len(lead_txt) > 10:
            page.update({"lead":lead_txt})

    #print(matches)

    for i in range(len(matches)):
        span = matches[i].span()
        name = matches[i].group(0).replace("=","")
        name = name.replace("\n","")
        name = name.replace(".","")
        p_start = span[1]+1
        if i == len(matches)-1:
            print("end",name)
            heading_text = page_string[p_start:]
        else:
            next_idx = i + 1
            p_end = matches[next_idx].span()[0]-1
            #print(p_start,p_end)
            heading_text = page_string[p_start:p_end]
            #print(heading_text)

        page.update({name:extract_sub_headings(heading_text,heading+1)})
        #print(page)
    return page

def extract_sub_headings(page_string,heading):
    matches = list(re.finditer(Heading_pattern(heading), page_string))
    page = {}
    if matches == None or matches == []:
        return page_string


    for i, match in enumerate(matches[0:-1]):
        span = match.span()
        name = match.group(0).replace("=","")
        p_start = span[1]
        p_end = matches[i + 1].span()[0]
        heading_text = page_string[p_start:p_end]
        page.update({name:extract_sub_headings(heading_text,heading+1)})

    return page


def extract_headline(page):
    outline_page = {"title":page["title"],"url":page["url"]}
    #print(page["text"])
    for i in range(2,6):
        outline = extract_page_headlines(page["text"],i)

        if outline is not None and outline is not {}:
            outline_page.update({"outline": outline})
            return outline_page

    outline_page.update({"outline": {"title":page["text"]}})
    return outline_page
