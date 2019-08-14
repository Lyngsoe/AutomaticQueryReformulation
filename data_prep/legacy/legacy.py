from tqdm import tqdm
import re


PAGE_START = "<page>"
PAGE_END = "</page>"
MATCH_START = "<([^/.]+?)([ .]*?)>"

def extract_page(file):
    page_content = ""
    line = file.readline()
    while line:
        if PAGE_END in line:
            return page_content if len(page_content) > 0 else None
        page_content+=line
        line = file.readline()

def to_wiki_pages(file):
    print("loading wiki pages")
    file.seek(0)
    pages = []
    line = file.readline()
    while line:
        if PAGE_START in line:
            page = extract_page(file)

            if page is not None:
                pages.append(page)

        line = file.readline()

    return pages

def extract_section_start(page_string):
    match = re.search(MATCH_START, page_string)

    if match is None:
        return None,None,None

    name = match.group(1).split(" ")[0]
    start = match.span()[0]
    end = match.span()[1]
    return name,start,end

def extract_section_end(name,page_string):
    match = re.search("</("+name+")>",page_string)

    if match is None:
        return None,None

    #name = match.group(1).split(" ")[0]
    start = match.span()[0]
    end = match.span()[1]
    return start,end

def parse_section(name,section_string):
    section_end,_ = extract_section_end(name,section_string)

    section ={}
    section_string = section_string[:section_end]

    while True:

        sub_section_name,_,sub_section_end = extract_section_start(section_string)

        if sub_section_name is None:
            if section == {}:
                section = section_string
            return section, section_end

        sub_section,pointer = parse_section(sub_section_name,section_string[sub_section_end:])
        section.update({sub_section_name:sub_section})

        section_string = section_string[pointer:]


def parse_page(page_string):
    parsed_page = {}

    while True:
        current_name,_, subsection_start = extract_section_start(page_string)

        if current_name is None:
            return parsed_page

        section_string = page_string[subsection_start:]
        sub_section, pointer = parse_section(current_name,section_string )
        parsed_page.update({current_name: sub_section})

        page_string = page_string[subsection_start+pointer:]



def parse_pages(pages):

    parsed_pages = []
    for page in tqdm(pages,desc="parsing pages"):
        parsed_page = parse_page(page)
        parsed_pages.append(parsed_page)
    return parsed_pages







wiki_path = "/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml"
wiki_file = open(wiki_path,'r')

pages = to_wiki_pages(wiki_file)
print("\n\n#pages extracted:",len(pages),"\n\n")
parsed_pages = parse_pages(pages[:3])


keys_set = set()
for page in parsed_pages:
    print(page.keys())
    for key in page.keys():
        keys_set.add(key)

print(keys_set)
#filtered_pages = filter_pages(pages)



