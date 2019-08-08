from tqdm import tqdm
import re

PAGE_START = "<page>"
PAGE_END = "</page>"

def extract_page(file):
    page_content = ""
    line = file.readline()
    while line:
        if PAGE_END in line:
            return page_content if len(page_content) > 0 else None
        page_content+=line
        line = file.readline()

def to_wiki_pages(file,length):
    unit_divisor = 1000
    file.seek(0)
    pages = []
    pbar = tqdm(total=length/2,desc="reading wiki pages",unit_divisor=unit_divisor,unit_scale=True)
    old_fp = 0
    line = file.readline()
    while line:
        if PAGE_START in line:
            page = extract_page(file)

            if page is not None:
                pages.append(page)


        update_value = file.tell() - old_fp
        pbar.update(update_value/unit_divisor)
        old_fp = file.tell()
        line = file.readline()

    return pages

def parse_section(name,page,pointer):
    section_end = re.search("</"+name+">",page).start()
    section ={}
    section_string = page[0:section_end]


    while pointer < section_end:
        sub_section_match = re.search("<(.+?)>",section_string)

        if sub_section_match is None:
            return section, section_end

        sub_section,pointer = parse_section(sub_section_match.group(1),page[sub_section_match.end():-1],pointer)
        section.update({sub_section_match.group(1):sub_section})

        section_string = page[pointer:section_end]

    return {},section_end

def parse_page(page):
    parsed_page = {}

    section = re.search("<(.+?)>",page)
    section_name = section.group(1)
    section_start_pos = section.end()
    parsed_page = parse_section(section_name,page[section_start_pos:-1],section_start_pos)

    return parsed_page


def parse_pages(pages):

    parsed_pages = []
    for page in tqdm(pages,desc="parsing pages"):
        parsed_page = parse_page(page)
        parsed_pages.append(parsed_page)
        break
    return parsed_pages







wiki_path = "/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml"
num_lines = sum(1 for line in open(wiki_path,'r'))
wiki_file = open(wiki_path,'r')

pages = to_wiki_pages(wiki_file,num_lines)

parsed_pages = parse_pages(pages)
print(parsed_pages)
print("\n\n#pages extracted:",len(pages))

#filtered_pages = filter_pages(pages)



