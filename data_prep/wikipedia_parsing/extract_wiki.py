from data_prep.wikipedia_parsing.wiki_extractor import parse_wiki_dump
from tqdm import tqdm

def extract_wiki(input_path = "/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml",
                 output_path="/home/jonas/data/wiki/extracted_wiki",
                 language="da"):

    output_path = output_path+"/"+language
    tqdm.write("extracting {} wiki".format(language))
    parse_wiki_dump(input_path,
                        output_path,
                        json_format=True,
                        html_format=False,
                        include_sections=True,
                        include_lists=False,
                        include_links=False,
                        include_tables=False,
                        filter_disambig_pages=True,
                        min_text_length=-1,
                        max_file_size="1M",
                        ignore_tags=["h1","ref"],
                        discard_elements=[],
                        num_processes=-1,
                        quiet=True)

    return output_path