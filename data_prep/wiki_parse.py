from data_prep.wiki_extractor import parse_wiki_dump

input_path = "/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml"
output_path = "/home/jonas/data/wiki/parsed_wiki"
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
                    ignore_tags=["h1"],
                    discard_elements=[],
                    num_processes=-1,
                    quiet=True)