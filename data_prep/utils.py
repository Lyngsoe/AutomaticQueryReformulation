import requests

def get_wikidata_id(page_name,lang):

    url = "https://{}.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&ppprop=wikibase_item&redirects=1&titles={}".format(lang,page_name)
    resp = requests.get(url)
    body = resp.json()

    query = body["query"]
    pages = query["pages"]

    if len(list(pages.keys())) != 1:
        raise Exception
        #raise Exception("multiple pages returned: {]".format(body))

    #print(body)
    key = list(pages.keys())[0]
    page = pages[key]
    pageprops = page["pageprops"]
    wikidata_id = pageprops["wikibase_item"]

    return wikidata_id


if __name__ == "__main__":

    main_topics = ["Category:Academic_disciplines",
                   "Category:Business",
                   "Category:Concepts",
                   "Category:Crime",
                   "Category:Culture",
                   "Category:Economy",
                   "Category:Education",
                   "Category:Energy",
                   "Category:Entertainment",
                   "Category:Events",
                   "Category:Food_and_drink",
                   "Category:Geography",
                   "Category:Government",
                   "Category:Health",
                   "Category:History",
                   "Category:Human_behavior",
                   "Category:Humanities",
                   "Category:Knowledge",
                   "Category:Language",
                   "Category:Law",
                   "Category:Life",
                   "Category:Mathematics",
                   "Category:Military",
                   "Category:Mind",
                   "Category:Music",
                   "Category:Nature",
                   "Category:Objects",
                   "Category:Organizations",
                   "Category:People",
                   "Category:Philosophy",
                   "Category:Politics",
                   "Category:Religion",
                   "Category:Science",
                   "Category:Society",
                   "Category:Sports",
                   "Category:Technology",
                   "Category:Universe",
                   "Category:World"]

    category_dict = {}
    for site in main_topics:
        wiki_id = get_wikidata_id(site, "en")
        category_dict.update({wiki_id:site})

    print(category_dict)