from cbor import cbor
from trec_car.read_data import *
from trec_car.read_data import Para

para_path = "/media/jonas/archive/master/data/car/train/base.train.cbor-paragraphs.cbor"
art_path = "/media/jonas/archive/master/data/car/train/base.train.cbor"
outline_path = "/media/jonas/archive/master/data/car/train/base.train.cbor-outlines.cbor"





with open(art_path, 'rb') as f:
    for p in iter_annotations(f):
        print('\npagename:', p.page_name)
        print('\npageid:', p.page_id)
        print('\nmeta:', p.page_meta)

        for ks in p.outline():
            for sec in ks:
                try:
                    print(sec.outline)
                except:
                    print(sec.para_id)
                    raise Exception
        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        # print(headings)

        if len(p.outline()) > 0:
            print('heading 1=', p.outline()[0].__str__())

            print('deep headings= ',
                  [(str(section.heading), len(children)) for (section, children) in p.deep_headings_list()])

            print('flat headings= ', ["/".join([str(section.heading) for section in sectionpath]) for sectionpath in
                                      p.flat_headings_list()])



with open(outline_path, 'rb') as f:
    for p in iter_outlines(f):
        print('\npagename:', p.page_name)

        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        print('headings= ', [(str(section.heading), len(children)) for (section, children) in headings])

        if len(p.outline()) > 2:
            print('heading 1=', p.outline()[0])

            print('deep headings= ',
                  [(str(section.heading), len(children)) for (section, children) in p.deep_headings_list()])

            print('flat headings= ', ["/".join([str(section.heading) for section in sectionpath]) for sectionpath in
                                      p.flat_headings_list()])


with open(para_path,'rb') as f:
    for p in iter_paragraphs(f):
        print('\n', p.para_id, ':')

        # Print just the text
        texts = [elem.text if isinstance(elem, ParaText)
                 else elem.anchor_text
                 for elem in p.bodies]
        print(' '.join(texts))

        # Print just the linked entities
        entities = [elem.page
                    for elem in p.bodies
                    if isinstance(elem, ParaLink)]
        print(entities)

        # Print text interspersed with links as pairs (text, link)
        mixed = [(elem.anchor_text, elem.page) if isinstance(elem, ParaLink)
                 else (elem.text, None)
                 for elem in p.bodies]
        print(mixed)




