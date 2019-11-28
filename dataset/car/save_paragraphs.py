from trec_car.read_data import iter_paragraphs
from trec_car.read_data import ParaText
import jsonlines
from tqdm import tqdm

debug = False
base_path = "/media/jonas/archive/master/data/car/"
save_path = base_path+"my_car/"
base_path += "test200-train/" if debug else "train/"
para_path = base_path+"base.train.cbor-paragraphs.cbor"
qa_writer = jsonlines.open(save_path + "paragraphs.jsonl", 'w', flush=True)
pbar = tqdm(desc="writing paragraphs")
with open(para_path,'rb') as f:

    for p in iter_paragraphs(f):

        texts = [elem.text if isinstance(elem, ParaText)
                 else elem.anchor_text
                 for elem in p.bodies]
        para = {"id":p.para_id,"text":' '.join(texts)}
        qa_writer.write(para)
        pbar.update()

pbar.close()