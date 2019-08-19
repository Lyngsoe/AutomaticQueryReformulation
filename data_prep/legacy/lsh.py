from laserembeddings import Laser
from datasketch import MinHash,MinHashLSH
import copy

def delete_paras_from_paragraphs(paras_to_delete,paragraphs):
    for p in paras_to_delete:
        paragraphs.remove(p[1])
    return paragraphs

def delete_para_from_annotations(paras_to_delete, annotations):
    new_annotations = {}
    for wiki,paras in annotations.items():
        for p in paras_to_delete:
            for para in paras:

                if para["id"] == p[1]["id"]:
                    para["id"] = p[0]["id"]
                    print("#################")
                    print(p[0])
                    print(p[1])
            new_annotations.update({wiki:paras})

    return new_annotations

def remove_duplicates(paragraphs,annotations):
    print("loading laser")
    laser = Laser()

    min_hashes = []
    num_perm = int(len(paragraphs)/2)
    lsh = MinHashLSH(threshold=0.5,num_perm=num_perm)

    print("embedding paragraphs")
    for i,p in enumerate(paragraphs):
        mh = MinHash(num_perm=num_perm)

        embeddings = laser.embed_sentences([p["text"]], lang='da')
        for emb in embeddings:
            mh.update(emb)

        min_hashes.append(mh)
        lsh.insert("{}".format(i), mh)

    paras_to_delete = []

    for i,mh in enumerate(min_hashes):
        result = lsh.query(mh)
        cur_para = paragraphs[i]
        try:
            result.remove(str(i))
        except ValueError:
            pass

        if len(result) > 0:
            for res in result:
                res_para = paragraphs[int(res)]
                if (cur_para,res_para) not in paras_to_delete and (res_para,cur_para) not in paras_to_delete:
                    print("replace:",res_para,"with:",cur_para)
                    paras_to_delete.append((cur_para,res_para))

    paragraphs = delete_paras_from_paragraphs(paras_to_delete,paragraphs)
    annotations = delete_para_from_annotations(paras_to_delete,annotations)

    return paragraphs,annotations
