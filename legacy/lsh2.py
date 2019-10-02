from datasketch import MinHash,MinHashLSH
import copy
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def delete_paras_from_paragraphs(paras_to_delete,paragraphs):
    for p in paras_to_delete:
        del_idx = None
        for i,para in enumerate(paragraphs):
            if para["id"] == p[1]:
                del_idx = i
                break
        if del_idx is not None:
            paragraphs.pop(del_idx)
    return paragraphs

def delete_para_from_annotations(paras_to_delete, annotations):
    new_annotations = {}
    for wiki,paras in annotations.items():
        for p in paras_to_delete:
            for para_id in paras:
                delete_id = p[1]
                if para_id == delete_id:
                    new_paras = copy.deepcopy(paras)
                    new_paras.remove(para_id)
                    new_paras.append(p[0])
                    paras=new_paras
        new_annotations.update({wiki: paras})

    return new_annotations

def create_min_hash(embedding,id,num_perm):
    mh = MinHash(num_perm=num_perm)
    #for emb in embedding:
        #mh.update(emb)
    mh.update(embedding)
    return (id,mh)


def find_duplicates(embedder):
    min_hashes = []
    num_perm = 128
    number_of_cpus = multiprocessing.cpu_count()

    with tqdm(total=len(embedder.para2emb.keys()), desc="min hashing paragraphs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus) as executor:
            future_to_min_hash = {
                executor.submit(create_min_hash, embedder(para_id), para_id, num_perm=num_perm): (para_id, fp) for
                para_id, fp in embedder.para2emb.items()}
            for future in concurrent.futures.as_completed(future_to_min_hash):
                results = future_to_min_hash[future]
                try:
                    min_hashes.append(future.result())
                except Exception as exc:
                    tqdm.write('%r generated an exception: %s' % (results, exc))
                    raise exc

                pbar.update()
        pbar.close()

    lsh = MinHashLSH(threshold=0.90, num_perm=num_perm)

    for id, mh in tqdm(min_hashes, desc="adding MinHash to LSH index"):
        if id not in lsh.keys:
            lsh.insert("{}".format(id), mh)

    paras_to_delete = []
    for id, mh in tqdm(min_hashes, desc="finding similar paragraphs"):
        result = lsh.query(mh)
        cur_para = id
        try:
            result.remove(id)
        except ValueError:
            pass

        if len(result) > 0:
            for res in result:
                res_para = res
                if (cur_para, res_para) not in paras_to_delete and (res_para, cur_para) not in paras_to_delete:
                    # print("replace:",res_para,"with:",cur_para)
                    paras_to_delete.append((cur_para, res_para))


    return paras_to_delete