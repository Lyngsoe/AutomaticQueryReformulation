from embedding_method.laser import LaserSentenceEmbeddings
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
    for emb in embedding:
        mh.update(emb)
    return (id,mh)

def remove_duplicates(paragraphs,annotations,language="da"):
    laser = LaserSentenceEmbeddings(max_batch_size=10)
    embeddings = laser([para["text"] for para in paragraphs], method='sentence', language=language)
    min_hashes = []
    num_perm = int(len(paragraphs)/2)
    number_of_cpus = multiprocessing.cpu_count()

    with tqdm(total=len(paragraphs), desc="min hashing paragraphs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus) as executor:
            future_to_min_hash = {executor.submit(create_min_hash,embedding,paragraphs[i]["id"],num_perm=num_perm): (i,embedding) for i,embedding in enumerate(embeddings)}
            for future in concurrent.futures.as_completed(future_to_min_hash):
                results = future_to_min_hash[future]
                try:
                    min_hashes.append(future.result())
                except Exception as exc:
                    print('%r generated an exception: %s' % (results, exc))

                pbar.update(1)

    lsh = MinHashLSH(threshold=0.95,num_perm=num_perm)

    for id,mh in tqdm(min_hashes,desc="adding MinHash to LSH index"):
        lsh.insert("{}".format(id), mh)

    paras_to_delete = []

    for id,mh in tqdm(min_hashes,desc="finding similar paragraphs"):
        result = lsh.query(mh)
        cur_para = id
        try:
            result.remove(id)
        except ValueError:
            pass


        if len(result) > 0:
            for res in result:
                res_para = res
                if (cur_para,res_para) not in paras_to_delete and (res_para,cur_para) not in paras_to_delete:
                    print("replace:",res_para,"with:",cur_para)
                    paras_to_delete.append((cur_para,res_para))


    paragraphs = delete_paras_from_paragraphs(paras_to_delete,paragraphs)
    annotations = delete_para_from_annotations(paras_to_delete,annotations)

    return paragraphs,annotations
