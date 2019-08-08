para2wiki,wiki2para = dataset.load_annotation()

recall10 = []
for query in tqdm(queries,desc="calc recall"):
    retrieved_doc = set()
    for para in query.get("results"):
        if para["rank"] < 11:
            retrieved_doc.add(para["para_id"])

    relevant_docs = wiki2para.get(query["wiki_id"])

    intersec = retrieved_doc.intersection(retrieved_doc)
    recall10.append(len(intersec)/10)


print("recall 10:",np.mean(recall10))

