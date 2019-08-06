from lupyne import engine

indexer = engine.Indexer(directory="index")                                  # Indexer combines Writer and Searcher; RAMDirectory and StandardAnalyzer are defaults
indexer.set('fieldname', engine.Field.Text, stored=True)    # default indexed text settings for documents
indexer.add(fieldname="hej")                                 # add document
indexer.commit()                                            # commit changes and refresh searcher

# Now search the index:
hits = indexer.search('hej', field='fieldname')    # parsing handled if necessary
assert len(hits) == 1
for hit in hits:                                    # hits support mapping interface
    print(hit)
    assert hit['fieldname'] == "hej"
# closing is handled automatically