from data_prep.wikipedia_parsing.bfs_engine import BFS


def get_topic(title,language):
    categories_id = {'Q6642719': 'Category:Academic_disciplines', 'Q6353120': 'Category:Business', 'Q5550686': 'Category:Concepts', 'Q6478924': 'Category:Crime', 'Q2944929': 'Category:Culture', 'Q9715089': 'Category:Economy', 'Q4103249': 'Category:Education', 'Q8413436': 'Category:Energy', 'Q6337045': 'Category:Entertainment', 'Q7214908': 'Category:Events', 'Q5645580': 'Category:Food_and_drink', 'Q1457673': 'Category:Geography', 'Q54070': 'Category:Government', 'Q7486603': 'Category:Health', 'Q1457595': 'Category:History', 'Q6697416': 'Category:Human_behavior', 'Q6172603': 'Category:Humanities', 'Q2945448': 'Category:Knowledge', 'Q1458484': 'Category:Language', 'Q4026563': 'Category:Law', 'Q5550747': 'Category:Life', 'Q4619': 'Category:Mathematics', 'Q5850187': 'Category:Military', 'Q6643238': 'Category:Mind', 'Q8255': 'Category:Music', 'Q4049293': 'Category:Nature', 'Q6576895': 'Category:Objects', 'Q5613113': 'Category:Organizations', 'Q4047087': 'Category:People', 'Q1983674': 'Category:Philosophy', 'Q4103183': 'Category:Politics', 'Q1457903': 'Category:Religion', 'Q1458083': 'Category:Science', 'Q1457756': 'Category:Society', 'Q1457982': 'Category:Sports', 'Q4884546': 'Category:Technology', 'Q52075235': 'Category:Universe', 'Q7386634': 'Category:World'}


    bfs = BFS(categories_id,language)

    topic = bfs.search(title)
    print(topic," : ",title)
    return topic