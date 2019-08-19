from mediawiki import MediaWiki
from data_prep.utils import get_wikidata_id

class BFS:
    def __init__(self,goal_dict,language,max_iter=1000):
        self.explored = []
        self.frontier = []
        self.max_iter = max_iter
        self.goals = goal_dict
        self.language = language

        self.wikipedia = MediaWiki(lang="da")

    def search(self,start_node):

        #print("starting BFS, max {} iterations".format(self.max_iter))

        self.frontier.append(start_node)

        i=0
        while i < self.max_iter:
            #print("iteration:",i)
            i+=1

            cur_node = self.frontier.pop(0)
            #print("cur_node:",cur_node)
            is_goal, topic = self.is_goal(cur_node)
            if is_goal:
                return topic

            cand_nodes = self.get_candidate_nodes(cur_node)

            for cn in cand_nodes:
                self.frontier.append(cn)

            self.explored.append(cur_node)
            #print("\n\n##############\n",self.frontier)

    def get_candidate_nodes(self,cur_node):

        cur_page = self.wikipedia.page(cur_node)
        #print(cur_page.html)
        cand_nodes = []
        for category in sorted(cur_page.categories):
            category_page = self.wikipedia.page(category)

            if category_page.title not in self.frontier and category_page.title not in self.explored:
                #print("added_to_cand:",category_page.title)
                cand_nodes.append(category_page.title)

        return cand_nodes

    def is_goal(self,cur_node):
        try:
            wikidata_id = get_wikidata_id(cur_node,self.language)
            #print("wiki_id:",wikidata_id)
            topic = self.goals.get(wikidata_id,None)
        except:
            return False,None

        if topic is None:
            return False,None
        else:
            return True,topic