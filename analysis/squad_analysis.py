import json
import matplotlib.pyplot as plt
import numpy as np
import jsonlines

base_path = "/media/jonas/archive/master/data/squad/"

ctl = json.load(open(base_path+"context_token_length.json",'r'))
qtl = json.load(open(base_path+"question_token_length.json",'r'))
cwl = json.load(open(base_path+"context_word_length.json",'r'))


print("max_token_length: {}".format(np.mean(ctl)))
print("max_token_length: {}".format(np.mean(qtl)))
print("max_token_length: {}".format(np.mean(cwl)))

print(len(qtl))
fig = plt.figure()
st = fig.suptitle("SQuAD length in #subwords", fontsize="x-large")
plt_paragraph = fig.add_subplot(211)
#plt.ylim(4,10)
plt_question = fig.add_subplot(212,sharey=plt_paragraph,sharex=plt_paragraph)


plt_paragraph.set_title('Paragraphs')
plt_paragraph.set_xlabel('Length')
plt_paragraph.set_ylabel('#Paragraphs')

plt_question.set_title('Questions')
plt_question.set_xlabel('Length')
plt_question.set_ylabel('#Questions')

plt_paragraph.hist(cwl,bins=50)
plt_question.hist(qtl,bins=20)


fig.set_figheight(10)
fig.set_figwidth(11)
#fig.legend(loc='lower right',bbox_to_anchor=(0.8, 0.2))
#plt.subplots_adjust(wspace=0.6,hspace=0.6)

#plt.tight_layout(rect=[0,0,0.7,0.95])
fig.savefig("fig/squadlength.png",bbox_inches = "tight")

