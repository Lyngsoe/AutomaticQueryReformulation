import jsonlines
import matplotlib.pyplot as plt
import analysis.plot_paths as plot_spec
import numpy as np
import time

def load_results(path):
    r = jsonlines.open(path+"norm_reward.jsonl")
    base_reward = []
    reward = []
    batch_size = 100
    base_reward_temp = []
    reward_temp = []
    for line in r:
        base_reward_temp.append(line["base_reward"])
        reward_temp.append(line["reward"])
        if len(base_reward_temp) >= batch_size:
            base_reward.append(np.mean(base_reward_temp))
            base_reward_temp = []
            reward.append(np.mean(reward_temp))
            reward_temp = []


    test_reward = []
    sentences = []
    batch_size = 10
    base_reward_temp = []
    r = jsonlines.open(path + "results.jsonl")
    for line in r:
        base_reward_temp.append(line["reward"])
        if len(base_reward_temp) >= batch_size:
            #sentences.append(line["sentence"])
            test_reward.append(np.mean(base_reward_temp))
            base_reward_temp = []
    return base_reward,reward,test_reward,sentences


def create_plot(exp_cur):
    fig = plt.figure()
    base_line_train = exp_cur[0][1]
    base_line_test = exp_cur[0][2]
    st = fig.suptitle(exp_cur[0][0], fontsize="x-large")
    plt_base_reward = fig.add_subplot(221)
    #plt.ylim(4,10)
    plt_test = fig.add_subplot(222,sharey=plt_base_reward)
    plt_reward = fig.add_subplot(223,sharey=plt_base_reward)

    max_len = 0
    max_len_test = 0

    for exp in exp_cur[1:]:

        exp_path = exp[0]
        base_reward,reward,test_reward,sentences = load_results(exp_path)



        plt_base_reward.set_title('Train base reward')
        plt_base_reward.set_xlabel('Train Iteration')
        plt_base_reward.set_ylabel('Reward')

        plt_reward.set_title('Train reward')
        plt_reward.set_xlabel('Train Iteration')

        plt_test.set_title('Test reward')
        plt_test.set_xlabel('Train Iteration')
        #plt_test.set_ylabel('Loss')

        if len(reward) > max_len:
            max_len = len(reward)
        if len(test_reward) > max_len_test:
            max_len_test = len(test_reward)

        plt_base_reward.plot(base_reward)
        plt_reward.plot(reward)
        plt_test.plot(test_reward,label=exp[1])

    base_line = np.array([base_line_train for i in range(max_len)])
    plt_reward.plot(base_line)
    plt_base_reward.plot(base_line)

    base_line = np.array([base_line_test for i in range(max_len_test)])
    plt_test.plot(base_line)

    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.legend(loc='lower right',bbox_to_anchor=(0.8, 0.2))
    #plt.subplots_adjust(wspace=0.6,hspace=0.6)

    #plt.tight_layout(rect=[0,0,0.7,0.95])
    fig.savefig("fig/{}.png".format(exp_cur[0][0].strip(" ")),bbox_inches = "tight")
    print("Saved: fig/{}.png".format(exp_cur[0][0].strip(" ")))


base_path = "/media/jonas/archive/master/data/rl_squad_sub/experiments/"

rank_rl_one_question = [
    ("Rank Reward: Who else appeared with Beyonce in Telephone?",0.333),
    (base_path + "Transformer__12-18_16:02/", "Dropout 0.2"),
    (base_path + "Transformer__12-18_16:03/", "Dropout 0.3"),
]

recall_rl_one_question = [
    ("Recall Reward: Who else appeared with Beyonce in Telephone?",0.0606),
    (base_path + "Transformer__12-18_18:41/", "Dropout 0.2 - debut else appeared with beyonce in telephone ?"),
    (base_path + "Transformer__12-18_18:42/", "Dropout 0.3 - who else performed with beyonce in telephone ?"),
]



base_path = "/media/jonas/archive/master/data/rl_squad/experiments/"
recall_rl = [
    ("Recall Reward Q2Q Transformer",0.1867,0.2205),
    ("/media/jonas/archive/master/data/rl_squad/cluster_exp/19_12_19/experiments/" + "RL_Transformer__12-18_21:12/", "Moving Avg"),
    (base_path + "Transformer__12-19_15:11/", "Moving Avg per Question"),
]

base_path = "/media/jonas/archive/master/data/rl_squad/experiments/"
recall_one_question = [
    ("Recall Reward One Question",0.015,0.015),
    (base_path + "Transformer__12-19_16:50/", "Dropout 0.2"),
    (base_path + "Transformer__12-19_21:48/", "Dropout 0.3"),
    (base_path + "Transformer__12-20_02:47/", "Dropout 0.4"),
]



base_path = "/media/jonas/archive/master/data/rl_squad/experiments/"
Rank_one_question = [
    ("Rank Reward One Question",0.047619,0.047619),
    (base_path + "Transformer__12-20_07:26/", "Dropout 0.2"),
]


plots = [
    recall_one_question
]

live = True

while live:
    for p in plots:
        create_plot(p)
    time.sleep(60)

for p in plots:
    create_plot(p)