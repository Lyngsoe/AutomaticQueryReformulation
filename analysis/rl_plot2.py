import jsonlines
import matplotlib.pyplot as plt
import analysis.plot_paths as plot_spec
import numpy as np
import time

def load_results(path):
    r = jsonlines.open(path+"norm_reward.jsonl")
    norm_rewards = []
    base_reward = []
    reward = []
    batch_size = 30
    norm_rewards_temp = []
    base_reward_temp = []
    reward_temp = []
    for line in r:
        norm_rewards_temp.append(line["reward"] - line["base_reward"])
        base_reward_temp.append(line["base_reward"])
        reward_temp.append(line["reward"])
        if len(base_reward_temp) >= batch_size:
            norm_rewards.append(np.mean(norm_rewards_temp))
            norm_rewards_temp = []
            base_reward.append(np.mean(base_reward_temp))
            base_reward_temp = []
            reward.append(np.mean(reward_temp))
            reward_temp = []
    return norm_rewards,base_reward,reward


def create_plot(exp_cur):
    fig = plt.figure()
    st = fig.suptitle(exp_cur[0], fontsize="x-large")
    plt_base_reward = fig.add_subplot(221)
    #plt.ylim(4,10)
    plt_normalized = fig.add_subplot(222,sharey=plt_base_reward)
    plt_reward = fig.add_subplot(223,sharey=plt_base_reward)

    max_len = 0

    for exp in exp_cur[1:]:

        exp_path = exp[0]
        norm_rewards,base_reward,reward = load_results(exp_path)



        plt_base_reward.set_title('Train base reward')
        plt_base_reward.set_xlabel('Train Iteration')
        plt_base_reward.set_ylabel('Reward')

        plt_reward.set_title('Train reward')
        plt_reward.set_xlabel('Train Iteration')

        plt_normalized.set_title('Normalized train reward')
        plt_normalized.set_xlabel('Train Iteration')
        #plt_test.set_ylabel('Loss')

        if len(reward) > max_len:
            max_len = len(reward)

        plt_base_reward.plot(base_reward)
        plt_reward.plot(reward)
        plt_normalized.plot(norm_rewards,label=exp[1])

    horiz_line_data = np.array([0.0606 for i in range(max_len)])
    plt_reward.plot(horiz_line_data)

    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.legend(loc='lower right',bbox_to_anchor=(0.8, 0.2))
    #plt.subplots_adjust(wspace=0.6,hspace=0.6)

    #plt.tight_layout(rect=[0,0,0.7,0.95])
    fig.savefig("fig/{}.png".format(exp_cur[0].strip(" ")),bbox_inches = "tight")
    print("Saved: fig/{}.png".format(exp_cur[0].strip(" ")))


base_path = "/media/jonas/archive/master/data/rl_squad_sub/experiments/"
exps_rl_amw = [
    "Reinforcement learning with subset of dataset",
    (base_path + "Transformer__12-17_10:22/", "MovingAvg 0 start"),
    (base_path + "Transformer__12-17_12:24/", "MovingAvg q0 start"),
    (base_path + "Transformer__12-17_13:57/", "MovingAvg avg start"),
    #(base_path + "Transformer__12-17_15:07/", "MovingAvg reverse"),

]

exps_rl_1_percent = [
    "Reinforcement learning 1% question",
    (base_path + "Transformer__12-17_15:49/", "MovingAvg 1% batch size 8"),
]

exps_rl_one_question_old = [
    "old Reinforcement learning Question: Who else appeared with Beyonce in Telephone?",
    (base_path + "Transformer__12-17_20:02/", "lr 10e-7, Sample size 1"),
    (base_path + "Transformer__12-17_20:15/", "lr 10e-5, Sample size 1"),
    (base_path + "Transformer__12-17_20:38/", "lr 10e-7, Sample size 8"),
    (base_path + "Transformer__12-17_20:39/", "lr 10e-5, Sample size 8"),
    (base_path + "Transformer__12-17_22:12/", "lr 10e-7, Sample size 1 drop 0.3"),
    (base_path + "Transformer__12-17_22:13/", "lr 10e-7, Sample size 1 drop 0.2"),
]

exps_rl_one_question = [
    "Reinforcement learning Question: Who else appeared with Beyonce in Telephone?",
    (base_path + "Transformer__12-17_22:13/", "Dropout 0.2"),
    (base_path + "Transformer__12-17_22:12/", "Dropout 0.3"),
    (base_path + "Transformer__12-18_07:30/", "Dropout 0.4"),
    (base_path + "Transformer__12-18_07:31/", "Dropout 0.5"),
]



plots = [
    #exps_rl_amw,
    #exps_rl_1_percent,
    #exps_rl_one_question_old,
    exps_rl_one_question
]

live = True

while live:
    for p in plots:
        create_plot(p)
    time.sleep(30)

for p in plots:
    create_plot(p)