import jsonlines
import matplotlib.pyplot as plt
import analysis.plot_paths as plot_spec

def load_results(path):
    r = jsonlines.open(path+"results.jsonl")
    train_loss = []
    test_loss = []
    epoch = -1
    for line in r:
        e = line["epoch"]
        if e > epoch:
            epoch+=1
            train_loss.append(line["train_loss"])
            test_loss.append(line["test_loss"])

    return train_loss,test_loss


def create_plot(exp_cur):
    fig = plt.figure()
    st = fig.suptitle(exp_cur[0], fontsize="x-large")
    plt_train = fig.add_subplot(121)
    #plt.ylim(4,10)
    plt_test = fig.add_subplot(122,sharey=plt_train)


    for exp in exp_cur[1:]:

        exp_path = exp[0]
        train_loss,test_loss = load_results(exp_path)



        plt_train.set_title('Train loss')
        plt_train.set_xlabel('Train Iteration')
        plt_train.set_ylabel('Loss')


        plt_test.set_title('Test loss')
        plt_test.set_xlabel('Train Iteration')
        #plt_test.set_ylabel('Loss')


        plt_train.plot(train_loss)
        plt_test.plot(test_loss,label=exp[1])

    fig.legend(loc='center right',bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.7,0.95])
    fig.savefig("fig/{}.png".format(exp_cur[0].strip(" ")),bbox_inches = "tight")
    print("Saved: fig/{}.png".format(exp_cur[0].strip(" ")))



plots = [
    #plot_spec.exps_q2q
    plot_spec.exps_all_correct,
    plot_spec.exps_all_subwords_correct,
    #plot_spec.exps_lstm,
    #plot_spec.exps_attn_lstm,
    #plot_spec.exps_trans_subwords,
    #plot_spec.exps_attn_lstm_subwords,
    #plot_spec.exps_lstm_subwords,
]

for p in plots:
    create_plot(p)