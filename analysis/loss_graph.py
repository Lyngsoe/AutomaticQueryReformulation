import jsonlines
import matplotlib.pyplot as plt


def load_results(path):
    r = jsonlines.open(path+"results.jsonl")
    train_loss = []
    epoch = 0
    for line in r:
        train_loss.append(line["train_loss"])
        epoch+=1

    return train_loss

exps = [
    #("/media/jonas/archive/master/data/squad/cluster_exp/experiments/Transformer__11-13_16:49/","Transformer Medium"),
    #("/media/jonas/archive/master/data/squad/cluster_exp/experiments/Transformer__11-13_16:50/","Transformer Large"),
    #("/media/jonas/archive/master/data/squad/cluster_exp/experiments/Transformer__11-13_21:03/","Transformer Small"),
    ("/media/jonas/archive/master/data/squad/cluster_exp/experiments/LSTM_auto_encoder_1__11-13_09:18/","LSTM Medium"),
    ("/media/jonas/archive/master/data/squad/cluster_exp/experiments/LSTM_auto_encoder_1__11-15_11:10/","LSTM Large"),
    #("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_14:05/","LSTM Large"),
    #("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_15:12/","LSTM Medium"),
    #("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_15:26/","LSTM Small"),
]

for exp in exps:

    exp_path = exp[0]
    results = load_results(exp_path)

    plt.plot(results)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")


plt.title("Training loss with subword embedding input")
plt.legend([exp[1] for exp in exps])
#plt.savefig("fig/transformer_train_loss.png")
plt.savefig("fig/lstm_train_loss.png")
#plt.savefig("fig/trans_subwords_train_loss.png")
plt.show()