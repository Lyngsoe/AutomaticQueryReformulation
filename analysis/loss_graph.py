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
#base_path = "/home/jonas/data/squad/cluster_exp/"
base_path = "/home/jonas/data/25_11_19/"
exps = [
    #(base_path + "transformer_medium/", "Transformer Medium"),
    #(base_path + "transformer_large/", "Transformer Large"),
    #(base_path + "transformer_small/", "Transformer Small"),
    #(base_path + "transformer_small_lr/", "Transformer Small lr+"),
    #(base_path + "transformer_medium_lr/", "Transformer Medium lr+"),
    (base_path + "adam_trans_medium/", "Transformer Medium"),
    (base_path + "adam_trans_large/", "Transformer Large"),
    (base_path + "adam_trans_small/", "Transformer Small"),
    #(base_path+"Transformer__11-13_16:49/","Transformer Medium"),
    #(base_path+"Transformer__11-13_16:50/","Transformer Large"),
    #(base_path+"Transformer__11-13_21:03/","Transformer Small"),
    # (base_path+"LSTM_auto_encoder_1__11-13_09:18/","LSTM Medium"),
    # (base_path+"LSTM_auto_encoder_1__11-15_11:10/","LSTM Large"),
    # ("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_14:05/","LSTM Large"),
    # ("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_15:12/","LSTM Medium"),
    # ("/media/jonas/archive/master/data/squad2/cluster_exp/experiments/Transformer__11-11_15:26/","LSTM Small"),
]

for exp in exps:

    exp_path = exp[0]
    results = load_results(exp_path)

    plt.plot(results)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")


plt.title("Training loss with subword embedding input")
plt.legend([exp[1] for exp in exps])
plt.ylim(3.5,6.5)
plt.savefig("fig/transformer_train_loss.png")
#plt.savefig("fig/lstm_train_loss.png")
#plt.savefig("fig/trans_subwords_train_loss.png")

plt.show()