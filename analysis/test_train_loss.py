import jsonlines
import matplotlib.pyplot as plt


def load_results(path):
    r = jsonlines.open(path+"results.jsonl")
    train_loss = []
    test_loss = []
    for line in r:
        train_loss.append(line["train_loss"])
        test_loss.append(line["test_loss"])

    return train_loss,test_loss

#base_path = "/home/jonas/data/squad/cluster_exp/"
#base_path = "/home/jonas/data/25_11_19/"
base_path = "/media/jonas/archive/master/data/squad/experiments/Transformer__12-06_15:14/"


train_loss,test_loss = load_results(base_path)

plt.plot(train_loss)
plt.plot(test_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.title("Transformer loss for subsample dataset")
plt.legend(["Train","Test"])
#plt.ylim(0,1)
plt.savefig("fig/transformer_sub_test_train.png")
#plt.savefig("fig/lstm_train_loss.png")
#plt.savefig("fig/trans_subwords_train_loss.png")

plt.show()