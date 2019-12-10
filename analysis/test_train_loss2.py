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

#base_path = "/media/jonas/archive/master/data/squad/cluster_exp/09_12_19/experiments/"
#base_path = "/home/jonas/data/25_11_19/"
base_path = "/media/jonas/archive/master/data/squad/experiments/"

## Subsample data small transformer with 1,3,6 layers
exps_sub_1 = [
    (base_path + "Transformer__12-07_16:47/", "1 layer"),
    (base_path + "Transformer__12-07_23:48/", "3 layers"),
    (base_path + "Transformer__12-07_15:00/", "6 layers"),
]

## small vs large 1 layer
exps_1_layer = [
    (base_path + "Transformer__12-08_09:39/", "Large"),
    (base_path + "Transformer__12-08_09:45/", "Small"),
]

## small vs large 1 layer
exps_6_layer = [
    (base_path + "Transformer__12-06_15:19/", "0.2 dropout"),
    (base_path + "Transformer__12-06_15:20/", "0.5 dropout"),
]


## attn lstm
exps_attn = [
    (base_path + "LSTM_attn__12-09_18:04/", "Attn LSTM"),
    ("/media/jonas/archive/master/data/squad/cluster_exp/09_12_19/experiments/" + "Transformer__12-06_15:19/", "Transformer Large"),
]

max_epochs=250

fig = plt.figure()
st = fig.suptitle("Transformer vs LSTM", fontsize="x-large")
plt_train = fig.add_subplot(121)
plt_test = fig.add_subplot(122,sharey = plt_train)


for exp in exps_attn:

    exp_path = exp[0]
    train_loss,test_loss = load_results(exp_path)



    plt_train.set_title('Train loss')
    plt_train.set_xlabel('Train iteration')
    plt_train.set_ylabel('Loss')


    plt_test.set_title('Test loss')
    plt_test.set_xlabel('Train iteration')
    #plt_test.set_ylabel('Loss')


    plt_train.plot(train_loss[:max_epochs])
    plt_test.plot(test_loss[:max_epochs],label=exp[1])

fig.legend(loc='center right',bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0,0,0.8,0.95])
fig.savefig("fig/transformer_test_train.png",bbox_inches = "tight")

plt.show()
