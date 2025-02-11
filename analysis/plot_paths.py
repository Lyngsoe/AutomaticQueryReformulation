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
    "Transformer Small vs Large 1 layer",
    (base_path + "Transformer__12-08_09:39/", "Large"),
    (base_path + "Transformer__12-08_09:45/", "Small"),
]

exps_6_layer = [
    "Transformer 6 layer",
    (base_path + "Transformer__12-06_15:19/", "0.2 dropout"),
    (base_path + "Transformer__12-06_15:20/", "0.5 dropout"),
]


## attn lstm
exps_attn = [
    "Attn lstm vs Transformer",
    (base_path + "LSTM_attn__12-09_18:04/", "Attn LSTM"),
    ("/media/jonas/archive/master/data/squad/cluster_exp/09_12_19/experiments/" + "Transformer__12-06_15:19/", "Transformer Large"),
]

base_path = "/media/jonas/archive/master/data/squad/cluster_exp/10_12_19/experiments/"
##  transformer with 1 layer
exps_1_layer_trans = [
    "Transformer 1 layer",
    (base_path + "Transformer__12-09_11:27/", "Small"),
    (base_path + "Transformer__12-09_11:28/", "Medium"),
    (base_path + "Transformer__12-09_11:30/", "Large"),
]

base_path = "/media/jonas/archive/master/data/squad/cluster_exp/12_12_19/experiments/"
##  all exps
exps_trans = [
    "Question Generation by Transformer with BERT embeddings",
    (base_path + "Transformer__12-12_16:59/", "Transformer Large"),
    (base_path + "Transformer__12-12_17:00/", "Transformer Medium"),
    (base_path + "Transformer__12-12_17:01/", "Transformer Small"),
]
base_path = "/media/jonas/archive/master/data/squad/cluster_exp/12_12_19/experiments/"
##  all exps
exps_lstm = [
    "Question Generation by LSTM with BERT embeddings",
    (base_path + "LSTM__12-11_01:34/", "LSTM Large"),
    (base_path + "LSTM__12-11_01:36/", "LSTM Medium"),
    (base_path + "LSTM__12-11_01:37/", "LSTM Small"),
]
base_path = "/media/jonas/archive/master/data/squad/cluster_exp/12_12_19/experiments/"
##  all exps
exps_attn_lstm = [
    "Question Generation by LSTM Attention with BERT embeddings",
    (base_path + "LSTM_attn__12-11_01:27/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-11_01:29/", "LSTM Attn Medium"),
    (base_path + "LSTM_attn__12-11_01:31/", "LSTM Attn Small"),
]


base_path = "/media/jonas/archive/master/data/squad2/cluster_exp/12_12_19/experiments/"
##  all exps
exps_trans_subwords = [
    "Question Generation by Transformer with Wordpiece indices",
    (base_path + "Transformer__12-12_16:54/", "Transformer Large"),
    (base_path + "Transformer__12-12_16:55/", "Transformer Medium"),
    (base_path + "Transformer__12-12_16:56/", "Transformer Small"),
]

base_path = "/media/jonas/archive/master/data/squad2/cluster_exp/12_12_19/experiments/"
##  all exps
exps_lstm_subwords = [
    "Question Generation by LSTM with Wordpiece indices",
    (base_path + "LSTM__12-11_13:25/", "LSTM Large"),
    (base_path + "LSTM__12-11_13:27/", "LSTM Medium"),
    (base_path + "LSTM__12-11_13:28/", "LSTM Small"),
]

base_path = "/media/jonas/archive/master/data/squad2/cluster_exp/12_12_19/experiments/"
##  all exps
exps_attn_lstm_subwords = [
    "Question Generation by LSTM Attention with Wordpiece indices",
    #(base_path + "LSTM_attn__12-11_13:32/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-11_13:34/", "LSTM Attn Medium"),
    (base_path + "LSTM_attn__12-11_13:39/", "LSTM Attn Small"),
]



base_path = "/media/jonas/archive/master/data/squad/cluster_exp/12_12_19/experiments/"
##  all exps
exps_all = [
    "Question Generation with BERT embeddings",
    (base_path + "LSTM__12-11_01:34/", "LSTM Large"),
    (base_path + "LSTM__12-11_01:36/", "LSTM Medium"),
    (base_path + "LSTM__12-11_01:37/", "LSTM Small"),
    (base_path + "LSTM_attn__12-11_01:27/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-11_01:29/", "LSTM Attn Medium"),
    (base_path + "LSTM_attn__12-11_01:31/", "LSTM Attn Small"),
    (base_path + "Transformer__12-12_08:59/", "Transformer Large"),
    (base_path + "Transformer__12-12_09:00/", "Transformer Medium"),
    (base_path + "Transformer__12-12_09:01/", "Transformer Small"),
]

base_path = "/media/jonas/archive/master/data/squad2/cluster_exp/12_12_19/"
##  all exps
exps_all_subwords = [
    "Question Generation with Wordpiece indices",
    (base_path + "LSTM__12-11_13:25/", "LSTM Large"),
    (base_path + "LSTM__12-11_13:27/", "LSTM Medium"),
    (base_path + "LSTM__12-11_13:28/", "LSTM Small"),
    (base_path + "LSTM_attn__12-11_13:32/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-11_13:34/", "LSTM Attn Medium"),
    (base_path + "LSTM_attn__12-11_13:39/", "LSTM Attn Small"),
    (base_path + "Transformer__12-12_09:06/", "Transformer Large"),
    (base_path + "Transformer__12-12_09:07/", "Transformer Medium"),
    (base_path + "Transformer__12-12_09:08/", "Transformer Small"),
]


base_path = "/media/jonas/archive/master/data/squad/cluster_exp/15_12_19/experiments/"
##  all exps
exps_small_trans_drop = [
    "Transformer small",
    (base_path + "Transformer__12-15_16:42/", "0.3 dropout"),
    (base_path + "Transformer__12-15_16:43/", "0.4 dropout"),
    (base_path + "Transformer__12-15_16:44/", "0.5 dropout"),
]

base_path = "/media/jonas/archive/master/data/squad/cluster_exp/17_12_19/experiments/"
##  all exps
exps_all_correct = [
    "Question Generation with BERT embeddings",
    (base_path + "Transformer__12-16_17:42/", "Transformer Large"),
    (base_path + "Transformer__12-16_17:44/", "Transformer Medium"),
    (base_path + "Transformer__12-16_17:45/", "Transformer Small"),
    (base_path + "LSTM__12-16_21:43/", "LSTM Large"),
    (base_path + "LSTM__12-16_21:49/", "LSTM Small"),
    (base_path + "LSTM_attn__12-16_21:46/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-16_21:51/", "LSTM Attn Small"),
]

base_path = "/media/jonas/archive/master/data/squad/cluster_exp/17_12_19/experiments/"
##  all exps
exps_all_subwords_correct = [
    "Question Generation with Wordpiece indices",
    (base_path + "Transformer__12-16_22:15/", "Transformer Large"),
    (base_path + "Transformer__12-16_22:17/", "Transformer Medium"),
    (base_path + "LSTM__12-17_08:03/", "LSTM Large"),
    (base_path + "LSTM__12-17_08:06/", "LSTM Small"),
    (base_path + "LSTM_attn__12-17_08:40/", "LSTM Attn Large"),
    (base_path + "LSTM_attn__12-17_08:41/", "LSTM Attn Small"),
]

base_path = "/media/jonas/archive/master/data/rl_squad/experiments/"
##  all exps
exps_q2q = [
    "Question to Question with Bert embeddings",
    (base_path + "Transformer_q2q_medium/", "Transformer Medium"),
]