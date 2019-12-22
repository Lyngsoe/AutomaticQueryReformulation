from training.rl_trainer import Trainer
from models.rl_transformer import RLTransformer
from models.losses.recall_reward import RecallRewardMean
from models.losses.rank_loss import RankReward
from models.baselines.moving_avg import MovingAverage,MovingStdAverage
from search_engine.elastic_search import ELSearch
import torch
from tqdm import tqdm
import numpy as np
from dataset.bertify import construct_sentence,prune
import json
import os
import jsonlines
from embedding_method.embedders import get_embedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


documents = ["bdcf574de33f3ab82c6faf4c5d425aa2583c331399e7f4b35f3f2affafc67717", "8c9f92f94f88c4594d053cb7d7e49e745556947bb851661ae2f5cdbeba3ef111", "96b553fb15cb067b2513b4571d5a13def34a0d84a89873bda4e5da539d650b4f", "fcba1d4c927d3b966fd9db2bc9270f6beea8b972a02588db817344d3f02a7dd3", "b39c2185f74e94c998b531b14b6c42b776fdf5b027aab935d01795dd0c7972a3", "97e74af5c815eb104b965c2c79ca64e07fc8285b515e4e73a88dfb233b08c7eb", "87dd2933b8bcbb78d09713e5bd64a81b78feb9e6e8aeb727a91bcfd64ac84ed5", "88a9a61282aade49153f083b202265615fba375a9afc44d3a0aa96a9d712e5e1", "f4ee2661e33e2d61eff8bda862313da9790a7a55d9bdb4f9ea1b28f8c7a8d901", "138bc1985a4036e4876a4aa4d9c69a526032cd745cd0d96d457d18321777b8ff", "5cab936f91fb1456427db35a4bc02131d1f230c5ff6743a12f9312be05707a48", "4368aba3e0568d25b377961448fb8d564d32627ba8a9d0a10b48c7fcebce85b2", "0a92fe3684a256b98b404d6a26cfcc1c1579d25c97a5ac1b7b16c7311a5bfdd7", "f3ae90edead8ac7e6bbdc888afd0910ec4de17bdad57b89d5cacfb7d069b31bf", "7b9cbf99c6f57a0940b6016a5f0349d87e4744faeb08beb4b6069a9d3d667b59", "ef15329a5908e7bc0270d3eaf19960bef7032064e3b12070c0c673c83c91e0c3", "b83d1acd3e7b207926855f71e14faafd15177b1cb80576a80a642694b8e74f28", "32e80164efcba6deebe31173e2ceca17e46a1edd86278f839b24c556451523d5", "0b51d3394e619caefcc222ccf88807bc277f583fb5b585fdec80770ccbe992f3", "80c98e5f46506caad15a7bc8ad32c037b0610371b0ce80adb3a590e95ea9f9e5", "7032bde5b1b0a6b89acd9b3a468c3567a8bb982455530dd15c8c0f5a0d87d353", "726940d16d126cbd987b9bae52510981cdc9a6a19cd25b7a976da06b2c418f7d", "e5ef1be8b0afa186c61428a2de484e9aa0bee228f9d0c9056dd4fbaace7052e8", "deb447591c67c3eb6da09a7b31d0cae7eef820c6cf26f7a4e5b37ce62d8f502a", "7894eeeb6f7a324bfc1afd270ead57be9ba08b2ee8fa058e8c708a9b2af554f6", "614757e4e1769380a26d90f9e202acc7e1cd192d9e3255cc23fce35f59765e1a", "7c727985b2c46f31827f71b0c5b96689f90c59685a567f6a2dd77a71e79596fa", "a026057a644238049255d8be60e84c90954a8163218d0b141ab2185a25682a8a", "3276bd7be5f5ed647ff05cb79649e11c183f4c26a72ea0ee45d710d6c7e25bca", "b73eaed9dc259508afe3a3dcd3b44cad0103b5dbb3d3d0d218278e2c3d0a2cfb", "8bf2954c890767bfb5abfd3b01435fac913c9436741fc18ebcfa1a706f68081f", "fc355711b10f0e2ff7d6e8d06662cdfab5499bb15c9b794af4d43181597e97e0", "97ed8dc8c051322eea8d52ee568f2019e71385bbe6e9ad3b6ce933a0a83e25d9", "a0b69b8b8f125d383a3121a799d64245e3884f34ebc43584b9f6c98c96b763d3", "1cf21201321bd9f0e962b6a0783dffd0baf6311c685797e7248c9bdf87d98a8e", "57ec1a313624d83628f48c90baba8f809eb79943ac7e10ea1abf20ac93e14b4b", "870b9babd24d3ac5ba1edbbc33ef4b26f48319bcdd6ad9088695a3407a0e8871", "d566dc3cecf17a7ad67613c5c23076ca606f6c82c1eade210778b34b66446ac4", "f14274f28aeacdb2acac9d9197ae3590eb2dde4f8f9a428b0c0ff73a2730d566", "4cc147d197e2876558808ae80b0f3163929c7615a6895e12ba1b0f54f7abfdaf", "b564992dda1804c2bfb856137184d2853a3524b08f0da5713b7cf901fd445968", "06082516706adc8ff8604c773c9b12c585ad970319f3c95a45e16f08534a4ecd", "46a497a5786dae9d2d3aececc24e43a94cc5aaa25cd7d661182db8a3660890cc", "e7f5458c064964f62181c1ec8d8255e3f53d53c3681ae3a540b0c115a9a59830", "940f9efb1e5f306fcc58ed4fb3b84253511dd7d090efaba4c8d64e8819437ef4", "fdd885806e8f22143114d15ec4783f97cb6d99d926eed82baa3b8ba0eda2da41", "01f183e1570dc892397947e9ec2d40ce1b32ff01e6798cf83eeb72d59b232948", "801a042035520ea02d639dbfd7925e9959802dda11bfa6e8066766d58083d1f2", "a71e716c32a41ab50763b821815e9025fbd96ba07960e14315129119e5889176", "fc7f2f826d63aa0a48289892e50233972eb0725cbcfc932d53aacb7492dc3e8f", "79d9503d29c50e86ceddbda6541c1d6be267677e1c949d78d3feacd6df32fd7f", "849bcd108582e0e973aad11a66d1db7a6fa50122eeb225cadcf4731fd2f18a98", "c383cc90c834be0ffd3de94d7a807303d036bb4a730d63b7ec567d28c5dc29a9", "ee5a66be6a48169e7a6b1fdd01b998f926bc3cea5a25410acc8725fe2ab5e0b7", "44899650ae50ac3935e70e8956c603e18aeaf2a74069f51cd71513ae36bb7375", "caa1264bdd7dae14329018a6d804937ea262dee3b5804e92921a23ddcf845a62", "55bf0546d88aceb5af02e6a1666734810d8c279115ad7bdfebb3107f5af78a95", "17300cbf34b3d694709815b058723d6e3202fd2b47e3e1478bd25d2a1cc631f9", "cc0f6633a7c5b0ad2fca012399fc4b7b54adf79903a95039504102bea96e4ef2", "94edd3cebe291511a247aa15b9469a5733423b365b5dd04c740a9c77a0cb2ddc", "6e33c066afe93c09153a66f962cc525b52a69cf76d4dbffd302bcfb1801fc1dc", "0a1250f33d84c6ff28235a5ef8eb598262aeea31eab4947ef4d046a2f17aaf97", "ec5ce01fcb4aa4296c870c07672fa4b8582907f2ced12c86a1cf9dad04c33a4d", "e6c0d9f1a0ae1615e01cb8f427d142582283d4ba1d1df33032ad1a0e7f0c2d2c", "89bb06127ad5b3e9a842877afe8935a2fef38a8b97a429e1c9f7b90ebced51c9", "bb17b1e62c9ef406216f0287ffc58422ee2455564413524f850b11167c0a6dd3"]
q_id = "56bf9b57a10cfb14005511b3"
c_id = "b83d1acd3e7b207926855f71e14faafd15177b1cb80576a80a642694b8e74f28"

relevant_documents = [{"c_id":"b83d1acd3e7b207926855f71e14faafd15177b1cb80576a80a642694b8e74f28","paragraphs":documents,"q_id":q_id}]

text = "Who else appeared with Beyonce in Telephone?"

bert = get_embedder("bertsub", "en")
q_tokens,q_emb,q_token_ids = bert([text])[0]
q_mask = [0 for i in range(len(q_emb))]
x_tensor = torch.tensor(q_emb,device=device).type(torch.float64).unsqueeze(1)
q_mask = torch.tensor(q_mask, device=device).type(torch.float64).unsqueeze(0)
bert = None


#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"
#base_path = "/scratch/s134280/squad/"

drops_value = [0.2,0.3,0.4,0.5]

for drop in drops_value:

    vocab_size = 30522
    emb_size = 768 # embedding dimension
    d_model = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
    n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8 # the number of heads in the multiheadattention models
    dropout = drop # the dropout value
    dff = d_model*4 # dimension of feed forward
    batch_size = 1
    lr = 10e-7

    epochs = 500
    epoch_size=100
    l2 = 0

    #base_line = MovingAverage(0.42499358759755385)
    base_line = MovingStdAverage()
    #reward_function = RecallRewardMean(base_line)
    reward_function = RankReward(base_line)
    specs = {
        "vocab_size":vocab_size,
        "emb_size":emb_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "nhead": nhead,
        "dropout": dropout,
        "dff": dff,
        "lr": lr,
        "l2": l2,
        "epochs": epochs,
        "base_line":base_line.name,
        "reward_function":reward_function.name
    }




    search_engine = ELSearch("squad")

    load = True

    if load:
        load_path = "/media/jonas/archive/master/data/squad/cluster_exp/17_12_19/experiments/Transformer__12-16_17:45"
        model = RLTransformer(base_path, reward_function, input_size=emb_size,num_layers=n_layers, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr,l2=l2)
        epoch = model.load(load_path +"/latest",train=True)

        specs.update({"load_path":load_path})
    else:
        model = RLTransformer(base_path, reward_function, input_size=emb_size, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr,l2=l2)


    exp_path = model.save_path + model.exp_name
    os.makedirs(exp_path + "/latest", exist_ok=True)
    os.makedirs(exp_path + "/best", exist_ok=True)
    specs.update({"model_name:": model.exp_name})
    json.dump(specs, open(exp_path + "/specs.json", 'w'))
    result_file_path = exp_path + "/norm_reward.jsonl"
    test_file_path = exp_path+"/results.jsonl"

    epoch=0




    while epoch < epochs:
        pbar = tqdm(total=epoch_size, desc="training batches for epoch {}".format(epoch))
        train_iter = 0


        for train_iter in range(epoch_size):
            train_iter += 1
            predictions = model.train_predict(x_tensor,q_mask)

            predicted_sentence = construct_sentence(predictions)
            sentence_cutoff = prune(predicted_sentence)
            search_results = search_engine.search(sentence_cutoff)

            reward,base_reward,normarlized_reward = model.reward_function(search_results,relevant_documents)

            loss = model.update_policy(normarlized_reward)
            normarlized_reward = np.mean(normarlized_reward)
            model.reward_function.base_line.update(reward,q_id)

            pbar.set_description("training batches for epoch {} with training loss: {:.3f}, normalized reward: {:.6f} base reward {:.6f}".format(epoch, loss,normarlized_reward,base_reward))

            if os.path.isfile(result_file_path):
                mode = 'a'
            else:
                mode = 'w'
            w = jsonlines.open(result_file_path, mode)
            w.write({"reward": reward, "base_reward": base_reward, "normarlized_reward": normarlized_reward})
            w.close()


            pbar.update()
        pbar.close()

        predictions = model.predict(x_tensor, q_mask)
        predicted_sentence = construct_sentence(predictions)
        sentence_cutoff = prune(predicted_sentence)
        search_results = search_engine.search(sentence_cutoff)
        reward, base_reward, normarlized_reward = model.reward_function(search_results, relevant_documents)
        tqdm.write("\n######\n{}\nbase reward: {:.6f}\ntest reward {:.6f}\n######\n".format(sentence_cutoff[0],base_reward,reward[0]))
        if os.path.isfile(test_file_path):
            mode = 'a'
        else:
            mode = 'w'
        w = jsonlines.open(test_file_path, mode)
        w.write({"reward": reward,"sentence":sentence_cutoff})
        w.close()




        epoch += 1