import jsonlines
from search_engine.elastic_search import ELSearch
from analysis.process_results import eval
from training.dataloaders.rl_squad_dataloader import RLSquadDataloader
#from training.dataloaders.rl_squad_sample_dataloader import RLSquadDataloader
from models.my_transformer import MyTransformer
from models.rl_transformer import RLTransformer
import torch
from tqdm import tqdm
import numpy as np
from dataset.bertify import construct_sentence,prune

def get_documents_from_result(results):

    hits = results["hits"]["hits"]

    retrieved_documents = []
    for hit in hits:
        retrieved_documents.append(hit["_id"])
    return retrieved_documents

base_path = "/media/jonas/archive/master/data/rl_squad/"
search_engine = ELSearch("squad")

results = []



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

vocab_size = 30522
emb_size = 768 # embedding dimension
d_model = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
dff = 4*d_model # dimension of feed forward
batch_size = 8
lr = 0.0001
l2 = 0
epochs = 100

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
}

load_path = "/media/jonas/archive/master/data/rl_squad_sub/experiments/Transformer__12-17_22:13"
#model = RLTransformer(base_path,reward_function=None, input_size=emb_size, output_size=vocab_size, device=device, nhead=nhead,dropout=dropout, d_model=d_model, dff=dff, lr=lr)
#epoch = model.load(load_path  + "/latest/", train=True)


eval_data = RLSquadDataloader(base_path=base_path,batch_size=1,max_length=300,eval=True)
pbar = tqdm(total=5928, desc="evaluating batches for epoch")
for q_batch, relevant_documents, q_txt,q_mask in iter(eval_data):

    q_emb = torch.tensor(q_batch, device=device).type(torch.float64)
    q_emb = q_emb.reshape(q_emb.size(1), 1, q_emb.size(2))
    q_mask = torch.tensor(q_mask, device=device).type(torch.float64)

    #predictions = model.predict(q_emb,q_mask)
    #print(predictions.shape)
    #predicted_sentence = construct_sentence(predictions)
    #print(predicted_sentence)
    #sentence_cutoff = prune(predicted_sentence)
    #text = sentence_cutoff
    text = q_txt
    relevant_docs = relevant_documents[0]
    search_result = search_engine.search(text)[0]
    results_docs = get_documents_from_result(search_result)
    results.append((relevant_docs,results_docs))
    pbar.update()

eval(results)
