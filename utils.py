import os

def create_exp_dir(model_name):
    os.makedirs("results/"+model_name,exist_ok=True)