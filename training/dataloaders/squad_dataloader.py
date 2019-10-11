from training.dataloaders.base_dataloader import BaseDataloader
from training.dataloaders.query_to_laser import QueryToLaser
from training.dataloaders.target_to_laserbpe import TargetToLaserBPE
from embedding_method.embedders import get_embedder

class SquadDataloader:
    def __init__(self,base_path,max_length,language,eval,batch_size):
        embedder = get_embedder("laser",language)
        self.input_loader = QueryToLaser(base_path=base_path,max_length=max_length,embedder=embedder)
        self.output_loader = TargetToLaserBPE(base_path=base_path,max_length=max_length,embedder=embedder,language=language)
        self.dataloader = BaseDataloader(base_path=base_path,input_loader=self.input_loader,output_loader=self.output_loader,query_file_name="qas.jsonl",eval=eval,batch_size=batch_size)

    def __iter__(self):
        return iter(self.dataloader)