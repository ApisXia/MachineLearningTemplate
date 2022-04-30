from model import SampleModel
from dataloader import SampleLoader
import trainer
import torch

net = SampleModel()

batch_size = 16
num_workers = 0

train_dataset = SampleLoader('train', batch_size, num_workers)
val_dataset = SampleLoader('val', batch_size, num_workers)

exp_name = 'PatchBatch_w060'

trainer = trainer.Trainer(net,
                          torch.device("cuda"),
                          train_dataset,
                          val_dataset,
                          exp_name)
 
trainer.train_model(4000)
