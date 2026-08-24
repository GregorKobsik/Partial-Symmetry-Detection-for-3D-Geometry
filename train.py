import os
from datetime import datetime
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from siamese_models import (
    ContrastiveSimilarityModel,
    PretrainingFinetuningModel,
)
from datasets import (
    PretrainPartNetDataModule,
    FinetunePartNetDataModule,
)

models = {
    "contrastive_similarity": ContrastiveSimilarityModel,
    "pretrain_contrastive_partnet": PretrainingFinetuningModel,
    "icp_only_partnet": PretrainingFinetuningModel,
}
datasets = {
    "contrastive_similarity": PretrainPartNetDataModule,
    "pretrain_contrastive_partnet": PretrainPartNetDataModule,
    "icp_only_partnet": FinetunePartNetDataModule,
}
encoders = [
    "VNPointNet",
    "VNDGCNN",
]
DEFAULT_DATA_PATH = "data/PartNet/patches/"

parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    dest="model",
    default="contrastive_similarity",
    type=str,
    choices=models.keys(),
)
parser.add_argument(
    "-enc",
    "--encoder",
    dest="encoder",
    default="VNPointNet",
    type=str,
    choices=encoders,
)
parser.add_argument("-d", "--data_path", dest="data_path", default=DEFAULT_DATA_PATH, type=str)
parser.add_argument("-b", "--batch_size", dest="batch_size", default=10, type=int)
parser.add_argument("-e", "--num_epochs", dest="num_epochs", default=5, type=int)
args = parser.parse_args()

# variables
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
OS_CWD = os.getcwd()

# dataset
dataset_args = {
    "file_path": args.data_path,
    "batch_size": args.batch_size,
    "lra_feat": args.encoder == "RIConv2",
}
dataset = datasets[args.model](**dataset_args)

# model
model_args = {
    "model": args.model,
    "encoder_type": args.encoder,
    "n_epochs": args.num_epochs,
    "pretrain": not args.model.startswith("icp_only"),
}
model = models[args.model](**model_args, **dataset_args)

# training
trainer_args = {
    "accelerator": "gpu",
    "max_epochs": args.num_epochs,
    "logger": TensorBoardLogger(save_dir=OS_CWD, name="logs", version=f"run_{SLURM_JOB_ID}"),
    "default_root_dir": OS_CWD + "/logs",
    "log_every_n_steps": 50,
    "limit_val_batches": 10,
    "fast_dev_run": False,
}
trainer = pl.Trainer(**trainer_args)
print("INFO: training started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
trainer.fit(model, datamodule=dataset)
print("INFO: training finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
