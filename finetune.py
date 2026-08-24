import os
from glob import glob
from datetime import datetime
from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from siamese_models import PretrainingFinetuningModel
from datasets import FinetunePartNetDataModule

# variables
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
OS_CWD = os.getcwd()
DEFAULT_DATA_PATH = "data/PartNet/all/"

# arguments
parser = ArgumentParser()
parser.add_argument("-id", "--slurm_id", dest="slurm_id", default=None, type=str)
parser.add_argument("-d", "--data_path", dest="data_path", default=DEFAULT_DATA_PATH, type=str)
parser.add_argument("-b", "--batch_size", dest="batch_size", default=10, type=int)
parser.add_argument("-e", "--num_epochs", dest="num_epochs", default=5, type=int)
args = parser.parse_args()

# checkpoint
dir_path = f"logs/run_{args.slurm_id}"
ckpt_paths = f"{dir_path}/checkpoints/*.ckpt"
ckpt_path = sorted(glob(ckpt_paths))[0]
with open(f"{dir_path}/hparams.yaml") as f:
    hparams = yaml.safe_load(f)
print("Pretrained:")
print("SLURM_JOB_ID", args.slurm_id)
print("CHECKPOINT_PATH", ckpt_path)
print(hparams)

# dataset
dataset_args = {
    "file_path": args.data_path,
    "batch_size": args.batch_size,
    "lra_feat": hparams["encoder_type"] == "RIConv2",
}
dataset = FinetunePartNetDataModule(**dataset_args)

# model
model_args = {"n_epochs": args.num_epochs, "pretrain": False}
if args.slurm_id is not None:
    model = PretrainingFinetuningModel.load_from_checkpoint(ckpt_path, **model_args, **dataset_args)
else:
    print("WARNING: no checkpoint loaded")

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
