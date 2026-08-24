import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from icp_distance import icp_distance


print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

parser = ArgumentParser()
parser.add_argument("-in", "--input_file_list", dest="input_file_list", default=None, type=str)
args = parser.parse_args()
file_paths = open(args.input_file_list).read().splitlines()

for path in tqdm(file_paths):
    out_dirname = os.path.dirname(path)
    if os.path.exists(f"{out_dirname}/icp_distances_1024.pt"):
        continue
    os.makedirs(out_dirname, exist_ok=True)
    torch.save(torch.ones(1), f"{out_dirname}/icp_distances_1024.pt")

    pnts = torch.load(path)[:, 0]
    icp_distances = icp_distance(pnts)
    torch.save(icp_distances, f"{out_dirname}/icp_distances_1024.pt")

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
