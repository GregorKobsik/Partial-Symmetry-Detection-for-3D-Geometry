# TODO: load model and embed patches als clr_features

from argparse import ArgumentParser
import numpy as np

from sym_cl import embed_features

parser = ArgumentParser()
parser.add_argument("-s", "--shape_name", dest="shape_name", default="filigree", type=str)
parser.add_argument("-id", "--model_id", dest="model_id", default=37809, type=int)
args = parser.parse_args()
data_dir = f"data/precomputed/{args.shape_name}"

# load data
point_cloud = np.load(f"{data_dir}/pcl.npy")
patches_idx = np.load(f"{data_dir}/patches_idx.npy")

######################
### Embed Features ###
######################

features, features_dist = embed_features(point_cloud, patches_idx, args.model_id, silent=False)

# save data
np.save(f"{data_dir}/clr_features.npy", features)
np.save(f"{data_dir}/clr_distances.npy", features_dist)
#yaml.dump(vars(model.hparams), open(f"{data_dir}/clr_hparams.yaml", "w"))
