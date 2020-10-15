from omegaconf import OmegaConf
import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from torch_points3d.datasets.panoptic.scannet import ScannetDataset
from torch_points3d.models.panoptic.structures import PanopticLabels


data_config = OmegaConf.load(os.path.join(DIR_PATH, "conf/scannet-panoptic.yaml"))
dataset = ScannetDataset(data_config.data)

data = dataset.train_dataset[0]
for key in PanopticLabels._fields:
    print(key in data)