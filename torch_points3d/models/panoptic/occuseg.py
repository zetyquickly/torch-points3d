import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_mean

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL


log = logging.getLogger(__name__)

class OccuSegModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option)
        self.loss_names = ["spatial_term"]

    def set_input(self, data, device):
        pass
    
    def _compute_spatial_term(self, di, gt_di, gt_instance_labels, keep_mask):
        di_filtered = di * keep_mask
        displmnt_cluster_error = scatter_mean(torch.norm(di_filtered + gt_di, dim = 1), gt_instance_labels, dim = 0)
        displmnt_scene_error = displmnt_cluster_error.sum() / (gt_instance_labels.max() + 1)
        return displmnt_scene_error


    def _compute_loss(self, preds, labels, ignore_index):
        self.spatial_term = self._compute_spatial_term(
            preds.di, 
            labels.object_displmnt_mask, 
            labels.instance_labels, 
            # assuming that instance_mask contains classes that we want to keep (i.e. no floors, walls, ceilings)
            labels.instance_mask
        )

    def forward(self, *args, **kwargs):
        pass

    def backward(self):
        pass