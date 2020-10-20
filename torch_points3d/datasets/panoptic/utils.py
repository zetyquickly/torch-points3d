import numpy as np
import torch

def set_extra_labels(data, instance_classes, num_max_objects):
    """ Adds extra labels for the instance and object segmentation tasks
    - num_instances: number of instances
    - center_label: [num_max_objects, 3] one bbox center per instance
    - instance_labels: [num_points]
    - vote_label: [num_points, 3] displacement of bbox center with respect to the object points
    - instance_mask: [num_points] boolean mask
    - object_center_label: [num_max_objects, 3] one per instance mean position vector
    - object_displmnt_mask: [num_points, 3] displacement of each object point with respect to the object center
    """
    # Initaliase variables
    num_points = data.pos.shape[0]
    semantic_labels = data.y

    # compute votes *AFTER* augmentation
    instances = np.unique(data.instance_labels)
    centers = []
    object_centers = []
    point_votes = torch.zeros([num_points, 3])
    object_point_displmnt = torch.zeros([num_points, 3])
    instance_labels = torch.zeros(num_points, dtype=torch.long)
    instance_idx = 1
    for i_instance in instances:
        # find all points belong to that instance
        ind = np.where(data.instance_labels == i_instance)[0]
        # find the semantic label
        instance_class = semantic_labels[ind[0]].item()
        if instance_class in instance_classes:  # We keep this instance
            pos = data.pos[ind, :3]
            max_pox = pos.max(0)[0]
            min_pos = pos.min(0)[0]
            center = 0.5 * (min_pos + max_pox)
            point_votes[ind, :] = center - pos
            centers.append(torch.tensor(center))
            object_center = pos.mean(axis=0)
            object_point_displmnt[ind, :] = pos - object_center
            object_centers.append(torch.tensor(object_center))
            instance_labels[ind] = instance_idx
            instance_idx += 1

    num_instances = len(centers)
    if num_instances > num_max_objects:
        raise ValueError(
            "We have more objects than expected. Please increase the NUM_MAX_OBJECTS variable.")
    data.center_label = torch.zeros((num_max_objects, 3))
    data.object_center_label = torch.zeros((num_max_objects, 3))
    if num_instances:
        data.center_label[:num_instances, :] = torch.stack(centers)
        data.object_center_label[:num_instances, :] = torch.stack(object_centers)

    data.vote_label = point_votes.float()
    data.object_displmnt_mask = object_point_displmnt.float()
    data.instance_labels = instance_labels
    data.instance_mask = instance_labels != 0
    data.num_instances = torch.tensor([num_instances])
    return data