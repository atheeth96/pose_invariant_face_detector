import math
import torch
import torch.nn as nn

import torch.nn.functional as F

class NaiveTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
    
def _pairwise_distances(embeddings, squared=False):
  
    dot_product = torch.matmul(embeddings, embeddings.T)

    square_norm = torch.diagonal(dot_product,0)

    distances =square_norm.unsqueeze(0)- 2.0 * dot_product + square_norm.unsqueeze(1)

    distances = torch.max(distances, torch.tensor([0.0]))

    if not squared:
        
        mask = torch.eq(distances,torch.tensor([0])).type(torch.FloatTensor)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)
        
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
   
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor)
    indices_not_equal = indices_equal^1
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = ((i_not_equal_j&i_not_equal_k)&j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.equal(labels.unsqueeze(0),labels.unsqueeze(1))
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = (i_equal_j&(i_equal_k^1))

    # Combine the two masks
    mask = (distinct_indices&valid_labels)

    return mask


def _get_anchor_positive_triplet_mask(labels):
    
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor)
    indices_not_equal = indices_equal^1

    labels_equal = torch.eq(labels.unsqueeze(0),labels.unsqueeze(1))

    mask = indices_not_equal& labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    
    labels_equal = torch.eq(labels.unsqueeze(0),labels.unsqueeze(1))

    mask = labels_equal^1

    return mask




def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
  
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)


    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

   
    mask = _get_triplet_mask(labels).type(torch.FloatTensor)
    triplet_loss = mask*triplet_loss

    triplet_loss = torch.max(triplet_loss, torch.tensor([0.0]))

    valid_triplets = (triplet_loss>torch.tensor([1e-16])).type(torch.FloatTensor)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
   
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).type(torch.BoolTensor)
    

    anchor_positive_dist = (mask_anchor_positive*pairwise_dist)

    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1).values

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).type(torch.BoolTensor)
    

    max_anchor_negative_dist = torch.max(pairwise_dist, dim=1)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1)

    triplet_loss =torch.max(hardest_positive_dist - hardest_negative_dist + margin, torch.tensor([0.0]))
   
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss



