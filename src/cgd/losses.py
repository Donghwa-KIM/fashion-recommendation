import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

class BatchHardTripletSelector(nn.Module):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def forward(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg
    
class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.Loss = nn.TripletMarginLoss(margin = margin, p = 2.0)

    def forward(self, anchor, pos, neg):
        
        return self.Loss(anchor, pos, neg)
    
    

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,smoothing, temperature_scale):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.temperature_scale = temperature_scale
        
    def forward(self, logits, target):
        '''
        true distribution -> true + noise distribution
        '''
        logits_scaled = logits/self.temperature_scale
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(logits_scaled, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
