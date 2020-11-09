
from torch import nn
from torch.nn import functional as F
import torch


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return torch.flatten(self.gem(x, p=self.p, eps=self.eps), start_dim=1)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, 'the input tensor of sL2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)

    
class CGD(nn.Module):
    def __init__(self, gd_config, feature_dim, base_dim, num_classes):
        '''
        gd_config: pooling order
        feature_dim: embedding size for cgd
        base_dim: embedding size for base model 
        num_classes: number of pair ID
        
        '''
        super().__init__()

        self.global_descriptors, self.main_modules = [], []
        for p in gd_config:
            self.global_descriptors.append(GeM(p=p))
            self.main_modules.append(nn.Sequential(nn.Linear(base_dim, int(feature_dim/ len(gd_config)), bias =False), L2Norm()))
        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.main_modules = nn.ModuleList(self.main_modules)
        
        # Auxiliary Module
        self.auxiliary_module = nn.Sequential(nn.BatchNorm1d(base_dim), nn.Linear(base_dim, num_classes,bias=True) )
        
    def forward(self, features):
        cgd = []
        for i in range(len(self.global_descriptors)):
            gd = self.global_descriptors[i](features)
            if i==0:
                classes = self.auxiliary_module(gd)
            gd_emb = self.main_modules[i](gd)
            cgd.append(gd_emb)
        cgd = F.normalize(torch.cat(cgd, dim=-1), dim=-1)
        return cgd, classes
    
