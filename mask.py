import torch
import numpy as np
import random
def normalize(cam,eps=1e-20):
    cam=cam.clone()
    max=torch.max(cam.squeeze())
    min=torch.min(cam.squeeze())
    normalized_cam=(cam-min)/(max-cam+eps)
    normalized_cam = normalized_cam.clamp_min(0)
    normalized_cam = normalized_cam.clamp_max(1)
    return normalized_cam

def compute_cam(feature,score):
    grad = torch.autograd.grad(score.sum(), feature,retain_graph=True)[0]
    node_weight = torch.mean(grad, dim=1, keepdim=True)
    feature_weight=torch.mean(grad, dim=0, keepdim=True)
    node_cam=torch.sum(node_weight * feature, dim=1, keepdim=True).detach()
    feature_cam=torch.sum(feature_weight * feature, dim=0, keepdim=True).detach()
    temp=feature_cam.cpu().detach().numpy()
    normalized_node_cam = normalize(node_cam).squeeze().detach()
    normalized_feature_cam = normalize(feature_cam).squeeze().detach()
    return normalized_node_cam,normalized_feature_cam


def drop_edge(edge_index,edge_weight,rate,retain_prob,retain,throw_edge,edge_degree,use_degree_drop):
    num_edge=edge_index.size(1)
    edge_mask = torch.rand(num_edge, device=edge_index.device) >= rate
    
    index = torch.LongTensor(random.sample(range(len(retain)), (int)(len(retain)*retain_prob))).to(retain.device)
    retain=torch.index_select(retain,0,index)
    edge_mask[retain]=True
    index = torch.LongTensor(random.sample(range(len(throw_edge)), (int)(len(retain)*2))).to(retain.device)
    throw=torch.index_select(throw_edge,0,index)
    edge_mask[throw]=False
    edge_index = edge_index[:, edge_mask]
    edge_weight = edge_weight[edge_mask]
    return edge_index,edge_weight
    
def drop_feature(x, drop_prob,retain_prob,retain,throw):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) >= drop_prob
    index = torch.LongTensor(random.sample(range(len(retain)), (int)(len(retain)*retain_prob))).to(retain.device)
    retain=torch.index_select(retain,0,index)
    drop_mask[retain]=True
    index = torch.LongTensor(random.sample(range(len(throw)), (int)(len(retain)*2))).to(retain.device)
    throw=torch.index_select(throw,0,index)
    drop_mask[throw]=False
    temp=torch.stack([drop_mask]*x.shape[0])
    return torch.mul(x,temp)