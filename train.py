import argparse
import os
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from dataset import get_dataset
import torch
import wandb
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
import torch_geometric
import numpy as np
from model import Encoder, Model
from eval import label_classification
from mask import compute_cam,drop_edge,drop_feature
import scipy.sparse as sp
from recorder import Recorder
from torch_geometric.utils import degree
debug=True
first_time=False
U=0
Lambda=0
mean=0
std=0.1
direction=0
edge1=0
weight1=0
edge2=0
weight2=0
use_spec_aug=[]
thetas=[]
pre_theta=0
weight_decay_of_spec=1
Lambda_1=0
def get_T(z0,z1,z2):
    norm1=torch.norm(z0-z1,dim=1)
    norm2=torch.norm(z0-z2,dim=1)
    print(norm1.shape)
    norm1=norm1.mean()
    norm2=norm2.mean()
    return ((norm1+norm2)/2).item()

def adj_to_edgeidx(adj):
    adj=sp.coo_matrix(adj.cpu())
    values=adj.data
    indices=np.vstack((adj.row,adj.col))
    edge_index=torch.LongTensor(indices).to(device)
    edge_weight=torch.FloatTensor(values).to(device)
    return edge_index,edge_weight

def augment_spe(epsilon,epoch,num_epochs):
    global mean,std,Lambda,U,direction,edge1,weight1,edge2,weight2,Lambda_1
    print("mean=",mean)
    positive=torch.where(Lambda>0)[0]
    negative=torch.where(Lambda<0)[0]
    print(positive.shape,negative.shape)
    print(torch.mean(direction[positive]),torch.mean(direction[negative]))
    mean+=direction*epsilon*Lambda*((num_epochs-epoch)/num_epochs)
    #noise_1=torch.tensor([torch.normal(m,std) for m in mean]).to(device)
    #print("noise_1==",noise_1)
    Lambda_1=Lambda+mean

    Lambda_1=torch.diag(Lambda_1)

    adj1=torch.matmul(U,Lambda_1)
    adj1=torch.matmul(adj1,U.T)
    
    zeros=torch.zeros_like(adj1)
    adj1=torch.where(adj1>0.2,adj1,zeros)
    adj2=adj1.clone()

    edge1,weight1=adj_to_edgeidx(adj1)
    edge2,weight2=adj_to_edgeidx(adj2)
    print("average weight=",torch.mean(weight1))
    #edge2,weight2=edge1,weight1
    
def augmentation(x,edge_degree,retain_edge,retain_feature,throw_edge,throw_feature,config):
    edge_index_1,edge_weight_1 = drop_edge(edge1,weight1, config['drop_edge_rate_1'],config['retain_prob'],retain_edge,throw_edge,edge_degree,True)
    edge_index_2,edge_weight_2 = drop_edge(edge2,weight2, config['drop_edge_rate_2'],config['retain_prob'],retain_edge,throw_edge,edge_degree,True)
    edge_index_1,edge_weight_1=edge1,weight1
    edge_index_2,edge_weight_2=edge2,weight2
    x_1=drop_feature(x,config['drop_feature_rate_1'],config['retain_prob'],retain_feature,throw_feature)
    x_2=drop_feature(x,config['drop_feature_rate_2'],config['retain_prob'],retain_feature,throw_feature)
    return x_1,edge_index_1,edge_weight_1,x_2,edge_index_2,edge_weight_2
    
def scipy_to_torch(x):
    values=x.data
    indices=np.vstack((x.row,x.col))
    idx=torch.LongTensor(indices)
    val=torch.FloatTensor(values)
    shape=x.shape
    x=torch.sparse.FloatTensor(idx,val,torch.Size(shape))
    return x
    
def init_edge(x,edge_index,config):
    if use_spec_aug:
        adj=torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
        adj=scipy_to_torch(adj)

        global Lambda, U, mean, direction
        path=osp.join('eigs',config['dataset'],'lambda.pt')
        if osp.exists(path)==False:
            adj=adj.to(device)
            (Lambda,U) = torch.eig(adj.to_dense(),eigenvectors=True)
            Lambda=torch.tensor(Lambda).to(device)
            Lambda=Lambda[:,0].to(device)
            U=U.to(device)
            
            torch.save(Lambda,osp.join('eigs',config['dataset'],'lambda.pt'))
            torch.save(U,osp.join('eigs',config['dataset'],'U.pt'))
            
        else:
            Lambda=torch.load(osp.join('eigs',config['dataset'],'lambda.pt')).to(device)
            U=torch.load(osp.join('eigs',config['dataset'],'U.pt')).to(device)
        mean=torch.zeros([Lambda.shape[0]]).to(device)
        global thetas, pre_theta
        thetas=[]
        pre_theta=0
        direction=torch.zeros([x.shape[0]]).to(device)
        augment_spe(config['epsilon_1'],1,config['num_epochs'])
    else:
        global edge1,edge2,weight1,weight2
        edge1=edge_index
        edge2=edge_index
        _,num_edge=edge_index.shape
        weight1=torch.ones([num_edge]).to(device)
        weight2=torch.ones([num_edge]).to(device)
    
def train_cam(model:Model,optimizer,x,edge_index,y,num_epochs,config):

    x=x.to(device)
    edge_index=edge_index.to(device)
    model=model.to(device)
    start = t()
    prev = start
    retain_edge_num=(int)(config['retain_rate']*edge_index.shape[1])
    retain_feature_num=(int)(config['retain_rate']*x.size(1))
    throw_edge_num=(int)((1-config['retain_rate'])*edge_index.shape[1])
    throw_feature_num=(int)((1-config['retain_rate'])*x.size(1))
    if config['retain_rate']==0:
        throw_edge_num=0
        throw_feature_num=0
    x.requires_grad_(True)
    src, dst =edge_index[0], edge_index[1]
    D=degree(edge_index[0])
    Dedge=D[src]+D[dst]
    init_edge(x,edge_index,config)
    recorder=Recorder()
    node_cams=[]
    feature_cams=[]
    edge_cams=[]
    feature_cam=torch.load(osp.join('cams',config['dataset'],'feature_cam.pt'))
    node_cam=torch.load(osp.join('cams',config['dataset'],'node_cam.pt'))
    edge_cam=torch.load(osp.join('cams',config['dataset'],'edge_cam.pt'))
    
    _,retain_edge=torch.topk(edge_cam,retain_edge_num)
    _,retain_feature=torch.topk(feature_cam,retain_feature_num)
    _,throw_edge=torch.topk(edge_cam,throw_edge_num,largest=False)
    _,throw_feature=torch.topk(feature_cam,throw_feature_num,largest=False)



    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        x_1, edge_index_1, edge_weight_1, x_2, edge_index_2, edge_weight_2=\
            augmentation(x,Dedge,retain_edge,retain_feature,throw_edge,throw_feature,config)
        z0,_=model(x,edge1,weight1)
        z1,wx1 = model(x_1, edge_index_1, edge_weight_1)
        z2,wx2 = model(x_2, edge_index_2, edge_weight_2)
        loss = model.loss(z1, z2, batch_size=0)
        from eval import get_dis_with_center
        sim_y,sim_i,delta,micro=get_dis_with_center(z0,y,z1,z2)
        print("sim==",sim_y,sim_i,micro)
        '''
        node_cam,feature_cam=compute_cam(x,loss)
        edge_cam = (node_cam[src] + node_cam[dst]) /2   
        node_cams.append(node_cam)
        feature_cams.append(feature_cam)
        edge_cams.append(edge_cam)
        '''

        if use_spec_aug:
            
            if epoch%20==0:
                global thetas,pre_theta,direction
                temp1=torch.mm(U.T,wx1)
                temp2=torch.mm(U.T,wx2)
                temp=torch.mul(temp1,temp2)
                theta=torch.sum(temp,dim=1)
                idx=torch.where(theta>0)
                sizeg=idx[0].shape[0]
                idx=torch.where(theta<0)
                sizel=idx[0].shape[0]
                print("positive rate=",sizeg/(sizeg+sizel))
                thetas.append(theta)
                epsilon=config['epsilon_2']
                direction=torch.zeros((x.shape[0])).to(device)
                thetas=torch.stack(thetas)
                theta=torch.mean(thetas,dim=0)
                print("theta mean==",torch.mean(theta))
                if epoch == 20:
                    pre_theta=theta
                minus=(theta-pre_theta).cpu().clone().detach().numpy()
                idx=np.where(minus>epsilon)
                direction[idx]=1
                idx=np.where(minus<-epsilon)
                direction[idx]=-1
                print(torch.sum(direction))
                
                pre_theta=theta
                thetas=[]
                augment_spe(config['epsilon_1'],epoch,config['num_epochs'])
                recorder.record_direction(direction,Lambda_1,theta)
        
        loss.backward()
        optimizer.step()
        res={
            'NCE_loss':loss.item()
        }
        if not debug:
            wandb.log(res)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    if use_spec_aug:
        recorder.show_directions(Lambda)
    '''
    feature_cams=torch.stack(feature_cams)
    feature_cams=torch.mean(feature_cams,dim=0)
    torch.save(feature_cams,osp.join('cams',config['dataset'],'feature_cam.pt'))
    
    node_cams=torch.stack(node_cams)
    node_cams=torch.mean(node_cams,dim=0)
    torch.save(node_cams,osp.join('cams',config['dataset'],'node_cam.pt'))
    
    edge_cams=torch.stack(edge_cams)
    edge_cams=torch.mean(edge_cams,dim=0)
    torch.save(edge_cams,osp.join('cams',config['dataset'],'edge_cam.pt'))
    '''
    return loss.item(),z1,z2



def test(model: Model, x, edge_index,edge_weight, y, final=False):
    model.eval()
    z,_ = model(x, edge_index,edge_weight)

    micro,macro=label_classification(z, y, ratio=0.1)

    return z,micro,macro


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='I')
    parser.add_argument('--retain_rate', type=float, default=0.2)
    args = parser.parse_args()


    assert args.gpu_id in range(0, 8)

    torch.cuda.set_device(args.gpu_id)
    global device
    device='cuda:'+str(args.gpu_id)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    config['dataset']=args.dataset
    if args.method=='I':
        config['retain_rate']=0.05
        config['retain_prob']=0.5
        config['use_spec_aug']=False
    else:
        config['use_spec_aug']=True
        config['epsilon_1']=-0.04
        config['epsilon_2']=0.02
        config['retain_rate']=0
        config['retain_prob']=0
    #config['num_epochs']=500
    global use_spec_aug
    use_spec_aug=config['use_spec_aug']

    learning_rate = config['learning_rate']

    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu':F.rrelu})[config['activation']]
    num_layers = config['num_layers']

    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data.x=data.x.to(device)
    data.edge_index=data.edge_index.to(device)
    data.y=data.y.to(device)
    model=model.to(device)

    loss,z1,z2 = train_cam(model,optimizer, data.x, data.edge_index,data.y,num_epochs,config)
    


    print("=== Final ===")
    embed_0,micro,macro=test(model, data.x, edge1,weight1, data.y, final=True)

    print(T,micro)
        
    res={
        'micro':micro,'macro':macro
    }
    if not debug:
        wandb.log(res)

if __name__ == '__main__':
    import time
    init_seed=12345
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(init_seed)
    if debug:
        begin=time.time()
        main()
        end=time.time()
        print("time==",end-begin)
    else:
        wandb.login()
        
        curPath = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(curPath, "sweep1.yaml")    
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = f.read()
        sweep_config = yaml.load(config,Loader=yaml.FullLoader)

        sweep_id=wandb.sweep(sweep_config,project='GCL_final_S')
        
        wandb.agent(sweep_id,main)