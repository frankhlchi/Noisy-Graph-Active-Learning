from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math 
from models import GMLP, Edge_Model
from utils import load_citation, accuracy, get_A_r, load_citation_cleaned
from utils_gcn import load_gcn_citation, gcn_accuracy, normalize,sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pickle 
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize

import warnings
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
parser.add_argument('--rand_seed', type=int, default=1, help='Random seed')
parser.add_argument('--init_split', type=int, default=0, help='initial split')

parser.add_argument('--gcn_epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--gcn_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--gcn_weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--gcn_hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--gcn_dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


parser.add_argument('--data_batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--budget_size', type=int, default=10,
                    help='budget size')
parser.add_argument('--selection_thre', type=float, default=0.0,
                    help='selection_threshold')
parser.add_argument('--nn_coverage', type=float, default=0.5,
                    help='nearesr neighbor coverage')
parser.add_argument('--denoise_perc', type=float, default=0.1,
                    help='denoise data selection metric')
parser.add_argument('--agree_epochs', type=int, default=300,
                    help='Number of epochs to train.')

parser.add_argument('--pos_conf', type=float, default=0.7 ,
                    help='positive confidence threshold.')
parser.add_argument('--neg_conf', type=float, default=1 ,
                    help='negative confidence threshold.')
parser.add_argument('--pos_unc', type=float, default=1,
                    help='positive uncertainty threshold.')
parser.add_argument('--neg_unc', type=float, default =1,
                    help='negative uncertainty threshold.')
parser.add_argument('--small_init', type=bool, default=True,
                    help='if true initialize the model with 2 samples per class otherwise 4')

parser.add_argument('--noise_level', type=float, default=0,
                    help='adding random noise perc')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



def seed_set(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def Ncontrast(x_dis, adj_label, pos_label=1, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label* pos_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def Ncontrast_large(x_dis, adj_label, rand_indx, pos_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    if len(pos_label)>1:
        x_dis_sum_pos = torch.sum(x_dis*adj_label* pos_label[rand_indx,:][:,rand_indx], 1)
    else:
        x_dis_sum_pos = torch.sum(x_dis*adj_label* pos_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    features_batch = features
    adj_label_batch = adj_label
    return features_batch, adj_label_batch


def get_batch_large(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    all_index = np.array(list(set(np.arange(len(labels))) - set(idx_train.cpu().numpy())))
    rand_indx = np.random.choice(all_index , batch_size - len(idx_train), replace=False)
    rand_indx = torch.tensor(np.concatenate((idx_train.cpu(), rand_indx))).cuda()
    
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch, rand_indx


def get_batch_large_psudo(batch_size, pseudo_label_size, pseudo_label_index):
    """
    get a batch of feature & adjacency matrix
    """
    all_index = np.array(list(set(np.arange(len(labels))) -\
                                 set(idx_train.cpu().numpy()) - set(pseudo_label_index[:pseudo_label_size].cpu().numpy())))
    rand_indx = np.random.choice(all_index , batch_size - len(idx_train) \
                                 - len(pseudo_label_index[:pseudo_label_size]), replace=False)
    rand_indx = torch.tensor(np.concatenate((idx_train.cpu(),pseudo_label_index[:pseudo_label_size].cpu(), rand_indx))).cuda()
    
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch, rand_indx


def train(pos_label):
    if  data_name != 'pubmed' and data_name != 'Coauthor-CS' :
        features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
        model.train()
        optimizer.zero_grad()
        output, x_dis = model(features_batch)
        loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, pos_label, tau = args.tau)
    else:
        features_batch, adj_label_batch, rand_indx = get_batch_large(batch_size=args.batch_size)
        model.train()
        optimizer.zero_grad()
        output, x_dis = model(features_batch)
        loss_train_class = F.nll_loss(output[:len(idx_train)], labels[idx_train])
        loss_Ncontrast = Ncontrast_large(x_dis, adj_label_batch, rand_indx, pos_label, tau = args.tau)
    
    loss_train = loss_train_class + loss_Ncontrast * args.alpha
    loss_train.backward()
    optimizer.step()
    return 

def train_self(pos_label,  pseudo_label_index):
    if  data_name != 'pubmed' and data_name != 'Coauthor-CS':
        features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
        model.train()
        optimizer.zero_grad()
        output, x_dis = model(features_batch)
        loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train_psudo = F.nll_loss(output[pseudo_label_index], pseudo_label[pseudo_label_index])
        loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, pos_label, tau = args.tau)
    else:
        
        pseudo_label_index = pseudo_label_index[torch.randperm(len(pseudo_label_index))]
        pseudo_label_size = min(int(args.batch_size/2), len(pseudo_label_index))
        features_batch, adj_label_batch, rand_indx = get_batch_large_psudo(batch_size=args.batch_size,\
                                                                           pseudo_label_size = pseudo_label_size,\
                                                                          pseudo_label_index =pseudo_label_index)
        model.train()
        optimizer.zero_grad()
        output, x_dis = model(features_batch)
        loss_train_class = F.nll_loss(output[:len(idx_train)], labels[idx_train])
        loss_train_psudo = F.nll_loss(output[len(idx_train): len(idx_train) + pseudo_label_size ], pseudo_label[pseudo_label_index[:pseudo_label_size]])
        loss_Ncontrast = Ncontrast_large(x_dis, adj_label_batch, rand_indx, pos_label, tau = args.tau)
    
    loss_train = loss_train_class/2 + loss_train_psudo/2  + loss_Ncontrast * args.alpha
    
    if  data_name != 'pubmed'and data_name != 'Coauthor-CS':
        acc_train = accuracy(output[idx_train], labels[idx_train])
    else:
        acc_train = accuracy(output[:len(idx_train)], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return 



def test():
    model.eval()
    output, rep = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return acc_test, acc_val

def test_mc(sampe_num = 5):
    model.eval()
    enable_dropout(model)
    dropout_predictions = torch.zeros((sampe_num, len(features), labels.max().item() + 1))
    for sample in range(sampe_num):
        print('sample_num:', sampe_num)
        output, rep = model(features)
        dropout_predictions[sample,:,:] = torch.exp(output)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    epistemic_std = torch.std(dropout_predictions, axis=0)
    mean_pred = torch.mean(dropout_predictions, axis=0)  
    return mean_pred, epistemic_std

def train_gcn(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = gcn_accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test_gcn():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = gcn_accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test

def vali_gcn():
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = gcn_accuracy(output[idx_val], labels[idx_val])
    print("Vali set results:",
          "loss= {:.4f}".format(loss_val.item()),
          "accuracy= {:.4f}".format(acc_val.item()))
    return acc_val


def enable_dropout(model):
#Function to enable the dropout layers during test
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def feature_cos_sim(features):
    n_feature = F.normalize(features, p=2.0, dim = 1)
    sim = np.matmul(n_feature.detach().cpu().numpy(), n_feature.detach().cpu().numpy().T)
    return sim

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


## get data
seed_set(args.rand_seed)
data_name = args.data
print (data_name, ' random seed:',args.rand_seed)
noise_level = args.noise_level
print ('noise_level:',noise_level)


#if args.data in ['cora', 'citeseer', 'pubmed']:
if False:
    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, 'AugNormAdj', True)
else:
    
    root_path = osp.expanduser('./data')
    dataset =  get_dataset('./data', args.data )

    data = dataset[0]
    adj, features, labels = data.edge_index, data.x, data.y
    
    print ('adding %s perc noise into graph'%noise_level)
    path_to_file = './data/%s_%s_%s.pt'%(data_name, noise_level, args.rand_seed)
    if noise_level ==0:
        pass
    elif osp.exists(path_to_file):
        adj = torch.load(path_to_file)
    else:
        noise_num = int(len(adj[0]) * noise_level)
        np.random.seed(0)
        counter = 0
        ori = []
        des = []
        while counter < noise_num:
            pair = np.random.choice(len(labels), 2, replace=False)
            if labels[pair[0]] != labels[pair[1]] and labels[pair[1]] != labels[pair[0]]:
                ori += [pair[0]] 
                des += [pair[1]]
                ori += [pair[1]] 
                des += [pair[0]]
                counter+=2
        adj = torch.hstack([adj, torch.tensor([ori,des])])
        torch.save(adj, path_to_file)
    
    adj = torch.sparse_coo_tensor(adj, torch.ones(len(adj[0])) )

    test_size = 1000
    init_size = len(data.y.unique())*2*10
    vali_size = 50
    init_idx = args.init_split
    np.random.seed(args.rand_seed)
    print ('random seed', args.rand_seed)
    
    train_idx_list = [] 
    for ind in data.y.unique():
        train_idx_list+= list(torch.nonzero((data.y==ind).float()).flatten()[init_idx*2:(init_idx+1)*2].numpy())
    rest_list = [i for i in range(np.max(train_idx_list) +1, len(data.y))]
    selected_data = np.random.choice(rest_list, test_size + vali_size, replace=False)
    vali_idx = selected_data[0: vali_size]
    test_idx = selected_data[ vali_size: vali_size + test_size]
    idx_train = torch.tensor(train_idx_list)
    print ('initial set ', init_idx)
    print (idx_train)
    idx_val =  torch.tensor(vali_idx)
    idx_test = torch.tensor(test_idx)
    
    if args.cuda:
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        adj = adj.cuda()

        
init_split = args.init_split

#adj, features, labels, _, _, _ = load_citation(args.data, 'AugNormAdj', True)
adj_label = get_A_r(adj, args.order)
#calculate unweighted adjacent graph
adj_dense = adj.to_dense().bool().float().detach().cpu()

nclass=labels.max().item() + 1
batch_size = nclass*2
print ('data selection batch size', batch_size)
budget_size = len(labels.unique()) * (args.budget_size)
print ('budget size', budget_size)
num_epoch = args.epochs
#calculate the cosine feature similarity 
sim = feature_cos_sim (features)
#apply it on edges
sim_vec = (sim * adj_dense.numpy()).sum(axis=1)
#calculated the reverse percentile 
sim_vec= np.argsort(np.argsort(-1*sim_vec))
sim_vec = sim_vec/sim_vec.max()
iteration = 0

init_size = len(idx_train)
print ('inital split %s'%init_split, 'random seed %s'%args.rand_seed)
print ('inital training set size: %i, testing size: %i, validation size: %i'%(init_size,len(idx_test),len(idx_test)))

#initialize the edge probability matrix
adj_label_reweight =  torch.ones_like(adj.to_dense())
flag = True
counter = 0

while flag :
    plan_size = min(batch_size, budget_size - len(idx_train))
    print ('selected data size:', len(idx_train))
    print ('planned selected number of samples', plan_size)
    ## run the representation model 
    model = GMLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                )
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        adj_label_reweight = adj_label_reweight.cuda()

    best_accu = 0
    best_val_acc = 0

    for epoch in range(num_epoch):
        if len(idx_train) == budget_size and counter > 1 and len(pseudo_label_index) > 0:
            train_self(adj_label_reweight,  pseudo_label_index)
        else:
            train(adj_label_reweight)
        tmp_test_acc, val_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    
    model.eval()
    output, rep = model(features)
    pred_dis = F.softmax(output)
    #calculate the mean prediction by running MC-dropout version representation model
    pred_dis, pred_std = test_mc()
    clas_max_predict, max_class_ind =  pred_dis.max(axis=1)
    pseudo_label = max_class_ind.flatten().cuda()
    pseudo_mask = pseudo_label == pseudo_label.reshape(-1,1)
    pseudo_mask = pseudo_mask.bool().detach().cpu()
    print ('clas_max_predict',clas_max_predict)
    pseudo_label_bool = clas_max_predict > args.pos_conf
    pseudo_label_index = torch.nonzero( pseudo_label_bool.float()).flatten().cuda()
    print ('pseudo_label_index',len(pseudo_label_index))
    pseudo_label_index = torch.tensor(list(set(pseudo_label_index.cpu().numpy()) - set(idx_train.cpu().numpy()))).cuda()
    print ('pseudo_label_index',len(pseudo_label_index))
    
    # get the initial positive training edges from pseudo labels
    pos_edge_all = torch.nonzero((adj_dense * pseudo_mask).fill_diagonal_(0)).T
    pos_score = torch.min(clas_max_predict[pos_edge_all[0]] , clas_max_predict[pos_edge_all[1]])
    # filter out edges not meeting the confidence threshold
    confi_mask_p = pos_score >= args.pos_conf
    confi_mask_p = confi_mask_p.cuda()
    pos_edge_all = pos_edge_all[:, confi_mask_p]
    #get the negative training edges from pseudo labels
    neg_edge_all = torch.nonzero((adj_dense * (~pseudo_mask)).fill_diagonal_(0)).T
        
    if data_name == 'Coauthor-CS':
        edge_batch_size = 5000
        pos_edge = pos_edge_all[:,:edge_batch_size].cuda()
        neg_edge= neg_edge_all[:,:edge_batch_size].cuda()
    else:
        pos_edge = pos_edge_all.cuda()
        neg_edge= neg_edge_all.cuda()
        
    print ('pos edge:', len(pos_edge[0]),'neg edge:',  len(neg_edge[0]))
    edge_sample =  torch.hstack([pos_edge, neg_edge] ).cuda()
    print ('edge_sample',edge_sample)
    #set edge label 
    k = len(neg_edge[0])
    edge_label = torch.ones(len(edge_sample[0])).cuda()
    edge_label[-k:] = 0
    
    #initialize the edge predictor model 
    mlp_model = Edge_Model(len(features[0]), 256, 7, 0.1).cuda()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0001, weight_decay=5e-3)
    mlp_model.train()
    
    min_loss = 10**6
    for epoch in range(args.agree_epochs):     
        optimizer.zero_grad() 
        mlp_out = mlp_model(features)
        pred = F.sigmoid(torch.sum(torch.mul(mlp_out[edge_sample[0]], mlp_out[edge_sample [1]]),dim=1))
        BCE = nn.BCELoss()
        loss =  BCE(pred, edge_label)
        
        if min_loss > loss:
            min_loss = loss
            mlp_out = mlp_model(features)
        print('edge model loss: ',loss.item())
        loss.backward() 
        optimizer.step() 
        
        if data_name == 'Coauthor-CS':
            p_idx =  epoch % math.ceil(len(pos_edge_all[0])/edge_batch_size)
            n_idx =  epoch % math.ceil(len(neg_edge_all[0])/edge_batch_size)
            print ('p_idx',p_idx)
            print ('n_idx',n_idx)
            pos_edge = pos_edge_all[:,edge_batch_size*(p_idx):edge_batch_size*(p_idx+1)].cuda()
            neg_edge= neg_edge_all[:,edge_batch_size*(n_idx):edge_batch_size*(n_idx+1)].cuda()
            edge_sample =  torch.hstack([pos_edge, neg_edge]).cuda()
            #set edge label 
            k = len(neg_edge[0])
            edge_label = torch.ones(len(edge_sample[0])).cuda()
            edge_label[-k:] = 0
            print ('edge_sample',edge_sample)

        
        
    del mlp_model
    torch.cuda.empty_cache()

 
    #generate the edge probability for all edges
    all_edge = adj.coalesce().indices()
    edge_pred = F.sigmoid(torch.sum(torch.mul(mlp_out[all_edge[0]], mlp_out[all_edge[1]]),dim=1)).detach()
    adj_label_reweight = torch.ones_like(adj.to_dense())
    adj_label_reweight[all_edge[0], all_edge[1]] = edge_pred
    adj_label_reweight[all_edge[1], all_edge[0]] = edge_pred
    
    #run the data selection module if budget is not run out
    if len(idx_train) < budget_size:
        selected_idx = idx_train.detach().cpu()
        rep = rep.cpu().detach().numpy()
        dis_matrix = pairwise_distances(rep,metric="euclidean", n_jobs=-1)
        dis_matrix[:, selected_idx] = dis_matrix.max()
        #decide the number of nodes which are well-represented
        nn_k = int(len(labels) * (len(selected_idx)/budget_size)* args.nn_coverage)
        #get the well-represented node index
        filtered_node = np.argsort((dis_matrix[selected_idx,:]).min(axis=0))[:nn_k]
        all_filtered_node = list(filtered_node) + list(selected_idx) 
        #get the nodes qualified for clustering
        candidate_list = list(set([i for i in range(len(labels))]) - set(all_filtered_node ))
        #run k-means on the candidate nodes
        kmeans = KMeans(n_clusters= plan_size, random_state=0).fit(rep[candidate_list])
        center = kmeans.cluster_centers_
        #calculate the distance percentile between cluster centroid and nodes
        dis_matrix =  pairwise_distances(center, rep, metric="euclidean", n_jobs=-1)
        dis_matrix = np.argsort(np.argsort(dis_matrix))
        dis_matrix = dis_matrix/dis_matrix.max()
        #combine two metrics
        dis_matrix =  dis_matrix * (1-args.denoise_perc )+  sim_vec  * args.denoise_perc 
        
        forbidden_list = all_filtered_node + list(idx_test.detach().cpu().numpy()) + list(idx_val.detach().cpu().numpy())
        # set the nodes which cannot be selected to label with the largest number
        dis_matrix[:, forbidden_list] = dis_matrix.max()
        #get the new selected node
        new_selected = np.argsort(dis_matrix,axis=1)[:,:1].reshape(1,-1)[0]
        cur_idx = list(selected_idx.numpy()) + list(new_selected)
        idx_train= torch.tensor(cur_idx.copy()).cuda()

    #get the weighted average estimated adjcent matrix between the prediction and the original one  
    factor = (len(idx_train) -init_size)/(budget_size -init_size)
    #adj_label_reweight = adj_label_reweight* factor + torch.ones_like(adj.to_dense())*(1- factor)
    iteration +=1
    if len(idx_train) == budget_size:
        if counter > 3:
            flag = False
        else:
            counter +=1
            
    del model
    torch.cuda.empty_cache()



log_file = open(r"class_log_%s.txt"%data_name, encoding="utf-8",mode="a+")  
with log_file as file_to_be_write:  
    print('seed', 'split' ,'vali_acc', file=file_to_be_write, sep=',')
    print(args.rand_seed, args.init_split, best_val_acc.item(), file=file_to_be_write, sep=',')

#cache the result 
print ('fitting GCN:', 'inital split %s'%init_split, 'random seed %s'%args.rand_seed)
with open('./index/selected_%s_%s_%s.pickle'%(data_name, args.rand_seed, init_split), 'wb') as handle:
    pickle.dump(idx_train.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

# fit GCN  
np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)
if args.cuda:
    torch.cuda.manual_seed(args.rand_seed)
seed_set(args.rand_seed)

adj_label_reweight = adj.to_dense().bool().float()*adj_label_reweight
adj= sparse_mx_to_torch_sparse_tensor(normalize(coo_matrix(adj_label_reweight.detach().cpu().numpy())))
print ('idx_train:', len(idx_train))
print (idx_train)
print ('idx_val',len(idx_val))
print ('idx_test',len(idx_test))

model = GCN(nfeat=features.shape[1],
            nhid=args.gcn_hidden,
            nclass=labels.max().item() + 1,
            dropout=args.gcn_dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.gcn_lr, weight_decay=args.gcn_weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
# Train model
t_total = time.time()
for epoch in range(args.gcn_epochs):
    train_gcn(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
acc_test = test_gcn()
acc_vali = vali_gcn()

log_file = open(r"log_%s.txt"%data_name, encoding="utf-8",mode="a+")  
with log_file as file_to_be_write:  
    print('seed', 'split' ,'alpha','tau', 'batch_size', 'budget_size', 'selection_thre', 'nn_coverage', 'denoise_perc', 'pos_conf', 'pos_unc', 'neg_unc', 'vali_acc','test_acc', file=file_to_be_write, sep=',')
    print(args.rand_seed, args.init_split, args.alpha, args.tau, args.batch_size, args.budget_size, args.selection_thre, args.nn_coverage, args.denoise_perc, args.pos_conf,  args.pos_unc, args.neg_unc, acc_vali.item(), acc_test.item(), file=file_to_be_write, sep=',')


