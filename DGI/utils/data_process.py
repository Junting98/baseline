import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import itertools
import os.path
import pickle
import os
import scipy.sparse as sp
import torch
mapping = {"Case_Based":0,"Genetic_Algorithms":1,"Neural_Networks":2,"Probabilistic_Methods":3,"Reinforcement_Learning":4,"Rule_Learning":5,"Theory":6}
motifs = {
    1: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"+"})]),
    2: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"-"})]),
    3: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"-"})]),
    4: nx.DiGraph([(1,2,{'s':"+"}),(1,3,{'s':"+"})]),
    5: nx.DiGraph([(1,2,{'s':"+"}),(1,3,{'s':"-"})]),
    6: nx.DiGraph([(1,2,{'s':"-"}),(1,3,{'s':"-"})]),
    7: nx.DiGraph([(2,1,{'s':"-"}),(3,1,{'s':"-"})]),
    8: nx.DiGraph([(2,1,{'s':"+"}),(3,1,{'s':"-"})]),
    9: nx.DiGraph([(2,1,{'s':"+"}),(3,1,{'s':"+"})]),
    10: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"+"}),(1,3,{'s':'+'})]),
    11: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"-"}),(1,3,{'s':'+'})]),
    12: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"-"}),(1,3,{'s':'+'})]),
     13: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"-"}),(1,3,{'s':'-'})]),
     14: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"+"}),(1,3,{'s':'-'})]),
     15: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"-"}),(1,3,{'s':'-'})]),
        16: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"+"}),(3,1,{'s':'+'})]),
    17: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"-"}),(3,1,{'s':'+'})]),
    18: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"-"}),(3,1,{'s':'+'})]),
     19: nx.DiGraph([(1,2,{'s':'+'}),(2,3,{'s':"-"}),(3,1,{'s':'-'})]),
     20: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"+"}),(3,1,{'s':'-'})]),
     21: nx.DiGraph([(1,2,{'s':'-'}),(2,3,{'s':"-"}),(3,1,{'s':'-'})]),
}


motifs1 = {
    0: nx.DiGraph([(1,2),(2,3)]),
    1: nx.DiGraph([(1,2),(1,3)]),
    2: nx.DiGraph([(2,1),(3,1)]),
    3: nx.DiGraph([(1,2),(2,3),(1,3)]),
    #4: nx.DiGraph([(1,2),(2,3),(3,1)]),
    4: nx.DiGraph([(1,2),(2,3),(3,2)]),
    5: nx.DiGraph([(1,2),(1,3),(3,1)]),
    6: nx.DiGraph([(2,1),(3,1),(1,3)]),
    7: nx.DiGraph([(1,2),(2,3),(1,3),(3,2)]),
    8: nx.DiGraph([(1,2),(2,3),(1,3),(2,1)]),
    9: nx.DiGraph([(1,2),(2,3),(3,1),(1,3)]),
}




def preprocess(dataset='data/cora',directed='directed',split=0.8):
    f = open(dataset+'.cites','r')
    mask,labels,attribute = get_dataset(dataset)
    G = None
    if directed == "directed":
        G = nx.DiGraph()
    for line in f.readlines():
        edge = map(int,line.strip().split())
        G.add_edge(mask[edge[1]], mask[edge[0]])
    return mask,labels,attribute,G



def get_dataset(dataset):
    f = open(dataset+".content", 'r')
    lines = f.readlines()
    mask = {}
    labels = np.zeros(len(lines))
    attribute = np.zeros((len(lines),1433))
    idx = 0
    for line in lines:
        line = line.strip().split()
        mask[int(line[0])] = idx
        attribute[idx] = np.array(map(int,line[1:-1]))
        labels[idx] = mapping[line[-1]]
        idx+=1
    return mask, labels, attribute


def mcounter(gr, mask, mo=motifs1):
    # mcount: the count of each motif types
    # M_instance: Dictionary (u:v) , u is the node, v is the node that is in the same motifs as v
    # M_types: dictionary(u,v), u is the motif type, v is the node in the motif types
    # M_type_dict: dictionary(u,v), u is the node, v is the motif types that u are in.

    #H = max(nx.connected_component_subgraphs(G.to_undirected(reciprocal=False)), key=len)
    #mcount = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo)+2)))))
    mcount = np.zeros(len(mo))
    nodes = gr.nodes()
    M_types = [[]]*(len(mo))
    M_instance = {mask[node]:[] for node in nodes}
    M_type_dict = {mask[node]:[] for node in nodes}
    M_instance_dict = {m:[] for m in mo.keys()}
    g2 = gr.to_undirected()
    for node in nodes:
        if node % 100 == 0:
            print(node)
        neighbors = list(g2.neighbors(node))+[node]
        triplets = list(itertools.combinations(neighbors,3))
        triplets = map(list, map(np.sort, triplets))
        for trip in triplets:
            sub_gr = gr.subgraph(trip)
            for k,v in mo.items():
                if (nx.is_isomorphic(v,sub_gr)):
                    masked_trip = map(lambda x: mask[x], trip)
                    M_types[k] = list(set(M_types[k]+masked_trip))
                    M_instance_dict[k].append(masked_trip)
                    for t in masked_trip:
                        M_instance[t] = list(set(M_instance[t]+trip)-{t})
                        M_type_dict[t] = list(set(M_type_dict[t] + [k]))
                    mcount[k] += 1
    """
    isolated_nodes = []
    for k,v in M_type_dict.items():
        if v ==[]:
            isolated_nodes.append(k)
    gs = gr.subgraph(isolated_nodes)
    gs2 = gs.to_undirected()
    print(isolated_nodes)
    for node in isolated_nodes:
        neighbors = list(gs2.neighbors(node))+[node]
        sub_gs = gs.subgraph(neighbors)
        for k,v in ms.items():
            if (nx.is_isomorphic(v,sub_gs)):
                M_types[k+len(mo)] = list(set(M_types[k+len(mo)]+neighbors))
                M_types[k+len(mo)] = list(set(M_types[k+len(mo)]+neighbors))
                for t in neighbors:
                    M_instance[t] = list(set(M_instance[t]+neighbors)-{t})
                    M_type_dict[t] = list(set(M_type_dict[t] + [k+len(mo)]))
                mcount[k+len(mo)] += 1
    """
    return mcount, M_instance, M_types, M_type_dict,M_instance_dict

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def save_dict(mcount,M_instance,M_types,M_type_dict,M_instance_dict,dataset='cora',path="data/dicts/"):
    directory = path+dataset+'/'
    if os.path.isdir(directory):
        f = open(directory+"mcount.pkl",'wb')
        pickle.dump(mcount,f)
        f.close()
        f = open(directory+"M_instance.pkl",'wb')
        pickle.dump(M_instance,f)
        f.close()
        f = open(directory+"M_types.pkl",'wb')
        pickle.dump(M_types,f)
        f.close()
        f = open(directory+"M_type_dict.pkl",'wb')
        pickle.dump(M_type_dict,f)
        f.close()
        f = open(directory+"M_instance_dict.pkl",'wb')
        pickle.dump(M_instance_dict,f)
        f.close
    else:
        os.mkdir(directory)
        f = open(directory+"mcount.pkl",'wb')
        pickle.dump(mcount,f)
        f.close()
        f = open(directory+"M_instance.pkl",'wb')
        pickle.dump(M_instance,f)
        f.close()
        f = open(directory+"M_types.pkl",'wb')
        pickle.dump(M_types,f)
        f.close()
        f = open(directory+"M_type_dict.pkl",'wb')
        pickle.dump(M_type_dict,f)
        f.close()
        f = open(directory+"M_instance_dict.pkl",'wb')
        pickle.dump(M_instance_dict,f)
        f.close


def load_dict(dataset='cora',path="data/dicts/"):
    directory = path+dataset+"/"
    mcount = pickle.load(open(directory+"mcount.pkl",'rb'))
    M_instance = pickle.load(open(directory+"M_instance.pkl",'rb'))
    M_types = pickle.load(open(directory+"M_types.pkl",'rb'))
    M_type_dict = pickle.load(open(directory+"M_type_dict.pkl",'rb'))
    M_instance_dict = pickle.load(open(directory+"M_instance_dict.pkl", 'rb'))
    return mcount,M_instance,M_types,M_type_dict,M_instance_dict


def motif_adjacency(G,M_types,sparse):
    adj_list = []
    n=0
    for nodes in M_types:
        n += len(nodes)
        subgr = G.subgraph(nodes)
        adj = nx.adjacency_matrix(subgr)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + spsparse.eye(adj.shape[0])).todense()
        adj_list.append(adj)
    return adj_list,n


def transform(mtype,n_nodes):
    newm = np.zeros((len(mtype),n_nodes))
    for i in range(len(mtype)):
        newm[i][mtype[i]]=1
    return torch.LongTensor(newm)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def process_small_data(m_instance,n_nodes,n_batch):
    output = [] #it should be N_batch*J*N
    length = len(m_instance)
    for i in range(n_batch):
        index = np.random.choice(length,int(length/1.25))
        output.append(np.array(m_instance[index]))
    return output


def process_large_data(m_instance,n_nodes,n_batch):
    output = []
    splits = np.array_split(np.random.permutation(range(len(m_instance))),n_batch)
    for i in splits:
        output.append(m_instance[i])
    return output


def get_mask(nodes):
    count = 0
    mask = {}
    reverse_mask = {}
    for node in nodes:
        mask[node] = count
        reverse_mask[count] = node
        count += 1
    return mask,reverse_mask


def get_batch(n_batch,M_instance_dict,n_nodes):
    batch_indices = [[]]*len(M_instance_dict)
    for i in range(len(M_instance_dict)):
        batch_indices[i] = process_large_data(np.array(M_instance_dict[i]),n_nodes,n_batch)
    new_batch = [[]] * n_batch
    for i in range(n_batch):
        for j in range(len(batch_indices)):
            new_batch[i].append(batch_indices[j][i])
    return new_batch

def get_negatives(n_batch,M_instance_dict,batch_indices):
    negatives = [[]] * n_batch
    for i in range(n_batch):
        negatives[i] = process_negatives(M_instance_dict,batch_indices[i])
    return negatives

def process_negatives(M_instance_dict,batch_indices):
    negative = [[]]* len(M_instance_dict)
    for i in range(len(M_instance_dict)):
        new = []
        for j in range(len(M_instance_dict)):
            if j == i:
                continue
            if new == []:
                new = batch_indices[j]
            else:
                new = np.concatenate((new,batch_indices[j]))
        index = np.random.permutation(range(new.shape[0]))[:batch_indices[i].shape[0]]
        negative[i] = new[index]
    return negative


def miniBatch(train_index,n_batch):
    splits = np.array_split(np.random.permutation(range(len(train_index))),n_batch)
    return splits


def process_label(label):
    print(label.shape)
    print(label[1])
    new_label = np.zeros((label.shape[0],7))
    for i in range(label.shape[0]):
        new_label[i, int(label[i])] = 1
    return new_label
