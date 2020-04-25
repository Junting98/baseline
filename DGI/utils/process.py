import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
mapping = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
           "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
mapping1 = {"Agents": 0, "IR": 1, "DB": 2, "AI": 3, "HCI": 4, "ML": 5}


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess(dataset='data/pubmed', directed='directed'):
    f = open("../data/" + dataset + '.cites', 'r')
    mask, labels, attribute = get_dataset(dataset)
    G = None
    if directed == "directed":
        G = nx.DiGraph()
    for line in f.readlines():
        edge = line.strip().split()
        if edge[0] != edge[1]:
            G.add_edge(int(mask[edge[0]]), int(mask[edge[1]]))
    return mask, labels, attribute, G


def get_dataset(dataset='data/pubmed'):
    f = open("../data/" + dataset + ".content", 'r')
    lines = f.readlines()
    mask = {}
    labels = np.zeros(len(lines))
    if 'cora' in dataset:
        attribute = np.zeros((len(lines), 1433))
    else:
        attribute = np.zeros((len(lines), 500))
    idx = 0
    for line in lines:
        line = line.strip().split()
        mask[line[0]] = idx
        attribute[idx] = np.array(list(map(int, line[1:-1])))
        if 'cora' in dataset:
            labels[idx] = mapping[line[-1]]
        else:
            labels[idx] = int(line[-1]) - 1
        idx += 1
    return mask, labels, attribute


def train_test_split(n_nodes, testing):
    # splits = np.array_split(np.random.permutation(range(n_nodes)), 5)
    #
    # train_index = np.concatenate(splits[:3], axis=0)
    # val_index = torch.LongTensor(splits[3])
    # test_index = torch.LongTensor(splits[-1])
    print(n_nodes)
    print(testing)
    splits = np.array_split(np.random.permutation(range(n_nodes)), 20)
    n_testing = int(testing * 100 / 5)
    n_val = 4

    test_index = np.concatenate(splits[:n_testing], axis=0)
    val_index = np.concatenate(splits[n_testing:n_testing + n_val], axis=0)
    train_index = np.concatenate(splits[n_testing + n_val:], axis=0)
    return train_index, val_index, test_index


def load_social_data(dataset):
    mat_contents = sio.loadmat(dataset + ".mat")
    adj = mat_contents["Network"].toarray()
    if not os.path.exists(dataset + '.cites'):
        f = open(dataset + '.cites', "w+")
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] != 0:
                    f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    adj = np.array(adj, dtype=int)
    features = mat_contents["Attributes"]
    label = mat_contents["Label"]
    G = nx.convert_matrix.from_numpy_matrix(adj, parallel_edges=False, create_using=nx.Graph)
    return label.flatten() - 1, features, G


def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    # """Load data."""
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    #
    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)
    #
    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended
    #
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)

    mask, temp_label, features, G = preprocess(dataset_str)
    G = G.subgraph(max(nx.connected_component_subgraphs(G.copy().to_undirected()), key=len).nodes())
    G = G.to_undirected()
    temp_label = temp_label[G.nodes()]
    features = features[G.nodes()]
    labels = np.zeros((len(temp_label), len(np.unique(temp_label))))
    print(labels.shape)
    for i in range(len(temp_label)):
        labels[i, int(temp_label[i])] = 1

    adj = nx.adjacency_matrix(G)
    idx_train, idx_val, idx_test = train_test_split(len(list(G.nodes())), 0.2)

    return adj, csr_matrix(features), labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
