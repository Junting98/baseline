import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn import preprocessing
mapping = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
           "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
mapping1 = {"Agents": 0, "IR": 1, "DB": 2, "AI": 3, "HCI": 4, "ML": 5}

###############################################
# Some code adapted from tkipf/gcn            #
# https://github.com/tkipf/gcn                #
###############################################


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess(dataset='data/cora', directed='directed', split=0.8):
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


def get_dataset(dataset):
    f = open("../data/" +dataset + ".content", 'r')
    lines = f.readlines()
    mask = {}
    labels = np.zeros(len(lines))
    if 'cora' in dataset:
        attribute = np.zeros((len(lines), 1433))
    elif 'citeseer' in dataset:
        attribute = np.zeros((len(lines), 3703))
    else:
        attribute = np.zeros((len(lines), 500))
    idx = 0
    for line in lines:
        line = line.strip().split()
        mask[line[0]] = idx
        attribute[idx] = np.array(map(int, line[1:-1]))
        if 'cora' in dataset:
            labels[idx] = mapping[line[-1]]
        elif 'citeseer' in dataset:
            labels[idx] = mapping1[line[-1]]
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
    splits = np.array_split(np.random.permutation(range(n_nodes)), 20)
    n_testing = int(testing * 100 / 5)
    n_val = 4

    test_index = np.concatenate(splits[:n_testing], axis=0)
    val_index = np.concatenate(splits[n_testing:n_testing + n_val], axis=0)
    train_index = np.concatenate(splits[n_testing + n_val:], axis=0)
    return train_index, val_index, test_index


def load_data(dataset_str):
    """Read the data and preprocess the task information."""
    # dataset_G = "data/{}-airports.edgelist".format(dataset_str)
    # dataset_L = "data/labels-{}-airports.txt".format(dataset_str)
    # label_raw, nodes = [], []
    # with open(dataset_L, 'r') as file_to_read:
    #     while True:
    #         lines = file_to_read.readline()
    #         if not lines:
    #             break
    #         node, label = lines.split()
    #         if label == 'label': continue
    #         label_raw.append(int(label))
    #         nodes.append(int(node))
    # lb = preprocessing.LabelBinarizer()
    # labels = lb.fit_transform(label_raw)
    # G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    # adj = nx.adjacency_matrix(G, nodelist=nodes)
    # features = sp.csr_matrix(adj)
    # print(labels.shape)

    mask, temp_label, features, G = preprocess(dataset_str)

    G = G.subgraph(max(nx.connected_component_subgraphs(G.copy().to_undirected()), key=len).nodes())
    G = G.to_undirected()
    adj = nx.adjacency_matrix(G)
    features = adj
    temp_label = temp_label[G.nodes()]

    labels = np.zeros((len(temp_label), len(np.unique(temp_label))))
    print(labels.shape)
    for i in range(len(temp_label)):
        labels[i, int(temp_label[i])] = 1

    # Randomly split the train/validation/test set
    indices = np.arange(adj.shape[0]).astype('int32')
    np.random.shuffle(indices)
    # idx_train = indices[:adj.shape[0] // 3]
    # idx_val = indices[adj.shape[0] // 3: (2 * adj.shape[0]) // 3]
    # idx_test = indices[(2 * adj.shape[0]) // 3:]
    idx_train, idx_val, idx_test = train_test_split(len(indices), 0.2)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    degreeValues = set(degreeNode)

    neighbor_list = []
    degreeTasks = []
    adj = adj.todense()
    for value in degreeValues:
        degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
        degreeTasks.append((value, degreePosition))

        d_list = []
        for idx in degreePosition:
            neighs = [int(i) for i in range(adj.shape[0]) if adj[idx, i] > 0]
            d_list += neighs
        neighbor_list.append(d_list)
        assert len(d_list) == value * len(degreePosition), 'The neighbor lists are wrong!'
    return adj, csr_matrix(features), y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
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
