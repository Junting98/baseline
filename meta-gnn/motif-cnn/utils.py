from __future__ import print_function
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import json
import random
import subprocess
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from metrics import *
import scipy.io as sio

mapping = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
           "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
mapping1 = {"Agents": 0, "IR": 1, "DB": 2, "AI": 3, "HCI": 4, "ML": 5}


def calc_motif_submatch(motif, motif_def, shape, dataset):
    '''Compute motif using subgraph matching'''
    graph_path = '../motif-cnn/data/{}/links.txt'.format(dataset)
    node_info_path = '../motif-cnn/data/{}/node_info.txt'.format(dataset)
    motif_path = '../motif-cnn/data/{}/motif.json'.format(dataset)
    result_path = '../vflib/instances.txt'
    submatch_path = '../vflib/call_submatch.py'

    def submatch(motif_json):
        motif_json = {'1': motif_json}
        with open(motif_path, 'w') as f:
            json.dump(motif_json, f)
        try:
            print('Call subgraph match')
            subprocess.call(['python', submatch_path, '-G', graph_path, '-N', node_info_path, '-M', motif_path],
                            cwd='../vflib/')
        except Exception as e:
            print(e)
            print('Subgraph match failed')
            exit(1)
        print('Parse results')
        instances = []
        with open(result_path, 'r') as f:
            for line in f:
                instances.append(tuple([int(x) for x in line.strip().split()]))
        return instances

    print("WTF")
    quit(0)
    with open(motif_def, 'r') as f:
        motif_json = json.load(f)
    try:
        motif_json = motif_json[motif]
    except KeyError as e:
        print('Motif ' + motif + ' definition not found!')
        exit(1)
    ret_ind = motif_json.pop('m', None)
    instances = submatch(motif_json)
    ret = []
    for i in range(len(ret_ind)):
        ret.append(sp.dok_matrix(shape))
    for ins in instances:
        for i in range(len(ret_ind)):
            v1 = ins[ret_ind[i][0]]
            v2 = ins[ret_ind[i][1]]
            ret[i][v1, v2] += 1
    ret = [x.tocsr() for x in ret]
    return ret


def load_motif_adj(motifs, dataset_str, types, n_nodes, motif_def='./motif_def.json', calc_motif=True):
    # motif adjacency matrix
    A = []

    # load original graph (directed)
    # file format
    # first line: n_nodes n_edges
    # rest lines: source_node target_node
    with open('data/{}/links.txt'.format(dataset_str), 'r') as f:
        adj = sp.dok_matrix((n_nodes, n_nodes))
        for line in f:
            s, t = [int(x) for x in line.strip().split()]
            adj[s, t] = 1
    adj = adj.tocsr()
    # for motif in motifs:
    #     if motif == 'edge':
    #         A.append([adj])
    #         continue
    #     motif_file_exist = os.path.isfile('data/{}/{}.motif'.format(dataset_str, motif))
    #     if calc_motif or (not motif_file_exist):
    #         A_m = calc_motif_submatch(motif, motif_def, (n_nodes, n_nodes), dataset_str)
    #         A.append(A_m)
    #         with open('data/{}/{}.motif'.format(dataset_str, motif), 'wb') as f:
    #             pkl.dump(A_m, f)
    #     else:
    #         with open('data/{}/{}.motif'.format(dataset_str, motif), 'rb') as f:
    #             A.append(pkl.load(f))
    A.append([adj])
    return A


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


def preprocess(dataset='data/cora', directed='directed', split=0.8):
    f = open("/home/junting/Downloads/baselines/Meta-GNN/motif-cnn/data/{}/".format(dataset) + dataset + '.cites', 'r')
    mask, labels, attribute = get_dataset(dataset)
    G = None
    if directed == "directed":
        G = nx.DiGraph()
    for line in f.readlines():
        edge = line.strip().split()
        if edge[0] != edge[1]:
            G.add_edge(int(mask[edge[0]]), int(mask[edge[1]]))
    return mask, labels, G, attribute


def get_dataset(dataset):
    f = open("/home/junting/Downloads/baselines/Meta-GNN/motif-cnn/data/{}/".format(dataset) + dataset + ".content",
             'r')
    lines = f.readlines()
    mask = {}
    labels = []
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
            # labels[idx] = mapping[line[-1]]
            labels.append((idx, mapping[line[-1]]))
        elif 'citeseer' in dataset:
            # labels[idx] = mapping1[line[-1]]
            labels.append((idx, mapping1[line[-1]]))
        else:
            # labels[idx] = int(line[-1]) - 1
            labels.append((idx, int(line[-1]) - 1))
        idx += 1
    return mask, labels, attribute


def get_mask(nodes):
    count = 0
    mask = {}
    reverse_mask = {}
    for node in nodes:
        mask[node] = count
        reverse_mask[count] = node
        count += 1
    return mask, reverse_mask


def load_social_data(dataset):
    mat_contents = sio.loadmat('/home/junting/Downloads/Meta-GNN/motif-cnn/data/{}/'.format(dataset) + dataset + ".mat")
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


def load_data(dataset_str, motifs, load_ind=True, calc_motif=True, motif_def='./motif_def.json'):
    '''Load HIN dataset
    load_ind: load train/val/test indices
    calc_motif: Re-calculate motif
    motif_def: Path to motif definition json'''
    print('Load {} dataset'.format(dataset_str))
    n_motifs = len(motifs)
    # load node type info
    # file format: node_id \t type_id \t info_text
    # types = dict()
    # with open('data/{}/node_info.txt'.format(dataset_str), 'r') as f:
    #     for line in f:
    #         tok = line.strip().split()
    #         node_id, type_id = int(tok[0]), int(tok[1])
    #         types[node_id] = type_id
    # n_nodes = len(types)
    # # turn into numpy vector
    # types = np.array([type_id for (node_id, type_id) in sorted(types.items())])
    #
    # # load features
    # # file format: dumped scipy sparse matrix
    # with open('data/{}/features.pkl'.format(dataset_str), 'rb') as f:
    #     try:
    #         features = pkl.load(f)
    #     except UnicodeDecodeError as e:
    #         features = pkl.load(f, encoding='latin-1')
    # assert features.shape[0] == n_nodes
    # print(features)
    # quit()
    # # compute motif adjacency matrix
    # # format: list of lists
    # #         each element (corresponding to each motif) is a list of scipy
    # #         csr matrices (N * N, corresponding to each motif position)
    # adj = load_motif_adj(motifs, dataset_str, types,
    #                      calc_motif=calc_motif, motif_def=motif_def)
    #
    # # load label
    # # file format: node_id \t label_id
    # # note that only labeled nodes are included in the file
    # labels = []
    # with open('data/{}/labels.txt'.format(dataset_str), 'rb') as f:
    #     for line in f:
    #         node_id, label = [int(x) for x in line.strip().split()]
    #         labels.append((node_id, label))
    # n_labeled = len(labels)
    if 'blog' in dataset_str or 'flickr' in dataset_str:
        label, features, G = load_social_data(dataset_str)
        features = features.todense()
        mask = {}
        for i in range(len(label)):
            mask[str(i)] = i
    else:
        mask, label, G, features = preprocess(dataset_str)
    G = G.subgraph(max(nx.connected_component_subgraphs(G.copy().to_undirected()), key=len).nodes())
    new_mask, _ = get_mask(G.nodes())
    # f = open('../motif-cnn/data/{}/node_info.txt'.format(dataset_str), 'w+')
    # g = open('../motif-cnn/data/{}/links.txt'.format(dataset_str), 'w+')
    # f1 = open('../motif-cnn/data/{}/{}.cites'.format(dataset_str, dataset_str), 'r+')
    # for line in f1.readlines():
    #     line = line.strip().split()
    #     node1 = mask[line[0]]
    #     node2 = mask[line[1]]
    #     if node1 in new_mask.keys():
    #         g.write(str(new_mask[node1]) + '\t' + str(new_mask[node2]) + '\n')
    # g.close()
    #
    # for node in G.nodes():
    #     f.write(str(new_mask[node]) + '\t0\tabc\n')
    # f.close()

    types = dict()
    with open('data/{}/node_info.txt'.format(dataset_str), 'r') as f:
        for line in f:
            tok = line.strip().split()
            node_id, type_id = int(tok[0]), int(tok[1])
            types[node_id] = type_id
    n_nodes = len(types)
    # turn into numpy vector
    types = np.array([type_id for (node_id, type_id) in sorted(types.items())])
    adj = load_motif_adj(motifs, dataset_str, types, n_nodes,
                         calc_motif=calc_motif, motif_def=motif_def)
    labels = []
    features = features[G.nodes()]
    for pair in range(len(label)):
        labels.append((new_mask[pair], label[pair]))
    # for pair in label:
    #     if pair[0] not in new_mask.keys():
    #         continue
    #     labels.append((new_mask[pair[0]], pair[1]))
    # shuffle labels
    if load_ind:
        #     try:
        #         train_ind = np.loadtxt('data/{}/train.ind'.format(dataset_str), dtype=int)
        #         val_ind = np.loadtxt('data/{}/val.ind'.format(dataset_str), dtype=int)
        #         test_ind = np.loadtxt('data/{}/test.ind'.format(dataset_str), dtype=int)
        #     except Exception as e:
        #         print('load index failed')
        #         exit(1)

        train_ind, val_ind, test_ind = train_test_split(n_nodes, 0.6)

        train_mask = np.zeros(n_nodes, dtype=np.bool)
        val_mask = np.zeros(n_nodes, dtype=np.bool)
        test_mask = np.zeros(n_nodes, dtype=np.bool)

        for ind in train_ind:
            train_mask[ind] = 1
        for ind in val_ind:
            val_mask[ind] = 1
        for ind in test_ind:
            test_mask[ind] = 1
    else:
        n_train = int(n_labeled * 0.1)
        n_test = int(n_labeled * 0.8)
        n_val = int(n_labeled * 0.1)

        train_mask = np.zeros(n_nodes, dtype=np.bool)
        val_mask = np.zeros(n_nodes, dtype=np.bool)
        test_mask = np.zeros(n_nodes, dtype=np.bool)
        random.shuffle(labels)
        for i in range(n_labeled):
            if i < n_train:
                train_mask[labels[i][0]] = 1  # labels element is in format (node_id, label)
            elif i >= n_train and i < n_train + n_val:
                val_mask[labels[i][0]] = 1
            elif i >= n_labeled - n_test:
                test_mask[labels[i][0]] = 1
        train_ind = np.where(train_mask)[0]
        val_ind = np.where(val_mask)[0]
        test_ind = np.where(test_mask)[0]

    labels = to_one_hot(labels, n_nodes, multilabel=False)
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def to_one_hot(labels, N, multilabel=False):
    '''In: list of (node_id, label) tuples, #nodes N
       Out: N * |label| matrix'''
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded


def sparse_to_tuple(sparse_mx):
    '''Convert sparse matrix to tuple representation.'''

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = csr_matrix(mx).tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype('int64')
        values = np.array(values).astype('float32')
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    '''Row-normalize feature matrix and convert to tuple representation'''
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    '''Symmetrically normalize adjacency matrix.'''
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum.astype(float), -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return (d_mat_inv * adj).tocoo()


def preprocess_adj(adj):
    '''Preprocess adjacency matrix for Motif motif-cnn model and convert to tuple representation.
       Return - A list of normalized motif adjacency matrices in tuple format.
                Shape: (n_motifs, (coords, values, mat_shape)), mat_shape should be of (n_positions, N, N)'''
    normalized_adjs = []
    for m in range(0, len(adj)):
        normalized_adj_m = []
        normalized_adj_m.append(sp.eye(adj[m][0].shape[0]))
        for k in range(0, len(adj[m])):
            adj_normalized = normalize_adj(adj[m][k])
            normalized_adj_m.append(adj_normalized)
        normalized_adjs.append(normalized_adj_m)

    normalized_adjs = sparse_to_tuple(normalized_adjs)
    return normalized_adjs


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    '''Construct feed dictionary.'''
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def compute_f1(preds, labels, mask):
    macro_f1 = masked_macro_f1(preds, labels, mask)
    micro_f1 = masked_micro_f1(preds, labels, mask)
    return macro_f1, micro_f1
