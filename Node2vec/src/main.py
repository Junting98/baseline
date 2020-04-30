'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import networkx as nx
import node2vec
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
from sklearn.model_selection import KFold

mapping = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
           "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
mapping1 = {"Agents": 0, "IR": 1, "DB": 2, "AI": 3, "HCI": 4, "ML": 5}


def get_mask(nodes):
    count = 0
    mask = {}
    reverse_mask = {}
    for node in nodes:
        mask[node] = count
        reverse_mask[count] = node
        count += 1
    return mask, reverse_mask


def preprocess(dataset='data/cora', directed='directed', split=0.8):
    f = open("../../data/" + dataset + '.cites', 'r')
    mask, labels, attribute = get_dataset(dataset)
    G = None
    if directed == "directed":
        G = nx.DiGraph()
    for line in f.readlines():
        edge = line.strip().split()
        if edge[0] != edge[1]:
            G.add_edge(int(mask[edge[0]]), int(mask[edge[1]]))
    return mask, labels, G, None


def get_dataset(dataset):
    f = open("../../data/" + dataset + ".content", 'r')
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


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='/home/jtwang/baseline/data/europe.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='/home/jtwang/baseline/Node2vec/emb/europe.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    print(model)
    model.wv.save_word2vec_format(args.output)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


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


def predict(dataset, labels, G):
    G = G.subgraph(max(nx.connected_component_subgraphs(G.copy().to_undirected()), key=len).nodes())
    labels = labels[G.nodes()]
    new_mask, _ = get_mask(G.nodes())
    embeddings = np.zeros((len(labels), 128))
    f = open('../emb/{}.emb'.format(dataset), 'r+')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        if int(line[0]) not in new_mask.keys():
            continue
        embeddings[new_mask[int(line[0])]] = np.array(line[1:], dtype=np.float64)
    kf = KFold(n_splits=5)
    acc = 0
    for _ in range(5):
        train_index, val_index, test_index = train_test_split(len(labels), 0.6)
        clf = LogisticRegression(penalty='l2', multi_class='ovr', max_iter=100)
        model = clf.fit(embeddings[train_index], labels[train_index])
        yhat = model.predict(embeddings[test_index])
        curr_acc = np.sum((yhat == labels[test_index])) / float(len(test_index))
        acc += curr_acc
        print(curr_acc)
    print(acc / 5.0)


if __name__ == "__main__":
    dataset = 'europe'
    # mask, labels, G, _ = preprocess(dataset)
    # f = open("/home/jtwang/baseline/data/{}.cites".format(dataset), 'r+')
    # g = open('/home/jtwang/baseline/data/{}.cites2'.format(dataset), 'w+')
    # for line in f.readlines():
    #     line = line.strip().split()
    #     node1 = line[0]
    #     node2 = line[1]
    #     if node1 is node2:
    #         continue
    #     idx1 = mask[node1]
    #     idx2 = mask[node2]
    #     g.write(str(idx1) + '\t' + str(idx2) + '\n')
    # args = parse_args()
    # main(args)
    dest = open('/home/jtwang/baseline/Node2vec/emb/{}.content'.format(dataset), 'w+')
    label = open('/home/jtwang/baseline/data/{}.label.txt'.format(dataset), 'r+')
    emb = open('/home/jtwang/baseline/Node2vec/emb/{}.emb'.format(dataset), 'r+')
    label_map = {}
    for line in label.readlines()[1:]:
        line = line.strip().split()
        label_map[line[0]] = line[1]

    for line in emb.readlines()[1:]:
        line = line.strip().split()
        node = line[0]
        dest.write(' '.join(line) + ' '+ label_map[node] +'\n')


    # predict(dataset, labels, G)
