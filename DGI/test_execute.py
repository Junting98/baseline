import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from models import DGI, LogReg
from utils import process,data_process
import networkx as nx
from sklearn.model_selection import KFold

dataset = 'cora'
root = "/home/junting/Downloads/baselines/DGI/"
# training params
batch_size = 1
nb_epochs = 10000
patience = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 128
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

#adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
mask,labels,features,G = data_process.preprocess(dataset=root+"/data/"+dataset)
G = G.to_undirected()
labels = data_process.process_label(labels)

adj = nx.adjacency_matrix(G)
features = data_process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
kf = KFold(n_splits=5,shuffle=False)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
#idx_train = torch.LongTensor(idx_train)
#idx_val = torch.LongTensor(idx_val)
#idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
    print(logits.shape)
    print(lbl.shape)
    loss = b_xent(logits, lbl)
    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

print(features.shape)
embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
for idx_train, idx_test in kf.split(features.squeeze()):
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    if torch.cuda.is_available():
        idx_train.cuda()
        idx_test.cuda()
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

