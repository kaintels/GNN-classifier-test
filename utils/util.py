import numpy as np
import networkx as nx
import torch
import dgl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, Precision, Recall

def get_feature_and_label(path):
    feature = []
    label = []
    for i in path:
        data = np.load(i)
        feature.append(data["x"])
        label.append(data["y"])

    feature = np.concatenate(feature)
    label = np.concatenate(label)
    print("feature ok")
    return feature, label

def laplacian_mat(feature):
    corr = np.corrcoef(feature.T)
    abs_corr = np.abs(corr)
    adjacency_matrix = abs_corr - np.identity(len(abs_corr))
    laplacian = nx.laplacian_matrix(nx.from_numpy_matrix(adjacency_matrix)).toarray()
    print("lap ok")
    return laplacian

def get_graph(laplacian, feature, label):
    G = nx.from_numpy_matrix(laplacian)
    edge_index = torch.from_numpy(np.array(list(G.edges())))
    x = torch.FloatTensor(feature)
    y = torch.LongTensor(label)
    data = dgl.graph((edge_index.t().contiguous()[0], edge_index.t().contiguous()[1]), num_nodes=len(feature))
    data.ndata['x'] = x
    data.ndata['y'] = y
    data = dgl.add_self_loop(data)
    return data.to("cuda:0")


def metrics(num_class=3):
    acc_metrics = Accuracy().cuda()
    pre_metrics = Precision(num_classes=num_class,average=None).cuda()
    rec_metrics = Recall(num_classes=num_class,average=None).cuda()
    f1_metrics = F1Score(num_classes=num_class,average="macro").cuda()
    confusion = ConfusionMatrix(num_classes=num_class).cuda()

    return acc_metrics, pre_metrics, rec_metrics, f1_metrics, confusion

def train_setting(target_model=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    return criterion, optimizer
