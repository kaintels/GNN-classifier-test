import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
from sklearn.datasets import load_breast_cancer
from torch_geometric.data import Data
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, Precision, Recall
dtype=torch.float16

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

device = set_device()

def get_feature_and_label():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def laplacian_mat(feature):
    corr = np.corrcoef(feature.T)
    abs_corr = np.abs(corr)
    adjacency_matrix = abs_corr - np.identity(len(abs_corr))
    laplacian = nx.laplacian_matrix(nx.from_numpy_array(adjacency_matrix)).toarray()
    print("lap ok")
    return laplacian

def get_graph(laplacian, feature, label):
    G = nx.from_numpy_array(laplacian)
    edge_index = torch.from_numpy(np.array(list(G.edges())))
    x = torch.FloatTensor(feature)
    y = torch.LongTensor(label)
    data = Data(x=x, 
                y=y,
                edge_index=edge_index.T)
    return data.to(device)


def metrics(num_class=3):
    acc_metrics = Accuracy(num_classes=num_class, task='multiclass').to(device)
    pre_metrics = Precision(num_classes=num_class,task='multiclass', average=None).to(device)
    rec_metrics = Recall(num_classes=num_class,task='multiclass',average=None).to(device)
    f1_metrics = F1Score(num_classes=num_class,task='multiclass',average="macro").to(device)
    confusion = ConfusionMatrix(num_classes=num_class,task='multiclass').to(device)

    return acc_metrics, pre_metrics, rec_metrics, f1_metrics, confusion

def train_setting(target_model=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    return criterion, optimizer