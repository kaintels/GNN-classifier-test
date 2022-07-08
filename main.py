import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import torch
import dgl
from glob import glob
from dgl.nn.pytorch import GraphConv
from models.model import GraphConvNet
from utils.util import get_feature_and_label, laplacian_mat, get_graph, metrics, train_setting

seed_num = 777
EPOCH = 5000
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
dgl.seed(seed_num)

model = GraphConvNet().cuda()

train_path = glob("./training/*.npz")
test_path = glob("./testing/*.npz")

x_train, y_train = get_feature_and_label(train_path)
x_test, y_test = get_feature_and_label(test_path)

x_train_lap = laplacian_mat(x_train)
x_test_lap = laplacian_mat(x_test)

train_graph = get_graph(x_train_lap, x_train, y_train)
test_graph = get_graph(x_test_lap, x_test, y_test)

print(train_graph)
print(test_graph)

acc_metrics, pre_metrics, rec_metrics, f1_metrics, confusion = metrics(3)
criterion, optimizer = train_setting(target_model=model)

for epoch in range(EPOCH):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph, train_graph.ndata['x'])
    loss = criterion(out, train_graph.ndata['y'])
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 1:
        print(loss.item())

model.eval()
with torch.no_grad():
    out = model(test_graph, test_graph.ndata['x'])
    acc_metrics(out, test_graph.ndata['y'])
    pre_metrics(out, test_graph.ndata['y'])
    rec_metrics(out, test_graph.ndata['y'])
    confusion(out, test_graph.ndata['y'])

print('Test Accuracy: ', 100.*acc_metrics.compute().item(), "%")
print('Test each Pre: ', 100.*pre_metrics.compute())
print('Test each Rec: ', 100.*rec_metrics.compute())

f1_list = []
for i in range(len(pre_metrics.compute())):
    f1 = 2/(1/pre_metrics.compute()[i]+1/rec_metrics.compute()[i]) * 100
    f1_list.append(f1)

print(f1_list)
print("mean F1 : ", torch.mean(torch.tensor(f1_list)).item(), "%")

print('Comfusion : ', confusion.compute())