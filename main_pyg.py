import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from glob import glob
from models.model_pyg import GraphConvNet
from utils.util_pyg import get_feature_and_label, laplacian_mat, get_graph, metrics, train_setting, set_device

from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric import seed_everything

device = set_device()

seed_num = 777
EPOCH = 5000
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
seed_everything(seed_num)

model = GraphConvNet().to(device)

x_train, x_test, y_train, y_test = get_feature_and_label()


x_train_lap = laplacian_mat(x_train)
x_test_lap = laplacian_mat(x_test)

train_graph = get_graph(x_train_lap, x_train, y_train)
test_graph = get_graph(x_test_lap, x_test, y_test)

print(train_graph)
print(test_graph)

acc_metrics, pre_metrics, rec_metrics, f1_metrics, confusion = metrics(2)
criterion, optimizer = train_setting(target_model=model)

for epoch in range(EPOCH):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph)
    loss = criterion(out, train_graph.y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 1:
        print(loss.item())

model.eval()
with torch.no_grad():
    outx = model(test_graph)
    acc_metrics(outx, test_graph.y)
    pre_metrics(outx, test_graph.y)
    rec_metrics(outx, test_graph.y)
    confusion(outx, test_graph.y)

    print(classification_report(outx.argmax(1).cpu(), test_graph.y.cpu()))
    print(confusion_matrix(outx.argmax(1).cpu(), test_graph.y.cpu()))



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