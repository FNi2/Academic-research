import torch
from torch import nn
from d2l import torch as d2l
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#实现一个具有单隐藏层的多层感知机，它博涵256个隐藏单元
