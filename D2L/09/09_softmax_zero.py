import torch
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np


batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
num_inputs=784 #展平图像为向量
num_outputs=10 # 有10个类所以模型输出为10
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)#定义权重w
b=torch.zeros(num_outputs,requires_grad=True)

# 定义softmax
def softmax(X):
    X_exp=torch.exp(X)#对每个元素做指数运算
    partition =X_exp.sum(1,keepdim=True)#按照行求和
    return X_exp/partition #矩阵中的各个元素/对应行元素之和
#验证一下是否是正确的
X=torch.normal(0,0.01,(2,5))# 创建均值为0方差为1的两行五列的X
X_prob=softmax(X)
print('1.验证softmax:',X_prob,X_prob.sum(1))
#实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b) # -1，每次喂数据的量，就是batchsize

y=torch.tensor([0,2])
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print('2.根据标号拿出预测值:',y_hat[[0,1],y])
# 实现交叉熵损失
def cross_entropy(y_hat,y): #给定预测和真实标号Y
    return -torch.log(y_hat[range(len(y_hat)),y])# 锁定y轴在x轴上根据labels收取预测值，交叉熵损失中除了真值=1，其他都是0，这里直接算针织对应的预测概率
print('3.交叉熵损失:',cross_entropy(y_hat,y))

#将预测类别与真实元素y进行比较
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1: #shape和列数大于1的时候
        y_hat=y_hat.argmax(axis=1)#把每一行元素最大的下标存到y_hat
    cmp=y_hat.type(y.dtype)==y #y_hat和y的数据类型转换，作比较变成布尔
    return float(cmp.type(y.dtype).sum())#转换成和y一样的形状求和
print('4.预测正确的概率:',accuracy(y_hat,y)/len(y))# 预测正确的样本数除以y的长度就是预测正确的概率

#计算模型在数据迭代器上的精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()#将模型设置为评估模式，输入后得出的结果用来评估模型的准确率，不做反向传播
    metric =Accumulator(2) # 累加器
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1] #返回分类正确的样本数和总样本数

# accumulator的实现
class Accumulator: #作用是累加
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
if __name__=='__main__':
    print(evaluate_accuracy(net,test_iter))

# softmax回归的训练
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y),accuracy(y_hat,y),
                       y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y),
                   y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

class Animator:
    def __init__(self,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',
                 fmts=('-','m--','g-.','r:'),nrows=1,ncols=1,figsize=(3.5,2.5)):
        if legend is None:
            legend=[]
        d2l.use_svg_display()
        self.fig,self.axes=d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes=[self.axes, ]
        self.config_axes=lambda :d2l.set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts=None,None,fmts

    def add(self,x,y):
        if not hasattr(y,"__len__"):
            y=[y]
        n=len(y)
        if not hasattr(x, "__len__"):
            x=[x]*n
        if not self.X:
            self.X=[[]for _ in range(n)]
        if not self.Y:
            self.Y=[[]for _ in range(n)]
        for i ,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator=Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics

lr = 0.1
def updater(batch_size):
     return d2l.sgd([w,b],lr,batch_size)

if __name__ == '__main__':
    num_epochs=10
    train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)


# 对图像进行分类的预测

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues=d2l.get_fashion_mnist_labels(y)
    preds=d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles=[true+'\n'+pred for true,pred in zip(trues,preds)]
    d2l.show_images(
        X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n]
    )
    d2l.plt.show()
if __name__ == '__main__':
    predict_ch3(net,test_iter)




