import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
#读取数据集
trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
print('训练数据集：',len(mnist_train),'测试数据集：',len(mnist_test))
print('训练数据集图片大小：',mnist_train[0][0].shape)

#两个可视化数据集的函数
def get_fashion_mnist_labels(labels): #返回fashion_mnist数据集的文本标签
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels ]
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_rows*scale,num_cols*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i ,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
#几个样本的图像及其相应的标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28,  28), 2, 9, titles=get_fashion_mnist_labels(y));
d2l.plt.show()

#读取一小批量数据，大小为batchsize
batch_size=256
def get_dataloader_workers(): #使用4个进程来读取数据
    return 4
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
timer=d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f}sec')
# 便于重用函数
def load_data_fasion_mnist(batch_size,resize:None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return(data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers()),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=get_dataloader_workers()))