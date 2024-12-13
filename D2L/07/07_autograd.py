import torch
print('1.自动梯度计算')
x=torch.arange(4.0,requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量
print('x:', x)
y=2*torch.dot(x,x) # 2.记录目标值的计算
print('y:', y)
y.backward()   # 3.执行它的反向传播函数
print('x.grad:',x.grad) # 4.访问得到的梯度
print('x.grad == 4*x:',x.grad==4*x)
print('2.计算另一个函数')
x.grad.zero_()
y=x.sum()
y.backward()
print('x.grad:',x.grad)
print('3.非标量变量的反向传播')
x.grad.zero_()
y=x*x
y.sum().backward()
print('x.grad:',x.grad)
print('4.将某些计算移动到记录的计算图之外')
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
z.sum().backward()
print('x.grad==u:',x.grad==u)
x.grad.zero_()
y.sum().backward()
print('x.grad==2*x:',x.grad==2*x)
print('5.Python控制流的梯度计算')
def f(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()
print('6.a.grad==d/a',a.grad==d/a)
