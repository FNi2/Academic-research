import torch
# 张量的创建
x1 = torch.arange(12)
print('1.有12个元素的张量：\n',x1)
print('2.张量的形状：\n',x1.shape)
print('3.张量中元素的总数：\n',x1.numel())

y1=x1.reshape((3,4)) # 改变一个张量的形状而不改变元素数量和元素值
print('4.改变形状后的张量\n',y1)

z=torch.zeros((2,3,4))
print('5.全0张量\n',z)  # 创建一个张量，其中所有元素都设置为0
w=torch.ones((2,3,4))
print('6.全1张量\n',w)# 创建一个张量，其中所有元素都设置为1
q=torch.tensor([[1,2,3,4],[2,1,4,3],[2,3,4,1]])
print('7.特定值张量\n',q) # 通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值

# 张量的运算

x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
print('8.张量的运算，加减乘除求幂\n',x+y,x-y,x*y,x/y,x**y)  # **运算符是求幂运算
print('9.按元素做指数运算\n',torch.exp(x))

x2=torch.arange(12,dtype=torch.float32).reshape(3,4)
y2=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print('10.连结：\n',torch.cat((x2,y2),dim=0),torch.cat((x2,y2),dim=1)) # 连结（concatenate） ,将它们端到端堆叠以形成更大的张量。
print('11.逻辑运算符 构建二元张量:\n',x2==y2) # 通过 逻辑运算符 构建二元张量
print('12.张量所有元素的和:\n',x2.sum()) # 张量所有元素的和

# 广播机制
a=torch.arange(3).reshape(3,1)
b=torch.arange(2).reshape(1,2)
print('13.广播机制:\n',a+b)

# 元素访问
x4=torch.arange(12,dtype=torch.float32).reshape(3,4)
print('14.元素访问:\n',x4[-1],x4[1:3])  # 用 [-1] 选择最后一个元素， 用 [1:3] 选择第二个和第三个元素]
x4[1,2]=9 # 写入元素。
x4[0:2,:]=12 # 写入元素。
print('15.写入元素:\n',x4)

#转换为其他python对象

a2= x.numpy()
print('16.转换为numpy张量:\n',type(a2))
b2=torch.tensor(a2)
print('17.转换为torch张量:\n',type(b2))

a3=torch.tensor([3.5])
print('18.转换为python标量:\n',a3,a3.item(),float(a3),int(a3))

