import torch
x=torch.tensor([3.0])
y=torch.tensor([2.0])
#标量
print('1.标量只有一个元素：\n',x+y,x*y,x/y,x**y)
x2=torch.arange(4)
#向量
print('2.向量视为标量值组成的列表：\n',x2)
print('3.访问张量元素,长度，形状：\n',x2[3],len(x2),x2.shape)
# 矩阵
A=torch.arange(20).reshape(5,4)
print('4.创建矩阵：\n',A)
print('5.矩阵的转置：\n',A.T)
#矩阵计算
A2=torch.arange(20,dtype=torch.float32).reshape(5,4)
B2=A2.clone()# 通过分配新内存，将A的一个副本分配给B
print('6.矩阵相加\n',A2+B2)# 矩阵相加
print('7.矩阵相乘：\n',A2*B2)# 矩阵相乘
#矩阵和标量计算
a3=2
X3=torch.arange(24).reshape(2,3,4)
print('8.标量+矩阵：\n',a3+X3)
print('9.标量*矩阵：\n',a3*X3)
#矩阵求和
A3=torch.arange(40).reshape(2,5,4)
print('10.矩阵求和：\n',A3,A3.sum())

A3_sum_axis0=A3.sum(axis=0)# 沿着 axis0 的方向进行求和
print('11.沿着 axis0 的方向进行求和\n',A3_sum_axis0,A3_sum_axis0.shape)
A3_sum_axis1=A3.sum(axis=1)# 沿着axis1的方向进行求和
print('12.沿着 axis1 的方向进行求和\n',A3_sum_axis1,A3_sum_axis1.shape)
A3_sum_axis01=A3.sum(axis=[0,1])# 沿着两个方向求和
print('13.沿着两个方向进行求和\n',A3_sum_axis01,A3_sum_axis01.shape)


#求均值
A4=torch.arange(20,dtype=torch.float32).reshape(5,4)
print('14.求均值\n',A4.mean(dtype=float))
print('15.求均值\n',A4.mean(axis=0,dtype=float))

#保持维度不变，方便使用广播机制
sum_A4=A.sum(axis=1,keepdims=True)
print('16.保持维度不变\n',sum_A4)
print('17.某个轴的累积总和\n',A4.cumsum(axis=0))

#向量的点积
X5=torch.arange(4,dtype=torch.float32)
Y5=torch.ones(4,dtype=torch.float32)
print('18.向量的点积\n',torch.dot(X5,Y5))

#矩阵和向量的乘积
print('19.矩阵和向量的乘积\n',torch.mv(A4,X5))
#矩阵和矩阵的乘积
B=torch.ones(4,3)
print('20.矩阵和矩阵的乘积\n',torch.mm(A4,B))

#向量范数
# L2范数：所有元素平方求和开根号
u=torch.tensor([3.0,-4.0])
print('21.L2范数\n',torch.norm(u))
# L1范数：每个元素的绝对值求和
print('22.L1范数\n',torch.abs(u).sum())
#矩阵范数
# F范数：每个矩阵元素的平方求和开根号
k=torch.ones(4,9)
print('23.F范数\n',torch.norm(k))

