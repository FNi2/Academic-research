import os
import pandas as pd
import torch
# 创建csv文件
os.makedirs(os.path.join('..','data'), exist_ok=True) # 在上级目录创建data文件夹
data_file=os.path.join('..','data','house_tiny.csv')  # 创建文件
with open(data_file,'w')as f: # 创建文件
    f.write('NumRooms,Alley,prince\n') #列名
    f.write('NA,Pave,127500\n') #每行表示一个数据样本
    f.write('2,NA,10600\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n') # 第4行的值
# 加载CSV文件
data=pd.read_csv(data_file)
print('1.原始数据：\n',data)

# 数据预处理，处理缺失的数据（插值）

inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
#数值预处理
inputs=inputs.fillna(inputs.mean()) # 用均值填充NaN
#非数值预处理
# 利用pandas中的get_dummies函数来处理离散值或者类别值。
# [对于 inputs 中的类别值或离散值，我们将 “NaN” 视为一个类别。] 由于 “Alley”列只接受两种类型的类别值 “Pave” 和 “NaN”
inputs=pd.get_dummies(inputs,dummy_na=True)

print('2.预处理后的数据：\n',inputs)


x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print('3.转换为张量:\n',x,y)