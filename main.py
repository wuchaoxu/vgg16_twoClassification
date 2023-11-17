import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader#工具取黑盒子，用函数来提取数据集中的数据（小批次）
from data import *
'''数据集'''
annotation_path='cls_train.txt'#读取数据集生成的文件
with open(annotation_path,'r') as f:
    lines=f.readlines()
np.random.seed(10101)#函数用于生成指定随机数
np.random.shuffle(lines)#数据打乱
np.random.seed(None)
num_val=int(len(lines)*0.1)#十分之一数据用来测试
num_train=len(lines)-num_val
#输入图像大小
input_shape=[224,224]   #导入图像大小
train_data=DataGenerator(lines[:num_train],input_shape,True)  #训练集
val_data=DataGenerator(lines[num_train:],input_shape,False)  #测试集
train_data.rand()
"""
DataGenerator 类是一个自定义的数据生成器，用于从文件中读取图像和标签，并将它们转换为可供模型使用的格式。
第一个参数 lines[:num_train] 是训练集的文件名列表，第二个参数 input_shape 是图像大小，第三个参数 True 表示这是训练数据集。
同样地，第二个 DataGenerator 实例是测试数据集，第三个参数 False 表示这是测试数据集。
"""
val_len=len(val_data)
#print(val_len)#返回测试集长度 #133
# 取黑盒子工具
"""加载数据"""
gen_train=DataLoader(train_data,batch_size=4)#训练集batch_size读取小样本，规定每次取多少样本
gen_test=DataLoader(val_data,batch_size=4)#测试集读取小样本
'''构建网络'''
device=torch.device('cuda'if torch.cuda.is_available() else "cpu")#电脑主机的选择
net=vgg16(True, progress=True,num_classes=2)#定于分类的类别
net.to(device)
'''True, progress=True 表示下载预训练权重，num_classes=2 表示分类的类别数为 2。最后一行代码 net.to(device) 将模型移动到指定的设备上。'''
'''选择优化器和学习率的调整方法'''
lr=0.0001#定义学习率
optim=torch.optim.Adam(net.parameters(),lr=lr)#导入网络和学习率
sculer=torch.optim.lr_scheduler.StepLR(optim,step_size=1)#调整优化器的学习率，步长设置为1
#torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
# '''训练'''
epochs=20#读取数据次数，每次读取顺序方式不同
for epoch in range(epochs):
    print("==========第{}轮训练开始==========".format(epoch+1))

    total_train=0 #定义总损失

    net.train()
    for data in gen_train:
        img,label=data
        with torch.no_grad():
            img =img.to(device)
            label=label.to(device)
        optim.zero_grad()
        output=net.forward(img)  #前向传播
        train_loss=nn.CrossEntropyLoss()(output,label).to(device)  #交叉熵损失函数
        train_loss.backward()#反向传播
        optim.step()#优化器更新
        total_train+=train_loss #损失相加


    sculer.step()  #调整学习率
    total_test=0#总损失
    total_accuracy=0#总精度


    net.eval()
    for data in gen_test:
        img,label =data #图片转数据
        with torch.no_grad():
            img=img.to(device)
            label=label.to(device)
            optim.zero_grad()#梯度清零
            out=net(img)#投入网络
            test_loss=nn.CrossEntropyLoss()(out,label).to(device)
            total_test+=test_loss#测试损失，无反向传播
            accuracy=(out.argmax(1)==label).sum()#.clone().detach().cpu().numpy()#正确预测的总和比测试集的长度，即预测正确的精度
            total_accuracy+=accuracy
    print("第{}轮训练集上的损失：{}".format(epoch+1, total_train))
    print("第{}轮测试集上的损失：{}".format(epoch+1, total_test))
    print("测试集上的精度：{:.1%}".format(total_accuracy/val_len))#百分数精度，正确预测的总和比测试集的长度
    #{:.1%} 是百分数精度，可以将小数转换为百分数。其中，.1 表示保留一位小数。

    torch.save(net.state_dict(),"DogandCat{}.pth".format(epoch+1))
    print("模型已保存")


