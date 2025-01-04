# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

model_file = 'my_CNN_with_emotion.pth'

def evaluate_model(model, X_test, y_test, epoch, device):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        inputs = X_test.to(device)  # 直接移动到GPU，不需要复制
        outputs = model(inputs)  # 进行预测
        _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别

        # 将 y_test 从 one-hot 编码转换为类别索引形式
        y_test_indices = np.argmax(y_test, axis=1)  # (batch_size, 2) -> (batch_size,)

        # 计算准确率
        f1 = f1_score(y_test_indices, predicted.cpu().numpy(), average='macro')  # 计算宏平均 F1 分数
        print(f'epoch:{epoch} F1 Score (macro): {f1:.4f}')  # 打印 F1 分数

        if (f1 > 0.882):
            torch.save(model.state_dict(), model_file)
            print("模型已保存为 {}".format(model_file))
            print("正确率到达0.882，保存文件！")
            sys.exit()  # 直接退出程序

num_epochs = 1000  # 或其他适当的数字

# 加载数据
data = np.load('../data/new_data_with_indicator.npz')
sentences_array = data['sentences']
tags_array = data['emotion_tag_with_indicator']

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(CNNModel, self).__init__()

        # 定义多个卷积层，增加卷积核尺寸
        self.conv2 = nn.Conv2d(1, 8, kernel_size=(2, input_dim))  # 2x300
        self.conv3 = nn.Conv2d(1, 8, kernel_size=(3, input_dim))  # 3x300
        self.conv5 = nn.Conv2d(1, 8, kernel_size=(5, input_dim))  # 5x300
        self.conv7 = nn.Conv2d(1, 8, kernel_size=(7, input_dim))  # 7x300
        self.conv10 = nn.Conv2d(1, 8, kernel_size=(10, input_dim))  # 10x300
        self.conv15 = nn.Conv2d(1, 8, kernel_size=(15, input_dim))  # 15x300

        # 计算拼接后的输入维度
        self.fc_input_dim = 8 * 6  # 8 个通道，6 个不同卷积核

        # 全连接层，增加隐藏层
        self.fc1 = nn.Linear(self.fc_input_dim, 32)  # 第一层隐藏层
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout层
        self.fc2 = nn.Linear(32, 16)  # 第二层隐藏层
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout层
        self.fc3 = nn.Linear(16, output_dim)  # 输出层

    def forward(self, x):
        # 进行卷积操作
        conv2_out = F.relu(self.conv2(x)).squeeze(3)  # 去掉最后一个维度
        conv3_out = F.relu(self.conv3(x)).squeeze(3)
        conv5_out = F.relu(self.conv5(x)).squeeze(3)
        conv7_out = F.relu(self.conv7(x)).squeeze(3)
        conv10_out = F.relu(self.conv10(x)).squeeze(3)
        conv15_out = F.relu(self.conv15(x)).squeeze(3)

        # 池化操作
        pooled2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)  # 全局最大池化并去掉维度
        pooled3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)
        pooled5 = F.max_pool1d(conv5_out, conv5_out.size(2)).squeeze(2)
        pooled7 = F.max_pool1d(conv7_out, conv7_out.size(2)).squeeze(2)
        pooled10 = F.max_pool1d(conv10_out, conv10_out.size(2)).squeeze(2)
        pooled15 = F.max_pool1d(conv15_out, conv15_out.size(2)).squeeze(2)

        # 拼接输出
        pooled = torch.cat((pooled2, pooled3, pooled5, pooled7, pooled10, pooled15), dim=1)

        x = pooled.view(pooled.size(0), -1)  # 展平
        # 全连接层，使用 ReLU 激活和 Dropout
        x = F.relu(self.fc1(x))  # 第一隐藏层
        x = self.dropout1(x)  # Dropout
        x = F.relu(self.fc2(x))  # 第二隐藏层
        x = self.dropout2(x)  # Dropout
        x = self.fc3(x)  # 输出层
        return x

# 创建模型实例
input_dim = 300  # 输入维度
output_dim = 11  # 输出维度 (11 分类)
model = CNNModel(input_dim=input_dim, output_dim=output_dim)

criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss 进行多分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    print("模型权重已加载。")
else:
    print("模型文件不存在，无法加载。")

# 设置设备为GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 将模型移动到设备上
criterion.to(device)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(sentences_array, tags_array,
                                                    test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 调整形状
X_train_tensor = X_train_tensor.unsqueeze(1)  # 增加通道维度，变为 (batch_size, 1, 50, 300)
X_test_tensor = X_test_tensor.unsqueeze(1)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除梯度

    # 转为Tensor并移动到GPU
    inputs = X_train_tensor.to(device)  # 输入 shape: (batch_size, 1, 50, 300)
    labels = torch.tensor(y_train, dtype=torch.float32).to(device)

    outputs = model(inputs)  # outputs shape: (batch_size, 11)

    loss = criterion(outputs, labels)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 评估模型
    evaluate_model(model, X_test_tensor, y_test, epoch, device)

    # 每隔 100 个 epoch 保存一次模型
    if epoch % 100 == 0:
        torch.save(model.state_dict(), model_file)
        print("模型已保存为 {} epoch:{}".format(model_file, epoch))

torch.save(model.state_dict(), model_file)
print("模型已保存为 {}".format(model_file))
