# -*- coding: utf-8 -*-
import os

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def index_sentence(fc_sentence, word_to_index, max_length):
    words = fc_sentence.split()  # 假设以空格分隔
    sentence_len = len(words)
    fc_sentence_matrix = []

    for fc_word in words:
        if fc_word in word_to_index:
            fc_sentence_matrix.append(word_to_index[fc_word])
        else:
            fc_sentence_matrix.append(np.zeros(300))  # 用300维的零向量替代
    # 填充到最大长度
    if sentence_len < max_length:
        # 填充300维的零向量
        fc_sentence_matrix += [np.zeros(300)] * (max_length - sentence_len)
    else:
        fc_sentence_matrix = fc_sentence_matrix[:max_length]  # 截断

    return fc_sentence_matrix  # 返回填充后的句子的索引列表


# 创建模型实例
def final_CNN_with_emotion(csv_path, tag, save_path):
    emotion_dict = {
        1: "愉快/幸福",
        2: "悲伤/痛苦",
        3: "愤怒/生气",
        4: "惊讶/震惊",
        5: "恐惧/害怕",
        6: "羞耻",
        7: "平静/冷静",
        8: "希望/期待",
        9: "厌恶/讨厌",
        10: "孤独/寂寞",
        11: "好奇/兴趣"
    }
    model_file = 'my_CNN_with_emotion.pth'
    input_dim = 300  # 输入维度
    output_dim = 11  # 输出维度（二分类）
    model = CNNModel(input_dim=input_dim, output_dim=output_dim)

    # 使用交叉熵损失
    criterion = nn.BCEWithLogitsLoss()  # 对于二分类，通常使用BCEWithLogitsLoss

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(model_file):
        # model.load_state_dict(torch.load(model_file))
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        print("模型权重已加载。")
    else:
        print("模型文件不存在，无法加载。")

    # 设置设备为GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到设备上

    criterion.to(device)

    vector_table = {}
    with open('../data/sgns.zhihu.bigram', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)  # 转换为numpy数组
            vector_table[word] = vector
    print("词典已加载。")

    if os.path.exists(model_file):
        # model.load_state_dict(torch.load(model_file))
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        print("模型权重已加载。")
    else:
        print("模型文件不存在，无法加载。")

    df = pd.read_csv(csv_path)
    sentences = df[tag].tolist()
    predict_list = []

    for sentence in sentences:
        sentence = " ".join(jieba.cut(sentence, cut_all=False, HMM=True))
        # print("sentence:", sentence)
        sentence_vector = index_sentence(sentence, vector_table, max_length=100)
        sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32).to(device)
        sentence_tensor = sentence_tensor.unsqueeze(0)
        sentence_tensor = sentence_tensor.unsqueeze(0)
        # print(sentence_tensor.shape)
        result = model(sentence_tensor)
        predicted_class = torch.argmax(result).item() + 1
        predict_list.append(emotion_dict[predicted_class])

    df["emotions"] = predict_list
    # df = pd.DataFrame({"sentences":sentences,"tags":predict_list})
    csv_file_path = save_path  # 指定 CSV 文件的路径
    df.to_csv(csv_file_path, index=False)  # index=False 表示不保存行索引

if __name__ == "__main__":
    csv_path = "../test/筛选数据_“三权分立”变“三权合一”，特朗普2.0时代前瞻.csv"
    save_path = "../test/具体情感的.csv"
    final_CNN_with_emotion(csv_path=csv_path, tag="评论内容", save_path= save_path)
