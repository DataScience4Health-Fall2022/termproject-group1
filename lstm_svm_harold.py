#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

from rich import print
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='APNEA')
parser.add_argument('--method', type=str, default='lstm', help='Method')
parser.add_argument('--patient', type=str, default='21', help='Patient')
parser.add_argument('--factor', type=str, default='signal_ecg_ii', help='Factor')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

np.random.seed(args.random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(dataset, patient, factor, random_seed):
    with open(dataset, 'rb') as fr:
        df = pickle.load(fr)
    df = df[df['patient'] == patient]  # 筛选出某个病人的信息
    X_pos, Y_pos, X_neg, Y_neg = [], [], [], []
    for index, row in df.iterrows():
        x = row[factor]
        where_are_nan = np.isnan(x)
        where_are_inf = np.isinf(x)
        x[where_are_nan] = 0
        x[where_are_inf] = 0

        if 'APNEA-OBSTRUCTIVE' in row['event']:  # event 作为标签
            X_pos.append(x)
            Y_pos.append(1)
        elif 'NONE' in str(row['event']):
            X_neg.append(x)
            Y_neg.append(0)

    X_pos, Y_pos, X_neg, Y_neg = np.array(X_pos), np.array(Y_pos), np.array(X_neg), np.array(Y_neg)
    # 负样本下采样. 保证正负样本比为 1: 2
    sample_index = np.random.choice(X_neg.shape[0], size=int(X_pos.shape[0] * 2), replace=False, p=None)
    X_neg, Y_neg = X_neg[sample_index], Y_neg[sample_index]
    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((Y_pos, Y_neg), axis=0)

    if len(X.shape) == 2:
        X = StandardScaler().fit_transform(X)  # 标准化数据
        if args.method == 'lstm':  # 如果是 LSTM, 需要升一维作为特征
            X = np.expand_dims(X, axis=2)

    # 按 7: 3 分割训练测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_seed, stratify=Y)
    return X_train, X_test, Y_train, Y_test


class LSTM(nn.Module):
    def __init__(self, seq_len, n_features, embed_dim=64):
        super(LSTM, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embed_dim, self.hidden_dim = embed_dim, 2 * embed_dim
        
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embed_dim,
            num_layers=1,
            batch_first=True)
        self.output_layer = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        # x 经过两层 LSTM, 并将第二层 LSTM 的输出经过线性层得到模型预测
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        o = self.output_layer(hidden_n.squeeze(0))
        return o


def svm_exp():
    # SVM 实验
    X_train, X_test, Y_train, Y_test = load_data('dataset_OSAS.pickle', args.patient, args.factor, args.random_seed)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    print(np.sum(Y_train), np.sum(Y_test))
    model = svm.SVC(C=10, kernel='linear')  # SVM 模型初始化
    model.fit(X_train, Y_train)  # 训练
    Y_pred = model.predict(X_test)  # 测试
    # 计算测试集的预测结果与真实值计算准确率
    report = classification_report(Y_test, Y_pred, target_names=['None', 'APNEA'])
    print(report)


def lstm_exp():
    # LSTM 实验
    X_train, X_test, Y_train, Y_test = load_data('dataset_OSAS.pickle', args.patient, args.factor, args.random_seed)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    print(np.sum(Y_train), np.sum(Y_test))
    # 将数据转为 torch.Tensor 格式
    X_train, X_test = torch.FloatTensor(X_train).to(device), torch.FloatTensor(X_test).to(device)
    Y_train, Y_test = torch.FloatTensor(Y_train).to(device), torch.FloatTensor(Y_test).to(device)
    # 通过 Dataloader 加载数据, batch_size=16, 测试集随机打乱顺序
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    # 模型初始化
    model = LSTM(seq_len=X_train.shape[1], n_features=X_train.shape[2], embed_dim=128)
    model = model.to(device)

    train_model(model, train_loader, test_loader)


def train_model(model, train_loader, test_loader):
    # LSTM 模型训练
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)  # 优化器
    crite = nn.BCEWithLogitsLoss(reduction='mean').to(device)  # 交叉熵损失

    best_model = copy.deepcopy(model.state_dict())  # best_model 保存最优的参数
    best_loss = 10000  # 初始化损失
    for epoch in range(1, 501):  # 训练 500 个 epoch
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:  # 遍历训练集优化模型
            optim.zero_grad()
            Y_pred = model(X_batch)  # 模型计算
            loss = crite(Y_pred.squeeze(-1), Y_batch)  # 计算损失
            loss.backward()  # loss 回传
            optim.step()

            train_losses.append(loss.item())

        test_losses = []
        model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:  # 遍历测试集, 计算 test loss
                Y_pred = model(X_batch)  # 模型计算
                loss = crite(Y_pred.squeeze(-1), Y_batch)  # 计算损失
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)

        # 如果测试集损失比之前小, 说明这一轮得到的模型比之前好.
        # 记录新的最优损失和模型参数
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} test loss {test_loss}')

    model.load_state_dict(best_model)  # 加载最优模型
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            Y_pred = np.where(torch.sigmoid(Y_pred.squeeze(-1)).cpu().numpy() > 0.5, 1, 0)  # 如果 CPU 计算的, 去掉 .cpu()
            y_pred.extend(Y_pred.tolist())
            y_true.extend(Y_batch.cpu().numpy().tolist())  # 如果 CPU 计算的, 去掉 .cpu()
    # 计算测试集的预测结果与真实值计算准确率
    report = classification_report(y_true, y_pred, target_names=['None', 'APNEA'])
    print(report)


if __name__ == '__main__':
    if args.method == 'svm':
        svm_exp()
    else:
        lstm_exp()
