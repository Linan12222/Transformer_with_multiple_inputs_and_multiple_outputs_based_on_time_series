import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
import sys
import math

class StreamToLogger(object):
    """
    自定义流，用于同时将输出信息发送到标准输出和文件。
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这个函数在这里是为了兼容文件对象的接口
        self.terminal.flush()
        self.log.flush()

sys.stdout = StreamToLogger("console_output_TCN.txt")

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_out_len, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_out_len = seq_out_len
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        # 修改seq_decoder的定义
        self.seq_decoder = nn.Linear(look_back * input_dim, seq_out_len * output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        # 此处调整output的形状以匹配seq_decoder的输入
        output_flat = output.view(output.size(0), -1)
        output = self.seq_decoder(output_flat)

        # 调整output的形状以匹配目标数据的形状
        output = output.view(-1, self.seq_out_len, self.output_dim)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        max_len = x.size(0)
        position = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(max_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        x = x + pe[:x.size(0), :]
        return self.dropout(x)


# 数据处理函数，添加两个虚拟特征
def process_data(data, look_back, seq_out_len):
    X, y = [], []
    for i in range(len(data) - look_back - seq_out_len + 1):
        X_seq = data.iloc[i:(i + look_back), :].values
        y_seq = data.iloc[(i + look_back):(i + look_back + seq_out_len), :].values

        # 添加两个虚拟特征（例如全为0）到X和y
        X_seq_padded = np.pad(X_seq, ((0, 0), (0, 2)), mode='constant', constant_values=0)
        y_seq_padded = np.pad(y_seq, ((0, 0), (0, 2)), mode='constant', constant_values=0)

        X.append(X_seq_padded)
        y.append(y_seq_padded)

    return np.array(X), np.array(y)



plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 数据加载和处理
# 数据加载和处理
data = pd.read_excel('17+18-已处理.xlsx')
data = data.iloc[1:, 1:]
look_back = 4
# 设置循环
seq_out_lens = [1, 4, 8, 13, 16]
epoches = 50  # 定义epoches变量


for seq_out_len in seq_out_lens:
    print('seq_out_len:', seq_out_len)
    x, y = process_data(data, look_back, seq_out_len)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 数据转换
    X_test = torch.from_numpy(X_test).float()

    y_test = torch.from_numpy(y_test).float()

    # DataLoader
    # 转换为 PyTorch Tensor
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # 模型、损失函数和优化器
    input_dim = 16  # 更新后的输入维度
    output_dim = y_train.shape[-1]  # 输出特征数
    nhead = 4  # 头数
    nhid = 128
    nlayers = 2
    dropout = 0.2
    num_runs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 初始化指标存储列表
    mse_scores, rmse_scores, mae_scores, r2_scores, mape_scores = [], [], [], [], []

    # 运行模型 10 次
    for run in range(num_runs):
        # 模型定义
        model = TransformerModel(input_dim, output_dim, seq_out_len, nhead, nhid, nlayers, dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 模型训练
        model.to(device)
        for epoch in range(epoches):
            model.train()
            train_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # 模型测试
        model.eval()
        preds = model(X_test.to(device))
        predictions = preds.view(-1, seq_out_len, y_train.shape[-1]).cpu().detach().numpy()
        predictions_reshaped = predictions.reshape(-1, y_train.shape[-1])
        y_test_reshaped = y_test.numpy().reshape(-1, y_train.shape[-1])

        # 计算评估指标
        mse = mean_squared_error(y_test_reshaped, predictions_reshaped)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reshaped, predictions_reshaped)
        r2 = r2_score(y_test_reshaped, predictions_reshaped)
        # 定义一个小的阈值，用于确定目标值是否接近零
        epsilon = 1e-10

        # 初始化MAPE的累积总和和计数器
        mape_sum = 0
        count = 0

        # 遍历每个数据点
        for i in range(y_test_reshaped.shape[0]):
            for j in range(y_test_reshaped.shape[1]):
                # 只有当目标值不接近零时，才计算MAPE
                if abs(y_test_reshaped[i, j]) > epsilon:
                    mape_sum += np.abs((y_test_reshaped[i, j] - predictions_reshaped[i, j]) / y_test_reshaped[i, j])
                    count += 1

        # 计算MAPE
        mape = (mape_sum / count) * 100 if count > 0 else 0

        # 存储指标
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        mape_scores.append(mape)

        # 打印当前迭代的评估指标
        print(f'Run {run + 1}:')
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        print('R-squared:', r2)
        print('MAPE:', mape)
        print('-' * 50)

    # 计算平均评估指标
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)


    # 打印平均评估指标
    print('Average Metrics Over 10 Runs for seq_out_len =', seq_out_len)
    print('Average MSE:', avg_mse)
    print('Average RMSE:', avg_rmse)
    print('Average MAE:', avg_mae)
    print('Average R-squared:', avg_r2)
    print('Average MAPE:', avg_mape)
    print('-' * 100)