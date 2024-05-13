import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
df['Target'] = california.target
# New column 'Target' 생성 : california.target(우리가 예측해야 하는 값)을 출력값으로 넣어준다.
print(df.tail())
print()

# Convert to pytorch tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
# numpy array로 값 받기 위해서 : .values
# numpy array 값을 받아서 tensor로 전환하기 위해서 torch.from_numpy
# 값이 Double 형이라서 float 형으로 변환

# data 행/열 size 확인하기
print(len(data))    # 20640
print(data.shape[0])    # 20640
print(data.shape[1])    # 9
print()

x = data[:, : -1]
y = data[:, -1:]

print(x.size(), y.size())
# torch.Size([20640, 8]) torch.Size([20640, 1])

# Train / Valid / Test split ratio 지정
# Random split
ratios = [.6, .2, .2]   # list로 지정

# count 나누기
train_cnt = int(data.size(0) * ratios[0])
print(type(train_cnt))  # <class 'float'> -> 따라서 int 씌워주기
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt

# list 만들어주기
cnts = [train_cnt, valid_cnt, test_cnt]
print("Train %d / Valid %d / Test %d" % (train_cnt, valid_cnt, test_cnt))
print()

# Shuffling before split_for SGD(Stochastic Gradient Descent)
indices = torch.randperm(data.size(0))
print(indices)  # tensor([11441, 12712,  4400,  ..., 20399, 18459,  6008])
# |x| = (20640, 8) / # |y| = (20640, 1)
x = torch.index_select(x, dim=0, index=indices)
# dim=0 은 위 아래 방향. 즉, 위 아래 방향으로 indices 적용해서 섞어라는 말
y = torch.index_select(y, dim=0, index=indices)

# Split train, valid, test set with each count
x = list(x.split(cnts, dim=0))   # dim=0은 위 아래 방향으로 나누기
print(x)    # x의 tensor 값이 도출되고, 현재 튜플형태로 trian / valid / test로 나누어서 도출
y = y.split(cnts, dim=0)

# x,y 같이 묶기
for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())


# Preprocessing
scaler = StandardScaler()
# fit: 데이터 각 칼럼들의 분포를 보고 뮤와 시그마 -> 정규분포의 평균=0, 분산=1을 구하는 것 -> 이 또한 학습의 과정 중 하나
# You must fit with train data only. (Train Dataset만 학습할 것이므로)
# validation set과 test set에 대해서는 fit을 먹이면 안됨.
scaler.fit(x[0].numpy())    # 현재 x[0]은 tensor 값이므로 numpy array로 변환
# 표준 스케일링을 진행하기 위해서 학습 데이터에 대한 평균과 표준편차를 구한다.
# 학습 데이터만을 활용하여 정규화 진행

# train set
x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
# valid set
x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
# test set
x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()

df = pd.DataFrame(x[0].numpy(), columns=california.feature_names)
print(df.tail())


# Build Model & Optimizer
# Method1
# model = nn.Sequential(
#     nn.Linear(x[0].size(-1), 6),
#     nn.LeakyReLU(),
#     nn.Linear(6, 5),
#     nn.LeakyReLU(),
#     nn.Linear(5, 4),
#     nn.LeakyReLU(),
#     nn.Linear(4, 3),
#     nn.LeakyReLU(),
#     nn.Linear(3, y[0].size(-1)),
#     nn.LeakyReLU()

# )

# print(model)
# print()

# Method2
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()
        # 부모클래스의 생성자를 가져온다.
        self.linear1 = nn.Linear(input_dim, 6)
        self.linear2 = nn.Linear(6, 5)
        self.linear3 = nn.Linear(5, 4)
        self.linear4 = nn.Linear(4, 3)
        self.linear5 = nn.Linear(3, output_dim)
        self.act = nn.ReLU()
        # 활성함수는 학습을 하는 것이 아니므로 한 번만 적어주면 된다.

    def forward(self, x):
        # |x| = (batchsize, input_dim)
        # 선형함수(Linear) 통과 후 활성함수(act) 통과하는 프로세스
        h = self.act(self.linear1(x))   # |h| = (batchsize, 6)
        h = self.act(self.linear2(h))   # |h| = (6, 5)
        h = self.act(self.linear3(h))   # |h| = (5, 4)
        h = self.act(self.linear4(h))   # |h| = (4, 3)
        y = self.linear5(h)             # |y| = (batchsize, output_dim)
       # y = self.act(self.linear5(h))  # 마지막 함수는 활성화함수 통과 X

        return y

# model = MyModel(input_dim, output_dim)
model = MyModel(x[0].size(-1), y[0].size(-1))
print(model)
print()

# optimizer
optimizer = optim.Adam(model.parameters())

# Train
n_epochs = 4000     # 총 4000번 학습
batch_size = 256
print_interval = 100

from copy import deepcopy

lowest_loss = np.inf
best_model = None

early_stop = 100
lowest_epoch = np.inf

train_history, valid_history = [], []

for i in range(n_epochs):
    # Shuffle before mini-batch split.
    # x[0]: trining set
    indices = torch.randperm(x[0].size(0))
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)
    # |x_| = (total_size, input_dim)
    # |y_| = (total_size, input_dim)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    # |x_[i]| = (batch_size, input_dim)
    # |y_[i]| = (batch_size, output_dim)

    train_loss, valid_loss = 0, 0
    y_hat = []

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss)
    train_loss = train_loss / len(x_)

    # You need to declare to PYTORCH to stop build the computation graph.
    with torch.no_grad():
      # gradient를 쓰지마라. - validation set은 학습을 안 하는거니까.
      x_ = x[1].split(batch_size, dim=0)
      y_ = y[1].split(batch_size, dim=0)

      valid_loss = 0

      for x_i, y_i in zip(x_, y_):
          y_hat_i = model(x_i)
          loss = F.mse_loss(y_hat_i, y_i)

          valid_loss += loss
          ## back propagation 할 필요가 없음
          y_hat += [y_hat_i]
    valid_loss = valid_loss / len(x_)

    # Log each loss to plot after training is done.
    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i + 1) % print_interval == 0:
        print('Epoch %d train loss= %.4e valid_loss= %.4e lowest_loss= %.4e' % \
              (i + 1, train_loss, valid_loss, lowest_loss))
        
    
    # validation loss가 낮아지는지 확인해야 한다.
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i
    # 'state_dict()' returns model weights (weight parameter) as key-value.
    # Take a deep copy, if the valid loss is lowest ever.

    # 우리의 최종목표는 generalization error을 낮추는 것이기 때문에
    # validation의 lowest loss를 항상 기억하고 있어야한다.
    # validation의 lowest loss가 갱신되는 시점에 state_dict()을 통해
    # 스냅샷을 떠놓고 모든 학습이 종료되고 나서 best를 복원해준다.
    # 그래야 모든 학습이 종료되고 나서 overfitting이 되지 않은 최고의 모델로 진행가능

        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            # 가장 낮은 손실값을 갱신한 epoch에 early_stop 값을 더한 epoch가 현재 epoch보다 작은지 확인
            print('There is no improvement during last %d epochs.' % early_stop)
            break

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))

# Load best epoch's model.
model.load_state_dict(best_model)