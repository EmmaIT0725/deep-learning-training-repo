# 데이터 준비

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Pandas DataFrame
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target
df

# 데이터 분할

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# numpy data를 pytorch 데이터로 변환하는 작업
print(df.values)   # numpy value (array)
data = torch.from_numpy(df.values).float()
# print(data)    # torch value(tensor)

x = data[:, :-1]
y = data[:, -1:]    # 정답 데이터

print(x.shape, y.shape)

# Train / Valid / Test ratio

ratios = [.6, .2, .2]

# ratio에 따라 cnt 만들어두기 -> int
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt

cnts = [train_cnt , valid_cnt, test_cnt]    # list에 담음

print("Train %d / Valid %d / Test %d" % (cnts[0] , cnts[1], cnts[2]))

# Remember!
# Split 한대로 Random Permutation

# randperm 할 때, indice 부터 만들어주기
indices = torch.randperm(data.size(0))

print(indices)

print(type(x))    # List

x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = x.split(cnts, dim=0)
y = y.split(cnts, dim=0)

for x_i, y_i in zip(x, y):
  print(x_i.size(), y_i.size())

# 학습 데이터 기준 표준 스케일링 학습
scaler = StandardScaler()
# x[0]: Train dataset(Tensor)

scaler.fit(x[0].numpy())

x = [torch.from_numpy(scaler.transform(x[0].numpy())).float(),
     torch.from_numpy(scaler.transform(x[1].numpy())).float(),
     torch.from_numpy(scaler.transform(x[2].numpy())).float()]

"""# 학습코드 구현
* 선형 계층과 리키 렐루를 차례로 집어넣어준다.
* 모델의 마지막에는 시그모이드를 집어넣어주어서 이진 분류를 위한 준비를 마친다.
"""

data.size()
x[0].size()

model = nn.Sequential(
    nn.Linear(x[0].size(-1), 25),
    nn.LeakyReLU(),
    nn.Linear(25, 20),
    nn.LeakyReLU(),
    nn.Linear(20, 15),
    nn.LeakyReLU(),
    nn.Linear(15, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 5),
    nn.LeakyReLU(),
    nn.Linear(5, y[0].size(-1)),
    nn.Sigmoid(),   # 마지막에 시그모이드 넣어주어 이진 분류를 위한 준비 마치기
)

optimizer = optim.Adam(model.parameters())

"""학습에 필요한 하이퍼파라미터 조정해주기"""

n_epochs = 10000
batch_size = 32   # If the dataset is small, using a large batch size might result in fewer parameter updates, which could be problematic.
print_interval = 10
early_stop = 100

# loss value 초기화
lowest_loss = np.inf
best_model = None
lowest_epoch = np.inf

"""모델 학습 iteration 진행하는 반복문 코드

손실함수 : BCE (F.binary_cross_entropy)


코드의 재사용(머릿속에서 흐름이 자연스럽게 흘러가도록 반복하기)
"""

# 초기값 초기화하기
train_history, valid_history = [], []

# 훈련하기
for i in range(n_epochs):
  indices = torch.randperm(x[0].size(0))    # 숫자 0~340, 총 341를 섞어주기

  x_ = torch.index_select(x[0], dim=0, index=indices)
  y_ = torch.index_select(y[0], dim=0, index=indices)

  x_ = x_.split(batch_size, dim=0)    # Stocastic Gradient Descent(SGD) : 확률적 경사하강
  y_ = y_.split(batch_size, dim=0)

  train_loss, valid_loss = 0, 0
  y_hat = []

  for x_i, y_i in zip(x_, y_):    # 흐름을 머릿속으로 따라가면서 그려보기
    y_hat_i = model(x_i)
    loss = F.binary_cross_entropy(y_hat_i, y_i)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    train_loss += float(loss)

  train_loss = train_loss / len(x_)

  # valid data와 test data는 no_grad

  with torch.no_grad():
    # valid dataset은 검증을 하면 되므로, randperm 해서 굳이 섞을 필요없다.
    x_ = x[1].split(batch_size, dim=0)
    y_ = y[1].split(batch_size, dim=0)

    valid_loss = 0    # 초기화

    for x_i, y_i in zip(x_, y_):
      y_hat_i = model(x_i)
      loss = F.binary_cross_entropy(y_hat_i, y_i)

# torch.with no_grad()라서 Gradient is already detached
      # optimizer.zero_grad()
      # loss.backward

      # optimizer.step()

      valid_loss += float(loss)
      y_hat += [y_hat_i]

    valid_loss = valid_loss / len(x_)


    # 처음에 train_history, valid_history = [], [] 리스트로 두었다.
    # train_loss, valid_loss 넣을 container라고 생각하면 됨.
    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i + 1) % print_interval == 0:
      print("Epoch %d: train_loss = %.4e    valid_loss = %.4e    lowest_loss = %.4e" %
            (i+1,
             train_loss,
             valid_loss,
             lowest_loss,))

    if valid_loss <= lowest_loss:
      # True이면 갱신되어야 한다.
      lowest_loss = valid_loss
      lowest_epoch = i + 1

      best_model = deepcopy(model.state_dict())
      # 찰칵 찍혀야 한다.

    else:   # 조기 종료하기
      if early_stop > 0 and lowest_epoch + early_stop < i + 1:
        print("There is no improvement during last %d epochs." % early_stop)

        break


print("The best validation loss from epoch %d : %.4e" % (lowest_epoch, lowest_loss))
# best model 찍어내기
model.load_state_dict(best_model)

"""손실곡선 확인하기"""

plot_from = 2

plt.figure(figsize = (20, 10))
plt.grid(True)
plt.title("Train/Valid Loss History", fontsize=13)
# plt.plot(x, y)
plt.plot(range(plot_from, len(train_history)), train_history[plot_from:],
         range(plot_from, len(valid_history)), valid_history[plot_from:],)
plt.yscale('log')
plt.show()

# test_dataset으로 결과 확인하기

test_loss = 0
y_hat = []

# with no_grad() 해주어야한다. 까먹지 말기
with torch.no_grad():
  x_ = x[2].split(batch_size, dim=0)
  y_ = y[2].split(batch_size, dim=0)

  for x_i, y_i in zip(x_, y_):    # 흐름 머릿속에 그리면서 따라가기
    y_hat_i = model(x_i)
    loss = F.binary_cross_entropy(y_hat_i, y_i)


# torch.with no_grad()라서 Gradient is already detached
    # optimizer.zero_grad()
    # loss.backward

    # optimizer.step()

    test_loss += float(loss)
    y_hat += [y_hat_i]    # List

  test_loss = test_loss / len(x_)
# print(len(y_hat))   # 4 : batch_size로 나눈후 tensor[[...], [...], [...], [...]] 4개로 나뉘어짐


  sorted_history = sorted(zip(train_history, valid_history), key=lambda x: x[1])

# print(sorted_history[0])    # 가장 손실값이 낮은 것
  print("Train loss: %.4e \nValid loss: %.4e \nTest loss: %.4e" %
           (sorted_history[0][0],
            sorted_history[0][1],
            test_loss))

  # 위 아래로 붙이기
  y_hat = torch.cat(y_hat, dim=0)

  # 일반 회귀가 아닌 분류 문제 : Accuracy도 구할 수 있다.
  correct_cnt = ((y_hat > .5) == y[2]).sum()
  # y_hat >= 0.5: probabiliiy=1
  # 실제 정답은 y
  # 비교연산자가 True이면: 1
  total_cnt = float(y[2].size(0))
  print()
  print("Test Accuracy: %.4f" % (correct_cnt / total_cnt))

  # 예측값의 분포도 확인하기
  df = pd.DataFrame(torch.cat([y[2], y_hat], dim=1).detach().numpy(),
                    columns=["y", "y_hat"])
# print(df)
  '''
  detach():
  This method is used to detach a tensor from the computation graph in PyTorch.
  When you detach a tensor, it no longer tracks gradients.
  This is useful when you don't need gradients and just want the data for further operations without backpropagation.
  '''
  sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
  plt.show()

  # AUROC
  from sklearn.metrics import roc_auc_score

  roc_auc_score(df.values[:, 0], df.values[:, 1])

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import sklearn
# import torch

# print("numpy version:", np.__version__)
# print("pandas version:", pd.__version__)
# print("seaborn version:", sns.__version__)
# print("matplotlib version:", plt.matplotlib.__version__)
# print("scikit-learn version:", sklearn.__version__)
# print("torch version:", torch.__version__)
