# Linear Regression

# Load Dataset from sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn

# 보스턴 주택 가격 데이터셋 로드
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston', version=1)

print(boston.DESCR)
print()

# DataFrame 으로 나타내기
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['Target'] = boston.target    # boston Dataframe에 ['Target'] column 추가
df.tail()
# boston.data: 수치형으로 제시
# boston.target: 주택 가격
'''
PyTorch에서 .target은 주로 학습 데이터셋에서 타깃 변수(예측하고자 하는 값)에 접근할 때 사용됩니다. 
많은 기계학습 데이터셋은 입력 데이터(features)와 타깃 데이터(labels 또는 targets)로 구성되어 있습니다.

예를 들어 보스턴 주택 가격 데이터셋의 경우 boston.data는 주택의 특성(방 개수, 평균 거리, 범죄율 등)을 포함하고, 
boston.target은 해당 데이터에 대응하는 주택 가격을 포함합니다.

따라서 모델을 학습시킬 때 입력 데이터는 boston.data를 사용하고, 
타깃 변수는 boston.target을 사용하여 supervised learning을 수행할 수 있습니다.
'''

# sns.pairplot(df)
# plt.show()

# 값 바꿔가면서 찾아보기
cols = ["RM", "AGE", "PTRATIO", "LSTAT"]   
df[cols].describe()

# sns.pairplot(df[cols])
# plt.show()

# Train Linear Model with Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# pandas 값을 numpy[array]로 가져오는 방법 df[cols].values
# torch.from_numpy 를 통해서 numpy 값을 Tensor로 가져오기.
# torch.from_numpy(df[cols].values)가 DoubleTensor 값이라서 float 찍어주기 : (    ).float()

# numpy array로 배열
print(df[cols].values)
print()

# numpy 값을 tensor로 변환
print(torch.from_numpy(df[cols].values))
print()

a = torch.from_numpy(df[cols].values)
print(a.type())     # torch.DoubleTensor
print()

# Double 값을 Float 값으로 바꾸기
data = torch.from_numpy(df[cols].values).float()
print(data)
print()
print(data.shape)   # data.size()
# torch.Size([506, 4])
print()

## Split x and y
y = data[:, :1] # 첫 번째 값 하나(슬라이싱은 0부터)
x = data[:, 1:]

print(y.shape, x.size())
print()

# Define configurations.
n_epochs = 2000
learning_rate = 1e-3
# "e"는 10의 거듭제곱을 나타내며 1e-3 : 0.001
print_interval = 100

# Define model(Linear Regression).
# x.size(-1) : torch.Size([506, 4]) 중에서 '4'를 가리킴.
model = nn.Linear(x.size(-1), y.size(-1))
# 입력 및 출력 벡터의 차원을 넣어줌
print(model)
print()

# Instead of implementing gradient equation,
# we can use <optim class> to update model parameters, automatically.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# optim.SGD라는 class를 활용한다.
# optimizer은 lr=learning_rate의 비율로 gradient descent 해준다.

y_hat = model(x)
print(y_hat)        # 값 정상적으로 나온다.
print()

loss = F.mse_loss(y_hat, y)
print(loss)     # tensor(1367.2802, grad_fn=<MseLossBackward0>) 
                # 정상적으로 나온다.
print()

## Whole training samples are used in 1epoch.
# Thus, "N epochs" means that model saw a sample N-times.

# 여기서 무언가 문제가 있는데..
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()   # zero_grad(): 까먹으면 안된다.
                            # gradient를 0으로 초기화해주지 않으면 이전의 값들이 다 더해지게 된다.
                            # optimizer에게 명령을 함: 모델의 파라미터들의 gradient를 다 비워놔!

    loss.backward()         # loss를 theta로 미분함.
    optimizer.step()        # optimizer에게 명령을 함: 내가 gradient도 구해놨거든? learning rate도 알려줬으니
                            # 이제 descent 해! step(한 걸음 걸어가게 됨.)
    
    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss= %.4e' % (i+1, loss))

# optimizer.step()는 PyTorch에서 optimizer를 업데이트하는 중요한 단계입니다.
'''
일반적인 PyTorch 학습 과정은 다음과 같습니다:

1. optimizer 객체 생성 
(예: optimizer = torch.optim.SGD(model.parameters(), lr=0.01))
2. 순전파 단계 (feed - forward) : y_hat = model(x)
# outputs: y_hat / inputs: x
3. 손실함수 계산 : loss = criterion(y_hat, y)
# criterios: MSE, RMSE...
# outputs: y_hat / labels: y
4. 역전파 단계 (backward pass) : loss.backward()
# 역전파 단계에서 그래디언트를 사용하여 모델의 가중치를 실제로 업데이트 함(수식 생각하기)
5. 가중치 업데이트 : optimizer.step()
# 역전파 단계에서 계산된 그래디언트를 사용하여 모델의 가중치를 실제로 업데이트.
'''

print()
# Let's see the result!
print(y, y.size())
print()
print(y_hat, y_hat.shape)
print()

# concat: 리스트 배열로 들어간다.
torch.cat([y, y_hat], dim=1)    # 열로 합쳐짐.
print()
torch.cat([y, y_hat], dim=1).size()

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])
# y: 실제 정답 값.
# y_hat: 예측된 값들의 분포

sns.pairplot(df, height=5)
plt.show()

