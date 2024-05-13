# 데이터 준비
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# breast_cancer dataset을 pandas dataframe에 넣고 'class' column에 정답 data 넣기
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target
print(df)   # [569 rows x 31 columns]

# data split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(df.values)    # value 값은 numpy 값이므로 pytorch에 담아주기
data = torch.from_numpy(df.values).float()     # torch.from_numpy(df.values) 값이 double이라서 float으로 바꿔주기 
print(data)     # Matrix

x = data[:, :-1]    # 마지막 열 직전까지 / input data: 569차원의 벡터
y = data[:, -1:]    # 정답 열인 마지막 열만 / out data: 하나의 이진값으로 표현

print(x.size(), y.size())   # print(x.shape, y.shape)

# data를 train / valid / test set으로 split (6:2:2:)
ratios = [.6 , .2, .2]  # ratios[0], ratios[1], ratios[2] by list
print(data.size(0))
print(type(data.size(0)))   # <class 'int>

train_cnt = data.size(0) * ratios[0]
print(type(train_cnt))      # type: <class 'float'>
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]
print(cnts)
# data.size(0)의 type은 <class 'int'>

print('Train %d / Valid %d / Test %d samples.' % (train_cnt, valid_cnt, test_cnt))
# 6:2:2의 비율로 나눌 경우, 각각 341, 113, 115개의 샘플로 학습 / 검증 / 테스트 데이터셋이 구성됨.

# 각각 Random permutation(randperm) 진행
indice = torch.randperm(data.size(0))   # 0 ~ 568 random permutation
# print(indice)

# x, y를 index=indice대로 random permutation 해준다. [일단 shuffle해준 후 나눠진 비율(cnt)대로 나눌 예정]
x = torch.index_select(x, dim=0, index=indice)
y = torch.index_select(y, dim=0, index = indice)

# 이제 randomperm 한 것을 cnt대로 split 해주기
x = x.split(cnts, dim=0)
y = y.split(cnts, dim=0)

# train_set
# valid_set
# test_set

# x, y 합치기: 짝 지어주기
for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())
    '''
    torch.Size([341, 30]) torch.Size([341, 1])
    torch.Size([113, 30]) torch.Size([113, 1])
    torch.Size([115, 30]) torch.Size([115, 1])
    '''
# 여기까지 데이터 셋 나누고, randperm 진행

# 표준 스케일링 (정규화) 해주기 : column별로 값이 너무 상이하므로 맞춰주기
scaler = StandardScaler()
# print(x[0])     # x[0] : tensor 값
# print(x[0].numpy())
scaler.fit(x[0].numpy())
# x[0] : train dataset
# 학습데이터를 기준으로 표준 스케일링을 학습한 후, 해당 스케일러를 학습/검증/테스트 데이터셋에 똑같이 적용
'''
주어진 데이터 x[0]를 가지고 있는 Tensor를 스케일링하기 위해 Scikit-learn의 StandardScaler 또는 MinMaxScaler와 같은 스케일러를 사용하는 것
이 때 Tensor는 |x[0]| = (341, 30)
여기서 scaler.fit() 메서드는 스케일러를 데이터에 맞추는 역할을 한다. 
이 메서드를 호출하면 스케일러가 주어진 데이터에 대한 평균 및 표준편차(StandardScaler()) 또는 최솟값과 최댓값(MinMaxScaler)을 계산하여 내부적으로 저장
'''
x = [torch.from_numpy(scaler.transform(x[0].numpy())).float(), 
     torch.from_numpy(scaler.transform(x[1].numpy())).float(), 
     torch.from_numpy(scaler.transform(x[2].numpy())).float()]

# scaler.transform() 함수를 사용하여 각 데이터를 스케일링 
'''
scaler.fit()을 통해 저장된 평균과 표준편차를 0과 1로 각각 맞추기 위한 과정
scaler.transform() 함수는 주어진 데이터를 스케일링하여 평균이 0이 되고 표준편차가 1이 되도록 변환. 
이는 주어진 데이터의 분포를 <표준 정규 분포>로 맞추는 작업을 수행하는데, 
이는 일반적으로 머신러닝 모델의 학습 성능을 향상시킨다.

예를 들어, StandardScaler 클래스를 사용하는 경우에는 평균이 0이 되도록 데이터를 이동하고, 
표준편차가 1이 되도록 데이터를 조정. 이는 Z 점수 정규화라고도 불리며, 
이 작업을 수행함으로써 데이터의 단위가 서로 다른 경우에도 동일한 스케일로 학습을 진행할 수 있게 된다.

따라서, 스케일러를 적용하면 모델의 학습이 더욱 안정적으로 이루어지고, 최적화 과정에서 더 빠르게 수렴할 수 있게 된다.
'''

# 데이터 조정이 끝이났고, 이제는 학습모델 구현
# 선형 계층(Linear)과 리키 렐루(LeakyRelu)를 차례로 집어넣어준다.
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 25),       # x[0].size(-1) : (341, 30) 중에서 n차원인 30을 가리킴
    nn.LeakyReLU(),
    nn.Linear(25, 20),
    nn.LeakyReLU(),
    nn.Linear(20, 15),
    nn.LeakyReLU(),
    nn.Linear(15, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 5),
    nn.LeakyReLU(),
    nn.Linear(5, y[0].size(-1)),      # y[0].size(-1) : (341, 1) 중에서 m차원인 1
    nn.Sigmoid(),        # 모델 구조의 마지막에 Sigmoid 넣어주어서 이진 분류를 위한 준비 마치기
)

# Adam optimizer : 파라미터 가중치를 자동으로 설정
optimizer = optim.Adam(model.parameters())

# 학습에 필요한 하이퍼 파라미터 설정
# early stop(조기 종료)을 설정해두어서 n_epoch()을 크게 잡음

n_epoch = 10000
batch_size = 32     # 전체 데이터 셋이 569개이고, 학습 데이터셋은 341개 밖에 안되므로, 
                    # Batch_Size가 너무 크면 1 epoch의 iteration 횟수가 적음, 파라미터 업데이트 횟수가 적음
print_interval = 10
early_stop = 100
lowest_loss = np.inf
lowest_epoch = np.inf
best_model = None
'''
lowest_loss = np.inf는 최저 손실값을 초기화하는 코드.
여기서 np.inf는 NumPy에서 무한대를 나타내는 특수한 값. 
이는 손실값을 비교하는 과정에서 초기화된 최저 손실값보다 낮은 손실값을 발견했을 때 그 값을 최저 손실값으로 설정하기 위해 사용.

따라서 이 코드는 초기에는 최저 손실값을 무한대로 설정하고, 이후에 모델을 학습하면서 발견된 손실값이 
이 값보다 낮을 경우 최저 손실값을 업데이트하는데 사용된다. 
이를 통해 학습 중에 발견된 가장 낮은 손실값을 기록할 수 있다.
'''

# 모델 학습 iteration을 진행하는 반복문 코드
train_history, valid_history = [], []       # 빈 리스트 생성

# Stocastic Gradient Descent 를 하기 위한 과정
for i in range(n_epoch):
    indices = torch.randperm(x[0].size(0))       # 위 아래로 random permutation / indice는 dim을 변수로 안 받음
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)

    train_loss, valid_loss = 0, 0
    y_hat = []

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)     # BCE Loss 사용

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += float(loss)
        '''
        train_loss 변수는 훈련 중에 관찰된 손실값들의 누적값을 저장하는 변수이다. 
        각 반복(또는 배치)에서 관찰된 현재 손실값은 loss 변수에 저장되며, 
        이를 float을 통해 부동 소수점 형식으로 변환한 후 train_loss에 더하여 새로운 손실값을 누적한다. 
        이렇게 함으로써 train_loss에는 이전 손실값들의 합과 현재 손실값이 더해져서 누적되게 된다.
        '''
    train_loss = train_loss / len(x_)    

    # 학습(train)과 달리 검증(validation)은 역전파를 이용하여 학습을 수행하지 않는다.
    # 따라서 Gradient를 계산할 필요가 없기 때문에 torch.no_grad 함수를 호출하여 with 내부에서 검증 작업을 진행
    # 그래디언드를 계산하기 위한 배후 작업들이 없어지기 때문에 계산 오버헤드가 줄어들어 속도가 빨라지고 메모리 사용량이 줄어들게 된다.
    with torch.no_grad():
        # You don't need to shuffle the validation set.
        # Only Split is needed.
        '''
        # You don't need to shuffle the validation set.
        # Only Split is needed. 
        위의 의미는, 모델의 성능을 측정할 때 validation set에 대해 셔플링이 필요하지 않다는 것을 의미한다.

        셔플링은 데이터를 무작위로 섞는 것을 의미. 
        일반적으로 train(훈련) 데이터에 대해 셔플링을 수행하는 이유는 데이터가 순서대로 나열되어 있을 때 
        모델이 특정 패턴을 학습할 수 있기 때문. 
        하지만 validation(검증) 세트 또는 test(테스트) 세트에서는 모델이 제대로 학습하였는지
        일반화된 성능을 검증하고 평가하면 되는 것이라서 shuffling이 필요없다.
        '''
        x_ = x[1].split(batch_size, dim=0)
        y_ = y[1].split(batch_size, dim=0)

        valid_loss = 0

        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = F.binary_cross_entropy(y_hat_i, y_i)

            valid_loss += float(loss)

            y_hat += [y_hat_i]
        
        valid_loss = valid_loss / len(x_)

        train_history += [train_loss]
        valid_history += [valid_loss]

        if ( i + 1 ) % print_interval == 0:
            print('Epoch %d: train loss=%.4e valid_loss=%.4e lowest_loss=%.4e' % 
                (  i+1,
                  train_loss,
                  valid_loss,
                  lowest_loss,
                )
                )
            
        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = i

            best_model = deepcopy(model.state_dict())   # 모델 카피 찍기
        else:
            if early_stop > 0 and lowest_epoch + early_stop < i + 1:
                print("There is no improvement during last %d epoch" % early_stop)

                break
            '''
            1. early_stop > 0: early_stop이 0보다 크다는 것은, 조기 중지를 위해 일정한 기간을 두고 있다는 의미. 
            예를 들어, early_stop이 5이면, 모델이 최상의 성능을 낸 에포크 이후 5번의 에포크 동안 개선이 없다면 학습을 중지할 수 있다는 뜻.
            2. lowest_epoch + early_stop < i + 1: 여기서 lowest_epoch은 모델이 가장 좋은 성능을 보였던 에포크이고, i는 현재 에포크. 따라서 이 조건은 "가장 좋은 에포크 이후로 몇 번의 에포크가 지났는가"를 계산.
            lowest_epoch + early_stop은 모델이 최상의 성능을 낸 후 학습을 중지해야 하는 시점을 의미.

            이 두 조건을 합치면 다음과 같다:

            가장 좋은 성능을 낸 에포크 이후에 early_stop 횟수만큼의 에포크가 지났다면, 학습을 중지해야 한다는 의미.
            따라서, 이 코드 구문은 "모델이 더 이상 개선되지 않으면 조기에 학습을 멈추는 조건"을 확인하는 것
            '''

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))
model.load_state_dict(best_model)

# 손실 곡선 확인
plot_from = 2       # 그래프를 그릴 때 어디서부터 시작할지 정함

plt.figure(figsize=(20, 10))
plt.grid=True
plt.title("Train / Valid Loss History")
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:],
    # plot_from에서 train_history 길이까지의 범위를 생성하고, 해당 범위의 train_history 데이터를 그래프에 표시한다. 이로써 훈련 손실의 변화를 선 그래프로 나타낸다.
    # plot_from에서 valid_history 길이까지의 범위를 생성하고, 해당 범위의 valid_history 데이터를 그래프에 표시한다. 이로써 검증 손실의 변화를 선 그래프로 나타낸다.
    # plt.plot(x범위 , y범위) : y범위인 train_history[2:]는 train_history 리스트의 두 번째 인덱스부터 끝까지의 모든 값을 의미. 이 값들이 y축에 표시된다.
)
plt.yscale('log')
'''
y축을 로그 스케일(logarithmic scale)로 변환. 로그 스케일은 값이 매우 크게 변할 때나 작은 값을 더 명확하게 보기 위해 사용된다. 
로그 스케일을 사용하면 손실이 크게 변할 때 그래프의 가독성을 높일 수 있다.
'''
plt.show()

# Test set을 통한 결과확인
# 테스트 데이터 셋에 대해서 평균 손실 값을 구해본다.
'''
이 코드는 일반적으로 모델의 test(테스트 과정)을 시작할 때 사용된다. 
각 변수의 초기화와 이를 왜 초기화하는지에 대한 이유:

test_loss = 0:
test_loss는 테스트 데이터셋에 대해 모델의 손실을 추적하기 위한 변수. 
테스트 데이터에 대해 모델의 예측을 수행하면서, 예측 결과와 실제 레이블 간의 손실을 계산하여 
이 변수를 업데이트 한다.
변수를 0으로 초기화하는 것은 테스트 과정을 시작할 때 이전의 손실 값이 영향을 주지 않도록 하기 위해서. 
학습과정을 반복적으로 실행하거나 다른 테스트 케이스를 진행할 때 각 테스트는 서로 독립적이어야 하므로, 
항상 초기 상태로 설정하여 결과의 일관성을 유지하는 것이 좋음.

y_hat = [ ]:
y_hat은 모델이 테스트 데이터에 대해 예측한 결과를 저장할 리스트.
이 리스트는 모델의 예측 결과를 저장하기 위한 컨테이너 역할을 한다. 
테스트 데이터셋의 각 샘플에 대해 모델이 예측한 결과를 이 리스트에 추가할 수 있다.
빈 리스트로 초기화하는 것은 테스트 과정을 진행할 때 이전의 예측 결과가 영향을 주지 않도록 하기 위해서. 
이를 통해 이전 테스트의 결과와 새로운 테스트의 결과를 명확히 분리할 수 있다.

이 두 초기화 작업을 통해, 테스트 프로세스가 이전의 결과나 상태의 영향을 받지 않고 깨끗한 상태에서 시작할 수 있고, 
이렇게 하면 이후에 수행되는 계산이나 결과 분석이 이전 과정과 독립적으로 진행될 수 있어 결과의 정확성과 신뢰성을 높인다.
'''
# 변수의 초기화
test_loss = 0
y_hat = []

with torch.no_grad():
    x_ = x[2].split(batch_size, dim=0)
    y_ = y[2].split(batch_size, dim=0)

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)

        test_loss += float(loss)    # Gradient is already detached
        y_hat += [y_hat_i]

test_loss = test_loss / len(x_)
y_hat = torch.cat(y_hat, dim=0)     # BCE Loss 계산하기 위해서

sorted_history = sorted(zip(train_history, valid_history),
                        key=lambda x: x[1])

'''
1. zip(train_history, valid_history)는 두 리스트의 값을 튜플로 묶는다.
- train_history = [0.5, 0.3, 0.4]와 valid_history = [0.6, 0.2, 0.5]일 경우, zip(train_history, valid_history)는 [(0.5, 0.6), (0.3, 0.2), (0.4, 0.5)]을 반환
2. 이 묶은 튜플을 valid_history의 값, 즉 각 튜플의 두 번째 요소에 따라 정렬. (key = lambda x : x[1])
3. 결과적으로 sorted_history는 valid_history 값에 따라 정렬된 (훈련 손실, 검증 손실) 튜플들의 리스트이다.

이러한 정렬을 통해, valid_history 값을 기준으로 정렬된 train_history 값을 볼 수 있다. 
예를 들어, 검증 손실이 가장 낮은 때 훈련 손실이 얼마였는지, 
또는 검증 손실이 증가하거나 감소할 때 훈련 손실이 어떻게 변하는지 확인할 수 있다. 
이는 모델의 학습 및 일반화 성능을 분석하거나, 모델의 과적합 여부를 평가할 때 유용하다.
'''

print(sorted_history)
print("Train loss: %.4e" % sorted_history[0][0])
print("Valid loss: %.4e" % sorted_history[0][1])
print("Test loss: %.4e" % test_loss)

# 분류이므로 정확도(Accuracy) 계산도 가능
correct_cnt = (y[2] == (y_hat > .5)).sum()
total_cnt = float(y[2].size(0))

print("Test Accuracy : %.4f" % (correct_cnt / total_cnt))

# 예측값의 분포도 확인하기
df = pd.DataFrame(torch.cat([y[2], y_hat], dim=1).detach().numpy(),
                    columns = ["y", "y_hat"])
# print(df)
sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()

# AUROC
# scikit-learn 통해서 쉽게 구할 수 있다.
from sklearn.metrics import roc_auc_score

roc_auc_score(df.values[:, 0], df.values[:, 1])