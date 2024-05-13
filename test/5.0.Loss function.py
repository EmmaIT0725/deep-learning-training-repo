# Mean Square Error(MSE) Loss
import torch

def mse(x_hat, x):
    # |x_hat| = (batch_size, dim)
    # |x| = (batch_size, dim)
    # x, x_hat 모두 n차원의 벡터
    y = ((x - x_hat)**2).mean()

    return y


x = torch.FloatTensor([[1, 2],
                        [3, 4]])

x_hat = torch.FloatTensor([[1, 1],
                            [1, 1]])

print(x.size(), x_hat.size())
print()
mse(x_hat, x)


# Predefined MSE in PyTorch
# 이미 정의되어있는 함수를 가져와서 사용하기
import torch.nn.functional as F

F.mse_loss(x_hat, x)

# reduction의 default 값은 mean
F.mse_loss(x_hat, x, reduction='sum')
F.mse_loss(x_hat, x, reduction='mean')

# reduction의 값을 none 으로 주면, 차원 축소 안하겠다는 말.
# (x_hat - x)**2 여기까지만 수행
F.mse_loss(x_hat, x, reduction='none')

# 객체로 assign 해서 수행하는 방법
import torch.nn as nn
mse_loss = nn.MSELoss()

mse_loss(x_hat, x)