# Pytorch Auto Grad (자동 미분 기능)
import torch

x = torch.FloatTensor([[1, 2],
                       [3, 4]]).requires_grad_(True)

# .requires_grad_(True) : gradient(미분)에 참여할거니?

x1 = x + 5
x2 = x - 5
x3 = x1 * x2    # x**2 - 25
y = x3.sum()

print(x1)
print(x2)
print(x3)
print(y)

# 자연스럽게 x1, x2, x3, y는 x에 따라 .requires_grad_(True)의 성질을 그대로 가지게 된다.

y.backward()  # x4는 스칼라 값
# x, x1, x2, x3, y 모두 grad 속성에 그래디언트 값이 계산되어 저장되었을 것

# 파이토치로 미분한 값
print(x.grad)
print()
print(x)

'''
참고로 이 연산에 사용된 텐서들을 배열과 같은 곳에 저장할 경우 메모리 누수의 원인이 되므로 주의해야 한다.
.detach_()
'''

# x3.numpy()    에러 발생
x3.detach_().numpy
print()

# Gradient Descent
'''
Gradient Descent를 통해 Loss 함수 값을 minimize 하기 위한 parameter(입력 값)를 찾아나갈 수 있다.

어떤 가상의 함수를 모사하기 위해서 Loss 함수가 그 점수를 측정하고,
그 Loss를 최소화하기 위한 함수의 "파라미터"를 찾기 위함.

아래의 코드는 근본 원리대로 구현하는 방법
'''

import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.11, .22, .33],
                            [.44, .55, .66],
                            [.77, .88, .99]])

x = torch.rand_like(target)
'''
x = torch.rand_like(target)는 target과 동일한 크기의 텐서 x를 생성하고,
그 안의 각 요소를 0 ~ 1 사이에서 균일하게 무작위로 선택된 값으로 초기화하는 것을 의미
'''

print(x)
print()
# This means the final scalar will be differentiate by x.
x.requires_grad = True

# You cna get gradient of x, after differentiation
print(x)
print()

loss = F.mse_loss(x, target)
print(loss)
print()
# x가 target에 가까워질수록 loss도 작아짐.
# x_hat = argminL(x)

# 따라서 loss 값을 minimize 해 줄 필요가 있다.
# 효율적으로 파라미터를 업데이트하기 위해서 random으로 파라미터를 구하는 게 아니라
# 경사 하강법(Gradient Descent)을 통해서 파라미터를 업데이트 해주는 과정
threshold = 1e-5
learning_rate = 1
iter_cnt = 0    # iteration count

while loss > threshold:
    # loss > threshold 동안만 반복문 돌아간다.
    # loss <= threshold이면 반복문 종료
    iter_cnt += 1
    loss.backward() # Calculate gradients (미분)
    # loss를 미분하면 x.grad에 값이 들어간다.
    x = x - learning_rate * x.grad
    
    # You don't need to aware this now.
    x.detach_()
    x.requires_grad_(True)

    loss = F.mse_loss(x, target)

    print('%dth Loss: %.4e' % (iter_cnt, loss))
    print(x)
    print()