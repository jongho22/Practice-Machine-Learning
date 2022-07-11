# Logistic Regression 
# Training Data 특성과 분포를 나타내는 최적의 직선을 찾고
# 그 직선을 기준으로 데이터를 위(1) 또는 아래(0)등으로 분휴 해주는 알고리즘

import numpy as np

#학습데이터 준비
x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10,1)

#임의의 직선 z = Wx + b정의 
W = np.random.rand(1,1)
b = np.random.rand(1)
print(f"w = {W}, w.shanpe = {W.shape}, b = {b}, b.shape = {b.shape}")

#시그모이드 함수
def sigmoid(x) :
    return 1 / (1+np.exp(-x))

#손실함수 E(W,b) 정의
def loss_func(x,t) :
   
    delta = 1e-7

    z = np.dot(x,W) + b
    #출력 값 시그모이드 통과
    y = sigmoid(z)

    #cross-entropy
    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))

#수치미분 
def numerical_derivative(f,x) : #다변수 함수 , 모든 변수를 포함하고 있는 numpy 객체(배열,행렬)
    delta_x = 1e-4
    grad = np.zeros_like(x) #계산된 수치미분 값 저장 변수, 입력 값으로 들어온 numpy x와 동일한 형ㅐ로 0을 채움

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished : #변수 개수 만큼 반복
        idx = it.multi_index

        tmp_val = x[idx] #numpy 타입은 mutable 이므로 원래 값 보관
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()
    
    return grad

def error_val(x,t) :
    delta = 1e-7

    z = np.dot(x,W) + b
    y = sigmoid(z)

    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))

def predict(x) :

    z = np.dot(x,W) + b
    y = sigmoid(z)

    if y > 0.5 :
        result = 1
    else :
        result = 0

    return y, result

#학습률 초기화 및 손실함수가 최소가 될때까지 W,b 업데이트
learning_rate = 1e-2

f = lambda x : loss_func(x_data,t_data)
print(f"Inital error value = {error_val(x_data,t_data)},  Initial w = {W}, b = {b}")

for step in range(10001) :
    W -= learning_rate * numerical_derivative(f,W)
    b -= learning_rate * numerical_derivative(f,b)

    if step % 400 == 0 :
        print("step = ",step,"error value =", error_val(x_data,t_data), "w =", W, " b = ",b )

#미래 값 예측
(real_val, logical_val) = predict(3)
print(real_val,logical_val)

(real_val, logical_val) = predict(17)
print(real_val,logical_val)