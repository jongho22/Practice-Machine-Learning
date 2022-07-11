import numpy as np

#학습테이터 준비
x_data = np.array([1,2,3,4,5]).reshape(5,1)
t_data = np.array([2,3,4,5,6]).reshape(5,1)

#임의의 직선 y= wx + b정의 (임의의 값으로 가중치w, 편향 b 초기화) 
w = np.random.rand(1,1)
b = np.random.rand(1)
print(f"w = {w}, w.shanpe = {w.shape}, b = {b}, b.shape = {b.shape}")

#손실함수 E(w,b) 정의
def loss_func(x,t) :
    y = np.dot(x,w) + b

    return(np.sum((t-y)**2))/(len(x))
    
#손실함수 값 개선 함수
#입력변수 x,t
def error_val(x,t) :
    y = np.dot(x,w) + b

    return(np.sum((t-y)**2))/(len(x))

#학습을 마친후 임의의 데이터에 대해 미래 값 예측 함수
def predict(x) :
    y = np.dot(x,w) + b

    return y

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

learning_rate = 1e-2

f = lambda x : loss_func(x_data,t_data)

print(f"Inital error value = {error_val(x_data,t_data)},  Initial w = {w}, b = {b}")

#손실함수가 최소가 될 때까지 w,b 업데이트
for step in range(8001) :
    w -= learning_rate * numerical_derivative(f,w)
    b -= learning_rate * numerical_derivative(f,b)

    if (step % 400 == 0):
        print("step = ",step,"error value =", error_val(x_data,t_data), "w =", w, " b = ",b )

#미래 값 예측
print(predict(43))