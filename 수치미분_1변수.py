# 입력 변수가 하나 이상인 다 변수 함수의 경우, 입력 변수는 서로 독립적이기 때문에
# 수치미분 또한 변수의 개수만큼 개별적으로 계산하여야함

import numpy as np

def numerical_derivative(f,x) : #다변수 함수 , 모든 변수를 포함하고 있는 numpy 객체(배열,행렬)
    delta_x = 1e-4
    grad = np.zeros_like(x) #계산된 수치미분 값 저장 변수, 입력 값으로 들어온 numpy x와 동일한 형ㅐ로 0을 채움

    print(x) #1
    print(grad) #
    print('-'*30)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished : #변수 개수 만큼 반복
        idx = it.multi_index

        print(idx, x[idx]) #3

        tmp_val = x[idx] #numpy 타입은 mutable 이므로 원래 값 보관
        
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x) # f(x-delta_x)

        grad[idx] = (fx1 - fx2) / (2*delta_x)

        print(grad[idx]) #4
        print(grad) #5
        print('-'*30)

        x[idx] = tmp_val
        it.iternext()
    
    return grad

def func1(input_obj) :
    x = input_obj[0]
    return x**2

numerical_derivative(func1, np.array([3.0]))


