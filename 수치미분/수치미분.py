import numpy as np

# 수치미분 구현
def numerical_derivative(f,x) :
    delta_x = 1e-4 #리미트에 해당되는 작은 값
    return (f(x+delta_x)-f(x-delta_x)) / (2*delta_x) # 분자, 분모

def my_func1(x) :
    return x**2

def my_func2(x) :
    return 3*x*(np.exp(x))

result = numerical_derivative(my_func1,3)
result2 = numerical_derivative(my_func2,2)

print(result)
print(result2)

#유튜브 링크 : https://www.youtube.com/watch?v=3ELMIbGTIDs&list=PLS8gIc2q83OjStGjdTF2LZtc0vefCAbnX&index=10