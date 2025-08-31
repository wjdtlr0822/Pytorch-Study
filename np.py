import numpy as np

# 1) 1차원 배열 생성
a = np.array([1, 2, 3, 4])
print("a =", a)

# 2) 2차원 배열 생성 (행렬)
b = np.array([[1, 2], [3, 4]])
print("b =\n", b)

# 3) 기본 연산
print("a + 10 =", a + 10)
print("b * 2 =\n", b * 2)

# 4) 행렬 연산
print("b 전치 =\n", b.T)
print("b * b 전치 =\n", np.dot(b, b.T))  # 행렬 곱

# 5) 브로드캐스팅
c = np.array([1, 2])
d = np.array([[10], [20]])
print("c + d =\n", c + d)