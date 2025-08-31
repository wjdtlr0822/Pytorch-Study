import numpy as np

if __name__ == "__main__":
    a = np.array([1, 2, 3])
    print(a)
    print(a.T)

    print(np.dot(a,a))

    c = np.array([[1,2,3]])
    d = np.array([[10],[20],[30]])
    print(c+d)