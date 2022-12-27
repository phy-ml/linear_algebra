import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from config.definitions import ROOT_DIR
import time

# img_path = os.path.join(ROOT_DIR, 'supplimentary_materials','daylilies.jpg')
# img = Image.open(img_path)

def ex_1():
    A = np.random.randint(1,10,size=(5,5))
    b = np.random.randint(1,10,size=(5)).reshape(-1,1)
    print(A)
    print(b)

def ex_2():
    A = np.random.randint(1, 10, size=(5, 5))
    b = np.random.randint(1, 10, size=(5)).reshape(-1, 1)

    # A x = b
    sol = np.linalg.solve(A,b)
    print(sol)

def ex_3():
    A = np.random.randint(1, 10, size=(5, 5))
    b = np.random.randint(1, 10, size=(5))#.reshape(-1, 1)

    # A x = b
    sol = np.linalg.solve(A, b)

    # check the residual
    # r = Ax - b
    residual = (A @ sol )- b
    print(residual.mean())

def ex_4():
    # implement LU decomposition
    A = np.random.randint(1, 10, size=(5, 5))
    b = np.random.randint(1, 10, size=(5))

    lu, P = lu_factor(A)
    sol = lu_solve((lu, P), b)

    # check the residual
    # r = Ax - b
    residual = (A @ sol) - b
    print(residual.mean())

def ex_6():
    # solution by using A inv
    A = np.random.randint(1, 10, size=(5, 5))
    b = np.random.randint(1, 10, size=(5))

    x = np.linalg.inv(A) @ b
    r2 = A @ x - b
    print(r2)

def ex_8():
    num = 5000
    # compare the three different methods
    A = np.random.rand(num, num)
    b = np.random.rand(num)

    # method 1) inverse
    time_inv = time.time()
    inv_x = np.linalg.inv(A) @ b
    err_1 = A @ inv_x - b
    print(f"Time for inverse function :{time.time() - time_inv} and error :{err_1.mean()}")

    # method 2) numpy solve func
    time_numpy = time.time()
    numpy_x = np.linalg.solve(A,b)
    err_2 = A @ numpy_x - b
    print(f"Time for numpy solve function :{time.time() - time_numpy} and error :{err_2.mean()}")

    # method 3) LU decomposition
    time_lu = time.time()
    lu, P = lu_factor(A)
    lu_x = lu_solve((lu, P), b)
    err_3 = A @ lu_x - b
    print(f"Time for LU function :{time.time() - time_lu} and error :{err_3.mean()}")

def ex_9():
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,10],
                  [9,11,12]])
    b = np.array([[1],
                  [2],
                  [3],
                  [4]])

    # solve the overdetermined equations using QR decomposition
    Q,R = np.linalg.qr(A)

    # solve the overdetermined equations using least square

    # solve the overdetermined equation using inverse function

if __name__ == "__main__":
    ex_8()