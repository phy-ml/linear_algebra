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
    x1 = np.linalg.inv(R) @ (Q.T @ b)

    # solve the overdetermined equations using least square
    x2 = np.linalg.lstsq(A,b)[0]

    # print the norm from both the approach
    er_1 = np.linalg.norm(A @ x1 - b)
    er_2 = np.linalg.norm(A @ x2 - b)

    print(f"Norm for QR is :{er_1} and for least sqr :{er_2}")

def ex_10():
    # under determined system
    A = np.random.randn(3,5)
    b = np.random.randn(3,1)


    # compute x using SVD
    # Ax = b
    # U,S, Vinv x= b
    # x = Vinv.T @ Sinv@ U.T @ b
    U, S, Vinv = np.linalg.svd(A,full_matrices=False)
    x_1 = Vinv.T @ np.diag(1/S) @ U.T @ b

    # compute x using psudo inverse function
    x_2 = np.linalg.pinv(A) @ b

    # compute x using least square function
    x_3 = np.linalg.lstsq(A,b)

    print(f'Solution of SVD :{x_1}')
    print(f"Solution of psudo inverse :{x_2}")
    print(f"Solution of least square :{x_3[0]}")


if __name__ == "__main__":
    ex_10()