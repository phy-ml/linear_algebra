import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from config.definitions import ROOT_DIR

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

if __name__ == "__main__":
    ex_6()