import numpy as np
import matplotlib.pyplot as plt


def ex_1():
    t = np.linspace(0,2*np.pi, 4)

    # remove the last element
    t = t[:-1]

    v = np.asarray([np.cos(t), np.sin(t)])

    return t,v

def ex_2():
    T = np.array([[0.5,0],
                  [0,0.5]])
    x = [np.random.random() - 0.5, np.random.random() - 0.5]
    return T, x

def ex_3():
    plt.figure(figsize=(12,8))
    t,v = ex_1()
    T, x = ex_2()

    plt.plot(v[0,:],v[1,:],'o')
    plt.plot(x[0], x[1],'*')


    plt.show()

def ex_4():
    plt.figure(figsize=(12, 12))
    t, v = ex_1()
    nums = 10000
    x = np.zeros((nums, 2))

    T = np.array([[0.5,0],
                  [0,0.5]])

    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]

    plt.plot(v[0, :], v[1, :], 'o')

    for i in range(1,nums):
        k = np.random.randint(low=0, high=3)
        x[i,:] = T @ (x[i-1,:]- v[:, k]) + v[:,k]

    plt.plot(x[:,0], x[:,1],'*')
    #
    plt.show()

def ex_5():
    plt.figure(figsize=(12, 12))
    t, v = ex_1()
    nums = 10000
    x = np.zeros((nums, 2))

    T = np.array([[1/3, 0],
                  [0, 1/3]])

    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]

    plt.plot(v[0, :], v[1, :], 'o')

    for i in range(1, nums):
        k = np.random.randint(low=0, high=3)
        x[i, :] = T @ (x[i - 1, :] - v[:, k]) + v[:, k]

    plt.plot(x[:, 0], x[:, 1], '*')
    #
    plt.show()

def ex_6():
    plt.figure(figsize=(12, 12))
    t, v = ex_1()
    nums = 10000
    x = np.zeros((nums, 2))

    theta = np.pi /18

    T = 0.5 * np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]

    plt.plot(v[0, :], v[1, :], 'o')

    for i in range(1, nums):
        k = np.random.randint(low=0, high=3)
        x[i, :] = T @ (x[i - 1, :] - v[:, k]) + v[:, k]

    plt.plot(x[:, 0], x[:, 1], '*')
    #
    plt.show()

def ex_7():
    t = np.linspace(0,2*np.pi, 5)
    t = t[:-1]

    v = np.asarray([np.cos(t), np.sin(t)])

    plt.plot(v[0, :], v[1, :], 'o')
    # plt.plot(x[0], x[1], '*')

    nums = 10000
    x = np.zeros((nums, 2))

    theta = np.pi / 18

    T = np.array([[0.5,0],
                  [0,0.5]])

    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]


    for i in range(1, nums):
        k = np.random.randint(low=0, high=4)
        x[i, :] = T @ (x[i - 1, :] - v[:, k]) + v[:, k]

    plt.plot(x[:, 0], x[:, 1], '*')
    #
    plt.show()

def ex_8():
    t = np.linspace(0,2*np.pi, 5)
    t = t[:-1]

    v = np.asarray([np.cos(t), np.sin(t)])

    plt.plot(v[0, :], v[1, :], 'o')
    # plt.plot(x[0], x[1], '*')

    nums = 10000
    x = np.zeros((nums, 2))

    T = np.array([[1/3,0],
                  [0,1/3]])

    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]


    for i in range(1, nums):
        k = np.random.randint(low=0, high=4)
        x[i, :] = T @ (x[i - 1, :] - v[:, k]) + v[:, k]

    plt.plot(x[:, 0], x[:, 1], '*')
    #
    plt.show()

def ex_9():
    t = np.linspace(0, 2 * np.pi, 5)
    t = t[:-1]

    v = np.exp(1j * t)
    nums = 10000
    x = np.zeros((nums, 2))
    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]
    T = np.array([[0.5, 0],
                  [0, 0.5]])

    plt.plot(np.real(v), np.imag(v), 'o')

    k1 = 0
    for i in range(1,nums):
        k = np.random.randint(low=0, high=3)
        if k >= k1:
            k = k+1

        w = [np.real(v)[k], np.imag(v)[k]]
        x[i,:] = T @ (x[i-1,:] - w) + w
        k1 = k
    plt.plot(x[:, 0], x[:, 1], '*')
    plt.show()

def ex_10():
    T = np.array([[0.5, 0],
                  [0, 0.5]])

    t = np.linspace(0, 2 * np.pi, 5)
    t = t[:-1]

    v = np.exp(1j * t)
    nums = 10000
    x = np.zeros((nums, 2))
    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]

    plt.plot(np.real(v), np.imag(v), 'o')
    # plt.plot(x[0,0], x[1,0])


    k1 = 0

    for i in range(1, nums):
        k = np.random.randint(low=0, high=4)
        if (k != k1 + 2) and (k1 != k+2):
            w = [np.real(v)[k], np.imag(v)[k]]
            x[i,:] = T @ (x[i-1,:] - w) + w
            k1 = k

        else:
            x[i,:] = x[i-1,:]

    # plt.plot(x[:, 0], x[:, 1], '*')
    plt.show()

def ex_11():
    # barnsley Fern
    T1 = np.array([[0.85, 0.04],
                   [-0.04, 0.85]])
    T2 = np.array([[-0.15, 0.28],
                   [0.26, 0.24]])
    T3 = np.array([[0.2, -0.26],
                   [0.23, 0.22]])
    T4 = np.array([[0,0],
                   [0,0.16]])
    Q1 = np.array([0, 1.64])

    Q2 = np.array([-0.028, 1.05])

    Q3 = np.array([0, 1.6])

    Q4 = np.array([0,0])

    P1 = 0.85
    P2 = 0.07
    P3 = 0.07
    P4 = 0.01

    nums = 15000
    x = np.zeros((nums, 2))
    x[0,:] = np.random.random(2)

    plt.figure(figsize=(12, 12))
    plt.plot(x[0,0], x[0,1], '*')

    for i in range(1, nums):
        r = np.random.random(1)[0]
        if r <= P1:
            x[i,:] = T1 @ x[i-1,:] + Q1

        elif r <= P1 + P2:
            x[i, :] = T2 @ x[i - 1, :] + Q2

        elif r <= P1 + P2 + P3:
            x[i, :] = T3 @ x[i - 1, :] + Q3

        else:
            x[i, :] = T4 @ x[i - 1, :] + Q4

    plt.plot(x[:, 0], x[:, 1], '*')

    plt.show()

def ex_12():
    # Hexagon
    t = np.linspace(0, 2 * np.pi, 7)
    t = t[:-1]

    v = np.asarray([np.cos(t), np.sin(t)])

    plt.plot(v[0, :], v[1, :], 'o')
    # plt.plot(x[0], x[1], '*')

    nums = 10000
    x = np.zeros((nums, 2))

    T = np.array([[1/3, 0],
                  [0, 1/3]])
    #
    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]

    for i in range(1, nums):
        k = np.random.randint(low=0, high=6)
        x[i, :] = T @ (x[i - 1, :] - v[:, k]) + v[:, k]

    plt.plot(x[:, 0], x[:, 1], '*')
    plt.show()

def ex_13():
    t = np.linspace(0, 2 * np.pi, 7)
    t = t[:-1]

    v = np.exp(1j * t)
    print(v)
    nums = 10000
    x = np.zeros((nums, 2))
    x[0, :] = [np.random.random() - 0.5, np.random.random() - 0.5]
    T = np.array([[2/5, 0],
                  [0, 2/5]])

    plt.plot(np.real(v), np.imag(v), 'o')

    k1 = 0
    for i in range(1, nums):
        k = np.random.randint(low=0, high=5)
        if k >= k1:
            k = k + 1
        w = [np.real(v)[k], np.imag(v)[k]]
        x[i, :] = T @ (x[i - 1, :] - w) + w
        k1 = k
    plt.plot(x[:, 0], x[:, 1], '*')
    plt.show()

if __name__ == "__main__":
    ex_13()