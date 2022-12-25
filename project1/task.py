import numpy as np

# create two matrics namely A and B
def main():

    # task 1
    A = np.array([[1,2,-10,4],
                  [3,4,5,-6],
                  [3,3,-2,5]])

    B = np.array([[3,3,4,2]])

    # task 2
    def printA_B():
        print(f"Length of A is :{len(A)}")
        print(f"Length of B is :{len(B)}")

    # task 3
    def add_4th_row():
        new_row = np.row_stack([A,B])
        return new_row

    # task 4
    def new_matrix_D():
        d = add_4th_row()
        return d[1:4, 2:4]

    # task 5
    def matrix_E():
        return new_matrix_D().T

    # task 6
    def E_size():
        m,n = matrix_E().shape[0], matrix_E().shape[1]
        return m,n

    # task 8
    def column_A():
        print(A)
        maxcolA = A.max(axis=0)
        mincolA = A.min(axis=0)
        sumcolA = A.sum(axis=0)
        meancolA = A.mean(axis=0)
        print(maxcolA,mincolA,sumcolA,meancolA)

    print(column_A())


if __name__ == "__main__":
    main()