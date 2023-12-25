import os
import numpy as np
import scipy
from config.definitions import ROOT_DIR
import pandas as pd


# laod the score and differential files
score_path = os.path.join(ROOT_DIR, "supplimentary_materials","Scores.mat")


def ex_1():
    # read the score and differential files
    score = scipy.io.loadmat(score_path)
    diff = scipy.io.loadmat(diff_path)

    # create index and columns names for the loaded matrix
    teams = ["Baylor", "Iowa State", "Uni Kansan", "Kansas State", "Uni Okhla","Okhla State", "Texas Christians", "Uni Texas Austin", "Texas Tech", "West Virginia"]

    score = pd.DataFrame(score['Scores'],columns=teams,index=teams)
    diff = pd.DataFrame(diff['Differentials'], columns=teams,index=teams)

    return {'score':score,
            'diff':diff}

def ex_2():
    # colleys matrix

    # get the score
    score = ex_1()['score']

    games = abs(score)
    total = games.sum(axis=1)
    colley_matrix = 2*np.eye(len(total)) + np.diag(total) - games
    right_side = 1+ 0.5*score.sum(axis=1)

    return {'colley_mat':colley_matrix,
            'right_side':right_side}

def ex_3():
    # solve the system of linear equation for ranking
    vals = ex_2()
    rating = np.linalg.solve(vals['colley_mat'], vals['right_side'])

    return rating

def ex_4():
    vals = ex_2()
    rating = ex_3()

    vals['colley_mat']['rate'] = rating
    return vals['colley_mat'].sort_values(by='rate',ascending=False)
    # return vals

def ex_5():
    # get the diff and scores
    vals = ex_1()
    diff_score = vals['diff'].values

    # create blank matrix to store vals
    P = np.zeros((45,10))
    B = np.zeros((45,1))

    count = 0
    for i in range(1,10):
        for j in range(i+1):
            if i != j:
                # print(count)
                P[count,i] = 1
                P[count,j] = -1
                B[count] = diff_score[i,j]

                count += 1

    return P, B

def ex_6():
    P,B = ex_5()

    A = P.T @ P
    D = P.T @ B
    return A, D

def ex_7():
    A,D = ex_6()
    # substitute the last line of A with ones and D with zeros
    A[-1] = 1
    D[-1] = 0

    return A, D

def ex_8():
    # Solve the equation
    A,D = ex_7()

    return np.linalg.solve(A,D)

def ex_9():
    # display the masseys results
    # get the score matrix
    score = ex_1()['diff']

    rank = ex_8()

    score['rank'] = rank

    return score.sort_values(by='rank',ascending=False)

def ex_10():
    colleys_rank = ex_3()
    masseys_rank = ex_8()

    score = ex_1()['diff']
    score['colley'] = colleys_rank
    score['massey'] = masseys_rank

    return score.sort_values(by='massey',ascending=False)

if __name__ == "__main__":
    print(ex_10())
