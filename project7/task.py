import os
from config.definitions import ROOT_DIR
import scipy
import numpy as np

# load user_movies.mat, which contains following :
# 1) movies
# 2) user_movies
# 3) user_movies_sort
# 4) index_small
# 5) trial_user

# path to load the mat file
path = os.path.join(ROOT_DIR, "supplimentary_materials","users_movies.mat")

def ex_1():
    data = scipy.io.loadmat(path)
    # print(data)
    movies = data['movies']
    user_movies = data['users_movies']
    index_small = data['index_small']
    user_movies_sort = data['users_movies_sort']
    trial_user = data['trial_user']

    return {'movie':movies,
            'user_movies':user_movies,
            'index_small':index_small,
            'user_movies_sort':user_movies_sort,
            'trial_user':trial_user}

def ex_2():
    data = ex_1()

    print(f'Rating based on movies:\n')

    for i in data['index_small']:
        print(data['movie'][i])

    print('**'*20)


def ex_3():
    data = ex_1()
    user_movie_sort = data['user_movies_sort']

    rating = []

    for i in range(len(user_movie_sort)):
        temp = np.array(user_movie_sort[i])
        if np.prod(temp) != 0:
            rating.append(user_movie_sort[i])

    rating = np.array(rating)
    return rating

def ex_4():
    rating = ex_3()
    trial_user = np.array(ex_1()['trial_user'])

    # check l2 formula from numpy
    new_dist = []
    for i in range(len(rating)):
        new_dist.append(np.linalg.norm(rating[i]-trial_user, 2))

    return new_dist

def ex_5():
    # get the original array
    rating = ex_4()

    # sort the rating
    rating.sort()

    print(rating)

def ex_6():
    # get the original data
    rating = ex_3()
    rating_mean = rating - rating.mean(axis=1).reshape(-1,1)

    trial_user = np.array(ex_1()['trial_user'])
    trial_mean = trial_user - trial_user.mean().flatten().reshape(-1,1)

    return rating_mean, trial_mean

def ex_7():
    # compute pearson correlation between rating and trail users
    # get the vals of mean rating and trail_users
    rating_mean, trail_mean = ex_6()
    # print(rating_mean)
    p_coef = []

    for i in range(len(rating_mean)):
        coef = ((rating_mean[i] * trail_mean).sum()) / ( (np.sqrt((rating_mean[i]**2).sum())) * (np.sqrt((trail_mean**2).sum())) )
        p_coef.append(coef)
    return p_coef

def ex_8():
    # get the list of p_coefs
    coef = ex_7()
    coef.sort()
    print(coef)

def ex_9():
    # compare the euclidean dist and pearson coefs results
    euclid = ex_4()
    p_coef = ex_7()

    for i,j in zip(euclid, p_coef):
        print(f"E-Dist :{i} and P-Coef :{j}")

    # some plotting to visualize the relation of p_coef and eluicidian dist
    # import matplotlib.pyplot as plt
    # plt.scatter(p_coef,euclid, color = 'black')
    # plt.scatter(list(range(len(euclid))), p_coef, color='red')
    # plt.show()

def explain_input():
    # get base dataset
    base = ex_1()
    movie = base['movie']
    user_movies = base['user_movies']
    index_small = base['index_small']
    trial_user = base['trial_user']
    user_movies_sort = base['user_movies_sort']

    print(f'Movie array shape {movie.shape}')
    print(movie)
    print('---------------------------------------------')

    print(f'user_movies array shape {user_movies.shape}')
    print(user_movies)
    print('---------------------------------------------')

    #  'user_movies_sort':user_movies_sort
    print(f'user_movies_sort array shape {user_movies_sort.shape}')
    print(user_movies_sort)
    print('---------------------------------------------')

    print(f'index_small array shape {index_small.shape}')
    print(index_small)
    print('---------------------------------------------')

    print(f'trial_user array shape {trial_user.shape}')
    print(trial_user)
    print('---------------------------------------------')

def ex_10():
    # need to work on exercises 10-14 >> incomplete
    pass

if __name__ == "__main__":
    # print(ex_1())
    explain_input()
    ex_9()