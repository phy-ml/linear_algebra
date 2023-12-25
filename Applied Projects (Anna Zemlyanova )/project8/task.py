import os
from config.definitions import ROOT_DIR
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# path to load the mat file
path = os.path.join(ROOT_DIR, "supplimentary_materials","temperature.mat")

# read the file
data = scipy.io.loadmat(path)

def ex_1():
    weather_high = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])
    plt.plot(weather_high)
    plt.show()

def ex_2():
    weather_high = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])

    # use below given index
    index = [0,4,7,11]
    data = [weather_high[i] for i in index]
    vander_matrix = np.vander(data)

    coef = np.linalg.lstsq(vander_matrix, data, rcond=None)[0]

    # get the prediction
    pred = vander_matrix @ coef

    # plt.plot(data, label='Given')
    # plt.plot(pred, label='Pred')
    # plt.show()
    return coef, data

def ex_3():
    coef, data = ex_2()
    data = list()
    poly_val = np.polyval(coef, data)
    plt.plot(poly_val, data)
    plt.show()

def ex_4():
    # perform polynomial fit over 6 month of data
    weather_high = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])

    # use below given index
    index = [0, 2, 4, 7, 9, 11]
    data = [weather_high[i] for i in index]

    vander_matrix = np.vander(data)

    # use polynomial fit
    coef = np.polyfit(index, data, deg=5)

    # generate the y func
    y_fun = np.poly1d(coef)

    pred = y_fun(index)

    plt.plot(index,data,'o', label='Given')
    plt.plot(index,pred, 'x',label='Pred',color='black')
    plt.plot(index,pred, label='Pred')
    plt.legend()
    plt.show()

def ex_5():
    weather_high = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])
    index = list(range(len(weather_high)))

    # iterate through various degree of fittings and observe the changes

    # blank dict to save results
    dic = {}

    for deg in range(4,13):
        temp_coef = np.polyfit(index, weather_high, deg=deg)
        y_func = np.poly1d(temp_coef)
        temp_pred = y_func(index)
        dic[deg] = temp_pred

    # plot the predictions
    plt.figure(figsize=(12,8))
    plt.plot(index, weather_high, label='Given')
    for key, val in dic.items():
        plt.plot(index, val, label=f"Degree :{key}")

    plt.legend()
    plt.show()

def ex_6():
    weather_high = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])
    index = list(range(len(weather_high)))

    y_linear = np.interp(index, index, weather_high)
    y_pchip = interpolate.PchipInterpolator(x=index, y=weather_high)(index)
    y_spline = interpolate.CubicSpline(index, weather_high)(index)

    print(y_linear)
    print(y_pchip)
    print(y_spline)

def ex_7():
    return data

def ex_8():
    data = ex_7()['temperature']

    # load year and corresponding year
    year = data[:,0]
    temp = data[:,1]

    plt.plot(year, temp, 'o', color='black', ls='--')
    plt.show()

def ex_9():
    data = ex_7()['temperature']

    # load year and corresponding year
    year = data[:, 0]
    temp = data[:, 1]

    # the range for extrapolation
    extpol_year = np.arange(2017,2027, 1)
    # print(temp)

    # perform extrapolation

    # use linear function
    y_linear = np.interp(extpol_year, year, temp)

    # use the pchip
    y_pchip = interpolate.PchipInterpolator(year,temp)(extpol_year)

    # use the spline function
    y_spline = interpolate.CubicSpline(year, temp)(extpol_year)

    # plot the extrapolated points
    plt.plot(year, temp, color='black', ls='--')
    plt.plot(extpol_year, y_linear, ls='--', color='red', label='Linear')
    plt.plot(extpol_year, y_pchip, ls=':', color='blue', label='PChip')
    plt.plot(extpol_year, y_spline, ls='-.', color='green', label='Cubic Spline')
    plt.legend()
    plt.show()

def get_data():
    data = ex_7()['temperature']

    # load year and corresponding year
    year = data[:, 0]
    temp = data[:, 1]
    return year, temp

def ex_10():
    year, temp = get_data()

    # find the averge temperature
    print(f'Mean temperature is :{temp.mean()}')

    return temp.mean(())

def ex_11():
    # find the orthogonal projection of vector t on a vector of ones
    year, temp = get_data()

    ones_vec = np.ones_like(temp)

    # find the projection of t on ones
    proj_t = ((ones_vec @ temp) * ones_vec) / (ones_vec.T @ ones_vec)

    return proj_t, year, temp

def ex_12():
    proj_t, year, temp = ex_11()

    plt.plot(year, temp, 'o', color='black', ls='--')
    plt.plot(year, proj_t, '--', color='green')
    plt.show()

def ex_14():
    # assert the normality of the created projection
    proj_t, year, temp = ex_11()

    ones_vec = np.ones_like(temp)

    # get the projection matrix P
    proj_matrix = (ones_vec. T @ ones_vec) /(ones_vec.T @ ones_vec)
    # print(proj_matrix.T @ proj_matrix - proj_matrix)
    print(np.linalg.norm(proj_matrix*proj_matrix - proj_matrix))

def ex_15():
    # get the data
    year, temp = get_data()

    year = year.reshape(-1,1)
    temp = temp.reshape(-1, 1)

    # prepare the data
    one_col = np.ones_like(temp)

    # compute orthonormal basis
    B = np.hstack([one_col, year])
    Q = scipy.linalg.orth(B)

    # print(Q.shape)

    return Q

def ex_16():
    Q = ex_15()

    # compute the projection matrix
    p2 = Q @ Q.T

    # compute the norm
    norm = np.linalg.norm(p2.T @ p2 - p2)
    print(norm)

def ex_17():
    # perform same thing but use quadratic equations
    # get the data
    year, temp = get_data()

    year = year.reshape(-1, 1)
    temp = temp.reshape(-1, 1)

    # prepare the data
    one_col = np.ones_like(temp)

    # compute orthonormal basis
    B = np.hstack([one_col, year, year**2])
    Q = scipy.linalg.orth(B)

    return Q

def ex_18():
    # get the data
    year, temp = get_data()
    year = year.reshape(-1, 1)
    temp = temp.reshape(-1, 1)

    # get the ones projection
    B1, _, _ = ex_11()

    # get the linear matrix projection
    Q2 = ex_15()
    proj_2 = Q2 @ Q2.T
    pred_2 = proj_2 @ temp

    # get the quadratic projection
    Q3 = ex_17()
    proj_3 = Q3 @ Q3.T
    pred_3 = proj_3 @ temp



    # plot the data >> uncomment to see the different preds <<
    # plt.plot(year, temp, 'o', color='black', ls='--')
    # plt.plot(year, B1, '--', color='green')
    # plt.plot(year, pred_2, '--', color='red')
    # plt.plot(year, pred_3, '--', color='gold')
    # plt.show()

    return pred_3, year, temp

def ex_19():
    # get the quadratic pred
    pred_3, _, _ = ex_18()
    # print(pred_3.ravel())

    data = ex_7()['temperature']

    # load year and corresponding year
    year = data[:, 0]
    temp = data[:, 1]

    # the range for extrapolation
    extpol_year = np.arange(2016, 2116, 1)

    # use linear function
    interpol = interpolate.CubicSpline(year, pred_3.ravel())(extpol_year)

    # plot the extrapolated points
    plt.plot(year, temp, color='black', ls='--')
    plt.plot(extpol_year, interpol, ls='--', color='red', label='Spline')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ex_19()