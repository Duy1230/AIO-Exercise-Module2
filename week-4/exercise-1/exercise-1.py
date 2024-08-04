import numpy as np


def compute_mean(x):
    # your code here*******************
    return np.mean(x)


def compute_median(x):
    size = len(x)
    x = np.sort(x)
    if (size % 2 != 0):
        # your code here*******************
        return x[(size)//2]
    else:
        # your code here*******************
        return 1/2 * (x[(size-1)//2] + x[(size-1)//2 + 1])


def compute_std(x):
    mean = compute_mean(x)
    variance = 0
    # your code here*******************
    variance = np.sum([(i - mean)**2 for i in x]) / len(x)
    return np.sqrt(variance)


def compute_correlation_cofficient(X, Y):
    n = len(X)
    numberator = 0
    denominator = 0
    # your code here*******************
    numberator = n * \
        np.sum(X*Y) - np.sum(X)*np.sum(Y)
    denominator = np.sqrt(n*np.sum(X**2) - np.sum(X)**2) * \
        np.sqrt(n*np.sum(Y**2) - np.sum(Y)**2)

    return np.round(numberator / denominator, 2)


X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print("Mean: ", compute_mean(X))

X = [1, 5, 4, 4, 9, 13]
print("Median: ", compute_median(X))

X = [171, 176, 155, 167, 169, 182]
print("STD: ", compute_std(X))

X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print("Correlation : ", compute_correlation_cofficient(X, Y))
