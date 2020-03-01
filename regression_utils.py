import numpy as np
from sklearn.metrics import r2_score

def simple_linear_regression(x, y):
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    dev_x = (x - mean_x)
    dev_y = (y - mean_y)
    
    b1 = np.sum(dev_x * dev_y)/np.sum((dev_x ** 2))
    b0 = mean_y - (b1 * mean_x)
  ######################
    return b1, b0


def multiple_regression(x, y):
    """
    x: np array of shape (n, p) where n is the number of samples
    and p is the number of features.
    y: np array of shape (n, ) where n is the number of samples
    return b: np array of shape (n, )
    """
    x.insert(loc = 0, column = "intercept", value = 1)
    b = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), y)

    ######################
    return b


def predict(x, b):
    #x['default']=1
    
    yhat = np.dot(x,b)
    ######################
    return yhat


def calculate_r2(y, yhat):
    # y: np array of shape (n,) where n is the number of samples
    # yhat: np array of shape (n,) where n is the number of samples
    # yhat is calculated by predict()

    # calculate the residual sum of squares (rss) and total sum of squares (tss)
    rss = np.sum((y-yhat)**2)
    tss = np.sum((y-np.mean(y))**2)
    
    ######################

    r2 = 1.0 - rss/tss
    return r2

def calculate_adjusted_r2(r2, n, k):
    
    r2_adj = 1- ((1-r2)*(n-1)/(n-k-1))
    
    return r2_adj

def check_r2(y, yhat):
    return np.allclose(calculate_r2(y, yhat), r2_score(y, yhat))