import os, sys 
import numpy as np
import math
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Gaussian
from sklearn.metrics import r2_score
from statsmodels.genmod.families.links import identity, log
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def batchStudyGLM(y, X, kwargs):
    
    glm_results = list(map(lambda endog, exog: runGLM(endog, exog, **kwargs), y, X))

    return glm_results

    
def runGLM(y, X, p=0.7, model='sklearn', link='Poisson', custom_intercept=True):

    # for i in range(len(X)):
    #     print(np.unique(X[i,:])[:20])
    #     X[i,:] = X[i,:] / np.linalg.norm(X[i,:])
    #     print(np.unique(X[i,:])[:20])

    # y = y/np.sum(y)
    # y[y!=y] = 0

    if custom_intercept == True:
        avg = np.mean(y)

        intercept = np.ones(len(y)) * avg
        X = np.vstack((intercept, X))
    elif custom_intercept == False and model == 'statsmodels':
        intercept = np.ones(len(y)) 
        X = np.vstack((intercept, X))

    # renormalize full?
    # X = X / np.linalg.norm(X)
    ### TESTING REMOVE
    print(X.shape)
    X = X[:4,:]
    X[X != X] = 0
    # .reshape((1,-1))
    trainX, trainY, testX, testY = split_endog_exog(y, X, p=p)


    if model == 'sklearn':
        res, res_data, inp = fit_glm_sklearn(trainX, trainY, testX, testY, custom_intercept)
    elif model == 'statsmodels':
        res, res_data, inp = fit_glm_statsmodel(trainX, trainY, testX, testY, regularized=False)

    

    return res, res_data, inp


def fit_glm_sklearn(trainX, trainY, testX, testY, custom_intercept):

    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    # intercept added to X already
    if custom_intercept == True:
        clf = linear_model.PoissonRegressor(fit_intercept=False)
    else:
        clf = linear_model.PoissonRegressor(fit_intercept=True)
    # clf = make_pipeline(StandardScaler(), linear_model.PoissonRegressor())

    clf.fit(trainX.T, trainY)

    train_pred = clf.predict(trainX.T)

    train_score = clf.score(trainX.T, trainY)
    # train_score = mean_squared_error(train_pred, trainY)

    pred = clf.predict(testX.T)

    # for reg in trainX:
    #     fig = plt.figure(figsize=(8,3))
    #     plt.plot(reg)
    #     plt.show()

    # for reg in testX:
    #     fig = plt.figure(figsize=(8,3))
    #     plt.plot(reg)
    #     plt.show()

    # fig = plt.figure(figsize=(8,3))
    # plt.plot(np.arange(len(trainY)), trainY)
    # plt.show()

    # fig = plt.figure(figsize=(8,3))
    # plt.bar(np.arange(len(testY)), testY)
    # plt.show()

    # fig = plt.figure(figsize=(8,3))
    # ax = plt.subplot(1,1,1)
    # ax2 = ax.twinx()
    # ax2.plot(train_pred)
    # ax.plot(trainY)
    # plt.show()

    # stop()

    train_score = r2_score(trainY, train_pred)

    test_score = r2_score(testY, pred)


    test_score = clf.score(testX.T, testY)
    # test_score = mean_squared_error(testY, pred)

    # print(train_score, test_score)
    coeff = clf.coef_   

    intercept = clf.intercept_

    # pred = clf.predict(testX.T)

    fig = plt.figure(figsize=(12,3))
    ax=plt.subplot(1,1,1)
    ax.plot(trainY,color='k')
    # ax2 = ax.twinx()
    ax.plot(train_pred, color='r')
    # ax.set_xlim(0,500)
    plt.title(str(train_score))
    plt.show()


    test_score = 1 - (1-test_score) * (len(testX.T) - 1) / (len(testX.T) - len(testX) - 1)
    train_score = 1 - (1-train_score) * (len(trainX.T) - 1) / (len(trainX.T) - len(trainX) - 1)

    res = {'intercept': intercept, 'coeff': coeff, 'test_score': test_score, 'train_score': train_score}

    res_data = {'pred': pred, 'clf': clf}

    inp = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY}

    return res, res_data, inp

def fit_glm_statsmodel(trainX, trainY, testX, testY, regularized=False):

    poisson_model = sm.GLM(trainY, trainX.T, family=Gaussian(log()))

    if regularized:
        poisson_result = poisson_model.fit_regularized(alpha=0.1)
        b = poisson_result.params.reshape(-1, 1)
        intercept = b[0]
        coeff = b[1:]
        # DE = 1 - (poisson_result.deviance)/poisson_result.null_deviance
        DE = None
        AIC = poisson_result.aic
        BIC = poisson_result.bic
        fitted_vals = poisson_result.fittedvalues
        chi2 = poisson_result.pearson_chi2
        pvalues = poisson_result.pvalues
        tvalues= poisson_result.tvalues

    else:
        poisson_result = poisson_model.fit()
        b = poisson_result.params.reshape(-1, 1)
        intercept = b[0]
        coeff = b[1:]
        DE = 1 - (poisson_result.deviance)/poisson_result.null_deviance
        AIC = poisson_result.aic
        BIC = poisson_result.bic
        fitted_vals = poisson_result.fittedvalues
        chi2 = poisson_result.pearson_chi2
        pvalues = poisson_result.pvalues
        tvalues= poisson_result.tvalues

    trainPred = poisson_result.predict(trainX.T)
    testPred = poisson_result.predict(testX.T)

    train_score = r2_score(trainY, trainPred)

    test_score = r2_score(testY, testPred)

    test_score = 1 - (1-test_score) * (len(testX.T) - 1) / (len(testX.T) - len(testX) - 1)
    train_score = 1 - (1-train_score) * (len(trainX.T) - 1) / (len(trainX.T) - len(trainX) - 1)

    fig = plt.figure(figsize=(12,3))
    ax=plt.subplot(1,1,1)
    ax.plot(trainY,color='k')
    # ax2 = ax.twinx()
    ax.plot(trainPred, color='r')
    # ax.set_xlim(0,500)
    plt.title(str(train_score))
    # ax.set_ylim(0,10)
    # ax.set_xlim(300,600)
    plt.show()

    res = {'intercept': intercept, 'coeff': coeff, 'test_score': test_score, 'train_score': train_score, 'DE': DE, 'AIC': AIC, 'BIC': BIC, 'fitted_vals': fitted_vals, 'chi2': chi2, 'pvalues': pvalues, 'tvalues': tvalues}

    res_data = {'pred': testPred, 'model': poisson_model, 'result_obj':poisson_result}

    inp = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY}

    return res, res_data, inp


def split_endog_exog(y, X, p=0.7):
    """
    p is train-test split percentage
    """
    idx = int(p * len(y))

    trainY = y[:idx]
    testY = y[idx:]
    print(X.shape, y.shape, idx)
    trainX = X[:,:idx]
    testX = X[:,idx:]

    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    return trainX, trainY, testX, testY
