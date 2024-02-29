#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt

def plot_dec_bound(h, predict_func, x1min, x1max, x2min, x2max, X, y, title, label_1, label_2): 
    """
    Purpose:
        Plot decision boundary on a [x1min,x1max] x [x2min,x2max] grid and plot a sample of points. 

    Inputs: 
        h             double, size of step in the grid that you use for determining decision 
                              boundary (choose e.g. 0.001)
        predict_func  function, a function that takes as input an array with two entries and 
                                outputs a real value
        x1min         double, lower bound to use for x1
        x1max         double, upper bound to use for x1
        x2min         double, lower bound to use for x2
        x2max         double, upper bound to use for x2
        X             matrix, n x 2 matrix where each row is a feature vector of training sample
        y             vector, n-vector containing the labels corresponding to the rows of X
        title         string, title to put above plot and as file name for the saved plot
        label_1       string, label to use for x-axis
        label_2       string, label to use for y-axis

    Return value:
        -    
    """  

    # create a mesh to plot in
    xx1, xx2 = np.meshgrid(np.arange(x1min+h, x1max-h, h),
                       np.arange(x2min+h, x2max-h, h))

    Z = np.zeros((xx1.shape[1],xx2.shape[0]))
    i=0
    j=0

    for x1 in xx1[0,:]:
       j=0
       for x2 in xx2[:,0]:
           Z[i,j] = predict_func([x1,x2]) 
           j=j+1
       i=i+1

    # Put the result into a color plot. cmap determines the color scheme, alpha determines
    # the opacity
    plt.contourf(xx1, xx2, Z.T, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.xlabel(label_1)
    plt.ylabel(label_2)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Also plot the training points 
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)

    plt.title(title)
    plt.savefig(title,dpi=600,bbox_inches='tight')
    plt.show()

def sim_sample(n, p, alpha0, alpha1, beta0, beta1):
    """
    Purpose:
        Simulate a sample with n examples from P

    Inputs: 
        n             int, length of sample to generate
        p             float, probability parameter Bernoulli dist
        alpha0        float, parameter alpha_0
        alpha1        float, parameter alpha_1
        beta0         float, parameter beta_0
        beta1         float, parameter beta_1
        
    Return value:
        y             vector, vector of n outcomes
        X             matrix, n x 2 matrix of simulated features
    """
    y = np.random.binomial(1,p,n)

    X = np.zeros((n,2))
    alpha_vec = [alpha0, alpha1]
    beta_vec = [beta0, beta1]
    
    for i in range(n):
        X[i,:] = np.random.beta(alpha_vec[y[i]],beta_vec[y[i]],size=2)

    return y, X


def hist_fit(X, y, r):
  """
    Purpose:
        Fit a histogram in [0,1]^2 with r*r bins

    Inputs: 
        X             matrix, n x 2 matrix with sample of feature vectors
        y             vector, n-dimensional vector containing labels corresponding to X
        r             int, number of bins per feature

    Return value:
        pred_mat      matrix, r x r matrix of predicted label per cell of histogram partition
        edges         matrix, 2 x (r + 1) matrix containing boundaries of cells of partition
  """
  hist1, edges = np.histogramdd(X[y == 1,:], bins=r, range=([0,1],[0,1]))
  hist0, _ = np.histogramdd(X[y == 0,:], bins=r, range=([0,1],[0,1]))
  pred_mat = (hist1 > hist0)
  
  return pred_mat, edges


def hist_predict(x_vec, pred_mat, edges): 
  """
    Purpose:
        Evaluate ERM histogram classifier for a single feature vector

    Inputs: 
        x_vec         vector, 2-dimensional vector (x_1,x_2) for which histogram classifier is to be evaluated
        pred_mat      matrix, r x r matrix of predicted label per cell of histogram partition
        edges         matrix, 2 x (r + 1) matrix containing boundaries of cells of partition

    Return value:
        prediction    int, predicted label, either 0 or 1 
  """
  row_index = np.where((edges[0][:-1] <= x_vec[0]) & (x_vec[0] < edges[0][1:]))
  col_index = np.where((edges[1][:-1]<= x_vec[1]) & (x_vec[1] < edges[1][1:]))
  
  prediction = pred_mat[row_index,col_index]

  return prediction 


def hist_predict_mult(X, pred_mat, edges):
  """
    Purpose:
        Evaluate ERM histogram classifier for n feature vectors

    Inputs: 
        X            matrix, n x 2 matrix with a feature vector (x_1,x_2) in each row
        pred_mat     matrix, r x r matrix of predicted label per cell of histogram partition
        edges        matrix, 2 x (r + 1) matrix containing boundaries of cells of partition

    Return value:
        predictions   vector, n-dimensional vector containing the predictions of a fitted
                              histogram classifier for each feature vector in X
  """
  predictions = np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    predictions[i] = hist_predict(X[i,:], pred_mat, edges)
    
  return predictions


def pred_func_simple_linear_model(w1, w2, b):
    def pred_func(x):
        x1, x2 = x
        hx = w1*x1 + w2*x2 + b
        return 1 if hx > 0 else -1 
    return pred_func

def pred_func_flexible_linear_model(m1, m2, m3, m4, m5, b):
    def pred_func(x):
        x1, x2 = x
        hx = m1*x1 + m2*x2 + m3*x1*x2 + m4*x1*x1 + m5*x2*x2 + b
        return 1 if hx > 0 else -1 
    return pred_func

def find_emperical_risk(X,Y, fit_func, predict_func):
    pred_mat, edges = fit_func(X, Y)

    emperical_risk = 0
    for x,y in zip(X,Y):
        pred = predict_func(x, pred_mat, edges)
        if pred != y:
            emperical_risk += 1

    emperical_risk /= len(X)

    return emperical_risk