import collections
import copy
import os
import re
import csv
import enum
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from scipy.signal import butter, filtfilt
import importlib
from tabulate import tabulate
from numpy import nanmean, nanstd


def bsxfun(A,B,fun,dtype=int):
    '''bsxfun implemented from matlab bsxfun,\n
    functions:\n
        -ge: A greater than or equal B \n
        -le: A less than or equal B \n
        -gt: A greater than B \n
        -lt: A less than B \n
    '''
    if A.ndim == 1 and B.ndim == 1 :
        C = np.zeros((A.shape[0],B.shape[0]))
        for i in range(B.shape[0]):
            if fun == 'ge':
                C[:,i] = np.greater_equal(A,B[i],dtype=dtype)
            elif fun == 'gt':
                C[:,i] = np.greater(A,B[i],dtype=dtype)
            elif fun == 'le':
                C[:,i] = np.less_equal(A,B[i],dtype=dtype)
            elif fun == 'lt':
                C[:,i] = np.less(A,B[i],dtype=dtype)
            else:
                raise Exception('function is not defined')
    if A.ndim == 1 and B.ndim != 1:
        C = np.zeros((B.shape[0],A.shape[0]))
        for i in range(B.shape[0]):
            if fun == 'ge':
                C[i,:] = np.greater_equal(A,B[i,:],dtype=dtype)
            elif fun == 'gt':
                C[i,:] = np.greater(A,B[i,:],dtype=dtype)
            elif fun == 'le':
                C[i,:] = np.less_equal(A,B[i,:],dtype=dtype)
            elif fun == 'lt':
                C[i,:] = np.less(A,B[i,:],dtype=dtype)
            else:
                raise Exception('function is not defined')
    elif A.ndim == 2:
        C = np.zeros((A.shape[0],B.shape[0]))
        for i in range(A.shape[0]):
            if fun == 'ge':
                C[i,:] = np.greater_equal(A[i,:],B,dtype=dtype)
            elif fun == 'gt':
                C[i,:] = np.greater(A[i,:],B,dtype=dtype)
            elif fun == 'le':
                C[i,:] = np.less_equal(A[i,:],B,dtype=dtype)
            elif fun == 'lt':
                C[i,:] = np.less(A[i,:],B,dtype=dtype)
            else:
                raise Exception('function is not defined')
    return C

def paretofront(P):
    '''
     Filters a set of points P according to Pareto dominance, i.e., points
     that are dominated (both weakly and strongly) are filtered.
    
     Inputs: 
     - P    : N-by-D matrix, where N is the number of points and D is the 
              number of elements (objectives) of each point.
    
     Outputs:
     - P    : Pareto-filtered P
     - idxs : indices of the non-dominated solutions
    
    Example:\n
    p = [1 1 1; 2 0 1; 2 -1 1; 1, 1, 0];
     [f, idxs] = paretoFront(p)
         f = [1 1 1; 2 0 1]
         idxs = [1; 2]
    '''
    dim = P.shape[1]
    i   = P.shape[0]-1
    idxs= np.arange(0,i+1,1)
    while i >= 1:
        old_size = P.shape[0]
        a = bsxfun(P[i,:],P, fun='le')
        x = np.sum( bsxfun(P[i,:],P, fun='le'), axis=1,dtype=int)
        indices = np.not_equal(np.sum( bsxfun(P[i,:], P, fun='le'), axis=1,dtype=int),dim,dtype=int)
        indices[i] = True
        P = P[indices,:]
        idxs = idxs[indices]
        i = i - 1 - (old_size - P.shape[0]) + np.sum(indices[i:-1]);
    return P,idxs

A = np.array([8,17,20,24])
B = np.array([0,10,21])
p = np.array(([1, 1, 1], [2, 0, 1], [2, -1, 1], [1, 1, 0]))
p1 = np.array([1, 1, 1])
c = bsxfun(A,B,fun='gt')
d = bsxfun(p1,p,fun='le')
P,idxs = paretofront(p)
print('p')
print(P)
print('idxs')
print(idxs)