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

def paretofront_v2(P):
    '''
     Filters a set of points P according to Pareto dominance, i.e., points
     that are dominated (both weakly and strongly) are filtered.\n

     Note**: this version inserts numpy.nan to the non dominant solutions instead
     of eliminating them to pervent jagged arrays.
    
     Inputs: 
     - P    : N-by-D matrix, where N is the number of points and D is the 
              number of elements (objectives) of each point.
    
     Outputs:
     - P_copy   : Pareto-filtered P, dtype: float64.
    '''
    if P.ndim == 1:
        P = P[:,None]
    P_copy = P.astype('float64')
    dim = P.shape[1]
    i   = P.shape[0]-1
    idxs = np.arange(0,i+1,1)
    index = np.arange(0,i+1,1)
    while i >= 1:
        old_size = P.shape[0]
        a = bsxfun(P[i,:],P, fun='le')
        x = np.sum( bsxfun(P[i,:],P, fun='le'), axis=1,dtype=int)
        indices = np.not_equal(np.sum( bsxfun(P[i,:], P, fun='le'), axis=1,dtype=int),dim,dtype=int)
        indices[i] = True
        P = P[indices,:]
        idxs = idxs[indices]
        i = i - 1 - (old_size - P.shape[0]) + np.sum(indices[i:-1]);
    for i in index:
        if i not in idxs:
            P_copy[i,:] = np.nan
    return P_copy

def manual_paretofront(data_1,data_2,indices):
    data = np.column_stack((data_1,data_2))
    for i,j in enumerate(indices):
        indices[i]=j-1
    if data.dtype != 'float64':
        data = data.astype('float64')
    data[~indices,:] = np.nan
    return data
            

A = np.linspace(0,100,num=50)
B = np.linspace(100,0,num=50)
indices = np.array([1,50])
p = np.array(([1, 1, 1], [2, 0, 1], [2, -1, 1], [1, 1, 0]))
print('##############################\n')
data = manual_paretofront(A,B,indices)
print('##############################\n')
print(Index)