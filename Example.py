#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:39:51 2018

@author: b15678
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CanonicalTools import Constraint, Constraints, Optim


# Definition of the numerical model
def myModel(x):
    return (x[:, 0]/(300*x[:, 1]*np.sqrt((x[:, 3] - x[:, 2])*0.0002)))**0.6


# The constraints are enforced in a dictionnary
# The constraint n°0 is the mode, meaning the extreme points are not discrete.
# Optimization is much faster without any mode, as it uses the 'DiscreteCostFunction'.
# The constraint 'eq' correspond to equality constraint, 'ineq' to inequality in between the bounds
dic1 = {1: ['ineq', 1300.42, 1340.42], 2: ['eq', 2.1632e+06]}
dic2 = {0: ['eq', 30], 1: ['eq', 30], 2: ['eq', 949.137]}
dic3 = {0: ['eq', 50], 1: ['eq', 50]}
dic4 = {0: ['eq', 54.5], 1: ['eq', 54.5]}

# The bounds and the constraints are enforced using the class 'Constraint'
Cons1 = Constraint([160, 3580], dic1)
Cons2 = Constraint([12.55, 47.45], dic2)
Cons3 = Constraint([49, 51], dic3)
Cons4 = Constraint([54, 55], dic4)

# The constraints are concatenated using the class 'Constraints'
constraints = Constraints([Cons1, Cons2, Cons3, Cons4])

# Minimization of the probability of failure for the given thresholds, 
# it yields the reconstruction of the CDF lower envelope 
Res = []
start = time.time()
for threshold in [4]:
    solved, func_max, func_evals = Optim(func=myModel, threshold=threshold, constraints=constraints, maxiter=400, McNumber=5000, npop=40)
    Res.append([threshold, func_max, solved])
    name = 'Unimodal_Inequality_Constraint_2_Hydraulic_Model_end'
    x = [Res[i][0] for i in range(len(Res))]
    y = [Res[i][1] for i in range(len(Res))]
    d = {'1-Threshold': x, '2-Probability of failure': y}
    my_df = pd.DataFrame(d)
    my_df.to_csv('./saved/' + name + '.csv', index=False, header=False, sep=' ')
end = time.time()
print('time elapsed :', time.strftime('%H:%M:%S', time.localtime(end-start-3600)))


# %% ========================= PRINT RESULT ===================================

Print = False
if Print:
    name = 'Unimodal_Inequality_Constraint_2_Hydraulic_Mode'
    x = [Res[i][0] for i in range(len(Res))]
    y = [Res[i][1] for i in range(len(Res))]

    plt.figure(figsize=(12, 6.1))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.ylabel(r'$\inf_{\mu\in\mathcal{A}} \; F_{\mu}(h)$', fontsize=24)
    plt.xlabel('h', fontsize=24)

    # plot the curve
    plt.plot(x, y, marker='.')

    # export file for latex, put png to save as image
    plt.savefig('./saved/' + name + '.pdf')

    # export data as csv
    d = {'1-Threshold': x, '2-Probability of failure': z}
    my_df = pd.DataFrame(d)
    my_df.to_csv('./saved/' + name + '.csv', index=False,
                 header=False, sep=' ')
    print(my_df)
