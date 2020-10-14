#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:52:31 2018

@author: b15678
"""
import time
import openturns as ot
import pandas as pd
import numpy as np
from CanonicalTools import Constraint, Constraints, OptimSobol


'''
The optimization is only compatible with discrete measure,
the unimodal constraint is not handle yet.
The computation of the Sobol indices is realized using openturns,
warning messages can appear.
'''

def MyModel(x):
    return (x[:, 0]/(300*x[:, 1]*np.sqrt((x[:, 3] - x[:, 2])*0.0002)))**0.6


# =============================================================================
# ============================ CONSTRAINTS ====================================
# =============================================================================

dic1 = {1: ['eq', 1320.42], 2: ['eq', 2.1632e+06]}   # , 3: ['eq', 4.18077e+09]}
dic2 = {1: ['eq', 30], 2: ['eq', 949.137]}   # , 3: ['eq', 31422.3]}
dic3 = {1: ['eq', 50], 2: ['eq', 7501/3]}   # , 3: ['eq', 500200/4]}
dic4 = {1: ['eq', 54.5], 2: ['eq', 8911/3]}   # , 3: ['eq', 647569/4]}

Cons1 = Constraint([160, 3580], dic1)
Cons2 = Constraint([12.55, 47.45], dic2)
Cons3 = Constraint([49, 51], dic3)
Cons4 = Constraint([54, 55], dic4)

constraints = Constraints([Cons1, Cons2, Cons3, Cons4])

# =============================================================================
# ========================== INITIAL DISTRIBUTION =============================
# =============================================================================

distribution = []
lower = constraints.Lower()
upper = constraints.Upper()

# Variable #10 Q
distribution.append(ot.Gumbel(0.00524, 626.14))
distribution[0].setParameter(ot.GumbelAB()([1013, 558]))
distribution[0] = ot.TruncatedDistribution(distribution[0], float(lower[0]), float(upper[0]))
# Variable #22 Ks
distribution.append(ot.Normal(30, 7.5))
distribution[1] = ot.TruncatedDistribution(distribution[1], float(lower[1]), float(upper[1]))
# Variable #25 Zv
distribution.append(ot.Triangular(49, 50, 51))
# Variable #2 Zm
distribution.append(ot.Triangular(54, 54.5, 55))

# =============================================================================
# ================================= RUN =======================================
# =============================================================================

# Denote the input index
indexNumber = 3
# 1 for first order indice, 0 for total order
indexChoice = 1
# -1 for maximization, 1 for minimization
MINMAX = -1

Res = []
start = time.time()
solved, func_max, func_evals = OptimSobol(MyModel, distribution, indexNumber, constraints, indexChoice=indexChoice,
                                          solver='DE', maxiter=800, npop=40, MINMAX=MINMAX)
Res.append([indexNumber, func_max])
end = time.time()
print('time elapsed :', time.strftime('%H:%M:%S', time.localtime(end-start-3600)))


# %%
# =============================================================================
# ========================== RESULTS EXPORTATION ==============================
# =============================================================================

if False:
    name = 'Moment_Order_Constraint_1_v2'
    x = [sorted(Res)[i][0] for i in range(len(Res))]
    y = [sorted(Res)[i][1] for i in range(len(Res))]

#   export data as csv
    d = {'1-Threshold': x, '2-Probability of failure': y}
    my_df = pd.DataFrame(data=d)
    my_df.to_csv('./saved/'+name+'.csv', index=False, header=False, sep=' ')
    print(my_df)
