#!/usr/bin/env python3
# -*- coding: utf-8 -*
import numpy as np
import openturns as ot
from scipy.special import binom
from scipy.linalg import eigvalsh_tridiagonal as eigenValues
from random import shuffle
from DE_Noisy import DE_Noisy
from CanonicalSolver import optimizeDE

# =============================================================================
# ============================ UTILITIES TOOLS ================================
# =============================================================================


def Affine_Transformation(lower, upper, m):
    '''
    Affine transformation of the moments sequence in [a,b] to [0,1]
    '''
    m = np.append([1], m)
    c = np.zeros(len(m))
    for n in range(len(m)):
        c[n] = 1/((upper - lower)**n)*sum([binom(n, j)*(-1*lower)**(n-j)*m[j] for j in range(n+1)])
    return c


def QD_Algorithm(c):
    '''
    QD return the canonical moments corresponding to the moment sequence,
    beware, the sequence must start with 1
    '''
    rg = len(c)-1
    Mat = [[0]*rg, [c[i+1]/c[i] for i in range(rg)]]
    for i in range(2, rg+1):
        if i % 2 == 0:
            Mat.append([Mat[i-1][t+1]-Mat[i-1][t]+Mat[i-2][t+1] for t in range(len(Mat[i-1])-1)])
        else:
            Mat.append([Mat[i-1][t+1]/Mat[i-1][t]*Mat[i-2][t+1] for t in range(len(Mat[i-1])-1)])
    Zeta = [Mat[i][0] for i in range(1, len(Mat))]
    p = [Zeta[0]]
    for i in range(1, len(Zeta)):
        p.append(Zeta[i]/(1-p[i-1]))
    if np.all([0 <= p[i] <= 1 for i in range(len(p))]):
        return p
    else:
        # print('Moment sequence does not correpond to a probability measure')
        return [-1]*len(p)


def Canonical_to_Position(lower, upper, p):
    '''
    Convert a sequence of Canonical Moment to the correponding discrete measure, 
    return the positions and weights of the measure,
    Number of points is int((len(p)-1)/2)+1
    '''
    lower = np.array(lower)
    upper = np.array(upper)
    if np.all([0 <= p[i] <= 1 for i in range(len(p))]):
        zeta = np.zeros(len(p))
        zeta[0] = p[0]
        for i in range(1, len(p)):
            zeta[i] = (1-p[i-1])*p[i]
        n = int((len(p)-1)/2)+1
        denominator = [0]*(n+1)
        numerator = [0]*(n)
        alpha = [0]*(n)
        beta = [0]*(n)

        denominator[0] = np.poly1d([1])
        numerator[0] = np.poly1d([1])

        denominator[1] = np.poly1d([lower + (upper-lower)*p[0]], True)
        beta[0] = 1
        alpha[0] = (lower + (upper-lower)*p[0])[0]
        for i in range(1, n):
            alpha[i] = (lower + (upper-lower)*(zeta[2*i-1]+zeta[2*i]))[0]
            beta[i] = ((upper-lower)**2*zeta[2*i-2]*zeta[2*i-1])[0]
            eig = eigenValues(alpha, np.sqrt(beta[1:]))
            denominator[i+1] = np.poly1d([alpha[i]], True)*denominator[i] - beta[i]*denominator[i-1]
        numerator[1] = np.poly1d([alpha[1]], True)*numerator[0]
        for i in range(1, n-1):
            numerator[i+1] = np.poly1d([alpha[i+1]], True)*numerator[i] - beta[i+1]*numerator[i-1]

        weights = numerator[len(numerator)-1](eig)/np.poly1d.deriv(denominator[len(denominator)-1])(eig)

        return eig, weights
    else:
        return -1


def LHSdesign(Z, Wgt, mode, N):
    '''
    Create a LHS design for a mixture of Uniform
    '''
    dim = len(Z)
    Pos = [[]]*dim
    Part = [[]]*dim
    Design = np.zeros((N, 4))
    for i in range(dim):
        Z[i].append(mode[i])
        Z[i].sort()
        Pos[i] = int(Z[i].index(mode[i]))
        Part[i] = [[]]*(len(Z[i])-1)
        for j in range(Pos[i]):
            Part[i][j] = int((N+1)*(Z[i][j+1] - Z[i][j])*sum([Wgt[i][k]/(Z[i][Pos[i]] - Z[i][k]) for k in range(j+1)]))
        for j in range(Pos[i]+1, len(Z[i])):
            Part[i][j-1] = int((N+1)*(Z[i][j] - Z[i][j-1])*sum([Wgt[i][k-1]/(Z[i][k]-Z[i][Pos[i]]) for k in range(j, len(Z[i]))]))
        if sum(Part[i]) != N:
            # print('Error, decomposition lead to different number of design point')
            Part[i][len(Part[i])-1] += int(N-sum(Part[i]))
        for j in range(len(Z[i])-1):
            if Part[i][j] != 0:
                Design[sum(Part[i][0:j]):sum(Part[i][0:j+1]), i] = list(np.array(ot.LHSExperiment(ot.Uniform(Z[i][j], Z[i][j+1]), Part[i][j]).generate()))
        shuffle(Design[:, i])

    return Design


def CostFunction(func, p, m, lower, upper, N, mode, threshold, design, MinMax):
    '''
    Return the probability of failure correponding to the sequence of Canonical
    in the general case where the input distribution can be continuous. 
    Should be used with the NoisyDE solver.
    '''
    dim = len(lower)
    # We concatenate p per block of variable
    if len(m) == dim:
        pp = []
        t = 0
        for i in range(dim):
            pp.append(p[t:t+len(m[i])+1])
            t = t + len(m[i])+1
    else:
        print('error size of moment vector')
    Z = [[]]*dim
    Wgt = [[]]*dim
    NewMom = [[]]*dim
    m_copy = m.copy()
    for i in range(dim):
        if mode[i] != None:
            m_copy[i] = np.append([1], m_copy[i])
            NewMom[i] = [(j+1)*m_copy[i][j] - (j)*mode[i]*m_copy[i][j-1] for j in range(1, len(m_copy[i]))]
            Z[i], Wgt[i] = Canonical_to_Position([lower[i]], [upper[i]], QD_Algorithm(Affine_Transformation(lower[i], upper[i], NewMom[i])) + pp[i])
        else:
            Z[i], Wgt[i] = Canonical_to_Position([lower[i]], [upper[i]], QD_Algorithm(Affine_Transformation(lower[i], upper[i], m_copy[i])) + pp[i])

    if not np.any([type(Z[i]) == int for i in range(len(Z))]):
        if design == 'MC':
            PERT = []
            for i in range(dim):
                if mode[i] != None:
                    U = []
                    for j in range(len(m[i])+1):
                        U.append(ot.Uniform(float(min(mode[i], Z[i][j])), float(max(mode[i], Z[i][j]))))
                    PERT.append(ot.Mixture(U, Wgt[i]))
                else:
                    U = []
                    for j in range(len(m[i])+1):
                        U.append(ot.Dirac(Z[i][j]))
                    PERT.append(ot.Mixture(U, Wgt[i]))
            DIST = ot.ComposedDistribution(PERT)
            Sample = DIST.getSample(N)
            return MinMax*sum(func(Sample) <= threshold)/N

        elif design == 'LHS':
            Sample = LHSdesign(Z, Wgt, mode, N)
            return MinMax*sum(func(Sample) <= threshold)/N
    else:
        return 1



def DiscreteCostFunction(func, p, m, lower, upper, threshold, MinMax):
    '''
    Return the probability of failure correponding to the sequence of Canonical
    in the case where every input is discrete.
    '''
    dim = len(lower)
    # We concatenate p per block of variable
    if len(m) == dim:
        pp = []
        t = 0
        for i in range(dim):
            pp.append(p[t:t+len(m[i])+1])
            t = t + len(m[i])+1
    else:
        print('error size of moment vector')

    P = [[]]*(dim)
    Position = [[]]*(dim)
    Weight = [[]]*(dim)
    for i in range(dim):
        P[i] = list(QD_Algorithm(Affine_Transformation(lower[i], upper[i], m[i]))) + list(pp[i])
        Position[i], Weight[i] = Canonical_to_Position([lower[i]], [upper[i]], P[i])

    Cardinal = [len(pp[i]) for i in range(dim)]
    PositionShuffle = np.ones((dim, np.prod(Cardinal)))
    WeightShuffle = np.ones((dim, np.prod(Cardinal)))
    for i in range(dim):
        PositionShuffle[i] = np.tile(np.repeat(Position[i], [np.prod(Cardinal[0:i])], axis=0), np.prod(Cardinal[i+1:dim+1]))
        WeightShuffle[i] = np.tile(np.repeat(Weight[i], [np.prod(Cardinal[0:i])], axis=0), np.prod(Cardinal[i+1:dim+1]))
    WeightShuffle = np.prod(WeightShuffle, axis=0)
    a = sum(WeightShuffle*(func(np.transpose(PositionShuffle)) >= threshold))

    return MinMax*(1-a)


def Optim(func, threshold, constraints, McNumber=5000, design='MC', MinMax=1, **kwargs,):

    lower = constraints.Lower()
    upper = constraints.Upper()

    def CostSolver(mp):
        mode, moment = constraints.Get_Moment_Constraint_Vector(mp)
        p = constraints.Get_Canonical_Moment_Vector(mp)
        if (mode == [None]*len(mode)):
            return DiscreteCostFunction(func, p, moment, lower, upper, threshold, MinMax)
        else:
            return CostFunction(func, p, moment, lower, upper, McNumber, mode, threshold, design, MinMax)

    xmin, xmax = constraints.Create_Optim_Bounds()
    if (constraints.Get_Moment_Constraint_Vector(xmin)[0] == [None]*len(lower)):
        solved, func_max, func_evals = optimizeDE(CostSolver, xmin, xmax, **kwargs)
        return solved, MinMax*func_max, func_evals
    else:
        solved, func_max, func_evals = DE_Noisy(CostSolver, xmin, xmax, **kwargs)
        return solved, MinMax*func_max, func_evals

# =============================================================================
# ======================== CLASS OF CONSTRAINT ================================
# =============================================================================

class Constraint:
    """Class defining one constraint:
        - Bounds [a, b]
        - Moment Constraints in a dictionnary with specificic format {MomentOder: ['type', Values]}
        for example {1: ['eq', 0.5], 2: ['ineq', 0.27, 0.34]}"""
    def __init__(self, Bounds, MomCons):
        self.Bounds = Bounds
        self.MomCons = MomCons

    def GetConstraintsNumber(self):
        '''Number of constraint of this variabel'''
        return len(self.MomCons) - list(self.MomCons.keys()).count(0)

    def VerifyConstraints(self):
        import numpy as np
        Bool = []
        for i in self.MomCons.keys():
            if (self.MomCons[i][0] == 'eq') and (len(self.MomCons[i]) == 2):
                Bool.append(True)
            elif (self.MomCons[i][0] == 'ineq') and (len(self.MomCons[i]) == 3):
                Bool.append(True)
            else:
                Bool.append(False)
        if not np.all(Bool):
            print('Wrong Constraint Expression')
        return np.all(Bool)

    def GetIneqConstraints(self):
        '''return the orders of the ineqality constraints,
        the lower bounds, and the upper bounds of this constraints'''
        if self.VerifyConstraints():
            a = []
            for i in self.MomCons.keys():
                if (self.MomCons[i][0] == 'ineq') and (len(self.MomCons[i]) == 3):
                    a.append([i, self.MomCons[i][1], self.MomCons[i][2]])
            return a

    def GetEqConstraints(self):
        '''return the orders of the Eqality constraints,
        and the value of this constraints'''
        if self.VerifyConstraints():
            a = []
            for i in self.MomCons.keys():
                if (self.MomCons[i][0] == 'eq') and (len(self.MomCons[i]) == 2):
                    a.append([i, self.MomCons[i][1]])
            return a

    def PrintConstraints(self):
        for i in self.MomCons.keys():
            if (self.MomCons[i][0] == 'ineq'):
                print('Moment', i, 'in: [', self.MomCons[i][1], ',', self.MomCons[i][2], '] \n')
            if (self.MomCons[i][0] == 'eq'):
                print('Moment', i, 'equals to: ', self.MomCons[i][1], '\n')


class Constraints(list):
    '''List of Constraint'''
    def Lower(self):
        '''Return the lower bounds of all variable'''
        lower = []
        for i in range(len(self)):
            lower.append(self[i].Bounds[0])
        return lower

    def Upper(self):
        '''Return the upper bounds of all variable'''
        upper = []
        for i in range(len(self)):
            upper.append(self[i].Bounds[1])
        return upper

    def Create_Optim_Bounds(self):
        '''We create the Lower and Upper bounds needed for the solver,
        it must be a list, we put the bounds correpsonding to the inequality moment constraints,
        then the bounds of the free canonical moment equals to 0, 1'''
        eps = 0.0001
        OptimVecLower = []
        OptimVecUpper = []
        for i in range(len(self)):
            for j in range(len(self[i].GetIneqConstraints())):
                OptimVecLower.append(self[i].GetIneqConstraints()[j][1])
                OptimVecUpper.append(self[i].GetIneqConstraints()[j][2])
            OptimVecLower += [0 + eps]*(self[i].GetConstraintsNumber()+1)
            OptimVecUpper += [1 - eps]*(self[i].GetConstraintsNumber()+1)
        return OptimVecLower, OptimVecUpper

    def Get_Moment_Constraint_Vector(self, OptimVec):
        '''From a vector with the previous structure we shall recompose the moment Constraint vector'''
        MomentVec = [0]*len(self)
        Mode = [None]*len(self)
        t = 0
        for i in range(len(self)):
            MomentVec[i] = [0]*(self[i].GetConstraintsNumber())
            for j in range(len(self[i].GetEqConstraints())):
                if self[i].GetEqConstraints()[j][0] == 0:
                    Mode[i] = self[i].GetEqConstraints()[j][1]
                else:
                    MomentVec[i][self[i].GetEqConstraints()[j][0]-1] = self[i].GetEqConstraints()[j][1]
            for j in range(len(self[i].GetIneqConstraints())):
                if self[i].GetIneqConstraints()[j][0] == 0:
                    Mode[i] = OptimVec[t]
                    t += 1
                else:
                    MomentVec[i][self[i].GetIneqConstraints()[j][0]-1] = OptimVec[t]
                    t += 1
            t += (self[i].GetConstraintsNumber()+1)

        return Mode, MomentVec

    def Get_Canonical_Moment_Vector(self, OptimVec):
        '''From a vector with the previous structure we shall recompose the Canonical moment vector'''
        CanonicalVec = []  # *sum([(self[i].GetConstraintsNumber()+1) for i in range(len(self))])
        t = 0
        for i in range(len(self)):
            t += len(self[i].GetIneqConstraints())
            for j in range(self[i].GetConstraintsNumber()+1):
                CanonicalVec.append(OptimVec[t])
                t += 1
        return CanonicalVec

    def PrintConstraints(self):
        for i in range(len(self)):
            print('Varible', i, 'bounded between:', self[i].Bounds, '\n')
            self[i].PrintConstraints()
