import numpy as np
# from multiprocessing import Pool
from pathos.pools import ProcessPool as Pool
# from pyina.launchers import Mpi as Pool

# =============================================================================
# ==================== DIFFERENTIAL EVOLUTION SOLVER ==========================
# =============================================================================


def optimizeDE(cost, xmin, xmax, npop=40, maxiter=1000, maxfun=1e+6,
               convergence_tol=1e-5, ngen=100, crossover=0.9, percent_change=0.9, MINMAX=1, x0=[], radius=0.2, parallel=False, nodes=16):
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import ChangeOverGeneration as COG
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from pathos.helpers import freeze_support
    freeze_support()  # help Windows use multiprocessing

    stepmon = VerboseMonitor(20)  # step for showing live update
    evalmon = Monitor()
    ndim = len(xmin)

    solver = DifferentialEvolutionSolver2(ndim, npop)
    if parallel:
        solver.SetMapper(Pool(nodes).map)
    if x0 == []:
        solver.SetRandomInitialPoints(xmin, xmax)
    else:
        solver.SetInitialPoints(x0, radius)
    solver.SetStrictRanges(xmin, xmax)
    solver.SetEvaluationLimits(maxiter, maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    tol = convergence_tol
    solver.Solve(cost, termination=COG(tolerance=tol, generations=ngen),
                 strategy=Best1Exp, CrossProbability=crossover,
                 ScalingFactor=percent_change)
    solved = solver.bestSolution
    func_max = MINMAX * solver.bestEnergy
    func_evals = solver.evaluations
    return solved, func_max, func_evals


# =============================================================================
# ====================== SIMULATED ANNEALING SOLVER ===========================
# =============================================================================

def Fast_SA(x0, k, T0, lower, upper):
    T = T0*np.exp(-k)
    u = np.random.uniform(0, 1, size=len(x0))
    xc = np.sign(u-0.5)*T*((1+1./T)**abs(2*u-1)-1.0)*(np.array(upper)-np.array(lower))
    x_new = [max(min(x0[i]+xc[i], upper[i]), lower[i]) for i in range(len(xc))]
    return x_new, T


def Cauchy_SA(x0, k, T0, lower, upper):
    T = float(T0)/(1+k)
    nb = np.squeeze(np.random.uniform(-np.pi/2, np.pi/2, size=len(x0)))
    xc = 0.5*T*np.tan(nb)
    x_new = [max(min(x0[i]+xc[i], upper[i]), lower[i]) for i in range(len(xc))]
    return x_new, T


def Boltzmann_SA(x0, k, T0, lower, upper):
    T = float(T0)/np.log(1+k)
    std = np.minimum(np.sqrt(T)*len(x0), (np.array(upper)-np.array(lower))/3/0.5)
    xc = np.squeeze(np.random.normal(0, 1, size=len(x0)))
    x_new = [max(min(x0[i]+xc[i]*std[i]*0.5, upper[i]), lower[i]) for i in range(len(xc))]
    return x_new, T


def Exp_SA(x0, k, T0, lower, upper):
    T = T0*0.90**k
    std = np.minimum(np.sqrt(T)*len(np.array(x0)), (np.array(upper)-np.array(lower))/3/0.5)
    xc = np.squeeze(np.random.normal(0, 1, size=len(x0)))
    x_new = [max(min(x0[i]+xc[i]*std[i]*0.5, upper[i]), lower[i]) for i in range(len(xc))]
    return x_new, T


def optimizeSA(func, xmin, xmax, x0=[], T0=100, schedule='Exp', niter=150, n_reanneal=20):
    if len(x0) == 0:
        x0 = [0.5]*len(xmin)
    else:
        x0 = np.array(x0)       # initial point
        T = T0                  # initial temperature, set to 100 if not information are available
        x_best = x0.copy()      # best point
        f_best = func(x0)       # current minimum
        f_current = func(x0)    # current evaluated function
        x_current = x0.copy()   # current evaluated point
        x_new = x0.copy()       # new evaluated point
        func_eval = 0
        for j in range(n_reanneal):
            na = 0              # number of accepted solution
            k = 1
            while na < niter:
                x_new, T = eval(schedule+'_SA(x_current, k, T0, xmin, xmax)')
                f_new = func(x_new)
                func_eval += 1
                dE = (f_new-f_current)
                accept = False
                if dE < 0:      # always accept new point if value is lower
                    accept = True
                    if f_new < f_best:
                        f_best = f_new
                        x_best = x_new
                else:
                    p = 1/(1+np.exp(dE/float(T)))
                    if p > np.random.uniform(0, 1):
                        accept = True
                if accept is True:
                    x_current = x_new
                    f_current = f_new
                    na += 1
                k += 1
            print('Reannealing n°', j, ': \n', 'value :', f_best)
        return x_best, f_best, func_eval
