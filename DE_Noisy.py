import numpy as np

# =============================================================================
# ==================== DIFFERENTIAL EVOLUTION SOLVER ==========================
# =============================================================================


def getCount(arr, num1, num2):
    count = 0
    for i in range(len(arr)):
        if num1 <= arr[i] <= num2:
            count += 1
    return count


def compete(xe, xoff, Ye, Yoff, fun_cost, sigma, nmax):
    ''' If winner==1, this means the offspring win'''

    winner = 0
    fe = np.mean(Ye)
    foff = np.mean(Yoff)

    if np.abs(fe - foff) > 2*sigma:
        if foff <= fe:
            winner = 1
        # print('No evaluation needed', winner)
        return [winner, Ye, Yoff, fe, foff]
    else:
        alpha = min(fe, foff)
        beta = max(fe, foff)
        nu = (alpha + 4*sigma - beta)/(beta + 4*sigma - alpha)
        ns = int(min((1.96/(2*(1-nu)))**2, nmax))
        for j in range(ns):
            Ye.append(fun_cost(xe))
            Yoff.append(fun_cost(xoff))
        fe = np.mean(Ye)
        foff = np.mean(Yoff)
        if foff <= fe:
            winner = 1
        # print('More evaluation needed', winner, 'has won after', ns, 'evaluations')
        return [winner, Ye, Yoff, fe, foff]


def DE_Noisy(fun_cost, xmin, xmax, npop=40, maxiter=1000, maxfun=1e+6,
             convergence_tol=1e-5, ngen=100, crossover=0.9, MINMAX=1, x0=[], radius=0.2, sigma=0.0025, nmax=30):

    import openturns as ot

    dim = len(xmin)
    X = np.zeros([npop, dim])
    Y = [[]]*npop
    Y_mean = np.zeros(npop)
    fun_eval = 0

    # 1st Generation
    distribution = ot.ComposedDistribution([ot.Uniform(xmin[j], xmax[j]) for j in range(dim)])
    lhs = ot.LHSExperiment(distribution, npop)
    lhs.setAlwaysShuffle(True)
    sample = lhs.generate()
    for i in range(npop):
        X[i] = np.array(sample[i])
        Y[i] = [fun_cost(X[i]), fun_cost(X[i])]
        Y_mean[i] = np.mean(Y[i])
    fun_eval += npop

    # Next Generations
    k = 1
    argmin_id = Y_mean.argmin()
    Xoff = np.zeros([npop, dim])
    Yoff = [[]]*npop
    Yoff_mean = np.zeros(npop)
    fun_min = []

    while k <= maxiter:

        for i in range(npop):
            # Mutation
            id_rnd = np.random.choice(range(npop), 2, replace=False)
            scaling = 0.5 + 0.5*ot.RandomGenerator.Generate()
            Xoff[i] = X[argmin_id] + scaling*(X[id_rnd[0]] - X[id_rnd[1]])
            # Crossover
            for j in range(dim):
                if (ot.RandomGenerator.Generate() < crossover):
                    Xoff[i][j] = X[i][j]
                if (Xoff[i][j] < xmin[j]):
                    Xoff[i][j] = X[argmin_id][j] + ot.RandomGenerator.Generate()*(xmin[j] - X[argmin_id][j])
                elif (Xoff[i][j] > xmax[j]):
                    Xoff[i][j] = X[argmin_id][j] + ot.RandomGenerator.Generate()*(xmax[j] - X[argmin_id][j])

        for i in range(npop):
            # Noisy Selection
            Y[i].append(fun_cost(X[i]))
            Yoff[i] = [fun_cost(Xoff[i]), fun_cost(Xoff[i])]
            Yoff_mean[i] = np.mean(Yoff[i])
            if i == argmin_id:
                res_comp = compete(X[i], Xoff[i], Y[i], Yoff[i], fun_cost, sigma, nmax)
                if (res_comp[0] == 1):
                    X[i] = Xoff[i]
                    Y[i] = res_comp[2]
                    Y_mean[i] = res_comp[4]
                else:
                    Y[i] = res_comp[1]
                    Y_mean[i] = res_comp[3]
            elif (Yoff_mean[i] < Y_mean[i]):
                    X[i] = Xoff[i]
                    Y[i] = Yoff[i]
                    Y_mean[i] = Yoff_mean[i]

        fun_min.append(Y_mean.min())
        argmin_id = Y_mean.argmin()
        fun_argmin = X[argmin_id]
        fun_min_number = len(Y[argmin_id])
        fun_eval += npop

        if (k % 20 == 0):
            print('iteration', k, ': ', np.round(fun_min[k-1], 4), 'evaluated', fun_min_number, 'times at position', argmin_id)

        if (getCount(fun_min, fun_min[k-1] - convergence_tol, fun_min[k-1] + convergence_tol) >= ngen):
            print('Convergence criteria reached')
            return fun_argmin, fun_min, fun_eval

        if (fun_eval >= maxfun):
            print('Max number of function call reached')
            return fun_argmin, fun_min, fun_eval

        if (k == maxiter):
            print('Max number of population generation reached')
            return fun_argmin, fun_min, fun_eval
        k += 1


def DE_Simple(fun_cost, xmin, xmax, npop=40, maxiter=1000, maxfun=1e+6,
              convergence_tol=1e-5, ngen=100, scaling=0.8, crossover=0.9, MINMAX=1, x0=[], radius=0.2):

    import openturns as ot

    dim = len(xmin)
    X = np.zeros([npop, dim])
    Y = np.zeros(npop)
    fun_eval = 0

    # 1st Generation
    distribution = ot.ComposedDistribution([ot.Uniform(xmin[j], xmax[j]) for j in range(dim)])
    lhs = ot.LHSExperiment(distribution, npop)
    lhs.setAlwaysShuffle(True)
    design = lhs.generate()
    for i in range(npop):
        X[i] = np.array(design[i])
        Y[i] = fun_cost(X[i])
    fun_eval += npop

    # Next Generations
    k = 1
    argmin_id = Y.argmin()
    Xoff = np.zeros([npop, dim])
    Yoff = np.zeros(npop)
    fun_min = []
    while k <= maxiter:

        for i in range(npop):
            # Mutation
            id_rnd = np.random.choice(range(npop), 2, replace=False)
            Xoff[i] = X[argmin_id] + scaling*(X[id_rnd[0]] - X[id_rnd[1]])
            # Crossover
            for j in range(dim):
                if (ot.RandomGenerator.Generate() < crossover):
                    Xoff[i][j] = X[i][j]
                if (Xoff[i][j] < xmin[j]):
                    Xoff[i][j] = X[argmin_id][j] + ot.RandomGenerator.Generate()*(xmin[j] - X[argmin_id][j])
                elif (Xoff[i][j] > xmax[j]):
                    Xoff[i][j] = X[argmin_id][j] + ot.RandomGenerator.Generate()*(xmax[j] - X[argmin_id][j])

        for i in range(npop):
            # Selection
            Yoff[i] = fun_cost(Xoff[i])
            if (Yoff[i] < Y[i]):
                X[i] = Xoff[i]
                Y[i] = Yoff[i]

        fun_min.append(Y.min())
        argmin_id = Y.argmin()
        fun_argmin = X[argmin_id]
        fun_eval += npop

        if (getCount(fun_min, fun_min[k-1] - convergence_tol, fun_min[k-1] + convergence_tol) >= ngen):
            print('Convergence criteria reached')
            return fun_argmin, fun_min[k-1], fun_eval

        if (k % 20 == 0):
            print('iteration', k, ': ', fun_min[k-1])

        if (fun_eval >= maxfun):
            print('Max number of function call reached')
            return fun_argmin, fun_min[k-1], fun_eval

        k += 1
        if (k == maxiter):
            print('Max number of population generation reached')
            return fun_argmin, fun_min[k-2], fun_eval
