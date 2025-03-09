import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    x = np.array(x)  # Ensure x is a NumPy array
    sum1 = np.sum(x**2, axis=0)
    sum2 = np.sum(np.cos(c * x), axis=0)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    return term1 + term2 + a + np.exp(1)

def rand_positive_spanning_set(alpha, n):
    delta = round(1/np.sqrt(alpha))
    Lmatrix = np.diag(delta*np.random.choice([1.0,-1.0],n))
    for i in range(1,n):
        for j in range(i):
            Lmatrix[i][j] = np.random.randint(-delta+1,delta)

    Dmatrix = Lmatrix[np.random.permutation(n), :]
    Dmatrix = Dmatrix[:, np.random.permutation(n)]
    Dmatrix = np.hstack((Dmatrix, -np.sum(Dmatrix, axis=1, keepdims=True)))
    dirVect = [Dmatrix[:, i] for i in range(n + 1)]
    return dirVect

def mesh_adaptive_direct_search(f, x, step_size_lim_mad, max_iter, max_fcalls):
    alpha = 1
    y = f(x)
    n = len(x)
    fcalls = 1
    it_num = 0
    points = np.array(x)

    while alpha > step_size_lim_mad and fcalls < max_fcalls and it_num < max_iter:
        improved = False
        dirVect = rand_positive_spanning_set(alpha, n)
        for d in dirVect:
            d = d.reshape(-1, 1)
            x_new = x + alpha * d
            y_new = f(x_new)
            fcalls += 1
            if y_new < y:
                x = x_new
                y = y_new
                improved = True

                x_new = x + 3 * alpha * d
                y_new = f(x_new)
                fcalls += 1

                if y_new < y:
                    x = x_new
                    y = y_new
                break
            
        alpha = min(4 * alpha, 1) if improved else alpha / 4
        it_num += 1
        points = np.append(points, x, axis=1)
    
    return points, fcalls, it_num

def cross_entropy_method(f, x, m, melite, max_iter, max_fcalls):
    n = len(x)
    fcalls = 0
    it_num = 0
    
    mu = np.array(x).reshape(-1)
    mu_points = np.array(x)
    #mu_points = np.empty((n,0))
    sigma = np.eye(n) * 10.0
    PropDist = stats.multivariate_normal(mu, sigma)
    elite_points = np.empty((n, 0))

    while fcalls < max_fcalls and it_num < max_iter:
        samples = PropDist.rvs(m)
        values = np.array([f(sample) for sample in samples])
        fcalls += m

        elite_indices = values.argsort()[:melite]
        elite = samples[elite_indices]

        mu = (1/melite)*sum(elite)

        sigma = np.zeros((n, n))
        for sample in elite:
            diff = sample - mu
            outer = np.outer(diff, diff)
            sigma += outer
        
        sigma = sigma / melite

        #alternative way to calculate sigma
        #sigma = (1/melite)*sum(np.outer(sample - mu, sample - mu) for sample in elite)
        
        PropDist = stats.multivariate_normal(mu, sigma)

        it_num += 1
        elite_points = np.append(elite_points, elite.T, axis=1)
        mu_points = np.append(mu_points, mu.reshape(-1, 1), axis=1)
    
    return elite_points,mu_points, fcalls, it_num

def differential_evolution(f, x, pop_size, p, w, max_iter, max_fcalls):
    n = len(x)
    fcalls = 0
    it_num = 0
    points = np.zeros((n, 0))
    pop = np.random.uniform(x - 2, x + 2, (n, pop_size))
    points = np.append(points, pop, axis=1)

    while fcalls < max_fcalls and it_num < max_iter:
        
        for k in range(pop_size):
            
            idxs = np.random.choice(pop_size_de, 3, replace=False) #replace zakaze opakovani indexu
            a, b, c = pop[:, idxs[0]], pop[:, idxs[1]], pop[:, idxs[2]] #vyber 3 nahodne jedince
            
            z = a + w * (b - c) #w je diferencni vaha a p je pravdepodobnost krizeni
            j = np.random.randint(n) #volba nahodne dimenze
            
            x_new = np.array([z[i] if i == j or np.random.rand() < p else pop[i, k] for i in range(n)])
            
            if f(x_new) < f(pop[:, k]):
                pop[:, k] = x_new
            fcalls += 2

        it_num += 1
        points = np.append(points, pop, axis=1)

    # pop_min = np.argmin([f(pop[:, i]) for i in range(pop_size)]) #debug

    return points, fcalls, it_num

def particke_swarm_optimization(f, x, pop_size, w, c1, c2, max_iter, max_fcalls):
    n = len(x)
    fcalls = 0
    it_num = 0
    points = np.zeros((n, 0))
    popul = np.random.uniform(x - 2, x + 2, (n, pop_size))
    points = np.append(points, popul, axis=1)
    
    x_best = np.copy(popul)
    y_best = [float('inf')] * pop_size
    global_best_x = None
    global_best_y = float('inf')

    popul_v = np.zeros((n, pop_size))

    for i in range(pop_size):
        y = f(popul[:, i])
        fcalls += 1
        if y < y_best[i]:
            x_best[:,i] = popul[:, i]
            y_best[i] = y

        if y < global_best_y:
            global_best_x = popul[:, i]
            global_best_y = y

    while fcalls < max_fcalls and it_num < max_iter:
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            popul_v[:, i] = w * popul_v[:, i] + c1 * r1 * (x_best[:, i] - popul[:,i]) + c2 * r2 * (global_best_x - popul[:,i])
            popul[:, i] += popul_v[:, i]

            y = f(popul[:, i])
            fcalls += 1

            if y < y_best[i]:
                x_best[:,i] = np.copy(popul[:, i])
                y_best[i] = np.copy(y)
            if y < global_best_y:
                global_best_x = np.copy(popul[:, i])
                global_best_y = np.copy(y)

        it_num += 1
        points = np.append(points, x_best, axis=1)

    return points, fcalls, it_num



if __name__ == '__main__':

    #np.random.seed(1138) #seed pro debugovani 

    #Mesh adaptive direct search specific params
    step_size_lim_mads = 1e-6

    #Cross-entropy method specific params
    m = 40
    melite = 10

    #Differential evolution specific params
    pop_size_de = 10
    p_de = 0.9
    w_de = 0.8

    #Particle swarm optimization specific params
    pop_size_pso = 10
    w_pso = 0.9
    c1_pso = 1.2
    c2_pso = 1.2

    max_iter = 100
    max_fcalls = 1000
    startpoint = np.array([[-6.0],[-4.5]])

    x1leftLim = -8
    x1rightLim = 8
    x1leftLim = -8
    x1rightLim = 8

    X1range = np.linspace(x1leftLim, x1rightLim, 200)
    X2range = np.linspace(x1leftLim, x1rightLim, 200)
    x = np.meshgrid(X1range, X2range)
    Zc = ackley(x)

    print(f"\n\nGlobal minimum at (0,0) with value {ackley([[0],[0]])}")
    print(f"Starting point: {startpoint[0][0], startpoint[1][0]}")
    print(f"Max iterations: {max_iter}")
    print(f"Max function calls: {max_fcalls}")

    #Volani jednotlivych metod

    print("\nMesh adaptive direct search: ")
    points_mads, f_calls_mads, it_num_mads = mesh_adaptive_direct_search(ackley, startpoint, step_size_lim_mads, max_iter, max_fcalls)
    print(f"Converged after {it_num_mads} iterations and {f_calls_mads} function calls, distance to global minimum: {np.linalg.norm(points_mads[:,-1]-[0,0])}")

    print("\nCross entropy method: ")
    points_cem_elite, points_cem, f_calls_cem, it_num_cem  = cross_entropy_method(ackley, startpoint, m, melite, max_iter, max_fcalls)
    print(f"Converged after {it_num_cem} iterations and {f_calls_cem} function calls, distance to global minimum: {np.linalg.norm(points_cem[:,-1]-[0,0])}")

    print("\nDifferential evolution: ")
    points_de,f_calls_de, it_num_de =  differential_evolution(ackley, startpoint, pop_size_de, p_de, w_de ,max_iter, max_fcalls)
    # Calculate mean from last population
    mean_last_pop_de = np.mean(points_de[:, -pop_size_de:], axis=1)
    print(f"Converged after {it_num_de} iterations and {f_calls_de} function calls, distance to global minimum: {np.linalg.norm(mean_last_pop_de-[0,0])}")

    print("\nParticle swarm optimization: ")
    points_pso, f_calls_pso, it_num_pso = particke_swarm_optimization(ackley, startpoint, pop_size_pso, w_pso, c1_pso, c2_pso, max_iter, max_fcalls)
    # Calculate mean from last population
    mean_last_pop_pso = np.mean(points_pso[:, -pop_size_pso:], axis=1)
    print(f"Converged after {it_num_pso} iterations and {f_calls_pso} function calls, distance to global minimum: {np.linalg.norm(mean_last_pop_pso-[0,0])}")

    # Plot the surface.

    fig, axs = plt.subplots(2, 2, figsize=(18, 8) )
    fig.delaxes(axs[0,0])
    fig.tight_layout()
    fig.suptitle('Exercise 4 - Stochastic and population methods')

    # first plot -------------------------------------------------------------------
    ax = fig.add_subplot(2, 2, 1, projection='3d',)
    axs[0, 0] = ax
    
    ax.scatter(startpoint[0], startpoint[1], ackley([startpoint[0], startpoint[1]]), c='r', marker='o')
    #endpoint 3D
    ax.scatter(0, 0, ackley([[0],[0]]), c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.contour3D(x[0], x[1], Zc,60, cmap='viridis')

    #3D plot 
    ax.plot(points_mads[0], points_mads[1], ackley(points_mads), c='b', marker='o')
    ax.plot(points_cem[0], points_cem[1], ackley(points_cem), c='y', marker='o')
    
    #plot moving batches of proposal distrubution
    # for i in range(0,points_cem_elite.shape[1],10):
    #     ax.scatter(points_cem_elite[0,i:i+10], points_cem_elite[1,i:i+10], ackley(points_cem_elite[:,i:i+10]), c='r', marker='o')
    
    #second plot (2D) -------------------------------------------------------------------
    ax = axs[0, 1]
    ax.set_aspect('equal')
    ax.contour(x[0], x[1], Zc, 20)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')

    # #2D plot
    ax.plot(points_mads[0], points_mads[1], c='b', marker='o')
    ax.plot(points_cem[0], points_cem[1], c='y', marker='o')
    
    #print batches of 10 elite points from cross entropy method
    # for i in range(0,points_cem_elite.shape[1],10):
    #     ax.scatter(points_cem_elite[0,i:i+10], points_cem_elite[1,i:i+10], c='r', marker='o')
    ax.scatter(0, 0 ,c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    #legend 
    ax.legend(['startpoint','Mesh adaptive direct search', 'Cross entropy method', 'Global minimum'])

    
    # Third subplot -------------------------------------------------------------------
    ax = axs[1, 0]
    ax.set_aspect('equal')
    ax.contour(x[0], x[1], Zc, 20)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')
    ax.scatter(0, 0, c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)

    for i in range(0,points_de.shape[1],pop_size_de):
        ax.scatter(points_de[0,i:i+pop_size_de], points_de[1,i:i+pop_size_de], c="b", marker='.')
    

    ax.legend(['startpoint', 'Global minimum', 'Differential evolution'])

    # Fourth subplot -------------------------------------------------------------------
    ax = axs[1, 1]
    ax.set_aspect('equal')
    ax.contour(x[0], x[1], Zc, 20)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')
    ax.scatter(0, 0, c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)

    for i in range(0,points_pso.shape[1],pop_size_pso):
        ax.scatter(points_pso[0,i:i+pop_size_pso], points_pso[1,i:i+pop_size_pso], c="b", marker='.')

    ax.legend(['startpoint', 'Global minimum', 'Particle swarm optimization'])

    plt.show()
