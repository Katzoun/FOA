import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



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
    #diagonal matrix
    Lmatrix = np.diag(delta*np.random.choice([1,-1],n))
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

 


if __name__ == '__main__':

    #np.random.seed(158)

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
    grad_lim_size = 0.01
    pop_size_pso = 10
    w_pso = 0.9
    c1 = 1.2
    c2 = 1.2

    max_iter = 100
    max_fcalls = 1000
    startpoint = np.array([[-6],[-4.5]])

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

    # print("\nHooke-Jeeves method: ")
    # points_hj, f_calls_hj, it_num_hj  = hookes_jeeves(startpoint, start_step_size_hj, step_size_lim_hj,step_decay_hj, max_iter, max_fcalls)
    # print(f"Converged after {it_num_hj} iterations and {f_calls_hj} function calls, distance to global minimum: {np.linalg.norm(points_hj[-1]-[1,1])}")

    # print("\nNelder-Mead method: ")
    # points_nm,f_calls_nm, it_num_nm =  nelder_mead(starting_simplex, step_size_lim_nm, alpha, beta, gamma ,max_iter, max_fcalls)
    # print(f"Converged after {it_num_nm} iterations and {f_calls_nm} function calls, distance to global minimum: {np.linalg.norm(points_nm[-1][0]-[1,1])}")

    # print("\nQuasi-Newton DFP method: ")
    # points_dfp, f_calls_dfp, it_num_dfp = quasi_newton_dfp(startpoint, max_iter, max_fcalls, grad_lim_size)
    # print(f"Converged after {it_num_dfp} iterations and {f_calls_dfp} function calls, distance to global minimum: {np.linalg.norm(points_dfp[-1]-[1,1])}")

    # Plot the surface.
    fig = plt.figure(figsize=(18,8))
    #figure title

    fig.suptitle('Exercise 4 - Stochastic and population methods')
    
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    

    #startpoint 3D
    ax.scatter(startpoint[0], startpoint[1], ackley([startpoint[0], startpoint[1]]), c='r', marker='o')
    #endpoint 3D
    ax.scatter(0, 0, ackley([[0],[0]]), c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.contour3D(x[0], x[1], Zc,100, cmap='viridis')

    #3D plot 
    ax.plot(points_mads[0], points_mads[1], ackley(points_mads), c='b', marker='o')
    # ax.scatter(points_hj[1:,0], points_hj[1:,1], rosenbrock(points_hj[1:,0], points_hj[1:,1]), c='y', marker='o')
    # ax.scatter(points_nm[1:,0], points_nm[1:,1], rosenbrock(points_nm[1:,0], points_nm[1:,1]), c='orange', marker='o')
    # ax.scatter(points_dfp[1:,0], points_dfp[1:,1], rosenbrock(points_dfp[1:,0], points_dfp[1:,1]), c='purple', marker='o')
    
    #print(points_nm)
    #print(points_nm.shape)
    #print(points_nm[-1][0])

    #2D plot
    ax = fig.add_subplot(1, 2,2)
    ax.set_aspect('equal', 'box')
    ax.contour(x[0], x[1], Zc, 100)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')

    # #2D plot
    ax.plot(points_mads[0], points_mads[1], c='b', marker='o')
    # ax.scatter(points_hj[1:,0], points_hj[1:,1], c='y', marker='o')

    # ax.scatter(points_nm[0:,0:,0], points_nm[0:,0:,1], c='orange', marker='o')
    # ax.scatter(points_dfp[1:,0], points_dfp[1:,1], c='purple', marker='o')

    #2D plot lines between points
    # for i in range(1,len(points_ccs)):
    #     ax.plot([points_ccs[i-1][0],points_ccs[i][0]],[points_ccs[i-1][1],points_ccs[i][1]],c='b')
    # for i in range(1,len(points_hj)):
    #     ax.plot([points_hj[i-1][0],points_hj[i][0]],[points_hj[i-1][1],points_hj[i][1]],c='y')
    # for i in range(0,len(points_nm)):
    #       for j in range(1,3):
    #         ax.plot([points_nm[i][j-1][0],points_nm[i][j][0]],[points_nm[i][j-1][1],points_nm[i][j][1]],c='orange')
    # for i in range(1,len(points_dfp)):
    #     ax.plot([points_dfp[i-1][0],points_dfp[i][0]],[points_dfp[i-1][1],points_dfp[i][1]],c='purple')       
    #cil
    ax.scatter(0, 0 ,c='g', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    #legend 
    ax.legend(['startpoint','Cyclic Coordinate Search with acceleration step', 'Hookes Jeeves method', 'Nelder-Mead method', 'Quasi-Newton DFP method'])
    #plt.savefig('./cv3/rosenbrock.png')
    plt.show()
