import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def rosenbrock(x1, x2):
    a = 1
    b = 5
    return np.power((a-x1),2) + b*np.power((x2-np.power(x1,2)),2)

def rosenbrock_alpha(alpha,startpoint,direction):
    a = 1
    b = 5
    x1 = startpoint[0]+alpha*direction[0]
    x2 = startpoint[1]+alpha*direction[1]

    return np.power((a-x1),2) + b*np.power((x2-np.power(x1,2)),2)

def cyclic_coordinate_search_accelerated(startpoint, step_size_lim, max_iter, max_fcalls):
    delta = np.finfo(np.float64).max
    x = startpoint.astype(float)
    n = np.size(x)
    it_num = 0
    f_calls = 0

    points = np.array([startpoint], dtype=np.float64)

    while np.fabs(delta) > step_size_lim and it_num < max_iter and f_calls < max_fcalls:
        x1 = x
        for i in range(n):
            dir = np.zeros(n)
            dir[i] = 1
            x, f_calls_new = line_search(x, dir, max_fcalls-f_calls)
            f_calls += f_calls_new
            
            points = np.append(points, [x], axis=0)

        x, f_calls_new = line_search(x, x-x1, max_fcalls-f_calls)
        f_calls += f_calls_new
        points = np.append(points, [x], axis=0)
        delta = np.linalg.norm(x-x1)
        it_num += 1
    
    return points, f_calls, it_num

def hookes_jeeves(startpoint, start_step_size, step_size_lim, step_decay, max_iter, max_fcalls):
    x = startpoint.astype(float)
    n = np.size(x)
    it_num = 0
    f_calls = 0
    points = np.array([startpoint], dtype=np.float64)

    y = rosenbrock(x[0], x[1])
    f_calls += 1

    while start_step_size > step_size_lim and it_num < max_iter and f_calls < max_fcalls:
        improved = False
        xbest, ybest = x, y

        for i in range(0,n):
            for sgn in (-1,1):
                xnew = x + sgn*start_step_size*np.eye(n)[i]
                ynew = rosenbrock(xnew[0], xnew[1])
                f_calls += 1

                if ynew < ybest:
                    xbest, ybest = xnew, ynew
                    improved = True
                
        x,y = xbest, ybest
        points = np.append(points, [x], axis=0)

        if not improved:
            start_step_size *= step_decay
        
        it_num += 1

    return points, f_calls, it_num


def line_search(startpoint,direction, max_fcalls):
    alpha, _ , _, calls = optimize.fminbound(rosenbrock_alpha, -2, 2, args=(startpoint,direction), full_output=True, maxfun=max_fcalls, disp=0)

    newpoint = startpoint + alpha*direction
    return newpoint, calls

def nelder_mead(startpoint, max_iter, max_fcalls):
    x = startpoint.astype(float)
    n = np.size(x)
    it_num = 0
    f_calls = 0
    points = np.array([startpoint], dtype=np.float64)

    return points , f_calls, it_num



if __name__ == '__main__':

    step_size_lim_ccs = 0.0001      # for cyclic coordinate search

    step_size_lim_hj = 0.000001     # for Hooke Jeeves
    start_step_size_hj = 1          # for Hooke Jeeves
    step_decay_hj  = 0.5            # for Hooke Jeeves

    max_iter = 100
    max_fcalls = 1000
    startpoint = np.array([-1,-1])
    global_min = (1,1)

    x1leftLim = -2
    x1rightLim = 2
    x1leftLim = -2
    x1rightLim = 2

    X1range = np.linspace(x1leftLim, x1rightLim, 200)
    X2range = np.linspace(x1leftLim, x1rightLim, 200)
    X1c, X2c = np.meshgrid(X1range, X2range)
    Zc = rosenbrock(X1c,X2c)

    print(f"\n\nGlobal minimum at (1,1) with value {rosenbrock(1,1)}")
    print(f"Starting point: {startpoint}")
    print(f"Max iterations: {max_iter}")
    print(f"Max function calls: {max_fcalls}")

    #Volani jednotlivych metod

    print("\nCyclic coordinate search with acceleration step and optimal stepsize: ")
    points_ccs, f_calls_ccs, it_num_ccs = cyclic_coordinate_search_accelerated(startpoint, step_size_lim_ccs, max_iter, max_fcalls)
    print(f"Converged after {it_num_ccs} iterations and {f_calls_ccs} function calls, distance to global minimum: {np.linalg.norm(points_ccs[-1]-[1,1])}")

    print("\nHooke-Jeeves method: ")
    points_hj, f_calls_hj, it_num_hj  = hookes_jeeves(startpoint, start_step_size_hj, step_size_lim_hj,step_decay_hj, max_iter, max_fcalls)
    print(f"Converged after {it_num_hj} iterations and {f_calls_hj} function calls, distance to global minimum: {np.linalg.norm(points_hj[-1]-[1,1])}")

    print("\nNelder-Mead method: ")
    points_nm,f_calls_nm, it_num_nm =  nelder_mead(startpoint, max_iter, max_fcalls)
    print(f"Converged after {it_num_nm} iterations and {f_calls_nm} function calls, distance to global minimum: {np.linalg.norm(points_nm[-1]-[1,1])}")

    # Plot the surface.
    fig = plt.figure(figsize=(18,8))
    #figure title

    fig.suptitle('Exercise 3 - Cyclic coordinate search, Hooke-Jeeves, Nelder-Mead and Quasi-newton methods')
    
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    

    #startpoint 3D
    ax.scatter(startpoint[0], startpoint[1], rosenbrock(startpoint[0], startpoint[1]), c='r', marker='o')
    #endpoint 3D
    ax.scatter(1, 1, rosenbrock(1, 1), c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.contour3D(X1c, X2c, Zc,200, cmap='viridis')

    #3D plot cesty
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], rosenbrock(points_ccs[1:,0], points_ccs[1:,1]), c='b', marker='o')
    ax.scatter(points_hj[1:,0], points_hj[1:,1], rosenbrock(points_hj[1:,0], points_hj[1:,1]), c='y', marker='o')
    #ax.scatter(points_cg[1:,0], points_cg[1:,1], rosenbrock(points_cg[1:,0], points_cg[1:,1]), c='orange', marker='o')
    
    
    
    #2D plot
    ax = fig.add_subplot(1, 2,2)
    ax.set_aspect('equal', 'box')
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')

    #2D plot cesty
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], c='b', marker='o')
    ax.scatter(points_hj[1:,0], points_hj[1:,1], c='y', marker='o')
    #ax.scatter(points_cg[1:,0], points_cg[1:,1], c='orange', marker='o')


    #2D plot lines between points
    for i in range(1,len(points_ccs)):
        ax.plot([points_ccs[i-1][0],points_ccs[i][0]],[points_ccs[i-1][1],points_ccs[i][1]],c='b')
    for i in range(1,len(points_hj)):
        ax.plot([points_hj[i-1][0],points_hj[i][0]],[points_hj[i-1][1],points_hj[i][1]],c='y')
    # for i in range(1,len(points_cg)):
    #     ax.plot([points_cg[i-1][0],points_cg[i][0]],[points_cg[i-1][1],points_cg[i][1]],c='orange')

    #cil
    ax.scatter(1, 1 ,c='g', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    #legend 
    ax.legend(['startpoint','Cyclic Coordinate Search with acceleration step', 'Hookes Jeeves method', 'Global minimum'])
    #plt.savefig('rosenbrock.png')

    plt.show()