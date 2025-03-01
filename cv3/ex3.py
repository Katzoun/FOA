import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def rosenbrock_grad(x1, x2):
    a = 1
    b = 5
    grad = np.zeros(2)
    grad[0] = 2*(x1-a) - 4*b*x1*(x2-np.power(x1,2))
    grad[1] = 2*b*(x2-np.power(x1,2))
    return grad

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

def nelder_mead(starting_simplex, step_size_lim_nm, alpha, beta, gamma ,max_iter, max_fcalls):
    it_num = 0
    f_calls = 0
    delta = np.finfo(np.float64).max
    yarr = np.zeros(3)
    simplex = starting_simplex
    simplex_points = np.array([starting_simplex], dtype=np.float64)

    for i in range(3):
        yarr[i] = rosenbrock(simplex[i][0], simplex[i][1])
        f_calls += 1
    
    while delta > step_size_lim_nm and it_num < max_iter and f_calls < max_fcalls:
        p = np.argsort(yarr)
        yarr = yarr[p]
        simplex = simplex[p]

        xl, yl = simplex[0], yarr[0] #lowest
        xh, yh = simplex[2], yarr[2] #highest
        xs, ys = simplex[1], yarr[1] #second highest

        xm = np.mean(simplex[:2], axis=0)
        xr = xm + alpha*(xm-xh) #reflection point
        yr = rosenbrock(xr[0], xr[1])
        f_calls += 1

        if yr < yl:
            xe = xm + beta*(xr-xm) #expansion point
            ye = rosenbrock(xe[0], xe[1])
            f_calls += 1
            
            simplex[2], yarr[2] = (xe, ye) if ye < yr else (xr, yr)

        elif yr >= ys:
            if yr < yh:
                xh, yh = xr, yr
                simplex[2], yarr[2] = xr, yr
            
            xc = xm + gamma*(xh-xm) #contraction point
            yc = rosenbrock(xc[0], xc[1])
            f_calls += 1

            if yc < yh:
                simplex[2], yarr[2] = xc, yc
            else:
                for i in range(1,3):
                    simplex[i] = 0.5*(simplex[i]+xl)
                    yarr[i] = rosenbrock(simplex[i][0], simplex[i][1])
                    f_calls += 1
        
        else:
            simplex[2], yarr[2] = xr, yr
        
        simplex_points = np.append(simplex_points, [simplex], axis=0)
        delta = np.std(yarr, correction=False)
        it_num += 1
    
    return simplex_points , f_calls, it_num


def quasi_newton_dfp(startpoint ,max_iter, max_fcalls, grad_lim_size):
    x = startpoint.astype(float)
    n = np.size(x)
    it_num = 0
    f_calls = 0
    points = np.array([startpoint], dtype=np.float64)

    Qm = np.eye(n)
    grad = rosenbrock_grad(x[0], x[1])
    grad_prev = grad
    f_calls += 1
    grad_size = np.linalg.norm(grad)
    x, f_calls_new = line_search(x, -np.matmul(Qm,grad), max_fcalls-f_calls)
    f_calls += f_calls_new

    points = np.append(points, [x], axis=0)

    while it_num < max_iter and f_calls < max_fcalls and grad_size > grad_lim_size:
            
        x, f_calls_new = line_search(x, -np.matmul(Qm,grad), max_fcalls-f_calls)
        f_calls += f_calls_new
        points = np.append(points, [x], axis=0)

        grad = rosenbrock_grad(x[0], x[1])
        f_calls += 1
        grad_size = np.linalg.norm(grad)

        gamma = grad - grad_prev
        delta = x - points[-2]
        Qm = Qm - np.outer(np.matmul(Qm, gamma), np.matmul(gamma,Qm))/np.dot(np.matmul(gamma,Qm),gamma) + np.outer(delta,delta)/np.dot(delta,gamma)
         
        grad_prev = grad

        it_num += 1

    return points , f_calls, it_num



if __name__ == '__main__':

    #Cyclic coordinate search specific params
    step_size_lim_ccs = 0.0001  

    #Hooke-Jeeves specific params
    step_size_lim_hj = 0.000001     
    start_step_size_hj = 1          
    step_decay_hj  = 0.5            

    #Nelder-Mead specific params
    starting_simplex = np.array([[-1,-1],[-0.8,-0.8],[-0.5,-1.0]])
    step_size_lim_nm = 0.000001 
    alpha = 1   #reflection param
    beta = 2  #expansion param
    gamma = 0.5   #contraction param

    #Quasi-Newton DFP specific params
    grad_lim_size = 0.01

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
    points_nm,f_calls_nm, it_num_nm =  nelder_mead(starting_simplex, step_size_lim_nm, alpha, beta, gamma ,max_iter, max_fcalls)
    print(f"Converged after {it_num_nm} iterations and {f_calls_nm} function calls, distance to global minimum: {np.linalg.norm(points_nm[-1][0]-[1,1])}")

    print("\nQuasi-Newton DFP method: ")
    points_dfp, f_calls_dfp, it_num_dfp = quasi_newton_dfp(startpoint, max_iter, max_fcalls, grad_lim_size)
    print(f"Converged after {it_num_dfp} iterations and {f_calls_dfp} function calls, distance to global minimum: {np.linalg.norm(points_dfp[-1]-[1,1])}")

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

    #3D plot 
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], rosenbrock(points_ccs[1:,0], points_ccs[1:,1]), c='b', marker='o')
    ax.scatter(points_hj[1:,0], points_hj[1:,1], rosenbrock(points_hj[1:,0], points_hj[1:,1]), c='y', marker='o')
    ax.scatter(points_nm[1:,0], points_nm[1:,1], rosenbrock(points_nm[1:,0], points_nm[1:,1]), c='orange', marker='o')
    ax.scatter(points_dfp[1:,0], points_dfp[1:,1], rosenbrock(points_dfp[1:,0], points_dfp[1:,1]), c='purple', marker='o')
    
    #print(points_nm)
    #print(points_nm.shape)
    #print(points_nm[-1][0])

    #2D plot
    ax = fig.add_subplot(1, 2,2)
    ax.set_aspect('equal', 'box')
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')

    #2D plot
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], c='b', marker='o')
    ax.scatter(points_hj[1:,0], points_hj[1:,1], c='y', marker='o')

    ax.scatter(points_nm[0:,0:,0], points_nm[0:,0:,1], c='orange', marker='o')
    ax.scatter(points_dfp[1:,0], points_dfp[1:,1], c='purple', marker='o')

    #2D plot lines between points
    for i in range(1,len(points_ccs)):
        ax.plot([points_ccs[i-1][0],points_ccs[i][0]],[points_ccs[i-1][1],points_ccs[i][1]],c='b')
    for i in range(1,len(points_hj)):
        ax.plot([points_hj[i-1][0],points_hj[i][0]],[points_hj[i-1][1],points_hj[i][1]],c='y')
    for i in range(0,len(points_nm)):
          for j in range(1,3):
            ax.plot([points_nm[i][j-1][0],points_nm[i][j][0]],[points_nm[i][j-1][1],points_nm[i][j][1]],c='orange')
    for i in range(1,len(points_dfp)):
        ax.plot([points_dfp[i-1][0],points_dfp[i][0]],[points_dfp[i-1][1],points_dfp[i][1]],c='purple')       
    #cil
    ax.scatter(1, 1 ,c='g', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    #legend 
    ax.legend(['startpoint','Cyclic Coordinate Search with acceleration step', 'Hookes Jeeves method', 'Nelder-Mead method', 'Quasi-Newton DFP method'])
    #plt.savefig('rosenbrock.png')

    plt.show()