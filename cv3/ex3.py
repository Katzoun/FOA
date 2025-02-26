import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

def direction(grad):
    grad_norm = np.linalg.norm(grad)
    return -grad/grad_norm

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

    return points, f_calls, it_num




def line_search(startpoint,direction, max_fcalls):
    alpha, _ , _, calls = optimize.fminbound(rosenbrock_alpha, -2, 2, args=(startpoint,direction), full_output=True, maxfun=max_fcalls, disp=0)
    #print(alpha)
    
    newpoint = startpoint + alpha*direction
    return newpoint, calls


# def gradient_descent(startpoint, grad_size_lim, max_iter, search_method):
#     x = startpoint.astype(float)
    
#     points = np.empty((0, 2))
#     directions = np.empty((0, 2))
#     points = np.append(points, [startpoint], axis=0)

#     for i in range(max_iter):
#         grad = rosenbrock_grad(x[0], x[1])
#         if np.linalg.norm(grad) < grad_size_lim:
#             if search_method == "optimal":
#                 print(f"Gradient descent with optimal step converged (grad < {grad_size_lim}) at iteration {i}, distance to global minimum: {np.linalg.norm(x-[1,1])}")
#             elif search_method == "decaying":
#                 print(f"Gradient descent with decaying step converged (grad < {grad_size_lim}) at iteration {i}, distance to global minimum: {np.linalg.norm(x-[1,1])}")
#             break
        
#         elif i == max_iter-1:
#             if search_method == "optimal":
#                 print(f"Gradient descent with optimal step did not converge after {max_iter} iterations, distance to global minimum: {np.linalg.norm(x-[1,1])}")
#             elif search_method == "decaying":
#                 print(f"Gradient descent with decaying step did not converge after {max_iter} iterations, distance to global minimum: {np.linalg.norm(x-[1,1])}")

#         dir = direction(grad)

#         if search_method == "optimal":
#             x[0],x[1] = line_search([x[0],x[1]], dir)
#         elif search_method == "decaying":
#             x[0], x[1] = line_search_conjugate([x[0],x[1]], dir,i)
#         else:
#             print("Invalid search method")
#             raise ValueError
        
#         points = np.append(points, [x], axis=0)
#         directions = np.append(directions, [dir], axis=0)

#     return points, directions

if __name__ == '__main__':

    step_size_lim = 0.0001 
    start_step_size = 1 # for Hooke Jeeves
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
    print(f"Step size limit: {step_size_lim}")
    print(f"Max iterations: {max_iter}")
    print(f"Max function calls: {max_iter}")

    print("\nCyclic coordinate search with acceleration step and optimal stepsize: ")
    points_ccs, f_calls_ccs, it_num_ccs = cyclic_coordinate_search_accelerated(startpoint, step_size_lim, max_iter, max_fcalls)
    print(f"Converged after {it_num_ccs} iterations and {f_calls_ccs} function calls, distance to global minimum: {np.linalg.norm(points_ccs[-1]-[1,1])}")

    print("\nHooke Jeeves method: ")
    points_hj, f_calls_hj, it_num_hj  = hookes_jeeves(startpoint, start_step_size, step_size_lim, max_iter, "decaying")
    print(f"Converged after {it_num_ccs} iterations and {f_calls_ccs} function calls, distance to global minimum: {np.linalg.norm(points_ccs[-1]-[1,1])}")

    # print("\nConjugate gradient method with Polak-Ribiere equation and optimal stepsize: ")
    # points_cg, directions_cg = gradient_descent_conjugate(startpoint, grad_size_lim, max_iter)

    # Plot the surface.
    fig = plt.figure(figsize=(18,8))
    #figure title

    fig.suptitle('Methods')
    
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    

    #startpoint 3D
    ax.scatter(startpoint[0], startpoint[1], rosenbrock(startpoint[0], startpoint[1]), c='r', marker='o')
    #endpoint 3D
    ax.scatter(1, 1, rosenbrock(1, 1), c='g', marker='o')
    
    
    #3D plot cesty
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], rosenbrock(points_ccs[1:,0], points_ccs[1:,1]), c='b', marker='o')
    #gradient descent decaying
    #ax.scatter(points_gd_dec[1:,0], points_gd_dec[1:,1], rosenbrock(points_gd_dec[1:,0], points_gd_dec[1:,1]), c='y', marker='o')
    #conjugate descent
    #ax.scatter(points_cg[1:,0], points_cg[1:,1], rosenbrock(points_cg[1:,0], points_cg[1:,1]), c='orange', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


    ax.contour3D(X1c, X2c, Zc,200, cmap='viridis')

    ax = fig.add_subplot(1, 2,2)
    ax.set_aspect('equal', 'box')
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')

    #2D plot cesty
    ax.scatter(points_ccs[1:,0], points_ccs[1:,1], c='b', marker='o')
    #ax.scatter(points_gd_dec[1:,0], points_gd_dec[1:,1], c='y', marker='o')
    #ax.scatter(points_cg[1:,0], points_cg[1:,1], c='orange', marker='o')


    #2D plot lines between points
    for i in range(1,len(points_ccs)):
        ax.plot([points_ccs[i-1][0],points_ccs[i][0]],[points_ccs[i-1][1],points_ccs[i][1]],c='b')
    # for i in range(1,len(points_gd_dec)):
    #     ax.plot([points_gd_dec[i-1][0],points_gd_dec[i][0]],[points_gd_dec[i-1][1],points_gd_dec[i][1]],c='y')
    # for i in range(1,len(points_cg)):
    #     ax.plot([points_cg[i-1][0],points_cg[i][0]],[points_cg[i-1][1],points_cg[i][1]],c='orange')

    #cil
    ax.scatter(1, 1 ,c='g', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    #legend 
    ax.legend(['startpoint','Cyclic Coordinate Search with acceleration step', 'Global minimum'])
    #plt.savefig('rosenbrock.png')

    plt.show()