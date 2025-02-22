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

def line_search(startpoint,direction):
    alpha = optimize.fminbound(rosenbrock_alpha, -2, 2, args=(startpoint,direction))
    #print(alpha)
   
    newpoint = startpoint + alpha*direction
    return newpoint

def line_search_conjugate(startpoint,direction,k):
    alpha = np.power(0.9,k)
    #print(alpha)
   
    newpoint = startpoint + alpha*direction
    return newpoint

def gradient_descent(startpoint, grad_size_lim, max_iter, search_method):
    x = startpoint.astype(float)
    
    points = np.empty((0, 2))
    directions = np.empty((0, 2))
    points = np.append(points, [startpoint], axis=0)

    for i in range(max_iter):
        grad = rosenbrock_grad(x[0], x[1])
        if np.linalg.norm(grad) < grad_size_lim:
            if search_method == "optimal":
                print(f"Gradient descent with optimal step converged (grad < {grad_size_lim}) at iteration {i}, distance to global minimum: {np.linalg.norm(x-[1,1])}")
            elif search_method == "decaying":
                print(f"Gradient descent with decaying step converged (grad < {grad_size_lim}) at iteration {i}, distance to global minimum: {np.linalg.norm(x-[1,1])}")
            break
        
        elif i == max_iter-1:
            if search_method == "optimal":
                print(f"Gradient descent with optimal step did not converge after {max_iter} iterations, distance to global minimum: {np.linalg.norm(x-[1,1])}")
            elif search_method == "decaying":
                print(f"Gradient descent with decaying step did not converge after {max_iter} iterations, distance to global minimum: {np.linalg.norm(x-[1,1])}")

        dir = direction(grad)

        if search_method == "optimal":
            x[0],x[1] = line_search([x[0],x[1]], dir)
        elif search_method == "decaying":
            x[0], x[1] = line_search_conjugate([x[0],x[1]], dir,i)
        else:
            print("Invalid search method")
            raise ValueError
        
        points = np.append(points, [x], axis=0)
        directions = np.append(directions, [dir], axis=0)

    return points, directions

def gradient_descent_conjugate(startpoint, grad_size_lim, max_iter):
    x = startpoint.astype(float)
    points = np.empty((0, 2))
    directions = np.empty((0, 2))
    points = np.append(points, [startpoint], axis=0)
    prev_grad = np.zeros(2) #previous gradient

    for i in range(max_iter):
        grad = rosenbrock_grad(x[0], x[1])

        if np.linalg.norm(grad) < grad_size_lim:
            print(f"Conjugate gradient descent with optimal step converged (grad < {grad_size_lim}) at iteration {i}, distance to global minimum: {np.linalg.norm(x-[1,1])}")
            break
        elif i == max_iter-1:
            print(f"Conjugate gradient descent with optimal step did not converge after {max_iter} iterations, distance to global minimum: {np.linalg.norm(x-[1,1])}")

        if(i == 0):
            dir = direction(grad)
        else:
            #Polak Ribiere equation for beta
            beta = np.dot(grad,grad-prev_grad)/np.dot(prev_grad,prev_grad)
            beta = max(0,beta)

            dir = direction(grad) + beta*directions[i-1]

        x[0], x[1] = line_search([x[0],x[1]], dir)
        points = np.append(points, [x], axis=0)
        directions = np.append(directions, [dir], axis=0)
        prev_grad = grad

    return points, directions




if __name__ == '__main__':

    grad_size_lim = 0.01
    max_iter = 100
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
    print(f"Gradient size limit: {grad_size_lim}")
    print(f"Max iterations: {max_iter}")
    print("\nGradient descent methods: ")
    points_gd_opt, directions_gd_opt = gradient_descent(startpoint, grad_size_lim, max_iter, "optimal")
    points_gd_dec, directions_gd_dec = gradient_descent(startpoint, grad_size_lim, max_iter, "decaying")

    print("\nConjugate gradient method with Polak-Ribiere equation and optimal stepsize: ")
    points_cg, directions_cg = gradient_descent_conjugate(startpoint, grad_size_lim, max_iter)

    # Plot the surface.
    fig = plt.figure(figsize=(18,8))
    #figure title

    fig.suptitle('Gradient Descent and Conjugate Gradient Descent on Rosenbrock Function')
    
    ax = fig.add_subplot(1, 2, 1,projection='3d')

    #startpoint 3D
    ax.scatter(startpoint[0], startpoint[1], rosenbrock(startpoint[0], startpoint[1]), c='r', marker='o')
    #endpoint 3D
    ax.scatter(1, 1, rosenbrock(1, 1), c='g', marker='o')
    #gradient descent optimal
    ax.scatter(points_gd_opt[1:,0], points_gd_opt[1:,1], rosenbrock(points_gd_opt[1:,0], points_gd_opt[1:,1]), c='b', marker='o')
    #gradient descent decaying
    ax.scatter(points_gd_dec[1:,0], points_gd_dec[1:,1], rosenbrock(points_gd_dec[1:,0], points_gd_dec[1:,1]), c='y', marker='o')
    #conjugate descent
    ax.scatter(points_cg[1:,0], points_cg[1:,1], rosenbrock(points_cg[1:,0], points_cg[1:,1]), c='orange', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    ax.contour3D(X1c, X2c, Zc,200, cmap='viridis')
    
    ax = fig.add_subplot(1, 2,2)
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')


    #gradient descent optimal
    ax.scatter(points_gd_opt[1:,0], points_gd_opt[1:,1], c='b', marker='o')
    #gradient descent decaying
    ax.scatter(points_gd_dec[1:,0], points_gd_dec[1:,1], c='y', marker='o')
    #conjugate gradient
    ax.scatter(points_cg[1:,0], points_cg[1:,1], c='orange', marker='o')

    #plot lines between points
    for i in range(1,len(points_gd_opt)):
        ax.plot([points_gd_opt[i-1][0],points_gd_opt[i][0]],[points_gd_opt[i-1][1],points_gd_opt[i][1]],c='b')
    for i in range(1,len(points_gd_dec)):
        ax.plot([points_gd_dec[i-1][0],points_gd_dec[i][0]],[points_gd_dec[i-1][1],points_gd_dec[i][1]],c='y')
    for i in range(1,len(points_cg)):
        ax.plot([points_cg[i-1][0],points_cg[i][0]],[points_cg[i-1][1],points_cg[i][1]],c='orange')

    #cil
    ax.scatter(1, 1 ,c='g', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    #legend 
    ax.legend(['startpoint','Gradient descent with optimal step','Gradient descent with decaying step','Conjugate gradient with optimal step'])
    plt.savefig('rosenbrock.png')

    plt.show()