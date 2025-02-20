import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def direction(grad):
    grad_norm = np.linalg.norm(grad)
    return -grad/grad_norm

def rosenbrock_grad(x1, x2,a,b):
    grad = np.zeros(2)
    grad[0] = 2*(x1-a) - 4*b*x1*(x2-np.power(x1,2))
    grad[1] = 2*b*(x2-np.power(x1,2))
    return grad

def rosenbrock(x1, x2,a,b):
    return np.power((a-x1),2) + b*np.power((x2-np.power(x1,2)),2)

def gradient_descent(startpoint, grad_size_lim, max_iter, a, b):
    x = startpoint
    for i in range(max_iter):
        grad = rosenbrock_grad(x[0], x[1], a, b)
        if np.linalg.norm(grad) < grad_size_lim:
            break
        x = x + direction(grad)
    return x


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
    Zc = rosenbrock(X1c,X2c,1,5)
    
    #test
    # print(Zc.shape)
    # print(rosenbrock(1,1,1,5))
    # print(rosenbrock(-1,-1,1,5))
    # print("grads")
    # print(rosenbrock_grad(-1,-1,1,5))
    # print(rosenbrock_grad(1,1,1,5))
    print("direction")
    print(rosenbrock_grad(-1,-1,1,5))
    o = direction(rosenbrock_grad(-1,-1,1,5))
    i = np.linalg.norm(o)

    #print(o)
    #print(i)
    #gradient_descent(startpoint, grad_size_lim, max_iter, 1, 5)
    #print(gradient_descent(startpoint, grad_size_lim, max_iter, 1, 5))

    # Plot the surface.
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(startpoint[0], startpoint[1], rosenbrock(startpoint[0], startpoint[1],1,5), c='r', marker='o')
    ax.scatter(1, 1, rosenbrock(1, 1,1,5), c='g', marker='o')
    ax.contour3D(X1c, X2c, Zc,200, cmap='viridis')
   
    ax = fig.add_subplot(1, 2,2)
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')
    #cil
    ax.scatter(1, 1 ,c='g', marker='o')
    plt.show()