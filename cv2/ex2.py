import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def line_search(startpoint,direction):
    alpha = np.arange(-2,2,0.01)
    ylinevals = rosenbrock(startpoint[0]+alpha*direction[0],startpoint[1]+alpha*direction[1])
    a, _ , idxa, _= GoldenSectionSearch(alpha,ylinevals,50)
    #print(a)
    newpoint = startpoint + a*direction
    #print(newpoint)
    
    # fig1 = plt.figure()
    # plt.plot(alpha,ylinevals)
    # plt.show()

    return newpoint

def GoldenSectionSearch(xf,f,n): #removed a,b params
    
    idxa = 0
    #idxa = np.argmin(np.abs(xf - a))
    idxb = len(f)-1
    #idxb = np.argmin(np.abs(xf - b))
    ya = f[idxa]
    yb = f[idxb]
    a = xf[idxa]
    b = xf[idxb]

    phi = (1+np.sqrt(5))/2
    rho = phi -1 
    d = rho*b+(1-rho)*a
    # zjisteni, kteremu indexu odpovida d
    idxd = np.argmin(np.abs(xf - d))
    yd = f[idxd]
    d = xf[idxd]

    #print("\nGolden Section Search Algorithm")
    #print(f"Iterace {0:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  d={xf[idxd]:1.2f}")
    
    for i in range(1,n):
        c = rho*xf[idxa]+(1-rho)*xf[idxb]
        idxc = np.argmin(np.abs(xf - c))
        c = xf[idxc]
        yc = f[idxc]

        #print(f"Iterace {i:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  c={xf[idxc]:1.2f}  d={xf[idxd]:1.2f}")

        if yc < yd:
            idxb = idxd
            idxd = idxc
            b = xf[idxb]
            a = xf[idxa]
            d = xf[idxd]
            yd = yc
        else:
            idxa = idxb
            idxb = idxc
            b = xf[idxb]
            a = xf[idxa]
            d = xf[idxd]
    
    if a < b:
        return (a,b, idxa, idxb)
    else:
        return (b,a, idxb, idxa)


def gradient_descent(startpoint, grad_size_lim, max_iter):
    x = startpoint.astype(float)
    points = np.empty((0, 2))
    #  o = direction(rosenbrock_grad(-1,-1))
    # i = np.linalg.norm(o)
    # new_point = line_search(startpoint,o)
    
    # points = np.append(points, [new_point], axis=0)
    
    # o = direction(rosenbrock_grad(new_point[0],new_point[1]))
    # new_point = line_search(new_point,o)
    # points = np.append(points, [new_point], axis=0)
    # print(points)
    for i in range(max_iter):
        grad = rosenbrock_grad(x[0], x[1])
        if np.linalg.norm(grad) < grad_size_lim:
            break
        x[0], x[1] = line_search([x[0],x[1]], direction(grad))
        points = np.append(points, [x], axis=0)
        #print(x)
    return points


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

    #print(points.shape)
    #print(points)
    #debug
    # print(Zc.shape)
    # print(rosenbrock(1,1,1,5))
    # print(rosenbrock(-1,-1,1,5))
    # print("grads")
    # print(rosenbrock_grad(-1,-1,1,5))
    # print(rosenbrock_grad(1,1,1,5))
    #print("direction")
    #print(rosenbrock_grad(-1,-1))

    """ points = np.empty((0, 2))
    o = direction(rosenbrock_grad(-1,-1))
    new_point = line_search(startpoint,o)

    points = np.append(points, [new_point], axis=0)
    
    o = direction(rosenbrock_grad(new_point[0],new_point[1]))
    new_point = line_search(new_point,o)
    points = np.append(points, [new_point], axis=0)
    print(points) """

    #print(points)
    
    
    #fig1 = plt.figure()
    # plt.plot(alpha,vect)
    # plt.show()
    #print(o)
    #print(i)
    print("cc")
    points = gradient_descent(startpoint, grad_size_lim, max_iter)
    #print(gradient_descent(startpoint, grad_size_lim, max_iter, 1, 5))

    # Plot the surface.
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(startpoint[0], startpoint[1], rosenbrock(startpoint[0], startpoint[1]), c='r', marker='o')
    ax.scatter(1, 1, rosenbrock(1, 1), c='g', marker='o')
    ax.scatter(points[1:,0], points[1:,1], rosenbrock(points[1:,0], points[1:,1]), c='b', marker='o')
    ax.contour3D(X1c, X2c, Zc,200, cmap='viridis')
    
    ax = fig.add_subplot(1, 2,2)
    ax.contour(X1c, X2c, Zc, 200)
    ax.scatter(startpoint[0], startpoint[1], c='r', marker='o')
    ax.scatter(points[0:,0], points[0:,1], c='b', marker='o')
    #plot lines between points
    for i in range(1,len(points)):
        ax.plot([points[i-1][0],points[i][0]],[points[i-1][1],points[i][1]],c='b')
    #cil
    ax.scatter(1, 1 ,c='g', marker='o')
    plt.show()