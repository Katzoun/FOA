import numpy as np
import matplotlib.pyplot as plt


def fibonacciSearch(xf,f,a,b,n,e):
    
    #idxa = 0
    idxa = np.argmin(np.abs(xf - a))
    #idxb = len(f)-1
    idxb = np.argmin(np.abs(xf - b))

    #print(xf[idxb])
    phi = (1+np.sqrt(5))/2
    s = (1-np.sqrt(5))/(1+np.sqrt(5))
    rho = (1-np.power(s,n))/(phi*(1-np.power(s,n+1)))
    d = rho*b+(1-rho)*a
    # zjisteni, kteremu indexu odpovida d
    idxd = np.argmin(np.abs(xf - d))
    yd = f[idxd]
    d = xf[idxd]
    print("\nFibonacci Search Algorithm")
    print(f"Iterace {0:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  d={xf[idxd]:1.2f}")

    for i in range(1,n):

        if i == n-1:
            c = e*a+(1-e)*d
            idxc = np.argmin(np.abs(xf - c))
            c = xf[idxc]
        else:
            c = rho*x[idxa]+(1-rho)*x[idxb]
            idxc = np.argmin(np.abs(xf - c))
            c = xf[idxc]

        yc = f[idxc]


        # print(x[idxc],x[idxd])
        # print(yc,yd)
        print(f"Iterace {i:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  c={xf[idxc]:1.2f}  d={xf[idxd]:1.2f}")
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
        rho = (1-np.power(s,n-i))/(phi*(1-np.power(s,n+1-i)))

    if a < b:
        return (a,b)
    else:
        return (b,a)


def GoldenSectionSearch(xf,f,a,b,n):
    
    #idxa = 0
    idxa = np.argmin(np.abs(xf - a))
    #idxb = len(f)-1
    idxb = np.argmin(np.abs(xf - b))

    phi = (1+np.sqrt(5))/2
    rho = phi -1 
    d = rho*b+(1-rho)*a
    # zjisteni, kteremu indexu odpovida d
    idxd = np.argmin(np.abs(xf - d))
    yd = f[idxd]
    d = xf[idxd]

    print("\nGolden Section Search Algorithm")
    print(f"Iterace {0:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  d={xf[idxd]:1.2f}")
    
    for i in range(1,n):
        c = rho*x[idxa]+(1-rho)*x[idxb]
        idxc = np.argmin(np.abs(xf - c))
        c = xf[idxc]

        yc = f[idxc]
        print(f"Iterace {i:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  c={xf[idxc]:1.2f}  d={xf[idxd]:1.2f}")

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
        return (a,b)
    else:
        return (b,a)

def QuadraticFitSearch(xf,f,a,b,c,n):
    
    idxa = np.argmin(np.abs(xf - a))
    idxb = np.argmin(np.abs(xf - b))
    idxc = np.argmin(np.abs(xf - c))
    a = xf[idxa]
    b = xf[idxb]
    c = xf[idxc]
    ya = f[idxa]
    yb = f[idxb]
    yc = f[idxc]

    print("\nQuadratic fit Search Algorithm")
    print(f"Iterace {0:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  c={xf[idxc]:1.2f} ")

    for i in range(1,n-2):
        x = (ya*(b**2-c**2)+yb*(c**2-a**2)+yc*(a**2-b**2))/(2*(ya*(b-c)+yb*(c-a)+yc*(a-b)))
        idxx = np.argmin(np.abs(xf - x))
        x = xf[idxx]
        yx = f[idxx]

        print(f"Iterace {i:2d}:  a={xf[idxa]:1.2f}  b={xf[idxb]:1.2f}  c={xf[idxc]:1.2f}  x={xf[idxx]:1.2f}")
        if x > b:
            if yx > yb:
                c = x
                idxc = idxx 
                yc= yx
            else:
                a = b
                idxa = idxb
                ya = yb

                b = x
                idxb = idxx 
                yb= yx
        else:
            if yx > yb:
                a = x
                idxa = idxx 
                ya= yx
            else:
                c = b
                idxc = idxb
                yc = yb

                b = x
                idxb = idxx 
                yb= yx
    return(a,b,c)


    



if __name__ == '__main__':

    n= 8

    leftLim = -1
    rightLim = 5
    midpoint = (leftLim+rightLim)//2
    step = 0.01 #discretization step
    eps = 0.03 #parameter for fibonacci search


    x = np.arange(leftLim, rightLim, step)
    y = 0.2*np.exp(x-2)-x

    #nalezeni bracketu pomoci ruznych algo
    (leftBracketFib,rightBracketFib) = fibonacciSearch(x,y, leftLim, rightLim, n,eps)
    (leftBracketGold,rightBracketGold) = GoldenSectionSearch(x,y, leftLim, rightLim, n)
    (leftBracketQuad,midBracketQuad,rightBracketQuad) = QuadraticFitSearch(x,y, leftLim, midpoint, rightLim, n)
    
    print(f'\nInterval Fibonacci search: a={leftBracketFib:2.3f} b={rightBracketFib:2.3f}')
    print(f'Interval Golden Section search: a={leftBracketGold:2.3f} b={rightBracketGold:2.3f}')
    print(f'Interval Quadratic fit search: a={leftBracketQuad:2.3f} b={midBracketQuad:2.3f} c={rightBracketQuad:2.3f}')
    #minimum funkce (jen pro plot) 
    ymin = np.min(y)
    indmin = np.argmin(y)
    xmin = x[indmin]

    #vykresleni grafu
    figsize = 18,8 
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = 0.2*exp(x-2)-x')
    plt.vlines(x = leftBracketFib , ymin = -3, ymax = -2, colors = 'r', label = f"<a,b> = <{leftBracketFib:2.2f},{rightBracketFib:2.2f}>,  pomoci Fibonacci search")
    plt.vlines(x = rightBracketFib , ymin = -3, ymax = -2, colors = 'r')

    plt.vlines(x = leftBracketGold , ymin = -3, ymax = -2, colors = 'g', label = f"<a,b> = <{leftBracketGold:2.2f},{rightBracketGold:2.2f}>,  pomoci Golden section search")
    plt.vlines(x = rightBracketGold , ymin = -3, ymax = -2, colors = 'g')

    plt.vlines(x = leftBracketQuad , ymin = -3, ymax = -2, colors = 'purple', label = f"<a,c> = <{leftBracketQuad:2.2f},{rightBracketQuad:2.2f}>,  pomoci Quadratic fit search")
    plt.vlines(x = midBracketQuad , ymin = -3, ymax = -2, colors = 'purple', linestyles="dashed", label = f"b = {midBracketQuad:2.2f} (Quadratic fit search)")
    plt.vlines(x = rightBracketQuad , ymin = -3, ymax = -2, colors = 'purple')

    ax.plot(xmin, ymin, 'bo', label = f'Minimum v x = {xmin:2.2f}' )
    ax.plot(x, y)

    plt.legend()
    plt.grid()
    plt.show()