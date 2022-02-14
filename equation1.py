# Project 3
# Kayla Zantello and Lucio Infante

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import quad as integral

# ODE 1
# homogeneous part - y'' + 2y' + y = 0

# return solution to homogeneous part that was found by hand
def yh(x):
    return 4 * np.exp(-x) + 2 * x * np.exp(-x)

# arrays for x and y values of homogeneous part solution
x_hom = []
y_hom = []

# x values for solution to be solved at - 50 values from 0 to 10
xrange = np.linspace(0, 10, 50)

# use function defined above to populate x and y arrays
for n in xrange:
    x_hom.append(n)
    y_hom.append(yh(n))

# plot the results
plt.plot(x_hom, y_hom)
plt.title("Solution to the Homogeneous Part - y'' + 2y' + y = 0")
plt.xlabel("x")
plt.ylabel("y = yh(x)")
plt.show()

# solving entire ODE - y'' + 2y' + y = 2x
# plot solution found by hand
def gs(x):
    return 4*np.exp(-x) + 2*x*np.exp(-x) + 2*x - 4

# arrays for x and y values of the general solution
x_gs = []
y_gs = []

for n in xrange:
    x_gs.append(n)
    y_gs.append(gs(n))

# green's function
# function that finds g(t, s)
def gts(t, s):
    return -s*np.exp(s-t) + t*np.exp(s-t)

# function representing r(s)
def rs(s):
    return 2*s

# integral of g(t, s)*r(s)ds from 0 to t
def greens(t):
    yg = integral(lambda s: gts(t, s)*rs(s), 0, t)
    return yg[0]

x_greens = []
y_greens = []

for n in xrange:
    x_greens.append(n)
    y_greens.append(greens(n))

# plot solution using odeint
# y is vector where y[0] = y and y[1] = y'
def ode(y, x):
    return [y[1], 2*x - 2*y[1] - y[0]]

# initial conditions
y0 = [0, 0]

sol_odeint = odeint(ode, y0, xrange)

ys2 = sol_odeint[:, 0]

# subplots
def plotEq1():
    fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle("Solutions to the ODE: y'' + 2y' + y = 2x")
    axs[0, 0].plot(x_gs, y_gs, 'r')
    axs[0, 0].set(ylabel="y = y(x)", title="By Hand")
    axs[0, 1].plot(x_greens, y_greens, 'g')
    axs[0, 1].set_title("Green's Function")
    axs[1, 0].plot(xrange, ys2, 'orange')
    axs[1, 0].set(xlabel="x", ylabel="y = y(x)", title="odeint")
    axs[1, 1].plot(xrange, ys2, 'orange', label="odeint")
    axs[1, 1].plot(x_gs, y_gs, 'r*', label="By Hand")
    axs[1, 1].plot(x_greens, y_greens, 'g.', label="Green's Function")
    axs[1, 1].set(xlabel="x", title="All Methods")
    axs[1, 1].legend()
    plt.show()
