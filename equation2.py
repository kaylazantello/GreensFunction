# Project 3
# Kayla Zantello and Lucio Infante

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import quad as integral
import math

# x values for solution to be solved at - 50 values from 0 to 10
xr = np.linspace(0, 5, 50)

# return solution to homogeneous part that was found by hand
def yh(x):
    return 2*math.cos(x)

# array for y values of homogeneous part solution
y_hom = []

for n in xr:
    y_hom.append(yh(n))

# plot the results
plt.plot(xr, y_hom)
plt.title("Solution to the Homogeneous Part - y'' + y = 0")
plt.xlabel("x")
plt.ylabel("y = yh(x)")
plt.show()

# solving the entire ODE - y'' + y = x^2
# plot the solution found by hand
def gs(x):
    return 2*math.cos(x) + x**2 - 2

y_gs = []

for n in xr:
    y_gs.append(gs(n))

# green's function
# function that finds g(t, s)
def gts(t, s):
    return math.cos(s)*math.sin(t) - math.sin(s)*math.cos(t)

# function representing r(s)
def rs(s):
    return s**2

# integral of g(t, s)*r(s)ds from 0 to t
def greens(t):
    yg = integral(lambda s: gts(t, s)*rs(s), 0, t)
    return yg[0]

y_greens = []

for n in xr:
    y_greens.append(greens(n))

# plot solution using odeint
# y is vector where y[0] = y and y[1] = y'
def ode(y, x):
    return [y[1], x**2 - y[0]]

y0 = [0, 0]

sol_ode = odeint(ode, y0, xr)
ys2 = sol_ode[:, 0]

# subplots
def plotEq2():
    fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle("Solutions to the ODE: y'' + y = x^2")
    axs[0, 0].plot(xr, y_gs, 'r')
    axs[0, 0].set(ylabel="y = y(x)", title="By Hand")
    axs[0, 1].plot(xr, y_greens, 'g')
    axs[0, 1].set_title("Green's Function")
    axs[1, 0].plot(xr, ys2, 'orange')
    axs[1, 0].set(xlabel="x", ylabel="y = y(x)", title="odeint")
    axs[1, 1].plot(xr, y_gs, 'r*', label="By Hand")
    axs[1, 1].plot(xr, y_greens, 'g.', label="Green's Function")
    axs[1, 1].plot(xr, ys2, 'orange', label="odeint")
    axs[1, 1].set(xlabel="x", title="All Methods")
    axs[1, 1].legend()
    plt.show()
