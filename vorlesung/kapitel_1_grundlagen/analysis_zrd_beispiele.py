# %% Beispiele zur Analysis und ZRD


# %% Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import axes3d

# %% Farben

accent1 = [0.0, 0.6, 0.5411764705882353]
accent5 = [0.0, 0.6549019607843137, 0.8941176470588236]

# %% 2D Funktion (Contour)

x = np.linspace(0,10,100)
y = np.linspace(-5,5,100)

X, Y = np.meshgrid(x,y)
Z = 3*X + 5*Y**2
plt.figure(1)
plt.contourf(X, Y, Z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
     

# %% 2D Funktion (Surface)

ax = plt.figure(2).add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax.set(xlabel='x')
ax.set(ylabel='y')
plt.show()


# %% Optimierung

x = np.linspace(-3, 10, 1000)
y = 3*x**2 - 12*x + 11

fig = plt.figure(1)
plt.clf()

plt.plot(x, y, color=accent1)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2,6)
plt.ylim(-5,10)
plt.show()


# %% Linearisierung

x = np.linspace(-2,8,1000)
f = np.sin(x)

# Linearisierung um x_0=0
f_lin = x

fig = plt.figure(1)
plt.clf()
plt.plot(x,f,color=accent1, label='f(x)')
plt.plot(x,f_lin,color=accent5, label='Linearisierung')
plt.grid(True)
plt.xlabel('x')
plt.legend()
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.show()

# Linearisierung um x_0=pi/2
f_lin = 1.0*np.ones(f.size)

fig = plt.figure(2)
plt.clf()
plt.plot(x,f,color=accent1, label='f(x)')
plt.plot(x,f_lin,color=accent5, label='Linearisierung')
plt.grid(True)
plt.xlabel('x')
plt.legend()
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.show()

# %% Definition eines dynamischen Systems in Zustandsraumdarstellung

import control as ctrl
import numpy as np

A = np.array([[0.0, 1.0], [2.0, -9.0]])
B = np.array([1.0, 8.0])
C = np.array([2.0, 4.0])
D = 0.0

sys = ctrl.ss(A, B, C, D)
print(sys)

# %% Lotka-Volterra

params = { 'alpha': 1.0, 
            'beta': 1.0,
            'gamma': 2.0,
            'delta': 1.0 }

t_f = 20.0

def ode_rhs(t, x):
    x_dot = np.zeros((2,))
    x1, x2 = x
    x_dot[0] = -(params['alpha'] - params['beta']*x2)*x1
    x_dot[1] = (params['gamma'] - params['delta']*x1)*x2
    return x_dot

x_0 = np.array([3.0, 5.0])
scipy_integration_result = solve_ivp(ode_rhs, [0, t_f], x_0, max_step=0.05)



plt.figure(1)
plt.clf()
plt.plot(scipy_integration_result.y[0], scipy_integration_result.y[1], color=accent1)
plt.grid(True)
plt.xlabel('Bestand Raeuber')
plt.ylabel('Bestand Beute')
plt.show()


plt.figure(2)
plt.clf()
plt.plot(scipy_integration_result.t, scipy_integration_result.y[0], color=accent1)
plt.grid(True)
plt.xlabel('Zeit')
plt.ylabel('Bestand Raeuber')
plt.show()



plt.figure(3)
plt.clf()
plt.plot(scipy_integration_result.t, scipy_integration_result.y[1], color=accent1)
plt.grid(True)
plt.xlabel('Zeit')
plt.ylabel('Bestand Beute')
plt.show()

# %% Phasenraumdarstellung der Schwingungsdifferentialgleichung

A = np.array([[0.0, 1.0], [-1.0, -0.2]])

t_f = 50.0

def ode_rhs(t, x):
    x_dot = np.dot(A,x)
    return x_dot

x_0 = np.array([3.0, 5.0])
scipy_integration_result = solve_ivp(ode_rhs, [0, t_f], x_0, max_step=0.05)


fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(1,1,1)
ax.plot(scipy_integration_result.y[0], scipy_integration_result.y[1], color=accent1)
ax.grid(True)
ax.set(xlabel='x_1')
ax.set(ylabel='x_2')
ax.set_aspect('equal', 'box')
plt.show()