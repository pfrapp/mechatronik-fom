# Uebung 6 -- Starrkoerpermechanik, Elektromechanik und Num. Integration
# Mechatronik, 2022
# P. Rapp

# %% Packages

import numpy as np
import matplotlib.pyplot as plt


# %% Aufgabe 5: Runge Kutta 4-stufig RK4

# RK4 Integrationsschritt
def rk4(f, x_k, delta_t):
    '''
    Ein Schritt des RK4 Verfahrens

    Parameters
    ----------
    f : Function
        Rechte Seite der Differentialgleichung.
    x_k : Scalar
        Aktueller Funktionswert x_k.
    delta_t : Scalar
        Integrationsschrittweite.

    Returns
    -------
    Rueckgabewert ist der naechste Funktionswert x_{k+1}.

    '''
    s1 = f(x_k)
    s2 = f(x_k+0.5*delta_t*s1)
    s3 = f(x_k+0.5*delta_t*s2)
    s4 = f(x_k+delta_t*s3)
    x_kp1 = x_k + delta_t/6.0*(s1+2*s2+2*s3+s4)
    return x_kp1

# Euler vorwaerts Integrationsschritt
def euler_vorwaerts(f, x_k, delta_t):
    '''
    Ein Schritt des Euler vorwaerts Verfahrens

    Parameters
    ----------
    f : Function
        Rechte Seite der Differentialgleichung.
    x_k : Scalar
        Aktueller Funktionswert x_k.
    delta_t : Scalar
        Integrationsschrittweite.

    Returns
    -------
    Rueckgabewert ist der naechste Funktionswert x_{k+1}.

    '''
    x_kp1 = x_k + delta_t*f(x_k)
    return x_kp1

# Rechte Seite der Differentialgleichung
def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([x2, -np.sin(x1)])

# Schrittweite
delta_t = 0.1

# Simulationsdauer
t_sim = 8.0

# Zeitpunkt
t = np.arange(0.0, t_sim, delta_t)

# Analytische Loesung der linearisierten Dgl (approximative Loesung)
x_analytisch = np.pi/6*np.cos(t)

x_numerisch = np.zeros((2,t.size)) # 2 Zeilen
x_numerisch[:,0] = np.array([x_analytisch[0], 0.0])

for idx, tau in enumerate(t[1:]):
    xk = x_numerisch[:,idx]
    x_numerisch[:,idx+1] = rk4(f, xk, delta_t)

x_numerisch_euler = np.zeros((2,t.size)) # 2 Zeilen
x_numerisch_euler[:,0] = np.array([x_analytisch[0], 0.0])
for idx, tau in enumerate(t[1:]):
    xk = x_numerisch_euler[:,idx]
    x_numerisch_euler[:,idx+1] = euler_vorwaerts(f, xk, delta_t)

fig = plt.figure(2)
plt.clf()
plt.plot(t, x_analytisch, label='Analytische approximative Loesung', linewidth=2.0)
plt.plot(t, x_numerisch[0,:], '--', label='Numerische Loesung RK4')
plt.plot(t, x_numerisch_euler[0,:], '--', label='Numerische Loesung Euler vorwaerts')
plt.xlim(0,t_sim)
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True)
plt.legend()
plt.show()

