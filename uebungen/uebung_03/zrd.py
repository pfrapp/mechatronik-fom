# %% Mechatronik -- Uebung 3
# Analysis und Zustandsraumdarstellung
# P. Rapp


# %% Packages

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

# %% Aufgabe 1 -- Differentiation (Ableitung)

# Optimierung
x = np.linspace(-7,7,1000)
y = x**3 - 27*x + 5

fig = plt.figure(1)
plt.clf()
plt.plot(x,y,linewidth=2.0)
plt.grid(True)
plt.show()

# Optimierung
x = np.linspace(-1,2,1000)
y = np.sin(2.0*np.pi*x)

fig = plt.figure(2)
plt.clf()
plt.plot(x,y,linewidth=2.0)
plt.grid(True)
plt.show()

# %% Aufgabe 2 -- Zustandsraumdarstellung -- Uebertragungsfunktion anhand ZRD

# Definition des Systems in ZRD
A = np.array([[0.0, 1.0], [2.0, -9.0]])
B = np.array([1.0, 8.0])
C = np.array([2.0, 4.0])
D = 0.0

sys = ctrl.ss(A, B, C, D)
print(sys)

# Plot der Sprungantwort
h = ctrl.step_response(sys)
fig = plt.figure(1)
plt.clf()
plt.plot(h.t, h.y.flatten())
plt.grid(True)
plt.show()

# Konvertierung in Darstellung als Uebertragungsfunktion (transfer function, tf)
sys = ctrl.tf(sys)
print(sys)

# Berechnung der Pole (3 Wege)

# 1. Eigenwerte von A
np.linalg.eig(A)

# 2. Via ctrl.pole()
ctrl.pole(sys)

# 3. Als Nullstellen des charakteristschen Polynoms (Nenner der Uebertragungsfunktion)
np.roots([1,9,-2])

# %% Aufgabe 3 -- Stabilitaet im Zustandsraum

A_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
lam, _ = np.linalg.eig(A_1)
print(f'Koeffizienten des char. Polynoms: {np.poly(A_1)}')
print(f'Pole: {lam}')

A_2 = np.array([[0.0, 1.0], [-3.0, -5.0]])
lam, _ = np.linalg.eig(A_2)
print(f'Koeffizienten des char. Polynoms: {np.poly(A_2)}')
print(f'Pole: {lam}')


