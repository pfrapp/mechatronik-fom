# Uebung 5 -- Kinematik und Kinetik eines Massepunktes
# Mechatronik, 2022
# P. Rapp

# %% Packages

import numpy as np
import matplotlib.pyplot as plt


# %% Aufgabe 1: U-Bahn

# Parameter
a_A = 0.2
a_B = -0.6
delta_x = 3000
v_max = 90 / 3.6

# Berechnete Werte
t_1 = 125
t_2 = 161.67
t_F = 203.34

# Zeitachse
t = np.linspace(0,t_F,1000)

# Beschleunigung
def beschleunigung(t):
    if 0 <= t < t_1:
        return a_A
    elif t_1 <= t < t_2:
        return 0
    elif t_2 < t <= t_F:
        return a_B
    else:
        raise Exception('Invalid time')

# Geschwindigkeit
def geschwindigkeit(t):
    if 0 <= t < t_1:
        return a_A*t
    elif t_1 <= t < t_2:
        return a_A*t_1
    elif t_2 < t <= t_F:
        return a_A*t_1 + a_B*(t-t_2)
    else:
        raise Exception('Invalid time')

# Position
def position(t):
    if 0 <= t < t_1:
        return 0.5*a_A*t**2
    elif t_1 <= t < t_2:
        return 0.5*a_A*t_1**2 + a_A*t_1*(t-t_1)
    elif t_2 < t <= t_F:
        return 0.5*a_A*t_1**2 + a_A*t_1*(t_2-t_1) + a_A*t_1*(t-t_2) + 0.5*a_B*(t-t_2)**2
    else:
        raise Exception('Invalid time')

a = [beschleunigung(tau) for tau in t]
v = [geschwindigkeit(tau) for tau in t]
x = [position(tau) for tau in t]

fig = plt.figure(1)
plt.clf()

ax = fig.add_subplot(3,1,1)
ax.plot(t, x)
ax.set(xlabel='t (s)')
ax.set(ylabel='x (m)')
ax.set(title='Position')
ax.grid(True)

ax = fig.add_subplot(3,1,2)
ax.plot(t, v)
ax.set(xlabel='t (s)')
ax.set(ylabel='v (m/s)')
ax.set(title='Geschwindigkeit')
ax.grid(True)

ax = fig.add_subplot(3,1,3)
ax.plot(t, a)
ax.set(xlabel='t (s)')
ax.set(ylabel='a (m/s^2)')
ax.set(title='Beschleunigung')
ax.grid(True)


plt.show()
