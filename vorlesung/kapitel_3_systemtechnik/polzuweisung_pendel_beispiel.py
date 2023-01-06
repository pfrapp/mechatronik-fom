# %% Beispiel zur Polzuweisung
#
# Siehe Vorlesungsfolien,
# Kapitel 3 "Systemtechnische Methodik der Mechatronik"
# Abschnitt "Reglerentwurf im Zustandsraum"
#
# Mechatronik, 2022
# P. Rapp

# %% Packages

import numpy as np
import matplotlib.pyplot as plt

# Fuer Regelungstechnik
import control as ctrl


# %% Definition der Regelstrecke

# omega_0 ist die (Kreis)Frequenz der Schwingung des Pendels.
omega_0 = 1.0


A = np.array([[0.0, 1.0], [-omega_0**2, 0.0]])
B = np.array([0.0, 1.0])
C = np.eye(2)
D = np.array([[0.0], [0.0]])

strecke = ctrl.ss(A, B, C, D)


print(f'Pole der ungeregelten Strecke: {ctrl.poles(strecke)}')

# Wir simulieren von 0 bis 7 Sekunden.
t = np.linspace(0.0, 7.0, 200)

# Anfangsbedingung: x1 = 1.0, x2 = 0.0
x0 = np.array([1.0, 0.0])

response = ctrl.initial_response(strecke, T=t, X0=x0)

x1 = response.states[0,:]
x2 = response.states[1,:]

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(1,1,1)

ax.plot(t, x1, label='$x_1$')
ax.plot(t, x2, label='$x_2$')

ax.grid(True)
ax.set(xlabel='$t$')
ax.set(ylabel='$x$')
ax.legend()
ax.set(title=f'Ungeregelte Strecke, $\omega_0$ = {omega_0:4.2f}')

plt.show()

# %% Definition der Zustandsrueckfuehrung

K_params = np.array([3.0*omega_0**2, 4*omega_0])

# Kein Zustand im Regler, da der Regler rein algebraisch ist.
K_sys = ctrl.ss([], [], [], K_params, states=[])
print(f'Anzahl Zustaende im Regler: {K_sys.nstates}')

# %% Bloecke verbinden

block_strecke = ctrl.ss2io(strecke, inputs='u', outputs=['x1', 'x2'])
block_regler = ctrl.ss2io(K_sys, inputs=['x1', 'x2'], outputs='kx')
summations_block = ctrl.summing_junction(inputs='-kx', output='u')
geschlossener_kreis = ctrl.interconnect((block_strecke, block_regler, summations_block), inputs='u', outputs=['x1', 'x2'])

print(f'Pole des geschlossenen Kreises: {ctrl.poles(geschlossener_kreis)}')

response = ctrl.initial_response(geschlossener_kreis, T=t, X0=x0)

x1 = response.states[0,:]
x2 = response.states[1,:]

# Nachprozessierung zur Berechnung der Stellgroesse.
u = -np.dot(K_params, response.states)

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(2,1,1)

ax.plot(t, x1, label='$x_1$')
ax.plot(t, x2, label='$x_2$')

ax.grid(True)
ax.set(xlabel='$t$')
ax.set(ylabel='$x$')
ax.legend()

ax = fig.add_subplot(2,1,2)

ax.plot(t, u, label='$u$')

ax.grid(True)
ax.set(xlabel='$t$')
ax.set(ylabel='$u$')
ax.legend()

plt.suptitle(f'Geschlossener Kreis, $\omega_0$ = {omega_0:4.2f}')

plt.show()

# %% Test auf Steuerbarkeit

# Steuerbarkeitsmatrix (controllability matrix)
S = ctrl.ctrb(strecke.A, strecke.B)

# Berechnung der Determinanten der Steuerbarkeitsmatrix
S_det = np.linalg.det(S)

print(f'Determinante der Steuerbarkeitsmatrix = {S_det}')

if np.abs(S_det) > 1.0e-6:
    print('Das System ist steuerbar.')
else:
    print('Das System ist nicht steuerbar.')
    


# %% Berechnung der Reglerparameter ueber die Ackermannformel

wunschpole = [-2.0*omega_0, -2.0*omega_0]

K_acker = ctrl.acker(strecke.A, strecke.B, wunschpole)

print(f'Reglerparameter via Ackermannformel: {K_acker}')

