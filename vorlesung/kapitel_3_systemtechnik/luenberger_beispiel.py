# %% Beispiel zum Luenberger Beobachter
#
# Siehe Vorlesungsfolien,
# Kapitel 3 "Systemtechnische Methodik der Mechatronik"
# Abschnitt "Luenberger Beobachter"
#
# Mechatronik, 2023
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
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

strecke = ctrl.ss(A, B, C, D)

wunschpole_beobachter = np.array([-10*omega_0, -10*omega_0])

# Berechnung von L
L = ctrl.acker(A.T, C.T, wunschpole_beobachter).T

print(f'L = {L}')



