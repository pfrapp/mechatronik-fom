# %% Mechatronik -- Uebung 3
# Analysis und Zustandsraumdarstellung
# P. Rapp
#
# Adressierung der Frage, wie man sich die Linearisierung bzw.
# die Jacobi-Matrix graphisch vorstellen kann.
#


# %% Packages

import matplotlib.pyplot as plt
import numpy as np

# %% Vektorfeld berechnen

def vektorfeld(x):
    x1, x2 = x
    x1_punkt = x2
    x2_punkt = -np.sin(x1) - 1.2*x2
    return np.array([x1_punkt, x2_punkt])

X1 = np.arange(-10,10,0.2)
X2 = np.arange(-5,5,0.2)
X1, X2 = np.meshgrid(X1, X2)

X1_punkt = np.zeros_like(X1)
X2_punkt = np.zeros_like(X2)

n_rows, n_cols = X1.shape

for rr in range(n_rows):
    for cc in range(n_cols):
        x = np.array(np.array([X1[rr,cc], X2[rr,cc]]))
        x_punkt = vektorfeld(x)
        X1_punkt[rr,cc] = x_punkt[0]
        X2_punkt[rr,cc] = x_punkt[1]
    
# %% Vektorfeld plotten

pfeil_skalierung = 0.3
linien_skalierung = 0.1

fig = plt.figure(1)
plt.clf()

ax = fig.add_subplot(1,1,1)

for rr in range(n_rows):
    for cc in range(n_cols):
        arrow_start_x1 = X1[rr,cc]
        arrow_start_x2 = X2[rr,cc]
        arrow_dx1 = pfeil_skalierung * X1_punkt[rr,cc]
        arrow_dx2 = pfeil_skalierung * X2_punkt[rr,cc]
        
        if np.mod(rr, 4) == 0 and np.mod(cc, 4) == 0:
            ax.arrow(arrow_start_x1, arrow_start_x2, arrow_dx1, arrow_dx2,
                     head_width = 0.10, head_length = 0.2,
                     color = [0,0,0],
                     width = 0.02)
        
        arrow_length = np.hypot(arrow_dx1, arrow_dx2)
        arrow_dx1 *= linien_skalierung / arrow_length
        arrow_dx2 *= linien_skalierung / arrow_length
        ax.plot([arrow_start_x1, arrow_start_x1+arrow_dx1],
                [arrow_start_x2, arrow_start_x2+arrow_dx2],
                color = [0.0, 0.0, 1.0],
                linewidth = 1.0)

ax.set_aspect('equal', 'box')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xlim((-10,10))
ax.set_ylim((-5,5))

plt.show()



