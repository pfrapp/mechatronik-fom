# Mechatronik
# Beispiel-Loesung zur Uebung 2, Aufgabe 7
# P. Rapp

# %% Packages

import numpy as np
import matplotlib.pyplot as plt


# %% Aufg. 1 -- Vektoren

u = np.array([1.0, 2.0])
v = np.array([-2.0, 1.0])
w = np.array([3.0, 4.0])

# (a) Graphische Darstellung
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(1,1,1)
ax.arrow(0.0, 0.0, *u, color=[0,0.5,0], head_width=0.1, label='u')
ax.arrow(0.0, 0.0, *v, color=[0,0,0.5], head_width=0.1, label='v')
ax.arrow(0.0, 0.0, *w, color=[0.5,0,0.5], head_width=0.1, label='w')
ax.grid(True)
ax.set_aspect('equal', 'box')
ax.legend()
ax.set(xlabel='x')
ax.set(ylabel='y')
plt.show()

# (b)
print(f'1. |u|  = {np.linalg.norm(u,2)}')
print(f'2. |-u| = {np.linalg.norm(-u,2)}')
print(f'3. |v|  = {np.linalg.norm(v,2)}')
print(f'4. |w|  = {np.linalg.norm(w,2)}')

print(f'5. u.v = {np.dot(u,v)}')
print(f'6. v.u = {np.dot(v,u)}')
print(f'7. v.w = {np.dot(v,w)}')
print(f'8. u.w = {np.dot(u,w)}')

# (c) Winkel
def winkel_in_grad(a, b):
    a_abs, b_abs = np.linalg.norm(a, 2), np.linalg.norm(b, 2)
    alpha = np.arccos(np.dot(a,b) / (a_abs * b_abs))
    return np.rad2deg(alpha)

print(f'Winkel zwischen u und v ist {winkel_in_grad(u,v)} Grad')
print(f'Winkel zwischen v und w ist {winkel_in_grad(v,w)} Grad')
print(f'Winkel zwischen u und w ist {winkel_in_grad(u,w)} Grad')

# %% Aufg. 2 -- Rechnen mit Matrizen und Vektoren

u, v = np.array([1.0, 2.0]), np.array([5.0, 3.0])
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[-2.0, 5.0], [0.0, 1.0]])

# Teil 1 (a)
for alpha in [2.0, -3.0]:
    print(alpha*u)
    print(alpha*v)
    print(alpha*A)
    print(alpha*B)
  
# Teil 1 (b)
print(A+B)
print(A-B)
print(A+A.T)
print(A-A.T)

# Teil 2
print(np.dot(A,B))
print(np.dot(B,A))
print(np.dot(A,A.T))
print(np.dot(B,B))

# Teil 3
print(np.dot(A,u))
print(np.dot(B,v))
print(np.dot(v,A))
print(np.dot(u,B))



# %% Aufg. 3 -- Lineares Gleichungssystem

A = np.array([[1.0, 3.0], [-4.0, 2.0]])
b = np.array([10.0, 2.0])

print(np.linalg.solve(A,b))


A = np.array([[3.0,2.0,-1.0], [1.0,3.0,-2.0], [5.0,0.0,1.0]])
b = np.array([-4.0, -11.0, 8.0])

print(np.linalg.solve(A,b))


# %% Aufg. 4 -- Invertierung von Matrizen

A = np.array([[1.0, 2.0], [3.0, 4.0]])
print(np.linalg.det(A))
if abs(np.linalg.det(A)) > 1.0e-6:
    print(np.linalg.inv(A))

B = np.array([[1.0, 0.0], [0.0, 1.0]])
print(np.linalg.det(B))
if abs(np.linalg.det(B)) > 1.0e-6:
    print(np.linalg.inv(B))

C = np.array([[4.0, 2.0], [2.0, 1.0]])
print(np.linalg.det(C))
if abs(np.linalg.det(C)) > 1.0e-6:
    print(np.linalg.inv(C))

D = np.array([[-2.0, 0.0], [0.0, 10.0]])
print(np.linalg.det(D))
if abs(np.linalg.det(D)) > 1.0e-6:
    print(np.linalg.inv(D))


# %% Spontan entstandene Zusatzaufgabe

H = np.array([[2.0, 3.0, 1.0],
              [5.0, 0.0, -1.0],
              [-3.0, 4.0, 1]])

det_H = np.linalg.det(H)
print(f'Determinante von H = {det_H}')

# %% Aufg. 5 -- Eigenwerte und -vektoren
# Hinweis: Die Reihenfolge der EW und EV stimmt nicht notwendigerweise
# mit der der haendischen Rechnung ueberein.
# Der Grund ist, dass die Reihenfolge der EW lambda nicht festgelegt ist.
# Zusaetzlich sind die EV, die Sie via np.linalg.eig() erhalten,
# auf die Laenge 1.0 normiert (Euklidische Norm).

# (a)
A=np.array([[4,-10],[-5,-1]]) / 3.0

print('Koeffizienten des charakteristischen Polynoms')
print(np.poly(A))

Lam, V = np.linalg.eig(A)
print('Eigenwerte')
print(Lam)
print('Eigenvektoren')
print(V)


# (b)
B=np.array([[8,18,-10],[12,2,10],[3,3,10]]) / 10.0

print('Koeffizienten des charakteristischen Polynoms')
print(np.poly(B))

Lam, V = np.linalg.eig(B)
print('Eigenwerte')
print(Lam)
print('Eigenvektoren')
print(V)

