# Uebung 6 -- Starrkoerpermechanik, Elektromechanik und Num. Integration
# Mechatronik, 2022
#
# Aufgabe 6: Numerische Integration (Simulation) des Segways
#
# P. Rapp

# %% Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# %% Mechanische Parameter definieren


class Parameters():
    def __init__(self):
        #
        # Aufbau
        #
        
        # Masse
        self.m1 = 0.5   # 500 g
        # Laenge
        self.L = 0.3    # 30 cm
        # Massentraegheitsmoment unter der Annahme, dass
        # der Aufbau als duenner Stab modelliert werden kann.
        self.J1 = self.m1 * self.L**2 / 12.0
        # Distanz zum Schwerpunkt
        self.a = 0.5*self.L
        
        #
        # Einzelnes Rad
        #
        
        # Masse
        self.m2 = 0.2   # 200 g
        # Radius
        self.R = 0.05   # 5 cm
        # Massentraegheitsmoment unter der Annahme, dass
        # das Rad als duenne Kreisscheibe modelliert werden kann.
        self.J2 = 0.5 * self.m2 * self.R**2
        
        # Wir haben 2 Raeder, die Achse wird vernachlaessigt.
        self.m2 *= 2
        self.J2 *= 2
        
        #
        # Gravitation: 9.81 m/s^2
        #
        self.g = 9.81
        
        #
        # Damping coefficient.
        #
        self.d = 0.01
        
        #
        # Abgeleitete Parameter
        #
        self.J1_tilde = self.m1*self.a**2 + self.J1
        self.J2_tilde = (self.m1 + self.m2)*self.R**2 + self.J2
        
        
parameters = Parameters() 


# %% Rechte Seite der ZRD Differentialgleichung
#
# ODE: Ordinary differential equation.


def ode_rhs(x, u, params):
    ''' Rechte Seite der nichtlinearen Zustandsraumdarstellung
        x_dot = f(x,u)
        
        Der Zustandsvektor x enthaelt [theta, theta_dot, phi, phi_dot].
        Dabei ist theta der Neigewinkel und phi der Winkel des Rads.

        u ist das Motordrehmoment.        
    '''
    # Einzelne Zustaende extrahieren.
    x1, x2, x3, x4 = x
    
    # Hilfsgroessen.
    A = params.m1 * params.a * params.R * np.cos(x1)
    B = params.m1 * params.a * np.sin(x1)
    D = A**2 - params.J1_tilde * params.J2_tilde
    
    # Ableitungen der Zustaende berechnen (dies ist die rechte Seite
    # der ZRD Dgl)
    x1_dot = x2
    x3_dot = x4
    
    x2_dot = params.R*x2**2*A*B/D - params.g*params.J2_tilde*B/D + (A+params.J2_tilde)/D * params.d * (x2-x4) + (A+params.J2_tilde)/D * u
    x4_dot = -params.R*params.J1_tilde*x2**2*B/D + params.g*A*B/D - (A+params.J1_tilde)/D * params.d * (x2-x4) - (A+params.J1_tilde)/D * u
    
    x_dot = np.array([x1_dot, x2_dot, x3_dot, x4_dot])
    return x_dot

def ode_rhs_for_scipy(t, x):
    # Ungeregeltes System.
    u = 0.0
    return ode_rhs(x,u,parameters)



# %% Numerische Integration mit scipy

# Anfangsbedingung.
x_0 = np.array([np.deg2rad(10.0), 0.0, np.deg2rad(0), 0.0])

# Zeitdauer der Simulation.
t_f = 10.0

# Zeitpunkte an denen wir die Signale auswerten wollen.
t_eval = np.arange(0, t_f, 0.02)

scipy_integration_results = solve_ivp(ode_rhs_for_scipy, \
                                      [0, t_f], \
                                      x_0, \
                                      t_eval=t_eval)

# %% Darstellung der Szene

def rotations_matrix(beta):
    s, c = np.sin(beta), np.cos(beta)
    R = np.array([[c, -s], [s, c]])
    return R

def homogene_transformations_matrix(beta, trans):
    T = np.eye(3)
    T[:2,:2] = rotations_matrix(beta)
    T[:2,2] = trans
    return T
    
def plot_scene(axis, x, params):
    theta, theta_dot, phi, phi_dot = x
    s = params.R*phi
    # Schwerpunkt Aufbau
    p_1 = np.array([-s - params.L/2*np.sin(theta), params.R+params.L/2*np.cos(theta)])
    # Schwerpunkt Rad
    p_2 = np.array([-s, params.R])
    # Homogene Transformationsmatrizen
    T_1 = homogene_transformations_matrix(theta, p_1)
    T_2 = homogene_transformations_matrix(phi, p_2)
    
    # Aufbau
    aufbau_coords = np.array([[0.0, params.L/2, 1.0],
                              [0.0, -params.L/2, 1.0]]).T
    aufbau_coords = np.dot(T_1, aufbau_coords)
    axis.plot(aufbau_coords[0,:], aufbau_coords[1,:], linewidth=2.0)
    
    # Rad
    beta = np.linspace(0,2*np.pi,50)
    rad_coords = np.vstack((params.R*np.sin(beta), -params.R*np.cos(beta), np.ones(beta.shape)))
    origin = np.array([0.0, 0.0, 1.0]).reshape((3,1))
    rad_coords = np.hstack((origin, rad_coords))
    rad_coords = np.dot(T_2, rad_coords)
    axis.plot(rad_coords[0,:], rad_coords[1,:])
    
    axis.set_aspect('equal', 'box')
    axis.set(xlim=(-0.5,0.5))
    axis.set(ylim=(-0.5,0.5))
    axis.grid(True)
    
    
    
    
fig = plt.figure(1); plt.clf()
ax = fig.add_subplot(1,1,1)

plot_scene(ax, x_0, parameters)
ax.set(title='Anfangsbedingung')
plt.show()

# %% Plot results

t = scipy_integration_results.t
y = scipy_integration_results.y

fig = plt.figure(2); plt.clf()

ax = fig.add_subplot(2,2,1)
ax.plot(t, np.rad2deg(y[0,:]))
ax.grid(True)
ax.set(xlabel='t (s)')
ax.set(ylabel='x_1 = theta (deg)')

ax = fig.add_subplot(2,2,2)
ax.plot(t, np.rad2deg(y[1,:]))
ax.grid(True)
ax.set(xlabel='t (s)')
ax.set(ylabel='x_2 = theta_dot (deg/s)')


ax = fig.add_subplot(2,2,3)
ax.plot(t, np.rad2deg(y[2,:]))
ax.grid(True)
ax.set(xlabel='t (s)')
ax.set(ylabel='x_3 = phi (deg)')

ax = fig.add_subplot(2,2,4)
ax.plot(t, np.rad2deg(y[3,:]))
ax.grid(True)
ax.set(xlabel='t (s)')
ax.set(ylabel='x_4 = phi_dot (deg/s)')

plt.show()

# %% Erstellung eines Videos

fig = plt.figure(1); plt.clf()
ax = fig.add_subplot(1,1,1)

for idx, tau in enumerate(t):
    x = y[:,idx]
    ax.clear()
    plot_scene(ax, x, parameters)
    ax.set(title=f'{tau:6.2f} / {max(t):6.2f}')
    plt.show()
    plt.pause(0.01)