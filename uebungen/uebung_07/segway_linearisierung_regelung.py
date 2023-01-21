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
import control as ctrl

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


# %% Linearisierung
    
system_dynamics = {}

# Dieser Term wird haeufig verwendet und dahehr einer separaten Variable zugewiesen.
tmp = (parameters.m1*parameters.a*parameters.R)**2 - parameters.J1_tilde*parameters.J2_tilde

# Systemmatrix
system_dynamics['A'] = 1.0/tmp * \
    np.array([[0.0, tmp, 0.0, 0.0], \
              [-parameters.g*parameters.J2_tilde*parameters.m1*parameters.a,
                  parameters.d*(parameters.m1*parameters.a*parameters.R + parameters.J2_tilde),
                  0.0,
                  -parameters.d*(parameters.m1*parameters.a*parameters.R + parameters.J2_tilde)], \
              [0.0, 0.0, 0.0, tmp], \
              [parameters.g*parameters.m1**2*parameters.a**2*parameters.R,
                  -parameters.d*(parameters.m1*parameters.a*parameters.R + parameters.J1_tilde),
                  0.0,
                  parameters.d*(parameters.m1*parameters.a*parameters.R + parameters.J1_tilde)]])

# Steuermatrix
system_dynamics['B'] = 1.0/tmp * \
    np.array([0.0,\
              parameters.m1*parameters.a*parameters.R + parameters.J2_tilde, \
              0.0,\
              -(parameters.m1*parameters.a*parameters.R + parameters.J1_tilde)]).reshape((4,1))

# %% Numerische Kontrolle der Linearisierung (via finiter Differenzen).

x_center = np.array([0.0, 0.0, 0.0, 0.0])
u_center = 0.0

my_eps = 0.00001

dx1 = np.array([1.0, 0.0, 0.0, 0.0])
dx2 = np.array([0.0, 1.0, 0.0, 0.0])
dx3 = np.array([0.0, 0.0, 1.0, 0.0])
dx4 = np.array([0.0, 0.0, 0.0, 1.0])

df_dx1 = (ode_rhs(x_center + my_eps*dx1, u_center, parameters) \
       -ode_rhs(x_center - my_eps*dx1, u_center, parameters))/(2.0*my_eps)
df_dx2 = (ode_rhs(x_center + my_eps*dx2, u_center, parameters) \
       -ode_rhs(x_center - my_eps*dx2, u_center, parameters))/(2.0*my_eps)
df_dx3 = (ode_rhs(x_center + my_eps*dx3, u_center, parameters) \
       -ode_rhs(x_center - my_eps*dx3, u_center, parameters))/(2.0*my_eps)
df_dx4 = (ode_rhs(x_center + my_eps*dx4, u_center, parameters) \
       -ode_rhs(x_center - my_eps*dx4, u_center, parameters))/(2.0*my_eps)
    
df_du = (ode_rhs(x_center, u_center + my_eps, parameters) \
        -ode_rhs(x_center, u_center - my_eps, parameters))/(2.0*my_eps)

A_numerically = np.vstack((df_dx1, df_dx2, df_dx3, df_dx4)).T
print('Numerically determined system matrix A')
print(A_numerically)

print('Error is')
print(np.linalg.norm(system_dynamics['A'] - A_numerically))

print('Numerically determined input vector b')
print(df_du)

print('Error is')
print(np.linalg.norm(system_dynamics['B'].flatten() - df_du))


# %% Steuerbarkeit und Pole placement

# Die Determinante ist ungleich Null --> Das System ist steuerbar.
controllability_det = np.linalg.det(ctrl.ctrb(system_dynamics['A'], \
                                              system_dynamics['B']))
print(f'Determinante der Steuerbarkeitsmatrix: {controllability_det}')

controller = {}
my_poles = [-1+1j,-1-1j,-2.0+2j,-2-2j]
my_poles = [-0.5+1j,-0.5-1j,-2.0+2j,-2-2j]
controller['K'] = ctrl.acker(system_dynamics['A'], system_dynamics['B'], my_poles)


# Berechnung der Pole des geschlossenen Kreises
A_closed_loop = system_dynamics['A'] - np.dot(system_dynamics['B'], controller['K'])

# %% Regler via LQR

# x_1 = theta (Aufbau), x_2 = theta_dot
# x_3 = phi (Rad), x_4 = phi_dot
Q = np.diag([1.0,10.0,10.0,10.0])
# Q[1,3] = -20.0
# Q[3,1] = Q[1,3]
R = np.array([3000.0])

controller['K'], _, _ = ctrl.lqr(system_dynamics['A'], system_dynamics['B'], Q, R)

A_closed_loop = system_dynamics['A'] - np.dot(system_dynamics['B'], controller['K'])
print(np.linalg.eig(A_closed_loop))


# %% Geregelten Roboter simulieren

def limit_abs(u, lim):
    if u > lim:
        u = lim
    if u < -lim:
        u = -lim
    return u

def control_law(x):
    # Auswertung des Regelgesetzes (bzw. der Zustandsrueckfuehrung)
    u = -np.dot(controller['K'], x).item()
    return u

def ode_rhs_controlled(t, x):
    u = control_law(x)
    return ode_rhs(x,u,parameters)

x_0 = np.array([np.deg2rad(10.0), 0.0, np.deg2rad(360), 0.0])
t_f = 10.0
t_eval = np.arange(0, t_f, 0.02)

scipy_integration_results = solve_ivp(ode_rhs_controlled, [0, t_f], x_0, t_eval=t_eval)

# %% Ergebnisse darstellen

t = scipy_integration_results.t
y = scipy_integration_results.y

fig = plt.figure(3); plt.clf()

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


# %% Video des geregelten Segways

fig = plt.figure(5); plt.clf()
ax = fig.add_subplot(1,1,1)

for idx, tau in enumerate(t):
    x = y[:,idx]
    ax.clear()
    plot_scene(ax, x, parameters)
    ax.set(title=f'{tau:6.2f} / {max(t):6.2f}')
    plt.show()
    plt.pause(0.01)