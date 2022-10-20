import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

f0 = 0.004
k = 0.5
mu_v = 1e-3
mu_c = 1e-5
fh = f0*(1-k)
fc = f0
alpha = 10
beta = 0.001
# beta = 0

n = 1
m = 1
vig = np.linspace(0, n, n + 1)
coop = np.linspace(0, m, m + 1)
vig, coop = np.meshgrid(vig, coop)
vig = vig.flatten()
coop = coop.flatten()
rho0 = np.zeros_like(vig)
rho0[-1] = 1
dt = 100
T = 10000
t = np.arange(0, T+dt, dt)
t_inv = np.flip(t)
lambda_T = np.zeros_like(vig)


def average_r(rho):
    return fh*rho[-1] + (fc - beta)*rho[1] - beta*rho[0]


def cell_fraction_ODEs(rho, t):
    dhdt = (fh - average_r(rho))*rho[-1] - (mu_v + mu_c)*rho[-1]
    dsdt = - average_r(rho)*rho[2] + mu_v*rho[-1] - mu_c*rho[2]
    dcdt = (fc - beta - average_r(rho))*rho[1] - mu_v*rho[1] + mu_c*rho[-1]
    dbdt = -(beta + average_r(rho))*rho[0] + mu_v*rho[1] + mu_c*rho[2]

    drhodt = [dbdt, dcdt, dsdt, dhdt]

    return drhodt


def solve_cell_fractions(rho, t):
    solution = odeint(cell_fraction_ODEs, rho, t)

    healthy = solution[:, -1]
    cancerous = solution[:, n]
    senescent = solution[:, -(m + 1)]
    both = solution[:, 0]

    return healthy, cancerous, senescent, both


def lambda_ODEs(lambda_bc, t, h, c, s, b):
    dl1dt = fh*(lambda_bc[2]*s[t] + lambda_bc[1]*c[t] + lambda_bc[0]*b[t]) + 2*h[t] - mu_v*lambda_bc[2] - mu_c*lambda_bc[1] \
            - (fh - 2*fh*h[t] - (fc-beta)*c[t] + beta*b[t]
               - mu_v - mu_c)*lambda_bc[-1]
    dl2dt = - mu_c * lambda_bc[0] + (fh * h[t] + (fc - beta) * c[i] - beta * b[i]
                                + mu_c) * lambda_bc[2]
    dl3dt = (fc-beta) * (lambda_bc[-1] * h[i] + lambda_bc[2] * s[i] + lambda_bc[0] * b[i]) + 2 * h[i] \
            - mu_v * lambda_bc[0] - (fc - beta - fh * h[i] - 2 * (fc - beta) * c[i]
                                + beta * b[i] - mu_v) * lambda_bc[1]
    dl4dt = beta * (lambda_bc[-1] * h[i] + lambda_bc[2] * s[i] + lambda_bc[1] * c[i]) + (beta
                + fh * h[i] + (fc - beta) * c[i] - 2 * beta * b[i]) * lambda_bc[0]

    dldt = np.array([dl1dt, dl2dt, dl3dt, dl4dt])

    return dldt


def solve_lambdas(lambda_bc, t):
    solution = odeint(lambda_ODEs, lambda_bc, t, args=(h_inv, c_inv, s_inv, b_inv))

    lambda1 = solution[:, -1]
    lambda2 = solution[:, n]
    lambda3 = solution[:, -(m + 1)]
    lambda4 = solution[:, 0]

    return lambda1, lambda2, lambda3, lambda4



# def compute_betaopt():
#     return 1/(2*alpha)*(lambda3*c + lambda4*b - (c + b)*(lambda1*h + lambda2*s + lambda3*c + lambda4*b))


h, c, s, b = solve_cell_fractions(rho0, t)

h_inv = np.flip(h)
c_inv = np.flip(c)
s_inv = np.flip(s)
b_inv = np.flip(b)

l_1, l_2, l_3, l_4 = solve_lambdas(lambda_T, t_inv)

# plt.plot(t, h)
# plt.plot(t, s)
# plt.plot(t, c)
# plt.plot(t, b)
#
# plt.show()