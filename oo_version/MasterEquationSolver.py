import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random
from mpl_toolkits import mplot3d
from scipy import stats
import pickle

class Solver:
    def __init__(self):

        self.parameters = {
            'k': [0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5],
            'f0': [0.004, 0.004, 0.004, 0.0025, 0.004, 0.004, 0.004],
            'mu_v': [1e-3, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
            'mu_c': [1e-5, 1e-5, 1e-7, 1e-5, 1e-5, 1e-5, 1e-5],
            'lambda_v': [0, 0, 0, 0, 0, 1e-3, 0],
            'lambda_c': [0, 0, 0, 0, 0, 0, 5e-4]
        }

        self.removal_parameters = {
            'beta': [0.001, 0.005],
            'cell_type': ['cancerous', 'senescent']
        }
        self.cell_type = None

        self.n = 1
        self.m = 1

    def growth_rate(self, v, c, k, f0):
        return f0 * v * (1 - k * c)

    def mean_growth_rate(self, rho):
        return np.sum(self.growth_rate(self.v, self.c, self.k, self.f0) * rho)

    def solve_system(self, rho, t):
        if self.cell_type == 'cancerous':
            solution = odeint(self.remove_cancer, rho, t)
        elif self.cell_type == 'senescent':
            solution = odeint(self.remove_senescence, rho, t)
        else:
            solution = odeint(self.compute_master_equation, rho, t)

        healthy = solution[:, -1]
        cancerous = solution[:, self.n]
        senescent = solution[:, -(self.m + 1)]
        both = solution[:, 0]

        return healthy, cancerous, senescent, both

    def hazard_func(self, rho, t):
        if self.cell_type == 'cancerous':
            solution = odeint(self.remove_cancer, rho, t)
        elif self.cell_type == 'senescent':
            solution = odeint(self.remove_senescence, rho, t)
        else:
            solution = odeint(self.compute_master_equation, rho, t)

        healthy = solution[:, -1]

        return -np.gradient(np.log(healthy))

    def mean_rate_cancer(self, rho):
        return self.beta * np.sum(np.where(self.c == 0, rho, 0))

    def mean_rate_senescence(self, rho):
        return self.beta * np.sum(np.where(self.v == 0, rho, 0))

    def compute_master_equation(self, rho, t):
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) - self.mu_v * np.where(self.v > 0, rho, 0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) \
               + self.lambda_v * np.where(self.v > 0, np.roll(rho, 1), 0) \
               - self.lambda_v * np.where(self.v < self.n, rho, 0) \
               + self.lambda_c * np.where(self.c > 0, np.roll(rho, self.n + 1), 0) \
               - self.lambda_c * np.where(self.c < self.m, rho, 0)

    def remove_cancer(self, rho, t):
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) \
               - self.mu_v * np.where(self.v > 0, rho, 0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) \
               - self.beta * np.where(self.c == 0, rho, 0) \
               + self.mean_rate_cancer(rho) * rho

    def remove_senescence(self, rho, t):
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) \
               - self.mu_v * np.where(self.v > 0, rho, 0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) - self.beta * np.where(self.v == 0, rho, 0) \
               + self.mean_rate_senescence(rho) * rho

    def run(self):
        v = np.linspace(0, self.n, self.n + 1)
        c = np.linspace(0, self.m, self.m + 1)
        v, c = np.meshgrid(v, c)
        self.v = v.flatten()
        self.c = c.flatten()
        rho0 = np.zeros_like(self.v)
        rho0[-1] = 1
        self.rho0 = rho0
        self.t = np.linspace(0, 12500, 100)

        self.param_version = len(self.parameters['k'])

        results = {}
        removal_results = {}

        for i in range(0, self.param_version):
            self.k = self.parameters['k'][i]
            self.mu_c = self.parameters['mu_c'][i]
            self.mu_v = self.parameters['mu_v'][i]
            self.f0 = self.parameters['f0'][i]
            self.lambda_v = self.parameters['lambda_v'][i]
            self.lambda_c = self.parameters['lambda_c'][i]

            h, c, s, b = self.solve_system(self.rho0, self.t)
            hazard = self.hazard_func(self.rho0, self.t)

            results_name = "_".join([k + "=" + str(eval("self." + k, {"self": self})) for k in self.parameters.keys()])

            results[results_name] = {'h': h,
                                     'c': c,
                                     's': s,
                                     'b': b,
                                     'hazard': hazard}

            if i == 0:
                for rv in range(0, len(self.removal_parameters['beta'])):
                    self.beta = self.removal_parameters['beta'][rv]
                    self.cell_type = self.removal_parameters['cell_type'][rv]

                    h, c, s, b = self.solve_system(self.rho0, self.t)
                    hazard = self.hazard_func(self.rho0, self.t)

                    results_name = "_".join([k + "=" + str(eval("self." + k, {"self": self}))
                                             for k in self.parameters.keys()]) \
                                   + "_" + "_".join([k + "=" + str(eval("self." + k, {"self": self}))
                                                     for k in self.removal_parameters.keys()])

                    removal_results[results_name] = {'h': h,
                                                     'c': c,
                                                     's': s,
                                                     'b': b,
                                                     'hazard': hazard}

            self.cell_type = None

        self.results = results

        # for k in results:
        #     file = open(k, 'rb')
        #     data = pickle.load(file)
        #     print(data)
        #     file.close()

        self.removal_results = removal_results


class Plotter:
    def __init__(self, solver):

        self.solver = solver
        self.mu_v_init = 1e-3
        self.t = self.solver.t
        self.t_max = 10000 * self.mu_v_init

    def generate_table(self, r, c, r_headers, c_headers):
        # Figure formatting
        fig = plt.figure(figsize=(25, 25))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        # Set number of columns and rows for the table
        ax = fig.subplots(r, c)

        # Add column and row headings
        for i in range(r):
            rows = ax[i, 0]
            rows.axis("off")
            rows.text(0.25, 0.5, r_headers[i], fontsize=25, family='serif', style='italic')
            if c > 3:
                radars = ax[i, 3]
                radars.axis("off")

        for j in range(c):
            columns = ax[0, j]
            columns.set_title(c_headers[j], fontsize=25, pad=60, family='serif', weight='bold')

        return ax, fig

    def plot_init_system(self, params, ax):

        self.k = params['k'][0]
        self.mu_c = params['mu_c'][0]
        self.mu_v = params['mu_v'][0]
        self.f0 = params['f0'][0]
        self.lambda_v = params['lambda_v'][0]
        self.lambda_c = params['lambda_c'][0]

        results_name = "_".join(
            [k + "=" + str(eval("self." + k, {"self": self})) for k in params.keys()])
        results = self.solver.results[results_name]

        for j in range(len(ax)):
            ax1 = ax[j, 1]
            ax1.plot(self.t * self.mu_v_init, results['h'], 'r-.', alpha=0.4)
            ax1.plot(self.t * self.mu_v_init, results['s'], 'y-.', alpha=0.4)
            ax1.plot(self.t * self.mu_v_init, results['c'], 'c-.', alpha=0.4)
            ax1.plot(self.t * self.mu_v_init, results['b'], 'm-.', alpha=0.4)
            ax1.set_xlim((self.t * self.mu_v_init).min(), self.t_max)
            ax1.set_ylim((results['h']).min(), (results['h']).max())
            ax1.set_yticks([0, 0.5, 1])
            ax1.set_xticks([0, 5, 10])
            ax1.set_xlabel('Time, $\mu_vt$', fontsize=18)
            ax1.set_ylabel('Fraction of cells', fontsize=18)

            ax2 = ax[j, 2]
            ax2.plot(self.t * self.mu_v_init, results['hazard'], 'r-.', alpha=0.4)
            ax2.set_xlim((self.t * self.mu_v_init).min(), self.t_max)
            ax2.set_ylim(0)
            ax2.set_yticks([0, 0.1, 0.2])
            ax2.set_xticks([0, 5, 10])
            ax2.set_xlabel('Time, $\mu_vt$', fontsize=18)
            ax2.set_ylabel('Hazard rate', fontsize=18)

        return ax

    def get_plateau(self, c):

        tol = 0.0001
        mid = (np.max(c) - np.min(c)) / 2
        peak_array = np.where((c > mid) & (c - np.roll(c, 1) < tol) & (c - np.roll(c, 1) >= 0), 1, 0)
        index = np.argmax(peak_array)
        value = c[index]

        return value

    def get_t_half(self, c):

        index, min = self.get_local_min(c)
        max = self.get_plateau(c)
        mid = (max + min) / 2
        c_new = np.where(self.t < self.t[index], 0, c)
        time = np.interp(mid, c_new, self.t)

        return time

    def get_local_min(self, c):

        local_min = np.where((np.roll(c, 1) > c) & (np.roll(c, -1) > c), 1, 0)
        index = np.argmax(local_min)
        value = c[index]

        return index, value

    def get_local_max(self, c):

        local_max = np.where((np.roll(c, 1) < c) & (np.roll(c, -1) < c), 1, 0)
        local_max[0] = 0
        index = np.argmax(local_max)
        value = c[index]

        return index, value

    def make_percentage(self, c):

        percent = np.zeros(len(c) - 1)
        for i in range(1, len(c)):
            percent[i - 1] = (c[i] - c[0]) / c[0] * 100

        return percent

    def get_fingerprint_data(self):

        data_len = 9
        peak_cancer = np.zeros(data_len)
        peak_senescense = np.zeros(data_len)
        time_cancer = np.zeros(data_len)
        time_senescence = np.zeros(data_len)
        min_hazard = np.zeros(data_len)
        time_hazard = np.zeros(data_len)
        peak_hazard = np.zeros(data_len)

        all_c = np.array(
            [self.solver.results[k]['c'] for k in self.solver.results.keys()] + [self.solver.removal_results[k]['c'] for
                                                                                 k in
                                                                                 self.solver.removal_results.keys()])
        all_s = np.array(
            [self.solver.results[k]['s'] for k in self.solver.results.keys()] + [self.solver.removal_results[k]['s'] for
                                                                                 k in
                                                                                 self.solver.removal_results.keys()])
        all_hazard = np.array([self.solver.results[k]['hazard'] for k in self.solver.results.keys()] + [
            self.solver.removal_results[k]['hazard'] for k in self.solver.removal_results.keys()])

        for i in range(data_len):
            peak_cancer[i] = self.get_plateau(all_c[i])
            time_cancer[i] = self.get_t_half(all_c[i])

            index, min_hazard[i] = self.get_local_min(all_hazard[i])
            peak_hazard[i] = self.get_plateau(all_hazard[i])
            time_hazard[i] = self.get_t_half(all_hazard[i])

            peak_senescense[i] = np.amax(all_s[i])
            time_senescence[i] = self.t[np.argmax(all_s[i])]

        i_min, min_hazard[6] = self.get_local_min(all_hazard[6])
        i_max, peak_hazard[6] = self.get_local_max(all_hazard[6])
        mid = (peak_hazard[6] + min_hazard[6]) / 2
        c_new = np.where(self.t < self.t[i_min], min_hazard[6], all_hazard[6])
        c_new = np.where(self.t > self.t[i_max], peak_hazard[6], c_new)
        time_hazard[6] = np.interp(mid, c_new, self.t)

        self.peak_cancer_percent = -(self.make_percentage(peak_cancer))
        self.time_cancer_percent = self.make_percentage(time_cancer)
        self.peak_senescense_percent = -(self.make_percentage(peak_senescense))
        self.time_senescence_percent = self.make_percentage(time_senescence)
        self.min_hazard_percent = -(self.make_percentage(min_hazard))
        self.time_hazard_percent = self.make_percentage(time_hazard)
        self.peak_hazard_percent = -(self.make_percentage(peak_hazard))

    def plot_fingerprints(self, ax, stats, labels):

        init_stats = np.zeros(len(stats) + 1)
        angles0 = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        # close the plot
        stats = np.concatenate((stats, [stats[0]]))
        angles = np.concatenate((angles0, [angles0[0]]))

        ax.plot(angles, init_stats, 'o-', linewidth=2, alpha=0.4)
        ax.fill(angles, init_stats, alpha=0.15)
        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.50)
        ax.set_thetagrids(angles0 * 180 / np.pi, labels, fontsize=16)
        ax.set_rlim(-100, 100)
        ax.set_rticks([-100, -50, 0, 50, 100])
        ax.grid(True)

        return ax

    def run(self):

        params = self.solver.parameters
        removal_params = self.solver.removal_parameters
        self.get_fingerprint_data()
        data = np.array([self.peak_cancer_percent, self.time_cancer_percent, self.peak_senescense_percent,
                         self.time_senescence_percent, self.min_hazard_percent, self.time_hazard_percent,
                         self.peak_hazard_percent])
        labels = np.array(['decr. \n peak cancer', 'incr. time to \n cancer peak',
                           'decr. \n peak senescence', 'incr. time to \n peak senescence', 'decr. \n hazard min.',
                           'incr. time to \n hazard min', 'decr. \n hazard peak'])

        # Initial system plot
        # Define parameters
        self.k = params['k'][0]
        self.mu_c = params['mu_c'][0]
        self.mu_v = params['mu_v'][0]
        self.f0 = params['f0'][0]
        self.lambda_v = params['lambda_v'][0]
        self.lambda_c = params['lambda_c'][0]

        results_name = "_".join(
            [k + "=" + str(eval("self." + k, {"self": self})) for k in params.keys()])
        results = self.solver.results[results_name]

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.subplots(1, 1)

        ax1.plot(self.t * self.mu_v_init, results['h'], 'r', label='Healthy', linewidth=3)
        ax1.plot(self.t * self.mu_v_init, results['s'], 'y', label='Senescent', linewidth=3)
        ax1.plot(self.t * self.mu_v_init, results['c'], 'c', label='Cancerous', linewidth=3)
        ax1.plot(self.t * self.mu_v_init, results['b'], 'm', label='Both', linewidth=3)
        ax1.set_xlim((self.t * self.mu_v_init).min(), self.t_max)
        ax1.set_ylim((results['h']).min(), (results['h']).max())
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_xticks([0, 5, 10])
        ax1.set_xlabel('Time, $\mu_vt$', fontsize=18)
        ax1.set_ylabel('Fraction of cells', fontsize=18)
        ax1.legend(fontsize=18, loc='center right')

        plt.savefig('PresentationPlots/initial_evolution.png')
        plt.show()

        # # Plot table 1
        # self.r1 = 4
        # self.c1 = 4
        # self.r1_headers = ['Reducing $\mu_v$', 'Reducing $\mu_c$', 'Reducing $f_0$', 'Reducing $k$']
        # self.c1_headers = ['Intervention', '(I) Survival Function \n and Populations', '(II) Hazard Function',
        #                    '(III) Radar Plots']
        # ax1, table1 = self.generate_table(self.r1, self.c1, self.r1_headers, self.c1_headers)
        # ax1 = self.plot_init_system(params, ax1)
        #
        # for i in range(1, len(ax1) + 1):
        #
        #     self.k = params['k'][i]
        #     self.mu_c = params['mu_c'][i]
        #     self.mu_v = params['mu_v'][i]
        #     self.f0 = params['f0'][i]
        #     self.lambda_v = params['lambda_v'][i]
        #     self.lambda_c = params['lambda_c'][i]
        #
        #     results_name = "_".join(
        #         [k + "=" + str(eval("self." + k, {"self": self})) for k in params.keys()])
        #     results = self.solver.results[results_name]
        #
        #     pops = ax1[i - 1, 1]
        #     pops.plot(self.t * self.mu_v_init, results['h'], 'r', label='Healthy')
        #     pops.plot(self.t * self.mu_v_init, results['s'], 'y', label='Senescent')
        #     pops.plot(self.t * self.mu_v_init, results['c'], 'c', label='Cancerous')
        #     pops.plot(self.t * self.mu_v_init, results['b'], 'm', label='Both')
        #
        #     haz = ax1[i - 1, 2]
        #     haz.plot(self.t * self.mu_v_init, results['hazard'], 'r')
        #
        #     if i == 1:
        #         pops.legend(frameon=False, loc='center right', fontsize=18)
        #
        # for i in range(4):
        #     radar = table1.add_subplot(4, 4, 4 * (i + 1), polar=True)
        #     self.plot_fingerprints(radar, data[:, i], labels)
        #
        # table1.savefig('PresentationPlots/table1_plot.png')
        # table1.show()
        #
        # # Plot table 2
        # self.r2 = 4
        # self.c2 = 4
        # self.r2_headers = ['Reverting $\mu_v$ \n at rate $\lambda_v$', 'Reverting $\mu_c$ \n at rate $\lambda_c$',
        #                    'Removing \n cancerous cells', 'Removing \n senescent cells']
        # self.c2_headers = ['Intervention', '(I) Survival Function \n and Populations', '(II) Hazard Function',
        #                    '(III) Radar Plots']
        # ax2, table2 = self.generate_table(self.r2, self.c2, self.r2_headers, self.c2_headers)
        # ax2 = self.plot_init_system(params, ax2)
        #
        # for i in range(len(ax2) + 1, len(ax2) + 3):
        #
        #     self.k = params['k'][i]
        #     self.mu_c = params['mu_c'][i]
        #     self.mu_v = params['mu_v'][i]
        #     self.f0 = params['f0'][i]
        #     self.lambda_v = params['lambda_v'][i]
        #     self.lambda_c = params['lambda_c'][i]
        #
        #     results_name = "_".join(
        #         [k + "=" + str(eval("self." + k, {"self": self})) for k in params.keys()])
        #     results = self.solver.results[results_name]
        #
        #     pops = ax2[i - (len(ax2) + 1), 1]
        #     pops.plot(self.t * self.mu_v_init, results['h'], 'r', label='Healthy')
        #     pops.plot(self.t * self.mu_v_init, results['s'], 'y', label='Senescent')
        #     pops.plot(self.t * self.mu_v_init, results['c'], 'c', label='Cancerous')
        #     pops.plot(self.t * self.mu_v_init, results['b'], 'm', label='Both')
        #
        #     haz = ax2[i - (len(ax2) + 1), 2]
        #     haz.plot(self.t * self.mu_v_init, results['hazard'], 'r')
        #
        #     if i - (len(ax2) + 1) == 0:
        #         pops.legend(frameon=False, loc='center right', fontsize=18)
        #
        # for rv in range(0, len(removal_params['beta'])):
        #     self.k = params['k'][0]
        #     self.mu_c = params['mu_c'][0]
        #     self.mu_v = params['mu_v'][0]
        #     self.f0 = params['f0'][0]
        #     self.lambda_v = params['lambda_v'][0]
        #     self.lambda_c = params['lambda_c'][0]
        #     self.beta = removal_params['beta'][rv]
        #     self.cell_type = removal_params['cell_type'][rv]
        #
        #     results_name = "_".join([k + "=" + str(eval("self." + k, {"self": self}))
        #                              for k in params.keys()]) \
        #                    + "_" + "_".join([k + "=" + str(eval("self." + k, {"self": self}))
        #                                      for k in removal_params.keys()])
        #
        #     removal_results = self.solver.removal_results[results_name]
        #
        #     pops = ax2[rv + 2, 1]
        #     pops.plot(self.t * self.mu_v_init, removal_results['h'], 'r', label='Healthy')
        #     pops.plot(self.t * self.mu_v_init, removal_results['s'], 'y', label='Senescent')
        #     pops.plot(self.t * self.mu_v_init, removal_results['c'], 'c', label='Cancerous')
        #     pops.plot(self.t * self.mu_v_init, removal_results['b'], 'm', label='Both')
        #
        #     haz = ax2[rv + 2, 2]
        #     haz.plot(self.t * self.mu_v_init, removal_results['hazard'], 'r')
        #
        # for i in range(4):
        #     radar = table2.add_subplot(4, 4, 4 * (i + 1), polar=True)
        #     self.plot_fingerprints(radar, data[:, i + 4], labels)
        #
        # table2.savefig('PresentationPlots/table2_plot.png')
        # table2.show()


class QLearning:
    def __init__(self, solver, plotter):

        self.solver = solver
        self.v = solver.v
        self.c = solver.c
        self.rho0 = solver.rho0
        self.params = self.solver.parameters
        self.removal_params = self.solver.removal_parameters

        # Initialise parameter for ode solver
        self.solver.k = self.params['k'][0]
        self.solver.mu_c = self.params['mu_c'][0]
        self.solver.mu_v = self.params['mu_v'][0]
        self.solver.f0 = self.params['f0'][0]
        self.solver.lambda_v = self.params['lambda_v'][0]
        self.solver.lambda_c = self.params['lambda_c'][0]

        self.learning_rate = 0.1
        self.p_explore = 0.1
        self.discount = 0.9998

        self.T = 10000
        self.dt = 100
        self.N = int(self.T / self.dt)
        self.t_vec = np.arange(0, self.T + self.dt, self.dt)

        self.plotter = plotter

    def q_learning_action(self, s, q_matrix):
        rand_draw = random.uniform(0, 1)
        if rand_draw < self.p_explore:
            action_idx = random.randint(0, q_matrix.shape[1] - 1)
        else:
            action_idx = np.argmax(q_matrix[s, :])
        return action_idx

    def q_learning_update(self, s, a, reward, s2, q_matrix):
        q_matrix[s, a] = (1 - self.learning_rate) * q_matrix[s, a] + self.learning_rate * (
                reward + self.discount * np.amax(q_matrix[s2, :], axis=-1))
        s = s2
        return s, q_matrix

    def exp_decay(self, lambda_val, t):
        return math.exp(-lambda_val * t)

    def ode_solver(self, state, action, alpha, t):

        # Time array from previous to new state
        time_array = np.array([t, t + self.dt])
        self.solver.beta = action

        # Use remove_cancer() function from Solver to solve ODE over timestep dt and retrieve new state
        h, c, s, b = self.solver.solve_system(state, time_array)
        new_state = np.array([b[1], c[1], s[1], h[1]])

        # Calculate reward
        new_vitality_state = h[1]
        reward = new_vitality_state - alpha * action

        # Increment time
        t += self.dt

        # Determines if episode is over
        if t == self.T:
            done = True
        else:
            done = False

        return new_state, new_vitality_state, reward, t, done

    def generate_q_matrix(self, num_episodes, actions, alpha):

        # Rate of decay for learning rate and p_explore
        decay = np.array([0.0005, 0.0005])

        # Initialize states
        states = self.rho0
        vitality_states = np.zeros(self.N+1)
        vitality_states[0] = states[-1]

        # Build initial Q-matrix filled with zeros
        q_matrix = np.zeros([len(vitality_states), len(actions)])

        # Initialise lists that will hold information about Q-learning convergence
        # rewards = []
        # errors = []
        matrix_diffs = []

        for i in range(num_episodes):
            done = False
            t = 0
            state_idx = 0
            states = self.rho0

            reward_tot = 0
            error_tot = 0
            prev_q_matrix = q_matrix.copy()

            while not done:
                action_idx = self.q_learning_action(state_idx, q_matrix)  # get best action for state
                action = actions[action_idx]
                new_states, new_vitality, reward, t, done = self.ode_solver(states, action, alpha,
                                                                           t)  # test environment
                new_state_idx = state_idx + 1  # get new state index
                vitality_states[new_state_idx] = new_vitality
                # Update state and Q matrix
                state_idx, q_matrix = self.q_learning_update(state_idx, action_idx, reward, new_state_idx,
                                                             q_matrix)
                states = new_states

            # Decay learning rate and exploration
            self.p_explore = self.exp_decay(decay[0], i)
            self.learning_rate = self.exp_decay(decay[1], i)

            # # Run inference step for learnt actions every 100 episode
            # if i % 100 == 0:
            #     t = 0
            #     states = self.rho0
            #     action_vec = self.get_best_action(q_matrix, actions)
            #     state_idx = 0
            #     for action in action_vec:
            #         new_states, new_vitality, reward, t, done = self.ode_solver(states, action, alpha,
            #                                                                     t)  # test environment
            #         states = new_states
            #         reward_disc = (self.discount**t) * reward
            #         reward_tot += reward_disc
            #         new_state_idx = state_idx + 1
            #         if new_state_idx < 101:
            #             error = (reward + self.discount * np.amax(q_matrix[new_state_idx, :], axis=-1))
            #             error_tot += error
            #             state_idx = new_state_idx
            #
            #     rewards.append(reward_tot)
            #     errors.append(error_tot)

            # Compute differences in Q-matrix values
            matrix_diff = q_matrix.flatten() - prev_q_matrix.flatten()
            matrix_diffs.append(matrix_diff)

        return q_matrix, vitality_states, matrix_diffs

    def get_best_action(self, q_matrix, actions):

        action_vec = []

        for t in range(len(self.t_vec)):
            state_idx = t
            best_action_idx = np.argmax(q_matrix[state_idx, :])
            best_action = actions[best_action_idx]
            action_vec.append(best_action)

        return action_vec

    def plot_learned_protocol(self, q_matrix, actions):

        # Plot learned repair
        removal_vecs = []
        removal_vec = self.get_best_action(q_matrix, actions)
        removal_vecs.append(removal_vec)

        # Average with mean or mode method
        repair_mat = np.vstack(removal_vecs)
        removal_vec = []
        for n in range(repair_mat.shape[1]):
            removal_vec.append(stats.mode(repair_mat[:, n])[0])

        time_vec = np.arange(0, len(removal_vec))  # get time vec

        plt.scatter(time_vec.tolist(), removal_vec, color='#000080', s=7, alpha=0.7)
        plt.xlabel('Time, $t$', fontsize=32)
        plt.ylabel('Repair, $r(t)$', fontsize=32)
        plt.xlim(0, self.T)
        plt.ylim(-0.001, None)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        plt.show()

    def plot_init_vitality(self, rho, axes):

        self.solver.k = self.params['k'][0]
        self.solver.mu_c = self.params['mu_c'][0]
        self.solver.mu_v = self.params['mu_v'][0]
        self.solver.f0 = self.params['f0'][0]
        self.solver.lambda_v = self.params['lambda_v'][0]
        self.solver.lambda_c = self.params['lambda_c'][0]

        h, c, s, b = self.solver.solve_system(rho, self.t_vec)
        states = np.array([b, c, s, h])
        vitality = np.zeros(len(self.t_vec))
        for i in range(len(self.t_vec)):
            vitality[i] = states[-1, i]

        for j in range(len(axes)):
            ax1 = axes[j, 1]
            ax1.plot(self.t_vec * self.solver.mu_v, vitality, 'b-.', alpha=0.4)
            ax1.set_xlim((self.t_vec * self.solver.mu_v).min(), (self.t_vec * self.solver.mu_v).max())
            ax1.set_ylim(vitality.min(), vitality.max())
            ax1.set_yticks([0, 0.5, 1])
            ax1.set_xticks([0, 5, 10])
            ax1.set_xlabel('Time, $\mu_vt$', fontsize=18)
            ax1.set_ylabel('Vitality, $\phi$', fontsize=18)

        return axes

    def run(self):

        # Plot learned protocol for cancer cell removal
        self.solver.cell_type = 'cancerous'

        # Define learning parameters
        N_ep = 15000
        beta = 0.007
        actions = [0, beta]
        alpha = 20

        # start_time = time.time()
        #
        # for i in tqdm(range(10)):
        #     # Generate Q matrix for these parameters
        #     q_matrix_cancer, vitalities_c, diffs = self.generate_q_matrix(N_ep, actions, alpha)
        #
        #     # Save Q_matrix data and retrieve it for plotting
        #     np.savetxt('Data/q_matrix_'+str(i)+'.txt', q_matrix_cancer)
        #     np.savetxt('Data/vitalities_'+str(i)+'txt', vitalities_c)
        #     np.savetxt('Data/diff_'+str(i)+'.txt', diffs)
        #
        alphas = np.linspace(1, 30, 50)
        betas = np.linspace(0.002, 0.02, 50)
        betas = betas[33:]
        print(betas)

        # for a in tqdm(alphas):
        #     # Generate Q matrix for these parameters
        #     q_matrix_cancer, vitalities_c, diffs = self.generate_q_matrix(N_ep, actions, a)
        #
        #     # Save Q_matrix data and retrieve it for plotting
        #     np.savetxt('Data/q_matrix_alpha' + str(a) + '.txt', q_matrix_cancer)
        #     np.savetxt('Data/vitalities_alpha' + str(a) + 'txt', vitalities_c)
        #     np.savetxt('Data/diff_alpha' + str(a) + '.txt', diffs)

        for b in tqdm(betas):
            beta = b
            actions = [0, beta]

            # Generate Q matrix for these parameters
            q_matrix_cancer, vitalities_c, diffs = self.generate_q_matrix(N_ep, actions, alpha)

            # Save Q_matrix data and retrieve it for plotting
            np.savetxt('Data/q_matrix_beta' + str(b) + '.txt', q_matrix_cancer)
            np.savetxt('Data/vitalities_beta' + str(b) + 'txt', vitalities_c)
            np.savetxt('Data/diff_beta' + str(b) + '.txt', diffs)


        # q_matrix_cancer = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/q_matrix_test.txt')
        # rewards_c = np.genfromtxt('Data/cancer_reward_qlearning.txt')
        #
        # plt.scatter(range(N), rewards_c)
        # plt.show()

        # Plot learned protocol
        # action_vec = self.get_best_action(q_matrix_cancer, actions)

        # plt.plot(self.t_vec * self.solver.mu_v, action_vec, 'g', linewidth=2.5)
        # plt.title(r'$\alpha$ =' + str(alpha))
        # plt.xlabel('Time, $t$', fontsize=18)
        # plt.ylabel(r'$\beta$', fontsize=18)
        # plt.xlim((self.t_vec * self.solver.mu_v).min(), (self.t_vec * self.solver.mu_v).max())
        # plt.yticks([min(action_vec), (max(action_vec) + min(action_vec)) / 2,
        #                     max(action_vec)])
        # plt.xticks([0, 5, 10])
        # # plt.savefig(r'$\alpha$ =' + str(alpha) + '.png')
        # plt.savefig(str(N) + '_episodes.png')
        # plt.show()

        # Plot learned protocol for senescent cell removal
        # self.solver.cell_type = 'senescent'
        #
        # # Define learning parameters
        # beta = 0.009
        # actions = [0, beta]
        # alpha = 25

        # # Generate Q matrix for these parameters
        # q_matrix_senescence, vitalities_s, rewards_s = self.generate_q_matrix(N, actions, alpha)
        # # Save Q_matrix data and retrieve it for plotting
        # np.savetxt('Data/senescence_q_matrix.txt', np.array(q_matrix_senescence))
        # np.savetxt('Data/senescence_reward_qlearning.txt', rewards_s)
        # np.savetxt('Data/senescence_vitalities.txt', vitalities_s)
        #
        # q_matrix_senescence = np.genfromtxt('Data/senescence_q_matrix.txt')
        # # Plot learned protocol
        # self.plot_learned_protocol(q_matrix_senescence, actions)

        # beta = 0.01
        # actions = np.array([0, beta])
        #
        # # alphas = np.linspace(1, 150, 10)
        # alpha = 15
        #
        # num_episodes = 15000
        # # i = 10000
        #
        # # for i in range(500, num_episodes+500, 500):
        # # for alpha in alphas:
        # q_matrix, vitality_states = self.q_matrix(num_episodes, actions, alpha)
        # action_vec = self.get_best_action(q_matrix, actions)
        #
        # plt.plot(self.t_vec * self.solver.mu_v, action_vec, 'g', linewidth=2.5)
        # plt.title(r'$\alpha$ =' + str(alpha))
        # plt.xlabel('Time, $t$', fontsize=18)
        # plt.ylabel(r'$\beta$', fontsize=18)
        # plt.xlim((self.t_vec * self.solver.mu_v).min(), (self.t_vec * self.solver.mu_v).max())
        # plt.yticks([min(action_vec), (max(action_vec) + min(action_vec)) / 2,
        #                     max(action_vec)])
        # plt.xticks([0, 5, 10])
        # # plt.savefig(r'$\alpha$ =' + str(alpha) + '.png')
        # plt.savefig(str(num_episodes) + '_episodes.png')
        # plt.show()

        # # alphas = np.array([1, 10, 100, 1000])
        #
        # symbols = ['$\mu_v$', '$\mu_c$', '$f_0$', '$k$', '$\lambda_v$', '$\lambda_c$', r'$\beta_0$', r'$\beta_0$']
        # filenames = ['Reducing mu_v', 'Reducing mu_c', 'Reducing f0', 'Reducing k', 'Reverting mu_v', 'Reverting mu_c',
        #              'Removing cancer', 'Removing senescence']
        # intial_list = [0, 0.001, 1e-5, 0.004, 0.5, 0, 0]
        # actions_list = [0, 0.0001, 1e-7, 0.0025, 0.3, 0.001, 0.0005]
        #
        # # # # alphas = [500, 10000, 10000000, 400, 3, 1000, 2000]
        # # alphas = [100, 20, 10000, 50, 10, 50, 100]
        #
        #
        # # for a, alpha in enumerate(alphas):
        #
        # action_vecs = []
        # vitalities = []
        #
        # for i in range(len(actions_list)):
        #     if i == 0:
        #         for rv in range(0, len(self.removal_params['beta'])):
        #             self.solver.beta = self.removal_params['beta'][rv]
        #             self.solver.cell_type = self.removal_params['cell_type'][rv]
        #             action = self.solver.beta
        #             actions = np.array([0, action])
        #
        #             if action == 0.005:
        #                 alpha = 200
        #             else:
        #                 alpha = alphas[i]
        #
        #             q_matrix, vitality_states = self.q_matrix(actions, alpha)
        #             action_vec = self.get_best_action(q_matrix, actions)
        #             action_vecs.append(action_vec)
        #             vitalities.append(vitality_states)
        #
        #     else:
        #         self.solver.cell_type = None
        #         action = actions_list[i]
        #         intial = intial_list[i]
        #         actions = np.array([intial, action])
        #         alpha = alphas[i]
        #
        #         # Set correct parameters for action taken
        #         self.solver.k = self.params['k'][i]
        #         self.solver.mu_c = self.params['mu_c'][i]
        #         self.solver.mu_v = self.params['mu_v'][i]
        #         self.solver.f0 = self.params['f0'][i]
        #         self.solver.lambda_v = self.params['lambda_v'][i]
        #         self.solver.lambda_c = self.params['lambda_c'][i]
        #
        #         q_matrix, vitality_states = self.q_matrix(actions, alpha)
        #         action_vec = self.get_best_action(q_matrix, actions)
        #         action_vecs.append(action_vec)
        #         vitalities.append(vitality_states)
        #
        # action_vecs.append(action_vecs.pop(0))
        # action_vecs.append(action_vecs.pop(0))
        # vitalities.append(vitalities.pop(0))
        # vitalities.append(vitalities.pop(0))
        #
        # # # Plot table 1
        # r1 = 4
        # c1 = 3
        # r1_headers = ['Reducing $\mu_v$', 'Reducing $\mu_c$', 'Reducing $f_0$', 'Reducing $k$']
        # c1_headers = ['Intervention', '(I) Vitality', '(II) Optimal action']
        # ax1, table1 = self.plotter.generate_table(r1, c1, r1_headers, c1_headers)
        # ax1 = self.plot_init_vitality(self.rho0, ax1)
        #
        # for i in range(r1):
        #     vit = ax1[i, 1]
        #     vit.plot(self.t_vec * self.solver.mu_v, vitalities[i], 'g', linewidth=2.5)
        #
        #     control = ax1[i, 2]
        #     control.plot(self.t_vec * self.solver.mu_v, action_vecs[i], 'g', linewidth=2.5)
        #     control.set_xlabel('Time, $t$', fontsize=18)
        #     control.set_ylabel(symbols[i], fontsize=18)
        #     # control.set_ylim([0 - max(action_vecs[i]) * 0.05, max(action_vecs[i]) * 1.05])
        #     control.set_xlim((self.t_vec * self.solver.mu_v).min(), (self.t_vec * self.solver.mu_v).max())
        #     control.set_yticks(
        #         [min(action_vecs[i]), (max(action_vecs[i]) + min(action_vecs[i])) / 2, max(action_vecs[i])])
        #     control.set_xticks([0, 5, 10])
        #
        # # table1.savefig('table1_qlearning.png')
        # table1.show()
        #
        # # Plot table 2
        # r2 = 4
        # c2 = 3
        # r2_headers = ['Reverting $\mu_v$ \n at rate $\lambda_v$', 'Reverting $\mu_c$ \n at rate $\lambda_c$',
        #               'Removing \n cancerous cells', 'Removing \n senescent cells']
        # c2_headers = ['Intervention', '(I) Vitality', '(II) Optimal action']
        # ax2, table2 = self.plotter.generate_table(r2, c2, r2_headers, c2_headers)
        # ax2 = self.plot_init_vitality(self.rho0, ax2)
        #
        # for i in range(r2):
        #     vit = ax2[i, 1]
        #     vit.plot(self.t_vec * self.solver.mu_v, vitalities[i + r1], 'g', linewidth=2.5)
        #
        #     control = ax2[i, 2]
        #     control.plot(self.t_vec * self.solver.mu_v, action_vecs[i + r1], 'g', linewidth=2.5)
        #     control.set_xlabel('Time, $t$', fontsize=18)
        #     control.set_ylabel(symbols[i + r1], fontsize=18)
        #     # control.set_ylim([0 - max(action_vecs[i+r1]) * 0.05, max(action_vecs[i+r1]) * 1.05])
        #     control.set_xlim((self.t_vec * self.solver.mu_v).min(), (self.t_vec * self.solver.mu_v).max())
        #     control.set_yticks([min(action_vecs[i + r1]), (max(action_vecs[i + r1]) + min(action_vecs[i + r1])) / 2,
        #                         max(action_vecs[i + r1])])
        #     control.set_xticks([0, 5, 10])
        #
        # # table2.savefig('table2_qlearning.png')
        # table2.show()


class Control:
    def __init__(self, solver, qlearning):
        self.solver = solver
        self.v = solver.v
        self.c = solver.c
        self.rho0 = solver.rho0
        self.params = self.solver.parameters
        self.removal_params = self.solver.removal_parameters

        # Initialise parameter for ode solver
        self.solver.k = self.params['k'][0]
        self.solver.mu_c = self.params['mu_c'][0]
        self.solver.mu_v = self.params['mu_v'][0]
        self.solver.f0 = self.params['f0'][0]
        self.solver.lambda_v = self.params['lambda_v'][0]
        self.solver.lambda_c = self.params['lambda_c'][0]

        self.T = 10000
        self.dt = 100
        self.N = int(self.T / self.dt)
        self.t_vec = np.arange(0, self.T + self.dt, self.dt)

        self.qlearning = qlearning

        # Define parameters for optimization
        self.alphas = np.linspace(1, 30, 50)
        self.betas = np.linspace(0.002, 0.02, 50)
        self.highres_step = self.dt / 10
        self.T_std = 5 * self.dt

        self.T1 = np.arange(100, 4000, self.T_std)
        self.T2 = np.arange(2000, 10000, self.T_std)

    def cost_calcuator(self, init_state, actions, alpha, time_array, i1, i2):

        total_cost = 0

        if i1 > 0:
            time_off1 = time_array[:i1]
            self.solver.beta = actions[0]
            h, c, s, b = self.solver.solve_system(init_state, time_off1)
            states = np.array([b, c, s, h])
            vitalities = states[-1, :]
            init_state = states[:, -1]
            gamma = -np.log(self.qlearning.discount)
            costs = np.exp(-gamma * time_off1) * (alpha * actions[0] - vitalities)
            cost = np.sum(costs)
            total_cost += cost

        time_on = time_array[i1:i2]
        time_off2 = time_array[i2:]

        self.solver.beta = actions[1]
        h, c, s, b = self.solver.solve_system(init_state, time_on)
        states = np.array([b, c, s, h])
        vitalities = states[-1, :]
        init_state = states[:, -1]
        gamma = -np.log(self.qlearning.discount)
        costs = np.exp(-gamma * time_on) * (alpha * actions[1] - vitalities)
        cost = np.sum(costs)
        total_cost += cost

        self.solver.beta = actions[0]
        h, c, s, b = self.solver.solve_system(init_state, time_off2)
        states = np.array([b, c, s, h])
        vitalities = states[-1, :]
        gamma = -np.log(self.qlearning.discount)
        costs = np.exp(-gamma * time_off2) * (alpha * actions[0] - vitalities)
        cost = np.sum(costs)
        total_cost += cost

        return total_cost

    def optimize_switch_times_v2(self, T1_array, T2_array, actions, alpha, time_array, highres_step):

        T1_results = []
        T2_results = []
        cost_results = []

        for t1 in T1_array:
            for t2 in T2_array:
                if t2 > t1:
                    # Create new list of switching times respecting the condition that T1 < T2
                    T1_results.append(t1)
                    T2_results.append(t2)
                    # Find index of time array where action is switched on and off
                    action_array = np.where((t1 <= time_array) & (time_array < t2), 1, 0)
                    i1 = np.argmax(action_array)
                    i2 = np.argmin(action_array[i1:]) + i1
                    # Calculate cost for specific protocol
                    cost = self.cost_calcuator(self.rho0, actions, alpha, time_array, i1, i2)
                    cost_results.append(cost)

        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]

        # HIGH-RES STEP
        cost_results = []
        T1_results = []
        T2_results = []

        # Loop with higher resolution around current min +- T_std
        time_array = np.arange(0, 10000, highres_step)
        T1_list = np.arange(minT1 - self.dt, minT1 + self.dt, highres_step)
        T2_list = np.arange(minT2 - self.dt, minT2 + self.dt, highres_step)

        for t1 in T1_list:
            if t1 > 0:
                for t2 in T2_list:
                    if t2 > t1:
                        T1_results.append(t1)
                        T2_results.append(t2)
                        action_array = np.where((t1 <= time_array) & (time_array < t2), 1, 0)
                        i1 = np.argmax(action_array)
                        i2 = np.argmin(action_array[i1:]) + i1
                        cost = self.cost_calcuator(self.rho0, actions, alpha, time_array, i1, i2)
                        cost_results.append(cost)

        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]

        return minT1, minT2, min_cost

    def run(self):

        # Create figure to store plots for cancer and senescent cell removal
        # fig = plt.figure(figsize=(15,10))
        # # fig, ax = fig.subplots(2, 3, sharex=True)
        #
        # # # EXEMPLAR PLOTS FOR PROTOCOLS #################################################################################
        # # # Cancer protocol
        # t1_c = 2130
        # t2_c = 7510
        # y_c = np.piecewise(self.t_vec, [(self.t_vec <= t1_c) & (self.t_vec >= t2_c),
        #                               (t1_c <= self.t_vec) & (self.t_vec <= t2_c)], [0, 1])
        #
        # ax1 = fig.add_subplot(111)
        # ax1.plot(self.t_vec, y_c, linewidth=3, color='blue')
        # ax1.fill_between(self.t_vec, y_c, alpha=0.3, color='blue')
        # # ax1.set_title('Protocol for Cancer Cell Removal')
        # ax1.set_ylabel('Rate of Cell Removal')
        # ax1.set_yticks([0, 1])
        # ax1.set_yticklabels([0, r'$\beta_c$'])
        # ax1.set_xticks([t1_c, t2_c])
        # ax1.set_xticklabels(['$T_1$', '$T_2$'])
        # ax1.set_xlim(0)
        # ax1.set_ylim(0, 2)
        # ax1.spines['right'].set_visible(False)
        # ax1.spines['top'].set_visible(False)
        # plt.show()
        #
        # # Senescence protocol
        # t1_s = 100
        # t2_s = 2500
        # y_s = np.piecewise(self.t_vec, [(self.t_vec <= t1_s) & (self.t_vec >= t2_s),
        #                                 (t1_s <= self.t_vec) & (self.t_vec <= t2_s)], [0, 1])
        #
        # ax2 = fig.add_subplot(222)
        # ax2.plot(self.t_vec, y_s, linewidth=3, color='orange')
        # ax2.fill_between(self.t_vec, y_s, alpha=0.3, color='orange')
        # # ax2.set_title('Protocol for Senescent Cell Removal')
        # ax2.set_yticks([0, 1])
        # ax2.set_ylabel('Rate of Cell Removal')
        # ax2.set_yticklabels([0, r'$\beta_c$'])
        # ax2.set_xticks([t1_s, t2_s])
        # ax2.set_xticklabels(['$T_1$', '$T_2$'])
        # ax2.set_xlim(0)
        # ax2.set_ylim(0, 2)
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['top'].set_visible(False)

        ################################################################################################################
        # OPTIMAL SWITCHING TIMES FOR CANCER CELL REMOVAL
        ################################################################################################################
        self.solver.cell_type = 'cancerous'

        # Find optimal switching times for fixed beta and varying alphas
        # Define fixed beta
        beta = 0.007
        actions = [0, beta]

        # Initialise lists to store times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []

        for a in tqdm(self.alphas):
            minT1, minT2, min_cost = self.optimize_switch_times_v2(self.T1, self.T2, actions, a, self.t_vec, self.highres_step)

            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/alphas_cancer_t1.txt', best_T1)
        np.savetxt('Data/alphas_cancer_t2.txt', best_T2)

        # # Retrieve data from results folder and plot results
        # best_T1 = np.genfromtxt('Data/alphas_cancer_t1.txt')
        # best_T2 = np.genfromtxt('Data/alphas_cancer_t2.txt')
        # ax3 = fig.add_subplot(223)
        # ax3.plot(best_T1, self.alphas, color='blue', linewidth=3)
        # ax3.plot(best_T2, self.alphas, color='blue', linewidth=3)
        # ax3.set_xlim(0, 10000)
        # ax3.set_ylim(min(self.alphas))
        # ax3.set_yticks([min(self.alphas), max(self.alphas)])
        # ax3.set_xticks([0, 5000, 10000])
        # ax3.set_ylabel(r'Cost of Intervention, $\alpha$')
        # ax3.set_xlabel(r'Time, $t$')
        # ax3.set_title(r'Cancer cell removal at fixed $\beta = $' + str(beta))
        # ax3.fill_betweenx(self.alphas, best_T1, best_T2, alpha=0.3, color='blue')

        ################################################################################################################
        # Find optimal switching times for fixed alpha and varying betas
        # Define fixed alpha
        alpha = 20

        # Re-initialise lists to store switching times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []
        print(len(self.betas))

        for beta in tqdm(self.betas):
            # beta = 0.007
            actions = [0, beta]
            minT1, minT2, min_cost = self.optimize_switch_times_v2(self.T1, self.T2, actions, alpha, self.t_vec,
                                                                   self.highres_step)
            # print(minT1, minT2)

            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/betas_cancer_t1.txt', best_T1)
        np.savetxt('Data/betas_cancer_t2.txt', best_T2)

        # # Retrieve data from results folder and plot results
        # best_T1 = np.genfromtxt('Data/betas_cancer_t1.txt')
        # best_T2 = np.genfromtxt('Data/betas_cancer_t2.txt')
        # ax4 = fig.add_subplot(325)
        # ax4.plot(best_T1, self.betas, color='blue')
        # ax4.plot(best_T2, self.betas, color='blue')
        # ax4.set_ylim(min(self.betas))
        # ax4.set_xlim(0, 10000)
        # ax4.set_yticks([min(self.betas), max(self.betas)])
        # ax4.set_xticks([0, 5000, 10000])
        # ax4.set_ylabel(r'Rate of Cell Removal, $\beta$')
        # ax4.set_xlabel(r'Time, $t$')
        # ax4.set_title(r'Cancer cell removal at fixed $\alpha = $' + str(alpha))
        # ax4.fill_betweenx(self.betas, best_T1, best_T2, alpha=0.3, color='blue')

        # # print(cost_results)
        #
        # # gen1 = plt.axes(projection='3d')
        # # gen1.plot_trisurf(T1_results, T2_results, cost_results, cmap='viridis', edgecolor='none')
        # # gen1.set_xlabel('T1', fontweight='bold')
        # # gen1.set_ylabel('T2', fontweight='bold')
        # # gen1.set_zlabel('Cost', fontweight='bold')
        # # gen1.view_init(azim=0, elev=90)
        # # plt.savefig('minimize_cost_gen1.png')
        # # plt.show()
        # #
        # # gen2 = plt.axes(projection='3d')
        # # gen2.plot_trisurf(T1_results, T2_results, cost_results, cmap='viridis', edgecolor='none')
        # # gen2.set_xlabel('T1', fontweight='bold')
        # # gen2.set_ylabel('T2', fontweight='bold')
        # # gen2.set_zlabel('Cost', fontweight='bold')
        # # plt.savefig('minimize_cost_gen2.png')
        # # plt.show()
        #
        # # spec1 = plt.axes(projection='3d')
        # # spec1.plot_trisurf(T1_results, T2_results, cost_results, cmap='viridis', edgecolor='none')
        # # spec1.set_xlabel('T1', fontweight='bold')
        # # spec1.set_ylabel('T2', fontweight='bold')
        # # spec1.set_zlabel('Cost', fontweight='bold')
        # # spec1.view_init(azim=0, elev=90)
        # # plt.savefig('minimize_cost_spec1.png')
        # # plt.show()
        # #
        # # spec2 = plt.axes(projection='3d')
        # # spec2.plot_trisurf(T1_results, T2_results, cost_results, cmap='viridis', edgecolor='none')
        # # spec2.set_xlabel('T1', fontweight='bold')
        # # spec2.set_ylabel('T2', fontweight='bold')
        # # spec2.set_zlabel('Cost', fontweight='bold')
        # # plt.savefig('minimize_cost_spec2.png')
        # # plt.show()

        ################################################################################################################
        # OPTIMAL SWITCHING TIMES FOR SENESCENT CELL REMOVAL
        ################################################################################################################
        # self.solver.cell_type = 'senescent'
        #
        # # Find optimal switching times for fixed beta and varying alphas
        # Define fixed beta
        # beta = 0.005
        # actions = [0, beta]

        # # Initialise lists to store times and costs
        # min_costs = []
        # best_T1 = []
        # best_T2 = []
        #
        # for a in tqdm(self.alphas):
        #     minT1, minT2, min_cost = self.optimize_switch_times_v2(self.T1, self.T2, actions, a, self.t_vec,
        #                                                            self.highres_step)
        #
        #     min_costs.append(min_cost)
        #     best_T1.append(minT1)
        #     best_T2.append(minT2)
        #
        # best_T1 = np.array(best_T1)
        # best_T2 = np.array(best_T2)
        # np.savetxt('Data/alphas_senescence_t1.txt', best_T1)
        # np.savetxt('Data/alphas_senescence_t2.txt', best_T2)

        # Retrieve data from results folder and plot results
        # best_T1 = np.genfromtxt('Data/alphas_senescence_t1.txt')
        # best_T2 = np.genfromtxt('Data/alphas_senescence_t2.txt')
        # # Plot switching times against value of alpha
        # ax5 = fig.add_subplot(224)
        # ax5.plot(best_T1, self.alphas, color='orange', linewidth=3)
        # ax5.plot(best_T2, self.alphas, color='orange', linewidth=3)
        # ax5.set_xlim(0, 10000)
        # ax5.set_ylim(min(self.alphas))
        # ax5.set_yticks([min(self.alphas), max(self.alphas)])
        # ax5.set_xticks([0, 5000, 10000])
        # ax5.set_ylabel(r'Cost of Intervention, $\alpha$')
        # ax5.set_xlabel(r'Time, $t$')
        # ax5.set_title(r'Senescent cell removal at fixed $\beta = $' + str(beta))
        # ax5.fill_betweenx(self.alphas, best_T1, best_T2, alpha=0.3, color='orange')

        ################################################################################################################
        ## Find optimal switching times for fixed alpha and varying betas
        ## Define fixed alpha
        # alpha = 25
        #
        # # Re-initialise lists to store switching times and costs
        # min_costs = []
        # best_T1 = []
        # best_T2 = []
        #
        # for beta in tqdm(self.betas):
        #     actions = [0, beta]
        #     minT1, minT2, min_cost = self.optimize_switch_times_v2(self.T1, self.T2, actions, alpha, self.t_vec,
        #                                                            self.highres_step)
        #
        #     min_costs.append(min_cost)
        #     best_T1.append(minT1)
        #     best_T2.append(minT2)
        #
        # best_T1 = np.array(best_T1)
        # best_T2 = np.array(best_T2)
        # np.savetxt('Data/betas_senescence_t1.txt', best_T1)
        # np.savetxt('Data/betas_senescence_t2.txt', best_T2)
        #
        # # Retrieve data from results folder and plot results
        # best_T1 = np.genfromtxt('Data/betas_senescence_t1.txt')
        # best_T2 = np.genfromtxt('Data/betas_senescence_t2.txt')
        # # Plot switching times against value of alpha
        # ax6 = fig.add_subplot(326)
        # ax6.plot(best_T1, self.betas, color='orange')
        # ax6.plot(best_T2, self.betas, color='orange')
        # ax6.set_ylim(min(self.betas))
        # ax6.set_xlim(0, 10000)
        # ax6.set_yticks([min(self.betas), max(self.betas)])
        # ax6.set_xticks([0, 5000, 10000])
        # ax6.set_ylabel(r'Rate of Cell Removal, $\beta$')
        # ax6.set_xlabel(r'Time, $t$')
        # ax6.set_title(r'Senescent cell removal at fixed $\alpha = $' + str(alpha))
        # ax6.fill_betweenx(self.betas, best_T1, best_T2, alpha=0.3, color='orange')

        # fig.savefig('PresentationPlots/protocols.png')
        # fig.show()


class MatchProtocols:
    def __init__(self, solver, control):

        self.best_T1_alpha = np.genfromtxt(
            '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t1.txt')
        self.best_T2_alpha = np.genfromtxt(
            '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t2.txt')
        self.best_T1_beta = np.genfromtxt(
            '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/betas_cancer_t1.txt')
        self.best_T2_beta = np.genfromtxt(
            '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/betas_cancer_t2.txt')

        self.betas = control.betas
        self.alphas = control.alphas

        self.T = control.T
        self.dt = control.dt
        self.N = control.N
        self.t_vec = control.t_vec
        self.mu_v = solver.mu_v

    def get_switching_times(self, time_vector, action_vector):

        non_zero = np.flatnonzero(action_vector)
        first_index = non_zero[0]
        last_index = non_zero[-1]

        T1 = time_vector[first_index]
        T2 = time_vector[last_index]

        return T1, T2

    def retrieve_time_values(self, x_array, directory, actions, varying_beta):

        t1s = []
        t2s = []

        for i, x in enumerate(x_array):
            if varying_beta:
                actions = [0, x]
            q_matrix = np.genfromtxt(directory + str(x) + '.txt')
            action_vec = qlearn.get_best_action(q_matrix, actions)
            t1, t2 = self.get_switching_times(self.t_vec, action_vec)

            t1s.append(t1)
            t2s.append(t2)

        return np.array(t1s), np.array(t2s)

    def plot_matching_t(self, x_array, num_data, analytic_data, var_sym):

        plt.scatter(x_array, num_data*self.mu_v, marker='o', s=25, edgecolors='none', label='Q-learning')
        plt.plot(x_array, analytic_data*self.mu_v, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        plt.ylim(min(self.t_vec)*self.mu_v, max(self.t_vec)*self.mu_v)
        plt.legend()
        plt.xlabel(var_sym, fontsize=28)
        plt.ylabel('Switching Time', fontsize=28)
        plt.show()

    def plot_t1_t2(self, x_array, t1_num, t2_num, t1_analytic, t2_analytic, var_sym):

        plt.scatter(x_array, t1_num*self.mu_v, marker='o', s=25, edgecolors='none', label='T1 - Q-learning')
        plt.scatter(x_array, t2_num*self.mu_v, marker='o', s=25, edgecolors='none', label='T2 - Q-learning')
        plt.plot(x_array, t1_analytic*self.mu_v, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        plt.plot(x_array, t2_analytic*self.mu_v, 'm', linestyle='--', linewidth=3, alpha=0.6)
        plt.ylim(min(self.t_vec)*self.mu_v, max(self.t_vec)*self.mu_v)
        plt.legend()
        plt.xlabel(var_sym, fontsize=28)
        plt.ylabel('Switching Times', fontsize=28)
        plt.show()

    def run(self):

        # Plot matching switching times from Q-learning and cost minimisation
        # 1. Cancer cell removal with varying Alphas
        var_sym = r'$\alpha$'
        beta = 0.007
        actions = [0, beta]
        direc = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/Alphas/q_matrix_alpha'

        t1_list, t2_list = self.retrieve_time_values(self.alphas, direc, actions, varying_beta='False')
        self.plot_matching_t(self.alphas, t1_list, self.best_T1_alpha, var_sym)
        self.plot_matching_t(self.alphas, t2_list, self.best_T2_alpha, var_sym)

        # Plot T1 and T2 values on the same plot
        self.plot_t1_t2(self.alphas, t1_list, t2_list, self.best_T1_alpha, self.best_T2_alpha, var_sym)

        # 1. Cancer cell removal with varying Betas
        var_sym = r'$\beta$'
        direc = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/Betas/q_matrix_beta'

        t1_list, t2_list = self.retrieve_time_values(self.betas, direc, actions, varying_beta='True')
        self.plot_t1_t2(self.betas, t1_list, t2_list, self.best_T1_beta, self.best_T2_beta, var_sym)


if __name__ == "__main__":
    solv = Solver()
    solv.run()
    # plot = Plotter(solv)
    # plot.run()
    # qlearn = QLearning(solv, plot)
    # qlearn.run()
    # control = Control(solv, qlearn)
    # control.run()
    # matchprotocols = MatchProtocols(solv, control)
    # matchprotocols.run()
