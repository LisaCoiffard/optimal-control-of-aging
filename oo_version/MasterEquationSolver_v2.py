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

        # Define parameters for each possible intervention for which system is solved
        self.parameters = {
            'k': [0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5],
            'f0': [0.004, 0.004, 0.004, 0.0025, 0.004, 0.004, 0.004],
            'mu_v': [1e-3, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
            'mu_c': [1e-5, 1e-5, 1e-7, 1e-5, 1e-5, 1e-5, 1e-5],
            'lambda_v': [0, 0, 0, 0, 0, 1e-3, 0],
            'lambda_c': [0, 0, 0, 0, 0, 0, 5e-4]
        }
        # Specific cell removal parameters (includes senescent and cancer cell removal)
        self.removal_parameters = {
            'beta': [0.001, 0.005],
            'cell_type': ['cancerous', 'senescent']
        }

        # Initialise cell-type to None for no cell removal
        self.cell_type = None

        # Create arrays of vigor and cooperation parameters from 0 to n and 0 to m respectively
        self.n = 1
        self.m = 1
        v = np.linspace(0, self.n, self.n + 1)
        c = np.linspace(0, self.m, self.m + 1)
        v, c = np.meshgrid(v, c)
        self.v = v.flatten()
        self.c = c.flatten()

        # Initialise cell fraction array according to initial conditions (each cell fraction is initially at 0 except
        # for healthy cells at 1)
        rho0 = np.zeros_like(self.v)
        rho0[-1] = 1
        self.rho0 = rho0

        # Create time array over which to solve system of ODE's
        self.dt = 100
        self.t_long = np.arange(0, 12500 + self.dt, self.dt)

    def growth_rate(self, v, c, k, f0):
        """
        Compute state-dependent growth rate.

        Parameters
        ----------
        v: float
            Measure of state vigor, from 0 to n (inclusive)
        c: float
            Measure of state cooperation, from 0 to m (inclusive)
        k: float
            Cost of cooperation
        f0: float
            Strength of competition

        Returns
        -------
        float
            Value of competition for state (v, c)
        """
        return f0 * v * (1 - k * c)

    def mean_growth_rate(self, rho):
        """ Mean growth rate calculated as a sum of the product of growth rates and cell fraction over each possible
        state (v, c)."""
        return np.sum(self.growth_rate(self.v, self.c, self.k, self.f0) * rho)

    def solve_system(self, rho, t):
        """ Solve system of ODE's over time array, t and return final array of cell fraction for each state of all time
        points."""
        # Solve remove_cancer ODE's if the cell type of removal 'cancerous'
        if self.cell_type == 'cancerous':
            solution = odeint(self.remove_cancer, rho, t)
        # Solve remove_senescence ODE's if the cell type of removal 'senscent'
        elif self.cell_type == 'senescent':
            solution = odeint(self.remove_senescence, rho, t)
        # Solve compute_master_equation if there is no cell removal
        else:
            solution = odeint(self.compute_master_equation, rho, t)

        return solution

    def separate_solution(self, solution):
        """ Separates array of cell fractions into four arrays for each cell type."""
        healthy = solution[:, -1]
        cancerous = solution[:, self.n]
        senescent = solution[:, -(self.m + 1)]
        both = solution[:, 0]

        return healthy, cancerous, senescent, both

    def hazard_func(self, healthy_fraction, t):
        """ Computes the hazard function given the fraction of healthy cells."""
        return -np.gradient(np.log(healthy_fraction))

    def mean_rate_cancer(self, rho):
        """ Compute mean growth rate over all cell types - specific to cancer cell removal."""
        return self.beta * np.sum(np.where(self.c == 0, rho, 0))

    def mean_rate_senescence(self, rho):
        """ Compute mean growth rate over all cell types - specific to senescent cell removal."""
        return self.beta * np.sum(np.where(self.v == 0, rho, 0))

    def compute_master_equation(self, rho, t):
        """ Computes ODE for cell fraction evolution over time."""
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) - self.mu_v * np.where(self.v > 0, rho,
                                                                                                   0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) \
               + self.lambda_v * np.where(self.v > 0, np.roll(rho, 1), 0) \
               - self.lambda_v * np.where(self.v < self.n, rho, 0) \
               + self.lambda_c * np.where(self.c > 0, np.roll(rho, self.n + 1), 0) \
               - self.lambda_c * np.where(self.c < self.m, rho, 0)

    def remove_cancer(self, rho, t):
        """ Computes ODE for cell fraction evolution over time, specific to removing cancer cells."""
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) \
               - self.mu_v * np.where(self.v > 0, rho, 0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) \
               - self.beta * np.where(self.c == 0, rho, 0) \
               + self.mean_rate_cancer(rho) * rho

    def remove_senescence(self, rho, t):
        """ Computes ODE for cell fraction evolution over time, specific to removing senescent cells."""
        return (self.growth_rate(self.v, self.c, self.k, self.f0) - self.mean_growth_rate(rho)) * rho \
               + self.mu_v * np.where(self.v < self.n, np.roll(rho, -1), 0) \
               - self.mu_v * np.where(self.v > 0, rho, 0) \
               + self.mu_c * np.where(self.c < self.m, np.roll(rho, -(self.n + 1)), 0) \
               - self.mu_c * np.where(self.c > 0, rho, 0) - self.beta * np.where(self.v == 0, rho, 0) \
               + self.mean_rate_senescence(rho) * rho

    def run(self):

        param_version = len(self.parameters['k'])

        results = {}
        removal_results = {}

        for i in range(0, param_version):
            self.k = self.parameters['k'][i]
            self.mu_c = self.parameters['mu_c'][i]
            self.mu_v = self.parameters['mu_v'][i]
            self.f0 = self.parameters['f0'][i]
            self.lambda_v = self.parameters['lambda_v'][i]
            self.lambda_c = self.parameters['lambda_c'][i]

            rho = self.solve_system(self.rho0, self.t_long)
            h, c, s, b = self.separate_solution(rho)

            hazard = self.hazard_func(h, self.t_long)

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

                    rho = self.solve_system(self.rho0, self.t_long)
                    h, c, s, b = self.separate_solution(rho)

                    hazard = self.hazard_func(h, self.t_long)

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

        results = results
        removal_results = removal_results
        # Save results dicts
        file = open('Data/Solver/results', 'wb')
        pickle.dump(results, file)
        file.close()
        file = open('Data/Solver/removal_results', 'wb')
        pickle.dump(removal_results, file)
        file.close()

        # Save results of simulations as dictionaries with names according to their parameter values
        for k in results:
            file = open('Data/Solver/' + k, 'wb')
            pickle.dump(results[k], file)
            file.close()
        for k in removal_results:
            file = open('Data/Solver/' + k, 'wb')
            pickle.dump(removal_results[k], file)
            file.close()


class QLearning:
    def __init__(self, solver):

        # Import values from solver
        self.solver = solver
        self.rho0 = self.solver.rho0
        self.params = self.solver.parameters
        self.solver.k = self.params['k'][0]
        self.solver.mu_c = self.params['mu_c'][0]
        self.solver.mu_v = self.params['mu_v'][0]
        self.solver.f0 = self.params['f0'][0]
        self.solver.lambda_v = self.params['lambda_v'][0]
        self.solver.lambda_c = self.params['lambda_c'][0]

        # Q-learning parameters
        self.learning_rate = 0.1
        self.p_explore = 0.5
        self.discount = 0.9998
        self.lambda_val = 0.0005

        # Create time array for system simulations (different to that in Solver)
        self.T = 10000
        self.dt = 100
        self.N = int(self.T / self.dt)
        self.t_vec = np.arange(0, self.T + self.dt, self.dt)

        # Arrays of alpha and beta values over which to identify switching times
        self.alphas = np.linspace(1, 30, 25)
        self.betas = np.linspace(0.002, 0.02, 25)
        # Set fixed alpha and beta parameters for when the other is being varied
        self.alpha_c = 20
        self.beta_c = 0.007
        self.alpha_s = 30
        self.beta_s = 0.005

    def q_learning_action(self, diff_idx, idx, q_matrix):
        """
        Function draws a random float between 0 and 1 and compares it to exploration probability (p_explore). If it is
        less than p_explore action is chosen randomly, otherwise action corresponding to maximum reward is chosen.

        Parameters
        ----------
        diff_idx: int
            Index of current difference state
        idx: int
            Index of current state
        q_matrix: 3-dimensional array
            Current Q-matrix with reward for each possible action at each (diff_idx, idx) state

        Returns
        -------
        int
            Index of the action to be taken
        """
        rand_draw = random.uniform(0, 1)
        if rand_draw < self.p_explore:
            action_idx = random.randint(0, q_matrix.shape[-1] - 1)
        else:
            action_idx = np.argmax(q_matrix[diff_idx, idx, :])

        return action_idx

    def q_learning_update(self, diff_idx, idx, new_diff_idx, new_idx, action_idx, reward, q_matrix):
        """
        Update Q-value of matrix for a specific state, action combination according to the Bellman equation; and updates
        the current diff and state indices.

        Parameters
        ----------
        diff_idx: int
            Index of current state difference
        idx: int
            Index of current state
        new_diff_idx: int
            Index of new state difference
        new_idx: int
            Index of next state
        action_idx: int
            Action index for which to calculate reward
        reward: float
            Reward calculated of taking action_idx at state (diff_idx, idx)
        q_matrix: 3-dimensional array
            Current matrix of Q-values for each possible state, action combination

        Returns
        -------
        int
            Updated index for difference state
        int
            Updated index for state
        3-dimensional array
            Update Q-matrix
        """
        q_matrix[diff_idx, idx, action_idx] = \
            (1 - self.learning_rate) * q_matrix[diff_idx, idx, action_idx] \
            + self.learning_rate * (reward + self.discount * np.amax(q_matrix[new_diff_idx, new_idx, :], axis=-1))
        diff_idx = new_diff_idx
        idx = new_idx

        return diff_idx, idx, q_matrix

    def exp_decay(self, decay_rate, t):
        """ Compute value of decaying a number according to a given decay rate and time point."""
        return math.exp(-decay_rate * t)

    def cell_removal_solver(self, rho0, action, alpha, dp, t):
        """
        Solve system over time array [t, t+dt] given array of initial cell fractions and action as the rate of cell
        removal (set to 0 if there is no removal). Calculate reward after time evolution according to action taken.
        Round final cell fractions and update time point.

        Parameters
        ----------
        rho0: Array of floats
            Initial cell fractions for each cell type (both, cancerous, senescent, healthy)
        action: float
            Value of the rate of cell removal according to action taken (0 for no cell removal)
        alpha: float
            Cost of intervention
        dp: int
            Number of decimal places at which to round solutions
        t: int
            Current time value

        Returns
        -------
        float
            Value of new difference state rounded to dp decimal places
        float
            Value of fraction of healthy cells state rounded to dp decimal places
        array of floats
            Updated cell fractions for each cell type
        float
            Reward calculated as the difference between the fraction of healthy cells (vitality) and the product of
            action and alpha
        int
            Updated time value
        boolean
            True if time value equals T+dt, otherwise False
        """


        # Time array from previous to new state
        time_array = np.array([t, t + self.dt])
        self.solver.beta = action

        # Use solve_system() function from Solver to solve ODE over timestep dt and retrieve new state
        cell_fractions = self.solver.solve_system(rho0, time_array)

        diffs = cell_fractions[1] - cell_fractions[0]
        diffs = np.around(diffs, dp)
        new_diff = diffs[-1]

        cell_fractions = np.around(cell_fractions[-1], dp)
        new_state = cell_fractions[-1]

        reward = new_state - alpha * action

        # Increment time
        t += self.dt

        return new_diff, new_state, cell_fractions, reward, t, t == self.T + self.dt

    def q_matrix_convergence_test(self, num_episodes, actions, alpha, res, dp):
        """
        Function constructs and updates Q-matrix over num_episodes episodes, given array of possible actions and cost of
        intervention.

        :param num_episodes: Integer number of episodes over which to run simulation to get converged Q-matrix
        :param actions: Array of possible actions to take
        :param alpha: Cost of intervention (float)
        :param res: Size of step between subsequent states in discretized vitality array (float)
        :param dp: Number of decimal places at which to round states (int)

        :return: Converged Q-matrix of Q-values for each possible combination of previous and current state for each
        possible action to be chosen from.
        """

        # Initialize array of discretized vitality states
        vitality_states = np.arange(0, 1.0 + res, res)
        vitality_states = np.around(vitality_states, dp)
        diff_states = np.arange(-0.2, 0.2 + res, res)
        diff_states = np.around(diff_states, dp)

        # Initialize Q-matrix of depth 2 with square grid of discretized vitality states for each dimension
        q_matrix = np.zeros([len(diff_states), len(vitality_states), len(actions)])

        # Initialise lists that will hold information about Q-learning convergence
        rewards = []
        errors = []
        no_action_matrix_diffs = []
        action_matrix_diffs = []

        for i in tqdm(range(num_episodes)):
            done = False
            t = 0
            init_cell_fraction = self.rho0
            diff = 0.0
            diff_idx = int(np.where(diff_states == diff)[0])
            state_idx = len(vitality_states) - 1

            # Initialize convergence testing values after each new episode
            reward_tot = 0
            error_tot = 0
            prev_q_matrix = q_matrix.copy()

            while not done:
                action_idx = self.q_learning_action(diff_idx, state_idx, q_matrix)  # get best action for state
                action = actions[action_idx]
                new_diff, new_state, init_cell_fraction, reward, t, done = \
                    self.cell_removal_solver(init_cell_fraction, action, alpha, dp, t)  # test environment

                # Recover index of new state from discretized vitality states array
                new_state_idx = int(np.where(vitality_states == new_state)[0])
                new_diff_idx = int(np.where(diff_states == new_diff)[0])

                # Update state and Q matrix
                diff_idx, state_idx, q_matrix = self.q_learning_update(diff_idx, state_idx, new_diff_idx, new_state_idx,
                                                                             action_idx, reward, q_matrix)

            # Decay learning rate and exploration
            self.p_explore = self.exp_decay(self.lambda_val, i)
            self.learning_rate = self.exp_decay(self.lambda_val, i)

            # Run inference step for learnt actions every 100 episode
            if i % 100 == 0:
                t = 0
                init_cell_fraction = self.rho0
                diff = 0.0
                diff_idx = int(np.where(diff_states == diff)[0])
                state_idx = len(vitality_states) - 1

                action_vec = self.get_protocol(q_matrix[:, :, 0], q_matrix[:, :, 1], actions, alpha, res, dp)

                for action in action_vec:
                    new_diff, new_state, init_cell_fraction, reward, t, done = \
                        self.cell_removal_solver(init_cell_fraction, action, alpha, dp, t)

                    reward_disc = (self.discount ** t) * reward
                    reward_tot += reward_disc

                    new_state_idx = int(np.where(vitality_states == new_state)[0])
                    new_diff_idx = int(np.where(diff_states == new_diff)[0])

                    error = reward + self.discount * np.amax(q_matrix[new_diff_idx, new_state_idx, :], axis=-1)
                    error_tot += error
                    state_idx = new_state_idx

                rewards.append(reward_tot)
                errors.append(error_tot)

                # Compute differences in Q-matrix values
                no_action_matrix_diff = q_matrix[:, :, 0].flatten() - prev_q_matrix[:, :, 0].flatten()
                no_action_matrix_diffs.append(no_action_matrix_diff)
                action_matrix_diff = q_matrix[:, :, 1].flatten() - prev_q_matrix[:, :, 1].flatten()
                action_matrix_diffs.append(action_matrix_diff)

        return q_matrix, action_matrix_diffs, no_action_matrix_diffs, rewards, errors

    def get_protocol(self, no_action_mat, action_mat, actions, alpha, res, dp):
        """
        Given final, separated Q-matrices for each possible action run simulation over total time frame (T + dt) by
        chosing action that has the greatest Q-value for given previous and current state indices.

        :param no_action_mat: 2-D matrix of Q-values for each possible discretized (previous, current) state combination
        for action, 0
        :param action_mat: 2-D matrix of Q-values for each possible discretized (previous, current) state combination
        for action, beta
        :param actions: Array of each possible action [0, beta]
        :param alpha: Cost of intervention (float)
        :param res: Size of step between subsequent states in discretized vitality array (float)
        :param dp: Number of decimal places at which to round states (int)

        :return: List of best action to take over entire time frame (until T+dt)
        """

        t = 0
        cell_fraction = self.rho0
        done = False

        vitality_states = np.arange(0, 1.0 + res, res)
        vitality_states = np.around(vitality_states, dp)
        diff_states = np.arange(-0.2, 0.2 + res, res)
        diff_states = np.around(diff_states, dp)

        state = vitality_states[-1]
        diff = 0.0

        # Initialize list to store action values over each time step of the simulation
        action_vect = []

        while not done:

            # Get indices for previous and current states according to discretized vitality states array
            diff_idx = int(np.where(diff_states == diff)[0])
            idx = int(np.where(vitality_states == state)[0])

            # Select action that has the highest Q-value given previous and current state indices
            if no_action_mat[diff_idx, idx] >= action_mat[diff_idx, idx]:
                action = actions[0]
            else:
                action = actions[1]
            action_vect.append(action)

            # Compute new states by evolving system over timestep dt
            new_diff, new_state, cell_fraction, reward, t, done = self.cell_removal_solver(cell_fraction, action, alpha, dp, t)
            diff = new_diff
            state = new_state

        return action_vect

    def run(self):

        N = 15000
        resolution = 1e-3
        dec_places = 3

        # Run test for Q-matrix convergence
        self.solver.cell_type = 'cancerous'
        actions = [0, self.beta_c]
        alpha = self.alphas[0]

        # q_matrix, action_matrix_diffs, no_action_matrix_diffs, rewards, errors = \
        #     self.q_matrix_convergence_test(N, actions, alpha, resolution, dec_places)
        # action_matrix = q_matrix[:, :, 1]
        # no_action_matrix = q_matrix[:, :, 0]
        # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
        #            'action_matrix_alpha' + str(alpha) + '_beta' + str(self.beta_c) + '.txt', action_matrix)
        # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
        #            'no_action_matrix_alpha' + str(alpha) + '_beta' + str(self.beta_c) + '.txt', no_action_matrix)
        # # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
        # #            'action_matrix_diffs1.txt', action_matrix_diffs)
        # # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
        # #            'no_action_matrix_diffs1.txt', no_action_matrix_diffs)
        # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/rewards1.txt', rewards)
        # np.savetxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/errors1.txt', errors)


        # Get protocol for test case
        action_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
                   'action_matrix_alpha' + str(alpha) + '_beta' + str(self.beta_c) + '.txt')
        no_action_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
                                      'no_action_matrix_alpha' + str(alpha) + '_beta' + str(self.beta_c) + '.txt')
        action_protocol = self.get_protocol(no_action_matrix, action_matrix, actions, alpha, resolution, dec_places)
        plt.plot(self.t_vec, action_protocol)
        plt.show()

        ###############################################################################################################
        # CANCER CELL REMOVAL
        ###############################################################################################################

        self.solver.cell_type = 'cancerous'

        # Find optimal switching times for fixed beta and varying alphas
        actions = [0, self.beta_c]

        for a in self.alphas:
            q_matrix = self.generate_q_matrix(N, actions, a, resolution, dec_places)
            action_matrix = q_matrix[:, :, 1]
            no_action_matrix = q_matrix[:, :, 0]
            np.savetxt('Data/Q-Learning/Alphas/action_mat_' + str(a) + '_cancer.txt', action_matrix)
            np.savetxt('Data/Q-Learning/Alphas/no_action_mat_' + str(a) + '_cancer.txt', no_action_matrix)
            action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/action_mat_' + str(a) + '_cancer.txt')
            no_action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/no_action_mat_' + str(a) + '_cancer.txt')
            action_protocols = self.get_protocol(no_action_matrix, action_matrix, actions, a, resolution, dec_places)
            np.savetxt('Data/Q-Learning/Alphas/Protocols/' + str(a) + '_cancer.txt', action_protocols)

        # Find optimal switching times for fixed alpha and varying betas
        for b in self.betas:
            actions = [0, b]
            q_matrix = self.generate_q_matrix(N, actions, self.alpha_c, resolution, dec_places)
            action_matrix = q_matrix[:, :, 1]
            no_action_matrix = q_matrix[:, :, 0]
            np.savetxt('Data/Q-Learning/Betas/action_mat_' + str(b) + '_cancer.txt', action_matrix)
            np.savetxt('Data/Q-Learning/Betas/no_action_mat_' + str(b) + '_cancer.txt', no_action_matrix)
            action_matrix = np.genfromtxt('Data/Q-Learning/Betas/action_mat_' + str(b) + '_cancer.txt')
            no_action_matrix = np.genfromtxt('Data/Q-Learning/Betas/no_action_mat_' + str(b) + '_cancer.txt')
            action_protocols = self.get_protocol(no_action_matrix, action_matrix, actions, self.alpha_c, resolution,
                                                 dec_places)
            np.savetxt('Data/Q-Learning/Betas/Protocols/' + str(b) + '_cancer.txt', action_protocols)

        ###############################################################################################################
        # SENESCENT CELL REMOVAL
        ###############################################################################################################
        self.solver.cell_type = 'senescent'

        # Find optimal switching times for fixed beta and varying alphas
        actions = [0, self.beta_s]

        for a in self.alphas:
            q_matrix = self.generate_q_matrix(N, actions, a, resolution, dec_places)
            action_matrix = q_matrix[:, :, 1]
            no_action_matrix = q_matrix[:, :, 0]
            np.savetxt('Data/Q-Learning/Alphas/action_mat_' + str(a) + '_senescent.txt', action_matrix)
            np.savetxt('Data/Q-Learning/Alphas/no_action_mat_' + str(a) + '_senescent.txt', no_action_matrix)
            action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/action_mat_' + str(a) + '_senescent.txt')
            no_action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/no_action_mat_' + str(a) + '_senescent.txt')
            action_protocols = self.get_protocol(no_action_matrix, action_matrix, actions, a, resolution,
                                                 dec_places)
            np.savetxt('Data/Q-Learning/Alphas/Protocols/' + str(a) + '_senescent.txt', action_protocols)

        # Find optimal switching times for fixed alpha and varying betas
        for b in self.betas:
            actions = [0, b]
            q_matrix = self.generate_q_matrix(N, actions, self.alpha_s, resolution, dec_places)
            action_matrix = q_matrix[:, :, 1]
            no_action_matrix = q_matrix[:, :, 0]
            np.savetxt('Data/Q-Learning/Betas/action_mat_' + str(b) + '_senescent.txt', action_matrix)
            np.savetxt('Data/Q-Learning/Betas/no_action_mat_' + str(b) + '_senescent.txt', no_action_matrix)
            action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/action_mat_' + str(b) + '_senescent.txt')
            no_action_matrix = np.genfromtxt('Data/Q-Learning/Alphas/no_action_mat_' + str(b) + '_senescent.txt')
            action_protocols = self.get_protocol(no_action_matrix, action_matrix, actions, self.alpha_s, resolution,
                                                 dec_places)
            np.savetxt('Data/Q-Learning/Alphas/Protocols/' + str(b) + '_senescent.txt', action_protocols)


class Control:
    def __init__(self, solver, qlearning):

        # Import values from solver
        self.solver = solver
        self.rho0 = self.solver.rho0
        self.params = self.solver.parameters
        self.solver.k = self.params['k'][0]
        self.solver.mu_c = self.params['mu_c'][0]
        self.solver.mu_v = self.params['mu_v'][0]
        self.solver.f0 = self.params['f0'][0]
        self.solver.lambda_v = self.params['lambda_v'][0]
        self.solver.lambda_c = self.params['lambda_c'][0]

        # Import values from Q-learning
        self.qlearning = qlearning
        self.alphas = self.qlearning.alphas
        self.betas = self.qlearning.betas
        self.beta_c = self.qlearning.beta_c
        self.alpha_c = self.qlearning.alpha_c
        self.beta_s = self.qlearning.beta_s
        self.alpha_s = self.qlearning.alpha_s

        # Define high-resolution steps for cost minimization
        self.highres_step = self.qlearning.dt / 10
        self.T_std = 5 * self.qlearning.dt

        # Define arrays for trial T1 and T2 values
        self.T1 = np.arange(100, 4000, self.T_std)
        self.T2 = np.arange(2000, 10000, self.T_std)

    def cost_calcuator(self, init_state, actions, alpha, time_array, i1, i2):
        """
        Function that calculates total cost of turning on action at time i1 and turning it off at time i2.

        :param init_state: Initial condition of cell fractions at start of the simulation
        :param actions: List of two possible actions (turned off: 0 and turned on: beta-value)
        :param alpha: Cost of intervention
        :param time_array: Array of time over which to perform simulation
        :param i1: Action switch on time
        :param i2: Action switch off time

        :return: Total cost of turning on intervention at i1 and turning it off at i2 over whole time_array
        """

        total_cost = 0

        # Define first time array with no intervention (if i1 is 0, go straight to intervention)
        if i1 > 0:
            time_off1 = time_array[:i1 + 1]
            self.solver.beta = actions[0]
            states = self.solver.solve_system(init_state, time_off1)
            vitalities = states[:, -1]
            init_state = states[-1, :]
            gamma = -np.log(self.qlearning.discount)
            costs = np.exp(-gamma * time_off1) * (alpha * actions[0] - vitalities)
            cost = np.sum(costs)
            total_cost += cost

        # Define time array of intervention and second time array of no intervention
        time_on = time_array[i1:i2 + 1]
        time_off2 = time_array[i2:]

        # Run solver over time array where intervention takes place
        self.solver.beta = actions[1]
        states = self.solver.solve_system(init_state, time_on)
        vitalities = states[:, -1]
        init_state = states[-1, :]
        gamma = -np.log(self.qlearning.discount)
        costs = np.exp(-gamma * time_on) * (alpha * actions[1] - vitalities)
        cost = np.sum(costs)
        total_cost += cost

        # Run solver over time array where intervention is switched off again
        self.solver.beta = actions[0]
        states = self.solver.solve_system(init_state, time_off2)
        vitalities = states[:, -1]
        gamma = -np.log(self.qlearning.discount)
        costs = np.exp(-gamma * time_off2) * (alpha * actions[0] - vitalities)
        cost = np.sum(costs)
        total_cost += cost

        return total_cost

    def optimize_switch_times(self, T1_array, T2_array, actions, alpha, time_array, highres_step):
        """
        Function which trials different start and stop times of intervention according to arrays T1 and T2 and the
        condition that T2 > T1. The total cost of switching on the intervention at all possible T1 and T2 values is
        calculated and the minimum value with its associated times is returned. A second loop is then run with higher
        resolution around the returned times to find a more precise value of switch on and off times.

        :param T1_array: Array of trial values of intervention switch-on times
        :param T2_array: Array of trial values of intervention switch-off times
        :param actions: List of two possible actions (turned off: 0 and turned on: beta-value)
        :param alpha: Cost of intervention
        :param time_array: Array of times for the entire simulation
        :param highres_step: Value of interval between consequent times in time arrays when finding more precise switch-
        on and -off times

        :return: Values of switch-on (minT1) and -off (minT2) times with the lost total cost (also returned as min_cost)
        """

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
        T1_list = np.arange(minT1 - self.qlearning.dt, minT1 + self.qlearning.dt, highres_step)
        T2_list = np.arange(minT2 - self.qlearning.dt, minT2 + self.qlearning.dt, highres_step)

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

        ################################################################################################################
        # OPTIMAL SWITCHING TIMES FOR CANCER CELL REMOVAL
        ################################################################################################################
        self.solver.cell_type = 'cancerous'

        # Find optimal switching times for fixed beta and varying alphas
        actions = [0, self.beta_c]

        # Initialise lists to store times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []

        for a in tqdm(self.alphas):
            minT1, minT2, min_cost = self.optimize_switch_times(self.T1, self.T2, actions, a, self.qlearning.t_vec,
                                                                self.highres_step)
            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/Control/New/alphas_cancer_t1.txt', best_T1)
        np.savetxt('Data/Control/New/alphas_cancer_t2.txt', best_T2)

        # Find optimal switching times for fixed alpha and varying betas
        # Re-initialise lists to store switching times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []

        for beta in tqdm(self.betas):
            actions = [0, beta]
            minT1, minT2, min_cost = self.optimize_switch_times(self.T1, self.T2, actions, self.alpha_c,
                                                                self.qlearning.t_vec, self.highres_step)
            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/Control/New/betas_cancer_t1.txt', best_T1)
        np.savetxt('Data/Control/New/betas_cancer_t2.txt', best_T2)

        ################################################################################################################
        # OPTIMAL SWITCHING TIMES FOR SENESCENT CELL REMOVAL
        ################################################################################################################
        self.solver.cell_type = 'senescent'

        # Find optimal switching times for fixed beta and varying alphas
        actions = [0, self.beta_s]

        # Initialise lists to store times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []

        for a in tqdm(self.alphas):
            minT1, minT2, min_cost = self.optimize_switch_times(self.T1, self.T2, actions, a, self.qlearning.t_vec,
                                                                self.highres_step)
            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/Control/New/alphas_cancer_t1.txt', best_T1)
        np.savetxt('Data/Control/New/alphas_cancer_t2.txt', best_T2)

        # Find optimal switching times for fixed alpha and varying betas
        alpha = 30

        # Re-initialise lists to store switching times and costs
        min_costs = []
        best_T1 = []
        best_T2 = []

        for beta in tqdm(self.betas):
            actions = [0, beta]
            minT1, minT2, min_cost = self.optimize_switch_times(self.T1, self.T2, actions, self.alpha_s,
                                                                self.qlearning.t_vec, self.highres_step)
            min_costs.append(min_cost)
            best_T1.append(minT1)
            best_T2.append(minT2)

        # Save data from cost minimization
        best_T1 = np.array(best_T1)
        best_T2 = np.array(best_T2)
        np.savetxt('Data/Control/New/betas_cancer_t1.txt', best_T1)
        np.savetxt('Data/Control/New/betas_cancer_t2.txt', best_T2)


class DataProcessing:
    def __init__(self, solver, qlearning):

        self.solver = solver
        self.qlearn = qlearning

        self.t_vec = self.qlearn.t_vec

        self.alphas = self.qlearn.alphas
        self.betas = self.qlearn.betas
        self.alpha_c = self.qlearn.alpha_c
        self.beta_c = self.qlearn.beta_c
        self.alpha_s = self.qlearn.alpha_s
        self.beta_s = self.qlearn.beta_s

    def get_plateau(self, dataset, tolerance):
        """
        Determine first value at which data set forms a peak plateau according to a certain tolerance for determining
        how 'flat' the plateau is.

        :param dataset: Array of data
        :param tolerance: Tolerance of what difference between consequent values classifies as 'flat'

        :return: First value of the array at which data flattens at its peak
        """

        # Determine mid point of data
        mid = (np.max(dataset) - np.min(dataset)) / 2
        # Create an array of zero's and one's where a one has to satisfy the conditions that the data is above
        # mid-point, the difference with the next value is less than the tolerance and the difference with the previous
        # value >= 0
        peak_array = np.where((dataset > mid) & (dataset - np.roll(dataset, 1) < tolerance) &
                              (dataset - np.roll(dataset, 1) >= 0), 1, 0)
        index = np.argmax(peak_array)
        value = dataset[index]

        return value

    def get_t_half(self, dataset, tolerance):
        """
        Determine time at which mid-point between minimum and maximum of an array is reached.

        :param dataset: Array of data
        :param tolerance: Tolerance of what difference between consequent values classifies as 'flat'

        :return: Time value at which midpoint between max and min is reached
        """

        # Find array values of local minimum and maximal plateau to be treated as min and max from which to calculate
        # mid-point
        index, min_val = self.get_local_min(dataset)
        max_val = self.get_plateau(dataset, tolerance)
        mid_val = (max_val + min_val) / 2
        # Create a new array where all values before the local minimum are zero
        dataset_new = np.where(self.solver.t_long < self.solver.t_long[index], 0, dataset)
        # Use interp to find time array value associated with mid-point of the array
        time = np.interp(mid_val, dataset_new, self.solver.t_long)

        return time

    def get_local_min(self, dataset):
        """
        Return array index and associated value for local minimum of an array.

        :param dataset: Array of data

        :return: Index and associated value of local min
        """

        # Create modified array where set 1 where left and right neighbouring values are both higher than current value,
        # otherwise set to 0
        min_array = np.where((np.roll(dataset, 1) > dataset) & (np.roll(dataset, -1) > dataset), 1, 0)
        index = np.argmax(min_array)
        value = dataset[index]

        return index, value

    def get_local_max(self, dataset):
        """
        Return array index and associated value for local maximum of an array.

        :param dataset: Array of data

        :return: Index and associated value of local max
        """

        # Create modified array where set 1 where left and right neighbouring values are both lower than current value,
        # otherwise set to 0
        max_array = np.where((np.roll(dataset, 1) < dataset) & (np.roll(dataset, -1) < dataset), 1, 0)
        max_array[0] = 0
        index = np.argmax(max_array)
        value = dataset[index]

        return index, value

    def make_percentage(self, array):
        """
        Return array of values as a percentage of first value of the array

        :param array: Array of values to be converted to percentage

        :return: Array of percentages with respect to first value of input array
        """

        percent = np.zeros(len(array) - 1)
        for i in range(1, len(array)):
            percent[i - 1] = (array[i] - array[0]) / array[0] * 100

        return percent

    def get_switching_times(self, time_vector, action_vector):

        non_zero = np.flatnonzero(action_vector)
        first_index = non_zero[0]
        last_index = non_zero[-1]

        T1 = time_vector[first_index]
        T2 = time_vector[last_index]

        return T1, T2

    def retrieve_time_values(self, x_array, directory, actions, intervention_type, varying_beta):

        t1s = []
        t2s = []

        for i, x in enumerate(x_array):
            if varying_beta:
                actions = [0, x]
            action_vec = np.genfromtxt(directory + str(x) + intervention_type + '.txt')
            t1, t2 = self.get_switching_times(self.t_vec, action_vec)

            t1s.append(t1)
            t2s.append(t2)

        return np.array(t1s), np.array(t2s)

    def run(self):

        ################################################################################################################
        # GENERATE FINGERPRINT DATA FROM SOLVER DATASETS
        ################################################################################################################

        # Set tolerance for obtaining plateau values
        tolerance = 0.0001

        # Retrieve solver data
        file = open('Data/Solver/results', 'rb')
        results = pickle.load(file)
        file.close()
        file = open('Data/Solver/removal_results', 'rb')
        removal_results = pickle.load(file)
        file.close()

        # Initialize arrays of radar plot data
        radar_results = {}
        data_len = len(results) + len(removal_results)

        cancer_data = np.array([results[k]['c'] for k in results.keys()]
                               + [removal_results[k]['c'] for k in removal_results.keys()])
        senescence_data = np.array([results[k]['s'] for k in results.keys()]
                                   + [removal_results[k]['s'] for k in removal_results.keys()])
        hazard_data = np.array([results[k]['hazard'] for k in results.keys()]
                               + [removal_results[k]['hazard'] for k in removal_results.keys()])

        for i in range(data_len):
            k = self.solver.parameters['k'][i]
            mu_c = self.solver.parameters['mu_c'][i]
            mu_v = self.solver.parameters['mu_v'][i]
            f0 = self.solver.parameters['f0'][i]
            lambda_v = self.solver.parameters['lambda_v'][i]
            lambda_c = self.solver.parameters['lambda_c'][i]

            results_name = "_".join([l + "=" + str(eval(l)) for l in self.solver.parameters.keys()])

            # Compute results as percentages
            peak_cancer = self.get_plateau(cancer_data[i], tolerance)
            time_cancer = self.get_t_half(cancer_data[i], tolerance)
            index, min_hazard = self.get_local_min(hazard_data[i])
            peak_hazard = self.get_plateau(hazard_data[i], tolerance)
            time_hazard = self.get_t_half(hazard_data[i], tolerance)
            peak_senescense = np.amax(senescence_data[i])
            time_senescence = self.t_vec[np.argmax(senescence_data[i])]

            if i == 6:
                i_min, min_hazard[i] = self.get_local_min(hazard_data[i])
                i_max, peak_hazard[i] = self.get_local_max(hazard_data[i])
                mid = (peak_hazard[i] + min_hazard[i]) / 2
                c_new = np.where(self.t_vec < self.t_vec[i_min], min_hazard[i], hazard_data[i])
                c_new = np.where(self.t_vec > self.t_vec[i_max], peak_hazard[i], c_new)
                time_hazard[i] = np.interp(mid, c_new, self.t_vec)


            # radar_results[results_name] = {'peak_cancer': ,
            #                               'peak_senescence': ,'time_cancer': ,
            # 'time_senescence', 'min_hazard',
            #              'time_hazard', 'peak_hazard'}


        #
        # peak_cancer = np.zeros(data_len)
        # peak_senescense = np.zeros(data_len)
        # time_cancer = np.zeros(data_len)
        # time_senescence = np.zeros(data_len)
        # min_hazard = np.zeros(data_len)
        # time_hazard = np.zeros(data_len)
        # peak_hazard = np.zeros(data_len)
        #
        #
        #
        # for i in range(data_len):
        #     peak_cancer[i] = self.get_plateau(all_c[i])
        #     time_cancer[i] = self.get_t_half(all_c[i])
        #
        #     index, min_hazard[i] = self.get_local_min(all_hazard[i])
        #     peak_hazard[i] = self.get_plateau(all_hazard[i])
        #     time_hazard[i] = self.get_t_half(all_hazard[i])
        #
        #     peak_senescense[i] = np.amax(all_s[i])
        #     time_senescence[i] = self.t[np.argmax(all_s[i])]
        #
        # i_min, min_hazard[6] = self.get_local_min(all_hazard[6])
        # i_max, peak_hazard[6] = self.get_local_max(all_hazard[6])
        # mid = (peak_hazard[6] + min_hazard[6]) / 2
        # c_new = np.where(self.t < self.t[i_min], min_hazard[6], all_hazard[6])
        # c_new = np.where(self.t > self.t[i_max], peak_hazard[6], c_new)
        # time_hazard[6] = np.interp(mid, c_new, self.t)
        #
        # peak_cancer_percent = -(self.make_percentage(peak_cancer))
        # time_cancer_percent = self.make_percentage(time_cancer)
        # peak_senescense_percent = -(self.make_percentage(peak_senescense))
        # time_senescence_percent = self.make_percentage(time_senescence)
        # min_hazard_percent = -(self.make_percentage(min_hazard))
        # time_hazard_percent = self.make_percentage(time_hazard)
        # peak_hazard_percent = -(self.make_percentage(peak_hazard))

        ################################################################################################################
        # GENERATE PROTOCOL SWITCHING ON/OFF TIMES FROM Q-LEARNING DATASETS
        ################################################################################################################

        # CANCER CELL REMOVAL
        intervention = '_cancer'

        # Protocols for varying alphas
        directory = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Alphas/Protocols/'
        actions = [0, self.beta_c]
        t1_num, t2_num = self.retrieve_time_values(self.alphas, directory, actions, intervention, varying_beta=False)
        np.savetxt(directory + 't1_num.txt', t1_num)
        np.savetxt(directory + 't2_num.txt', t2_num)

        # Protocols for varying betas
        directory = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Betas/Protocols/'
        t1_num, t2_num = self.retrieve_time_values(self.betas, directory, actions, intervention, varying_beta=True)
        np.savetxt(directory + 't1_num.txt', t1_num)
        np.savetxt(directory + 't2_num.txt', t2_num)


class Plotter:
    def __init__(self, qlearning):
        self.qlearn = qlearning

        self.alphas = self.qlearn.alphas
        self.betas = self.qlearn.betas

    def plot_matching_protocols(self, t1_numerical, t2_numerical, t1_analytical, t2_analytical, x_array):
        plt.scatter(x_array, t1_numerical, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
        plt.plot(x_array, t1_analytical, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        plt.ylim(0, 2000)

        plt.scatter(x_array, t2_numerical, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
        plt.plot(x_array, t2_analytical, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        plt.ylim(0, 10000)

        plt.show()

    def run(self):
        ################################################################################################################
        # PLOT Q-LEARNING CONVERGENCE TESTS
        ################################################################################################################
        N = int(15000/100)

        directory = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Tests/'
        # action_matrix_diffs = np.genfromtxt(directory + 'action_matrix_diffs.txt')
        # no_action_matrix_diffs = np.genfromtxt(directory + 'no_action_matrix_diffs.txt')
        errors = np.genfromtxt(directory + 'errors1.txt')
        rewards = np.genfromtxt(directory + 'rewards1.txt')

        plt.plot(range(N), errors)
        plt.title('Errors plot')
        plt.show()
        plt.plot(range(N), rewards)
        plt.title('Rewards plot')
        plt.show()

        ################################################################################################################
        # PLOT MATCHING T1'S AND T2'S FOR CONTROL AND Q-LEARNING
        ################################################################################################################
        # CANCER CELL REMOVAL - for varying alphas
        dir_qlearn = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Alphas/Protocols/'
        dir_control = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/New/'

        t1_num = np.genfromtxt(dir_qlearn + 't1_num.txt')
        t2_num = np.genfromtxt(dir_qlearn + 't2_num.txt')
        t1_analy = np.genfromtxt(dir_control + 'alphas_cancer_t1.txt')
        t2_analy = np.genfromtxt(dir_control + 'alphas_cancer_t2.txt')
        print(t1_analy[0], t2_analy[0])
        self.plot_matching_protocols(t1_num, t2_num, t1_analy, t2_analy, self.alphas)

        dir_qlearn = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Q-Learning/Betas/Protocols/'

        t1_num = np.genfromtxt(dir_qlearn + 't1_num.txt')
        t2_num = np.genfromtxt(dir_qlearn + 't2_num.txt')
        t1_analy = np.genfromtxt(dir_control + 'betas_cancer_t1.txt')
        t2_analy = np.genfromtxt(dir_control + 'betas_cancer_t2.txt')
        self.plot_matching_protocols(t1_num, t2_num, t1_analy, t2_analy, self.betas)


if __name__ == "__main__":
    solv = Solver()
    # solv.run()
    qlearn = QLearning(solv)
    qlearn.run()
    control = Control(solv, qlearn)
    # control.run()
    process = DataProcessing(solv, qlearn)
    # process.run()
    plot = Plotter(qlearn)
    # plot.run()