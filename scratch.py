import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import math

mu_v = 1e-3
T = 10000
dt = 100
N = int(T / dt)
t_vec = np.arange(0, T + dt, dt)

N_ep = 15000
# beta = 0.007
# actions = [0, beta]
# alpha = 20

# reward = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Rewards tests/reward_test3.txt')
#
# plt.plot(range(int(N_ep/100)), reward, color='orange')
# plt.savefig('/Users/Lilou/PycharmProjects/summer_project/oo_version/rewards1.png')
# plt.show()
#
# error = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Rewards tests/error_test3.txt')
#
# plt.plot(range(int(N_ep/100)), error, color='blue')
# plt.savefig('/Users/Lilou/PycharmProjects/summer_project/oo_version/errors1.png')
# plt.show()

diff = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Rewards tests/diff_test1.txt')

episodes_list = [0, 15000]
mean_diffs_on = []
mean_diffs_off = []

# for i in tqdm(range(N_ep)):
#     action_off = diff[i, :101]
#     action_on = diff[i, 101:]
#
#     action_off_mean = np.mean(action_off)
#     action_on_mean = np.mean(action_on)
#     mean_diffs_off.append(action_off_mean)
#     mean_diffs_on.append(action_on_mean)
#
# plt.plot(range(N_ep), mean_diffs_off, label='no action')
# plt.plot(range(N_ep), mean_diffs_on, label='action')
# plt.plot(range(N_ep), np.ones_like(range(N_ep))*1e-4)
# plt.plot(range(N_ep), -np.ones_like(range(N_ep))*1e-4)
#
# plt.legend()
# plt.ylim(-0.001, 0.001)
# plt.xlim(0, N_ep)
#
# # plt.savefig('/Users/Lilou/PycharmProjects/summer_project/oo_version/diffs_test4.png')
# plt.show()


# for i in tqdm(range(10)):
#     # fig = plt.figure(figsize=(7, 10))
#
#     q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/q_matrix_' + str(i)
#                              + '.txt')
#     action_vec = get_best_action(q_matrix, actions)
#     plt.plot(t_vec*mu_v, action_vec, linewidth=1)

# t1 = 2130
# t2 = 7510
# y = np.piecewise(t_vec, [(t_vec <= t1) & (t_vec >= t2),
#                                 (t1 <= t_vec) & (t_vec <= t2)], [0, 1])
# y = y * 0.007
#
# plt.plot(t_vec*mu_v, y, '-.', linewidth=3, color='orange', alpha=0.7)
# plt.fill_between(t_vec*mu_v, y, alpha=0.2, color='orange')
# plt.xlim(0, 10)
# plt.ylim(0)
# plt.yticks([0, beta], [0, r'$\beta$'])
# plt.xticks([t1*mu_v, t2*mu_v], ['$T_1$', '$T_2$'])
# plt.title('Q-Learning Protocol', fontsize=24)
#     # plt.savefig('protocol.png')
# plt.show()


def plot_learned_protocol(actions):

    # Plot learned repair
    removal_vecs = []
    dirs = '/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/q_matrix_'
    for i in range(9):
        q_matrix = np.genfromtxt(dirs + str(i) + '.txt')
        removal_vec = get_best_action(q_matrix, actions)
        removal_vecs.append(removal_vec)

    repair_mat = np.vstack(removal_vecs)
    removal_vec = []
    for n in range(repair_mat.shape[1]):
        print(repair_mat[:, n])
        removal_vec.append(stats.mode(repair_mat[:, n])[0])

    time_vec = np.arange(0, len(removal_vec))  # get time vec

    plt.scatter(time_vec.tolist(), removal_vec, color='#000080', s=7, alpha=0.7)
    plt.xlabel('Time, $t$')
    plt.ylabel('Repair, $r(t)$')
    # plt.xlim(0, T)
    # plt.ylim(-0.001, None)
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)

    plt.show()


# def get_T1(time_vec, action_vector, N):
#
#     error_count = []
#     for i in range(len(time_vec)):  # count classification errors
#         err_lower = np.count_nonzero(action_vector[:i])
#         err_upper = len(action_vector[i:]) - np.count_nonzero(action_vector[i:])
#         error_count.append(err_lower + err_upper)
#     error_count = np.array(error_count)
#     err, min_err_idx = min((val, idx) for (idx, val) in enumerate(error_count))  # get T1 with min error
#     T1 = time_vec[min_err_idx]
#
#     return T1

def get_T1(time_vector, action_vector, N, tol):

    done = False
    non_zero = np.flatnonzero(action_vector)
    index = non_zero[0]
    last_index = non_zero[-1]
    # next_index = non_zero[1]
    i = 1
    while not done:
        no_action = len(action_vector[index:last_index]) - np.count_nonzero(action_vector[index:last_index])
        if no_action > tol * N:
            index = non_zero[i+1]
            # index = next_index
            # next_index = non_zero[i+1]
            done = False
            i += 1
        else:
            done = True
    T1 = time_vector[index]

    return T1


def get_times(time_vector, action_vector, N):

    done = False
    non_zero = np.flatnonzero(action_vector)
    first_index = non_zero[0]
    last_index = non_zero[-1]
    T1 = time_vector[first_index]
    T2 = time_vector[last_index]

    return T1, T2


# plot_learned_protocol(actions)


learning_rate = 0.1
p_explore = 0.1
decay = np.array([0.0005, 0.0005])
learning_rates = [learning_rate]
exploration_probs = [p_explore]


def exp_decay(lambda_val, t):
    return math.exp(-lambda_val * t)


# for i in range(N_ep):
#     p_explore = exp_decay(decay[0], i)
#     exploration_probs.append(p_explore)
#
#     learning_rate = exp_decay(decay[1], i)
#     learning_rates.append(learning_rate)
#
# plt.plot(range(N_ep+1), exploration_probs, label='Exploration probability', linewidth=7)
# plt.plot(range(N_ep+1), learning_rates, label='Learning rate', linewidth=2)
# plt.legend()
# plt.show()


# alphas = np.linspace(1, 30, 50)
# betas = np.linspace(0.002, 0.02, 50)
#
# # Create figure to store plots for cancer and senescent cell removal
# fig = plt.figure(figsize=(15,10))
#
# # # Cancer protocol
# t1_c = 2130
# t2_c = 7510
# y_c = np.piecewise(t_vec, [(t_vec <= t1_c) & (t_vec >= t2_c),
#                               (t1_c <= t_vec) & (t_vec <= t2_c)], [0, 1])
#
# ax1 = fig.add_subplot(231)
# ax1.plot(t_vec, y_c, linewidth=3, color='blue')
# ax1.fill_between(t_vec, y_c, alpha=0.3, color='blue')
# # ax1.set_title('Protocol for Cancer Cell Removal')
# ax1.set_ylabel('Rate of Cell Removal', fontsize=24)
# ax1.set_yticks([0, 1])
# ax1.set_yticklabels([0, r'$\beta_c$'])
# ax1.set_xticks([t1_c, t2_c])
# ax1.set_xticklabels(['$T_1$', '$T_2$'])
# ax1.set_xlim(0)
# ax1.set_ylim(0, 1.2)
# ax1.set_title('Cancer cell removal', fontsize=24)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
#
# # Senescence protocol
# t1_s = 100
# t2_s = 2500
# y_s = np.piecewise(t_vec, [(t_vec <= t1_s) & (t_vec >= t2_s),
#                                 (t1_s <= t_vec) & (t_vec <= t2_s)], [0, 1])
#
# ax2 = fig.add_subplot(232)
# ax2.plot(t_vec, y_s, linewidth=3, color='orange')
# ax2.fill_between(t_vec, y_s, alpha=0.3, color='orange')
# # ax2.set_title('Protocol for Senescent Cell Removal')
# ax2.set_yticks([0, 1])
# ax2.set_ylabel('Rate of Cell Removal', fontsize=24)
# ax2.set_yticklabels([0, r'$\beta_c$'])
# ax2.set_xticks([t1_s, t2_s])
# ax2.set_xticklabels(['$T_1$', '$T_2$'])
# ax2.set_xlim(0)
# ax2.set_title('Senescent cell removal', fontsize=24)
# ax2.set_ylim(0, 1.2)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
#
# Retrieve data from results folder and plot results
# best_T1 = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t1.txt')
# best_T2 = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t2.txt')
# ax3 = fig.add_subplot(233)
# ax3.plot(best_T1*mu_v, alphas, color='blue', linewidth=3)
# ax3.plot(best_T2*mu_v, alphas, color='blue', linewidth=3)
# ax3.set_xlim(0, 10000*mu_v)
# ax3.set_ylim(min(alphas))
# ax3.set_yticks([min(alphas), max(alphas)])
# ax3.set_xticks([0, 5000*mu_v, 10000*mu_v])
# ax3.set_ylabel(r'Cost of Intervention, $\alpha$', fontsize=24)
# ax3.set_xlabel(r'Time, $t$', fontsize=24)
# ax3.fill_betweenx(alphas, best_T1*mu_v, best_T2*mu_v, alpha=0.3, color='blue')
#
# # Retrieve data from results folder and plot results
# best_T1 = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_senescence_t1.txt')
# best_T2 = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_senescence_t2.txt')
# # Plot switching times against value of alpha
# ax5 = fig.add_subplot(234)
# ax5.plot(best_T1*mu_v, alphas, color='orange', linewidth=3)
# ax5.plot(best_T2*mu_v, alphas, color='orange', linewidth=3)
# ax5.set_xlim(0, 10000*mu_v)
# ax5.set_ylim(min(alphas))
# ax5.set_yticks([min(alphas), max(alphas)])
# ax5.set_xticks([0, 5000*mu_v, 10000*mu_v])
# ax5.set_ylabel(r'Cost of Intervention, $\alpha$', fontsize=24)
# ax5.set_xlabel(r'Time, $t$', fontsize=24)
# ax5.fill_betweenx(alphas, best_T1*mu_v, best_T2*mu_v, alpha=0.3, color='orange')
#
# # t1 = 2130
# # t2 = 7510
# # y = np.piecewise(t_vec, [(t_vec <= t1) & (t_vec >= t2),
# #                                 (t1 <= t_vec) & (t_vec <= t2)], [0, 1])
# # y = y * 0.007
# #
# # ax4 = fig.add_subplot(235, rowspan=2)
# # ax4.plot(t_vec*mu_v, y, '-.', linewidth=3, color='orange', alpha=0.7)
# # ax4.fill_between(t_vec*mu_v, y, alpha=0.2, color='orange')
# # q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Rewards tests/q_matrix_test1.txt')
# # action_vec = get_best_action(q_matrix, actions)
# # ax4.plot(t_vec*mu_v, action_vec, linewidth=3)
# # ax4.set_xlim(0, 10)
# # ax4.set_ylim(0)
# # ax4.set_yticks([0, beta])
# # ax4.set_xticks([t1*mu_v, t2*mu_v])
# # ax4.set_xticklabels(['$T_1$', '$T_2$'])
# # ax4.set_yticklabels([0, r'$\beta$'])
# # ax4.set_title('Q-Learning Protocol', fontsize=24)
# #
# # fig.savefig('control_protocols.png')
# # fig.show()

# Retrieve data from cost minimization for t1 and t2 - cancer cell removal
# best_T1_alpha = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t1.txt')
# best_T2_alpha = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t2.txt')
#
# # Find values of t1 and t2 for different values of alpha using Q learning
alphas = np.linspace(1, 30, 50)
beta = 0.007
actions = [0, beta]

# tolerance = 0.7
#
t1_values = []
t2_values = []
# for i, a in tqdm(enumerate(alphas)):
#     q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/Alphas/q_matrix_alpha'
#                              + str(a) + '.txt')
#     action_vec = get_best_action(q_matrix, actions)
#     t1_analytical = best_T1_alpha[i]
#     # t1_num = get_T1_new(t_vec, action_vec, N)
#     t1_num = get_T1(t_vec, action_vec, N, tolerance)
#     # plt.scatter(t1_analytical * mu_v, beta, s=7, color='red')
#     # plt.plot(t_vec * mu_v, action_vec, linewidth=1, color='green')
#     # plt.scatter(t1_num*mu_v, beta, s=7, color='blue')
#     # plt.show()
#     t1_values.append(t1_num)
#
# # t1_values = np.array(t1_values)
# # print(t1_values)
# # print(best_T1)
# # print(best_T1-t1_values)
#
# plt.scatter(alphas, t1_values, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
# plt.plot(alphas, best_T1_alpha, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
# plt.ylim(0, 10000)
# plt.show()
#
# best_T1_beta = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/betas_cancer_t1.txt')
# best_T2_beta = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/betas_cancer_t2.txt')
#
# # Find values of t1 and t2 for different values of alpha using Q learning
# betas = np.linspace(0.002, 0.02, 50)
# tolerance = 0.5
#
# t1_values = []
# for i, b in tqdm(enumerate(betas)):
#     beta = b
#     actions = [0, beta]
#
#     q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/Betas/q_matrix_beta'
#                              + str(b) + '.txt')
#     action_vec = get_best_action(q_matrix, actions)
#     t1_analytical = best_T1_beta[i]
#     t1_num = get_T1(t_vec, action_vec, N, tolerance)
#     # plt.scatter(t1_analytical * mu_v, beta, s=7, color='red')
#     # plt.plot(t_vec * mu_v, action_vec, linewidth=1, color='green')
#     # plt.scatter(t1_num*mu_v, beta, s=7, color='blue')
#     # plt.show()
#     t1_values.append(t1_num)
#
# # t1_values = np.array(t1_values)
# # print(t1_values)
# # print(best_T1)
# # print(best_T1-t1_values)
#
# plt.scatter(betas, t1_values, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
# plt.plot(betas, best_T1_beta, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
# plt.ylim(0, 10000)
# plt.show()


# beta = 0.005
# actions = [0, beta]
# q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/senescent_removal.txt')
# action_vec = get_best_action(q_matrix, actions)
# plt.plot(t_vec, action_vec)
# plt.show()



# tolerances = np.arange(0.1, 1, 0.1)
# print(tolerances)
#
# for tol in tolerances:
#
#     t1_values = []
#     t2_values = []
#     t1s = []
#     t2s = []
#     best_T1_alpha = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t1.txt')
#     best_T2_alpha = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Control/alphas_cancer_t2.txt')
#
#     for i, a in tqdm(enumerate(alphas)):
#         q_matrix = np.genfromtxt('/Users/Lilou/PycharmProjects/summer_project/oo_version/Data/Results/Alphas/q_matrix_alpha'
#                                  + str(a) + '.txt')
#         action_vec = get_best_action(q_matrix, actions)
#         t1_analytical = best_T1_alpha[i]
#         t2_analytial = best_T2_alpha[i]
#         t1_num, t2_num = get_switching_times(t_vec, action_vec, N, tol)
#         t1, t2 = get_times(t_vec, action_vec, N)
#         # plt.scatter(t1_analytical * mu_v, beta, s=7, color='red')
#         # plt.scatter(t2_analytial * mu_v, beta, s=7, color='red')
#         # plt.plot(t_vec * mu_v, action_vec, linewidth=1, color='green')
#         # plt.scatter(t1_num*mu_v, beta, s=7, color='blue')
#         # plt.scatter(t2_num*mu_v, beta, s=7, color='blue')
#         # plt.show()
#         t1_values.append(t1_num)
#         t2_values.append(t2_num)
#         t1s.append(t1)
#         t2s.append(t2)
#
#     plt.scatter(alphas, t1_values, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
#     plt.plot(alphas, best_T1_alpha, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
#     plt.ylim(0, 10000)
#     plt.legend()
#     plt.title('Tolerance = ' + str(tolerance) + ' for T1')
#     plt.show()
#
#     plt.scatter(alphas, t2_values, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
#     plt.plot(alphas, best_T2_alpha, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
#     plt.ylim(0, 10000)
#     plt.legend()
#     plt.title('Tolerance = ' + str(tolerance) + ' for T2')
#     plt.show()
#
# plt.scatter(alphas, t1s, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
# plt.plot(alphas, best_T1_alpha, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
# plt.ylim(0, 10000)
# plt.legend()
# plt.title('No tolerance')
# plt.show()
#
# plt.scatter(alphas, t2s, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
# plt.plot(alphas, best_T2_alpha, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
# plt.ylim(0, 10000)
# plt.legend()
# plt.title('No tolerance')
# plt.show()



