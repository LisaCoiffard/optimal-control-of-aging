import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25, 20))

T = 10000
dt = 100
N = int(T / dt)
t_vec = np.arange(0, T + dt, dt)

t1_c = 3000
t2_c = 7000
y_c = np.piecewise(t_vec, [(t_vec <= t1_c) & (t_vec >= t2_c),
                                (t1_c <= t_vec) & (t_vec <= t2_c)], [0, 1])

ax1 = fig.add_subplot(321)
ax1.plot(t_vec, y_c, linewidth=2, color='blue')
ax1.fill_between(t_vec, y_c, alpha=0.3, color='blue')
ax1.set_title('Protocol for Cancer Cell Removal')
ax1.set_yticks([0, 1])
ax1.set_yticklabels([0, r'$\beta_c$'])
ax1.set_xticks([0, 5000, 10000])
ax1.set_xlim(0)
ax1.set_ylim(0, 2)

# Senescence protocol
t1_s = 500
t2_s = 2000
y_s = np.piecewise(t_vec, [(t_vec <= t1_s) & (t_vec >= t2_s),
                                (t1_s <= t_vec) & (t_vec <= t2_s)], [0, 1])

ax2 = fig.add_subplot(322)
ax2.plot(t_vec, y_s, linewidth=2, color='orange')
ax2.fill_between(t_vec, y_s, alpha=0.3, color='orange')
ax2.set_title('Protocol for Senescent Cell Removal')
ax2.set_yticks([0, 1])
ax2.set_yticklabels([0, r'$\beta_c$'])
ax2.set_xticks([0, 5000, 10000])
ax2.set_xlim(0)
ax2.set_ylim(0, 2)

fig.show()