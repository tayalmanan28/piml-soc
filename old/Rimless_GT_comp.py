import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import os

true_data = spio.loadmat('data_rimless_wheel_t_30.mat')
#print(true_data['data'][:, :, 30])
Z = true_data['data'][:, :, 30].T
print(Z<=0)

fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title('t = %0.2f' % (6.3))
ax2.imshow(1*(Z<=0.0), cmap='bwr',
origin='lower', extent=(-0.2, 0.6, -0.6, 1.3), aspect='auto')
fig2.savefig(os.path.join('Ground_Truth.png'))