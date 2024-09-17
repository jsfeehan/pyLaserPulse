import matplotlib.pyplot as plt
import numpy as np

from pyLaserPulse.data import paths

cs = paths.fibres.cross_sections.Yb_Al_silica

data = np.loadtxt(cs)

nt = data[:, 0].shape[0]
x = np.linspace(-1 * int(nt / 2), int(nt / 2), nt)
kernel = np.exp(-x**2 / 10**2)

smooth_data = np.zeros_like(data)
smooth_data[:, 0] = data[:, 0]
for i in range(2):
    smooth_data[:, i+1] = np.convolve(data[:, i+1], kernel, mode='same')
    smooth_data[:, i+1] *= np.amax(data[:, i+1]) / np.amax(smooth_data[:, i+1])

np.savetxt(cs + '_2.dat', smooth_data, delimiter='\t')

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(data[:, 0], data[:, 1], c='k')
ax1.plot(smooth_data[:, 0], smooth_data[:, 1], c='darkorange', ls='--')
ax2.plot(data[:, 0], data[:, 2], c='k')
ax2.plot(smooth_data[:, 0], smooth_data[:, 2], c='darkorange', ls='--')
plt.show()