import numpy as np
import matplotlib.pyplot as plt

A = np.random.rand(20, 20) * 256

plt.figure()
plt.imshow(A, vmin=0, vmax=256, cmap="gray")

plt.xticks(np.arange(20), np.arange(20))
plt.yticks(np.arange(20), np.arange(20))
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()