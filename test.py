import numpy as np
import matplotlib.pyplot as plt

training_steps = np.arange(100000)

lr_fn = 10 ** (- 3 * training_steps / training_steps[-1])
plt.plot(training_steps, lr_fn)

lr_fn = 1. - ((1 - 0.01) * training_steps / training_steps[-1])
plt.plot(training_steps, lr_fn)

plt.yscale('log')
plt.savefig('tmp.png')

print(lr_fn.min())