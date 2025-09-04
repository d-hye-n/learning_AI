#basic step function
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print(step_function(0.5))

#numpy ver
import numpy as np

def step_function_np(x):
    y = x > 0
    return y.astype(np.int32)

print(step_function_np(np.array([-1.0, 1.0, 2.0])))

#plotting
import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
y = step_function_np(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
