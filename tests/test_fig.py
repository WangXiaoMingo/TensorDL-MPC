import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots(dpi=72, figsize=(8,6))

x = np.arange(-2*np.pi, 2*np.pi, 0.01)
y = np.sin(x)

line,  = ax.plot(x, y)

def init():
    line.set_ydata(np.sin(x))
    return line

def animate(i):
    line.set_ydata(np.sin(x+i/10.0))
    return line

animation = animation.FuncAnimation(fig=fig,
                                       func=animate,
                                       frames=100, # total 100 frames
                                       init_func=init,
                                       interval=20,# 20 frames per second
                                       blit=False)
animation.save('sinx.gif', writer='imagemagick')
plt.show()