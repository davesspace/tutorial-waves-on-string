"""
d2y/dt^2 = c^2 d2y/dx^2
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')  # comment out for "light" theme
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 3


class String():

    def __init__(self, x, y0, c):
        self.x, self.y, self.y0, self.c = np.copy(
            x), self.pad_array(y0), self.pad_array(y0), c
        self.y_prev = np.copy(self.y0)

    def pad_array(self, arr):
        return np.concatenate((arr[:1], arr, arr[-1:]))

    def increment(self, dt):
        """Increment shape of string by dt"""
        r = (self.c * dt/np.gradient(self.x))**2
        temp = np.copy(self.y)
        self.y[1:-1] = 2 * self.y[1:-1] - self.y_prev[1:-1] + \
            r * (self.y[2:] - 2 * self.y[1:-1] + self.y[:-2])
        self.y_prev = temp
        self.y[[0, 1, -2, -1]] = self.y0[[0, 1, -2, -1]]


plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams["font.size"] = 16

fps = 30

d0 = 0.1  # initial displacement
d0_loc = 0.8

c = 10

x = np.linspace(0, 100, 256)
y = np.empty_like(x)

dt = 0.5*((x[-1] - x[0])/len(x))/c

y[np.where(x <= d0_loc*x[-1])] = d0/(d0_loc*x[-1]) * \
    x[np.where(x <= d0_loc*x[-1])]
y[np.where(x > d0_loc*x[-1])] = -d0/((1.0-d0_loc)*x[-1]) * \
    (x[np.where(x > d0_loc*x[-1])] - x[-1])
# y = d0*np.sin(2*np.pi/(x[-1] - x[0]) * x)

string = String(x, y, c)

fig, ax = plt.subplots()
line, =  ax.plot(x, y)

ax.axhline(y=0, alpha=0.3, color="black", ls="-.")

ax.set_ylim([-1.1*d0, 1.1*d0])

ax.set(xlabel="x", ylabel="Displacement")
ax.set_xlim([x[0], x[-1]])

fig.tight_layout()

t0 = time.time()
t = time.time()
t_next = time.time() + 1./fps


def init():
    return line,


def update(frame_no):
    global t

    while t <= time.time():
        string.increment(dt)
        t += dt

    line.set_ydata(string.y[1:-1])
    return line,


ani = FuncAnimation(fig, update, init_func=init, frames=range(
    1000), blit=True, interval=1000./fps, repeat=True)
