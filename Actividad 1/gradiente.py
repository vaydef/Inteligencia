import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0,np.pi,0.2)
y = np.arange(-2,np.pi,0.2)

xm, ym = np.meshgrid(x, y)
f = np.cos(xm)*np.cos(ym)*np.exp((-1*(xm**2))/5)
dx, dy = np.gradient(f)

fig, ax = plt.subplots()
ax.quiver(xm,ym,dx,dy)


plt.show()
