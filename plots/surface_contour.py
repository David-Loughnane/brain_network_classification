from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
#X = np.arange(-5, 5, 0.25)
X = np.array([.01, .1, 1, 10, 100, 1000])
print 'X'
print X.shape
print X
print ''
#Y = np.arange(-5, 5, 0.25)
Y = np.array([50,75,100, 125 ,150,200,400])
print 'Y'
print Y.shape
print Y
print ''

X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)

Z = np.array([[.1,.2,.3,.4,.5,.6],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1], [1,1,1,1,1,1]])

print 'Z'
print Z.shape
print Z
print ''


surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.RdBu,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()