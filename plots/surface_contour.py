from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rcParams
from matplotlib.ticker import LinearLocator, FormatStrFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.array([5,10,50,100,150,200,500,1000])
Y = np.array([100,150,170,180,190,200,250,300,400])

X, Y = np.meshgrid(X, Y)

Z = np.array([[ 69,  67, 	 60,	 61, 	 62, 	 61, 	 61, 	 62 ]
	,[ 65, 	 67, 	 62, 	 62, 	 61, 	 60, 	 61, 	 61, ]
	,[ 64, 	 69, 	 65, 	 64, 	 60, 	 59, 	 57, 	 59, ]
	,[ 56, 	 64, 	 58, 	 60, 	 61, 	 60, 	 61, 	 60, ]
	,[ 67, 	 66, 	 56, 	 57, 	 58, 	 57, 	 58, 	 59, ]
	,[ 76, 	 78, 	 65, 	 67, 	 67, 	 66, 	 67, 	 66, ]
	,[ 59, 	 64, 	 57, 	 57, 	 57, 	 59, 	 57, 	 60, ]
	,[ 59, 	 64, 	 57, 	 57, 	 57, 	 59, 	 57, 	 60, ]
	,[ 50, 	 61, 	 68, 	 64, 	 64, 	 63, 	 64, 	 62, ]])

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.brg,
                       linewidth=0, antialiased=False)


fig.colorbar(surf, shrink=0.5, aspect=10)

rcParams.update({'font.size': 9})
ax.set_xlabel("'LASSO 'C'", fontsize=10)
ax.set_ylabel('Parcels', fontsize=10)
ax.set_zlabel('Accuracy', fontsize=10)



#plt.show()
plt.savefig('grampa_lasso_3d.png')