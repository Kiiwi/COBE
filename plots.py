import matplotlib.pylab as plt
import numpy as np

#data_53 = 'cobe_dmr_53GHz_n16.npy'
data_53 = 'cobe_dmr_53ghz_lnL.npy'
data_90 = 'cobe_dmr_90ghz_lnL.npy'

a = np.load(data_53)
b = np.load(data_90)

x, y, z, cmb, rms = [a[:,i] for i in range(5)]
x_values = a[:,0]
y_values = a[:,1]
z_values = a[:,2]
cmb_values = a[:,3]
xgrid, ygrid = np.meshgrid(x,y, indexing='ij')

plt.contour(xgrid,ygrid,cmb_values)
#plt.plot(cmb_values)
plt.show()
#print(cmb_values)

#print(a[:,0])
