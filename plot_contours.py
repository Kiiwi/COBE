# Simple contour plotting script for visualizing the lnL computed by
# cmb_likelihood.py.
# For convenience, it takes as input either the .npy file or the .dat file.
# In the .dat case you also have to supply the number of grid points in each
# direction so that we can define the grid correctly.

import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    if len(sys.argv)<2:
        print('Wrong number if input arguments.')
        print('Usage: python plot_contours.py resultfile.npy')
        print('Or: python plot_contours.py resultfile.dat numpoints_Q numpoints_n')
        sys.exit()

    inputfile = sys.argv[1]
    if inputfile[inputfile.rfind('.'):]=='.npy':
        a = np.load(inputfile)
        Q_values = a[0,:]
        n_values = a[1,:]
        lnL = a[2:,:]
        qgrid, ngrid = np.meshgrid(Q_values,n_values, indexing='ij')


    else: # ascii file
        n_Q = int(sys.argv[2])
        n_n = int(sys.argv[3])
        a = np.loadtxt(inputfile)
        qgrid = np.reshape(a[:,0],(n_Q, n_n))
        ngrid = np.reshape(a[:,1],(n_Q, n_n))
        lnL = np.reshape(a[:,2],(n_Q, n_n))
        Q_values = qgrid[:,0]
        n_values = ngrid[0,:]


    P = np.exp(lnL - np.max(lnL))
    dn = n_values[2] - n_values[1]
    dQ = Q_values[2] - Q_values[1]

    P = P/(np.sum(P)*dn*dQ)
    P_n = np.zeros(len(n_values))
    P_Q = np.zeros(len(Q_values))
    for i in range(len(n_values)):
        P_n[i] = np.sum(P[:,i])
        P_Q[i] = np.sum(P[i,:])
    P_Q = P_Q/ (sum(P_Q)*dQ)
    P_n = P_n / (sum(P_n)*dn)
    P_n_53 = [  1.18135983e-08, 1.66036605e-07, 1.78389677e-06, 1.41779689e-05, 8.16982002e-05,3.43434304e-04, 1.10365229e-03, 2.94768503e-03, 7.09085536e-03,1.59802063e-02,3.39894530e-02, 6.80880041e-02, 1.28009619e-01,2.25005204e-01,3.68232248e-01, 5.58590802e-01, 7.81674983e-01,1.00388335e+00, 1.17670401e+00, 1.25144110e+00, 1.19995992e+00,1.03039471e+00, 7.86676762e-01, 5.29916204e-01,3.12401379e-01,1.59851636e-01, 7.01609935e-02, 2.64560133e-02,8.19590385e-03,2.24971110e-03, 4.52339841e-04, 8.85676869e-05,1.21038921e-05,1.13117261e-06, 1.65186281e-07, 6.50726262e-09, 3.00880298e-10,3.19201217e-11,5.46077774e-13,1.72409027e-15]
    P_Q_53 = [1.46257343e-66, 9.51721227e-33, 1.21220074e-16, 7.96590516e-11, 1.93719592e-07, 2.46835633e-05, 5.60162065e-04, 4.33749366e-03, 1.66223615e-02, 3.95054150e-02, 6.72127237e-02, 9.00067855e-02, 1.01214645e-01, 1.00013885e-01, 8.97117084e-02, 7.48044593e-02, 5.90142767e-02, 4.46383227e-02, 3.27022805e-02, 2.33859708e-02, 1.64237767e-02, 1.13813351e-02, 7.81163118e-03, 5.32607457e-03, 3.61589382e-03, 2.44898724e-03, 1.65720406e-03, 1.12178639e-03, 7.60345421e-04, 5.16435675e-04, 3.51719220e-04, 2.40306426e-04,1.64775996e-04, 1.13426581e-04, 7.84026870e-05, 5.44277426e-05, 3.79525182e-05, 2.65847481e-05, 1.87078133e-05, 1.32259634e-05]
    plt.plot(Q_values, P_n_53,'r--', Q_values, P_n,'bs')
    plt.legend(["53 GHz", "90 GHz"])
    plt.xlabel("Q-values")
    plt.ylabel("P(Q)")
    plt.show()
    mu_n = sum(P_n*n_values)*dn
    sigma_n = np.sqrt(sum(P_n*(n_values-mu_n)**2)*dn)

    mu_Q = sum(P_Q*Q_values)*dQ
    sigma_Q = np.sqrt(sum(P_Q*(Q_values-mu_Q)**2)*dQ)

    print(mu_n)
    print(sigma_n)
    print(mu_Q)
    print(sigma_Q)

    lnL -= np.amax(lnL) # arbitrarily "normalizing" to make the numbers more manageable

    # For a Gaussian distribution, the 1, 2 and 3 sigma (68%, 95% and
    # 99.7%) confidence regions correspond to where -2 lnL increases by
    # 2.3, 6.17 and 11.8 from its minimum value. 0.1 is close to the
    # peak.
    #my_levels = [0.1, 2.3, 6.17, 11.8]
    #cs = plt.contourf(ngrid,qgrid, -2.*lnL, levels=my_levels)
    #plt.clabel(cs, inline=1, fontsize=10)
    #plt.grid()
    #plt.show()
