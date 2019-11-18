from gapp import dgp,covariance
from numpy import loadtxt, savetxt
import numpy as np
import matplotlib.pyplot as plt
# 0 = Square expo
# 1 = Matern 7/2
# 2 = Matern 9/2
# 3 = Cauchy

# defining H0 and density parameter
H_0 = 73.8
den_matter = 0.302

# load the data
(X, Y, Sigma) = loadtxt("gapp_data_updated.txt", unpack = 'True')

# nstar points of the function will be reconstructed between xmin and xmax
xmin = 0.1
xmax = 3
nstar = 100

# loop for all covariance functions and nested loop for varying initial values of the hyperparameters
for kernels_iter in range (0, 4):
    for theta_iter in range(1, 3):

        # intial values of the hyperparemeters
        initheta = np.array([theta_iter, theta_iter])

        # intialization of the Gaussian Process for different kernels
        if kernels_iter == 0:
            g = dgp.DGaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar))
        elif  kernels_iter == 1:
            g = dgp.DGaussianProcess(X, Y, Sigma, covfunction=covariance.Matern72, cXstar=(xmin, xmax, nstar))
        elif  kernels_iter == 2:
            g = dgp.DGaussianProcess(X, Y, Sigma, covfunction=covariance.Matern92, cXstar=(xmin, xmax, nstar))
        elif kernels_iter == 3:
            g = dgp.DGaussianProcess(X, Y, Sigma, covfunction=covariance.Cauchy, cXstar=(xmin, xmax, nstar))

        # training of the hyperparameters and reconstruction of the function
        (rec, theta) = g.gp(theta=initheta)

        # reconstruction of the first, second and third derivatives.
        # theta is fixed to the previously determined value.
        (drec, theta) = g.dgp(thetatrain='False')
        (d2rec, theta) = g.d2gp()
        (d3rec, theta) = g.d3gp()

        # save the output in different files
        savetxt("{0}_f_({1},{1}).txt".format(kernels_iter,theta_iter), rec)
        savetxt("{0}_df_({1},{1}).txt".format(kernels_iter,theta_iter), drec)
        savetxt("{0}_d2f_({1},{1}).txt".format(kernels_iter,theta_iter), d2rec)
        savetxt("{0}_d3f_({1},{1}).txt".format(kernels_iter,theta_iter), d3rec)

        # creating plots for each kernel
        import plot_loop_update
        plot_loop_update.plot(X, Y, Sigma, rec, drec, d2rec, d3rec, kernels_iter, theta_iter)

        z = np.loadtxt("{0}_f_({1},{1}).txt".format(kernels_iter, theta_iter))[:, 0]
        H = np.loadtxt("{0}_f_({1},{1}).txt".format(kernels_iter, theta_iter))[:, 1]
        dH = np.loadtxt("{0}_df_({1},{1}).txt".format(kernels_iter, theta_iter))[:, 1]

        # splitting z into an array and appending the initial condition
        z_array = np.split(z, 1)
        z_101 = np.insert(z_array, 0, 0)

        # importing Torsion and f_z
        import Torsion
        import f_z

        #transposing arrays to form columns and saving in a text file
        z_col = np.transpose([z_101])
        T_col = np.transpose([Torsion.Torsion(nstar,H_0,H)])
        f_col = np.transpose([f_z.function(nstar, H_0, den_matter, z_101, H, dH)])

        np.savetxt("{0}_z,T,f_({1},{1}).txt".format(kernels_iter, theta_iter), zip(z_col,T_col,f_col))


