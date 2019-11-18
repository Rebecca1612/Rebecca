from gapp import dgp,covariance
from numpy import loadtxt, savetxt
import numpy as np
import matplotlib.pyplot as plt

# load the data
(X, Y, Sigma) = loadtxt("gapp_data_updated.txt", unpack = 'True')

# nstar points of the function will be reconstructed between xmin and xmax
xmin = 0
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

        # creating one plot for each kernel
        import plot_loop_update
        plot_loop_update.plot(X, Y, Sigma, rec, drec, d2rec, d3rec,kernels_iter,theta_iter)


# 0 = Square expo
# 1 = Matern 7/2
# 2 = Matern 9/2
# 3 = Cauchy
