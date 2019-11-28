from gapp import dgp,covariance
from numpy import loadtxt, savetxt
import numpy as np
import matplotlib.pyplot as plt
# 0 = Square expo
# 1 = Matern 7/2
# 2 = Matern 9/2
# 3 = Cauchy

# defining Hubble constant, its uncertainty and the density parameter
H_0 = 73.8
H_0_Sigma = 1.1
den_matter = 0.302

# load the data
(X, Y, Sigma) = loadtxt("gapp_data.txt", unpack = 'True')

# nstar points of the function will be reconstructed between xmin and xmax
xmin = 0.1
xmax = 3
nstar = 100

# loop for all covariance functions and nested loop for varying initial values of the hyperparameters
for kernels_iter in range (0, 4):

        theta_iter = 1
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

        # create plots for each kernel
        import plot_loop_update
        plot_loop_update.plot(X, Y, Sigma, rec, drec, d2rec, d3rec, kernels_iter, theta_iter)

        # load z, H, and uncertainty
        z,H, rec_Sigma = np.loadtxt("{0}_f_({1},{1}).txt".format(kernels_iter,theta_iter), unpack=True)

        # load H' and sigma_H'
        dH = np.loadtxt("{0}_df_({1},{1}).txt".format(kernels_iter,theta_iter))[:,1]
        rec_dSigma = np.loadtxt("{0}_df_({1},{1}).txt".format(kernels_iter,theta_iter))[:,2]

        # load H'' and sigma_H''
        d2H = np.loadtxt("{0}_d2f_({1},{1}).txt".format(kernels_iter,theta_iter))[:,1]
        rec_d2Sigma = np.loadtxt("{0}_d2f_({1},{1}).txt".format(kernels_iter,theta_iter))[:,2]

        # append the initial condition to z
        z_101 = np.insert(z, 0, 0)
        H_101 = np.insert(H,0,H_0)
        Sigma_101 = np.insert(rec_Sigma, H_0_Sigma,0)

        # import Torsion and f_z and their sigma_f and transposing them into columns
        import Torsion
        T_col = np.transpose([Torsion.Torsion(nstar,H_0,H_101)])
        import f_z
        f_col = np.transpose([f_z.function(nstar, H_0, den_matter, z_101, H, dH,T_col)])
        import Tf_Sigma
        S_col = np.transpose([Tf_Sigma.Tf_Sigma(nstar, H_0_Sigma,dH, d2H, H_101, H_0, den_matter, z_101,f_col,rec_Sigma,rec_dSigma)])

        #Save T,f,S as columns in a text file
        np.savetxt("{0}_T,f,S_({1},{1}).txt".format(kernels_iter, theta_iter), zip(T_col,f_col,S_col))

        # load the data
        (T, f, S) = loadtxt("{0}_T,f,S_({1},{1}).txt".format(kernels_iter, theta_iter), unpack = 'True')

        # define the min and max for T
        Tmin = 32000
        Tmax = 320000
        nstar = 100

        # intial values of the hyperparemeters
        Tf_initheta = np.array([100000,100000])

        # intialization of the Gaussian Process for different kernels
        if kernels_iter == 0:
            Tf_g = dgp.DGaussianProcess(T, f, S, cXstar=(Tmin, Tmax, nstar))
        elif  kernels_iter == 1:
            Tf_g = dgp.DGaussianProcess(T, f, S, covfunction = covariance.Matern72, cXstar=(Tmin, Tmax, nstar))
        elif  kernels_iter == 2:
            Tf_g = dgp.DGaussianProcess(T, f, S, covfunction = covariance.Matern92, cXstar=(Tmin, Tmax, nstar))
        elif kernels_iter == 3:
            Tf_g = dgp.DGaussianProcess(T, f, S, covfunction = covariance.Cauchy, cXstar=(Tmin, Tmax, nstar))

        # training of the hyperparameters and reconstruction of the function
        (Tf_rec, Tf_theta) = Tf_g.gp(theta=Tf_initheta)

        # reconstruction of the first, second and third derivatives.
        # theta is fixed to the previously determined value.
        (Tf_drec, Tf_theta) = Tf_g.dgp(thetatrain='False')
        (Tf_d2rec, Tf_theta) = Tf_g.d2gp()
        (Tf_d3rec, Tf_theta) = Tf_g.d3gp()

        savetxt("{0}_Tf_({1},{1}).txt".format(kernels_iter,theta_iter), Tf_rec)
        savetxt("{0}_Tdf_({1},{1}).txt".format(kernels_iter,theta_iter), Tf_drec)
        savetxt("{0}_Td2f_({1},{1}).txt".format(kernels_iter,theta_iter), Tf_d2rec)
        savetxt("{0}_Td3f_({1},{1}).txt".format(kernels_iter,theta_iter), Tf_d3rec)

        # creating the plot
        import Tf_plot
        Tf_plot.plot(T, f, S, Tf_rec, Tf_drec, Tf_d2rec, Tf_d3rec,Tmin, Tmax,kernels_iter,theta_iter)
