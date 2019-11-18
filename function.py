from gapp import dgp,covariance
from numpy import loadtxt, savetxt
import numpy as np
import matplotlib.pyplot as plt

# load the data
(X, Y, Sigma) = loadtxt("gapp_data_updated.txt", unpack = 'True')
print X
# nstar points of the function will be reconstructed between xmin and xmax
xmin = 0.1
xmax = 3
nstar = 100

# intial values of the hyperparemeters
initheta = np.array([1,1])

# intialization of the Gaussian Process
g = dgp.DGaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

# reconstruction of the first, second and third derivatives.
# theta is fixed to the previously determined value.
(drec, theta) = g.dgp(thetatrain='False')
(d2rec, theta) = g.d2gp()
(d3rec, theta) = g.d3gp()

#save the output
savetxt("f_(1,1).txt", rec)
savetxt("df_(1,1).txt", drec)
savetxt("d2f_(1,1).txt", d2rec)
savetxt("d3f_(1,1).txt", d3rec)

# creating the plot
import plot
plot.plot(X, Y, Sigma, rec, drec, d2rec, d3rec)

z,H, rec_Sigma = np.loadtxt("f_(1,1).txt", unpack=True)
dH = np.loadtxt("df_(1,1).txt")[:, 1]

# splitting z into an array and appending the initial condition
z_array = np.split(z,1)
z_101 = np.insert(z_array,0,0)

# defining H0 and density parameter
H_0 = 73.8
den_matter = 0.302

# function to find values for Torsion with initial condition
def Torsion():

    T = np.zeros([nstar+1])
    T[0] = -6*H_0**2
    #print "initial torsion = " , T[0]

    for torsion in range (0,nstar):
        T[torsion +1] = -6*H[torsion]**2

    return [T]

# function which finds values for f(z) 
def function():

    f = np.zeros([nstar+1])
    f[0] = -6*H_0**2*(1-den_matter)
    #print "initial = ", f[0]

    for i in range(0, nstar):
        F = 6*(z_101[i+1]-z_101[i])*(dH[i]/ H[i])
        G = H[i]**2 - H_0**2*den_matter*(1+z_101[i])**3 + f[i]/6
        f[i+1] = (F*G)+f[i]

    return [f]

print function()
z_col = np.transpose([z_101])
T_col = np.transpose([Torsion()])
f_col = np.transpose([function()])

np.savetxt('z,T,f.txt', zip(z_col,T_col,f_col))

(Q, W, E) = loadtxt("z,T,f.txt", unpack = 'True')



# initheta_1 = np.array([1,1])
#
# # intialization of the Gaussian Process
# g = dgp.DGaussianProcess(W, E, cXstar=(xmin, xmax, nstar))
#
# # training of the hyperparameters and reconstruction of the function
# (rec_1, theta_1) = g.gp(theta=initheta_1)
# print(rec)
# # reconstruction of the first, second and third derivatives.
# # theta is fixed to the previously determined value.
# (drec_1, theta_1) = g.dgp(thetatrain='False')
# (d2rec_1, theta_1) = g.d2gp()
# (d3rec_1, theta_1) = g.d3gp()
#
# #save the output
# savetxt("f_(1,1)_1.txt", rec)
# savetxt("df_(1,1)_1.txt", drec)
# savetxt("d2f_(1,1)_1.txt", d2rec)
# savetxt("d3f_(1,1)_1.txt", d3rec)
#
# plot.plot(Q, W, E, rec_1, drec_1, d2rec_1, d3rec_1)
