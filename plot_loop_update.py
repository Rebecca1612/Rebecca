import matplotlib.pyplot as plt
from numpy import loadtxt

def plot(X, Y, Sigma, rec, drec, d2rec, d3rec,kernels_iter,theta_iter):

    plt.subplot(221)
    plt.xlim(0, 3)
    plt.ylim(0,300)
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                     facecolor='lightblue')
    plt.plot(rec[:, 0], rec[:, 1])
    #plt.errorbar(X, Y, Sigma, color='red', fmt='_')
    plt.xlabel('z')
    plt.ylabel('H(z)')

    plt.subplot(222)
    plt.xlim(0, 3)
    plt.ylim(-20,100)
    plt.fill_between(drec[:, 0], drec[:, 1] + drec[:, 2],
                     drec[:, 1] - drec[:, 2], facecolor='lightblue')
    plt.plot(drec[:, 0], drec[:, 1])
    plt.xlabel('z')
    plt.ylabel("H'(z)")
    plt.ylabel.labelpad = 25

    plt.subplot(223)
    plt.xlim(0, 3)
    plt.fill_between(d2rec[:, 0], d2rec[:, 1] + d2rec[:, 2],
                     d2rec[:, 1] - d2rec[:, 2], facecolor='lightblue')
    plt.plot(d2rec[:, 0], d2rec[:, 1])
    plt.xlabel('z')
    plt.ylabel("H''(z)")

    plt.subplot(224)
    plt.xlim(0, 3)
    plt.fill_between(d3rec[:, 0], d3rec[:, 1] + d3rec[:, 2],
                     d3rec[:, 1] - d3rec[:, 2], facecolor='lightblue')
    plt.plot(d3rec[:, 0], d3rec[:, 1])
    plt.xlabel('z')
    plt.ylabel("H'''(z)")

    if kernels_iter == 1:
        plt.suptitle('Matern72_({0},{0})'.format(theta_iter))
        plt.savefig('Matern72_plot_({0},{0}).pdf'.format(theta_iter))
    elif kernels_iter == 0:
        plt.suptitle('Squared_Exponential_({0},{0})'.format(theta_iter))
        plt.savefig('Squared_Exonential_plot_({0},{0}).pdf'.format(theta_iter))
    elif kernels_iter == 2:
        plt.suptitle('Matern92_({0},{0})'.format(theta_iter))
        plt.savefig('Matern92_plot_({0},{0}).pdf'.format(theta_iter))
    else:
        plt.suptitle('Cauchy_({0},{0})'.format(theta_iter))
        plt.savefig('Cauchy_plot_({0},{0}).pdf'.format(theta_iter))

    plt.clf()

if __name__=="__main__":
    (X,Y,Sigma) = loadtxt("gapp_data.txt", unpack='True')
    rec = loadtxt("{0}_f_({1},{1}).txt".format(kernels_iter, theta_iter))
    drec = loadtxt("{0}_df_({1},{1}).txt".format(kernels_iter, theta_iter))
    d2rec = loadtxt("{0}_d2f_({1},{1}).txt".format(kernels_iter, theta_iter))
    d3rec = loadtxt("{0}_d3f_({1},{1}).txt".format(kernels_iter, theta_iter))

    plot(X, Y, Sigma, rec, drec, d2rec, d3rec, kernels_iter, theta_iter)
