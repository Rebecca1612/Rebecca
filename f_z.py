import numpy as np

def function(nstar, H_0, den_matter, z_101, H, dH,T_col):

    f = np.zeros([nstar+1])
    f[0] = -6*(H_0**2)*(1-den_matter) - T_col[0]

    for i in range(0, nstar):
        F = 6*(z_101[i+1]-z_101[i])*(dH[i]/ H[i])
        G = H[i]**2 - H_0**2*den_matter*(1+z_101[i])**3 - T_col[i]/6 + f[i]/6
        f[i+1] = (F*G)+f[i]

    return [f]

if __name__=="__main__":
    function(nstar, H_0, den_matter, z_101,H, dH,T_col)
