import numpy as np

# function defining and calulating Sigma
def Tf_Sigma(nstar, H_0_Sigma,dH, d2H, H_101, H_0, den_matter, z_101,f_col,rec_Sigma,rec_dSigma):

    S = np.zeros([nstar+1])
    S[0] = H_0_Sigma

    for j in range(1, nstar+1):
        df_dz = 6*(dH[j-1]/H_101[j])*(H_101[j]**2 - H_0**2*den_matter*(1+z_101[j])**3 + f_col[j]/6)
        dz_dH = 1/dH[j-1]
        dz_d2H = 1/d2H[j-1]

        df_dH = abs(df_dz * dz_dH)
        df_d2H = abs(df_dz * dz_d2H)

        S[j] = (((df_dH)**2 * rec_Sigma[j-1]**2) + ((df_d2H)**2 * rec_dSigma[j-1]**2))**0.5

    return [S]
if __name__=="__main__":
    Tf_Sigma(nstar, H_0_Sigma,dH, d2H, H_101, H_0, den_matter, z_101,f_col,rec_Sigma,rec_dSigma)
