import numpy as np

def Torsion(nstar,H_0,H_101):

    T = np.zeros([nstar+1])
    T[0] = 6*H_0**2
    #print "initial torsion = " , T[0]

    for torsion in range (1,nstar+1):
        T[torsion] = 6*(H_101[torsion])**2

    return [T]

if __name__=="__main__":
    Torsion(nstar,H_0,H_101)
