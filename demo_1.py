import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.io import savemat

from create_synthetic_dataset import swiss_roll
import sgwt

def distanz(x):
    N = x.shape[1]
    xx = (x * x).sum(0).reshape(1,N)
    xz = np.dot(x.T, x)
    sq = np.tile(xx.T, (1, N))
    d = np.abs(sq - 2*xz + sq.T)

    return np.sqrt(d)

   
if __name__ == '__main__':

    load_data = True
    if load_data:
        x = np.load('x.npy')
        n_points = x.shape[1]
        print 'Loaded Swiss Roll point cloud with %d points' % n_points
    else:
        n_points = 500
        print 'Creating Swiss Roll point cloud with %d points' % n_points
        x = swiss_roll(n_points)
        np.save('x.npy', x)
        savemat('x.mat', {'x':x}, oned_as='row')
        
    print 'Computing edge weights and graph Laplacian'
    d = squareform(pdist(x.T))
    s = 0.1
    A = np.exp(-d**2 / (2 * s**2))
    L = sgwt.laplacian(A)

    # Design filters for transform
    N_scales = 4
    l_max = sgwt.rough_l_max(L)
    print 'Measuring the largest eigenvalue, l_max = %.2f' % l_max
    print 'Designing transform in spectral domain'
    (g, _, t) = sgwt.filter_design(l_max, N_scales)

    arange = (0.0, l_max)

    # TODO
    # Display filter design in spectral domain
    sgwt.view_design(g,t,arange);

    # Chebyshev polynomial approximation
    m = 50 # Order of polynomial approximation
    print 'Computing Chebyshev polynomials of order %d for fast transform'  %m
    c = []
    for kernel in g:
        c.append(sgwt.cheby_coeff(kernel, m, m+1, arange))

    # Compute transform of delta at one vertex
    j_center = 32 - 1 # Vertex to center wavelets to be shown
    print 'Computing forward transform of delta at vertex %d' % j_center
    N = L.shape[0]
    d = sgwt.delta(N, j_center)
    # forward transform, using chebyshev approximation
    wp_all = sgwt.cheby_op(d, L, c, arange)

    plt.figure()
    print 'Plotting...'
    for i in range(N_scales + 1):
        plt.subplot(N_scales + 1, 1, i+1)
        plt.plot(wp_all[i])
        
    plt.show()
    
    print 'Done!'
