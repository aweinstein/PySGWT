import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sgwt

N = 16
G = nx.grid_2d_graph(N, N)
A = np.asarray(nx.adj_matrix(G, nodelist=sorted(G.nodes())))
L = sgwt.laplacian(A)

# Design filters for transform
N_scales = 4
l_max = sgwt.rough_l_max(L)
print 'Measuring the largest eigenvalue, l_max = %.2f' % l_max
print 'Designing transform in spectral domain'
(g, _, t) = sgwt.filter_design(l_max, N_scales)

arange = (0.0, l_max)

# Display filter design in spectral domain
#sgwt.view_design(g,t,arange);

# Chebyshev polynomial approximation
m = 50 # Order of polynomial approximation
print 'Computing Chebyshev polynomials of order %d for fast transform'  %m
c = []
for kernel in g:
    c.append(sgwt.cheby_coeff(kernel, m, m+1, arange))

# Compute transform of delta at one vertex
j_center = 135 # Vertex to center wavelets to be shown
print 'Computing forward transform of delta at vertex %d' % j_center
d = sgwt.delta(L.shape[0], j_center)
# forward transform, using chebyshev approximation
wp_all = sgwt.cheby_op(d, L, c, arange)


for n in range(N_scales + 1):
    alpha = wp_all[n].reshape((N,N))
    plt.matshow(alpha)
    plt.colorbar()

plt.show()
