import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import fminbound
import scipy.sparse.linalg as ssl
import matplotlib.pylab as plt

def laplacian(A, laplacian_type='raw'):
    """Compute graph laplacian from connectivity matrix.

    Parameters
    ----------
    A : Adjancency matrix
    
    Return
    ------
    L : Graph Laplacian as a lil (list of lists) sparse matrix
    """

    N = A.shape[0]
    # TODO: Raise exception if A is not square

    degrees = A.sum(1)
    # To deal with loops, must extract diagonal part of A
    diagw = np.diag(A)

    # w will consist of non-diagonal entries only
    ni2, nj2 = A.nonzero()
    w2 = A[ni2, nj2]
    ndind = (ni2 != nj2).nonzero() # Non-diagonal indices
    ni = ni2[ndind]
    nj = nj2[ndind]
    w = w2[ndind]

    di = np.arange(N) # diagonal indices

    if laplacian_type == 'raw':
        # non-normalized laplaciand L = D - A
        L = np.diag(degrees - diagw)
        L[ni, nj] = -w
        L = lil_matrix(L)
    elif laplacian_type == 'normalized':
        # TODO: Implement the normalized laplacian case
        #   % normalized laplacian D^(-1/2)*(D-A)*D^(-1/2)
        #   % diagonal entries
        #   dL=(1-diagw./degrees); % will produce NaN for degrees==0 locations
        #   dL(degrees==0)=0;% which will be fixed here
        #   % nondiagonal entries
        #   ndL=-w./vec( sqrt(degrees(ni).*degrees(nj)) );
        #   L=sparse([ni;di],[nj;di],[ndL;dL],N,N);
        print 'Not implemented'
    else:
        # TODO: Raise an exception
        print "Don't know what to do"

    return L

def rough_l_max(L):
    """Return a rough upper bound on the maximum eigenvalue of L.

    Parameters
    ----------
    L: Symmetric matrix

    Return
    ------
    l_max_ub: An upper bound of the maximum eigenvalue of L.
    """
    # TODO: Check if L is sparse or not, and handle the situation accordingly

    l_max = np.linalg.eigvalsh(L.todense()).max()

    # TODO: Fix this
    # At least for demo_1, this is much slower
    #l_max = ssl.arpack.eigsh(L, k=1, return_eigenvectors=False,
    #                         tol=5e-3, ncv=10)

    l_max_ub =  1.01 * l_max
    return l_max_ub
    
def set_scales(l_min, l_max, N_scales):
    """Compute a set of wavelet scales adapted to spectrum bounds.

    Returns a (possibly good) set of wavelet scales given minimum nonzero and
    maximum eigenvalues of laplacian.

    Returns scales logarithmicaly spaced between minimum and maximum
    'effective' scales : i.e. scales below minumum or above maximum will yield
    the same shape wavelet (due to homogoneity of sgwt kernel : currently
    assuming sgwt kernel g given as abspline with t1=1, t2=2)

    Parameters
    ----------
    l_min: minimum non-zero eigenvalue of the laplacian.
       Note that in design of transform with  scaling function, lmin may be
       taken just as a fixed fraction of lmax,  and may not actually be the
       smallest nonzero eigenvalue
    l_max: maximum eigenvalue of the laplacian
    N_scales: Number of wavelets scales

    Returns
    -------
    s: wavelet scales
    """
    t1=1
    t2=2
    s_min = t1 / l_max
    s_max = t2 / l_min
    # Scales should be decreasing ... higher j should give larger s
    s = np.exp(np.linspace(np.log(s_max), np.log(s_min), N_scales));

    return s

def kernel(x, g_type='abspline', a=2, b=2, t1=1, t2=2):
    """Compute sgwt kernel.

    This function will evaluate the kernel at input x

    Parameters
    ----------
    x : independent variable values
    type : 'abspline' gives polynomial / spline / power law decay kernel
    a : parameters for abspline kernel, default to 2
    b : parameters for abspline kernel, default to 2
    t1 : parameters for abspline kernel, default to 1
    t2 : parameters for abspline kernel, default to 2

    Returns
    -------
    g : array of values of g(x)
    """
    if g_type == 'abspline':
        g = kernel_abspline3(x, a, b, t1, t2)
    elif g_type == 'mh':
        g = x * np.exp(-x)
    else:
        print 'unknown type'
        #TODO Raise exception

    return g

def kernel_derivative(x, a, b, t1, t2):
    """Note: Note implemented in the MATLAB version."""
    return x

def kernel_abspline3(x, alpha, beta, t1, t2):
    """Monic polynomial / cubic spline / power law decay kernel

    Defines function g(x) with g(x) = c1*x^alpha for 0<x<x1
    g(x) = c3/x^beta for x>t2
    cubic spline for t1<x<t2,
    Satisfying g(t1)=g(t2)=1

    Parameters
    ----------
    x : array of independent variable values
    alpha : exponent for region near origin
    beta : exponent decay
    t1, t2 : determine transition region


    Returns
    -------
    r : result (same size as x)
"""
    # Convert to array if x is scalar, so we can use fminbound
    if np.isscalar(x):
        x = np.array(x, ndmin=1)

    r = np.zeros(x.size)

    # Compute spline coefficients
    # M a = v
    M = np.array([[1, t1, t1**2, t1**3],
                  [1, t2, t2**2, t2**3],
                  [0, 1, 2*t1, 3*t1**2],
                  [0, 1, 2*t2, 3*t2**2]])
    v = np.array([[1],
                  [1],
                  [t1**(-alpha) * alpha * t1**(alpha - 1)],
                  [-beta * t2**(-beta - 1) * t2**beta]])
    a = np.linalg.lstsq(M, v)[0]

    r1 = np.logical_and(x>=0, x<t1).nonzero()
    r2 = np.logical_and(x>=t1, x<t2).nonzero()
    r3 = (x>=t2).nonzero()
    r[r1] = x[r1]**alpha * t1**(-alpha)
    r[r3] = x[r3]**(-beta) * t2**(beta)
    x2 = x[r2]
    r[r2] = a[0]  + a[1] * x2 + a[2] * x2**2 + a[3] * x2**3

    return r

  
def filter_design(l_max, N_scales, design_type='default', lp_factor=20,
                  a=2, b=2, t1=1, t2=2):
    """Return list of scaled wavelet kernels and derivatives.
    
    g[0] is scaling function kernel, 
    g[1],  g[Nscales] are wavelet kernels

    Parameters
    ----------
    l_max : upper bound on spectrum
    N_scales : number of wavelet scales
    design_type: 'default' or 'mh'
    lp_factor : lmin=lmax/lpfactor will be used to determine scales, then
       scaling function kernel will be created to fill the lowpass gap. Default
       to 20.

    Returns
    -------
    g : scaling and wavelets kernel
    gp : derivatives of the kernel (not implemented / used)
    t : set of wavelet scales adapted to spectrum bounds
    """
    g = []
    gp = []
    l_min = l_max / lp_factor
    t = set_scales(l_min, l_max, N_scales)
    if design_type == 'default':
        # Find maximum of gs. Could get this analytically, but this also works
        f = lambda x: -kernel(x, a=a, b=b, t1=t1, t2=t2)
        x_star = fminbound(f, 1, 2)
        gamma_l = -f(x_star)
        l_min_fac = 0.6 * l_min
        g.append(lambda x: gamma_l * np.exp(-(x / l_min_fac)**4))
        gp.append(lambda x: -4 * gamma_l * (x/l_min_fac)**3 *
                  np.exp(-(x / l_min_fac)**4) / l_min_fac)
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, a=a, b=b, t1=t1,t2=t2))
            gp.append(lambda x,s=scale: kernel_derivative(scale * x) * s)
    elif design_type == 'mh':
        l_min_fac = 0.4 * l_min
        g.append(lambda x: 1.2 * np.exp(-1) * np.exp(-(x/l_min_fac)**4))
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, g_type='mh'))
    else:
        print 'Unknown design type'
        # TODO: Raise exception
        
    return (g, gp, t)

def cheby_coeff(g, m, N=None, arange=(-1,1)):
    """ Compute Chebyshev coefficients of given function.

    Parameters
    ----------
    g : function handle, should define function on arange
    m : maximum order Chebyshev coefficient to compute
    N : grid order used to compute quadrature (default is m+1)
    arange : interval of approximation (defaults to (-1,1) )

    Returns
    -------
    c : list of Chebyshev coefficients, ordered such that c(j+1) is 
      j'th Chebyshev coefficient
    """
    if N is None:
        N = m+1

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N+1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m+1)
    for j in range(m+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c

def delta(N, j):
    """Return vector with one nonzero entry equal to 1.

    Returns length N vector with r[j]=1, all others zero

    Parameters
    ----------
    N : length of vector
    j : position of "delta" impulse

    Returns
    -------
    r : returned vector
    """
    r = np.zeros((N,1))
    r[j] = 1
    return r

def cheby_op(f, L, c, arange):
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev
    basis) applied to input.

    Coefficients for multiple polynomials may be passed as a lis. This
    is equivalent to setting
    r[0] = cheby_op(f, L, c[0], arange)
    r[1] = cheby_op(f, L, c[1], arange)
    ...
 
    but is more efficient as the Chebyshev polynomials of L applied to f can be
    computed once and shared.

    Parameters
    ----------
    f : input vector
    L : graph laplacian (should be sparse)
    c : Chebyshev coefficients. If c is a plain array, then they are
       coefficients for a single polynomial. If c is a list, then it contains
       coefficients for multiple polynomials, such  that c[j](1+k) is k'th
       Chebyshev coefficient the j'th polynomial.
    arange : interval of approximation

    Returns
    -------
    r : If c is a list, r will be a list of vectors of size of f. If c is
       a plain array, r will be a vector the size of f.    
    """
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op(f, L, [c], arange)
        return r[0]

    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0

    Twf_old = f
    Twf_cur = (L*f - a2*f) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        Twf_new = (2/a1) * (L*Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k+1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r

def framebounds(g, lmin, lmax):
    """

    Parameters
    ----------
    g : function handles computing sgwt scaling function and wavelet
       kernels
    lmin, lmax : minimum nonzero, maximum eigenvalue

    Returns
    -------
    A , B : frame bounds
    sg2 : array containing sum of g(s_i*x)^2 (for visualization)
    x : x values corresponding to sg2
    """
    N = 1e4 # number of points for line search
    x = np.linspace(lmin, lmax, N)
    Nscales = len(g)

    sg2 = np.zeros(x.size)
    for ks in range(Nscales):
        sg2 += (g[ks](x))**2

    A = np.min(sg2)
    B = np.max(sg2)

    return (A, B, sg2, x)

def view_design(g, t, arange):
    """Plot the scaling and wavelet kernel.

    Plot the input scaling function and wavelet kernels, indicates the wavelet
    scales by legend, and also shows the sum of squares G and corresponding
    frame bounds for the transform.

    Parameters
    ----------
    g : list of  function handles for scaling function and wavelet kernels
    t : array of wavelet scales corresponding to wavelet kernels in g
    arange : approximation range

    Returns
    -------
    h : figure handle
    """
    x = np.linspace(arange[0], arange[1], 1e3)
    h = plt.figure()
    
    J = len(g) 
    G = np.zeros(x.size)

    for n in range(J):
        if n == 0:
            lab = 'h'
        else:
            lab = 't=%.2f' % t[n-1]
        plt.plot(x, g[n](x), label=lab)
        G += g[n](x)**2

    plt.plot(x, G, 'k', label='G')

    (A, B, _, _) = framebounds(g, arange[0], arange[1])
    plt.axhline(A, c='m', ls=':', label='A')
    plt.axhline(B, c='g', ls=':', label='B')
    plt.xlim(arange[0], arange[1])

    plt.title('Scaling function kernel h(x), Wavelet kernels g(t_j x) \n'
              'sum of Squares G, and Frame Bounds')
    plt.yticks(np.r_[0:4])
    plt.ylim(0, 3)
    plt.legend()

    return h


if __name__ == '__main__':
    from scipy.linalg import circulant
    import matplotlib.pyplot as plt
    
    N = 256
    jcenter = N/2 - 1
    d = delta(N, jcenter)
    circ = np.zeros(N)
    circ[0] = 2
    circ[1] = 1
    circ[-1] = 1
    L = circulant(circ)
    L = lil_matrix(L)
    lmax = rough_l_max(L)
    lmax = 4.039353399475705
    Nscales = 4
    (g, gp, t) = filter_design(lmax, Nscales)
    m = 50
    arange = (0, lmax)
    c = [cheby_coeff(g[i], m, m+1, arange) for i in range(len(g))]
    wpall = cheby_op(d, L, c, arange)

    for i in range(Nscales + 1):
        plt.subplot(Nscales + 1, 1, i+1)
        plt.plot(wpall[i])
    plt.show()

    
    print 'Done'
    
    
if __name__ == '__main__x':
    x = np.linspace(0, 1, 10)
    lmax = 6.2
    (g, gp, t) = filter_design(lmax, 4)
    ## print g[1](x)
    ## (g, gp) = filter_design(6.2, 4, design_type='mh')
    ## print g[0](x)
    ## print g[1](x)
    m = 50
    arange = (0, lmax)
    c = cheby_coeff(g[1], m, m+1, arange)
    #print c
    
