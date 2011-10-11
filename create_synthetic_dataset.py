import numpy as np

def swiss_roll(n, a=1, b=4, depth=5, do_rescale=True):
    """Return n random points laying in a swiss roll.

    The swiss roll manifold is the manifold typically used in manifold learning
    and other dimensionality reduction techniques. It is determined by the
    parametric equations

    x1 = pi * sqrt((b^2 - y^2)*t + a^2) * cos(pi * sqrt((b^2 - y^2)*t1 + a^2))
    x2 = depth * t2
    x3 = pi * sqrt((b^2 - y^2)*t + a^2) * sin(pi * sqrt((b^2 - y^2)*t1 + a^2))
    
    Parameters
    ----------
    n : Number of points
    a : Initial angle is a*pi
    b : End angle is b*pi
    depth: Depth of the roll
    do_rescale: If True, rescale to the plus/minus 1 range1
    
    Returns
    -------
    A 3-by-n ndarray [x1; x2; x3] with the points from the roll
    """
    y = np.random.rand(2, n)
    t = np.pi * np.sqrt((b*b - a*a) * y[0,:] + a*a)
    x2 = depth * y[1,:]
    x1 = t * np.cos(t)
    x3 = t * np.sin(t)

    if do_rescale:
        x1 = rescale(x1)
        x2 = rescale(x2)
        x3 = rescale(x3)
    
    #return (x1, x2, x3)
    return np.vstack([x1, x2, x3])

def rescale(x):
    """Rescale vector x into the [-1, 1] range.

    Parameters
    ----------
    x: 1 dimensional ndarray

    Returns:
    -------
    x: The original vector rescale to the range [-1, 1]
    """
    x -= x.mean()
    x /= np.max(np.abs(x))
    return x

if __name__ == '__main__':
    #(x1, x2, x3) = swiss_roll(1000)
    x = swiss_roll(1000)

    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x1, x2, x3, c='r')
    ax.scatter(x[0,:], x[1,:], x[2,:], c='r')

    plt.show()


