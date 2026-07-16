from scipy.sparse import coo_matrix


def smoothness_matrix(nx, ny, nz):
    """
    Create 3D first-order smoothness matrix.

    Parameters
    ----------
    nx, ny, nz : int
        Number of model cells in each direction

    Returns
    -------
    L : sparse matrix
        Smoothness operator
    """

    n = nx * ny * nz

    rows = []
    cols = []
    data = []

    counter = 0


    # X-direction smoothing
    for ix in range(nx - 1):
        for iy in range(ny):
            for iz in range(nz):

                m1 = ix*ny*nz + iy*nz + iz
                m2 = (ix+1)*ny*nz + iy*nz + iz

                rows += [counter, counter]
                cols += [m1, m2]
                data += [1, -1]

                counter += 1


    # Y-direction smoothing
    for ix in range(nx):
        for iy in range(ny - 1):
            for iz in range(nz):

                m1 = ix*ny*nz + iy*nz + iz
                m2 = ix*ny*nz + (iy+1)*nz + iz

                rows += [counter, counter]
                cols += [m1, m2]
                data += [1, -1]

                counter += 1


    # Z-direction smoothing
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz - 1):

                m1 = ix*ny*nz + iy*nz + iz
                m2 = ix*ny*nz + iy*nz + (iz+1)

                rows += [counter, counter]
                cols += [m1, m2]
                data += [1, -1]

                counter += 1


    L = coo_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(counter, n)
    )

    return L