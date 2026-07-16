import numpy as np


def create_model_grid(
    nx=25,
    ny=22,
    nz=7,
    dx=2,
    dy=2,
    dz=2
):
    """
    Create 3D inversion model grid.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells in X, Y, Z directions

    dx, dy, dz : float
        Cell dimensions in km

    Returns
    -------
    Xc, Yc, Zc : ndarray
        3D voxel center coordinates

    cell_volume : float
        Cell volume in m^3

    n_model : int
        Total number of model parameters
    """

    xs = np.linspace(
        -10,
        dx*(nx-1)+10,
        nx
    )

    ys = np.linspace(
        -10,
        dy*(ny-1)+10,
        ny
    )

    zs = np.linspace(
        0,
        dz*(nz-1),
        nz
    )


    Xc, Yc, Zc = np.meshgrid(
        xs,
        ys,
        zs,
        indexing="ij"
    )


    # km^3 to m^3
    cell_volume = (
        dx * dy * dz * 1e9
    )


    n_model = nx * ny * nz


    return (
        Xc,
        Yc,
        Zc,
        cell_volume,
        n_model
    )