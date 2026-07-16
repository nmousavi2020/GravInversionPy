import numpy as np

from scipy.sparse.linalg import cg, LinearOperator


def tikhonov_inversion(
    x_obs,
    y_obs,
    gz_obs,
    Xc,
    Yc,
    Zc,
    cell_volume,
    L,
    nx,
    ny,
    nz,
    lambda_reg=1e-3,
    maxiter=600
):
    """
    Perform 3D Tikhonov gravity inversion.

    Parameters
    ----------
    x_obs, y_obs : ndarray
        Observation coordinates (km)

    gz_obs : ndarray
        Observed gravity data

    Xc,Yc,Zc : ndarray
        Model grid coordinates

    cell_volume : float
        Voxel volume (m3)

    L : sparse matrix
        Smoothness operator

    nx,ny,nz : int
        Model dimensions

    lambda_reg : float
        Regularization strength

    Returns
    -------
    model : ndarray
        3D density model

    info : int
        CG solver status
    """

    n_model = nx * ny * nz


    from .forward import (
        forward_gravity,
        transpose_gravity
    )


    def A_mv(m):

        model = m.reshape(
            (nx, ny, nz)
        )


        data_term = transpose_gravity(
            forward_gravity(
                model,
                Xc,
                Yc,
                Zc,
                x_obs,
                y_obs,
                cell_volume
            ),
            Xc,
            Yc,
            Zc,
            x_obs,
            y_obs,
            cell_volume
        )


        smooth_term = (
            lambda_reg**2 *
            (L.T @ (L @ m))
        )


        return data_term + smooth_term



    A = LinearOperator(
        (
            n_model,
            n_model
        ),
        matvec=A_mv
    )


    print("Solving Tikhonov inversion...")


    m0 = np.zeros(
        n_model
    )


    m_est, info = cg(
        A,
        transpose_gravity(
            gz_obs,
            Xc,
            Yc,
            Zc,
            x_obs,
            y_obs,
            cell_volume
        ),
        x0=m0,
        maxiter=maxiter
    )


    model = m_est.reshape(
        (nx, ny, nz)
    )


    return model, info