import numpy as np


G_CONST = 6.67430e-11


def forward_gravity(
    density_model,
    Xc,
    Yc,
    Zc,
    x_obs,
    y_obs,
    cell_volume
):
    """
    Calculate forward gravity response.

    Parameters
    ----------
    density_model : ndarray
        3D density model (kg/m3)

    Xc, Yc, Zc : ndarray
        Model voxel coordinates (km)

    x_obs, y_obs : ndarray
        Observation coordinates (km)

    cell_volume : float
        Cell volume in m3

    Returns
    -------
    gz : ndarray
        Calculated gravity response
    """

    m_flat = density_model.ravel()[:, None]

    dx = Xc.ravel()[:, None] - x_obs[None, :]
    dy = Yc.ravel()[:, None] - y_obs[None, :]

    dz = Zc.ravel()[:, None]

    r3 = (
        dx**2 +
        dy**2 +
        dz**2
    )**1.5 + 1e-12


    gz = (
        G_CONST *
        cell_volume *
        (m_flat * dz / r3).sum(axis=0)
    )

    return gz



def transpose_gravity(
    residual,
    Xc,
    Yc,
    Zc,
    x_obs,
    y_obs,
    cell_volume
):
    """
    Calculate transpose operation G.T @ residual.
    """

    dx = Xc.ravel()[:, None] - x_obs[None, :]
    dy = Yc.ravel()[:, None] - y_obs[None, :]

    dz = Zc.ravel()[:, None]


    r3 = (
        dx**2 +
        dy**2 +
        dz**2
    )**1.5 + 1e-12


    result = (
        G_CONST *
        cell_volume *
        dz / r3
    ) @ residual


    return result.ravel()