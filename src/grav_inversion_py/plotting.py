import matplotlib.pyplot as plt


def plot_observed_gravity(
    x,
    y,
    gz,
    filename="outputs/observed_gravity.png"
):
    """
    Plot observed gravity data.

    Parameters
    ----------
    x : ndarray
        X coordinates (km)

    y : ndarray
        Y coordinates (km)

    gz : ndarray
        Gravity anomaly (mGal)

    filename : str
        Output PNG file
    """

    plt.figure(figsize=(8, 6), dpi=300)

    sc = plt.scatter(
        x,
        y,
        c=gz,
        cmap="RdBu_r",
        s=25
    )

    plt.colorbar(
        sc,
        label="Gravity gz (mGal)"
    )

    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title("Observed Gravity Data")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
def plot_density_slice(
    density_model,
    depth_index=None,
    filename="outputs/density_slice.png"
):
    """
    Plot a horizontal slice of inverted 3D density model.

    Parameters
    ----------
    density_model : ndarray
        3D inverted density model

    depth_index : int
        Z slice index

    filename : str
        Output image name
    """

    import numpy as np
    import matplotlib.pyplot as plt


    nx, ny, nz = density_model.shape


    if depth_index is None:
        depth_index = nz // 2


    slice_density = density_model[:, :, depth_index]


    plt.figure(
        figsize=(7, 6),
        dpi=300
    )


    im = plt.imshow(
        slice_density.T,
        origin="lower",
        cmap="viridis",
        aspect="auto"
    )


    plt.colorbar(
        im,
        label="Density (kg/m$^3$)"
    )


    plt.xlabel("X cell")
    plt.ylabel("Y cell")

    plt.title(
        f"Inverted Density Slice (Z index={depth_index})"
    )


    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()
def plot_observed_vs_forward(
    gz_obs,
    gz_pred,
    filename="outputs/observed_vs_forward.png"
):
    """
    Plot observed gravity versus forward calculated gravity.
    """

    import numpy as np
    import matplotlib.pyplot as plt


    rms = np.sqrt(
        np.mean(
            (gz_obs - gz_pred)**2
        )
    )


    plt.figure(
        figsize=(7, 6),
        dpi=300
    )


    plt.scatter(
        gz_obs,
        gz_pred,
        s=35,
        alpha=0.7,
        label=f"RMS = {rms:.3f} mGal"
    )


    minimum = min(
        gz_obs.min(),
        gz_pred.min()
    )

    maximum = max(
        gz_obs.max(),
        gz_pred.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="1:1 line"
    )


    plt.xlabel(
        "Observed Gravity (mGal)"
    )

    plt.ylabel(
        "Forward Gravity (mGal)"
    )


    plt.title(
        "Observed vs Forward Gravity"
    )


    plt.legend()
    plt.grid(alpha=0.3)


    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()
def plot_forward_contour(
    x,
    y,
    gz,
    filename="outputs/gravity_forward_contour.png"
):
    """
    Plot interpolated forward gravity contour map.

    Parameters
    ----------
    x, y : ndarray
        Observation coordinates (km)

    gz : ndarray
        Forward calculated gravity (mGal)

    filename : str
        Output PNG file
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata


    xi = np.linspace(
        x.min(),
        x.max(),
        100
    )

    yi = np.linspace(
        y.min(),
        y.max(),
        100
    )


    XI, YI = np.meshgrid(
        xi,
        yi
    )


    ZI = griddata(
        (x, y),
        gz,
        (XI, YI),
        method="cubic"
    )


    plt.figure(
        figsize=(7, 5),
        dpi=300
    )


    cf = plt.contourf(
        XI,
        YI,
        ZI,
        levels=20,
        cmap="RdBu_r"
    )


    plt.colorbar(
        cf,
        label="Forward Gravity gz (mGal)"
    )


    plt.xlabel(
        "X (km)"
    )

    plt.ylabel(
        "Y (km)"
    )


    plt.title(
        "Forward Calculated Gravity"
    )


    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()