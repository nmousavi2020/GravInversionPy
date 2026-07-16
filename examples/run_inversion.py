import numpy as np

from grav_inversion_py.io import load_gravity_data
from grav_inversion_py.model import create_model_grid
from grav_inversion_py.forward import forward_gravity
from grav_inversion_py.regularization import smoothness_matrix
from grav_inversion_py.inversion import tikhonov_inversion
from grav_inversion_py.plotting import plot_density_slice
from grav_inversion_py.plotting import plot_observed_vs_forward
from grav_inversion_py.plotting import plot_forward_contour

# =====================================
# 1. Load gravity data
# =====================================

x_full, y_full, gz_full = load_gravity_data(
    "data/Gravity_syn.txt"
)


print(
    f"Loaded {len(gz_full)} gravity points"
)


# =====================================
# 2. Regular grid subsampling
# =====================================

NX_full = 106
NY_full = 90

sub_factor_x = 3
sub_factor_y = 3


x_grid = x_full.reshape(
    NX_full,
    NY_full
)

y_grid = y_full.reshape(
    NX_full,
    NY_full
)

gz_grid = gz_full.reshape(
    NX_full,
    NY_full
)


x_obs = x_grid[
    ::sub_factor_x,
    ::sub_factor_y
].ravel()

y_obs = y_grid[
    ::sub_factor_x,
    ::sub_factor_y
].ravel()

gz_obs = gz_grid[
    ::sub_factor_x,
    ::sub_factor_y
].ravel()


print(
    f"Using {len(gz_obs)} points for inversion"
)


# =====================================
# 3. Create model grid
# =====================================

NX = 25
NY = 22
NZ = 7


Xc, Yc, Zc, volume, n_model = create_model_grid(
    NX,
    NY,
    NZ,
    dx=2,
    dy=2,
    dz=2
)


print(
    "Model cells:",
    n_model
)


# =====================================
# 4. Regularization matrix
# =====================================

L = smoothness_matrix(
    NX,
    NY,
    NZ
)


# =====================================
# 5. Tikhonov inversion
# =====================================

density_model, info = tikhonov_inversion(
    x_obs,
    y_obs,
    gz_obs,
    Xc,
    Yc,
    Zc,
    volume,
    L,
    NX,
    NY,
    NZ,
    lambda_reg=1e-3
)


print(
    "CG information:",
    info
)


# =====================================
# 6. Forward gravity calculation
# =====================================

gz_pred = forward_gravity(
    density_model,
    Xc,
    Yc,
    Zc,
    x_obs,
    y_obs,
    volume
)


residual = gz_obs - gz_pred

rms = np.sqrt(
    np.mean(
        residual**2
    )
)


print(
    f"RMS misfit = {rms:.4f} mGal"
)


# =====================================
# 7. Save density model
# =====================================

density_output = np.column_stack(
    [
        Xc.ravel(),
        Yc.ravel(),
        Zc.ravel(),
        density_model.ravel()
    ]
)


np.savetxt(
    "outputs/Density_calc.txt",
    density_output,
    header="X(km) Y(km) Z(km) Density(kg/m3)",
    fmt="%.6f"
)


# =====================================
# 8. Save predicted gravity
# =====================================

gravity_output = np.column_stack(
    [
        x_obs,
        y_obs,
        gz_pred
    ]
)


np.savetxt(
    "outputs/Gravity_calc.txt",
    gravity_output,
    header="X(km) Y(km) gz_pred(mGal)",
    fmt="%.6f"
)


print("✔ Inversion completed")
print("✔ Results saved in outputs/")

# =====================================
# 9. Plot inverted density slice
# =====================================

plot_density_slice(
    density_model,
    depth_index=NZ//2,
    filename="outputs/density_slice.png"
)

print("✔ Density plot saved")

# =====================================
# 10. Observed vs Forward plot
# =====================================

plot_observed_vs_forward(
    gz_obs,
    gz_pred,
    filename="outputs/observed_vs_forward.png"
)

print("✔ Observed vs Forward plot saved")
# =====================================
# 11. Forward gravity contour
# =====================================

plot_forward_contour(
    x_obs,
    y_obs,
    gz_pred,
    filename="outputs/gravity_forward_contour.png"
)

print("✔ Forward gravity contour saved")