import numpy as np

from grav_inversion_py.io import load_gravity_data
from grav_inversion_py.model import create_model_grid
from grav_inversion_py.forward import forward_gravity


x, y, gz = load_gravity_data(
    "data/Gravity_syn.txt"
)


Xc, Yc, Zc, volume, n_model = create_model_grid()


density = np.ones(
    Xc.shape
) * 1000


gz_calc = forward_gravity(
    density,
    Xc,
    Yc,
    Zc,
    x,
    y,
    volume
)


print("✔ Forward gravity calculated")
print("Points:", len(gz_calc))
print("Minimum:", gz_calc.min())
print("Maximum:", gz_calc.max())