from grav_inversion_py.io import load_gravity_data
from grav_inversion_py.plotting import plot_observed_gravity


x, y, gz = load_gravity_data(
    "data/Gravity_syn.txt"
)


plot_observed_gravity(
    x,
    y,
    gz
)


print("✔ Observed gravity plot saved")