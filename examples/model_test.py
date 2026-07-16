from grav_inversion_py.model import create_model_grid


Xc, Yc, Zc, volume, n_model = create_model_grid()


print("✔ Model grid created")
print("Grid size:", Xc.shape)
print("Cell volume:", volume)
print("Number of cells:", n_model)