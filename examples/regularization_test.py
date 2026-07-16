from grav_inversion_py.regularization import smoothness_matrix


L = smoothness_matrix(
    25,
    22,
    7
)


print("✔ Smoothness matrix created")
print("Shape:", L.shape)
print("Non-zero elements:", L.nnz)