import numpy as np


def load_gravity_data(filename):
    """
    Load gravity observation data.

    Format:
    X(km) Y(km) gz(mGal)
    """

    data = np.loadtxt(filename)

    x = data[:, 0]
    y = data[:, 1]
    gz = data[:, 2]

    return x, y, gz



def save_gravity_data(filename, x, y, gz):
    """
    Save gravity data.
    """

    data = np.column_stack(
        [
            x,
            y,
            gz
        ]
    )

    np.savetxt(
        filename,
        data,
        header="X(km) Y(km) gz(mGal)",
        fmt="%.6f"
    )