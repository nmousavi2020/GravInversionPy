\# GravInversionPy



\## 3D Gravity Forward Modeling and Tikhonov Inversion in Python



\*\*Author:\*\* Naeim Mousavi



GravInversionPy is an open-source Python package for 3D gravity inversion and forward modeling. It implements Tikhonov regularization with a 3D smoothness constraint to recover subsurface density distributions from gravity observations.



\## Features



\* Gravity data loading

\* Regular grid subsampling

\* 3D voxel model generation

\* Gravity forward modeling

\* Transpose gravity operator

\* 3D smoothness regularization

\* Tikhonov inversion using Conjugate Gradient solver

\* Density model visualization

\* Forward gravity validation plots



\## Package Structure



```text

GravInversionPy

│

├── data

│   └── Gravity\_syn.txt

│

├── outputs

│   ├── density\_slice.png

│   ├── observed\_vs\_forward.png

│   └── gravity\_forward\_contour.png

│

├── src

│   └── grav\_inversion\_py

│       ├── \_\_init\_\_.py

│       ├── io.py

│       ├── model.py

│       ├── forward.py

│       ├── regularization.py

│       ├── inversion.py

│       └── plotting.py

│

├── examples

│   └── run\_inversion.py

│

├── tests

│

├── LICENSE

├── README.md

└── pyproject.toml

```



\## Installation



Clone the repository:



```bash

git clone https://github.com/nmousavi2020/GravInversionPy.git

cd GravInversionPy

```



Install the package:



```bash

pip install -e .

```



\## Requirements



\* Python >= 3.12

\* NumPy

\* SciPy

\* Matplotlib



\## Quick Start



Run the complete inversion workflow:



```bash

python examples/run\_inversion.py

```



The workflow includes:



```

Gravity observations

&#x20;         |

&#x20;         ↓

Regular grid subsampling

&#x20;         |

&#x20;         ↓

3D voxel model generation

&#x20;         |

&#x20;         ↓

Forward gravity calculation

&#x20;         |

&#x20;         ↓

Tikhonov regularized inversion

&#x20;         |

&#x20;         ↓

Recovered density model

&#x20;         |

&#x20;         ↓

Visualization and validation

```



\## Method



The package solves the inverse gravity problem:



\[

Gm = d

]



using Tikhonov regularization:



\[

(G^T G + \\lambda^2 L^T L)m = G^T d

]



where:



\* \*\*G\*\* is the gravity forward operator

\* \*\*m\*\* is the unknown density model

\* \*\*d\*\* is the observed gravity data

\* \*\*L\*\* is the 3D smoothness operator

\* \*\*λ\*\* is the regularization parameter



The solution is obtained using the Conjugate Gradient (CG) iterative solver.



\## Output Examples



\### Inverted Density Model



!\[Density Slice](outputs/density\_slice.png)



\### Observed vs Forward Gravity



!\[Observed vs Forward](outputs/observed\_vs\_forward.png)



\### Forward Gravity Map



!\[Forward Gravity](outputs/gravity\_forward\_contour.png)



\## Data Format



Input gravity file:



```text

X(km)   Y(km)   gz(mGal)



10.0    5.0     12.5

12.0    5.0     13.1

...

```



\## Author



\*\*Naeim Mousavi\*\*



\## Citation



If you use GravInversionPy in research, please cite:



Naeim Mousavi (2026).

GravInversionPy: Python tools for 3D gravity forward modeling and Tikhonov inversion.



\## License



MIT License



