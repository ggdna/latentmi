# latentmi

[![Documentation Status](https://readthedocs.org/projects/latentmi/badge/?version=latest)](https://latentmi.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/latentmi.svg)](https://badge.fury.io/py/latentmi) [![arXiv](https://img.shields.io/badge/arXiv-2409.02732-b31b1b.svg)](https://arxiv.org/abs/2409.02732)

Latent MI (LMI) approximation is a method for estimating mutual information in high dimensions, using the idea that real world high-dimensional data has underlying low-dimensional structure. For more details, see our [manuscript (NeurIPS 2024 spotlight 🌟)](https://arxiv.org/abs/2409.02732). `latentmi` is our Python implementation of LMI approximation, built with the hope that practicioners find it pleasant to use :) 


## Installation

```bash
$ pip install latentmi
```

## Usage

```python
from latentmi import lmi

Xs = # some samples of a high dimensional variable
Ys = # some samples of a high dimensional variable

pmis, embedding, model = lmi.estimate(Xs, Ys)

MI_estimate = np.nanmean(pmis) # voila !
```

More detailed instructions can be found in the [demo notebook](https://latentmi.readthedocs.io/en/latest/example.html).


## License

`latentmi` was created by Gokul Gowri. It is licensed under the terms of the MIT license.

## Credits

`latentmi` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

The implementation of the `ksg` estimator is adapted from Greg Ver Steeg's [Nonparametric Entropy Estimation Toolbox](https://github.com/gregversteeg/NPEET/tree/master).
