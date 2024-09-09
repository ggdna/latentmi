# latentmi

Latent MI (LMI) approximation is a method for estimating mutual information high dimensions. It is built on the idea that real world high-dimensional data has underlying low-dimensional structure. For more details, see our [manuscript](https://arxiv.org/abs/2409.02732). `latentmi` is our Python implementation of LMI approximation, built with the hope that practicioners can use it with minimal pain and suffering :) 


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

More detailed instructions can be found in the example usage notebook.


## License

`latentmi` was created by Gokul Gowri. It is licensed under the terms of the MIT license.

## Credits

`latentmi` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
