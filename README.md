# latentmi

A package for approximating mutual information in high dimensions, using the method introduced in our [paper](linksoon). Latent MI approximation relies on the assumption that high-dimensional data has low-dimensional structure.

## Installation

```bash
$ pip install latentmi
```

## Usage

```python
from latentmi import lmi

Xs = # some samples of a high dimensional variable
Ys = # some samples of a high dimensional variable

pmis, embedding, model = lmi.lmi(Xs, Ys)

MI_estimate = np.nanmean(pmis) # voila !
```

More detailed instructions can be found in the example usage notebook.


## License

`latentmi` was created by Gokul Gowri. It is licensed under the terms of the MIT license.

## Credits

`latentmi` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
