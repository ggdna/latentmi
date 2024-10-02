# Troubleshooting

Here we'll list some common issues and how they can be fixed. Hope this helps!

### how should I choose `N_dims` ?

In most cases, LMI estimates are lower than true MI values. This aligns with the result that $I(X; Y) \geq I(Z_x; Z_y)$ (from the data processing inequality). So, a reasonable way to choose `N_dims` is to try several possibilities and choose that which yields that the highest LMI estimate. Choosing `N_dims` higher than 20 is unlikely to work well, as KSG tends to degrade in these dimensionalities.

### my MI estimate is `nan`

The most likely reason is that you are using `mean` of pointwise estimates rather than `nanmean`. If you get a warning that there are `NaN`s in the embedding, then there is some deeper problem -- first check for NaNs in the input data, then consider playing with some parameters like `N_dims` or `lr`.

### there are a few extremely high pMI contributions

If you have duplicated (x, y) pairs in the data you can get an extremely high pMI estimate as an artifact. This is a known artifact of the KSG estimator, which is why you can't bootstrap KSG estimates (as discussed by [Holmes and Nemenman](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.022404)).

### I expect my variables to be dependent but my LMI estimates are 0

LMI estimates are not theoretically guaranteed to identify independence. It is possible to get an LMI estimate of 0 despite having dependent variables. If you believe this is happening, you can try to modify the parameters of your LMI estimate to see if the results change. In particular, it is worth changing `N_dims` to be higher, and changing the regularizer to `regularizer="models.AEMINE"`.