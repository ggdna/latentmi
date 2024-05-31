# Troubleshooting

Here we'll list some common issues and how they can be fixed. Hope this helps!

### my MI estimate is `nan`

The most likely reason is that you are using `mean` of pointwise estimates rather than `nanmean`. If you get a warning that there are `NaN`s in the embedding, then there is some deeper problem -- first check for NaNs in the input data, then consider playing with some parameters like `N_dims` or `lr`.

### my MI estimate is high because of a few extremely high pMI contributions

If you have duplicated (x, y) pairs in the data you can get an extremely high pMI estimate as an artifact. This is a known artifact of the KSG estimator, which is why you can't bootstrap KSG estimates (as discussed by [Holmes and Nemenman](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.022404)).