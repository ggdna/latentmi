import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from tqdm.notebook import tqdm
import warnings

from . import models
from . import ksg

class EarlyStopper:
    """
    Early stopping that returns best weights
    trying to replicate the Keras callback
    """
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return self.best_state
        return False


def train(model, X_train, Y_train, X_test, Y_test,
          batch_size=512, lr=0.0001, epochs=300, patience=30, 
          quiet=True):
    """
    training loop for LMI models

    :param model: LMI model

    :param X_train: train samples, shape (N_samples, N_features)
    :param Y_train: train samples, shape (N_samples, N_features)
    :param X_test: test samples, shape (N_samples, N_features)
    :param Y_test: test samples, shape (N_samples, N_features)

    :param batch_size: samples per batch, defaults to 512
    :param lr: learning rate for Adam optimizer, defaults to 1e-4
    :param epochs: max number of epochs, defaults to 300
    :param patience: epochs without val. loss decline before early stopping, 
                     defaults to 300
    :param quiet: suppress training progress display, defaults to True
    """
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)
    
    train_dataloader = DataLoader(list(zip(X_train, Y_train)), 
                                batch_size=batch_size, 
                                shuffle=True)

    val_dataloader = DataLoader(list(zip(X_test, Y_test)), 
                                batch_size=batch_size, 
                                shuffle=True)
    
    val_losses = []

    early_stopper = EarlyStopper(patience=patience)
    
    # with tqdm(range(epochs), unit='Epoch', disable=quiet) as tepoch:
    for epoch in range(epochs):
        
        for i, (X, Y) in enumerate(train_dataloader):
            
            model.train() 
            model_loss = model.learning_loss(X, Y)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
        
        # validation 
        with torch.no_grad():
            epoch_validate_loss = []
            for i, data in enumerate(val_dataloader):
                X, Y = data
                epoch_validate_loss.append(model.learning_loss(X, Y).item())
            val_losses.append(np.mean(epoch_validate_loss))

        # tepoch.set_postfix(val_loss=val_losses[-1])
        # early stopping
        es = early_stopper.early_stop(val_losses[-1], model)
        if es or epoch==epochs:
            if not quiet:
                print('\repoch %d (of max %d) %s \U0001F389\U0001F389' 
                         %(epoch, epochs, '\U0001F33B'*int(10*(epoch/epochs))),
                         end='')
                sys.stdout.flush()
                sys.stdout.write('\n')
                print("success! training stopped at epoch %d" % epoch)
                print('final validation loss:', val_losses[-1])
            model.load_state_dict(es)
            return
        

def learn_representation(Xs, Ys, train_indices, test_indices,
       regularizer='models.AECross', 
       alpha=1, lam=1,
       N_dims=8, batch_size=512, lr=0.0001, epochs=300,
       validation_split=0.3, patience=30, quiet=True, device='cpu'):
    """
    train paired AE model and embed data

    :param Xs: input data, array with shape (N_samples, N_dims). ordering must align with Y.
    :param Ys: input data, array with shape(N_samples, N_dims). ordering must align with X.

    :param train_indices: list, indices of train samples
    :param test_indices:list, indices of test samples

    :param regularizer: type of regularization, defaults to AECross. 
                        can be changed to \'models.AEMINE\' or 
                        \'models.AEInfoNCE\' but not recommended.
    :param alpha: self-reconstruction loss weight, defaults to 1
    :param lam: cross-reconstruction regularization weight, defaults to 1
    :param N_dims: dimensions in each latent representation, defaults to 8
    
    :param batch_size: samples per batch, defaults to 512
    :param lr: learning rate for Adam optimizer, defaults to 1e-4
    :param epochs: max number of epochs, defaults to 300
    :param validation_split: fraction train/test split, defaults to 0.5
    :param patience: epochs without val. loss decline before early stopping, 
                     defaults to 300
    :param quiet: suppress training progress display, defaults to True
    """
    X_train = Xs[train_indices]
    Y_train = Ys[train_indices]
    
    X_test = Xs[test_indices]
    Y_test = Ys[test_indices]
    
    if 128 < N_dims or 128 < N_dims:
        warnings.warn("Hidden layer smaller than latent dimension. Consider reducing N_dims")
        
    # assert X_train.shape[1] // 4 > 0, "Hidden layer with size 0. Consider tiling input."
    # assert Y_train.shape[1] // 4 > 0, "Hidden layer with size 0. Consider tiling input."
    
    model = eval(regularizer)(X_train.shape[1], Y_train.shape[1], N_dims, 
                              alpha=alpha, lam=lam).to(device)
    
    train(model, X_train, Y_train, X_test, Y_test, 
          batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
          quiet=quiet)
    
    with torch.no_grad():
        model.eval()
        Zx, Zy = model.encode(Xs, Ys)
        
        Zx, Zy = Zx.cpu(), Zy.cpu()


        return Zx.cpu(), Zy.cpu(), model

def estimate(Xs, Ys, estimate_var=False, regularizer='models.AECross', 
         alpha=1, lam=1,
         N_dims=8, k=4, validation_split=0.5, estimate_on_val=True,
         batch_size=512, lr=0.0001, epochs=300, patience=30,
         quiet=True, device=None):
    """
    return pMIs (with NaNs for points not included in KSG estimate), [variance estimate], [variance estimate error], embeddings, trained model

    :param Xs: input data, array with shape (N_samples, N_dims). ordering must align with Y.
    :param Ys: input data, array with shape(N_samples, N_dims). ordering must align with X.

    :param estimate_var: indicates whether to return a complementary variance estimate and variance estimate error, defaults to False

    :param regularizer: type of regularization, defaults to AECross. 
                        can be changed to \'models.AEMINE\' or 
                        \'models.AEInfoNCE\' but not recommended.
    :param alpha: self-reconstruction loss weight, defaults to 1
    :param lam: cross-reconstruction regularization weight, defaults to 1
    :param N_dims: dimensions in each latent representation, defaults to 8
    :param k: k value used in the kNN calculation for KSG estimate, defaults to 4
    
    :param batch_size: samples per batch, defaults to 512
    :param lr: learning rate for Adam optimizer, defaults to 1e-4
    :param epochs: max number of epochs, defaults to 300
    :param validation_split: fraction train/test split, defaults to 0.5
    :param patience: epochs without val. loss decline before early stopping, 
                     defaults to 300
    :param quiet: suppress training progress display, defaults to True
    :param device: device for torch to train model, defaults to cuda if available, else cpu.

    :return: array of pointwise mutual information estimates, order aligned with input. NaNs
             values not included in KSG estimate. mean of this array is MI estimate.
    :return: a single variance estimate for the LMI approximation
    :return: the standard error of the variance estimate
    :return: tuple of arrays of coordinates of latent embeddings. first index is X embeddings, second index is Y.
    :return: Pytorch object for trained representation learning model
    """

    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    oXs = Xs
    oYs = Ys

    Xs = torch.from_numpy(np.nan_to_num((Xs - Xs.mean(axis=0)) / Xs.std(axis=0))).float().to(device)
    Ys = torch.from_numpy(np.nan_to_num((Ys - Ys.mean(axis=0)) / Ys.std(axis=0))).float().to(device)

    Xs = torch.clip(Xs, min=-10, max=10)
    Ys = torch.clip(Ys, min=-10, max=10)

    assert len(Xs) == len(Ys), "X and Y must be same length!"
    
    N_train = int(len(Xs) * (1 - validation_split))
    
    indices = np.arange(len(Xs))
    np.random.shuffle(indices)
    
    train_indices = indices[:N_train]
    test_indices = indices[N_train:]

    
    Zx, Zy, model = learn_representation(Xs, Ys, train_indices, test_indices,
                regularizer=regularizer, N_dims=N_dims, batch_size=batch_size,
                patience=patience, epochs=epochs, 
                lr=lr, quiet=quiet,
                alpha=alpha, lam=lam, device=device)

    if torch.isnan(Zx).any() or torch.isnan(Zy).any():
        warnings.warn("NaNs in embedding! converted to 0s")

    Zx = torch.nan_to_num(Zx)
    Zy = torch.nan_to_num(Zy)
    
    estimate = 0

    if estimate_on_val:

        # make nan array
        estimate = np.zeros(len(Xs))
        estimate += np.NaN

        # fill val pMIs
        estimate[indices[N_train:]] = ksg.mi(Zx.cpu()[indices[N_train:]], 
        Zy.cpu()[indices[N_train:]], k)
    
    else:
        estimate += ksg.mi(Zx.cpu(), Zy.cpu(), k)
    
    if not estimate_var:
        return estimate, (Zx.cpu(), Zy.cpu()), model
    else:
        var_estimate, se_var_estimate = estimate_variance(oXs, oYs, n_partitions=9, regularizer=regularizer, 
                      alpha=alpha, lam=lam, N_dims=N_dims, k=k, validation_split=validation_split, estimate_on_val=estimate_on_val,
                      batch_size=batch_size, lr=lr, epochs=epochs, patience=patience, quiet=quiet, device=device)
        return estimate, var_estimate, se_var_estimate, (Zx.cpu(), Zy.cpu()), model

def estimate_variance(Xs, Ys, n_partitions=9, regularizer='models.AECross', 
                      alpha=1, lam=1, N_dims=8, k=3, validation_split=0.5, estimate_on_val=True,
                      batch_size=512, lr=0.0001, epochs=300, patience=30, quiet=True, device=None):

    # :param n_partitions: number of different partitions of the data to estimate variance on. defaults to 9.  

    assert len(Xs) == len(Ys), "Xs and Ys must be the same size!"
   
    XsYs = list(zip(Xs, Ys)) # combine Xs and Ys into a list of tuples for easier shuffling and partitioning
    # [([x, x, x], [y, y, y]), ([x, x, x], [y, y, y]), ...]
    data_size = len(XsYs)
    part_sizes = np.array([i for i in range(2, n_partitions + 2)]) # the number of sections in each partition (the first partition has 2 sections, etc.)

    partitions = [] # contains n_partitions different partitions of n_i sections where i goes from 2 to n_partitions + 1
    for i in range(0, n_partitions):
        sec_size = data_size // part_sizes[i] # number of samples in each section
        np.random.shuffle(XsYs) # shuffle the data before creating the sections
        partitions.append([XsYs[j*sec_size:(j+1)*sec_size] for j in range(part_sizes[i])])

    lmis = [] # contains the LMI estimates for each partition
    for part in partitions:
        part_lmi_est = []
        for sec in part:
            Xs_sec, Ys_sec = zip(*sec) # unzip the section into Xs and Ys
            Xs_sec = np.array(Xs_sec)
            Ys_sec = np.array(Ys_sec)

            pmis_part , _, _ = estimate(Xs_sec, Ys_sec, estimate_var=False, regularizer=regularizer, alpha=alpha, lam=lam, N_dims=N_dims, k=k, validation_split=validation_split, estimate_on_val=estimate_on_val, batch_size=batch_size, lr=lr, epochs=epochs, patience=patience, quiet=quiet, device=device);

            lmi_estimate_part = np.nanmean(pmis_part)
            part_lmi_est.append(lmi_estimate_part)
        lmis.append(part_lmi_est)

    # calculating the variance of the LMI estiamtes from the subsamples
    part_variances = np.array([None] * n_partitions) # contains the variance estimates for each partition (the first partition is 0)
    for i in range(0, n_partitions):
        part_variances[i] = np.var(lmis[i], ddof = 1)

    var_estimate = sum((part_sizes - 1) / part_sizes * part_variances) / sum(part_sizes - 1)
    sml = var_estimate * data_size
    var_s = 2 * sml**2 / sum(part_sizes - 1)
    se_var_estimate = np.sqrt(var_s / data_size**2)
    
    return var_estimate, se_var_estimate
