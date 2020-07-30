import numpy as np
import torch
from scipy.stats import skew


class PCA:
    """
    A sklearn.decomposition.PCA - like wrapper on top of pytorch pca_lowrank
    """
    def __init__(self, n_components, svd_solver='auto'):
        assert svd_solver == 'auto'
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, x):
        assert len(x.shape) == 2
        assert x.dtype == 'float32'
        n, d = x.shape

        # TODO: Check if fits in GPU, Implement incremental PCA
        _x = torch.from_numpy(x)
        _mu = _x.mean(dim=0, keepdim=True)
        Z, S, V = torch.pca_lowrank((_x-_mu).cuda(), q=self.n_components, niter=3, center=False)
        # Z, S, V = torch.pca_lowrank((_x-_mu), q=self.n_components, niter=3, center=False)
        self.mean_ = _mu.numpy()
        self.components_ = V.cpu().numpy().T
        self.explained_variance_ = np.square(S.cpu().numpy()) / (n-1)

        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        return (x-self.mean_) @ self.components_.T

    def inverse_transform(self, z):
        return z @ self.components_ + self.mean_


def aligned_pca(X_a, X_b, comps=None):
    """
    Perform PCA on A and on B and then try to resolve polarity ambiguity using two heuristics:
    - Check the skewness of the projection of the samples in A and in B along each PCA component
    - If the skewness is not significant (close to 0), simply point the eigenvectors in the general positive
      direction
    """
    def do_pca(X):
        pca = PCA(comps)
        z = pca.fit_transform(X)
        sk = skew(z, axis=0)
        mu = np.mean(pca.components_, axis=1)
        return pca, sk, mu

    print('PCA A...')
    pca_a, sk_a, mu_a = do_pca(X_a)
    print('PCA B...')
    pca_b, sk_b, mu_b = do_pca(X_b)
    print('Synchronizing...')
    sk_significant = np.minimum(np.abs(sk_a), np.abs(sk_b)) > 6e-3
    print('Using skew-based logic for {}/{} dimensions.'.format(np.count_nonzero(sk_significant), len(sk_significant)))
    pca_a.components_[sk_significant] *= np.sign(sk_a[sk_significant]).reshape(-1, 1)
    pca_b.components_[sk_significant] *= np.sign(sk_b[sk_significant]).reshape(-1, 1)
    pca_a.components_[np.logical_not(sk_significant)] *= np.sign(mu_a[np.logical_not(sk_significant)]).reshape(-1, 1)
    pca_b.components_[np.logical_not(sk_significant)] *= np.sign(mu_b[np.logical_not(sk_significant)]).reshape(-1, 1)
    return pca_a, pca_b
