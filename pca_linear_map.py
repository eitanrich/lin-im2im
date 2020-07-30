import numpy as np
from pca import aligned_pca
import faiss
import time


class PCALinMapping:
    """
    Learn a supervised or unsupervised linear mapping between the PCA embedding of two domains.
    See https://arxiv.org/pdf/2007.12568.pdf for details.
    Options in args:
      n_components: Number of PCA components (eigenvectors) to use - the dimension of the PCA representation.
      pairing: 'paired' = Skip ICP and just compute Q (i.e. supervised). Otherwise - perform ICP iterations.
      matching: ICP matching method: 'nn' = Regular nearest-neighbors. 'cyc-nn' = use cycle-consistent pairs only.
      transform_type: 'orthogonal' = constrain the linear transformation to be orthogonal. 'linear' = least squares.
      n_iters: Max number of ICP iterations
    """
    def __init__(self, args=None):
        self.args = args
        self.fitted = False
        self.pca_b = self.pca_a = self.Q = None

    def fit(self, x_a, x_b, res=None):
        """
        Perform PCA on the two domains and learn the linear transformation.
        :param x_a: Samples from A [m, d] (rows = samples)
        :param x_b: Samples from B [m, d] (rows = samples)
        :param res: Optional GPU resource for faiss. Can be used if called multiple times.
        """
        print('Got {} samples in A and {} in B.'.format(x_a.shape[0], x_b.shape[0]))
        t0 = time.time()
        self.pca_a, self.pca_b = aligned_pca(x_a, x_b, comps=self.args.n_components)
        z_a = self.pca_a.transform(x_a)
        z_b = self.pca_b.transform(x_b)
        print('PCA representations: ', z_a.shape, z_b.shape, 'took:', time.time()-t0)

        Q = np.eye(self.args.n_components, dtype=np.float32)

        if res is None:
            res = faiss.StandardGpuResources()
        nbrs_b = faiss.GpuIndexFlatL2(res, self.args.n_components)
        nbrs_b.add(z_b)

        print('Learning {} transformation using {} sets:'.format(self.args.transform_type, self.args.pairing))
        for it in range(self.args.n_iters):
            t0 = time.time()

            # Step 1 - Matching
            if self.args.pairing == 'paired':
                if it > 0:
                    break
                assert z_a.shape == z_b.shape
                A, B = z_a, z_b
            else:
                print('Iter {}: '.format(it), end='')
                # Find nearest-neighbors to z_A Q in B:
                d_qa_to_b, i_qa_to_b = nbrs_b.search(z_a @ Q, 1)
                i_qa_to_b = i_qa_to_b.squeeze()

                if self.args.matching == 'nn':
                    A = z_a
                    B = z_b[i_qa_to_b]
                    print('Found {} NNs. Mean NN l2 = {:.3f}. '.format(len(np.unique(i_qa_to_b)),
                                                                       np.mean(d_qa_to_b)), end='')
                else:
                    # Find nearest-neighbors in the reverse direction, for cycle-consistency:
                    sel_b = np.unique(i_qa_to_b)
                    assert len(sel_b) > 100, 'Only {} unique NNs'.format(len(sel_b))
                    nbrs_aQ = faiss.GpuIndexFlatL2(res, self.args.n_components)
                    nbrs_aQ.add(z_a @ Q)
                    _d_iqb_to_a, _i_iqb_to_a = nbrs_aQ.search(z_b[sel_b], 1)
                    i_iqb_to_a = -np.ones(shape=[z_b.shape[0]], dtype=int)
                    i_iqb_to_a[sel_b] = _i_iqb_to_a.squeeze()
                    # Check for cycle-consistency
                    cyc_consistent_a = i_iqb_to_a[i_qa_to_b] == np.arange(len(i_qa_to_b))
                    if np.count_nonzero(cyc_consistent_a) < 1000:
                        print('(only {} consisten pairs) '.format(np.count_nonzero(cyc_consistent_a)), end='')
                        cyc_consistent_a = np.ones_like(cyc_consistent_a)
                    A = z_a[cyc_consistent_a]
                    B = z_b[i_qa_to_b[cyc_consistent_a]]
                    print('{} B-NNs / {} consistent, mean NN l2 = {:.3f}. '.format(len(sel_b),
                        np.count_nonzero(cyc_consistent_a), np.mean(d_qa_to_b[cyc_consistent_a])), end='')

            # Step 2 - Mapping (updating Q):
            prev_Q = Q
            if self.args.transform_type == 'orthogonal':
                U, S, V = np.linalg.svd(A.T @ B)
                Q = U @ V
            else:
                Q = np.linalg.inv(A.T @ A) @ A.T @ B

            if np.allclose(Q, prev_Q):
                print('Converged - terminating ICP iterations.')
                break

            print('took {:.2f} sec.'.format(time.time()-t0))

        self.fitted = True
        self.Q = Q
        return self

    def transform_a_to_b(self, x_a, _Q=None, n_comps=None):
        """
        Apply the learned linear transformation
        :param x_a: The samples to be transformed
        :param _Q: Option to provide the transformation matrix
        :param n_comps: Use only the first n_comps PCA coefficients
        :return: T(x_a)
        """
        assert self.fitted or _Q is not None
        n_comps = n_comps or self.pca_a.components_.shape[0]
        Q = self.Q if _Q is None else _Q
        mu_a = self.pca_a.mean_.reshape(1, -1)
        mu_b = self.pca_b.mean_.reshape(1, -1)
        z_a = (x_a - mu_a) @ self.pca_a.components_[:n_comps].T
        z_ab = z_a @ Q[:n_comps, :n_comps]
        return z_ab @ self.pca_b.components_[:n_comps] + mu_b

    def reconstruct_a(self, x_a, n_comps=None):
        """
        Represent x_a in its PCA space and reconstruct it back (just to see the PCA reconstruction quality)
        :param x_a: Samples from A
        :param n_comps: Use only the first n_comps PCA coefficients
        :return: Reconstructed samples
        """
        n_comps = n_comps or self.pca_a.components_.shape[0]
        mu_a = self.pca_a.mean_.reshape(1, -1)
        return (x_a - mu_a) @ self.pca_a.components_[:n_comps].T @ self.pca_a.components_[:n_comps] + mu_a

    def reconstruct_b(self, x_b, n_comps=None):
        """
        Same as reconstruct_a, but for domain B
        """
        n_comps = n_comps or self.pca_b.components_.shape[0]
        mu_b = self.pca_b.mean_.reshape(1, -1)
        return (x_b - mu_b) @ self.pca_b.components_[:n_comps].T @ self.pca_b.components_[:n_comps] + mu_b
