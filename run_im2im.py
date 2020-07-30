import os
import torch
import argparse
import numpy as np
import matplotlib
from data import get_data
from pca_linear_map import PCALinMapping
from matplotlib import pyplot as plt
import time
from imageio import imwrite
from utils import mosaic
import yaml
import pickle as pkl


def save_results(a_to_b, X_A_test, X_B_test, args, save_model=False):
    outfolder = os.path.join('results', 'run-{}'.format(time.strftime('%b-%d-%H:%M:%S', time.localtime(time.time()))))
    os.makedirs(outfolder, exist_ok=True)

    with open(os.path.join(outfolder, 'args.yaml'), 'w') as f:
        print(yaml.dump(vars(args), default_flow_style=False), file=f)

    print('Applying the learned transformation on test data...')
    X_A_test_to_A = a_to_b.reconstruct_a(X_A_test)
    X_A_test_to_B = a_to_b.transform_a_to_b(X_A_test)
    n_cols = 1 if X_A_test.shape[0] < 15 else None
    imwrite(os.path.join(outfolder, 'input_a.jpg'), mosaic(X_A_test, n_cols=n_cols))
    imwrite(os.path.join(outfolder, 'reconstructed_a.jpg'), mosaic(X_A_test_to_A, n_cols=n_cols))
    if X_B_test is not None:
        imwrite(os.path.join(outfolder, 'target_b.jpg'), mosaic(X_B_test, n_cols=n_cols))
        X_B_test_to_B = a_to_b.reconstruct_b(X_B_test)
        imwrite(os.path.join(outfolder, 'reconstructed_b.jpg'), mosaic(X_B_test_to_B, n_cols=n_cols))
    imwrite(os.path.join(outfolder, 'a_to_b.jpg'), mosaic(X_A_test_to_B, n_cols=n_cols))
    plt.matshow(1.0-a_to_b.Q[:50, :50], cmap='bwr', fignum=1)
    plt.savefig(os.path.join(outfolder, 'Q.png'))
    # plt.matshow(a_to_b.pca_a.components_.T @ a_to_b.Q @ a_to_b.pca_b.components_)
    # plt.savefig(os.path.join(outfolder, 'T.png'))

    if save_model:
        pkl.dump(a_to_b, open(os.path.join(outfolder, 'model.pkl'), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['celeba', 'ffhq', 'shoes'], default='celeba')
    parser.add_argument('--resolution', help='resized image height(=width) after cropping', type=int, default=64)
    parser.add_argument('--a_transform', choices=['identity', 'rot90', 'vflip', 'edges', 'Canny-edges', 'colorize', 'super-res', 'inpaint'], default='identity')
    parser.add_argument('--pairing', choices=['paired', 'matching', 'nonmatching', 'few-matches'], default='matching')
    parser.add_argument('--matching', choices=['nn', 'cyc-nn'], default='cyc-nn')
    parser.add_argument('--transform_type', choices=['orthogonal', 'linear'], default='orthogonal')
    parser.add_argument('--n_iters', type=int, default=50)
    parser.add_argument('--n_components', type=int, default=1000)
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--n_test', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    print('Loading {} data for {} - {} ...'.format(args.pairing, args.dataset, args.a_transform))
    X_A, X_B, X_A_test, X_B_test = get_data(args)

    print('Learning {} transformation in {} PCA dimensions...'.format(args.transform_type, args.n_components))

    a_to_b = PCALinMapping(args).fit(X_A, X_B)
    save_results(a_to_b, X_A_test, X_B_test, args)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
