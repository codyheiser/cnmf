# -*- coding: utf-8 -*-
"""
consensus non-negative matrix factorization (cNMF) adapted from (Kotliar, et al. 2019)

@author: C Heiser
2020
"""
import numpy as np
import pandas as pd
import os, errno
import glob
import shutil
import datetime
import uuid
import itertools
import yaml
import subprocess
import scipy.sparse as sp
import warnings

from scipy.spatial.distance import squareform
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import sparsefuncs
from sklearn.preprocessing import normalize

from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list

import matplotlib.pyplot as plt
import scanpy as sc
from ._version import get_versions


def save_df_to_npz(obj, filename):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        np.savez_compressed(
            filename,
            data=obj.values,
            index=obj.index.values,
            columns=obj.columns.values,
        )


def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep="\t")


def load_df_from_npz(filename):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with np.load(filename, allow_pickle=True) as f:
            obj = pd.DataFrame(**f)
    return obj


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def worker_filter(iterable, worker_index, total_workers):
    return (
        p for i, p in enumerate(iterable) if (i - worker_index) % total_workers == 0
    )


def fast_euclidean(mat):
    D = mat.dot(mat.T)
    squared_norms = np.diag(D).copy()
    D *= -2.0
    D += squared_norms.reshape((-1, 1))
    D += squared_norms.reshape((1, -1))
    D = np.sqrt(D)
    D[D < 0] = 0
    return squareform(D, checks=False)


def fast_ols_all_cols(X, Y):
    pinv = np.linalg.pinv(X)
    beta = np.dot(pinv, Y)
    return beta


def fast_ols_all_cols_df(X, Y):
    beta = fast_ols_all_cols(X, Y)
    beta = pd.DataFrame(beta, index=X.columns, columns=Y.columns)
    return beta


def var_sparse_matrix(X):
    mean = np.array(X.mean(axis=0)).reshape(-1)
    Xcopy = X.copy()
    Xcopy.data **= 2
    var = np.array(Xcopy.mean(axis=0)).reshape(-1) - (mean ** 2)
    return var


def get_highvar_genes_sparse(
    expression, expected_fano_threshold=None, minimal_mean=0.01, numgenes=None
):
    # Find high variance genes within those cells
    gene_mean = np.array(expression.mean(axis=0)).astype(float).reshape(-1)
    E2 = expression.copy()
    E2.data **= 2
    gene2_mean = np.array(E2.mean(axis=0)).reshape(-1)
    gene_var = pd.Series(gene2_mean - (gene_mean ** 2))
    del E2
    gene_mean = pd.Series(gene_mean)
    gene_fano = gene_var / gene_mean

    # Find parameters for expected fano line
    top_genes = gene_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_var) / gene_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_fano.quantile([0.10, 0.90])
    winsor_box = (
        (gene_fano > w_fano_low)
        & (gene_fano < w_fano_high)
        & (gene_mean > w_mean_low)
        & (gene_mean < w_mean_high)
    )
    fano_median = gene_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A ** 2) * gene_mean + (B ** 2)
    fano_ratio = gene_fano / gene_expected_fano

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T = None

    else:
        if not expected_fano_threshold:
            T = 1.0 + gene_counts_fano[winsor_box].std()
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame(
        {
            "mean": gene_mean,
            "var": gene_var,
            "fano": gene_fano,
            "expected_fano": gene_expected_fano,
            "high_var": high_var_genes_ind,
            "fano_ratio": fano_ratio,
        }
    )
    gene_fano_parameters = {
        "A": A,
        "B": B,
        "T": T,
        "minimal_mean": minimal_mean,
    }
    return (gene_counts_stats, gene_fano_parameters)


def get_highvar_genes(
    input_counts, expected_fano_threshold=None, minimal_mean=0.01, numgenes=None
):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var / gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_counts_var) / gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = (
        (gene_counts_fano > w_fano_low)
        & (gene_counts_fano < w_fano_high)
        & (gene_counts_mean > w_mean_low)
        & (gene_counts_mean < w_mean_high)
    )
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A ** 2) * gene_counts_mean + (B ** 2)

    fano_ratio = gene_counts_fano / gene_expected_fano

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T = None

    else:
        if not expected_fano_threshold:
            T = 1.0 + gene_counts_fano[winsor_box].std()
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame(
        {
            "mean": gene_counts_mean,
            "var": gene_counts_var,
            "fano": gene_counts_fano,
            "expected_fano": gene_expected_fano,
            "high_var": high_var_genes_ind,
            "fano_ratio": fano_ratio,
        }
    )
    gene_fano_parameters = {
        "A": A,
        "B": B,
        "T": T,
        "minimal_mean": minimal_mean,
    }
    return (gene_counts_stats, gene_fano_parameters)


def compute_tpm(input_counts):
    """
    Default TPM normalization
    """
    tpm = input_counts.copy()
    tpm.layers["raw_counts"] = tpm.X.copy()
    sc.pp.normalize_total(tpm, target_sum=1e6)
    return tpm


def subset_adata(adata, subset):
    print("Subsetting AnnData on {}".format(subset), end="")
    # initialize .obs column for choosing cells
    adata.obs["adata_subset_combined"] = 0
    # create label as union of given subset args
    for i in range(len(subset)):
        adata.obs.loc[adata.obs[subset[i]] == 1, "adata_subset_combined"] = 1
    adata = adata[adata.obs["adata_subset_combined"] == 1, :].copy()
    adata.obs.drop(columns="adata_subset_combined", inplace=True)
    print(" - now {} cells and {} genes".format(adata.n_obs, adata.n_vars))
    return adata


def cnmf_markers(adata, spectra_score_file, n_genes=30, key="cnmf"):
    """
    read in gene spectra score output from cNMF and save top gene loadings 
    for each usage as dataframe in adata.uns

    Parameters:
        adata (AnnData.AnnData): AnnData object
        spectra_score_file (str): '<name>.gene_spectra_score.<k>.<dt>.txt' file from cNMF containing gene loadings
        n_genes (int): number of top genes to list for each usage (rows of df)
        key (str): prefix of adata.uns keys to save

    Returns:
        AnnData.AnnData: adata is edited in place to include gene spectra scores
        (adata.varm["cnmf_spectra"]) and list of top genes by spectra score (adata.uns["cnmf_markers"])
    """
    # load Z-scored GEPs which reflect gene enrichment, save to adata.varm
    spectra = pd.read_csv(spectra_score_file, sep="\t", index_col=0).T
    adata.varm["{}_spectra".format(key)] = spectra.values
    # obtain top n_genes for each GEP in sorted order and combine them into df
    top_genes = []
    for gep in spectra.columns:
        top_genes.append(
            list(spectra.sort_values(by=gep, ascending=False).index[:n_genes])
        )
    # save output to adata.uns
    adata.uns["{}_markers".format(key)] = pd.DataFrame(
        top_genes, index=spectra.columns.astype(str)
    ).T


def cnmf_load_results(adata, cnmf_dir, name, k, dt, key="cnmf", **kwargs):
    """
    Load results of cNMF.
    Given adata object and corresponding cNMF output (cnmf_dir, name, k, dt to identify),
    read in relevant results and save to adata object inplace, and output plot of gene
    loadings for each GEP usage.

    Parameters:
        adata (AnnData.AnnData): AnnData object
        cnmf_dir (str): relative path to directory containing cNMF outputs
        name (str): name of cNMF replicate
        k (int): value used for consensus factorization
        dt (int): distance threshold value used for consensus clustering
        key (str): prefix of adata.uns keys to save
        n_points (int): how many top genes to include in rank_genes() plot
        **kwargs: keyword args to pass to cnmf_markers()

    Returns:
        AnnData.AnnData: adata is edited in place to include overdispersed genes
            (adata.var["cnmf_overdispersed"]), usages (adata.obs["usage_#"],
            adata.obsm["cnmf_usages"]), gene spectra scores (adata.varm["cnmf_spectra"]),
            and list of top genes by spectra score (adata.uns["cnmf_markers"]).
    """
    # read in cell usages
    usage = pd.read_csv(
        "{}/{}/{}.usages.k_{}.dt_{}.consensus.txt".format(
            cnmf_dir, name, name, str(k), str(dt).replace(".", "_")
        ),
        sep="\t",
        index_col=0,
    )
    usage.columns = ["usage_" + str(col) for col in usage.columns]
    # normalize usages to total for each cell
    usage_norm = usage.div(usage.sum(axis=1), axis=0)
    usage_norm.index = usage_norm.index.astype(str)
    # add usages to .obs for visualization
    adata.obs = pd.merge(
        left=adata.obs, right=usage_norm, how="left", left_index=True, right_index=True
    )
    # replace missing values with zeros for all factors
    adata.obs.loc[:, usage_norm.columns].fillna(value=0, inplace=True)
    # add usages as array in .obsm for dimension reduction
    adata.obsm["cnmf_usages"] = adata.obs.loc[:, usage_norm.columns].values

    # read in overdispersed genes determined by cNMF and add as metadata to adata.var
    overdispersed = np.genfromtxt(
        "{}/{}/{}.overdispersed_genes.txt".format(cnmf_dir, name, name), dtype=str
    )
    adata.var["cnmf_overdispersed"] = 0
    adata.var.loc[
        [x for x in adata.var.index if x in overdispersed], "cnmf_overdispersed"
    ] = 1

    # read top gene loadings for each GEP usage and save to adata.uns['cnmf_markers']
    cnmf_markers(
        adata,
        "{}/{}/{}.gene_spectra_score.k_{}.dt_{}.txt".format(
            cnmf_dir, name, name, str(k), str(dt).replace(".", "_")
        ),
        key=key,
        **kwargs
    )


class cNMF:
    def __init__(self, output_dir=".", name=None):
        """
        Parameters
        ----------

        output_dir : path, optional (default=".")
            Output directory for analysis files.

        name : string, optional (default=None)
            A name for this analysis. Will be prefixed to all output files.
            If set to None, will be automatically generated from date (and random string).
        """

        self.output_dir = output_dir
        if name is None:
            now = datetime.datetime.now()
            rand_hash = uuid.uuid4().hex[:6]
            name = "%s_%s" % (now.strftime("%Y_%m_%d"), rand_hash)
        self.name = name
        self.paths = None

    def _initialize_dirs(self):
        if self.paths is None:
            # Check that output directory exists, create it if needed.
            check_dir_exists(self.output_dir)
            check_dir_exists(os.path.join(self.output_dir, self.name))
            check_dir_exists(os.path.join(self.output_dir, self.name, "cnmf_tmp"))

            self.paths = {
                "normalized_counts": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".norm_counts.h5ad",
                ),
                "nmf_replicate_parameters": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".nmf_params.df.npz",
                ),
                "nmf_run_parameters": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".nmf_idvrun_params.yaml",
                ),
                "nmf_genes_list": os.path.join(
                    self.output_dir, self.name, self.name + ".overdispersed_genes.txt"
                ),
                "tpm": os.path.join(
                    self.output_dir, self.name, "cnmf_tmp", self.name + ".tpm.h5ad"
                ),
                "tpm_stats": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".tpm_stats.df.npz",
                ),
                "iter_spectra": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".spectra.k_%d.iter_%d.df.npz",
                ),
                "iter_usages": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".usages.k_%d.iter_%d.df.npz",
                ),
                "merged_spectra": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".spectra.k_%d.merged.df.npz",
                ),
                "local_density_cache": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".local_density_cache.k_%d.merged.df.npz",
                ),
                "consensus_spectra": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".spectra.k_%d.dt_%s.consensus.df.npz",
                ),
                "consensus_spectra__txt": os.path.join(
                    self.output_dir,
                    self.name,
                    self.name + ".spectra.k_%d.dt_%s.consensus.txt",
                ),
                "consensus_usages": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".usages.k_%d.dt_%s.consensus.df.npz",
                ),
                "consensus_usages__txt": os.path.join(
                    self.output_dir,
                    self.name,
                    self.name + ".usages.k_%d.dt_%s.consensus.txt",
                ),
                "consensus_stats": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".stats.k_%d.dt_%s.df.npz",
                ),
                "clustering_plot": os.path.join(
                    self.output_dir, self.name, self.name + ".clustering.k_%d.dt_%s.png"
                ),
                "gene_spectra_score": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".gene_spectra_score.k_%d.dt_%s.df.npz",
                ),
                "gene_spectra_score__txt": os.path.join(
                    self.output_dir,
                    self.name,
                    self.name + ".gene_spectra_score.k_%d.dt_%s.txt",
                ),
                "gene_spectra_tpm": os.path.join(
                    self.output_dir,
                    self.name,
                    "cnmf_tmp",
                    self.name + ".gene_spectra_tpm.k_%d.dt_%s.df.npz",
                ),
                "gene_spectra_tpm__txt": os.path.join(
                    self.output_dir,
                    self.name,
                    self.name + ".gene_spectra_tpm.k_%d.dt_%s.txt",
                ),
                "k_selection_plot": os.path.join(
                    self.output_dir, self.name, self.name + ".k_selection.png"
                ),
                "k_selection_stats": os.path.join(
                    self.output_dir, self.name, self.name + ".k_selection_stats.df.npz"
                ),
            }

    def get_norm_counts(
        self, counts, tpm, high_variance_genes_filter=None, num_highvar_genes=None
    ):
        """
        Parameters
        ----------

        counts : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing raw counts. Filtered such that
            no genes or cells with 0 counts
        
        tpm : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing tpm normalized data matching
            counts

        high_variance_genes_filter : np.array, optional (default=None)
            A pre-specified list of genes considered to be high-variance.
            Only these genes will be used during factorization of the counts matrix.
            Must match the .var index of counts and tpm.
            If set to None, high-variance genes will be automatically computed, using the
            parameters below.

        num_highvar_genes : int, optional (default=None)
            Instead of providing an array of high-variance genes, identify this many most overdispersed genes
            for filtering

        Returns
        -------

        normcounts : anndata.AnnData, shape (cells, num_highvar_genes)
            A counts matrix containing only the high variance genes and with columns (genes)normalized to unit
            variance

        """

        if high_variance_genes_filter is None:
            ## Get list of high-var genes if one wasn't provided
            if sp.issparse(tpm.X):
                (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(
                    tpm.X, numgenes=num_highvar_genes
                )
            else:
                (gene_counts_stats, gene_fano_params) = get_highvar_genes(
                    np.array(tpm.X), numgenes=num_highvar_genes
                )

            high_variance_genes_filter = list(
                tpm.var.index[gene_counts_stats.high_var.values]
            )

        ## Subset out high-variance genes
        print(
            "Selecting {} highly variable genes".format(len(high_variance_genes_filter))
        )
        norm_counts = counts[:, high_variance_genes_filter]
        norm_counts = norm_counts[tpm.obs_names, :].copy()

        ## Scale genes to unit variance
        if sp.issparse(tpm.X):
            sc.pp.scale(norm_counts, zero_center=False)
            if np.isnan(norm_counts.X.data).sum() > 0:
                print("Warning: NaNs in normalized counts matrix")
        else:
            norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
            if np.isnan(norm_counts.X).sum().sum() > 0:
                print("Warning: NaNs in normalized counts matrix")

        ## Save a \n-delimited list of the high-variance genes used for factorization
        open(self.paths["nmf_genes_list"], "w").write(
            "\n".join(high_variance_genes_filter)
        )

        ## Check for any cells that have 0 counts of the overdispersed genes
        zerocells = norm_counts.X.sum(axis=1) == 0
        if zerocells.sum() > 0:
            print(
                "Warning: %d cells have zero counts of overdispersed genes - ignoring these cells for factorization."
                % (zerocells.sum())
            )
            sc.pp.filter_cells(norm_counts, min_counts=1)

        return norm_counts

    def save_norm_counts(self, norm_counts):
        self._initialize_dirs()
        norm_counts.write(self.paths["normalized_counts"], compression="gzip")

    def get_nmf_iter_params(
        self, ks, n_iter=100, random_state_seed=None, beta_loss="kullback-leibler"
    ):
        """
        Create a DataFrame with parameters for NMF iterations.


        Parameters
        ----------
        ks : integer, or list-like.
            Number of topics (components) for factorization.
            Several values can be specified at the same time, which will be run independently.

        n_iter : integer, optional (defailt=100)
            Number of iterations for factorization. If several ``k`` are specified, this many
            iterations will be run for each value of ``k``.

        random_state_seed : int or None, optional (default=None)
            Seed for sklearn random state.

        """

        if type(ks) is int:
            ks = [ks]

        # Remove any repeated k values, and order.
        k_list = sorted(set(list(ks)))

        n_runs = len(ks) * n_iter

        np.random.seed(seed=random_state_seed)
        nmf_seeds = np.random.randint(low=1, high=(2 ** 32) - 1, size=n_runs)

        replicate_params = []
        for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
            replicate_params.append([k, r, nmf_seeds[i]])
        replicate_params = pd.DataFrame(
            replicate_params, columns=["n_components", "iter", "nmf_seed"]
        )

        _nmf_kwargs = dict(
            alpha=0.0,
            l1_ratio=0.0,
            beta_loss=beta_loss,
            solver="mu",
            tol=1e-4,
            max_iter=400,
            regularization=None,
            init="random",
        )

        ## Coordinate descent is faster than multiplicative update but only works for frobenius
        if beta_loss == "frobenius":
            _nmf_kwargs["solver"] = "cd"

        return (replicate_params, _nmf_kwargs)

    def save_nmf_iter_params(self, replicate_params, run_params):
        self._initialize_dirs()
        save_df_to_npz(replicate_params, self.paths["nmf_replicate_parameters"])
        with open(self.paths["nmf_run_parameters"], "w") as F:
            yaml.dump(run_params, F)

    def _nmf(self, X, nmf_kwargs):
        """
        Parameters
        ----------
        X : pandas.DataFrame,
            Normalized counts dataFrame to be factorized.

        nmf_kwargs : dict,
            Arguments to be passed to ``non_negative_factorization``

        """
        (usages, spectra, niter) = non_negative_factorization(X, **nmf_kwargs)

        return (spectra, usages)

    def run_nmf(
        self, worker_i=1, total_workers=1,
    ):
        """
        Iteratively run NMF with prespecified parameters.

        Use the `worker_i` and `total_workers` parameters for parallelization.

        Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'], defaults below::

            ``non_negative_factorization`` default arguments:
                alpha=0.0
                l1_ratio=0.0
                beta_loss='kullback-leibler'
                solver='mu'
                tol=1e-4,
                max_iter=200
                regularization=None
                init='random'
                random_state, n_components are both set by the prespecified self.paths['nmf_replicate_parameters'].


        Parameters
        ----------
        norm_counts : pandas.DataFrame,
            Normalized counts dataFrame to be factorized.
            (Output of ``normalize_counts``)

        run_params : pandas.DataFrame,
            Parameters for NMF iterations.
            (Output of ``prepare_nmf_iter_params``)

        """
        self._initialize_dirs()
        run_params = load_df_from_npz(self.paths["nmf_replicate_parameters"])
        norm_counts = sc.read(self.paths["normalized_counts"])
        _nmf_kwargs = yaml.load(
            open(self.paths["nmf_run_parameters"]), Loader=yaml.FullLoader
        )

        jobs_for_this_worker = worker_filter(
            range(len(run_params)), worker_i, total_workers
        )
        for idx in jobs_for_this_worker:

            p = run_params.iloc[idx, :]
            print("[Worker %d]. Starting task %d." % (worker_i, idx))
            _nmf_kwargs["random_state"] = p["nmf_seed"]
            _nmf_kwargs["n_components"] = p["n_components"]

            (spectra, usages) = self._nmf(norm_counts.X, _nmf_kwargs)
            spectra = pd.DataFrame(
                spectra,
                index=np.arange(1, _nmf_kwargs["n_components"] + 1),
                columns=norm_counts.var.index,
            )
            save_df_to_npz(
                spectra, self.paths["iter_spectra"] % (p["n_components"], p["iter"])
            )

    def combine_nmf(self, k, remove_individual_iterations=False):
        run_params = load_df_from_npz(self.paths["nmf_replicate_parameters"])
        print("Combining factorizations for k=%d." % k)

        self._initialize_dirs()

        combined_spectra = None
        n_iter = sum(run_params.n_components == k)

        run_params_subset = run_params[run_params.n_components == k].sort_values("iter")
        spectra_labels = []

        for i, p in run_params_subset.iterrows():

            spectra = load_df_from_npz(
                self.paths["iter_spectra"] % (p["n_components"], p["iter"])
            )
            if combined_spectra is None:
                combined_spectra = np.zeros((n_iter, k, spectra.shape[1]))
            combined_spectra[p["iter"], :, :] = spectra.values

            for t in range(k):
                spectra_labels.append("iter%d_topic%d" % (p["iter"], t + 1))

        combined_spectra = combined_spectra.reshape(-1, combined_spectra.shape[-1])
        combined_spectra = pd.DataFrame(
            combined_spectra, columns=spectra.columns, index=spectra_labels
        )

        save_df_to_npz(combined_spectra, self.paths["merged_spectra"] % k)
        return combined_spectra

    def consensus(
        self,
        k,
        density_threshold_str="0.5",
        local_neighborhood_size=0.30,
        show_clustering=True,
        skip_density_and_return_after_stats=False,
        close_clustergram_fig=True,
    ):
        merged_spectra = load_df_from_npz(self.paths["merged_spectra"] % k)
        norm_counts = sc.read(self.paths["normalized_counts"])

        if skip_density_and_return_after_stats:
            density_threshold_str = "2"
        density_threshold_repl = density_threshold_str.replace(".", "_")
        density_threshold = float(density_threshold_str)
        n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0] / k)

        # Rescale topics such to length of 1.
        l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra ** 2).sum(axis=1))).T

        if not skip_density_and_return_after_stats:
            # Compute the local density matrix (if not previously cached)
            topics_dist = None
            if os.path.isfile(self.paths["local_density_cache"] % k):
                local_density = load_df_from_npz(self.paths["local_density_cache"] % k)
            else:
                #   first find the full distance matrix
                topics_dist = squareform(fast_euclidean(l2_spectra.values))
                #   partition based on the first n neighbors
                partitioning_order = np.argpartition(topics_dist, n_neighbors + 1)[
                    :, : n_neighbors + 1
                ]
                #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
                distance_to_nearest_neighbors = topics_dist[
                    np.arange(topics_dist.shape[0])[:, None], partitioning_order
                ]
                local_density = pd.DataFrame(
                    distance_to_nearest_neighbors.sum(1) / (n_neighbors),
                    columns=["local_density"],
                    index=l2_spectra.index,
                )
                save_df_to_npz(local_density, self.paths["local_density_cache"] % k)
                del partitioning_order
                del distance_to_nearest_neighbors

            density_filter = local_density.iloc[:, 0] < density_threshold
            l2_spectra = l2_spectra.loc[density_filter, :]

        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans_model.fit(l2_spectra)
        kmeans_cluster_labels = pd.Series(
            kmeans_model.labels_ + 1, index=l2_spectra.index
        )

        # Find median usage for each gene across cluster
        median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

        # Normalize median spectra to probability distributions.
        median_spectra = (median_spectra.T / median_spectra.sum(1)).T

        # Compute the silhouette score
        stability = silhouette_score(
            l2_spectra.values, kmeans_cluster_labels, metric="euclidean"
        )

        # Obtain the reconstructed count matrix by re-fitting the usage matrix and computing the dot product: usage.dot(spectra)
        refit_nmf_kwargs = yaml.load(
            open(self.paths["nmf_run_parameters"]), Loader=yaml.FullLoader
        )
        refit_nmf_kwargs.update(
            dict(n_components=k, H=median_spectra.values, update_H=False)
        )

        # ensure dtypes match for factorization
        if median_spectra.values.dtype != norm_counts.X.dtype:
            norm_counts.X = norm_counts.X.astype(median_spectra.values.dtype)

        _, rf_usages = self._nmf(norm_counts.X, nmf_kwargs=refit_nmf_kwargs)
        rf_usages = pd.DataFrame(
            rf_usages, index=norm_counts.obs.index, columns=median_spectra.index
        )
        rf_pred_norm_counts = rf_usages.dot(median_spectra)

        # Compute prediction error as a frobenius norm
        if sp.issparse(norm_counts.X):
            prediction_error = (
                ((norm_counts.X.todense() - rf_pred_norm_counts) ** 2).sum().sum()
            )
        else:
            prediction_error = ((norm_counts.X - rf_pred_norm_counts) ** 2).sum().sum()

        consensus_stats = pd.DataFrame(
            [k, density_threshold, stability, prediction_error],
            index=["k", "local_density_threshold", "stability", "prediction_error"],
            columns=["stats"],
        )

        if skip_density_and_return_after_stats:
            return consensus_stats

        save_df_to_npz(
            median_spectra,
            self.paths["consensus_spectra"] % (k, density_threshold_repl),
        )
        save_df_to_npz(
            rf_usages, self.paths["consensus_usages"] % (k, density_threshold_repl)
        )
        save_df_to_npz(
            consensus_stats, self.paths["consensus_stats"] % (k, density_threshold_repl)
        )
        save_df_to_text(
            median_spectra,
            self.paths["consensus_spectra__txt"] % (k, density_threshold_repl),
        )
        save_df_to_text(
            rf_usages, self.paths["consensus_usages__txt"] % (k, density_threshold_repl)
        )

        # Compute gene-scores for each GEP by regressing usage on Z-scores of TPM
        tpm = sc.read(self.paths["tpm"])
        # ignore cells not present in norm_counts
        if tpm.n_obs != norm_counts.n_obs:
            tpm = tpm[norm_counts.obs_names, :].copy()
        tpm_stats = load_df_from_npz(self.paths["tpm_stats"])

        if sp.issparse(tpm.X):
            norm_tpm = (
                np.array(tpm.X.todense()) - tpm_stats["__mean"].values
            ) / tpm_stats["__std"].values
        else:
            norm_tpm = (tpm.X - tpm_stats["__mean"].values) / tpm_stats["__std"].values

        usage_coef = fast_ols_all_cols(rf_usages.values, norm_tpm)
        usage_coef = pd.DataFrame(
            usage_coef, index=rf_usages.columns, columns=tpm.var.index
        )

        save_df_to_npz(
            usage_coef, self.paths["gene_spectra_score"] % (k, density_threshold_repl)
        )
        save_df_to_text(
            usage_coef,
            self.paths["gene_spectra_score__txt"] % (k, density_threshold_repl),
        )

        # Convert spectra to TPM units, and obtain results for all genes by running
        # last step of NMF with usages fixed and TPM as the input matrix
        norm_usages = rf_usages.div(rf_usages.sum(axis=1), axis=0)
        refit_nmf_kwargs.update(dict(H=norm_usages.T.values,))

        # ensure dtypes match for factorization
        if norm_usages.values.dtype != tpm.X.dtype:
            tpm.X = tpm.X.astype(norm_usages.values.dtype)

        _, spectra_tpm = self._nmf(tpm.X.T, nmf_kwargs=refit_nmf_kwargs)
        spectra_tpm = pd.DataFrame(
            spectra_tpm.T, index=rf_usages.columns, columns=tpm.var.index
        )
        save_df_to_npz(
            spectra_tpm, self.paths["gene_spectra_tpm"] % (k, density_threshold_repl)
        )
        save_df_to_text(
            spectra_tpm,
            self.paths["gene_spectra_tpm__txt"] % (k, density_threshold_repl),
        )

        if show_clustering:
            if topics_dist is None:
                topics_dist = squareform(fast_euclidean(l2_spectra.values))
                # (l2_spectra was already filtered using the density filter)
            else:
                # (but the previously computed topics_dist was not!)
                topics_dist = topics_dist[density_filter.values, :][
                    :, density_filter.values
                ]

            spectra_order = []
            for cl in sorted(set(kmeans_cluster_labels)):

                cl_filter = kmeans_cluster_labels == cl

                if cl_filter.sum() > 1:
                    cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter])
                    cl_dist[
                        cl_dist < 0
                    ] = 0  # Rarely get floating point arithmetic issues
                    cl_link = linkage(cl_dist, "average")
                    cl_leaves_order = leaves_list(cl_link)

                    spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
                else:
                    ## Corner case where a component only has one element
                    spectra_order += list(np.where(cl_filter)[0])

            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            width_ratios = [0.5, 9, 0.5, 4, 1]
            height_ratios = [0.5, 9]
            fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(
                len(height_ratios),
                len(width_ratios),
                fig,
                0.01,
                0.01,
                0.98,
                0.98,
                height_ratios=height_ratios,
                width_ratios=width_ratios,
                wspace=0,
                hspace=0,
            )

            dist_ax = fig.add_subplot(
                gs[1, 1],
                xscale="linear",
                yscale="linear",
                xticks=[],
                yticks=[],
                xlabel="",
                ylabel="",
                frameon=True,
            )

            D = topics_dist[spectra_order, :][:, spectra_order]
            dist_im = dist_ax.imshow(
                D, interpolation="none", cmap="viridis", aspect="auto", rasterized=True
            )

            left_ax = fig.add_subplot(
                gs[1, 0],
                xscale="linear",
                yscale="linear",
                xticks=[],
                yticks=[],
                xlabel="",
                ylabel="",
                frameon=True,
            )
            left_ax.imshow(
                kmeans_cluster_labels.values[spectra_order].reshape(-1, 1),
                interpolation="none",
                cmap="Spectral",
                aspect="auto",
                rasterized=True,
            )

            top_ax = fig.add_subplot(
                gs[0, 1],
                xscale="linear",
                yscale="linear",
                xticks=[],
                yticks=[],
                xlabel="",
                ylabel="",
                frameon=True,
            )
            top_ax.imshow(
                kmeans_cluster_labels.values[spectra_order].reshape(1, -1),
                interpolation="none",
                cmap="Spectral",
                aspect="auto",
                rasterized=True,
            )

            hist_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs[1, 3], wspace=0, hspace=0
            )

            hist_ax = fig.add_subplot(
                hist_gs[0, 0],
                xscale="linear",
                yscale="linear",
                xlabel="",
                ylabel="",
                frameon=True,
                title="Local density histogram",
            )
            hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            hist_ax.yaxis.tick_right()

            xlim = hist_ax.get_xlim()
            ylim = hist_ax.get_ylim()
            if density_threshold < xlim[1]:
                hist_ax.axvline(density_threshold, linestyle="--", color="k")
                hist_ax.text(
                    density_threshold + 0.02,
                    ylim[1] * 0.95,
                    "filtering\nthreshold\n\n",
                    va="top",
                )
            hist_ax.set_xlim(xlim)
            hist_ax.set_xlabel(
                "Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering"
                % (
                    sum(~density_filter),
                    len(density_filter),
                    100 * (~density_filter).mean(),
                )
            )

            fig.savefig(
                self.paths["clustering_plot"] % (k, density_threshold_repl), dpi=250
            )
            if close_clustergram_fig:
                plt.close(fig)

    def k_selection_plot(self, close_fig=True):
        """
        Borrowed from Alexandrov Et Al. 2013 Deciphering Mutational Signatures
        publication in Cell Reports
        """
        run_params = load_df_from_npz(self.paths["nmf_replicate_parameters"])
        stats = []
        for k in sorted(set(run_params.n_components)):

            stats.append(
                self.consensus(k, skip_density_and_return_after_stats=True).stats
            )

        stats = pd.DataFrame(stats)
        stats.reset_index(drop=True, inplace=True)

        save_df_to_npz(stats, self.paths["k_selection_stats"])

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(stats.k, stats.stability, "o-", color="b")
        ax1.set_ylabel("Stability", color="b", fontsize=15)
        for tl in ax1.get_yticklabels():
            tl.set_color("b")
        # ax1.set_xlabel('K', fontsize=15)

        ax2.plot(stats.k, stats.prediction_error, "o-", color="r")
        ax2.set_ylabel("Error", color="r", fontsize=15)
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        ax1.set_xlabel("Number of Components", fontsize=15)
        ax1.grid(True)
        plt.tight_layout()
        fig.savefig(self.paths["k_selection_plot"], dpi=250)
        if close_fig:
            plt.close(fig)


def pick_k(k_selection_stats_path):
    k_sel_stats = load_df_from_npz(k_selection_stats_path)
    return int(k_sel_stats.loc[k_sel_stats.stability.idxmax, "k"])


def prepare(args):
    argdict = vars(args)

    cnmf_obj = cNMF(output_dir=argdict["output_dir"], name=argdict["name"])
    cnmf_obj._initialize_dirs()
    print("Reading in counts from {} - ".format(argdict["counts"]), end="")
    if argdict["counts"].endswith(".h5ad"):
        input_counts = sc.read(argdict["counts"])
    else:
        ## Load txt or compressed dataframe and convert to scanpy object
        if argdict["counts"].endswith(".npz"):
            input_counts = load_df_from_npz(argdict["counts"])
        else:
            input_counts = pd.read_csv(argdict["counts"], sep="\t", index_col=0)

        if argdict["densify"]:
            input_counts = sc.AnnData(
                X=input_counts.values,
                obs=pd.DataFrame(index=input_counts.index),
                var=pd.DataFrame(index=input_counts.columns),
            )
        else:
            input_counts = sc.AnnData(
                X=sp.csr_matrix(input_counts.values),
                obs=pd.DataFrame(index=input_counts.index),
                var=pd.DataFrame(index=input_counts.columns),
            )
    print("{} cells and {} genes".format(input_counts.n_obs, input_counts.n_vars))

    # use desired layer if not .X
    if args.layer is not None:
        print("Using layer '{}' for cNMF".format(args.layer))
        input_counts.X = input_counts.layers[args.layer].copy()

    if sp.issparse(input_counts.X) & argdict["densify"]:
        input_counts.X = np.array(input_counts.X.todense())

    if argdict["tpm"] is None:
        tpm = compute_tpm(input_counts)
    elif argdict["tpm"].endswith(".h5ad"):
        subprocess.call(
            "cp %s %s" % (argdict["tpm"], cnmf_obj.paths["tpm"]), shell=True
        )
        tpm = sc.read(cnmf_obj.paths["tpm"])
    else:
        if argdict["tpm"].endswith(".npz"):
            tpm = load_df_from_npz(argdict["tpm"])
        else:
            tpm = pd.read_csv(argdict["tpm"], sep="\t", index_col=0)

        if argdict["densify"]:
            tpm = sc.AnnData(
                X=tpm.values,
                obs=pd.DataFrame(index=tpm.index),
                var=pd.DataFrame(index=tpm.columns),
            )
        else:
            tpm = sc.AnnData(
                X=sp.csr_matrix(tpm.values),
                obs=pd.DataFrame(index=tpm.index),
                var=pd.DataFrame(index=tpm.columns),
            )

    if argdict["subset"]:
        tpm = subset_adata(tpm, subset=argdict["subset"])

    n_null = tpm.n_vars - tpm.X.sum(axis=0).astype(bool).sum()
    if n_null > 0:
        sc.pp.filter_genes(tpm, min_counts=1)
        print(
            "Removing {} genes with zero counts; final shape {}".format(
                n_null, tpm.shape
            )
        )
    tpm.write(cnmf_obj.paths["tpm"], compression="gzip")

    if sp.issparse(tpm.X):
        gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
        gene_tpm_stddev = var_sparse_matrix(tpm.X) ** 0.5
    else:
        gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
        gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)

    input_tpm_stats = pd.DataFrame(
        [gene_tpm_mean, gene_tpm_stddev], index=["__mean", "__std"]
    ).T
    save_df_to_npz(input_tpm_stats, cnmf_obj.paths["tpm_stats"])

    if argdict["genes_file"] is not None:
        highvargenes = open(argdict["genes_file"]).read().rstrip().split("\n")
    else:
        highvargenes = None

    norm_counts = cnmf_obj.get_norm_counts(
        input_counts,
        tpm,
        num_highvar_genes=argdict["numgenes"],
        high_variance_genes_filter=highvargenes,
    )
    cnmf_obj.save_norm_counts(norm_counts)
    (replicate_params, run_params) = cnmf_obj.get_nmf_iter_params(
        ks=argdict["components"],
        n_iter=argdict["n_iter"],
        random_state_seed=argdict["seed"],
        beta_loss=argdict["beta_loss"],
    )
    cnmf_obj.save_nmf_iter_params(replicate_params, run_params)


def factorize(args):
    argdict = vars(args)

    cnmf_obj = cNMF(output_dir=argdict["output_dir"], name=argdict["name"])
    cnmf_obj._initialize_dirs()

    cnmf_obj.run_nmf(worker_i=argdict["worker_index"], total_workers=argdict["n_jobs"])


def combine(args):
    argdict = vars(args)

    cnmf_obj = cNMF(output_dir=argdict["output_dir"], name=argdict["name"])
    cnmf_obj._initialize_dirs()
    run_params = load_df_from_npz(cnmf_obj.paths["nmf_replicate_parameters"])

    if type(args.components) is int:
        ks = [args.components]
    elif argdict["components"] is None:
        ks = sorted(set(run_params.n_components))
    else:
        ks = argdict["components"]

    for k in ks:
        cnmf_obj.combine_nmf(k)


def consensus(args):
    argdict = vars(args)

    cnmf_obj = cNMF(output_dir=argdict["output_dir"], name=argdict["name"])
    cnmf_obj._initialize_dirs()
    run_params = load_df_from_npz(cnmf_obj.paths["nmf_replicate_parameters"])

    if argdict["auto_k"]:
        argdict["components"] = pick_k(cnmf_obj.paths["k_selection_stats"])

    if type(argdict["components"]) is int:
        ks = [argdict["components"]]
    elif argdict["components"] is None:
        ks = sorted(set(run_params.n_components))
    else:
        ks = argdict["components"]

    for k in ks:
        merged_spectra = load_df_from_npz(cnmf_obj.paths["merged_spectra"] % k)
        cnmf_obj.consensus(
            k,
            argdict["local_density_threshold"],
            argdict["local_neighborhood_size"],
        )
        tpm = sc.read(cnmf_obj.paths["tpm"])
        tpm.X = tpm.layers["raw_counts"].copy()
        cnmf_load_results(
            tpm,
            cnmf_dir=cnmf_obj.output_dir,
            name=cnmf_obj.name,
            k=k,
            dt=argdict["local_density_threshold"],
            key="cnmf",
        )
        tpm.write(
            os.path.join(
                cnmf_obj.output_dir,
                cnmf_obj.name,
                cnmf_obj.name
                + "_k{}_dt{}.h5ad".format(
                    str(k), str(argdict["local_density_threshold"]).replace(".", "_"),
                ),
            ),
            compression="gzip",
        )

    if argdict["cleanup"]:
        files = (
            glob.glob("{}/{}/*.consensus.*".format(args.output_dir, args.name))
            + glob.glob(
                "{}/{}/cnmf_tmp/*.consensus.*".format(args.output_dir, args.name)
            )
            + glob.glob("{}/{}/*.gene_spectra_*".format(args.output_dir, args.name))
            + glob.glob(
                "{}/{}/cnmf_tmp/*.gene_spectra_*".format(args.output_dir, args.name)
            )
            + glob.glob(
                "{}/{}/cnmf_tmp/*.local_density_cache.*".format(
                    args.output_dir, args.name
                )
            )
            + glob.glob("{}/{}/cnmf_tmp/*.stats.*".format(args.output_dir, args.name))
        )
        for file in files:
            os.remove(file)


def k_selection(args):
    argdict = vars(args)

    cnmf_obj = cNMF(output_dir=argdict["output_dir"], name=argdict["name"])
    cnmf_obj._initialize_dirs()

    cnmf_obj.k_selection_plot()


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="cnmf")
    parser.add_argument(
        "-V", "--version", action="version", version=get_versions()["version"],
    )
    subparsers = parser.add_subparsers()

    prepare_parser = subparsers.add_parser(
        "prepare", help="Prep scRNA-seq data for cNMF analysis.",
    )
    prepare_parser.add_argument(
        "counts",
        type=str,
        nargs="?",
        help="Input (cell x gene) counts matrix as .h5ad, df.npz, or tab delimited text file",
    )
    prepare_parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    prepare_parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        help="Total number of workers to distribute jobs to",
        default=1,
    )
    prepare_parser.add_argument(
        "-k",
        "--components",
        type=int,
        help='Numper of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"',
        nargs="*",
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    prepare_parser.add_argument(
        "-n",
        "--n-iter",
        type=int,
        help="Numper of factorization replicates",
        default=50,
    )
    prepare_parser.add_argument(
        "--subset",
        help="AnnData.obs column name to subset on before performing NMF. Cells to keep should be True or 1",
        nargs="*",
    )
    prepare_parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        help="Key from .layers to use. Default '.X'.",
    )
    prepare_parser.add_argument(
        "--seed", type=int, help="Seed for pseudorandom number generation", default=18,
    )
    prepare_parser.add_argument(
        "--genes-file",
        type=str,
        help="File containing a list of genes to include, one gene per line. Must match column labels of counts matrix.",
        default=None,
    )
    prepare_parser.add_argument(
        "--numgenes",
        type=int,
        help="Number of high variance genes to use for matrix factorization.",
        default=2000,
    )
    prepare_parser.add_argument(
        "--tpm",
        type=str,
        help="Pre-computed (cell x gene) TPM values as df.npz or tab separated txt file. If not provided TPM will be calculated automatically",
        default=None,
    )
    prepare_parser.add_argument(
        "--beta-loss",
        type=str,
        choices=["frobenius", "kullback-leibler", "itakura-saito"],
        help="Loss function for NMF.",
        default="frobenius",
    )
    prepare_parser.add_argument(
        "--densify",
        dest="densify",
        help="Treat the input data as non-sparse",
        action="store_true",
        default=False,
    )
    prepare_parser.set_defaults(func=prepare)

    factorize_parser = subparsers.add_parser(
        "factorize", help="Run NMF iteratively to generate factors for consensus.",
    )
    factorize_parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    factorize_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    factorize_parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        help="Total number of workers to distribute jobs to",
        default=1,
    )
    factorize_parser.add_argument(
        "--worker-index",
        type=int,
        help="Index of current worker (the first worker should have index 0)",
        default=0,
    )
    factorize_parser.set_defaults(func=factorize)

    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine factors from NMF iterations and calculate stats for choosing consensus.",
    )
    combine_parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    combine_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    combine_parser.add_argument(
        "-k",
        "--components",
        type=int,
        help='Number of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"',
        nargs="*",
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    combine_parser.set_defaults(func=combine)

    consensus_parser = subparsers.add_parser(
        "consensus", help="Calculate consensus factors from NMF iterations.",
    )
    consensus_parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    consensus_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    consensus_parser.add_argument(
        "-k",
        "--components",
        type=int,
        help='Numper of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"',
        nargs="*",
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    consensus_parser.add_argument(
        "--auto-k",
        help="Automatically pick k value for consensus based on maximum stability",
        action="store_true",
    )
    consensus_parser.add_argument(
        "--local-density-threshold",
        type=str,
        help="Threshold for the local density filtering. This string must convert to a float >0 and <=2",
        default="0.1",
    )
    consensus_parser.add_argument(
        "--local-neighborhood-size",
        type=float,
        help="Fraction of the number of replicates to use as nearest neighbors for local density filtering",
        default=0.30,
    )
    consensus_parser.add_argument(
        "--cleanup",
        help="Remove excess files after saving results to clean workspace",
        action="store_true",
    )
    consensus_parser.set_defaults(func=consensus)

    k_selection_parser = subparsers.add_parser(
        "k_selection_plot",
        help="Plot stats across k values to choose optimal k for consensus.",
    )
    k_selection_parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    k_selection_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    k_selection_parser.set_defaults(func=k_selection)

    args = parser.parse_args()
    args.func(args)
