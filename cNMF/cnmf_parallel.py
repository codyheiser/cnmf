# -*- coding: utf-8 -*-
"""
consensus non-negative matrix factorization (cNMF) adapted from (Kotliar, et al. 2019)
entire pipeline run in parallel using GNU parallel

@author: C Heiser
2020
"""
import sys, os
import subprocess as sp
from ._version import get_versions


def parallel(args):
    argdict = vars(args)

    # convert arguments from list to string for passing to cnmf.py
    argdict["components"] = " ".join([str(k) for k in argdict["components"]])
    if argdict["subset"]:
        argdict["subset"] = " ".join([str(k) for k in argdict["subset"]])

    # remove arguments from dictionary prior to prepare
    counts_arg = argdict["counts"]
    del argdict["counts"]
    local_dens_thresh_arg = argdict["local_density_threshold"]
    del argdict["local_density_threshold"]
    local_neighborhood_size_arg = argdict["local_neighborhood_size"]
    del argdict["local_neighborhood_size"]

    # Run prepare
    prepare_opts = [
        "--{} {}".format(k.replace("_", "-"), argdict[k])
        for k in argdict.keys()
        if (argdict[k] is not None) and not isinstance(argdict[k], bool)
    ]
    prepare_cmd = "cnmf prepare {} ".format(counts_arg)
    prepare_cmd += " ".join(prepare_opts)
    print("Preparing directories and preprocessing:  {}".format(prepare_cmd))
    sp.call(prepare_cmd, shell=True)

    # Run factorize
    workind = " ".join([str(x) for x in range(argdict["n_jobs"])])
    factorize_cmd = (
        "nohup parallel cnmf factorize --output-dir %s --name %s --worker-index {} ::: %s"
        % (argdict["output_dir"], argdict["name"], workind)
    )
    print("Running iterative NMF:  {}".format(factorize_cmd))
    sp.call(factorize_cmd, shell=True)

    # Run combine
    combine_cmd = "cnmf combine --output-dir %s --name %s --components %s" % (
        argdict["output_dir"],
        argdict["name"],
        argdict["components"],
    )
    print("Combining NMF replicates:  {}".format(combine_cmd))
    sp.call(combine_cmd, shell=True)

    # Plot K selection
    Kselect_cmd = "cnmf k_selection_plot --output-dir %s --name %s" % (
        argdict["output_dir"],
        argdict["name"],
    )
    print("Plotting K selection parameters:  {}".format(Kselect_cmd))
    sp.call(Kselect_cmd, shell=True)

    # Delete individual iteration files
    clean_cmd = "rm %s/%s/cnmf_tmp/*.iter_*.df.npz" % (
        argdict["output_dir"],
        argdict["name"],
    )
    print("Cleaning up workspace:  {}".format(clean_cmd))
    sp.call(clean_cmd, shell=True)

    if argdict["auto_k"]:
        consensus_cmd = "cnmf consensus --output-dir {} --name {} --auto-k --local-density-threshold {} --local-neighborhood-size {}".format(
            argdict["output_dir"],
            argdict["name"],
            local_dens_thresh_arg,
            local_neighborhood_size_arg,
        )
        if argdict["cleanup"]:
            consensus_cmd = " ".join([consensus_cmd, "--cleanup"])
        print("Building consensus factors:  {}".format(consensus_cmd))
        sp.call(consensus_cmd, shell=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="cnmf_p")
    parser.add_argument(
        "-V", "--version", action="version", version=get_versions()["version"],
    )

    parser.add_argument(
        "counts",
        type=str,
        nargs="?",
        help="Input (cell x gene) counts matrix as .h5ad, df.npz, or tab delimited text file",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        help="Total number of workers to distribute jobs to",
        default=1,
    )
    parser.add_argument(
        "-k",
        "--components",
        type=int,
        help='Number of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"',
        nargs="*",
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    parser.add_argument(
        "-n",
        "--n-iter",
        type=int,
        help="Number of factorization replicates",
        default=50,
    )
    parser.add_argument(
        "--subset",
        help="AnnData.obs column name to subset on before performing NMF. Cells to keep should be True or 1",
        nargs="*",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        help="Key from .layers to use. Default '.X'.",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for pseudorandom number generation", default=18,
    )
    parser.add_argument(
        "--genes-file",
        type=str,
        help="File containing a list of genes to include, one gene per line. Must match column labels of counts matrix.",
        default=None,
    )
    parser.add_argument(
        "--numgenes",
        type=int,
        help="Number of high variance genes to use for matrix factorization.",
        default=2000,
    )
    parser.add_argument(
        "--tpm",
        type=str,
        help="Pre-computed (cell x gene) TPM values as df.npz or tab separated txt file. If not provided TPM will be calculated automatically",
        default=None,
    )
    parser.add_argument(
        "--beta-loss",
        type=str,
        choices=["frobenius", "kullback-leibler", "itakura-saito"],
        help="Loss function for NMF.",
        default="frobenius",
    )
    parser.add_argument(
        "--densify",
        dest="densify",
        help="[prepare] Treat the input data as non-sparse",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--auto-k",
        help="Automatically pick k value for consensus based on maximum stability",
        action="store_true",
    )
    parser.add_argument(
        "--local-density-threshold",
        type=str,
        help="Threshold for the local density filtering. This string must convert to a float >0 and <=2",
        default="0.1",
    )
    parser.add_argument(
        "--local-neighborhood-size",
        type=float,
        help="Fraction of the number of replicates to use as nearest neighbors for local density filtering",
        default=0.30,
    )
    parser.add_argument(
        "--cleanup",
        help="Remove excess files after saving results to clean workspace",
        action="store_true",
    )

    args = parser.parse_args()
    parallel(args)
