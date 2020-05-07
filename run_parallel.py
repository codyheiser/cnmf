import argparse, sys, os
import subprocess as sp

"""
Run all of the steps through plotting the K selection plot of cNMF sequentially using GNU
parallel to run the factorization steps in parallel. The same optional arguments are available
as for running the individual steps of cnmf.py

Example command:
python run_parallel.py --output-dir $output_dir \
            --name test --counts path_to_counts.df.npz \
            -k 6 7 8 9 --n-iter 5 --total-workers 2 \
            --seed 5
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "counts",
        type=str,
        nargs="?",
        help="[prepare] Input (cell x gene) counts matrix as .h5ad, df.npz, or tab delimited text file",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="[all] Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="cNMF",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="[all] Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "--total-workers",
        type=int,
        help="[all] Total number of workers to distribute jobs to",
        default=1,
    )

    parser.add_argument(
        "-k",
        "--components",
        type=int,
        help='[prepare] Numper of components (k) for matrix factorization. Several can be specified with "-k 8 9 10"',
        nargs="*",
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    parser.add_argument(
        "-n",
        "--n-iter",
        type=int,
        help="[prepare] Number of factorization replicates",
        default=50,
    )
    parser.add_argument(
        "--subset",
        help="[prepare] AnnData.obs column name to subset on before performing NMF",
        nargs="*",
    )
    parser.add_argument(
        "--subset-val",
        dest="subset_val",
        help="[prepare] Value to match in AnnData.obs[args.subset]",
        nargs="*",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        help="[prepare] Key from .layers to use. Default '.X'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="[prepare] Seed for pseudorandom number generation",
        default=18,
    )
    parser.add_argument(
        "--genes-file",
        type=str,
        help="[prepare] File containing a list of genes to include, one gene per line. Must match column labels of counts matrix.",
        default=None,
    )
    parser.add_argument(
        "--numgenes",
        type=int,
        help="[prepare] Number of high variance genes to use for matrix factorization.",
        default=2000,
    )
    parser.add_argument(
        "--tpm",
        type=str,
        help="[prepare] Pre-computed (cell x gene) TPM values as df.npz or tab separated txt file. If not provided TPM will be calculated automatically",
        default=None,
    )
    parser.add_argument(
        "--beta-loss",
        type=str,
        choices=["frobenius", "kullback-leibler", "itakura-saito"],
        help="[prepare] Loss function for NMF.",
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
        "--worker-index",
        type=int,
        help="[factorize] Index of current worker (the first worker should have index 0)",
        default=0,
    )

    parser.add_argument(
        "--auto-k",
        help="[consensus] Automatically pick k value for consensus based on maximum stability",
        action="store_true",
    )
    parser.add_argument(
        "--local-density-threshold",
        type=str,
        help="[consensus] Threshold for the local density filtering. This string must convert to a float >0 and <=2",
        default="0.1",
    )
    parser.add_argument(
        "--local-neighborhood-size",
        type=float,
        help="[consensus] Fraction of the number of replicates to use as nearest neighbors for local density filtering",
        default=0.30,
    )
    parser.add_argument(
        "--show-clustering",
        dest="show_clustering",
        help="[consensus] Produce a clustergram figure summarizing the spectra clustering",
        action="store_true",
    )
    parser.add_argument(
        "--cleanup",
        help="[consensus] Remove excess files after saving results to clean workspace",
        action="store_true",
    )

    # Collect args
    args = parser.parse_args()
    argdict = vars(args)

    # convert arguments from list to string for passing to cnmf.py
    argdict["components"] = " ".join([str(k) for k in argdict["components"]])
    if argdict["subset"]:
        argdict["subset"] = " ".join([str(k) for k in argdict["subset"]])
        argdict["subset_val"] = " ".join([str(k) for k in argdict["subset_val"]])

    # Directory containing cNMF and this script
    cnmfdir = os.path.dirname(sys.argv[0])
    if len(cnmfdir) == 0:
        cnmfdir = "."

    # Run prepare
    counts_arg = argdict["counts"]
    del argdict["counts"]
    prepare_opts = [
        "--{} {}".format(k.replace("_", "-"), argdict[k])
        for k in argdict.keys()
        if (argdict[k] is not None) and not isinstance(argdict[k], bool)
    ]
    prepare_cmd = "python {}/cnmf.py prepare {} ".format(cnmfdir, counts_arg)
    prepare_cmd += " ".join(prepare_opts)
    print(prepare_cmd)
    sp.call(prepare_cmd, shell=True)

    # Run factorize
    workind = " ".join([str(x) for x in range(argdict["total_workers"])])
    factorize_cmd = (
        "nohup parallel python %s/cnmf.py factorize --output-dir %s --name %s --worker-index {} ::: %s"
        % (cnmfdir, argdict["output_dir"], argdict["name"], workind)
    )
    print(factorize_cmd)
    sp.call(factorize_cmd, shell=True)

    # Run combine
    combine_cmd = "python %s/cnmf.py combine --output-dir %s --name %s" % (
        cnmfdir,
        argdict["output_dir"],
        argdict["name"],
    )
    print(combine_cmd)
    sp.call(combine_cmd, shell=True)

    # Plot K selection
    Kselect_cmd = "python %s/cnmf.py k_selection_plot --output-dir %s --name %s" % (
        cnmfdir,
        argdict["output_dir"],
        argdict["name"],
    )
    print(Kselect_cmd)
    sp.call(Kselect_cmd, shell=True)

    # Delete individual iteration files
    clean_cmd = "rm %s/%s/cnmf_tmp/*.iter_*.df.npz" % (
        argdict["output_dir"],
        argdict["name"],
    )
    print(clean_cmd)
    sp.call(clean_cmd, shell=True)

    if argdict["auto_k"]:
        consensus_cmd = "python {}/cnmf.py consensus --output-dir {} --name {} --auto-k --local-density-threshold {}".format(
            cnmfdir,
            argdict["output_dir"],
            argdict["name"],
            argdict["local_density_threshold"],
        )
        if argdict["show_clustering"]:
            consensus_cmd = " ".join([consensus_cmd, "--show-clustering"])
        if argdict["cleanup"]:
            consensus_cmd = " ".join([consensus_cmd, "--cleanup"])
        print(consensus_cmd)
        sp.call(consensus_cmd, shell=True)


if __name__ == "__main__":
    main()
