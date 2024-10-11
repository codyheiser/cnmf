# Consensus Non-negative Matrix Factorization (cNMF)

[![Latest Version][tag-version]][repo-url]

cNMF is an analysis pipeline for inferring gene expression programs from single-cell RNA-Seq (scRNA-Seq) data. It takes a counts matrix (N cells X G genes) as input and produces a (K x G) matrix of gene expression programs (GEPs) and a (N x K) matrix specifying the usage of each program for each cell in the data.

You can read more about the method in [Kotliar, et al. 2019](https://elifesciences.org/articles/43803).

# Installing cNMF

You can install a local version of the package by cloning this repository and running the following command from the main directory:

```
pip install -e .
```

This will install all python dependencies. To run the `cnmf_p` pipeline, you need to have [GNU Parallel](https://www.gnu.org/software/parallel/) installed on your machine as well.

## Parallelizing the factorization of cNMF

cNMF runs NMF multiple times and combines the results of each replicates to obtain a more robust consensus estimate. Since many replicates are run, typically for many choices of K, this can be much faster if replicates are run in parallel.

This cNMF code is designed to be agnostic to the method of parallelization. It divides up all of the factorization replicates into a specified number of "jobs" which could correspond to cores on a computer, nodes on a compute cluster, or virtual machines in the cloud. In any of these cases, if you are running cNMF for 5 values of K (K= 1..5) and 100 iterations each, there would be 500 total jobs. Those jobs could be divided up amongst 100 workers that would each run 5 jobs, 500 workers that would each run 1 job, or 1 worker that would run all 500 jobs.

You specify the total number of jobs in the prepare command (step 1) with the `--n-jobs` flag. Then, for step 2 you run all of the jobs for a specific worker using the `--worker-index` flag. Step 3 combines the results files that were output by all of the workers. The workers are indexed from 0 to (`n-jobs` - 1).

The following terminal command will show all arguments for parallel processing:

```
cnmf_p -h
```

# Input data and Scanpy

Input counts data can be provided to cNMF in 2 ways:

1. as a raw tab-delimited text file containing row labels with cell IDs (barcodes) and column labels as gene IDs
2. as a scanpy file ending in `.h5ad` containg counts as the data feature. Because Scanpy uses sparse matrices by default, the .h5ad data structure can take up much less memory than the raw counts matrix and is much faster to load.

# Step by step guide 

You can see all possible command line options by running:

```
cnmf {prepare, factorize, combine, consensus, k_selection_plot} -h
```

See the [simulated dataset tutorial](Tutorials/analyze_simulated_example_data.ipynb) and the [PBMC dataset tutorial](Tutorials/analyze_pbmc_example_data.ipynb) for a step by step walkthrough with example data. We also describe the key ideas and parameters for each step below.

### Step 1 - normalize the input matrix and prepare the run parameters

Example command:

```
cnmf prepare ./example_data/counts_prefiltered.txt --output-dir ./example_data --name example_cNMF -k 5 6 7 8 9 10 11 12 13 -n 30 -j 1
```

Path structure

  - `--output-dir` - the output directory into which all results will be placed. Default: `.`
  - `--name` - a subdirectory output_dir/name will be created and all output files will have name as their prefix. Default: `cNMF`

Input data

  - `counts` - path to the cell x gene counts file. This is expected to be a tab-delimited text file or a Scanpy object saved in the `.h5ad` format
  - `--tpm` [Optional] - Pre-computed Cell x Gene data in transcripts per million or other per-cell normalized data. If none is provided, TPM will be calculated automatically. This can be helpful if a particular normalization is desired. These can be loaded in the same formats as the counts file. Default: `None`
  - `--genes-file` [Optional] - List of over-dispersed genes to be used for the factorization steps. If not provided, over-dispersed genes will be calculated automatically and the number of genes to use can be set by the `--numgenes` parameter below. Default: `None`

Parameters

  - `-k` - space separated list of K values that will be tested for cNMF
  - `-n`, `--n-iter` - number of NMF iterations to run for each K.
  - `-j`, `--n-jobs` - specifies how many workers (e.g. cores on a machine or nodes on a compute farm) can be used in parallel. Default: `1`
  - `--seed` - the master seed that will be used to generate the individual seed for each NMF replicate. Default: `None`
  - `--numgenes` - the number of higest variance genes that will be used for running the factorization. Removing low variance genes helps amplify the signal and is an important factor in correctly inferring programs in the data. However, don't worry, at the end the spectra is re-fit to include estimates for all genes, even those that weren't included in the high-variance set. Default: `2000`
  - `--beta-loss` - Loss function for NMF, from one of `frobenius`, `kullback-leibler`, `itakura-saito`. Default: `frobenius`
  - `--densify` - flag indicating that unlike most single-cell RNA-Seq data, the input data is not sparse. Causes the data to be treated as dense. Not recommended for most single-cell RNA-Seq data Default: `False`
  - `--subset` - column(s) in `.obs` of `.h5ad` counts file to subset dataset on before running cNMF. Can specify more than one as a list. Keeps cells with values of `1` or `True` in any of the respective columns.
  - `-l`, `--layer` - key in `.layers` of `.h5ad` counts file to use for cNMF instead of `.X`.

This command generates a filtered and normalized matrix for running the factorizations on. It first subsets the data down to a set of over-dispersed genes that can be provided as an input file or calculated here. While the final spectra will be computed for all of the genes in the input counts file, the factorization is much faster and can find better patterns if it only runs on a set of high-variance genes. A per-cell normalized input file may be provided as well so that the final gene expression programs can be computed with respect to that normalization.

In addition, this command allocates specific factorization jobs to be run to distinct workers. The number of workers are specified by `--n-jobs`, and the total number of jobs is `--n-iter` X the number of Ks being tested.

In the example above, we are assuming that no parallelization is to be used (`--n-jobs` 1) and so all of the jobs are being allocated to a single worker.

__Please note that the input matrix should not include any cells or genes with 0 total counts. Furthermore if any of the cells end up having 0 counts for the over-dispersed genes, they will be ignored for analysis and resulting usage scores in those cells will be reported as 0 for all factors in the final AnnData object. It is best practice to filter out cells and genes with low counts prior to running cNMF.__

### Step 2 factorize the matrix

Next NMF is run for all of the replicates specified in the previous command. The tasks have been allocated to workers indexed from 0 ... (`--n-jobs` -1). You can run all of the NMF replicates allocated to a specific index like below for index 0 corresponding to the first worker:

```
cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 0 
```

This is running all of the jobs for worker 1. If you specified a single worker in the prepare step (`--n-jobs` 1) like in the command above, this will run all of the factorizations. However, if you specified more than 1 total worker, you would need to run the commands for those workers as well with separate commands, e.g.:

```
cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 1 
cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 2
...
```

You should submit these commands to distinct processors or machines so they are all run in parallel. See the [tutorial on simulated data](Tutorials/analyze_simulated_example_data.ipynb) and [PBMC tutorial](Tutorials/analyze_pbmc_example_data.ipynb) for examples of how you could submit all of the workers to run in parallel either using [GNU parralel](https://www.gnu.org/software/parallel/) or an [UGER scheduler](http://www.univa.com/resources/files/univa_user_guide_univa__grid_engine_854.pdf). 

__Tip: The implementation of NMF in scikit-learn by default will use more than 1 core if there are multiple cores on a machine. We find that we get the best performance by using 2 workers when using GNU parallel.__

### Step 3 combine the individual spectra results files for each K into a merged file

Since a separate file has been created for each replicate for each K, we combine the replicates for each K as below:

Example command:

```
cnmf combine --output-dir ./example_data --name example_cNMF
```

After this, you can optionally delete the individual spectra files like so (running `cnmf_p` does this automatically):

```
rm ./example_data/example_cNMF/cnmf_tmp/example_cNMF.spectra.k_*.iter_*.df.npz
```

### Step 4 select an optimal K by considering the trade-off between stability and error

This will iterate through all of the values of K that have been run and will calculate the stability and error. It then outputs a `.png` image file plotting this relationship into the output_dir/name directory.

Example command:

```
cnmf k_selection_plot --output-dir ./example_data --name example_cNMF
```

This outputs a K selection plot to example_data/example_cNMF/example_cNMF.k_selection.png. There is no universally definitive criteria for choosing K but we will typically use the largest value that is reasonably stable and/or a local maximum in stability. See the discussion and methods section and the response to reviewer comments in [the manuscript](https://elifesciences.org/articles/43803) for more discussion about selecting K.

### Step 5 obtain consensus estimates for the programs and their usages at the desired value of K

The last step is to cluster the spectra after first optionally filtering out outliers. This step outputs 4 files:
    - GEP estimate in units of TPM
    - GEP estimate in units of TPM Z-scores, reflecting whether having a higher usage of a program would be expected to decrease or increase gene expression)
    - Unnormalized GEP usage estimate. 
    - Clustergram diagnostic plot, showing how much consensus there is amongst the replicates and a histogram of distances between each spectra and its K nearest neighbors 

We recommend that you use the diagnostic plot to determine the threshold to filter outliers. By default cNMF sets the number of neighbors to use for this filtering as 30% of the number of iterations done (`0.30`). This can be modified from the command line.

Example command:

```
cnmf consensus --output-dir ./example_data --name example_cNMF --local-density-threshold 0.05 --show-clustering --auto-k --cleanup
```

  - `-k` - value(s) of K to compute consensus clusters for. Must be among the options provided to the prepare step
  - `--local-density-threshold` - the threshold on average distance to K nearest neighbors to use. 2.0 or above means that nothing will be filtered out. Default: 0.5
  - `--local-neighborhood-size` - Percentage of replicates to consider as nearest neighbors for local density filtering. E.g. if you run 100 replicates, and set this to .3, 30 nearest neighbors will be used for outlier detection. Default: 0.3
  - `--show-clustering` - Controls whether or not the clustergram image is output. Default: `False`
  - `--auto-k` - pick K value automatically from output of `combine`. Chooses K with highest stability value to run consensus on.
  - `--cleanup` - removes unnecessary files from output directory, while still allowing for running `consensus` again on results of `combine`.

### Results

cNMF deposits an `.h5ad` file in `output-dir/name/` for each `consensus` run on a particular K value, e.g. `example_data_k10_dt0_05.h5ad` for the above commands. This AnnData object can easily be read using Scanpy, and has the following attributes:

  - raw counts from original file in `.X` slot
  - NMF usages normalized within each cell (N x K fractional usages) in `.obsm["cnmf_usages"]`. This can be used to prime dimension-reduced embeddings such as PCA, t-SNE, and UMAP.
  - NMF usages for each cell as columns of `.obs`, used for plotting - `.obs[["usage_1","usage_2","usage_3",...]]`.
  - GEP spectra describing the loadings for each overdispersed gene in all K factors - `.varm["cnmf_spectra"]`.
  - ranked GEP loadings showing top genes in each factor - `.uns["cnmf_markers"]`.

---

Full documentation is available at [codyheiser.github.io/cnmf/](https://codyheiser.github.io/cnmf/)

---

## Contributing to the cNMF package

After making changes, lint, format and document code before committing:

```bash
make format  # black-formatting
make lint  # lint Python code
make doc  # pdoc3 documentation
```

Then, following `git commit`, create new version tag and push to remote:

```bash
git tag -a vX.X.X -m "tag message"
git push --follow-tags
```

[tag-version]: https://img.shields.io/github/v/tag/codyheiser/cNMF
[repo-url]: https://github.com/codyheiser/cNMF
