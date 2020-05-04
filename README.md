# clustermatch
_Title:_ Clustermatch: discovering hidden relations in highly-diverse kinds of qualitative and quantitative data without standardization  
_Authors:_ Milton Pividori, Andres Cernadas, Luis de Haro, Fernando Carrari, Georgina Stegmayer and Diego H. Milone
Bioinformatics, DOI: https://doi.org/10.1093/bioinformatics/bty899

sinc(i) (Research institute for signals, systems and computational intelligence) - http://sinc.unl.edu.ar

\* Corresponding author: mpividori@sinc.unl.edu.ar

## Description

Clustermatch is an efficient clustering method for processing highly diverse
data. It can handle very different data types (such as numerical and
categorical), in the presence of linear or non-linear relationships, also with
noise, and without the need of any previous pre-processing. The article
describing the method has been sent for publication.

If you want to quickly test Clustermatch, you can access an online web-demo from
[here](http://sinc.unl.edu.ar/web-demo/clustermatch/).

Mirrors:

 * Github: https://github.com/sinc-lab/clustermatch
 * Bitbucket (Mercurial): https://bitbucket.org/sinc-lab/clustermatch

## Installation
You can easily install Clustermatch with pip by running:

```bash
$ pip install clustermatch
```

This will install a command line utility (run `clustermatch -h` for usage instructions)
that it is considered alpha and still under development. Follow the instructions
below if you want to create your own environment and use the Python API to run
Clustermatch.

Clustermatch works with Python 3.6 (it should work with version 3.5 too). You
also need a C compiler (like GCC) to install `minepy` and run the simulations,
although it's not necessary to use Clustermatch. In Ubuntu you can install GCC
by running:

```bash
$ sudo apt-get install build-essential
```

The recommended way to install the Python environment needed is using the
[Anaconda](https://anaconda.org/)/[Miniconda](https://conda.io/miniconda.html)
distribution. Once conda is installed, move to the folder where Clustermatch
was unpacked and follow these steps:

```bash
$ conda env create -n cm -f environment.yml
$ conda activate cm
```

This will create a conda environment named `cm`. The last step activates the
environment. You can run the test suite to make sure everything works in your
system:

```bash
$ python -m unittest discover .
......................................................................

Ran 92 tests in 47.056s

OK
```

Keep in mind that if you want to fully reproduce the results in the manuscript,
then you need to install the full environment using the file
`environment_full.yml`, which has additional dependencies. The one we used
before (`environment.yml`) has the minimum set of packages needed to run
Clustermatch.


## Reproducing results

You can reproduce one of the manuscripts results by running an experiment using
an artificial dataset with several linear and non-linear transformations and
see how the method behave (replace `{CLUSTERMATCH_FOLDER}` with the path
of the Clustermatch folder):

```bash
$ export PYTHONPATH={CLUSTERMATCH_FOLDER}
$ cd {CLUSTERMATCH_FOLDER}/experiments
$ python main.py --data-transf transform_rows_nonlinear03 --noise-perc 45 --n-jobs 4 --n-reps 1 --n-features 50
Running now:
{
  "clustering_algorithm": "spectral",
  "clustering_metric": "ari",
  "data_generator": "Blobs (data_seed_mode=False). n_features=50, n_samples=1000, centers=3, cluster_std=0.10, center_box=(-1.0, 1.0)",
  "data_noise": {
    "magnitude": 0.0,
    "percentage_measures": 0.0,
    "percentage_objects": 0.45
  },
  "data_transform": "Nonlinear row transformation 03. 10 simulated data sources; Functions: x^4, log, exp2, 100, log1p, x^5, 10000, log10, 0.0001, log2",
  "k_final": null,
  "n_reps": 1
}
```

The arguments to the `main.py` scripts are: the data transformation function
(`--data-transf transform_rows_nonlinear03`), the noise percentage (`--noise-perc 45`), the number of
cores used (`--n-jobs 4`) and the number of repetitions (`--n-reps 1`). We are using just `1`
repetition and 50 features (`--n-features 50`) so as to speed up the
experiment. If you want to fully run this experiment as it was done in the
manuscript (Figure 3), use this command (for all noise levels):

```bash
python main.py --data-transf transform_rows_nonlinear03 --noise-perc 45 --n-jobs 4 --n-reps 20
```

Once finished, you will find the output in directory
`results_transform_rows_nonlinear03_0.45/{TIMESTAMP}/`:

```bash
$ cat results_transform_rows_nonlinear03_0.45/20180829_161133/output000.txt

[...]

method              ('metric', 'mean')    ('metric', 'std')    ('time', 'mean')
----------------  --------------------  -------------------  ------------------
00. Clustermatch                  1.00                  nan               31.56
01. SC-Pearson                    0.11                  nan                0.33
02. SC-Spearman                   0.29                  nan                0.67
03. SC-DC                         0.45                  nan               37.19
04. SC-MIC                        0.88                  nan               45.73
```

## Usage

If you installed the command line utility (`clustermatch`), you can run it like this:

```bash
$ cd {CLUSTERMATCH_FOLDER}
$ clustermatch -i experiments/tomato/data/real_sample.xlsx -k 3 -o partition.xls
```

The file `partition.xls` will contain the partition of the data (`real_sample.xlsx`).
Check out the help (`clustermatch -h`) for more options.

You can also try the method by loading a sample of the tomato dataset used in
the manuscript. For that, follow these instructions:

```bash
$ cd {CLUSTERMATCH_FOLDER}
$ ipython
```
```python
In [1]: from clustermatch.utils.data import merge_sources
In [2]: from clustermatch.cluster import calculate_simmatrix, get_partition_spectral
In [3]: data_files = ['experiments/tomato/data/real_sample.xlsx']
In [4]: merged_sources, feature_names, sources_names = merge_sources(data_files)
In [5]: cm_sim_matrix = calculate_simmatrix(merged_sources, n_jobs=4)
In [6]: partition = get_partition_spectral(cm_sim_matrix, 3)
```

The variable `partition` will have the clustering solution for the number of
clusters specified (`3` in this case).  You can specify multiple input data
files by filling the list `data_files`.

Clustermatch is able to process different data types (numerical, ordinal or
categorical) with no previous preprocessing required. The current
implementation considers a variable as categorical if it contains text. The
rest, numerical and ordinal, are processed in a similar way, so you are
responsible for coding your ordinal varibles appropriately (for example,
`low`, `normal` and `high` could be coded as 0, 1, 2; otherwise, if left as text,
will be considered as categorical).
