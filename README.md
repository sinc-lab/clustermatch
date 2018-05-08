# clustermatch


## Description

Clustermatch is an efficient clustering method for processing highly diverse data. It can handle
very different data types (such as numerical and categorical), in the presence of linear or
non-linear relationships, also with noise, and without the need of any previous pre-processing.
The article describing the method has been sent for publication.

If you want to quickly test Clustermatch, you can access an online web-demo from
[here](http://sinc.unl.edu.ar/web-demo/clustermatch/).


## Installation

Clustermatch works with Python 3.5 (it should work with version 3.6 too).
The recommended way to install the environment needed is using the [Anaconda](https://anaconda.org/)/[Miniconda](https://conda.io/miniconda.html) distribution.
Once conda is installed, move to the folder where Clustermatch was unpacked and follow this steps:

```bash
$ conda env create -n cm -f environment.yml
$ conda activate cm
```

This will create a conda environment named `cm`. The last step activates the environment.
You can run the test suite to make sure everything works in your system:

```bash
$ python -m unittest discover .
......................................................................

Ran 92 tests in 59.562s

OK
```


## Reproducing results

You can reproduce one of the manuscripts results by running an experiment using an artificial
dataset with several linear and non-linear transformations and see how the method
behave:

```bash
$ export PYTHONPATH=/path/to/clustermatch/
$ python main.py transform_rows_nonlinear03 20 4 1 --n-features 50
Running now:
{
  "data_generator": "Blobs. n_features=50, n_samples=1000, centers=3, cluster_std=0.10, center_box=(-1.0, 1.0)",
  "data_noise": {
    "magnitude": 0.0,
    "percentage_measures": 0.0,
    "percentage_objects": 0.2
  },
  "data_transform": "Nonlinear row transformation 03. 10 simulated data sources; Functions: x^4, log, exp2, 100, log1p, x^5, 10000, log10, 0.0001, log2",
  "n_reps": 1
}
```

The arguments to the `main.py` scripts are: the data transformation function (`transform_rows_nonlinear03`), the noise percentage (`20`), the number of cores
used (`4`) and the number of repetitions (`1`). We are using just `1` repetition and 50 features (`--n-features 50`) so as to speed up the experiment.
If you want to fully run this experiment as it was done in the manuscript (Figure 3), use this comand:

```bash
python main.py transform_rows_nonlinear03 20 4 20
```

Once finished, you will find the output in directory `results_transform_rows_nonlinear03_0.2/{TIMESTAMP}/`:

```bash
$ $ cat results_transform_rows_nonlinear03_0.2/20180508_093733/output000.txt

[...]

method                 ('ari', 'mean')    ('ari', 'std')    ('time', 'mean')
-------------------  -----------------  ----------------  ------------------
00.20. Clustermatch               1.00               nan               26.50
01. SC-Pearson                    0.23               nan                0.38
02. SC-Spearman                   0.29               nan                0.89
03. SC-DC                         1.00               nan               40.90
04. SC-MIC                        1.00               nan               60.62
```

## Usage

You can also try the method by loading a sample of the tomato dataset used in the manuscript. For that,
follow this instructions:

```
$ cd {CLUSTERMATCH_FOLDER}
$ ipython
In [1]: from utils.data import merge_sources
In [2]: from clustermatch.cluster import calculate_simmatrix, get_partition_spectral
In [3]: data_files = ['experiments/tomato_data/sample.xlsx']
In [4]: merged_sources, feature_names, sources_names = merge_sources(data_files)
In [5]: cm_sim_matrix = calculate_simmatrix(merged_sources, n_jobs=4)
In [6]: partition = get_partition_spectral(cm_sim_matrix, 3)
```

The variable `partition` will have the clustering solution for the number of clusters specified (`3` in this case).