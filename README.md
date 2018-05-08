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


## Usage

```bash

```