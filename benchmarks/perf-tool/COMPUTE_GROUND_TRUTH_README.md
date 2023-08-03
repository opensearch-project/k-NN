# Compute Ground Truth
- [Welcome!](#welcome)
- [Install Prerequisites](#install-prerequisites)
- [Usage](#usage)

## Welcome!

This readme provides details how you can run the ground truth scripts to generate the ground truth for any dataset. The scripts are still work in progress, feel free to contribute.

## Install Prerequisites
### Setup

Requires Python 3.8 or greater to be installed. One of
the easier ways to do this is through Conda, a package and environment
management system for Python.

First, follow the
[installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
to install Conda on your system.

Next, create a Python 3.8 environment:
```
conda create -n compute-truth python=3.8
```

After the environment is created, activate it:
```
source activate compute-truth
```

```
pip install -r requirements.txt
```

Install the required packages

```
sudo yum install lapack
sudo yum install blas 
```

## Usage

### Quick Start
In order to compute the ground truth you first need is a dataset in hdf5 file format. The code assumes that dataset file will be 
a standard hdf5 file which will have `train` and `test` datasets in it, which represents the corpus and the queries for which
ground truth needs to be generated.

To generate the ground truth script you can run the below command

```
python compute-ground-truth.py --input_file <FULL-PATH-OF-HDF5-DATASET-FILE> --output_file <FULL-PATH-OF-OUTPUT-HDF5-DATASET-FILE> --corpus_size <INTEGER> --query_client <NUMBER_OF_PARALLEL_CLIENTS_NEEDED_TO_RUN_THE_QUERIES>
```

### Parameters

| Parameter Name        | Description                                                                                                        | Default                               |   
|-----------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| input_file            | Full path of the file containing the dataset.                                                                      | No Default                            |
| output_file           | Full path of the file and file name where ground truth needs to be generated. File will be created if not present. | No Default                            |
| corpus_size(Optional) | Total count of vectors input vectors that are indexed.                                                             | Obtained from dataset if not provided |
| query_client(Optional)| Number of clients/python processes needs to be used for for running all the queries.                               | 2                                     | 

### Recommendation/Things to keep in mind
1. As we are doing brute force Nearest Neighbors search which is time consuming, it is recommended to use a machine which has more cores and higher value of query_client, so more and more queries can be parallelized.
2. You can use `htop` command to view how many cores are getting used during the computation. Example if you have 96 cores in the machine and 20k queries atleast use `20 query_client` so that results can be computed quickly.(Read next point for more details)
3. The code use `numpy.dot` functions for calculating distance which use Blas internally to parallelize distance calculation. So even with smaller values of `query_client` you will start to see `100% CPU utilization`. But as those operations are short-lived we can bump up the `query_client` value.
