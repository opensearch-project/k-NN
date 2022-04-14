# OpenSearch Benchmarks for k-NN

## Overview

This directory contains code and configurations to run k-NN benchmarking 
workloads using OpenSearch benchmarks. 

The tenets of these benchmarks:
1. Modularity
2. Reproducibility
3. Extendability

The [helpers](extensions) directory contains common code shared between workloads. Each 
workload folder contains more details about what that particular workload is 
meant to test.

## Getting Started

### Setup

OpenSearch Benchmarks requires Python 3.8 or greater to be installed. One of 
the easier ways to do this is through Conda, a package and environment 
management system for Python.

First, follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
to install Conda on your system.

Next, create a Python 3.8 environment:
```
conda create -n knn-osb python=3.8
```

After the environment is created, activate it:
```
source activate knn-osb
```

Lastly, clone the k-NN repo and install all required python packages:
```
git clone https://github.com/opensearch-project/k-NN.git
cd k-NN/benchmarks/osb
pip install -r requirements.txt
```

After all of this completes, you should be ready to run your first benchmark!

### Running a benchmark

Before running a benchmark, make sure you have the endpoint of your cluster and
 and the machine you are running the benchmarks from can access it. 
 Additionally, ensure that all data has been pulled to the client.

First, go to the directory of the workload that you want to run:
```
cd knn-no-train
```

Next, open up the `params.json` file and choose your configuration.

Lastly, set the URL and PORT of your cluster and run the command. 

```
export URL=
export PORT=

opensearch-benchmark execute_test \ 
    --target-hosts $URL:$PORT \ 
    --workload-path ./workload.json \ 
    --track-params ./params.json \
    --pipeline benchmark-only
```

- Target Hosts - host you want to run the test against
- Workload Path - path to the workload you want to run
- Track Params - path to parameters to configure your workload
- Pipeline - tell OSB to use external cluster

## Adding a benchmark

Adding additional benchmarks is very simple. First, place any custom parameter 
sources or runners in the `helpers` directory so that other tests can use them 
and also update the [documentation](#custom-extensions) accordingly.

Next, create a new directory for your workload and create a `workload.json` 
file, where you define your test, `workload.py`, where you can pull in custom 
extension points (just copy this file from an existing workload), `params.json`, 
where you define the parameters of your test, and a `README.md` file, that 
describes the test and how to run it.

## Custom Extensions

OpenSearch Benchmarks is very extendible. To fit the plugins needs, we add customer parameter sources and custom 
runners. Parameter sources allow users to supply custom parameters to an operation. Runners are what actually performs 
the operations against OpenSearch.

### Custom Parameter Sources

Custom parameter sources are defined in [helpers/param_sources.py](extensions/param_sources.py).

| Name               | Description                                                            | Parameters                                                                                                                                                                                                         |
|--------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bulk-from-data-set | Provides bulk payloads containing vectors from a data set for indexing | 1. data_set_format - (hdf5, bigann)<br/>2. data_set_path - path to data set<br/>3. index - name of index for bulk ingestion<br/> 4. field - field to place vector in <br/> 5. bulk_size - vectors per bulk request |


### Custom Runners

Custom runners are defined in [helpers/runners.py](extensions/param_sources.py).

| Syntax             | Description                                         | Parameters                                                                                   |
|--------------------|-----------------------------------------------------|:---------------------------------------------------------------------------------------------|
| custom-vector-bulk | Bulk index a set of vectors in an OpenSearch index. | 1. bulk-from-data-set                                                                        |
| custom-refresh     | Run refresh with retry capabilities.                | 1. index - name of index to refresh<br/> 2. retries - number of times to retry the operation |


