# OpenSearch Benchmarks for k-NN

## Overview

This directory contains code and configurations to run k-NN benchmarking 
workloads using OpenSearch Benchmarks.

The [extensions](extensions) directory contains common code shared between 
procedures. The [procedures](procedures) directory contains the individual 
test procedures for this workload.

## Getting Started

### OpenSearch Benchmarks Background

OpenSearch Benchmark is a framework for performance benchmarking an OpenSearch 
cluster. For more details, checkout their 
[repo](https://github.com/opensearch-project/opensearch-benchmark/). 

Before getting into the benchmarks, it is helpful to know a few terms:
1. Workload - Top level description of a benchmark suite. A workload will have a `workload.json` file that defines different components of the tests 
2. Test Procedures - A workload can have a schedule of operations that run the test. However, a workload can also have several test procedures that define their own schedule of operations. This is helpful for sharing code between tests
3. Operation - An action against the OpenSearch cluster
4. Parameter source - Producers of parameters for OpenSearch operations
5. Runners - Code that actually will execute the OpenSearch operations

### Setup

OpenSearch Benchmarks requires Python 3.8 or greater to be installed. One of 
the easier ways to do this is through Conda, a package and environment 
management system for Python.

First, follow the 
[installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
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
  the machine you are running the benchmarks from can access it. 
 Additionally, ensure that all data has been pulled to the client.

Currently, we support 2 test procedures for the k-NN workload: train-test and 
no-train-test. The train test has steps to train a model included in the 
schedule, while no train does not. Both test procedures will index a data set 
of vectors into an OpenSearch index and then run a set of queries against them. 

Once you have decided which test procedure you want to use, open up 
[params/train-params.json](params/train-params.json) or 
[params/no-train-params.json](params/no-train-params.json) and 
fill out the parameters. Notice, at the bottom of `no-train-params.json` there 
are several parameters that relate to training. Ignore these. They need to be 
defined for the workload but not used. 

Once the parameters are set, set the URL and PORT of your cluster and run the 
command to run the test procedure. 

```
export URL=
export PORT=
export PARAMS_FILE=
export PROCEDURE={no-train-test | train-test}

opensearch-benchmark execute_test \ 
    --target-hosts $URL:$PORT \ 
    --workload-path ./workload.json \ 
    --workload-params ${PARAMS_FILE} \
    --test-procedure=${PROCEDURE} \
    --pipeline benchmark-only
```

## Current Procedures

### No Train Test

The No Train Test procedure is used to test `knn_vector` indices that do not 
use an algorithm that requires training.

#### Workflow

1. Delete old resources in the cluster if they are present
2. Create an OpenSearch index with `knn_vector` configured to use the HNSW algorithm
3. Wait for cluster to be green
4. Ingest data set into the cluster
5. Refresh the index
6. Run queries from data set against the cluster

#### Parameters

| Name                                    | Description                                                                      |
|-----------------------------------------|----------------------------------------------------------------------------------|
| target_index_name                       | Name of index to add vectors to                                                  |
| target_field_name                       | Name of field to add vectors to                                                  |
| target_index_body                       | Path to target index definition                                                  |
| target_index_primary_shards             | Target index primary shards                                                      |
| target_index_replica_shards             | Target index replica shards                                                      |
| target_index_dimension                  | Dimension of target index                                                        |
| target_index_space_type                 | Target index space type                                                          |
| target_index_bulk_size                  | Target index bulk size                                                           |
| target_index_bulk_index_data_set_format | Format of vector data set                                                        |
| target_index_bulk_index_data_set_path   | Path to vector data set                                                          |
| target_index_bulk_index_clients         | Clients to be used for bulk ingestion (must be divisor of data set size)         |
| hnsw_ef_search                          | HNSW ef search parameter                                                         |
| hnsw_ef_construction                    | HNSW ef construction parameter                                                   |
| hnsw_m                                  | HNSW m parameter                                                                 |
| query_k                                 | The number of neighbors to return for the search                                 |
| query_clients                           | Number of clients to use for running queries                                     |
| query_data_set_format                   | Format of vector data set for queries                                            |
| query_data_set_path                     | Path to vector data set for queries                                              |

#### Metrics

The result metrics of this procedure will look like: 
```
------------------------------------------------------
    _______             __   _____
   / ____(_)___  ____ _/ /  / ___/_________  ________
  / /_  / / __ \/ __ `/ /   \__ \/ ___/ __ \/ ___/ _ \
 / __/ / / / / / /_/ / /   ___/ / /__/ /_/ / /  /  __/
/_/   /_/_/ /_/\__,_/_/   /____/\___/\____/_/   \___/
------------------------------------------------------

|                                                         Metric |                    Task |       Value |   Unit |
|---------------------------------------------------------------:|------------------------:|------------:|-------:|
|                     Cumulative indexing time of primary shards |                         |  0.00173333 |    min |
|             Min cumulative indexing time across primary shards |                         |           0 |    min |
|          Median cumulative indexing time across primary shards |                         |           0 |    min |
|             Max cumulative indexing time across primary shards |                         | 0.000616667 |    min |
|            Cumulative indexing throttle time of primary shards |                         |           0 |    min |
|    Min cumulative indexing throttle time across primary shards |                         |           0 |    min |
| Median cumulative indexing throttle time across primary shards |                         |           0 |    min |
|    Max cumulative indexing throttle time across primary shards |                         |           0 |    min |
|                        Cumulative merge time of primary shards |                         |           0 |    min |
|                       Cumulative merge count of primary shards |                         |           0 |        |
|                Min cumulative merge time across primary shards |                         |           0 |    min |
|             Median cumulative merge time across primary shards |                         |           0 |    min |
|                Max cumulative merge time across primary shards |                         |           0 |    min |
|               Cumulative merge throttle time of primary shards |                         |           0 |    min |
|       Min cumulative merge throttle time across primary shards |                         |           0 |    min |
|    Median cumulative merge throttle time across primary shards |                         |           0 |    min |
|       Max cumulative merge throttle time across primary shards |                         |           0 |    min |
|                      Cumulative refresh time of primary shards |                         |  0.00271667 |    min |
|                     Cumulative refresh count of primary shards |                         |         115 |        |
|              Min cumulative refresh time across primary shards |                         |           0 |    min |
|           Median cumulative refresh time across primary shards |                         |           0 |    min |
|              Max cumulative refresh time across primary shards |                         |     0.00135 |    min |
|                        Cumulative flush time of primary shards |                         |           0 |    min |
|                       Cumulative flush count of primary shards |                         |          43 |        |
|                Min cumulative flush time across primary shards |                         |           0 |    min |
|             Median cumulative flush time across primary shards |                         |           0 |    min |
|                Max cumulative flush time across primary shards |                         |           0 |    min |
|                                        Total Young Gen GC time |                         |       0.849 |      s |
|                                       Total Young Gen GC count |                         |          20 |        |
|                                          Total Old Gen GC time |                         |           0 |      s |
|                                         Total Old Gen GC count |                         |           0 |        |
|                                                     Store size |                         |    0.647921 |     GB |
|                                                  Translog size |                         |  0.00247511 |     GB |
|                                         Heap used for segments |                         |    0.284451 |     MB |
|                                       Heap used for doc values |                         |   0.0872688 |     MB |
|                                            Heap used for terms |                         |   0.0714417 |     MB |
|                                            Heap used for norms |                         | 6.10352e-05 |     MB |
|                                           Heap used for points |                         |           0 |     MB |
|                                    Heap used for stored fields |                         |    0.125679 |     MB |
|                                                  Segment count |                         |         257 |        |
|                                                 Min Throughput |      custom-vector-bulk |     18018.5 | docs/s |
|                                                Mean Throughput |      custom-vector-bulk |     18018.5 | docs/s |
|                                              Median Throughput |      custom-vector-bulk |     18018.5 | docs/s |
|                                                 Max Throughput |      custom-vector-bulk |     18018.5 | docs/s |
|                                        50th percentile latency |      custom-vector-bulk |     98.5565 |     ms |
|                                        90th percentile latency |      custom-vector-bulk |     100.033 |     ms |
|                                       100th percentile latency |      custom-vector-bulk |     103.792 |     ms |
|                                   50th percentile service time |      custom-vector-bulk |     98.5565 |     ms |
|                                   90th percentile service time |      custom-vector-bulk |     100.033 |     ms |
|                                  100th percentile service time |      custom-vector-bulk |     103.792 |     ms |
|                                                     error rate |      custom-vector-bulk |           0 |      % |
|                                                 Min Throughput |    refresh-target-index |       76.22 |  ops/s |
|                                                Mean Throughput |    refresh-target-index |       76.22 |  ops/s |
|                                              Median Throughput |    refresh-target-index |       76.22 |  ops/s |
|                                                 Max Throughput |    refresh-target-index |       76.22 |  ops/s |
|                                       100th percentile latency |    refresh-target-index |     12.7619 |     ms |
|                                  100th percentile service time |    refresh-target-index |     12.7619 |     ms |
|                                                     error rate |    refresh-target-index |           0 |      % |
|                                                 Min Throughput | knn-query-from-data-set |     1587.47 |  ops/s |
|                                                Mean Throughput | knn-query-from-data-set |     1649.97 |  ops/s |
|                                              Median Throughput | knn-query-from-data-set |     1661.79 |  ops/s |
|                                                 Max Throughput | knn-query-from-data-set |     1677.06 |  ops/s |
|                                        50th percentile latency | knn-query-from-data-set |     4.79125 |     ms |
|                                        90th percentile latency | knn-query-from-data-set |        5.38 |     ms |
|                                        99th percentile latency | knn-query-from-data-set |     46.8965 |     ms |
|                                      99.9th percentile latency | knn-query-from-data-set |     58.2049 |     ms |
|                                     99.99th percentile latency | knn-query-from-data-set |     59.6476 |     ms |
|                                       100th percentile latency | knn-query-from-data-set |     60.9245 |     ms |
|                                   50th percentile service time | knn-query-from-data-set |     4.79125 |     ms |
|                                   90th percentile service time | knn-query-from-data-set |        5.38 |     ms |
|                                   99th percentile service time | knn-query-from-data-set |     46.8965 |     ms |
|                                 99.9th percentile service time | knn-query-from-data-set |     58.2049 |     ms |
|                                99.99th percentile service time | knn-query-from-data-set |     59.6476 |     ms |
|                                  100th percentile service time | knn-query-from-data-set |     60.9245 |     ms |
|                                                     error rate | knn-query-from-data-set |           0 |      % |


--------------------------------
[INFO] SUCCESS (took 46 seconds)
--------------------------------
```

### Train Test

The Train Test procedure is used to test `knn_vector` indices that do use an 
algorithm that requires training.

#### Workflow

1. Delete old resources in the cluster if they are present
2. Create an OpenSearch index with `knn_vector` configured to load with training data
3. Wait for cluster to be green
4. Ingest data set into the training index
5. Refresh the index
6. Train a model based on user provided input parameters
7. Create an OpenSearch index with `knn_vector` configured to use the model
8. Ingest vectors into the target index
9. Refresh the target index
10. Run queries from data set against the cluster

#### Parameters

| Name                                    | Description                                                                      |
|-----------------------------------------|----------------------------------------------------------------------------------|
| target_index_name                       | Name of index to add vectors to                                                  |
| target_field_name                       | Name of field to add vectors to                                                  |
| target_index_body                       | Path to target index definition                                                  |
| target_index_primary_shards             | Target index primary shards                                                      |
| target_index_replica_shards             | Target index replica shards                                                      |
| target_index_dimension                  | Dimension of target index                                                        |
| target_index_space_type                 | Target index space type                                                          |
| target_index_bulk_size                  | Target index bulk size                                                           |
| target_index_bulk_index_data_set_format | Format of vector data set for ingestion                                          |
| target_index_bulk_index_data_set_path   | Path to vector data set for ingestion                                            |
| target_index_bulk_index_clients         | Clients to be used for bulk ingestion (must be divisor of data set size)         |
| ivf_nlists                              | IVF nlist parameter                                                              |
| ivf_nprobes                             | IVF nprobe parameter                                                             |
| pq_code_size                            | PQ code_size parameter                                                           |
| pq_m                                    | PQ m parameter                                                                   |
| train_model_method                      | Method to be used for model (ivf or ivfpq)                                       |
| train_model_id                          | Model ID                                                                         |
| train_index_name                        | Name of index to put training data into                                          |
| train_field_name                        | Name of field to put training data into                                          |
| train_index_body                        | Path to train index definition                                                   |
| train_search_size                       | Search size to use when pulling training data                                    |
| train_timeout                           | Timeout to wait for training to finish                                           |
| train_index_primary_shards              | Train index primary shards                                                       |
| train_index_replica_shards              | Train index replica shards                                                       |
| train_index_bulk_size                   | Train index bulk size                                                            |
| train_index_data_set_format             | Format of vector data set for training                                           |
| train_index_data_set_path               | Path to vector data set for training                                             |
| train_index_num_vectors                 | Number of vectors to use from vector data set for training                       |
| train_index_bulk_index_clients          | Clients to be used for bulk ingestion (must be divisor of data set size)         |
| query_k                                 | The number of neighbors to return for the search                                 |
| query_clients                           | Number of clients to use for running queries                                     |
| query_data_set_format                   | Format of vector data set for queries                                            |
| query_data_set_path                     | Path to vector data set for queries                                              |

#### Metrics

The result metrics of this procedure will look like: 
```
------------------------------------------------------
    _______             __   _____
   / ____(_)___  ____ _/ /  / ___/_________  ________
  / /_  / / __ \/ __ `/ /   \__ \/ ___/ __ \/ ___/ _ \                                                                                                                                                                                                                                           [63/1855]
 / __/ / / / / / /_/ / /   ___/ / /__/ /_/ / /  /  __/
/_/   /_/_/ /_/\__,_/_/   /____/\___/\____/_/   \___/
------------------------------------------------------
|                                                         Metric |                    Task |       Value |             Unit |
|---------------------------------------------------------------:|------------------------:|------------:|-----------------:|
|                     Cumulative indexing time of primary shards |                         |     2.92355 |              min |
|             Min cumulative indexing time across primary shards |                         |           0 |              min |
|          Median cumulative indexing time across primary shards |                         |    0.497817 |              min |
|             Max cumulative indexing time across primary shards |                         |     1.37717 |              min |
|            Cumulative indexing throttle time of primary shards |                         |           0 |              min |
|    Min cumulative indexing throttle time across primary shards |                         |           0 |              min |
| Median cumulative indexing throttle time across primary shards |                         |           0 |              min |
|    Max cumulative indexing throttle time across primary shards |                         |           0 |              min |
|                        Cumulative merge time of primary shards |                         |     1.34895 |              min |
|                       Cumulative merge count of primary shards |                         |          39 |                  |
|                Min cumulative merge time across primary shards |                         |           0 |              min |
|             Median cumulative merge time across primary shards |                         |    0.292033 |              min |
|                Max cumulative merge time across primary shards |                         |      0.6268 |              min |
|               Cumulative merge throttle time of primary shards |                         |     0.62845 |              min |
|       Min cumulative merge throttle time across primary shards |                         |           0 |              min |
|    Median cumulative merge throttle time across primary shards |                         |    0.155617 |              min |
|       Max cumulative merge throttle time across primary shards |                         |    0.290117 |              min |
|                      Cumulative refresh time of primary shards |                         |    0.369433 |              min |
|                     Cumulative refresh count of primary shards |                         |          96 |                  |
|              Min cumulative refresh time across primary shards |                         |           0 |              min |
|           Median cumulative refresh time across primary shards |                         |   0.0903833 |              min |
|              Max cumulative refresh time across primary shards |                         |     0.10365 |              min |
|                        Cumulative flush time of primary shards |                         |   0.0278667 |              min |
|                       Cumulative flush count of primary shards |                         |           2 |                  |
|                Min cumulative flush time across primary shards |                         |           0 |              min |
|             Median cumulative flush time across primary shards |                         |           0 |              min |
|                Max cumulative flush time across primary shards |                         |   0.0278667 |              min |
|                                        Total Young Gen GC time |                         |      13.106 |                s |
|                                       Total Young Gen GC count |                         |         263 |                  |
|                                          Total Old Gen GC time |                         |           0 |                s |
|                                         Total Old Gen GC count |                         |           0 |                  |
|                                                     Store size |                         |     2.60183 |               GB |
|                                                  Translog size |                         |     1.34787 |               GB |
|                                         Heap used for segments |                         |   0.0646248 |               MB |
|                                       Heap used for doc values |                         |  0.00899887 |               MB |
|                                            Heap used for terms |                         |   0.0203552 |               MB |
|                                            Heap used for norms |                         | 6.10352e-05 |               MB |
|                                           Heap used for points |                         |           0 |               MB |
|                                    Heap used for stored fields |                         |   0.0352097 |               MB |
|                                                  Segment count |                         |          71 |                  |
|                                                 Min Throughput |            delete-model |       10.55 |            ops/s |
|                                                Mean Throughput |            delete-model |       10.55 |            ops/s |
|                                              Median Throughput |            delete-model |       10.55 |            ops/s |
|                                                 Max Throughput |            delete-model |       10.55 |            ops/s |
|                                       100th percentile latency |            delete-model |     94.4726 |               ms |
|                                  100th percentile service time |            delete-model |     94.4726 |               ms |
|                                                     error rate |            delete-model |           0 |                % |
|                                                 Min Throughput |       train-vector-bulk |     44763.1 |           docs/s |
|                                                Mean Throughput |       train-vector-bulk |     52022.4 |           docs/s |
|                                              Median Throughput |       train-vector-bulk |     52564.8 |           docs/s |
|                                                 Max Throughput |       train-vector-bulk |       53833 |           docs/s |
|                                        50th percentile latency |       train-vector-bulk |     22.3364 |               ms |
|                                        90th percentile latency |       train-vector-bulk |      47.799 |               ms |
|                                        99th percentile latency |       train-vector-bulk |     195.954 |               ms |
|                                      99.9th percentile latency |       train-vector-bulk |     495.217 |               ms |
|                                       100th percentile latency |       train-vector-bulk |      663.48 |               ms |
|                                   50th percentile service time |       train-vector-bulk |     22.3364 |               ms |
|                                   90th percentile service time |       train-vector-bulk |      47.799 |               ms |
|                                   99th percentile service time |       train-vector-bulk |     195.954 |               ms |
|                                 99.9th percentile service time |       train-vector-bulk |     495.217 |               ms |
|                                  100th percentile service time |       train-vector-bulk |      663.48 |               ms |
|                                                     error rate |       train-vector-bulk |           0 |                % |
|                                                 Min Throughput |     refresh-train-index |        0.98 |            ops/s |
|                                                Mean Throughput |     refresh-train-index |        0.98 |            ops/s |
|                                              Median Throughput |     refresh-train-index |        0.98 |            ops/s |
|                                                 Max Throughput |     refresh-train-index |        0.98 |            ops/s |
|                                       100th percentile latency |     refresh-train-index |     1019.54 |               ms |
|                                  100th percentile service time |     refresh-train-index |     1019.54 |               ms |
|                                                     error rate |     refresh-train-index |           0 |                % |
|                                                 Min Throughput |       ivfpq-train-model |        0.01 | models_trained/s |
|                                                Mean Throughput |       ivfpq-train-model |        0.01 | models_trained/s |
|                                              Median Throughput |       ivfpq-train-model |        0.01 | models_trained/s |
|                                                 Max Throughput |       ivfpq-train-model |        0.01 | models_trained/s |
|                                       100th percentile latency |       ivfpq-train-model |      150952 |               ms |
|                                  100th percentile service time |       ivfpq-train-model |      150952 |               ms |
|                                                     error rate |       ivfpq-train-model |           0 |                % |
|                                                 Min Throughput |      custom-vector-bulk |     32367.4 |           docs/s |
|                                                Mean Throughput |      custom-vector-bulk |     36027.5 |           docs/s |
|                                              Median Throughput |      custom-vector-bulk |     35276.7 |           docs/s |
|                                                 Max Throughput |      custom-vector-bulk |       41095 |           docs/s |
|                                        50th percentile latency |      custom-vector-bulk |     22.2419 |               ms |
|                                        90th percentile latency |      custom-vector-bulk |      70.163 |               ms |
|                                        99th percentile latency |      custom-vector-bulk |     308.395 |               ms |
|                                      99.9th percentile latency |      custom-vector-bulk |     548.558 |               ms |
|                                       100th percentile latency |      custom-vector-bulk |     655.628 |               ms |
|                                   50th percentile service time |      custom-vector-bulk |     22.2419 |               ms |
|                                   90th percentile service time |      custom-vector-bulk |      70.163 |               ms |
|                                   99th percentile service time |      custom-vector-bulk |     308.395 |               ms |
|                                 99.9th percentile service time |      custom-vector-bulk |     548.558 |               ms |
|                                  100th percentile service time |      custom-vector-bulk |     655.628 |               ms |
|                                                     error rate |      custom-vector-bulk |           0 |                % |
|                                                 Min Throughput |    refresh-target-index |        0.23 |            ops/s |
|                                                Mean Throughput |    refresh-target-index |        0.23 |            ops/s |
|                                              Median Throughput |    refresh-target-index |        0.23 |            ops/s |
|                                                 Max Throughput |    refresh-target-index |        0.23 |            ops/s |
|                                       100th percentile latency |    refresh-target-index |     4331.17 |               ms |
|                                  100th percentile service time |    refresh-target-index |     4331.17 |               ms |
|                                                     error rate |    refresh-target-index |           0 |                % |
|                                                 Min Throughput | knn-query-from-data-set |      455.19 |            ops/s |
|                                                Mean Throughput | knn-query-from-data-set |      511.74 |            ops/s |
|                                              Median Throughput | knn-query-from-data-set |      510.85 |            ops/s |
|                                                 Max Throughput | knn-query-from-data-set |      570.07 |            ops/s |
|                                        50th percentile latency | knn-query-from-data-set |     14.1626 |               ms |
|                                        90th percentile latency | knn-query-from-data-set |     30.2389 |               ms |
|                                        99th percentile latency | knn-query-from-data-set |     71.2793 |               ms |
|                                      99.9th percentile latency | knn-query-from-data-set |     104.733 |               ms |
|                                     99.99th percentile latency | knn-query-from-data-set |     127.298 |               ms |
|                                       100th percentile latency | knn-query-from-data-set |     145.229 |               ms |
|                                   50th percentile service time | knn-query-from-data-set |     14.1626 |               ms |
|                                   90th percentile service time | knn-query-from-data-set |     30.2389 |               ms |
|                                   99th percentile service time | knn-query-from-data-set |     71.2793 |               ms |
|                                 99.9th percentile service time | knn-query-from-data-set |     104.733 |               ms |
|                                99.99th percentile service time | knn-query-from-data-set |     127.298 |               ms |
|                                  100th percentile service time | knn-query-from-data-set |     145.229 |               ms |
|                                                     error rate | knn-query-from-data-set |           0 |                % |


---------------------------------
[INFO] SUCCESS (took 295 seconds)
---------------------------------
```

## Adding a procedure

Adding additional benchmarks is very simple. First, place any custom parameter 
sources or runners in the [extensions](extensions) directory so that other tests 
can use them and also update the [documentation](#custom-extensions) 
accordingly.

Next, create a new test procedure file and add the operations you want your test 
to run. Lastly, be sure to update documentation.

## Custom Extensions

OpenSearch Benchmarks is very extendable. To fit the plugins needs, we add 
customer parameter sources and custom runners. Parameter sources allow users to 
supply custom parameters to an operation. Runners are what actually performs 
the operations against OpenSearch.

### Custom Parameter Sources

Custom parameter sources are defined in [extensions/param_sources.py](extensions/param_sources.py).

| Name                    | Description                                                            | Parameters                                                                                                                                                                                                                                                                                                                                                |
|-------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bulk-from-data-set      | Provides bulk payloads containing vectors from a data set for indexing | 1. data_set_format - (hdf5, bigann)<br/>2. data_set_path - path to data set<br/>3. index - name of index for bulk ingestion<br/> 4. field - field to place vector in <br/> 5. bulk_size - vectors per bulk request<br/> 6. num_vectors - number of vectors to use from the data set. Defaults to the whole data set.                                      |
| knn-query-from-data-set | Provides a query generated from a data set                             | 1. data_set_format - (hdf5, bigann)<br/>2. data_set_path - path to data set<br/>3. index - name of index to query against<br/>4. field - field to to query against<br/>5. k - number of results to return<br/>6. dimension - size of vectors to produce<br/> 7. num_vectors - number of vectors to use from the data set. Defaults to the whole data set. |


### Custom Runners

Custom runners are defined in [extensions/runners.py](extensions/runners.py).

| Syntax             | Description                                         | Parameters                                                                                                   |
|--------------------|-----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| custom-vector-bulk | Bulk index a set of vectors in an OpenSearch index. | 1. bulk-from-data-set                                                                                        |
| custom-refresh     | Run refresh with retry capabilities.                | 1. index - name of index to refresh<br/> 2. retries - number of times to retry the operation                 |
| train-model        | Trains a model.                                     | 1. body - model definition<br/> 2. timeout - time to wait for model to finish<br/> 3. model_id - ID of model |
| delete-model       | Deletes a model if it exists.                       | 1. model_id - ID of model                                                                                    |

### Testing

We have a set of unit tests for our extensions in 
[tests](tests). To run all the tests, run the following 
command:

```commandline
python -m unittest discover ./tests
```

To run an individual test:
```commandline
python -m unittest tests.test_param_sources.VectorsFromDataSetParamSourceTestCase.test_partition_hdf5
```
