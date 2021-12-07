# OpenSearch k-NN Benchmarking
- [Welcome!](#welcome)
- [Usage](#quick-start)

## Welcome!

This directory contains the code related to benchmarking the k-NN plugin. 
Benchmarks can be run against any OpenSearch cluster with the k-NN plugin 
installed. Benchmarks are highly configurable using the test configuration 
file.

## Install Prerequisites

### Python

Python 3.7 or above is required.

### Pip

Use pip to install the necessary requirements:

```
pip install -r requirements.txt
```

## Usage

### Quick Start

In order to run a benchmark, you must first create a test configuration yml 
file. Checkout [this example](https://github.com/opensearch-project/k-NN) file 
for benchmarking *faiss*'s IVF method. This file contains the definition for 
the benchmark that you want to run. At the top are [test parameters](LINK_ME). 
These define high level settings of the test, such as the endpoint of the 
OpenSearch cluster. 

Next, you define the actions that the test will perform. These actions are 
referred to as steps. First, you can define "setup" steps. These are steps that 
are run once at the beginning of the execution to configure the cluster how you 
want it. These steps do not contribute to the final metrics.

After that, you define the "steps". These are the steps that the test will be 
collecting metrics on. Each step emits certain metrics. These are run 
multiple times, depending on the test parameter "num_runs". At the end of the 
execution of all of the runs, the metrics from each run are collected and 
averaged.

Lastly, you define the "cleanup" steps. The "cleanup" steps are executed after 
each test run. For instance, if you are measuring index performance, you may 
want to delete the index after each run.

To run the test, execute the following command:
```
python knn-perf-tool.py [--log LOGLEVEL] test config-path.yml output.json

--log       log level of tool, options are: info, debug, warning, error, critical
```

The output will be a json document containing the results.

Additionally, you can get the difference between two test runs using the diff 
command:
```
python knn-perf-tool.py [--log LOGLEVEL] diff result1.json result2.json

--log       log level of tool, options are: info, debug, warning, error, critical
```

The output will be the delta between the two metrics.

### Test Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| endpoint | Endpoint OpenSearch cluster is running on | localhost |
| test_name | Name of test | No default |
| test_id | String ID of test | No default |
| num_runs | Number of runs to execute steps | 1 |
| show_runs | Whether to output each run in addition to the total summary | false |
| setup | List of steps to run once before metric collection starts | [] |
| steps | List of steps that make up one test run. Metrics will be collected on these steps. | No default |
| cleanup | List of steps to run after each test run | [] |

### Steps

Included are the list of steps that are currently supported. Each step contains 
a set of parameters that are passed in the test configuration file and a set 
of metrics that the test produces. 

#### create_index

Creates an OpenSearch index.

##### Parameters
| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| index_name | Name of index to create | No default |
| index_spec | Path to index specification | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end. | ms |

#### disable_refresh

Disables refresh for all indices in the cluster.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end. | ms |

#### refresh_index

Refreshes an OpenSearch index.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| index_name | Name of index to refresh | No default |
| store_kb | Size of index after step completes | KB |

##### Metrics

#### force_merge

Force merges an index to a specified number of segments.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| index_name | Name of index to force merge | No default |
| max_num_segments | Number of segments to force merge to | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end. | ms |

#### train_model

Trains a model.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| model_id | Model id to set | Test |
| train_index | Index to pull training data from | No default |
| train_field | Field to pull training data from | No default |
| dimension | Dimension of model | No default |
| description | Description of model | No default |
| max_training_vector_count | Number of training vectors to used | No default |
| method_spec | Path to method specification | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end | ms |

#### delete_model

Deletes a model from the cluster.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| model_id | Model id to delete | Test |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end | ms |

#### delete_index

Deletes an index from the cluster.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| index_name | Name of index to delete | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end | ms |

#### ingest

Ingests a dataset of vectors into the cluster.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| index_name | Name of index to ingest into | No default |
| field_name | Name of field to ingest into | No default |
| bulk_size | Documents per bulk request | 300 |
| dataset_format | Format the dataset is in. Currently only hdf5 is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs. This will use the "train" data as the data to ingest. | 'hdf5' |
| dataset_path | Path to dataset | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Took times returned per bulk request aggregated as total, p50, p90 and p99 (when applicable). Note - this number does not mean the time it took to made the vectors searchable. Time from the refresh step should be looked at as well to determine this.| ms |

#### query

Runs a set of queries against an index.

##### Parameters

| Parameter Name | Description | Default |  
| ----------- | ----------- | ----------- |
| k | Number of neighbors to return on search | 100 |
| r | r value in Recall@R  | 1 |
| index_name | Name of index to search | No default |
| field_name | Name field to search | No default |
| calculate_recall | Whether to calculate recall values | False |
| dataset_format | Format the dataset is in. Currently only hdf5 is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs. This will use the "test" data as the data to use for queries. | 'hdf5' |
| dataset_path | Path to dataset | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Took times returned per query aggregated as total, p50, p90 and p99 (when applicable) | ms |
| memory_kb | Native memory k-NN is using at the end of the query workload | KB |
| recall@R | ratio of when top R ground truth results were returned in K results | float 0.0-1.0 |
| recall@K | ratio of results returned that were ground truth nearest neighbors  | float 0.0-1.0 |

## Contributing 

### Linting

Use pylint to lint the code:
```
pylint knn-perf-tool.py okpt/**/*.py okpt/**/**/*.py
```

### Formatting

We use yapf and the google style to format our code. After installing yapf, you can format your code by running:

```
yapf --style google knn-perf-tool.py okpt/**/*.py okpt/**/**/*.py
```

### Updating requirements

Add new requirements to "requirements.in" and run `pip-compile`
