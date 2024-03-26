# IMPORTANT NOTE: No new features will be added to this tool . This tool is currently in maintanence mode. All new features will be added to [vector search workload]( https://github.com/opensearch-project/opensearch-benchmark-workloads/tree/main/vectorsearch)

# OpenSearch k-NN Benchmarking
- [Welcome!](#welcome)
- [Install Prerequisites](#install-prerequisites)
- [Usage](#usage)
- [Contributing](#contributing)

## Welcome!

This directory contains the code related to benchmarking the k-NN plugin. 
Benchmarks can be run against any OpenSearch cluster with the k-NN plugin 
installed. Benchmarks are highly configurable using the test configuration 
file.

## Install Prerequisites

### Setup

K-NN perf requires Python 3.8 or greater to be installed. One of 
the easier ways to do this is through Conda, a package and environment 
management system for Python.

First, follow the 
[installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
to install Conda on your system.

Next, create a Python 3.8 environment:
```
conda create -n knn-perf python=3.8
```

After the environment is created, activate it:
```
source activate knn-perf
```

Lastly, clone the k-NN repo and install all required python packages:
```
git clone https://github.com/opensearch-project/k-NN.git
cd k-NN/benchmarks/perf-tool
pip install -r requirements.txt
```

After all of this completes, you should be ready to run your first performance benchmarks!


## Usage

### Quick Start

In order to run a benchmark, you must first create a test configuration yml 
file. Checkout [this example](https://github.com/opensearch-project/k-NN/blob/main/benchmarks/perf-tool/sample-configs) file 
for benchmarking *faiss*'s IVF method. This file contains the definition for 
the benchmark that you want to run. At the top are 
[test parameters](#test-parameters). These define high level settings of the 
test, such as the endpoint of the OpenSearch cluster. 

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

| Parameter Name | Description                                                                        | Default    |  
|----------------|------------------------------------------------------------------------------------|------------|
| endpoint       | Endpoint OpenSearch cluster is running on                                          | localhost  |
| port           | Port on which OpenSearch Cluster is running on                                     | 9200       |
| test_name      | Name of test                                                                       | No default |
| test_id        | String ID of test                                                                  | No default |
| num_runs       | Number of runs to execute steps                                                    | 1          |
| show_runs      | Whether to output each run in addition to the total summary                        | false      |
| setup          | List of steps to run once before metric collection starts                          | []         |
| steps          | List of steps that make up one test run. Metrics will be collected on these steps. | No default |
| cleanup        | List of steps to run after each test run                                           | []         |

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

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Time to execute step end to end. | ms |
| store_kb | Size of index after refresh completes | KB |

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
| dataset_format | Format the data-set is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs. | 'hdf5' |
| dataset_path | Path to data-set | No default |
| doc_count | Number of documents to create from data-set | Size of the data-set |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Total time to ingest the dataset into the index.| ms |

#### ingest_multi_field

Ingests a dataset of multiple context types into the cluster.

##### Parameters

| Parameter Name | Description                                                                                                                                               | Default |  
| ----------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- |
| index_name | Name of index to ingest into                                                                                                                              | No default |
| field_name | Name of field to ingest into                                                                                                                              | No default |
| bulk_size | Documents per bulk request                                                                                                                                | 300 |
| dataset_path | Path to data-set                                                                                                                                          | No default |
| doc_count | Number of documents to create from data-set                                                                                                               | Size of the data-set |
| attributes_dataset_name | Name of dataset with additional attributes inside the main dataset                                                                                        | No default |
| attribute_spec | Definition of attributes, format is: [{ name: [name_val], type: [type_val]}] Order is important and must match order of attributes column in dataset file | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Total time to ingest the dataset into the index.| ms |

#### ingest_nested_field

Ingests a dataset with nested field into the cluster.

##### Parameters

| Parameter Name | Description                                                                                                                                                                                                      | Default |  
| ----------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- |
| index_name | Name of index to ingest into                                                                                                                                                                                     | No default |
| field_name | Name of field to ingest into                                                                                                                                                                                     | No default |
| dataset_path | Path to data-set                                                                                                                                                                                                 | No default |
| attributes_dataset_name | Name of dataset with additional attributes inside the main dataset                                                                                                                                               | No default |
| attribute_spec | Definition of attributes, format is: [{ name: [name_val], type: [type_val]}] Order is important and must match order of attributes column in dataset file. It should contains { name: 'parent_id', type: 'int'}  | No default |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Total time to ingest the dataset into the index.| ms |

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
| dataset_format | Format the dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs. | 'hdf5' |
| dataset_path | Path to dataset | No default |
| neighbors_format | Format the neighbors dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs. | 'hdf5' |
| neighbors_path | Path to neighbors dataset | No default |
| query_count | Number of queries to create from data-set | Size of the data-set |

##### Metrics

| Metric Name | Description                                                                                             | Unit |  
| ----------- |---------------------------------------------------------------------------------------------------------| ----------- |
| took | Took times returned per query aggregated as total, p50, p90, p99, p99.9 and p100 (when applicable)      | ms |
| memory_kb | Native memory k-NN is using at the end of the query workload                                            | KB |
| recall@R | ratio of top R results from the ground truth neighbors that are in the K results returned by the plugin | float 0.0-1.0 |
| recall@K | ratio of results returned that were ground truth nearest neighbors                                      | float 0.0-1.0 |

#### query_with_filter

Runs a set of queries with filter against an index.

##### Parameters

| Parameter Name | Description                                                                                                                                                                                                                               | Default              |  
| ----------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| k | Number of neighbors to return on search                                                                                                                                                                                                   | 100                  |
| r | r value in Recall@R                                                                                                                                                                                                                       | 1                    |
| index_name | Name of index to search                                                                                                                                                                                                                   | No default           |
| field_name | Name field to search                                                                                                                                                                                                                      | No default           |
| calculate_recall | Whether to calculate recall values                                                                                                                                                                                                        | False                |
| dataset_format | Format the dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs.                                                                               | 'hdf5'               |
| dataset_path | Path to dataset                                                                                                                                                                                                                           | No default           |
| neighbors_format | Format the neighbors dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs.                                                                     | 'hdf5'               |
| neighbors_path | Path to neighbors dataset                                                                                                                                                                                                                 | No default           |
| neighbors_dataset | Name of filter dataset inside the neighbors dataset                                                                                                                                                                                       | No default           |
| filter_spec | Path to filter specification                                                                                                                                                                                                              | No default           |
| filter_type | Type of filter format, we do support following types: <br/>FILTER inner filter format for approximate k-NN search<br/>SCRIPT score scripting with exact k-NN search and pre-filtering<br/>BOOL_POST_FILTER Bool query with post-filtering | SCRIPT               |
| score_script_similarity | Similarity function that has been used to index dataset. Used for SCRIPT filter type and ignored for others                                                                                                                               | l2                   |
| query_count | Number of queries to create from data-set                                                                                                                                                                                                 | Size of the data-set |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Took times returned per query aggregated as total, p50, p90 and p99 (when applicable) | ms |
| memory_kb | Native memory k-NN is using at the end of the query workload | KB |
| recall@R | ratio of top R results from the ground truth neighbors that are in the K results returned by the plugin | float 0.0-1.0 |
| recall@K | ratio of results returned that were ground truth nearest neighbors  | float 0.0-1.0 |


#### query_nested_field

Runs a set of queries with nested field against an index.

##### Parameters

| Parameter Name | Description                                                                                                                                                                                                                               | Default              |  
| ----------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| k | Number of neighbors to return on search                                                                                                                                                                                                   | 100                  |
| r | r value in Recall@R                                                                                                                                                                                                                       | 1                    |
| index_name | Name of index to search                                                                                                                                                                                                                   | No default           |
| field_name | Name field to search                                                                                                                                                                                                                      | No default           |
| calculate_recall | Whether to calculate recall values                                                                                                                                                                                                        | False                |
| dataset_format | Format the dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs.                                                                               | 'hdf5'               |
| dataset_path | Path to dataset                                                                                                                                                                                                                           | No default           |
| neighbors_format | Format the neighbors dataset is in. Currently hdf5 and bigann is supported. The hdf5 file must be organized in the same way that the ann-benchmarks organizes theirs.                                                                     | 'hdf5'               |
| neighbors_path | Path to neighbors dataset                                                                                                                                                                                                                 | No default           |
| neighbors_dataset | Name of filter dataset inside the neighbors dataset                                                                                                                                                                                       | No default           |
| query_count | Number of queries to create from data-set                                                                                                                                                                                                 | Size of the data-set |

##### Metrics

| Metric Name | Description | Unit |  
| ----------- | ----------- | ----------- |
| took | Took times returned per query aggregated as total, p50, p90 and p99 (when applicable) | ms |
| memory_kb | Native memory k-NN is using at the end of the query workload | KB |
| recall@R | ratio of top R results from the ground truth neighbors that are in the K results returned by the plugin | float 0.0-1.0 |
| recall@K | ratio of results returned that were ground truth nearest neighbors  | float 0.0-1.0 |

#### get_stats

Gets the index stats.

##### Parameters

| Parameter Name | Description                                                                                                                                                                                                                               | Default              |  
| ----------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------| 
| index_name | Name of index to search                                                                                                                                                                                                                   | No default           |

##### Metrics

| Metric Name | Description                                     | Unit       |  
| ----------- |-------------------------------------------------|------------|
| num_of_committed_segments | Total number of commited segments in the index  | integer >= 0 |
| num_of_search_segments | Total number of search segments in the index    | integer >= 0 |

### Data sets

This benchmark tool uses pre-generated data sets to run indexing and query workload. For some benchmark types existing dataset need to be 
extended. Filtering is an example of use case where such dataset extension is needed.

It's possible to use script provided with this repo to generate dataset and run benchmark for filtering queries.
You need to have existing dataset with vector data. This dataset will be used  to generate additional attribute data and set of ground truth neighbours document ids.

To generate dataset with attributes based on vectors only dataset use following command pattern:

```commandline
python add-filters-to-dataset.py <path_to_dataset_with_vectors> <path_of_new_dataset_with_attributes> True False
```

To generate neighbours dataset for different filters based on dataset with attributes use following command pattern:

```commandline
python add-filters-to-dataset.py <path_to_dataset_with_vectors> <path_of_new_dataset_with_attributes> False True
```

After that new dataset(s) can be referred from testcase definition in `ingest_extended` and `query_with_filter` steps.

To generate dataset with parent doc id based on vectors only dataset, use following command pattern:
```commandline
python add-parent-doc-id-to-dataset.py <path_to_dataset_with_vectors> <path_of_new_dataset_with_parent_id>
```
This will generate neighbours dataset as well. This new dataset(s) can be referred from testcase definition in `ingest_nested_field` and `query_nested_field` steps.

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
