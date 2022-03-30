# ANN No Train Workload

## Overview

A workload that indexes a data set of vectors in an OpenSearch k-NN index that does not require training.

## Parameters

1. target_index_name - name of index to ingest vectors into - Default("target_index")
2. target_index_body - path to body of index definition - Default("nmslib-index.json")
3. target_field_name - name of field to ingest vectors into - Default("target_field")
4. target_index_primary_shards - number of primary shards - Default(3)
5. target_index_replica_shards - number of replica shards - Default(1)
6. target_index_ef_search - HNSW efSearch parameter - Default(512)
7. target_index_ef_construction - HNSW efConstruction parameter - Default(512)
8. target_index_m - HNSW m parameter - Default(16)
9. target_index_dimension - Dimension of vectors - Default(128)
10. target_index_space_type - Space type to use - Default("l2")
11. bulk_size - Bulk size on ingestion - Default(200)
12. bulk_index_data_set_format - Format of data set (hdf5 or bigann) - No Default
13. bulk_index_data_set_path" - Path to data set - No Default
14. bulk_index_clients - Number of clients to use for indexing - Default(10)
