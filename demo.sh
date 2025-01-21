#!/bin/bash

./gradlew run -PcustomDistributionUrl=file://${HOME}/projects/OpenSearch/distribution/archives/darwin-tar/build/distributions/opensearch-min-3.0.0-SNAPSHOT-darwin-x64.tar.gz

# ping local cluster
curl localhost:9200

# Check test cluster status
curl -X GET "http://localhost:9200/_cluster/health?pretty"

# Create new knn index with 1 shard and 0 replicas
curl -X PUT "localhost:9200/my_knn_index?pretty" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index.knn": true,
    "index.number_of_shards": 1,
    "index.number_of_replicas": 0,
    "index.use_compound_file": false
  }
}'

# Check index settings
curl -X GET "localhost:9200/my_knn_index/_settings?pretty"

# Add mapping for knn_vector field with jVector engine
curl -X PUT "localhost:9200/my_knn_index/_mapping?pretty" -H 'Content-Type: application/json' -d'
{
  "properties": {
    "my_vector": {
      "type": "knn_vector",
      "dimension": 3,
      "method": {
        "name": "disk_ann",
        "space_type": "l2",
        "engine": "jvector"
      }
    }
  }
}'


# Check index mapping
curl -X GET "localhost:9200/my_knn_index/_mapping?pretty"

# Add document with knn_vector field
curl -X POST "localhost:9200/_bulk?pretty" -H 'Content-Type: application/json' -d'
{"index": {"_index": "my_knn_index"}}
{"my_vector": [1, 2, 3]}
{"index": {"_index": "my_knn_index"}}
{"my_vector": [4, 5, 6]}
{"index": {"_index": "my_knn_index"}}
{"my_vector": [7, 8, 9]}
'

# refresh index
curl -X POST "localhost:9200/my_knn_index/_refresh?pretty"


# Search for nearest neighbors
curl -X GET "localhost:9200/my_knn_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1, 2, 3],
        "k": 3
      }
    }
  }
}'

# Delete index
curl -X DELETE "localhost:9200/my_knn_index?pretty"


# Check test cluster location
ls -lah build/testclusters/integTest-0/data/nodes/0/indices