#!/bin/bash

set -ex


#reinstall 
./bin/opensearch-plugin remove opensearch-neural-search || true
./bin/opensearch-plugin remove opensearch-knn || true

./bin/opensearch-plugin install file:///usr/share/opensearch/knn-plugins/artifacts/plugins/opensearch-knn-3.0.0.0-SNAPSHOT.zip -b
./opensearch-docker-entrypoint.sh
