## Version 2.6.0.0 Release Notes

Compatible with OpenSearch 2.6.0


### Bug Fixes

* Remove latestSettings cache from KNNSettings ([#727](https://github.com/opensearch-project/k-NN/pull/727))

### Infrastructure

* Add p99.9, p100 and num_of_segments metrics to perf-tool ([#739](https://github.com/opensearch-project/k-NN/pull/739))
* Update bwc to 2.6.0-SNAPSHOT ([#723](https://github.com/opensearch-project/k-NN/pull/723))
* Add Windows Support to BWC Tests ([#726](https://github.com/opensearch-project/k-NN/pull/726))
* Add test for KNNWeight ([#759](https://github.com/opensearch-project/k-NN/pull/759))
* Set NoMergePolicy for codec tests ([#754](https://github.com/opensearch-project/k-NN/pull/754))

### Maintenance

* Replace KnnQueryVector by KnnFloatVectorQuery for Lucene knn ([#767](https://github.com/opensearch-project/k-NN/pull/767))

### Refactoring

* Refactor structure of stats module ([#736](https://github.com/opensearch-project/k-NN/pull/736))
