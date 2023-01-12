## Version 2.5.0.0 Release Notes

Compatible with OpenSearch 2.5.0

### Enhancements

* Extend SystemIndexPlugin for k-NN model system index ([#630](https://github.com/opensearch-project/k-NN/pull/630))
* Add Lucene specific file extensions to core HybridFS ([#721](https://github.com/opensearch-project/k-NN/pull/721))

### Bug Fixes

* Add fix to fromXContent and toXContent in ModelGraveyard ([#624](https://github.com/opensearch-project/k-NN/pull/624))
* Allow mapping service to be null for scenarios of shard recovery from translog ([#685](https://github.com/opensearch-project/k-NN/pull/685))
* Add backward compatibility and validation checks to ModelGraveyard XContent bug fix ([#692](https://github.com/opensearch-project/k-NN/pull/692))

### Infrastructure

* Add benchmark workflow for queries with filters ([#598](https://github.com/opensearch-project/k-NN/pull/598))
* Fix failing codec unit test ([#610](https://github.com/opensearch-project/k-NN/pull/610))
* Update bwc tests for 2.5.0 ([#661](https://github.com/opensearch-project/k-NN/pull/661))
* Add release configs for lucene filtering ([#663](https://github.com/opensearch-project/k-NN/pull/663))
* Update backwards compatibility versions ([#701](https://github.com/opensearch-project/k-NN/pull/701))
* Update tests for backwards codecs ([#710](https://github.com/opensearch-project/k-NN/pull/710))

### Documentation

* Update MAINTAINERS.md format ([#709](https://github.com/opensearch-project/k-NN/pull/709))

### Maintenance

* Fix the codec94 version import statements ([#684](https://github.com/opensearch-project/k-NN/pull/684))
* Add integ test for index close/open scenario ([#693](https://github.com/opensearch-project/k-NN/pull/693))
* Add Lucene 9.5 codec and make it new default ([#700](https://github.com/opensearch-project/k-NN/pull/700))
* Make version of lucene k-nn engine match lucene current version ([#691](https://github.com/opensearch-project/k-NN/pull/691))
* Increment version to 2.5.0-SNAPSHOT ([#632](https://github.com/opensearch-project/k-NN/pull/632))
