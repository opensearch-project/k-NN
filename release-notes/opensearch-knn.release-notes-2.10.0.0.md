## Version 2.10.0.0 Release Notes

Compatible with OpenSearch 2.10.0

### Features
* ~~Add Clear Cache API ([#740](https://github.com/opensearch-project/k-NN/pull/740))~~ Feature was mistakenly added to the release notes, although it was not included in the release.
### Enhancements
* Enabled the IVF algorithm to work with Filters of K-NN Query. ([#1013](https://github.com/opensearch-project/k-NN/pull/1013))
* Improved the logic to switch to exact search for restrictive filters search for better recall. ([#1059](https://github.com/opensearch-project/k-NN/pull/1059))
* Added max distance computation logic to enhance the switch to exact search in filtered Nearest Neighbor Search. ([#1066](https://github.com/opensearch-project/k-NN/pull/1066))
### Bug Fixes
* Update Faiss parameter construction to allow HNSW+PQ to work ([#1074](https://github.com/opensearch-project/k-NN/pull/1074))
### Maintenance
* Update Guava Version to 32.0.1 ([#1019](https://github.com/opensearch-project/k-NN/pull/1019))
### Refactoring
* Fix TransportAddress Refactoring Changes in Core ([#1020](https://github.com/opensearch-project/k-NN/pull/1020))
