# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Enhancements
### Bug Fixes 
### Infrastructure
### Documentation
### Maintenance
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.15...2.x)
### Features
* Adds dynamic query parameter ef_search [#1783](https://github.com/opensearch-project/k-NN/pull/1783)
* Adds dynamic query parameter ef_search in radial search faiss engine [#1790](https://github.com/opensearch-project/k-NN/pull/1790)
* Adds dynamic query parameter nprobes [#1792](https://github.com/opensearch-project/k-NN/pull/1792)
* Add binary format support with HNSW method in Faiss Engine [#1781](https://github.com/opensearch-project/k-NN/pull/1781)
* Add script scoring support for knn field with binary data type [#1826](https://github.com/opensearch-project/k-NN/pull/1826)
* Add painless script support for hamming with binary vector data type [#1839](https://github.com/opensearch-project/k-NN/pull/1839)
* Add binary format support with IVF method in Faiss Engine [#1784](https://github.com/opensearch-project/k-NN/pull/1784)
### Enhancements
* Switch from byte stream to byte ref for serde [#1825](https://github.com/opensearch-project/k-NN/pull/1825)
### Bug Fixes
* Fixing the arithmetic to find the number of vectors to stream from java to jni layer.[#1804](https://github.com/opensearch-project/k-NN/pull/1804)
* Fixed LeafReaders casting errors to SegmentReaders when segment replication is enabled during search.[#1808](https://github.com/opensearch-project/k-NN/pull/1808)
* Release memory properly for an array type [#1820](https://github.com/opensearch-project/k-NN/pull/1820)
* FIX Same Suffix Cause Recall Drop to zero [#1802](https://github.com/opensearch-project/k-NN/pull/1802)
### Infrastructure
* Apply custom patch only once by comparing the last patch id  [#1833](https://github.com/opensearch-project/k-NN/pull/1833)
### Documentation
### Maintenance
* Bump faiss commit to 33c0ba5 [#1796](https://github.com/opensearch-project/k-NN/pull/1796)
### Refactoring
