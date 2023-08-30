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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.9...2.x)
### Features
### Enhancements
* Enabled the IVF algorithm to work with Filters of K-NN Query. [#1013](https://github.com/opensearch-project/k-NN/pull/1013)
* Improved the logic to switch to exact search for restrictive filters search for better recall. [#1059](https://github.com/opensearch-project/k-NN/pull/1059)
* Added max distance computation logic to enhance the switch to exact search in filtered Nearest Neighbor Search. [#1066](https://github.com/opensearch-project/k-NN/pull/1066)
### Bug Fixes
* Update Faiss parameter construction to allow HNSW+PQ to work [#1074](https://github.com/opensearch-project/k-NN/pull/1074) 
### Infrastructure
### Documentation
### Maintenance
* Update Guava Version to 32.0.1 [#1019](https://github.com/opensearch-project/k-NN/pull/1019)
### Refactoring
* Fix TransportAddress Refactoring Changes in Core [#1020](https://github.com/opensearch-project/k-NN/pull/1020)
