
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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.18...2.x)
### Features
### Enhancements
### Bug Fixes
* Fixing the bug when a segment has no vector field present for disk based vector search (#2282)[https://github.com/opensearch-project/k-NN/pull/2282]
* Fix for NPE while merging segments after all the vector fields docs are deleted (#2365)[https://github.com/opensearch-project/k-NN/pull/2365]
* Allow validation for non knn index only after 2.17.0 (#2315)[https://github.com/opensearch-project/k-NN/pull/2315]
* Fixing the bug to prevent updating the index.knn setting after index creation(#2348)[https://github.com/opensearch-project/k-NN/pull/2348]
* Release query vector memory after execution (#2346)[https://github.com/opensearch-project/k-NN/pull/2346]
* Fix shard level rescoring disabled setting flag (#2352)[https://github.com/opensearch-project/k-NN/pull/2352]
* Fix filter rewrite logic which was resulting in getting inconsistent / incorrect results for cases where filter was getting rewritten for shards (#2359)[https://github.com/opensearch-project/k-NN/pull/2359]
* Fixing it to retrieve space_type from index setting when both method and top level don't have the value. [#2374](https://github.com/opensearch-project/k-NN/pull/2374)
### Infrastructure
### Documentation
### Maintenance
* Select index settings based on cluster version[2236](https://github.com/opensearch-project/k-NN/pull/2236)
### Refactoring
