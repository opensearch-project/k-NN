
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
- Add Support for Multi Values in innerHit for Nested k-NN Fields in Lucene and FAISS (#2283)[https://github.com/opensearch-project/k-NN/pull/2283]
- Add binary index support for Lucene engine. (#2292)[https://github.com/opensearch-project/k-NN/pull/2292]
- Add expand_nested_docs Parameter support to NMSLIB engine (#2331)[https://github.com/opensearch-project/k-NN/pull/2331]
### Enhancements
- Introduced a writing layer in native engines where relies on the writing interface to process IO. (#2241)[https://github.com/opensearch-project/k-NN/pull/2241]
- Allow method parameter override for training based indices (#2290) https://github.com/opensearch-project/k-NN/pull/2290]
- Optimizes lucene query execution to prevent unnecessary rewrites (#2305)[https://github.com/opensearch-project/k-NN/pull/2305]
- Add check to directly use ANN Search when filters match all docs. (#2320)[https://github.com/opensearch-project/k-NN/pull/2320]
### Bug Fixes
* Fixing the bug when a segment has no vector field present for disk based vector search (#2282)[https://github.com/opensearch-project/k-NN/pull/2282]
* Fix for NPE while merging segments after all the vector fields docs are deleted (#2365)[https://github.com/opensearch-project/k-NN/pull/2365]
* Allow validation for non knn index only after 2.17.0 (#2315)[https://github.com/opensearch-project/k-NN/pull/2315]
* Release query vector memory after execution (#2346)[https://github.com/opensearch-project/k-NN/pull/2346]
* Fix shard level rescoring disabled setting flag (#2352)[https://github.com/opensearch-project/k-NN/pull/2352]
* Fix filter rewrite logic which was resulting in getting inconsistent / incorrect results for cases where filter was getting rewritten for shards (#2359)[https://github.com/opensearch-project/k-NN/pull/2359]
### Infrastructure
* Updated C++ version in JNI from c++11 to c++17 [#2259](https://github.com/opensearch-project/k-NN/pull/2259)
* Upgrade bytebuddy and objenesis version to match OpenSearch core and, update github ci runner for macos [#2279](https://github.com/opensearch-project/k-NN/pull/2279)
### Documentation
### Maintenance
* Select index settings based on cluster version[2236](https://github.com/opensearch-project/k-NN/pull/2236)
* Added null checks for fieldInfo in ExactSearcher to avoid NPE while running exact search for segments with no vector field (#2278)[https://github.com/opensearch-project/k-NN/pull/2278]
* Upgrade jsonpath from 2.8.0 to 2.9.0[2325](https://github.com/opensearch-project/k-NN/pull/2325)
### Refactoring
