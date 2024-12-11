
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
* Removed JDK 11 and 17 version from CI runs [#1921](https://github.com/opensearch-project/k-NN/pull/1921)
### Documentation
### Maintenance
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.18...2.x)
### Features
- Add Support for Multi Values in innerHit for Nested k-NN Fields in Lucene and FAISS (#2283)[https://github.com/opensearch-project/k-NN/pull/2283]
### Enhancements
- Introduced a writing layer in native engines where relies on the writing interface to process IO. (#2241)[https://github.com/opensearch-project/k-NN/pull/2241]
- Allow method parameter override for training based indices (#2290) https://github.com/opensearch-project/k-NN/pull/2290]
### Bug Fixes
* Fixing the bug when a segment has no vector field present for disk based vector search (#2282)[https://github.com/opensearch-project/k-NN/pull/2282]
### Infrastructure
* Updated C++ version in JNI from c++11 to c++17 [#2259](https://github.com/opensearch-project/k-NN/pull/2259)
* Upgrade bytebuddy and objenesis version to match OpenSearch core and, update github ci runner for macos [#2279](https://github.com/opensearch-project/k-NN/pull/2279)
### Documentation
### Maintenance
* Select index settings based on cluster version[2236](https://github.com/opensearch-project/k-NN/pull/2236)
* Added null checks for fieldInfo in ExactSearcher to avoid NPE while running exact search for segments with no vector field (#2278)[https://github.com/opensearch-project/k-NN/pull/2278]
* Upgrade jsonpath from 2.8.0 to 2.9.0[2325](https://github.com/opensearch-project/k-NN/pull/2325)
### Refactoring
