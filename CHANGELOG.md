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
* Add binary format support with HNSW method in Faiss Engine [#1781](https://github.com/opensearch-project/k-NN/pull/1781)
### Enhancements
* Switch from byte stream to byte ref for serde [#1825](https://github.com/opensearch-project/k-NN/pull/1825)
### Bug Fixes
* Fixing the arithmetic to find the number of vectors to stream from java to jni layer.[#1804](https://github.com/opensearch-project/k-NN/pull/1804)
* Release memory properly for an array type [#1820](https://github.com/opensearch-project/k-NN/pull/1820)
* FIX Same Suffix Cause Recall Drop to zero [#1802](https://github.com/opensearch-project/k-NN/pull/1802)
### Infrastructure
### Documentation
### Maintenance
### Refactoring
