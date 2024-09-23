
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Enhancements
* Adds concurrent segment search support for mode auto [#2111](https://github.com/opensearch-project/k-NN/pull/2111)
### Bug Fixes 
* Add DocValuesProducers for releasing memory when close index [#1946](https://github.com/opensearch-project/k-NN/pull/1946)
### Infrastructure
* Removed JDK 11 and 17 version from CI runs [#1921](https://github.com/opensearch-project/k-NN/pull/1921)
### Documentation
### Maintenance
### Refactoring
* Does not create additional KNNVectorValues in NativeEngines990KNNVectorWriter when quantization is not needed [#2133](https://github.com/opensearch-project/k-NN/pull/2133)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.17...2.x)
### Features
* Add AVX512 support to k-NN for FAISS library [#2069](https://github.com/opensearch-project/k-NN/pull/2069)
### Enhancements
* Add short circuit if no live docs are in segments [#2059](https://github.com/opensearch-project/k-NN/pull/2059)
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
* Remove benchmarks folder from k-NN repo [#2127](https://github.com/opensearch-project/k-NN/pull/2127)
### Refactoring
