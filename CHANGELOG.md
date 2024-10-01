
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
* Introduce new setting to configure whether to build vector data structure or not during segment creation [#2007](https://github.com/opensearch-project/k-NN/pull/2007)
### Enhancements
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
* Optimize reduceToTopK in ResultUtil by removing pre-filling and reducing peek calls [#2146](https://github.com/opensearch-project/k-NN/pull/2146)
* Update Default Rescore Context based on Dimension [#2149](https://github.com/opensearch-project/k-NN/pull/2149)
* KNNIterators should support with and without filters [#2155](https://github.com/opensearch-project/k-NN/pull/2155)
* Adding Support to Enable/Disble Share level Rescoring and Update Oversampling Factor[#2172](https://github.com/opensearch-project/k-NN/pull/2172)
* Add exact search if no native engine files are available [#2136] (https://github.com/opensearch-project/k-NN/pull/2136)
### Bug Fixes
* KNN80DocValues should only be considered for BinaryDocValues fields [#2147](https://github.com/opensearch-project/k-NN/pull/2147)
### Infrastructure
### Documentation
### Maintenance
* Remove benchmarks folder from k-NN repo [#2127](https://github.com/opensearch-project/k-NN/pull/2127)
### Refactoring
* Minor refactoring and refactored some unit test [#2167](https://github.com/opensearch-project/k-NN/pull/2167)
