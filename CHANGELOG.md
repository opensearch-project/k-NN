
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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.17...2.x)
### Features
* Add AVX512 support to k-NN for FAISS library [#2069](https://github.com/opensearch-project/k-NN/pull/2069)
### Enhancements
* Introducing a loading layer in FAISS [#2033](https://github.com/opensearch-project/k-NN/issues/2033)
* Add short circuit if no live docs are in segments [#2059](https://github.com/opensearch-project/k-NN/pull/2059)
* Optimize reduceToTopK in ResultUtil by removing pre-filling and reducing peek calls [#2146](https://github.com/opensearch-project/k-NN/pull/2146)
* Update Default Rescore Context based on Dimension [#2149](https://github.com/opensearch-project/k-NN/pull/2149)
* KNNIterators should support with and without filters [#2155](https://github.com/opensearch-project/k-NN/pull/2155)
* Adding Support to Enable/Disble Share level Rescoring and Update Oversampling Factor[#2172](https://github.com/opensearch-project/k-NN/pull/2172)
* Add support to build vector data structures greedily and perform exact search when there are no engine files [#1942](https://github.com/opensearch-project/k-NN/issues/1942)
### Bug Fixes
* Add DocValuesProducers for releasing memory when close index [#1946](https://github.com/opensearch-project/k-NN/pull/1946)
* KNN80DocValues should only be considered for BinaryDocValues fields [#2147](https://github.com/opensearch-project/k-NN/pull/2147)
* Score Fix for Binary Quantized Vector and Setting Default value in case of shard level rescoring is disabled for oversampling factor[#2183](https://github.com/opensearch-project/k-NN/pull/2183)
### Infrastructure
### Documentation
* Fix sed command in DEVELOPER_GUIDE.md to append a new line character '\n'. [#2181](https://github.com/opensearch-project/k-NN/pull/2181)
### Maintenance
* Remove benchmarks folder from k-NN repo [#2127](https://github.com/opensearch-project/k-NN/pull/2127)
* Fix lucene codec after lucene version bumped to 9.12. [#2195](https://github.com/opensearch-project/k-NN/pull/2195)
### Refactoring
* Does not create additional KNNVectorValues in NativeEngines990KNNVectorWriter when quantization is not needed [#2133](https://github.com/opensearch-project/k-NN/pull/2133)
* Minor refactoring and refactored some unit test [#2167](https://github.com/opensearch-project/k-NN/pull/2167)
* Remove explicit K from exact search context  [#2212](https://github.com/opensearch-project/k-NN/pull/2212)
