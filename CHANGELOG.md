
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.5](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features

### Maintenance
* Added gradle task to generate task dependency graph ([#3032](https://github.com/opensearch-project/k-NN/pull/3032))
* Added new gradle task validateLibraryUsage so that System.loadLibrary cannot be run outside KNNLibraryLoader ([#3033](https://github.com/opensearch-project/k-NN/pull/3033))
* Add IT and bwc test with indices containing both vector and non-vector docs ([#3064](https://github.com/opensearch-project/k-NN/pull/3064))


### Bug Fixes
* Fix indexing for 16x and 8x compression [#3019](https://github.com/opensearch-project/k-NN/pull/3019)
* Block index creation for Faiss engine with cosine similarity and byte vectors due to normalization incompatibility [#3002](https://github.com/opensearch-project/k-NN/pull/3002)
* Update validation for cases when k is greater than total results [#3038](https://github.com/opensearch-project/k-NN/pull/3038)
* Add regex support to derived source transformer include / exclude check [#3031](https://github.com/opensearch-project/k-NN/pull/3031)
* Correct ef_search parameter for Lucene engine and override mergeLeafResults to return top K results [#3037](https://github.com/opensearch-project/k-NN/pull/3037)
* Fix efficient filtering in nested k-NN queries [#2990](https://github.com/opensearch-project/k-NN/pull/2990)
* Fix nested docs and exact search query when some docs has no vector field present. [#3051](https://github.com/opensearch-project/k-NN/pull/3051)
* Changed warmup seek to use long instead of int to avoid overflow [#3067](https://github.com/opensearch-project/k-NN/pull/3067)
* Fix memory-optimized-search reentrant search bug in byte index. [#3071](https://github.com/opensearch-project/k-NN/pull/3071)

### Refactoring
* Change ordering of build task and added tests to catch uninitialized libraries [#3024](https://github.com/opensearch-project/k-NN/pull/3024)

### Enhancements
* Index setting to disable exact search after ANN Search with Faiss efficient filters [#3022](https://github.com/opensearch-project/k-NN/pull/3022)
