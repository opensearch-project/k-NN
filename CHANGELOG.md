
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
* [Remote Vector Index Build] Client polling mechanism, encoder check, method parameter retrieval [#2576](https://github.com/opensearch-project/k-NN/pull/2576)
* [Remote Vector Index Build] Move client to separate module [#2603](https://github.com/opensearch-project/k-NN/pull/2603)
* Add filter function to KNNQueryBuilder with unit tests and integration tests [#2599](https://github.com/opensearch-project/k-NN/pull/2599)
* [Lucene On Faiss] Add a new mode, memory-optimized-search enable user to run vector search on FAISS index under memory constrained environment. [#2630](https://github.com/opensearch-project/k-NN/pull/2630)
### Enhancements
### Bug Fixes
* Fixing bug to prevent NullPointerException while doing PUT mappings [#2556](https://github.com/opensearch-project/k-NN/issues/2556)
* Add index operation listener to update translog source [#2629](https://github.com/opensearch-project/k-NN/pull/2629)
* [Remote Vector Index Build] Fix bug to support `COSINESIMIL` space type [#2627](https://github.com/opensearch-project/k-NN/pull/2627)
### Infrastructure
* Add github action to run ITs against remote index builder [2620](https://github.com/opensearch-project/k-NN/pull/2620)
### Documentation
### Maintenance
* Update minimum required CMAKE version in NMSLIB [#2635](https://github.com/opensearch-project/k-NN/pull/2635)
* Revert CMake version bump, instead add CMake policy version flag to build task to support modern CMake builds [#2645](https://github.com/opensearch-project/k-NN/pull/2645/files)
### Refactoring
* Switch derived source from field attributes to segment attribute [#2606](https://github.com/opensearch-project/k-NN/pull/2606)
* Migrate derived source from filter to mask [#2612](https://github.com/opensearch-project/k-NN/pull/2612)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
### Refactoring
