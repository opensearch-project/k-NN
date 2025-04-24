
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Enhancements
* Removing redundant type conversions for script scoring for hamming space with binary vectors [#2351](https://github.com/opensearch-project/k-NN/pull/2351)
### Bug Fixes
* [BUGFIX] Fix KNN Quantization state cache have an invalid weight threshold [#2666](https://github.com/opensearch-project/k-NN/pull/2666)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
* [Vector Profiler] Adding basic generic vector profiler implementation and tests. [#2624](https://github.com/opensearch-project/k-NN/pull/2624)
* [Vector Profiler] Adding main segment implementation for API and indexing. [#2653](https://github.com/opensearch-project/k-NN/pull/2653)
### Enhancements
### Bug Fixes
* [BUGFIX] FIX nested vector query at efficient filter scenarios [#2641](https://github.com/opensearch-project/k-NN/pull/2641)
### Infrastructure
### Documentation
### Maintenance
### Refactoring
