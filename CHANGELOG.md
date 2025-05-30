
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Infrastructure
* Add testing support to run all ITs with remote index builder [#2659](https://github.com/opensearch-project/k-NN/pull/2659)
* Fix KNNSettingsTests after change in MockNode constructor [#2700](https://github.com/opensearch-project/k-NN/pull/2700)
### Enhancements
* Removing redundant type conversions for script scoring for hamming space with binary vectors [#2351](https://github.com/opensearch-project/k-NN/pull/2351)
* [Remote Vector Index Build] Add tuned repository upload/download configurations per benchmarking results [#2662](https://github.com/opensearch-project/k-NN/pull/2662)
* Apply mask operation in preindex to optimize derived source [#2704](https://github.com/opensearch-project/k-NN/pull/2704)
### Bug Fixes
* [BUGFIX] Fix KNN Quantization state cache have an invalid weight threshold [#2666](https://github.com/opensearch-project/k-NN/pull/2666)
* [BUGFIX] Fix enable rescoring when dimensions > 1000. [#2671](https://github.com/opensearch-project/k-NN/pull/2671) 
* [BUGFIX] Honors slice counts for non-quantization cases [#2692](https://github.com/opensearch-project/k-NN/pull/2692)
* [BUGFIX] Block derived source enable if index.knn is false [#2702](https://github.com/opensearch-project/k-NN/pull/2702)
* [BUGFIX] Avoid opening of graph file if graph is already loaded in memory [#2719](https://github.com/opensearch-project/k-NN/pull/2719)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
### Bug Fixes
* [BUGFIX] FIX nested vector query at efficient filter scenarios [#2641](https://github.com/opensearch-project/k-NN/pull/2641)
### Infrastructure
### Documentation
### Maintenance
### Refactoring
