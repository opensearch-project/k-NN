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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.14...2.x)
### Features
* Use the Lucene Distance Calculation Function in Script Scoring for doing exact search [#1699](https://github.com/opensearch-project/k-NN/pull/1699)
### Enhancements
* Add KnnCircuitBreakerException and modify exception message [#1688](https://github.com/opensearch-project/k-NN/pull/1688)
* Add stats for radial search [#1684](https://github.com/opensearch-project/k-NN/pull/1684)
* Support script score when doc value is disabled and fix misusing DISI [#1696](https://github.com/opensearch-project/k-NN/pull/1696)
* Add validation for pq m parameter before training starts [#1713](https://github.com/opensearch-project/k-NN/pull/1713)
### Bug Fixes
* Block commas in model description [#1692](https://github.com/opensearch-project/k-NN/pull/1692)
* Update threshold value after new result is added [#1715](https://github.com/opensearch-project/k-NN/pull/1715)
### Infrastructure
### Documentation
### Maintenance
### Refactoring
