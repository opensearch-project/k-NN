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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.12...2.x)
### Features
* Add parent join support for lucene knn [#1182](https://github.com/opensearch-project/k-NN/pull/1182)
### Enhancements
* Increase Lucene max dimension limit to 16,000 [#1346](https://github.com/opensearch-project/k-NN/pull/1346)
* Tuned default values for ef_search and ef_construction for better indexing and search performance for vector search [#1353](https://github.com/opensearch-project/k-NN/pull/1353)
* Enabled Filtering on Nested Vector fields with top level filters [#1372](https://github.com/opensearch-project/k-NN/pull/1372)
### Bug Fixes
* Fix use-after-free case on nmslib search path [#1305](https://github.com/opensearch-project/k-NN/pull/1305)
* Allow nested knn field mapping when train model [#1318](https://github.com/opensearch-project/k-NN/pull/1318)
* Properly designate model state for actively training models when nodes crash or leave cluster [#1317](https://github.com/opensearch-project/k-NN/pull/1317)
* Fix script score queries not getting cached [#1367](https://github.com/opensearch-project/k-NN/pull/1367)
* Throw proper exception to invalid k-NN query [#1380](https://github.com/opensearch-project/k-NN/pull/1380)
### Infrastructure
* Upgrade gradle to 8.4 [1289](https://github.com/opensearch-project/k-NN/pull/1289)
* Refactor security testing to install from individual components [#1307](https://github.com/opensearch-project/k-NN/pull/1307)
### Documentation
### Maintenance
* Update developer guide to include M1 Setup [#1222](https://github.com/opensearch-project/k-NN/pull/1222)
* Upgrade urllib to 1.26.17 [#1278](https://github.com/opensearch-project/k-NN/pull/1278)
* Upgrade urllib to 1.26.18 [#1319](https://github.com/opensearch-project/k-NN/pull/1319)
* Upgrade guava to 32.1.3 [#1319](https://github.com/opensearch-project/k-NN/pull/1319)
### Refactoring
