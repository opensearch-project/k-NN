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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.16...2.x)
### Features
### Enhancements
### Bug Fixes
* Corrected search logic for scenario with non-existent fields in filter [#1874](https://github.com/opensearch-project/k-NN/pull/1874)
* Add script_fields context to KNNAllowlist [#1917] (https://github.com/opensearch-project/k-NN/pull/1917)
* Fix graph merge stats size calculation [#1844](https://github.com/opensearch-project/k-NN/pull/1844)
* Integrate Lucene Vector field with native engines to use KNNVectorFormat during segment creation [#1945](https://github.com/opensearch-project/k-NN/pull/1945)
* Disallow a vector field to have an invalid character for a physical file name. [#1936] (https://github.com/opensearch-project/k-NN/pull/1936)
### Infrastructure
### Documentation
### Maintenance
* Fix a flaky unit test:testMultiFieldsKnnIndex, which was failing due to inconsistent merge behaviors [#1924](https://github.com/opensearch-project/k-NN/pull/1924)
### Refactoring
* Introduce KNNVectorValues interface to iterate on different types of Vector values during indexing and search [#1897](https://github.com/opensearch-project/k-NN/pull/1897)
* Clean up parsing for query [#1824](https://github.com/opensearch-project/k-NN/pull/1824)
* Refactor engine package structure [#1913](https://github.com/opensearch-project/k-NN/pull/1913)
* Refactor method structure and definitions [#1920](https://github.com/opensearch-project/k-NN/pull/1920)
