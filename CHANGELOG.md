
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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.16...2.x)
### Features
* Integrate Lucene Vector field with native engines to use KNNVectorFormat during segment creation [#1945](https://github.com/opensearch-project/k-NN/pull/1945)
* k-NN query rescore support for native engines [#1984](https://github.com/opensearch-project/k-NN/pull/1984)
* Add support for byte vector with Faiss Engine HNSW algorithm [#1823](https://github.com/opensearch-project/k-NN/pull/1823)
### Enhancements
* Adds iterative graph build capability into a faiss index to improve the memory footprint during indexing and Integrates KNNVectorsFormat for native engines[#1950](https://github.com/opensearch-project/k-NN/pull/1950)
### Bug Fixes
* Corrected search logic for scenario with non-existent fields in filter [#1874](https://github.com/opensearch-project/k-NN/pull/1874)
* Add script_fields context to KNNAllowlist [#1917] (https://github.com/opensearch-project/k-NN/pull/1917)
* Fix graph merge stats size calculation [#1844](https://github.com/opensearch-project/k-NN/pull/1844)
* Disallow a vector field to have an invalid character for a physical file name. [#1936](https://github.com/opensearch-project/k-NN/pull/1936)
### Infrastructure
* Parallelize make to reduce build time [#2006] (https://github.com/opensearch-project/k-NN/pull/2006)
### Documentation
### Maintenance
* Fix a flaky unit test:testMultiFieldsKnnIndex, which was failing due to inconsistent merge behaviors [#1924](https://github.com/opensearch-project/k-NN/pull/1924)
### Refactoring
* Introduce KNNVectorValues interface to iterate on different types of Vector values during indexing and search [#1897](https://github.com/opensearch-project/k-NN/pull/1897)
* Integrate KNNVectorValues with vector ANN Search flow [#1952](https://github.com/opensearch-project/k-NN/pull/1952)
* Clean up parsing for query [#1824](https://github.com/opensearch-project/k-NN/pull/1824)
* Refactor engine package structure [#1913](https://github.com/opensearch-project/k-NN/pull/1913)
* Refactor method structure and definitions [#1920](https://github.com/opensearch-project/k-NN/pull/1920)
* Refactor KNNVectorFieldType from KNNVectorFieldMapper to a separate class for better readability. [#1931](https://github.com/opensearch-project/k-NN/pull/1931)
* Generalize lib interface to return context objects [#1925](https://github.com/opensearch-project/k-NN/pull/1925)
* Restructure mappers to better handle null cases and avoid branching in parsing [#1939](https://github.com/opensearch-project/k-NN/pull/1939)
* Added Quantization Framework and implemented 1Bit and multibit quantizer[#1889](https://github.com/opensearch-project/k-NN/issues/1889)
* Encapsulate dimension, vector data type validation/processing inside Library [#1957](https://github.com/opensearch-project/k-NN/pull/1957)
* Add quantization state cache [#1960](https://github.com/opensearch-project/k-NN/pull/1960)