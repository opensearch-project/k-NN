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
### Enhancements
* Optize Faiss Query With Filters: Reduce iteration and memory for id filter [#1402](https://github.com/opensearch-project/k-NN/pull/1402)
* Detect AVX2 Dynamically on the System [#1502](https://github.com/opensearch-project/k-NN/pull/1502)
* Validate zero vector when using cosine metric [#1501](https://github.com/opensearch-project/k-NN/pull/1501)
* Persist model definition in model metadata [#1527] (https://github.com/opensearch-project/k-NN/pull/1527)
* Added Inner Product Space type support for Lucene Engine [#1551](https://github.com/opensearch-project/k-NN/pull/1551)
* Add Range Validation for Faiss SQFP16 [#1493](https://github.com/opensearch-project/k-NN/pull/1493)
### Bug Fixes
* Disable sdc table for HNSWPQ read-only indices [#1518](https://github.com/opensearch-project/k-NN/pull/1518)
* Switch SpaceType.INNERPRODUCT's vector similarity function to MAXIMUM_INNER_PRODUCT [#1532](https://github.com/opensearch-project/k-NN/pull/1532)
* Add patch to fix arm segfault in nmslib during ingestion [#1541](https://github.com/opensearch-project/k-NN/pull/1541)
### Infrastructure
* Manually install zlib for win CI [#1513](https://github.com/opensearch-project/k-NN/pull/1513)
* Update k-NN build artifact script to enable SIMD on ARM for Faiss [#1543](https://github.com/opensearch-project/k-NN/pull/1543)
### Documentation
### Maintenance
* Bump faiss lib commit to 32f0e8cf92cd2275b60364517bb1cce67aa29a55 [#1443](https://github.com/opensearch-project/k-NN/pull/1443)
* Fix FieldInfo Parameters Mismatch [#1489](https://github.com/opensearch-project/k-NN/pull/1489)
* Upgrade faiss to 12b92e9 [#1509](https://github.com/opensearch-project/k-NN/pull/1509)
### Refactoring
