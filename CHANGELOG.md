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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.13...2.x)
### Features
* Support radial search in k-NN plugin [#814](https://github.com/opensearch-project/k-NN/issues/814)
### Enhancements
* Make the HitQueue size more appropriate for exact search [#1549](https://github.com/opensearch-project/k-NN/pull/1549)
* Support script score when doc value is disabled [#1573](https://github.com/opensearch-project/k-NN/pull/1573)
* Implemented the Streaming Feature to stream vectors from Java to JNI layer to enable creation of larger segments for vector indices [#1604](https://github.com/opensearch-project/k-NN/pull/1604)
* Remove unnecessary toString conversion of vector field and added some minor optimization in KNNCodec [1613](https://github.com/opensearch-project/k-NN/pull/1613)
* Serialize all models into cluster metadata [#1499](https://github.com/opensearch-project/k-NN/pull/1499)
### Bug Fixes
* Add stored fields for knn_vector type [#1630](https://github.com/opensearch-project/k-NN/pull/1630)
### Infrastructure
* Add micro-benchmark module in k-NN plugin for benchmark streaming vectors to JNI layer functionality. [#1583](https://github.com/opensearch-project/k-NN/pull/1583)
* Add arm64 check when SIMD is disabled [#1618](https://github.com/opensearch-project/k-NN/pull/1618)
* Skip rebuild from scratch after cmake is ran [#1636](https://github.com/opensearch-project/k-NN/pull/1636)
### Documentation
### Maintenance
### Refactoring
