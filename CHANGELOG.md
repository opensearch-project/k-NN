
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.6](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features

### Maintenance
* Improve unit tests by tightening asserts [#3112](https://github.com/opensearch-project/k-NN/pull/3112)

### Bug Fixes
* Fix thread leak in HNSW merge executor caused by per-call thread pool creation [#3119](https://github.com/opensearch-project/k-NN/pull/3119)
* Fix score conversion logic for radial exact search [#3110](https://github.com/opensearch-project/k-NN/pull/3110)

### Refactoring

### Enhancements
* Make Merge in nativeEngine can Abort [#2529](https://github.com/opensearch-project/k-NN/pull/2529)
* Use pre-quantized vectors from native engines for exact search [#3095](https://github.com/opensearch-project/k-NN/pull/3095)
* Use right Vector Scorer when segments are initialized using SPI and also corrected the maxConn for MOS [#3117](https://github.com/opensearch-project/k-NN/pull/3117)

