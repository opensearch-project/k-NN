
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.10](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Support knn queries on native engine fields inside percolator queries by scoring the percolated document with exact search [#3556](https://github.com/opensearch-project/k-NN/pull/3556)
* Add SQ 2-bit and 4-bit for half_float [#3603](https://github.com/opensearch-project/k-NN/pull/3603)

### Maintenance
* Fixed build issues happening due to Lucene 10.5.2 upgrade [#3631](https://github.com/opensearch-project/k-NN/pull/3631)
* Use clustered test vectors and the api-snapshot builder image so remote index build ITs pass on cuVS 26.08 [#3627](https://github.com/opensearch-project/k-NN/pull/3627)

### Bug Fixes
* Fix double free of off-heap vectors when a native index build fails [#3621](https://github.com/opensearch-project/k-NN/pull/3621)
* Honor query cancellation and timeout in memory-optimized search [#3620](https://github.com/opensearch-project/k-NN/pull/3620)
* Fix derived source returning another document's nested vectors when a parent has more nested docs than the first child lookup window [#3624](https://github.com/opensearch-project/k-NN/pull/3624)
* Return model not found instead of the internal model index when getting a model before any model is trained [#3598](https://github.com/opensearch-project/k-NN/pull/3598)
* Support rescoring with `expand_nested_docs` on the Lucene engine, so the final reduction to k counts parent documents instead of child documents [#3579](https://github.com/opensearch-project/k-NN/pull/3579)

### Refactoring

### Enhancements
* Set the CMAKE_BUILD_TYPE to Release to ensure that all targets get build with optimized code [#3622](https://github.com/opensearch-project/k-NN/pull/3622)
* Avoid FP16 -> FP32 -> FP16 round trip when merging `half_float` segments [#3610](https://github.com/opensearch-project/k-NN/pull/3610)
* Report `exact_search` timings in the Profile API for nested k-NN queries with `expand_nested_docs` on the Lucene engine [#3579](https://github.com/opensearch-project/k-NN/pull/3579)
* Send `half_float` vectors to the remote vector index build service as fp16 instead of fp32 [#3608](https://github.com/opensearch-project/k-NN/pull/3608)
