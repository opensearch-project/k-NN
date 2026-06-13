
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.8](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Add rescoring phase after radial search on quantized index [#3347](https://github.com/opensearch-project/k-NN/pull/3347)
* Add base64 encoded vector indexing support for knn_vector fields [#3350](https://github.com/opensearch-project/k-NN/pull/3350)

### Maintenance

### Bug Fixes
* Turn off ACORN for MOS to match default Lucene HNSW behavior [#3346](https://github.com/opensearch-project/k-NN/pull/3346)
* Preserve mixed-case derived source vector field names and add backward-compatible field resolution for previously lowercased segment metadata [#3313](https://github.com/opensearch-project/k-NN/pull/3313)
* Fix rescore flag not propagating over transport layer in multi-node clusters [#3343](https://github.com/opensearch-project/k-NN/pull/3343)
* Integrated proper ef_search functionality into MOS and Lucene with oversample_factor [#3331](https://github.com/opensearch-project/k-NN/pull/3331)
* Fix skip warm up in old indices when MOS is enabled [#3344](https://github.com/opensearch-project/k-NN/pull/3344)
* Check to see if Lucene's search budget has exhausted when deciding to exact search in MOS [#3354](https://github.com/opensearch-project/k-NN/pull/3354)

### Refactoring


### Enhancements
* Speedup cosine similarity using AVX512-FP16 [#3215](https://github.com/opensearch-project/k-NN/pull/3215)
