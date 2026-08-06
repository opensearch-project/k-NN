
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.8](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Add dynamic mapping for knn_vector fields via plugin inferencer and knn_vector dynamic templates [#3490](https://github.com/opensearch-project/k-NN/pull/3490)
* Add rescoring phase after radial search on quantized index [#3347](https://github.com/opensearch-project/k-NN/pull/3347)
* Add base64 encoded vector indexing support for knn_vector fields [#3350](https://github.com/opensearch-project/k-NN/pull/3350)
* Introduce system-generated search pipeline processor to automatically exclude knn_vector fields from _source in search responses [#3152](https://github.com/opensearch-project/k-NN/pull/3152)
* Parameterize integration test framework for compression level [#3416](https://github.com/opensearch-project/k-NN/pull/3416)
* Introduce extensible VectorSearchEngine API [#3288](https://github.com/opensearch-project/k-NN/pull/3443)

### Maintenance
* Upgrade Lucene to 10.5.0 [#3411](https://github.com/opensearch-project/k-NN/pull/3411)
* Re-enable flaky BWC test testKNNWarmupCustomLegacyFieldMapping and restore legacy index settings in BWC test fixtures [#2415](https://github.com/opensearch-project/k-NN/issues/2415)

### Bug Fixes
* Preserve raw non-XContent `_source` fields when derived source is enabled [#3402](https://github.com/opensearch-project/k-NN/pull/3402)
* Fix dimension-based oversampling not applying for 32x compression [#3455](https://github.com/opensearch-project/k-NN/pull/3455)
* Fix NPE in nested kNN search when index contains documents without nested object [#3368](https://github.com/opensearch-project/k-NN/pull/3368)
* Turn off ACORN for MOS to match default Lucene HNSW behavior [#3346](https://github.com/opensearch-project/k-NN/pull/3346)
* Preserve mixed-case derived source vector field names and add backward-compatible field resolution for previously lowercased segment metadata [#3313](https://github.com/opensearch-project/k-NN/pull/3313)
* Fix rescore flag not propagating over transport layer in multi-node clusters [#3343](https://github.com/opensearch-project/k-NN/pull/3343)
* Integrated proper ef_search functionality into MOS and Lucene with oversample_factor [#3331](https://github.com/opensearch-project/k-NN/pull/3331)
* Fix skip warm up in old indices when MOS is enabled [#3344](https://github.com/opensearch-project/k-NN/pull/3344)
* Check to see if Lucene's search budget has exhausted when deciding to exact search in MOS [#3354](https://github.com/opensearch-project/k-NN/pull/3354)
* Fix FAISS SQ merge failure when segment has no live vectors [#3381](https://github.com/opensearch-project/k-NN/pull/3381)
* Fix FAISS SQ merges when an input segment has no vectors [#3433](https://github.com/opensearch-project/k-NN/pull/3433)
* Fix isFaissSQfp16 to skip FP16 validation when SQ encoder uses bits=1 [#3366](https://github.com/opensearch-project/k-NN/pull/3366)
* Fix score corruption in multi-segment FAISS indices with ADC [#3385](https://github.com/opensearch-project/k-NN/pull/3385)
* Fix corrupt index file when remote build falls back to local build [#3448](https://github.com/opensearch-project/k-NN/pull/3448)
* Fix radial search max_distance threshold conversion for inner product with memory-optimized search [#3369](https://github.com/opensearch-project/k-NN/pull/3369)
* Fix inner-hits returning mask value for top level source excludes [#3446](https://github.com/opensearch-project/k-NN/pull/3446)
* Fix _source bloat on merge for derived source indices with _source.exclude [#3465](https://github.com/opensearch-project/k-NN/pull/3465)
* Disable radial search on quantized indices due to poor recall from quantization error [#3448](https://github.com/opensearch-project/k-NN/pull/3448)

### Refactoring
* Refactor ExactSearcher to use BulkVectorScorer directly and rename factory methods [#3361](https://github.com/opensearch-project/k-NN/pull/3361)

### Enhancements
* Set decay value to experimentally found value 0.95 and use decay method in MOS [#3447](https://github.com/opensearch-project/k-NN/pull/3447)
* Support `index.knn.advanced.approximate_threshold` for the Lucene engine [#3451](https://github.com/opensearch-project/k-NN/pull/3451)
* Convert document vectors to primitive arrays once per document in lateInteractionScore, instead of once per query-vector/document-vector pair [#3453](https://github.com/opensearch-project/k-NN/pull/3453)
