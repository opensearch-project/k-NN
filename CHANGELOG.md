
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.9](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Enable the approximate graph threshold for Faiss SQ x32 (sq bits=1) indices [#3434](https://github.com/opensearch-project/k-NN/pull/3434)
* Accept SQ 2-bit and 4-bit quantization at the mapping and codec layers [#3429](https://github.com/opensearch-project/k-NN/pull/3429)
* Build SQ B-bit HNSW graph with multi-bit symmetric distance for SQ bits ∈ {1, 2, 4} [#3431](https://github.com/opensearch-project/k-NN/pull/3431
* Enable remote vector index build for multi-bit SQ - bits ∈ {2, 4} [#3459](https://github.com/opensearch-project/k-NN/pull/3459)
* Set default oversample factor to 1 for SQ 2-bit and 4-bit encoders (x16 / x8 compression) [#3463](https://github.com/opensearch-project/k-NN/pull/3463)
* Support flat with x8 and x16 compression and make `method=flat` engine-agnostic [#3471](https://github.com/opensearch-project/k-NN/pull/3471)
* Add Intel SVS (Scalable Vector Search) as a sandbox tenant engine: `svs_vamana` with flat/sq/lvq/leanvec encoders [#XXXX](https://github.com/opensearch-project/k-NN/pull/XXXX)

### Maintenance
* Fixed multiple forbidden api warnings from the code []()

### Bug Fixes
* Fix knn query against a field alias returning zero hits silently [#3485](https://github.com/opensearch-project/k-NN/pull/3485)
* Add prefetch for Lucene engine's fp32 and binary vector data type [#3504](https://github.com/opensearch-project/k-NN/pull/3504)
* Fix when BQ file is not present in the segment as there is no vectors in the segment [#3511](https://github.com/opensearch-project/k-NN/pull/3511)
* Fix shared mutable PerLeafResult.EMPTY_RESULT causing NPE [#3534](https://github.com/opensearch-project/k-NN/pull/3534)
* Drop HasIndexSlice from ScalarQuantizedFloatVectorValues and expose float/quantized delegates via getters [#3486](https://github.com/opensearch-project/k-NN/pull/3486)
* Fix exact search and rescore scoring innerproduct and cosinesimil fields with L2 on model based and 2.17 to 2.19 indices [#3537](https://github.com/opensearch-project/k-NN/pull/3537)

### Refactoring
* Wire ResolvedIndexSpec consumers through spec-driven resolution flow [#3421](https://github.com/opensearch-project/k-NN/pull/3421)
* Refactor engine field mapper, deprecate mode parameter, add encoder validation [#3436](https://github.com/opensearch-project/k-NN/pull/3436)
* Centralize rescore and MOS logic in ResolvedIndexSpec [#3466](https://github.com/opensearch-project/k-NN/pull/3466)
* Add ScalarEncodingResolver and parameterize Faiss SQ format by encoding to unblock multi-bit SQ support [#3428](https://github.com/opensearch-project/k-NN/pull/3428)

### Enhancements
* Support `index.knn.advanced.approximate_threshold` for the Lucene engine [#3451](https://github.com/opensearch-project/k-NN/pull/3451)
* Convert document vectors to primitive arrays once per document in lateInteractionScore, instead of once per query-vector/document-vector pair [#3453](https://github.com/opensearch-project/k-NN/pull/3453)
* Terminate remote index build early when the merge has been aborted [#3488](https://github.com/opensearch-project/k-NN/pull/3488)
* Add NEON SIMD kernel for FP16 L2 similarity [#3512](https://github.com/opensearch-project/k-NN/pull/3512)
