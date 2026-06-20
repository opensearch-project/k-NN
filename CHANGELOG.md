
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.8](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Add rescoring phase after radial search on quantized index [#3347](https://github.com/opensearch-project/k-NN/pull/3347)
* Add base64 encoded vector indexing support for knn_vector fields [#3350](https://github.com/opensearch-project/k-NN/pull/3350)

### Maintenance

### Bug Fixes
* The KNN1030Codec does not properly support delegation for non-default codec(s). [#3093](https://github.com/opensearch-project/k-NN/pull/3093)
* Fix score conversion logic for radial exact search [#3110](https://github.com/opensearch-project/k-NN/pull/3110)
* Simplify DerivedSourceReaders lifecycle by removing manual ref-counting [#3138](https://github.com/opensearch-project/k-NN/pull/3138)
* Fix lucene reduce to topK when rescoring is enabled [#3124](https://github.com/opensearch-project/k-NN/pull/3124)
* Fix bugs in optimistic search for nested Cagra index [#3155](https://github.com/opensearch-project/k-NN/pull/3155)
* Fixed generating random entry points for CagraIndex in MOS when numVectors < entryPoints [#3161](https://github.com/opensearch-project/k-NN/pull/3161)
* Fix integer overflow for memory optimized search [#3130](https://github.com/opensearch-project/k-NN/pull/3130)
* Fix derived source returning incorrect vector value during indexing with dynamic templates [#3035](https://github.com/opensearch-project/k-NN/pull/3035)
* Fix FaissIdMap honor the given acceptOrds for sparse case. [#3196](https://github.com/opensearch-project/k-NN/pull/3196)
* Fix radial search bug returning 0 results for IndexHNSWCagra [#3201](https://github.com/opensearch-project/k-NN/pull/3201)
* Fix default encoder to SQ 1 bit for faiss 32x compression [#3210](https://github.com/opensearch-project/k-NN/pull/3210)
* Fix for prefetch failure due to out of bound exception [#3240](https://github.com/opensearch-project/k-NN/pull/3240)
* Fix copy_to functionality with vector fields [#3162](https://github.com/opensearch-project/k-NN/pull/3162)
* Turn off ACORN for MOS to match default Lucene HNSW behavior [#3346](https://github.com/opensearch-project/k-NN/pull/3346)
* Preserve mixed-case derived source vector field names and add backward-compatible field resolution for previously lowercased segment metadata [#3313](https://github.com/opensearch-project/k-NN/pull/3313)
* Fix rescore flag not propagating over transport layer in multi-node clusters [#3343](https://github.com/opensearch-project/k-NN/pull/3343)
* Integrated proper ef_search functionality into MOS and Lucene with oversample_factor [#3331](https://github.com/opensearch-project/k-NN/pull/3331)
* Fix skip warm up in old indices when MOS is enabled [#3344](https://github.com/opensearch-project/k-NN/pull/3344)
* Check to see if Lucene's search budget has exhausted when deciding to exact search in MOS [#3354](https://github.com/opensearch-project/k-NN/pull/3354)

### Refactoring


### Enhancements
