
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.6](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Support Lucene SQ Flat for 1 bit [#3154](https://github.com/opensearch-project/k-NN/pull/3154)
* Add 32x support for SQ encoder on Faiss [#3193](https://github.com/opensearch-project/k-NN/pull/3193)
* Faiss SQ 1bit MOS changes [#3182](https://github.com/opensearch-project/k-NN/pull/3182)
* Support compression to 1 bit for Lucene's scalar quantizer [#3144](https://github.com/opensearch-project/k-NN/pull/3144)

### Maintenance
* Improve unit tests by tightening asserts [#3112](https://github.com/opensearch-project/k-NN/pull/3112)

### Bug Fixes
* The KNN1030Codec does not properly support delegation for non-default codec(s). [#3093](https://github.com/opensearch-project/k-NN/pull/3093)
* Fix thread leak in HNSW merge executor caused by per-call thread pool creation [#3120](https://github.com/opensearch-project/k-NN/pull/3120) 
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

### Refactoring
* Refactor ExactSearcher to use VectorScorer instead of ExactKNNIterator [#3207](https://github.com/opensearch-project/k-NN/pull/3207)

### Enhancements
* Make Merge in nativeEngine can Abort [#2529](https://github.com/opensearch-project/k-NN/pull/2529)
* Use pre-quantized vectors from native engines for exact search [#3095](https://github.com/opensearch-project/k-NN/pull/3095)
* Use right Vector Scorer when segments are initialized using SPI and also corrected the maxConn for MOS [#3117](https://github.com/opensearch-project/k-NN/pull/3117)
* Use pre-quantized vectors for ADC [#3113](https://github.com/opensearch-project/k-NN/pull/3113)
* Adjusting the merge policy setting to make merges less aggressive [#3128](https://github.com/opensearch-project/k-NN/pull/3128)
* Upgrade Lucene to 10.4.0 [#3135](https://github.com/opensearch-project/k-NN/pull/3135)
* Speedup FP16 bulk similarity by precomputing the tail mask [#3172](https://github.com/opensearch-project/k-NN/pull/3172)
* Add Prefetch functionality to prefetch vectors during ANN Search for MemoryOptimizedSearch. [#3173](https://github.com/opensearch-project/k-NN/pull/3173)
* Add Prefetch functionality to Fp16 based indices during ANN Search for MemoryOptimizedSearch. [#3195](https://github.com/opensearch-project/k-NN/pull/3195)
* Add Prefetch functionality to SparseFloatVectorValues with Faiss Indices [#]()
* Optimize ByteVectorIdsExactKNNIterator by moving array conversion to constructor [#3171](https://github.com/opensearch-project/k-NN/pull/3171)
* Add VectorScorers for BinaryDocValues and nested best child scoring [#3179](https://github.com/opensearch-project/k-NN/pull/3179)
* Introduce NativeEngines990KnnVectorsScorer to decouple native SIMD scoring selection from FaissMemoryOptimizedSearcher [#3184](https://github.com/opensearch-project/k-NN/pull/3184)
* Add scorer-aware ByteVectorValues wrapper for FAISS Index [#3192](https://github.com/opensearch-project/k-NN/pull/3192)
* Add Hamming distance scorer for byte vectors in VectorScorers [#3214](https://github.com/opensearch-project/k-NN/pull/3214)
* Introduce VectorScorers to create VectorScorer instances based on the underlying vector storage format [#3183](https://github.com/opensearch-project/k-NN/pull/3183)
