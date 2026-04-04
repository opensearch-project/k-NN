## Version 3.6.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.6.0

### Features
* Add 1-bit compression support for the Lucene Scalar Quantizer (BBQ integration) ([#3144](https://github.com/opensearch-project/k-NN/pull/3144))
* Add Faiss Scalar Quantization 1-bit support with memory-optimized search, SIMD acceleration, and codec integration ([#3208](https://github.com/opensearch-project/k-NN/pull/3208))
* Add support for Lucene BBQ Flat format with 1-bit (32x) compression ([#3154](https://github.com/opensearch-project/k-NN/pull/3154))
* Add support for pre-quantized vector exact search to avoid redundant quantization during queries ([#3095](https://github.com/opensearch-project/k-NN/pull/3095))
* Add Hamming distance scorer for byte vectors in memory-optimized binary vector search ([#3214](https://github.com/opensearch-project/k-NN/pull/3214))
* Add NestedBestChildVectorScorer and KnnBinaryDocValuesScorer for exact search when Lucene's built-in scorers are unavailable ([#3179](https://github.com/opensearch-project/k-NN/pull/3179))
* Add prefetch functionality to prefetch vectors during ANN search for memory-optimized search ([#3173](https://github.com/opensearch-project/k-NN/pull/3173))
* Add scorer-aware ByteVectorValues wrapper for FAISS index to enable VectorScorer-driven scoring ([#3192](https://github.com/opensearch-project/k-NN/pull/3192))
* Introduce VectorScorers factory to create VectorScorer instances based on underlying vector storage format ([#3183](https://github.com/opensearch-project/k-NN/pull/3183))
* Use pre-quantized vectors for Asymmetric Distance Computation (ADC) to improve search performance ([#3113](https://github.com/opensearch-project/k-NN/pull/3113))
* Support native engine merge abort to prevent node stalls during shard relocation ([#2529](https://github.com/opensearch-project/k-NN/pull/2529))

### Enhancements
* Refactor ExactSearcher to use Lucene's VectorScorer API with bulk scoring instead of ExactKNNIterator ([#3207](https://github.com/opensearch-project/k-NN/pull/3207))
* Decouple native SIMD scoring selection from FaissMemoryOptimizedSearcher into FlatVectorsScorer decorator for broader scoring path coverage ([#3184](https://github.com/opensearch-project/k-NN/pull/3184))
* Integrate prefetch with FP16-based index for memory-optimized search including BulkSIMD support ([#3195](https://github.com/opensearch-project/k-NN/pull/3195))
* Integrate prefetch for SparseFloatVectorValues with Faiss indices (FP32, FP16, binary) ([#3197](https://github.com/opensearch-project/k-NN/pull/3197))
* Speed up FP16 bulk similarity by precomputing the tail mask, yielding up to 35% performance gain ([#3172](https://github.com/opensearch-project/k-NN/pull/3172))
* Optimize ByteVectorIdsExactKNNIterator by moving float-to-byte array conversion to constructor ([#3171](https://github.com/opensearch-project/k-NN/pull/3171))
* Use correct VectorScorer when segments are initialized via SPI and correct maxConn for memory-optimized search ([#3117](https://github.com/opensearch-project/k-NN/pull/3117))
* Adjust merge policy settings to make merges less aggressive, reducing CPU impact during concurrent search and merge ([#3128](https://github.com/opensearch-project/k-NN/pull/3128))
* Improve unit tests by tightening asserts for pre-quantized vector exact search ([#3112](https://github.com/opensearch-project/k-NN/pull/3112))

### Bug Fixes
* Fix derived source with dynamic templates causing vectors to be incorrectly returned during bulk indexing ([#3035](https://github.com/opensearch-project/k-NN/pull/3035))
* Fix FaissIdMap to honor given acceptOrds for sparse case by removing double ordinal-to-docID mapping ([#3196](https://github.com/opensearch-project/k-NN/pull/3196))
* Fix radial search returning 0 results for IndexHNSWCagra by adding proper range_search override ([#3201](https://github.com/opensearch-project/k-NN/pull/3201))
* Fix score conversion logic for filtered radial exact search with cosine space type ([#3110](https://github.com/opensearch-project/k-NN/pull/3110))
* Fix random entry point generation for CagraIndex in memory-optimized search when numVectors is less than entryPoints ([#3161](https://github.com/opensearch-project/k-NN/pull/3161))
* Fix memory-optimized search bugs on nested Cagra index including duplicate entry points and incorrect second deep-dive search behavior ([#3155](https://github.com/opensearch-project/k-NN/pull/3155))
* Fix Lucene reduce to topK when rescoring is enabled to prevent premature result reduction before rescoring phase ([#3124](https://github.com/opensearch-project/k-NN/pull/3124))
* Fix integer overflow for memory-optimized search when converting Faiss HNSW offsets from long to int ([#3130](https://github.com/opensearch-project/k-NN/pull/3130))
* Fix default encoder to SQ 1-bit for Faiss 32x compression ([#3210](https://github.com/opensearch-project/k-NN/pull/3210))
* Fix prefetch failure due to out-of-bound exception in FaissScorableByteVectorValues ([#3240](https://github.com/opensearch-project/k-NN/pull/3240))
* Fix KNN1030Codec to properly support delegation for non-default codecs on the read path ([#3093](https://github.com/opensearch-project/k-NN/pull/3093))

### Infrastructure
* Fix KNN build and run compatibility with Lucene 10.4.0 upgrade ([#3135](https://github.com/opensearch-project/k-NN/pull/3135))

### Maintenance
* Update changelog for 3.6.0 ([#3252](https://github.com/opensearch-project/k-NN/pull/3252))

### Refactoring
* Simplify DerivedSourceReaders lifecycle by removing manual ref-counting and aligning with Lucene's SegmentReader pattern ([#3138](https://github.com/opensearch-project/k-NN/pull/3138))
