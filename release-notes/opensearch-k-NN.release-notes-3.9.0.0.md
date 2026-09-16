## Version 3.9.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.9.0

### Features
* Add Intel SVS (Scalable Vector Search) as a sandbox tenant engine with svs_vamana method, flat/sq/lvq/leanvec encoders, and query-time parameters ([#3551](https://github.com/opensearch-project/k-NN/pull/3551))
* Add multi-bit scalar quantization (2-bit and 4-bit) for Faiss, and make flat method engine-agnostic with x16/x8 support ([#3544](https://github.com/opensearch-project/k-NN/pull/3544))
* Add radial search method that applies post-filtering on quantized indices using size × oversample_factor top-K ([#3491](https://github.com/opensearch-project/k-NN/pull/3491))
* Introduce resolved index spec for centralized index configuration resolution ([#3499](https://github.com/opensearch-project/k-NN/pull/3499))
* Support `index.knn.advanced.approximate_threshold` for the Lucene engine to control HNSW graph construction ([#3451](https://github.com/opensearch-project/k-NN/pull/3451))

### Enhancements
* Add NEON SIMD kernel for FP16 L2 similarity with ~20% throughput improvement on ARM Graviton3 ([#3512](https://github.com/opensearch-project/k-NN/pull/3512))
* Add native SIMD cosine scoring for FP16 and SQ formats, eliminating post-hoc score conversion in the memory-optimized search path ([#3386](https://github.com/opensearch-project/k-NN/pull/3386))
* Add memory prefetching for Lucene engine's fp32 and binary vector HNSW query-scoring path to reduce cache-miss stalls ([#3504](https://github.com/opensearch-project/k-NN/pull/3504))
* Enable approximate graph threshold for Faiss SQ x32 to allow skipping HNSW graph construction for small segments ([#3434](https://github.com/opensearch-project/k-NN/pull/3434))
* Move document vector conversion out of the query-vector loop in late interaction scoring, significantly reducing allocations for ColBERT-style workloads ([#3453](https://github.com/opensearch-project/k-NN/pull/3453))
* Flip default compression for 16x and 8x to SQ 2-bit and 4-bit respectively ([#3561](https://github.com/opensearch-project/k-NN/pull/3561))
* Terminate remote index build early when the associated merge is aborted ([#3488](https://github.com/opensearch-project/k-NN/pull/3488))
* Skip warmup on warm indices since data is fetched on demand from remote store ([#3565](https://github.com/opensearch-project/k-NN/pull/3565))
* Add configurable nproc count to native library build script for parallel compilation ([#3539](https://github.com/opensearch-project/k-NN/pull/3539))

### Bug Fixes
* Fix `_source` bloat on merge when derived source is used with `_source.excludes`/`_source.includes` ([#3465](https://github.com/opensearch-project/k-NN/pull/3465))
* Fix derived source ingestion failure for non-JSON (CBOR/SMILE) encoded documents ([#3529](https://github.com/opensearch-project/k-NN/pull/3529))
* Fix dimension-based oversampling not applying for 32x BQ compression when shard-level rescoring is enabled ([#3460](https://github.com/opensearch-project/k-NN/pull/3460))
* Fix exact search scoring with L2 instead of the configured space type for model-based and pre-3.0 Faiss/Nmslib fields ([#3537](https://github.com/opensearch-project/k-NN/pull/3537))
* Fix k-NN query against a field alias silently returning zero hits ([#3485](https://github.com/opensearch-project/k-NN/pull/3485))
* Fix native thread leak in Lucene HNSW merge executor caused by unbounded thread pool accumulation ([#3533](https://github.com/opensearch-project/k-NN/pull/3533))
* Fix shared mutable `PerLeafResult.EMPTY_RESULT` singleton causing NPE across concurrent queries ([#3534](https://github.com/opensearch-project/k-NN/pull/3534))
* Fix `FileNotFoundException` on quantization state file when BQ segment has no live vectors ([#3511](https://github.com/opensearch-project/k-NN/pull/3511))
* Fix SQ flat prefetch reading `.veq` instead of `.vec` by dropping `HasIndexSlice` from `ScalarQuantizedFloatVectorValues` ([#3486](https://github.com/opensearch-project/k-NN/pull/3486))
* Preserve raw non-XContent `_source` fields (e.g., no-op tombstones) when derived source is enabled ([#3402](https://github.com/opensearch-project/k-NN/pull/3402))

### Infrastructure
* Stabilize Remote Index Build integration tests after libcuvs 26.06 upgrade by using graph-friendly test data ([#3557](https://github.com/opensearch-project/k-NN/pull/3557))
* Fix flaky BWC test `WarmupIT.testKNNWarmupCustomLegacyFieldMapping` ([#3419](https://github.com/opensearch-project/k-NN/pull/3419))

### Maintenance
* Clean up changelog after 3.8 release ([#3500](https://github.com/opensearch-project/k-NN/pull/3500))
* Fix multiple forbidden API warnings in the codebase for build log clarity ([#3507](https://github.com/opensearch-project/k-NN/pull/3507))
