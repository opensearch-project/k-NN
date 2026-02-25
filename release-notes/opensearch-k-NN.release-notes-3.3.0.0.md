## Version 3.3.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.3.0

### Features
* Support native Maximal Marginal Relevance ([#2868](https://github.com/opensearch-project/k-NN/pull/2868))
* Support lateInteraction feature using painess script ([#2909](https://github.com/opensearch-project/k-NN/pull/2909))

### Maintenance
* Replace commons-lang with org.apache.commons:commons-lang3 ([#2863](https://github.com/opensearch-project/k-NN/pull/2863))
* Bump OpenSearch-Protobufs to 0.13.0 ([#2833](https://github.com/opensearch-project/k-NN/pull/2833))
* Bump Lucene version to 10.3 and fix build failures ([#2878](https://github.com/opensearch-project/k-NN/pull/2878))

### Bug Fixes
* Use queryVector length if present in MDC check ([#2867](https://github.com/opensearch-project/k-NN/pull/2867))
* Fix derived source deserialization bug on invalid documents ([#2882](https://github.com/opensearch-project/k-NN/pull/2882))
* Fix invalid cosine score range in LuceneOnFaiss ([#2892](https://github.com/opensearch-project/k-NN/pull/2892))
* Allows k to be nullable to fix filter bug ([#2836](https://github.com/opensearch-project/k-NN/issues/2836))
* Fix integer overflow for while estimating distance computations for efficient filtering ([#2903](https://github.com/opensearch-project/k-NN/pull/2903))
* Fix AVX2 detection on other platforms ([#2912](https://github.com/opensearch-project/k-NN/pull/2912))
* Fix byte[] radial search for faiss ([#2905](https://github.com/opensearch-project/k-NN/pull/2905))
* Use the unique doc id for MMR rerank rather than internal lucenue doc id which is not unique for multiple shards case. ([#2911](https://github.com/opensearch-project/k-NN/pull/2911))
* Fix local ref leak in JNI ([#2916](https://github.com/opensearch-project/k-NN/pull/2916))
* Fix rescoring logic for nested exact search ([#2921](https://github.com/opensearch-project/k-NN/pull/2921))

### Refactoring
* Refactored the KNN Stat files for better readability.

### Enhancements
* Added engine as a top-level optional parameter while creating vector field ([#2736](https://github.com/opensearch-project/k-NN/pull/2736))
* Migrate k-NN plugin to use GRPC transport-grpc SPI interface ([#2833](https://github.com/opensearch-project/k-NN/pull/2833))