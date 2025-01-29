
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
* Removed JDK 11 and 17 version from CI runs [#1921](https://github.com/opensearch-project/k-NN/pull/1921)
* Upgrade min JDK compatibility to JDK 21 [#2422](https://github.com/opensearch-project/k-NN/pull/2422)
### Documentation
### Maintenance
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.18...2.x)
### Features
- Add Support for Multi Values in innerHit for Nested k-NN Fields in Lucene and FAISS (#2283)[https://github.com/opensearch-project/k-NN/pull/2283]
- Add binary index support for Lucene engine. (#2292)[https://github.com/opensearch-project/k-NN/pull/2292]
- Add expand_nested_docs Parameter support to NMSLIB engine (#2331)[https://github.com/opensearch-project/k-NN/pull/2331]
- Add a new build mode, `FAISS_OPT_LEVEL=avx512_spr`, which enables the use of advanced AVX-512 instructions introduced with Intel(R) Sapphire Rapids (#2404)[https://github.com/opensearch-project/k-NN/pull/2404]
- Add cosine similarity support for faiss engine (#2376)[https://github.com/opensearch-project/k-NN/pull/2376]
- Add derived source feature for vector fields (#2449)[https://github.com/opensearch-project/k-NN/pull/2449]
### Enhancements
- Introduced a writing layer in native engines where relies on the writing interface to process IO. (#2241)[https://github.com/opensearch-project/k-NN/pull/2241]
- Allow method parameter override for training based indices (#2290) https://github.com/opensearch-project/k-NN/pull/2290]
- Optimizes lucene query execution to prevent unnecessary rewrites (#2305)[https://github.com/opensearch-project/k-NN/pull/2305]
- Add check to directly use ANN Search when filters match all docs. (#2320)[https://github.com/opensearch-project/k-NN/pull/2320]
- Use one formula to calculate cosine similarity (#2357)[https://github.com/opensearch-project/k-NN/pull/2357]
- Add WithFieldName implementation to KNNQueryBuilder (#2398)[https://github.com/opensearch-project/k-NN/pull/2398]
- Make the build work for M series MacOS without manual code changes and local JAVA_HOME config (#2397)[https://github.com/opensearch-project/k-NN/pull/2397]
- Remove DocsWithFieldSet reference from NativeEngineFieldVectorsWriter (#2408)[https://github.com/opensearch-project/k-NN/pull/2408]
### Bug Fixes
* Fixing the bug when a segment has no vector field present for disk based vector search (#2282)[https://github.com/opensearch-project/k-NN/pull/2282]
* Fixing the bug where search fails with "fields" parameter for an index with a knn_vector field (#2314)[https://github.com/opensearch-project/k-NN/pull/2314]
* Fix for NPE while merging segments after all the vector fields docs are deleted (#2365)[https://github.com/opensearch-project/k-NN/pull/2365]
* Allow validation for non knn index only after 2.17.0 (#2315)[https://github.com/opensearch-project/k-NN/pull/2315]
* Fixing the bug to prevent updating the index.knn setting after index creation(#2348)[https://github.com/opensearch-project/k-NN/pull/2348]
* Release query vector memory after execution (#2346)[https://github.com/opensearch-project/k-NN/pull/2346]
* Fix shard level rescoring disabled setting flag (#2352)[https://github.com/opensearch-project/k-NN/pull/2352]
* Fix filter rewrite logic which was resulting in getting inconsistent / incorrect results for cases where filter was getting rewritten for shards (#2359)[https://github.com/opensearch-project/k-NN/pull/2359]
* Fixing it to retrieve space_type from index setting when both method and top level don't have the value. [#2374](https://github.com/opensearch-project/k-NN/pull/2374)
* Fixing the bug where setting rescore as false for on_disk knn_vector query is a no-op (#2399)[https://github.com/opensearch-project/k-NN/pull/2399]
* Fixing bug where mapping accepts both dimension and model-id (#2410)[https://github.com/opensearch-project/k-NN/pull/2410]
### Infrastructure
* Updated C++ version in JNI from c++11 to c++17 [#2259](https://github.com/opensearch-project/k-NN/pull/2259)
* Upgrade bytebuddy and objenesis version to match OpenSearch core and, update github ci runner for macos [#2279](https://github.com/opensearch-project/k-NN/pull/2279)
### Documentation
### Maintenance
* Select index settings based on cluster version[2236](https://github.com/opensearch-project/k-NN/pull/2236)
* Added periodic cache maintenance for QuantizationStateCache and NativeMemoryCache [#2308](https://github.com/opensearch-project/k-NN/pull/2308)
* Added null checks for fieldInfo in ExactSearcher to avoid NPE while running exact search for segments with no vector field (#2278)[https://github.com/opensearch-project/k-NN/pull/2278]
* Added Lucene BWC tests (#2313)[https://github.com/opensearch-project/k-NN/pull/2313]
* Upgrade jsonpath from 2.8.0 to 2.9.0[2325](https://github.com/opensearch-project/k-NN/pull/2325)
* Bump Faiss commit from 1f42e81 to 0cbc2a8 to accelerate hamming distance calculation using _mm512_popcnt_epi64  intrinsic and also add avx512-fp16 instructions to boost performance [#2381](https://github.com/opensearch-project/k-NN/pull/2381)
* Enabled indices.breaker.total.use_real_memory setting via build.gradle for integTest Cluster to catch heap CB in local ITs and github CI actions [#2395](https://github.com/opensearch-project/k-NN/pull/2395/) 
* Fixing Lucene912Codec Issue with BWC for Lucene 10.0.1 upgrade[#2429](https://github.com/opensearch-project/k-NN/pull/2429)
### Refactoring
