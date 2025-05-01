## Version 3.0.0.0 Release Notes

Compatible with OpenSearch 3.0.0

### Breaking Changes
* Remove ef construction from Index Seeting [#2564](https://github.com/opensearch-project/k-NN/pull/2564)
* Remove m from Index Setting [#2564](https://github.com/opensearch-project/k-NN/pull/2564)
* Remove space type from index setting [#2564](https://github.com/opensearch-project/k-NN/pull/2564)
* Remove Knn Plugin enabled setting [#2564](https://github.com/opensearch-project/k-NN/pull/2564)

### Features
* [Remote Vector Index Build] Client polling mechanism, encoder check, method parameter retrieval [#2576](https://github.com/opensearch-project/k-NN/pull/2576)
* [Remote Vector Index Build] Move client to separate module [#2603](https://github.com/opensearch-project/k-NN/pull/2603)
* Add filter function to KNNQueryBuilder with unit tests and integration tests [#2599](https://github.com/opensearch-project/k-NN/pull/2599)
* [Lucene On Faiss] Add a new mode, memory-optimized-search enable user to run vector search on FAISS index under memory constrained environment. [#2630](https://github.com/opensearch-project/k-NN/pull/2630)
* [Remote Vector Index Build] Add metric collection for remote build process [#2615](https://github.com/opensearch-project/k-NN/pull/2615)
* [Explain API Support] Added Explain API support for Exact/ANN/Radial/Disk based KNN search on Faiss Engine [#2403] (https://github.com/opensearch-project/k-NN/pull/2403)
* Introduce Remote Native Index Build feature flag, settings, and initial skeleton [#2525](https://github.com/opensearch-project/k-NN/pull/2525)
* Implement vector data upload and vector data size threshold setting [#2550](https://github.com/opensearch-project/k-NN/pull/2550)
* Implement data download and IndexOutput write functionality [#2554](https://github.com/opensearch-project/k-NN/pull/2554)
* Introduce Client Skeleton + basic Build Request implementation [#2560](https://github.com/opensearch-project/k-NN/pull/2560)
* Add concurrency optimizations with native memory graph loading and force eviction [#2345](https://github.com/opensearch-project/k-NN/pull/2345)
### Enhancements
* Introduce node level circuit breakers for k-NN [#2509](https://github.com/opensearch-project/k-NN/pull/2509)
* Added more detailed error messages for KNN model training [#2378](https://github.com/opensearch-project/k-NN/pull/2378)
### Bug Fixes
* Fixing bug to prevent NullPointerException while doing PUT mappings [#2556](https://github.com/opensearch-project/k-NN/issues/2556)
* Add index operation listener to update translog source [#2629](https://github.com/opensearch-project/k-NN/pull/2629)
* Add parent join support for faiss hnsw cagra [#2647](https://github.com/opensearch-project/k-NN/pull/2647)
* [Remote Vector Index Build] Fix bug to support `COSINESIMIL` space type [#2627](https://github.com/opensearch-project/k-NN/pull/2627)
* Disable doc value storage for vector field storage [#2646](https://github.com/opensearch-project/k-NN/pull/2646)
* Fix KNN Quantization state cache have an invalid weight threshold [#2666](https://github.com/opensearch-project/k-NN/pull/2666)
* Fix enable rescoring when dimensions > 1000. [#2671](https://github.com/opensearch-project/k-NN/pull/2671)
* Fix derived source for binary and byte vectors [#2533](https://github.com/opensearch-project/k-NN/pull/2533/)
* Fix the put mapping issue for already created index with flat mapper [#2542](https://github.com/opensearch-project/k-NN/pull/2542)
* Fixing the bug to prevent index.knn setting from being modified or removed on restore snapshot [#2445](https://github.com/opensearch-project/k-NN/pull/2445)
### Infrastructure
* Add github action to run ITs against remote index builder [2620](https://github.com/opensearch-project/k-NN/pull/2620)
* Removed JDK 11 and 17 version from CI runs [#1921](https://github.com/opensearch-project/k-NN/pull/1921)
* Upgrade min JDK compatibility to JDK 21 [#2422](https://github.com/opensearch-project/k-NN/pull/2422)
### Maintenance
* Update minimum required CMAKE version in NMSLIB [#2635](https://github.com/opensearch-project/k-NN/pull/2635)
* Revert CMake version bump, instead add CMake policy version flag to build task to support modern CMake builds [#2645](https://github.com/opensearch-project/k-NN/pull/2645/files)
* Update package name to fix compilation issue [#2513](https://github.com/opensearch-project/k-NN/pull/2513)
* Update gradle to 8.13 to fix command exec on java 21 [#2571](https://github.com/opensearch-project/k-NN/pull/2571)
* Add fix for nmslib pragma on arm [#2574](https://github.com/opensearch-project/k-NN/pull/2574)
* Removes Array based vector serialization [#2587](https://github.com/opensearch-project/k-NN/pull/2587)
* Enabled indices.breaker.total.use_real_memory setting via build.gradle for integTest Cluster to catch heap CB in local ITs and github CI actions [#2395](https://github.com/opensearch-project/k-NN/pull/2395/)
* Fixing Lucene912Codec Issue with BWC for Lucene 10.0.1 upgrade[#2429](https://github.com/opensearch-project/k-NN/pull/2429)
* Enabled idempotency of local builds when using `./gradlew clean` and nest `jni/release` directory under `jni/build` for easier cleanup [#2516](https://github.com/opensearch-project/k-NN/pull/2516)
### Refactoring
* Switch derived source from field attributes to segment attribute [#2606](https://github.com/opensearch-project/k-NN/pull/2606)
* Migrate derived source from filter to mask [#2612](https://github.com/opensearch-project/k-NN/pull/2612)
* Consolidate MethodFieldMapper and LuceneFieldMapper into EngineFieldMapper [#2646](https://github.com/opensearch-project/k-NN/pull/2646)
* Small Refactor Post Lucene 10.0.1 upgrade [#2541](https://github.com/opensearch-project/k-NN/pull/2541)
* Refactor codec to leverage backwards_codecs [#2546](https://github.com/opensearch-project/k-NN/pull/2546)
* Blocking Index Creation using NMSLIB [#2573](https://github.com/opensearch-project/k-NN/pull/2573)
* Improve Streaming Compatibility Issue for MethodComponetContext and Remove OpenDistro URL [#2575](https://github.com/opensearch-project/k-NN/pull/2575)
* 3.0.0 Breaking Changes For KNN [#2564](https://github.com/opensearch-project/k-NN/pull/2564)
