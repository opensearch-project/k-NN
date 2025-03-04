
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
* [Remote Vector Index Build] Introduce Remote Native Index Build feature flag, settings, and initial skeleton [#2525](https://github.com/opensearch-project/k-NN/pull/2525)
* [Remote Vector Index Build] Implement vector data upload and vector data size threshold setting [#2550](https://github.com/opensearch-project/k-NN/pull/2550)
* [Remote Vector Index Build] Implement data download and IndexOutput write functionality [#2554](https://github.com/opensearch-project/k-NN/pull/2554)
* [Remote Vector Index Build] Introduce Client Skeleton + basic Build Request implementation [#2560](https://github.com/opensearch-project/k-NN/pull/2560)
* Add concurrency optimizations with native memory graph loading and force eviction (#2265) [https://github.com/opensearch-project/k-NN/pull/2345]
### Enhancements
* Introduce node level circuit breakers for k-NN [#2509](https://github.com/opensearch-project/k-NN/pull/2509)
### Bug Fixes
### Infrastructure
* Removed JDK 11 and 17 version from CI runs [#1921](https://github.com/opensearch-project/k-NN/pull/1921)
* Upgrade min JDK compatibility to JDK 21 [#2422](https://github.com/opensearch-project/k-NN/pull/2422)
### Documentation
### Maintenance
* Update package name to fix compilation issue [#2513](https://github.com/opensearch-project/k-NN/pull/2513)
* Update gradle to 8.13 to fix command exec on java 21 [#2571](https://github.com/opensearch-project/k-NN/pull/2571)
* Add fix for nmslib pragma on arm [#2574](https://github.com/opensearch-project/k-NN/pull/2574)
### Refactoring
* Small Refactor Post Lucene 10.0.1 upgrade [#2541](https://github.com/opensearch-project/k-NN/pull/2541)
* Refactor codec to leverage backwards_codecs [#2546](https://github.com/opensearch-project/k-NN/pull/2546)
* Remove usage of cluster level setting for circuit breaker [#2567](https://github.com/opensearch-project/k-NN/pull/2567)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
- Added more detailed error messages for KNN model training (#2378)[https://github.com/opensearch-project/k-NN/pull/2378]
### Bug Fixes
* Fix derived source for binary and byte vectors [#2533](https://github.com/opensearch-project/k-NN/pull/2533/)
* Fix the put mapping issue for already created index with flat mapper [#2542](https://github.com/opensearch-project/k-NN/pull/2542)
* Fixing the bug to prevent index.knn setting from being modified or removed on restore snapshot (#2445)[https://github.com/opensearch-project/k-NN/pull/2445]
### Infrastructure
### Documentation
### Maintenance
* Enabled indices.breaker.total.use_real_memory setting via build.gradle for integTest Cluster to catch heap CB in local ITs and github CI actions [#2395](https://github.com/opensearch-project/k-NN/pull/2395/) 
* Fixing Lucene912Codec Issue with BWC for Lucene 10.0.1 upgrade[#2429](https://github.com/opensearch-project/k-NN/pull/2429)
* Enabled idempotency of local builds when using `./gradlew clean` and nest `jni/release` directory under `jni/build` for easier cleanup [#2516](https://github.com/opensearch-project/k-NN/pull/2516)
### Refactoring
