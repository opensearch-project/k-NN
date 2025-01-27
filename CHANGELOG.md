
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
* Update package name to fix compilation issue [#2513](https://github.com/opensearch-project/k-NN/pull/2513)
### Refactoring
* Small Refactor Post Lucene 10.0.1 upgrade [#2541](https://github.com/opensearch-project/k-NN/pull/2541)
* Refactor codec to leverage backwards_codecs [#2546](https://github.com/opensearch-project/k-NN/pull/2546)

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
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
* Fixing the bug to prevent index.knn setting from being modified or removed on restore snapshot (#2445)[https://github.com/opensearch-project/k-NN/pull/2445]
* Fix derived source for binary and byte vectors [#2533](https://github.com/opensearch-project/k-NN/pull/2533/)
* Fix the put mapping issue for already created index with flat mapper [#2542](https://github.com/opensearch-project/k-NN/pull/2542)
### Infrastructure
### Documentation
### Maintenance
* Enabled indices.breaker.total.use_real_memory setting via build.gradle for integTest Cluster to catch heap CB in local ITs and github CI actions [#2395](https://github.com/opensearch-project/k-NN/pull/2395/) 
* Fixing Lucene912Codec Issue with BWC for Lucene 10.0.1 upgrade[#2429](https://github.com/opensearch-project/k-NN/pull/2429)
* Enabled idempotency of local builds when using `./gradlew clean` and nest `jni/release` directory under `jni/build` for easier cleanup [#2516](https://github.com/opensearch-project/k-NN/pull/2516)
### Refactoring
