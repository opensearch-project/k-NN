
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

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
* Enabled indices.breaker.total.use_real_memory setting via build.gradle for integTest Cluster to catch heap CB in local ITs and github CI actions [#2395](https://github.com/opensearch-project/k-NN/pull/2395/) 
* Fixing Lucene912Codec Issue with BWC for Lucene 10.0.1 upgrade[#2429](https://github.com/opensearch-project/k-NN/pull/2429)
* Enabled idempotency of local builds when using `./gradlew clean` and nest `jni/release` directory under `jni/build` for easier cleanup [#2516](https://github.com/opensearch-project/k-NN/pull/2516)
### Refactoring