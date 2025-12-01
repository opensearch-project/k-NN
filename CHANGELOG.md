
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.4](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Memory optimized search warmup ([#2954](https://github.com/opensearch-project/k-NN/pull/2954))
### Maintenance
* Onboard to s3 snapshots ([#2943](https://github.com/opensearch-project/k-NN/pull/2943))
* Gradle 9.2.0 and GitHub Actions JDK 25 Upgrade ([#2984](https://github.com/opensearch-project/k-NN/pull/2984))

### Bug Fixes
* Fix blocking old indices created before 2.18 to use memory optimized search. [#2918](https://github.com/opensearch-project/k-NN/pull/2918)
* Fix NativeEngineKnnQuery to return all part results for valid totalHits in response [#2965](https://github.com/opensearch-project/k-NN/pull/2965)
* Fix unsafe concurrent update query vector in KNNQueryBuilder [#2974](https://github.com/opensearch-project/k-NN/pull/2974)
* Fix Backwards Compatability on Segment Merge for Disk-Based vector search [#2987](https://github.com/opensearch-project/k-NN/pull/2987) 

### Refactoring
* Refactor to not use parallel for MMR rerank. [#2968](https://github.com/opensearch-project/k-NN/pull/2968)

### Enhancements
* Removed VectorSearchHolders map from NativeEngines990KnnVectorsReader [#2948](https://github.com/opensearch-project/k-NN/pull/2948)
* Native scoring for FP16 [#2922](https://github.com/opensearch-project/k-NN/pull/2922)
