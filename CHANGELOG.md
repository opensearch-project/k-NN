
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Bug Fixes
* Fix indexing for 16x and 8x compression [#3019](https://github.com/opensearch-project/k-NN/pull/3019)
### Enhancements
### Bug Fixes
* [BUGFIX] Fix KNN Quantization state cache have an invalid weight threshold [#2666](https://github.com/opensearch-project/k-NN/pull/2666)
* [BUGFIX] Fix enable rescoring when dimensions > 1000. [#2671](https://github.com/opensearch-project/k-NN/pull/2671)
* [BUGFIX] Honors slice counts for non-quantization cases [#2692](https://github.com/opensearch-project/k-NN/pull/2692)
### Infrastructure
### Documentation
### Maintenance
* Replace commons-lang with org.apache.commons:commons-lang3 [#2863](https://github.com/opensearch-project/k-NN/pull/2863)
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.19...2.x)
### Features
### Enhancements
### Bug Fixes
* Align field visibility across k-NN search paths ([#3659](https://github.com/opensearch-project/k-NN/pull/3659)).
* Fix the put mapping issue for already created index with flat mapper [#2542](https://github.com/opensearch-project/k-NN/pull/2542)
* [BUGFIX] Avoid opening of graph file if graph is already loaded in memory [#2719](https://github.com/opensearch-project/k-NN/pull/2719)
### Infrastructure
* Restore CI builds and rolling upgrades on 2.19 ([#3663](https://github.com/opensearch-project/k-NN/pull/3663)).
* Upgrade JaCoCo to 0.8.16 across projects to use ASM 9.11.1 [#3670](https://github.com/opensearch-project/k-NN/pull/3670)
### Documentation
### Maintenance
* Bump Guava from 32.1.3-jre to 33.8.0-jre and failureaccess from 1.0.1 to 1.0.3
* Bump OpenSearch to 2.19.8-SNAPSHOT to pick up Jackson 2.18.12 and zstd-jni 1.5.7-25 ([opensearch-project/OpenSearch#23629](https://github.com/opensearch-project/OpenSearch/pull/23629))
### Refactoring
