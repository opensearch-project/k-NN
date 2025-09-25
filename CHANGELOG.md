
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.3](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Support native Maximal Marginal Relevance [#2868](https://github.com/opensearch-project/k-NN/pull/2868)
### Maintenance
* Replace commons-lang with org.apache.commons:commons-lang3 [#2863](https://github.com/opensearch-project/k-NN/pull/2863)
* Bump OpenSearch-Protobufs to 0.13.0 [#2833](https://github.com/opensearch-project/k-NN/pull/2833)
* Bump Lucene version to 10.3 and fix build failures [#2878](https://github.com/opensearch-project/k-NN/pull/2878)

### Bug Fixes
* Use queryVector length if present in MDC check [#2867](https://github.com/opensearch-project/k-NN/pull/2867)
* Fix derived source deserialization bug on invalid documents [#2882](https://github.com/opensearch-project/k-NN/pull/2882)

### Refactoring
* Refactored the KNN Stat files for better readability.

### Enhancements
* Added engine as a top-level optional parameter while creating vector field [#2736](https://github.com/opensearch-project/k-NN/pull/2736)
* Migrate k-NN plugin to use GRPC transport-grpc SPI interface [#2833](https://github.com/opensearch-project/k-NN/pull/2833)
