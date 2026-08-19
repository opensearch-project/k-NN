
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.9](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Enable the approximate graph threshold for Faiss SQ x32 (sq bits=1) indices [#3434](https://github.com/opensearch-project/k-NN/pull/3434)

### Maintenance
* Fixed multiple forbidden api warnings from the code []()

### Bug Fixes
* Fix knn query against a field alias returning zero hits silently [#3485](https://github.com/opensearch-project/k-NN/pull/3485)
* Fix when BQ file is not present in the segment as there is no vectors in the segment []()

### Refactoring
* Wire ResolvedIndexSpec consumers through spec-driven resolution flow [#3421](https://github.com/opensearch-project/k-NN/pull/3421)
* Refactor engine field mapper, deprecate mode parameter, add encoder validation [#3436](https://github.com/opensearch-project/k-NN/pull/3436)
* Centralize rescore and MOS logic in ResolvedIndexSpec [#3466](https://github.com/opensearch-project/k-NN/pull/3466)

### Enhancements
* Support `index.knn.advanced.approximate_threshold` for the Lucene engine [#3451](https://github.com/opensearch-project/k-NN/pull/3451)
* Convert document vectors to primitive arrays once per document in lateInteractionScore, instead of once per query-vector/document-vector pair [#3453](https://github.com/opensearch-project/k-NN/pull/3453)
* Terminate remote index build early when the merge has been aborted [#3488](https://github.com/opensearch-project/k-NN/pull/3488)
