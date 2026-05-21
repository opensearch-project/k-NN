
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.7](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Add debug mode to MMR rerank that injects per-hit scoring details (original_score, max_similarity_to_selected, mmr_score, mmr_formula) into _source via the `debug` flag in the mmr search extension [#3254](https://github.com/opensearch-project/k-NN/pull/3254)
* Support derived source for knn with other fields [#3260](https://github.com/opensearch-project/k-NN/pull/3260)
* Added support for 1 bit SQ with remote build [#3270](https://github.com/opensearch-project/k-NN/pull/3270)

### Maintenance
* Update Gradle wrapper to 9.4.1 and Jacoco to 0.8.14 to match core OpenSearch [#3308](https://github.com/opensearch-project/k-NN/pull/3308)

### Bug Fixes
* Use KNN1040ScalarQuantizedVectorsFormat for Faiss SQ flat format to enable prefetch [#3302](https://github.com/opensearch-project/k-NN/pull/3302)

### Refactoring


### Enhancements
* Add the bulkscore logic in MOS when K is greater than number of docs in segment [#3285](https://github.com/opensearch-project/k-NN/pull/3285)
* Added capability to retrieve float data type vectors using doc_values [#3321](https://github.com/opensearch-project/k-NN/pull/3321)
* Add base64 binary encoding as default format for knn_vector docvalue_fields [#3324](https://github.com/opensearch-project/k-NN/pull/3324)
