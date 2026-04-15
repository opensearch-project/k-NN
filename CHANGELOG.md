
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.7](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Add debug mode to MMR rerank that injects per-hit scoring details (original_score, max_similarity_to_selected, mmr_score, mmr_formula) into _source via the `debug` flag in the mmr search extension [#3254](https://github.com/opensearch-project/k-NN/pull/3254)
* Support derived source for knn with other fields [#3260](https://github.com/opensearch-project/k-NN/pull/3260)

### Maintenance


### Bug Fixes

### Refactoring


### Enhancements
* Make Merge in nativeEngine can Abort [#2529](https://github.com/opensearch-project/k-NN/pull/2529)
* Use pre-quantized vectors from native engines for exact search [#3095](https://github.com/opensearch-project/k-NN/pull/3095)
* Use right Vector Scorer when segments are initialized using SPI and also corrected the maxConn for MOS [#3117](https://github.com/opensearch-project/k-NN/pull/3117)
* Use pre-quantized vectors for ADC [#3113](https://github.com/opensearch-project/k-NN/pull/3113)
* Upgrade Lucene to 10.4.0 [#3135](https://github.com/opensearch-project/k-NN/pull/3135)
* Optimize ByteVectorIdsExactKNNIterator by moving array conversion to constructor [#3171](https://github.com/opensearch-project/k-NN/pull/3171)
* Enhance ADC scoring with SIMD Vector API and add comprehensive tests [#3167](https://github.com/opensearch-project/k-NN/pull/3167)
