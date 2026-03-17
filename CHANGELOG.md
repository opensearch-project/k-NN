
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.6](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features
* Support Lucene BBQ Flat for 1 bit [#3154](https://github.com/opensearch-project/k-NN/pull/3154)

### Maintenance
* Improve unit tests by tightening asserts [#3112](https://github.com/opensearch-project/k-NN/pull/3112)

### Bug Fixes
* The KNN1030Codec does not properly support delegation for non-default codec(s). [#3093](https://github.com/opensearch-project/k-NN/pull/3093)
* Fix score conversion logic for radial exact search [#3110](https://github.com/opensearch-project/k-NN/pull/3110)
* Fix lucene reduce to topK when rescoring is enabled [#3124](https://github.com/opensearch-project/k-NN/pull/3124)
* Fix bugs in optimistic search for nested Cagra index [#3155](https://github.com/opensearch-project/k-NN/pull/3155)
* Fixed generating random entry points for CagraIndex in MOS when numVectors < entryPoints [#3161](https://github.com/opensearch-project/k-NN/pull/3161)
* Fix integer overflow for memory optimized search [#3130](https://github.com/opensearch-project/k-NN/pull/3130)
* Fix derived source returning incorrect vector value during indexing with dynamic templates [#3035](https://github.com/opensearch-project/k-NN/pull/3035)

### Refactoring

### Enhancements
* Make Merge in nativeEngine can Abort [#2529](https://github.com/opensearch-project/k-NN/pull/2529)
* Use pre-quantized vectors from native engines for exact search [#3095](https://github.com/opensearch-project/k-NN/pull/3095)
* Use right Vector Scorer when segments are initialized using SPI and also corrected the maxConn for MOS [#3117](https://github.com/opensearch-project/k-NN/pull/3117)
* Use pre-quantized vectors for ADC [#3113](https://github.com/opensearch-project/k-NN/pull/3113)
* Upgrade Lucene to 10.4.0 [#3135](https://github.com/opensearch-project/k-NN/pull/3135)
* Speedup FP16 bulk similarity by precomputing the tail mask [#3172](https://github.com/opensearch-project/k-NN/pull/3172)
* Optimize ByteVectorIdsExactKNNIterator by moving array conversion to constructor [#3171](https://github.com/opensearch-project/k-NN/pull/3171)
