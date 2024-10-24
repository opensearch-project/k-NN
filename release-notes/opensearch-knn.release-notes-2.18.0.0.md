## Version 2.18.0.0 Release Notes

Compatible with OpenSearch 2.18.0

### Features
* Add AVX512 support to k-NN for FAISS library [#2069](https://github.com/opensearch-project/k-NN/pull/2069)
### Enhancements
* Introducing a loading layer in FAISS [#2033](https://github.com/opensearch-project/k-NN/issues/2033)
* Add short circuit if no live docs are in segments [#2059](https://github.com/opensearch-project/k-NN/pull/2059)
* Optimize reduceToTopK in ResultUtil by removing pre-filling and reducing peek calls [#2146](https://github.com/opensearch-project/k-NN/pull/2146)
* Update Default Rescore Context based on Dimension [#2149](https://github.com/opensearch-project/k-NN/pull/2149)
* KNNIterators should support with and without filters [#2155](https://github.com/opensearch-project/k-NN/pull/2155)
* Adding Support to Enable/Disble Share level Rescoring and Update Oversampling Factor[#2172](https://github.com/opensearch-project/k-NN/pull/2172)
* Add support to build vector data structures greedily and perform exact search when there are no engine files [#1942](https://github.com/opensearch-project/k-NN/issues/1942)
* Add CompressionLevel Calculation for PQ [#2200](https://github.com/opensearch-project/k-NN/pull/2200)
* Remove FSDirectory dependency from native engine constructing side and deprecated FileWatcher [#2182](https://github.com/opensearch-project/k-NN/pull/2182)
* Update approximate_threshold to 15K documents [#2229](https://github.com/opensearch-project/k-NN/pull/2229)
* Update default engine to FAISS [#2221](https://github.com/opensearch-project/k-NN/pull/2221)
### Bug Fixes
* Add DocValuesProducers for releasing memory when close index [#1946](https://github.com/opensearch-project/k-NN/pull/1946)
* KNN80DocValues should only be considered for BinaryDocValues fields [#2147](https://github.com/opensearch-project/k-NN/pull/2147)
* Score Fix for Binary Quantized Vector and Setting Default value in case of shard level rescoring is disabled for oversampling factor[#2183](https://github.com/opensearch-project/k-NN/pull/2183)
* Java Docs Fix For 2.x[#2190](https://github.com/opensearch-project/k-NN/pull/2190)
### Documentation
* Fix sed command in DEVELOPER_GUIDE.md to append a new line character '\n'. [#2181](https://github.com/opensearch-project/k-NN/pull/2181)
### Maintenance
* Remove benchmarks folder from k-NN repo [#2127](https://github.com/opensearch-project/k-NN/pull/2127)
* Fix lucene codec after lucene version bumped to 9.12. [#2195](https://github.com/opensearch-project/k-NN/pull/2195)
### Refactoring
* Does not create additional KNNVectorValues in NativeEngines990KNNVectorWriter when quantization is not needed [#2133](https://github.com/opensearch-project/k-NN/pull/2133)
* Minor refactoring and refactored some unit test [#2167](https://github.com/opensearch-project/k-NN/pull/2167)
