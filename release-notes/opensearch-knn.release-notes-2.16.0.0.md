## Version 2.16.0.0 Release Notes

Compatible with OpenSearch 2.16.0

### Features
* Adds dynamic query parameter ef_search [#1783](https://github.com/opensearch-project/k-NN/pull/1783)
* Adds dynamic query parameter ef_search in radial search faiss engine [#1790](https://github.com/opensearch-project/k-NN/pull/1790)
* Adds dynamic query parameter nprobes [#1792](https://github.com/opensearch-project/k-NN/pull/1792)
* Add binary format support with HNSW method in Faiss Engine [#1781](https://github.com/opensearch-project/k-NN/pull/1781)
* Add script scoring support for knn field with binary data type [#1826](https://github.com/opensearch-project/k-NN/pull/1826)
* Add painless script support for hamming with binary vector data type [#1839](https://github.com/opensearch-project/k-NN/pull/1839)
* Add binary format support with IVF method in Faiss Engine [#1784](https://github.com/opensearch-project/k-NN/pull/1784)
* Add support for Lucene inbuilt Scalar Quantizer [#1848](https://github.com/opensearch-project/k-NN/pull/1848)
### Enhancements
* Switch from byte stream to byte ref for serde [#1825](https://github.com/opensearch-project/k-NN/pull/1825)
### Bug Fixes
* Fixing the arithmetic to find the number of vectors to stream from java to jni layer.[#1804](https://github.com/opensearch-project/k-NN/pull/1804)
* Fixed LeafReaders casting errors to SegmentReaders when segment replication is enabled during search.[#1808](https://github.com/opensearch-project/k-NN/pull/1808)
* Release memory properly for an array type [#1820](https://github.com/opensearch-project/k-NN/pull/1820)
* FIX Same Suffix Cause Recall Drop to zero [#1802](https://github.com/opensearch-project/k-NN/pull/1802)
### Infrastructure
* Apply custom patch only once by comparing the last patch id  [#1833](https://github.com/opensearch-project/k-NN/pull/1833)
### Documentation
* Update dev guide to fix clang linking issue on arm [#1746](https://github.com/opensearch-project/k-NN/pull/1746)
### Maintenance
* Bump faiss commit to 33c0ba5 [#1796](https://github.com/opensearch-project/k-NN/pull/1796)