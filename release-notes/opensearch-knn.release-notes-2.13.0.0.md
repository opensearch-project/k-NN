## Version 2.13.0.0 Release Notes

Compatible with OpenSearch 2.13.0

### Enhancements
* Optize Faiss Query With Filters: Reduce iteration and memory for id filter [#1402](https://github.com/opensearch-project/k-NN/pull/1402)
* Detect AVX2 Dynamically on the System [#1502](https://github.com/opensearch-project/k-NN/pull/1502)
* Validate zero vector when using cosine metric [#1501](https://github.com/opensearch-project/k-NN/pull/1501)
* Persist model definition in model metadata [#1527] (https://github.com/opensearch-project/k-NN/pull/1527)
* Added Inner Product Space type support for Lucene Engine [#1551](https://github.com/opensearch-project/k-NN/pull/1551)
* Add Range Validation for Faiss SQFP16 [#1493](https://github.com/opensearch-project/k-NN/pull/1493)
* SQFP16 Range Validation for Faiss IVF Models [#1557](https://github.com/opensearch-project/k-NN/pull/1557)
### Bug Fixes
* Disable sdc table for HNSWPQ read-only indices [#1518](https://github.com/opensearch-project/k-NN/pull/1518)
* Switch SpaceType.INNERPRODUCT's vector similarity function to MAXIMUM_INNER_PRODUCT [#1532](https://github.com/opensearch-project/k-NN/pull/1532)
* Add patch to fix arm segfault in nmslib during ingestion [#1541](https://github.com/opensearch-project/k-NN/pull/1541)
* Share ivfpq-l2 table allocations across indices on load [#1558](https://github.com/opensearch-project/k-NN/pull/1558)
### Infrastructure
* Manually install zlib for win CI [#1513](https://github.com/opensearch-project/k-NN/pull/1513)
* Update k-NN build artifact script to enable SIMD on ARM for Faiss [#1543](https://github.com/opensearch-project/k-NN/pull/1543)
### Maintenance
* Bump faiss lib commit to 32f0e8cf92cd2275b60364517bb1cce67aa29a55 [#1443](https://github.com/opensearch-project/k-NN/pull/1443)
* Fix FieldInfo Parameters Mismatch [#1490](https://github.com/opensearch-project/k-NN/pull/1490)
* Upgrade faiss to 12b92e9 [#1509](https://github.com/opensearch-project/k-NN/pull/1509)