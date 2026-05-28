## Version 3.7.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.7.0

### Features

* Add base64 binary encoding as default format for knn_vector docvalue_fields, providing ~2x throughput improvement over array format ([#3324](https://github.com/opensearch-project/k-NN/pull/3324))
* Add capability to retrieve float data type vectors using doc_values instead of reading _source ([#3321](https://github.com/opensearch-project/k-NN/pull/3321))
* Support derived source for knn_vector fields alongside other fields ([#3260](https://github.com/opensearch-project/k-NN/pull/3260))
* Add support for 1-bit scalar quantization with remote index build ([#3270](https://github.com/opensearch-project/k-NN/pull/3270))

### Enhancements

* Add bulk scoring logic in Memory Optimized Search when K exceeds the number of docs in a segment for improved SIMD/vectorization performance ([#3285](https://github.com/opensearch-project/k-NN/pull/3285))
* Use KNN1040ScalarQuantizedVectorsFormat for Faiss SQ flat format to enable I/O prefetch during exact search rescoring ([#3302](https://github.com/opensearch-project/k-NN/pull/3302))

### Infrastructure

* Add issues write permission to untriaged label workflow to fix 403 errors ([#3332](https://github.com/opensearch-project/k-NN/pull/3332))

### Maintenance

* Bump Gradle to 9.4.1 and JaCoCo to 0.8.14 to align with core OpenSearch ([#3308](https://github.com/opensearch-project/k-NN/pull/3308))
* Clean up Changelog.md file after 3.6 release ([#3266](https://github.com/opensearch-project/k-NN/pull/3266))
