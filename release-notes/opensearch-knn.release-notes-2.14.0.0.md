## Version 2.14.0.0 Release Notes

Compatible with OpenSearch 2.14.0

### Features
* Add k-NN clear cache api [#740](https://github.com/opensearch-project/k-NN/pull/740)
* Support radial search in k-NN plugin [#1617](https://github.com/opensearch-project/k-NN/pull/1617)
* Support filter and nested field in faiss engine radial search [#1652](https://github.com/opensearch-project/k-NN/pull/1652)
### Enhancements
* Make the HitQueue size more appropriate for exact search [#1549](https://github.com/opensearch-project/k-NN/pull/1549)
* Implement the Streaming Feature to stream vectors from Java to JNI layer to enable creation of larger segments for vector indices [#1604](https://github.com/opensearch-project/k-NN/pull/1604)
* Remove unnecessary toString conversion of vector field and added some minor optimization in KNNCodec [1613](https://github.com/opensearch-project/k-NN/pull/1613)
* Serialize all models into cluster metadata [#1499](https://github.com/opensearch-project/k-NN/pull/1499)
### Bug Fixes
* Add stored fields for knn_vector type [#1630](https://github.com/opensearch-project/k-NN/pull/1630)
* Enable script score to work with model based indices [#1649](https://github.com/opensearch-project/k-NN/pull/1649)
### Infrastructure
* Add micro-benchmark module in k-NN plugin for benchmark streaming vectors to JNI layer functionality. [#1583](https://github.com/opensearch-project/k-NN/pull/1583)
* Add arm64 check when SIMD is disabled [#1618](https://github.com/opensearch-project/k-NN/pull/1618)
* Skip rebuild from scratch after cmake is run [#1636](https://github.com/opensearch-project/k-NN/pull/1636)
